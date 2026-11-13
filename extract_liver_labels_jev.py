"""Extract structured liver labels from CT reports with Jev (TypeSafe AI).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore
from typing import Any

from dotenv import load_dotenv

load_dotenv()


FIELDS = [
    "liver_status",
    "liver_malignancy_status",
    "liver_malignancy_level",
    "Further_Exam_required",
    "liver_lesions_count",
    "liver_vague_multiple_present",
    "liver_steatosis",
    "liver_cirrhosis",
    "Hepatomegaly",
    "liver_vascular_disease",
    "liver_calcification",
    "Intrahepatic_duct_dilation",
]

COUNT_CHOICES = {str(i): f"Exactly {i} distinct, quantifiable liver lesions" for i in range(101)}
COUNT_CHOICES["0"] = (
    "No countable liver lesion: either no lesion is described, or only vague "
    "unquantified lesions are described without an individually quantifiable lesion."
)
COUNT_CHOICES["unknown"] = (
    "An exact count cannot be given without inventing a number, or exceeds "
    "the available integer choices (0-100)."
)
CHOICE_FIELDS = FIELDS[:3] + ["liver_lesions_count"]
YES_NO_FIELDS = [field for field in FIELDS if field not in CHOICE_FIELDS]
ALLOWED = {
    "liver_status": {"normal", "abnormal", "not_mentioned"},
    "liver_malignancy_status": {"Not malignant", "Malignant", "not_mentioned"},
    "liver_malignancy_level": {"Primary", "Secondary", "not_mentioned"},
    "liver_lesions_count": set(COUNT_CHOICES),
    **{field: {"Y", "N"} for field in YES_NO_FIELDS},
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "liver_status": {
        "type": "choice",
        "instructions": "Determine liver_status for the LIVER only.",
        "criteria": {
            "normal": "Liver described with no abnormal finding.",
            "abnormal": "Any liver finding described, even if stable.",
            "not_mentioned": (
                "Liver not described, OR only compared to prior with no indication "
                "of the prior state."
            ),
        },
    },
    "liver_malignancy_status": {
        "type": "choice",
        "instructions": (
            "Determine liver_malignancy_status for the LIVER, not an individual "
            "lesion. A liver with no lesion is 'Not malignant', not 'not_mentioned'. "
            "Reserve 'not_mentioned' for reports giving no basis to judge. "
            "Use the liver_status definitions when making this decision."
        ),
        "criteria": {
            "Malignant": (
                "For an abnormal liver, any lesion malignant or suspected malignant."
            ),
            "Not malignant": (
                "liver_status is normal; OR liver_status is abnormal with no "
                "lesion (steatosis/cirrhosis/calcification only); OR all lesions "
                "are benign or suspected benign."
            ),
            "not_mentioned": (
                "liver_status is not_mentioned; OR liver_status is abnormal "
                "and lesions have mixed or uncharacterized suspicion."
            ),
        },
    },
    "liver_malignancy_level": {
        "type": "choice",
        "instructions": "Determine liver_malignancy_level for the LIVER only.",
        "criteria": {
            "Primary": "Primary liver malignancy.",
            "Secondary": "Secondary liver malignancy.",
            "not_mentioned": "Liver malignancy level is not mentioned.",
        },
    },
    "Further_Exam_required": {
        "type": "noul",
        "instructions": (
            "Is follow-up or further characterization of a LIVER finding explicitly "
            "recommended in the report? Answer yes only if explicitly recommended; "
            "otherwise answer no."
        ),
    },
    "liver_lesions_count": {
        "type": "choice",
        "instructions": (
            "Count liver lesions only when described with quantifiable language. "
            "'X and Y' (same or different segments) = 2. One mass spanning or "
            "straddling two or more segments = 1. Do not invent a count for "
            "lesions described with unquantifiable language. Calcification "
            "in the liver is not counted as a lesion. Use 0 when no lesion is "
            "described. For vaguely described multiple lesions with no separately "
            "quantifiable lesion, the given count example assigns 0. "
        ),
        "criteria": COUNT_CHOICES,
    },
    "liver_vague_multiple_present": {
        "type": "noul",
        "instructions": (
            "Answer yes if vague or unquantified multiplicity language appears "
            "anywhere about liver lesions. Otherwise answer no. "
            "Use the cases given in the prompt: "
            "'some lesions, including a 2cm mass' -> Y, count 1; "
            "'multiples lesions hypodenses' -> Y, count 0; "
            "'lesion of 15 mm' -> N, count 1."
        ),
    },
    "liver_steatosis": {
        "type": "noul",
        "instructions": "Is liver_steatosis present? Answer Y or N for the LIVER only.",
    },
    "liver_cirrhosis": {
        "type": "noul",
        "instructions": "Is liver_cirrhosis present? Answer Y or N for the LIVER only.",
    },
    "Hepatomegaly": {
        "type": "noul",
        "instructions": "Is Hepatomegaly present? Answer Y or N for the LIVER only.",
    },
    "liver_vascular_disease": {
        "type": "noul",
        "instructions": "Is liver_vascular_disease present? Answer Y or N for the LIVER only.",
    },
    "liver_calcification": {
        "type": "noul",
        "instructions": "Is liver_calcification present? Answer Y or N for the LIVER only.",
    },
    "Intrahepatic_duct_dilation": {
        "type": "noul",
        "instructions": "Is Intrahepatic_duct_dilation present? Answer Y or N for the LIVER only.",
    },
}

TASK_SCOPE = (
    "You are a radiology report analyst. You will be given the text of a CT "
    "abdomen report. Your only task is to extract structured information about "
    "the LIVER only. Ignore other organs unless a finding is explicitly stated "
    "to originate from or affect the liver. Decide liver_status first, then "
    "liver_lesions_count and liver_vague_multiple_present, then "
    "liver_malignancy_status, then Further_Exam_required; later decisions use "
    "earlier answers."
)

PIPELINE_VERSION = "jev-liver-labels-v3-choice-lesion-count"


def strip_before_marker(text: str, marker: str) -> str:
    """Strip through the LAST occurrence, as in the original LLM script.

    Unlike the original, a missing marker raises an error rather than
    inadvertently sending raw, potentially identifying report headers.
    """
    if not marker:
        return text
    index = text.rfind(marker)
    if index < 0:
        raise ValueError("strip_marker_not_found")
    stripped = text[index + len(marker):].lstrip()
    if not stripped:
        raise ValueError("empty_report_after_stripping")
    return stripped


def load_reports(input_path: str, strip_marker: str | None, limit: int = 0):
    """Return (report_id, text, local_error) in stable input order."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    output: list[tuple[str, str, str | None]] = []
    if source.is_dir():
        iterator = ((file.stem, file.read_text(encoding="utf-8-sig", errors="replace"))
                    for file in sorted(source.glob("*.txt")))
    else:
        if source.suffix.lower() != ".csv":
            raise ValueError("Input must be a directory of .txt files or a .csv file")
        with source.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            required = {"report_id", "report_text"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("CSV must contain columns: report_id, report_text")
            # Read while handle is open; output is one tuple per input row.
            iterator = [(row.get("report_id", ""), row.get("report_text", ""))
                        for row in reader]

    for report_id, raw_text in iterator:
        if limit and len(output) >= limit:
            break
        report_id = str(report_id or "").strip()
        raw_text = str(raw_text or "")
        try:
            if not report_id:
                raise ValueError("missing_report_id")
            if not raw_text.strip():
                raise ValueError("empty_report")
            text = strip_before_marker(raw_text, strip_marker)
            output.append((report_id, text, None))
        except ValueError as err:
            # The error string is a fixed code, not text from the patient report.
            output.append((report_id, "", str(err)))
    return output


def cache_key(model: str, report_text: str) -> str:
    """Independent of threshold: cache raw Jev predictions."""
    policy = {
        "version": PIPELINE_VERSION,
        "model": model,
        "questions": QUESTIONS,
        "task_scope": TASK_SCOPE,
        "report_text": report_text,
    }
    payload = json.dumps(policy, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def valid_probability(value: Any) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def raw_answer_is_valid(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    choices = result.get("choices")
    choice_probabilities = result.get("choice_probabilities")
    nouls = result.get("nouls")
    if not all(isinstance(v, dict) for v in (choices, choice_probabilities, nouls)):
        return False
    for field in CHOICE_FIELDS:
        if choices.get(field) not in ALLOWED[field]:
            return False
        values = choice_probabilities.get(field)
        if not isinstance(values, dict) or set(values) != ALLOWED[field]:
            return False
        if not all(valid_probability(p) for p in values.values()):
            return False
        # The API may round probabilities; don't renormalize or change them.
        if abs(sum(values.values()) - 1.0) > 0.05:
            return False
    return all(valid_probability(nouls.get(field)) for field in YES_NO_FIELDS)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            result = json.load(fh)
        if not isinstance(result, dict):
            raise ValueError("cache_not_mapping")
        return {k: v for k, v in result.items()
                if isinstance(k, str) and raw_answer_is_valid(v)}
    except (OSError, ValueError, json.JSONDecodeError):
        print("Warning: cannot read cache; starting with an empty cache.")
        return {}


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, sort_keys=True)
    os.replace(temp, path)


def parse_jev_response(response: Any) -> dict[str, Any]:
    """Capture unmodified Jev Choice choices/distributions and Noul P(yes).
    """
    choices: dict[str, str] = {}
    choice_probabilities: dict[str, dict[str, float]] = {}
    nouls: dict[str, float] = {}
    for field in CHOICE_FIELDS:
        try:
            answer = response.choices[field]
            selection = answer.choice
            probabilities = answer.probabilities
        except (AttributeError, KeyError, TypeError) as err:
            raise ValueError("missing_or_invalid_choice") from err
        if selection not in ALLOWED[field]:
            raise ValueError("invalid_choice_category")
        if not isinstance(probabilities, dict) or set(probabilities) != ALLOWED[field]:
            raise ValueError("missing_choice_probabilities")
        if not all(valid_probability(p) for p in probabilities.values()):
            raise ValueError("invalid_choice_probabilities")
        if abs(sum(probabilities.values()) - 1.0) > 0.05:
            raise ValueError("invalid_choice_probability_sum")
        choices[field] = selection
        choice_probabilities[field] = dict(probabilities)

    for field in YES_NO_FIELDS:
        try:
            probability = float(response.nouls[field].noul)
        except (AttributeError, KeyError, TypeError, ValueError) as err:
            raise ValueError("missing_or_invalid_noul") from err
        if not valid_probability(probability):
            raise ValueError("out_of_range_noul")
        nouls[field] = probability

    result = {"choices": choices, "choice_probabilities": choice_probabilities,
              "nouls": nouls}
    if not raw_answer_is_valid(result):
        raise ValueError("invalid_raw_answer")
    return result


def make_labels(raw: dict[str, Any], yes_threshold: float) -> dict[str, Any]:
    """Create labels from Jev's independent answers; no hierarchical rules."""
    output = dict(raw["choices"])
    output.update({field: ("Y" if raw["nouls"][field] >= yes_threshold else "N")
                   for field in YES_NO_FIELDS})
    if not all(output.get(field) in ALLOWED[field] for field in FIELDS):
        raise ValueError("invalid_final_labels")
    chosen_count = output["liver_lesions_count"]
    output["liver_lesions_count"] = (int(chosen_count) if chosen_count != "unknown" else None)
    return output


def probability_columns() -> list[str]:
    """Stable, explicit per-category naming. For Nouls record raw P(yes)."""
    cols = ["report_id"]
    for field in CHOICE_FIELDS:
        cols.append(f"jev_raw_{field}")
        for label in QUESTIONS[field]["criteria"]:
            cols.append(f"jev_{field}__p_{label.replace(' ', '_')}")
    for field in YES_NO_FIELDS:
        cols.append(f"jev_{field}__p_Y")
    return cols


def probability_row(raw: dict[str, Any]) -> dict[str, str | float]:
    row: dict[str, str | float] = {}
    for field in CHOICE_FIELDS:
        row[f"jev_raw_{field}"] = raw["choices"][field]
        for label in QUESTIONS[field]["criteria"]:
            row[f"jev_{field}__p_{label.replace(' ', '_')}"] = raw["choice_probabilities"][field][label]
    for field in YES_NO_FIELDS:
        row[f"jev_{field}__p_Y"] = raw["nouls"][field]
    return row


def retryable(exc: Exception) -> bool:
    """Do not retry authentication, permission, schema, or parsing failures."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status in (408, 409, 425, 429) or status >= 500
    name = type(exc).__name__.lower()
    return "connection" in name or "timeout" in name


def safe_error(exc: Exception) -> str:
    """Avoid copying raw API exceptions (which may include patient text)."""
    if isinstance(exc, ValueError) and str(exc) in {
        "missing_or_invalid_choice", "invalid_choice_category",
        "missing_or_invalid_noul", "out_of_range_noul", "invalid_final_labels",
        "missing_choice_probabilities", "invalid_choice_probabilities",
        "invalid_choice_probability_sum", "invalid_raw_answer"
    }:
        return str(exc)
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return f"api_http_{status}"
    return type(exc).__name__


def call_jev(text: str, model: str,
             timeout: float, retries: int, semaphore: Semaphore) -> dict[str, Any]:
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    for attempt in range(retries + 1):
        try:
            with semaphore:
                with TypeSafeClient(model=model, timeout=timeout,
                                    retry=RetryPolicy(max_retries=0)) as client:
                    response = client.system_one(
                        state={"scope": TASK_SCOPE, "report_text": text},
                        questions=QUESTIONS,
                    )
            return parse_jev_response(response)
        except Exception as exc:
            if attempt >= retries or not retryable(exc):
                raise
            time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    raise RuntimeError("unreachable")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Folder of .txt reports, or CSV with report_id and report_text")
    parser.add_argument("--output", default="jev_liver_labels.csv")
    parser.add_argument("--probabilities-output", default=None,
                        help="Separate raw probabilities CSV (default: <output stem>.probabilities.csv)")
    parser.add_argument("--no-hierarchical-rules", action="store_true",
                        help="Deprecated compatibility flag: hierarchical rules are always disabled")
    parser.add_argument("--model", default="jev-latest",
                        help="Jev model (pin a version for reproducible research)")
    parser.add_argument("--yes-threshold", type=float, default=0.5,
                        help="Probability threshold for Noul Y/N fields (default 0.5)")
    parser.add_argument("--strip-before", default=None,
                        help="Strip up to the last occurrence of this report marker; "
                             "never send a report when the marker is missing")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent worker tasks (default 4)")
    parser.add_argument("--jev-max-concurrent", type=int, default=2,
                        help="Maximum in-flight Jev API calls (default 2)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="API timeout in seconds (default 60)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries for transient failures (default 3)")
    parser.add_argument("--cache-file", default="jev_cache.json")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N reports (0 = all)")
    args = parser.parse_args()

    if not 0 < args.yes_threshold < 1:
        parser.error("--yes-threshold must be strictly between 0 and 1")
    if args.workers < 1 or args.jev_max_concurrent < 1:
        parser.error("--workers and --jev-max-concurrent must be >= 1")
    if args.timeout <= 0 or args.retries < 0 or args.limit < 0:
        parser.error("--timeout must be > 0; --retries/--limit must be >= 0")

    reports = load_reports(args.input, args.strip_before, args.limit)
    if not reports:
        parser.error("No reports found")

    output_path = Path(args.output)
    probs_path = (Path(args.probabilities_output) if args.probabilities_output
                  else output_path.with_name(output_path.stem + ".probabilities.csv"))
    error_path = output_path.with_name(output_path.stem + ".errors.csv")
    cache_path = Path(args.cache_file)
    paths = [output_path.resolve(), probs_path.resolve(), cache_path.resolve(),
             error_path.resolve()]
    if len(set(paths)) != len(paths):
        parser.error("Output, probabilities, cache and errors paths must be different")
    if Path(args.input).is_file() and Path(args.input).resolve() in paths:
        parser.error("Output files cannot overwrite your input CSV")

    cache = {} if args.no_cache else load_cache(cache_path)
    label_columns = ["report_id"] + [f"jev_{field}" for field in FIELDS]
    prob_columns = probability_columns()
    output: list[dict[str, Any]] = [
        {"report_id": rid, **{name: "" for name in label_columns[1:]}}
        for rid, _, _ in reports
    ]
    probability_output: list[dict[str, Any]] = [
        {"report_id": rid, **{name: "" for name in prob_columns[1:]}}
        for rid, _, _ in reports
    ]
    errors: list[dict[str, str]] = []
    by_key: dict[str, dict[str, Any]] = {}
    cached_records = 0

    def write_result(index: int, raw: dict[str, Any]) -> None:
        labels = make_labels(raw, args.yes_threshold)
        output[index].update({f"jev_{field}": labels[field] for field in FIELDS})
        probability_output[index].update(probability_row(raw))

    for i, (rid, text, local_error) in enumerate(reports):
        if local_error:
            errors.append({"report_id": rid, "error": local_error})
            continue
        key = cache_key(args.model, text)
        if key in cache:
            write_result(i, cache[key])
            cached_records += 1
        else:
            if key not in by_key:
                by_key[key] = {"text": text, "indices": []}
            by_key[key]["indices"].append(i)

    if by_key and not os.getenv("TYPESAFE_API_KEY"):
        parser.error("TYPESAFE_API_KEY is missing. Set it in .env or the environment.")

    print(f"Loaded {len(reports)} reports; {cached_records} cache hits, "
          f"{len(by_key)} unique API calls needed; model={args.model}; "
           "hierarchical_rules=off.")
    semaphore = Semaphore(args.jev_max_concurrent)
    completed = 0
    cache_unsaved = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    call_jev, record["text"], args.model,
                    args.timeout, args.retries, semaphore
                ): (key, record["indices"])
                for key, record in by_key.items()
            }
            for future in as_completed(futures):
                key, indices = futures[future]
                try:
                    raw = future.result()
                    for i in indices:
                        write_result(i, raw)
                    if not args.no_cache:
                        cache[key] = raw
                        cache_unsaved += 1
                        if cache_unsaved >= 25:
                            save_cache(cache_path, cache)
                            cache_unsaved = 0
                except Exception as exc:
                    error = safe_error(exc)
                    for i in indices:
                        errors.append({"report_id": reports[i][0], "error": error})
                completed += 1
                print(f"  [{completed}/{len(by_key)}] processed", flush=True)
    finally:
        if cache_unsaved and not args.no_cache:
            save_cache(cache_path, cache)

    write_csv(output_path, label_columns, output)
    write_csv(probs_path, prob_columns, probability_output)
    print(f"Wrote raw probabilities to {probs_path}.")
    print(f"Wrote {output_path} ({len(output)} rows, {len(errors)} failures).")
    if errors:
        error_path = output_path.with_name(output_path.stem + ".errors.csv")
        write_csv(error_path, ["report_id", "error"], errors)
        print(f"Failure details: {error_path}")
    else:
        stale = output_path.with_name(output_path.stem + ".errors.csv")
        if stale.exists():
            stale.unlink()


if __name__ == "__main__":
    main()
