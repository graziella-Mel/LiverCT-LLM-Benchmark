"""
Extract liver status (normal / abnormal / not_mentioned) labels from CT abdomen
radiology reports using 2 independent LLM APIs (OpenAI, Anthropic). 

Setup
-----
    pip install -r requirements.txt
    Create a .env file with:
        OPENAI_API_KEY=...
        ANTHROPIC_API_KEY=...


Usage
-----
    # output a single CSV with columns: report_id, report_text
    python extract_liver_labels.py --input reports.csv --output liver_labels.csv --strip-before "VCT64 ABDOMINO-PELVIEN"

"""


from dotenv import load_dotenv
import argparse
import csv
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()

SYSTEM_PROMPT = """You are a radiology report analyst. You will be given the text
of a CT abdomen report. Your only task is to extract structured information about
the LIVER only. Ignore other organs unless a
finding is explicitly stated to originate from or affect the liver. 


Respond with ONLY a JSON object, no other text, no markdown fences:
{
  "liver_status": "normal" | "abnormal" | "not_mentioned",
  "liver_malignancy_status": "Not malignant" | "Malignant" | "not_mentioned",
  "liver_malignancy_level": "Primary" | "Secondary" | "not_mentioned",
  "Further_Exam_required": "Y" | "N",
  "liver_lesions_count": <integer, or 0 if no lesion>,
  "liver_vague_multiple_present": "Y" | "N",
  "liver_steatosis": "Y" | "N",
  "liver_cirrhosis": "Y" | "N",
  "Hepatomegaly": "Y" | "N",
  "liver_vascular_disease": "Y" | "N",
  "liver_calcification": "Y" | "N",
  "Intrahepatic_duct_dilation": "Y" | "N",
  "evidence": "<verbatim sentence(s) from the report that justify your answer, or empty string>",
}


Decide fields IN THIS ORDER. Later steps use earlier answers.
1. liver_status: "not_mentioned" — liver not described, OR only compared to prior with no indication of the prior state.
   "abnormal" — any liver finding described, even if stable.
   "normal" — liver described with no abnormal finding.
2.  liver_lesions_count/ liver_vague_multiple_present
    - liver_lesions_count: Count lesions only when described with quantifiable language.
        "X and Y" (same or different segments) = 2. One mass spanning/straddling two or more segments = 1. 
        0 only if no lesion is described. 
        Don't invent count for lesions described with unquantifiable language. 
        Calcification in the liver is not counted as a lesion
    - liver_vague_multiple_present: "Y" if vague/unquantified multiplicity language appears anywhere about liver lesions. Else "N".
     "some lesions, including a 2cm mass"  -> "Y", count 1
     "multiples lesions hypodenses"          -> "Y", count 0
     "lesion of 15 mm"                   -> "N", count 1
3. liver_malignancy_status:
    Scope: the LIVER, not an individual lesion. A liver with no lesion is
    "Not malignant" — not "not_mentioned". Reserve "not_mentioned" for reports
    giving no basis to judge.
        liver_status "not_mentioned"  -> "not_mentioned"
        liver_status "normal"         -> "Not malignant"
        liver_status "abnormal":
            no lesion (steatosis/cirrhosis/calcification only) -> "Not malignant"
            any lesion malignant or suspected malignant        -> "Malignant"
            all lesions benign or suspected benign             -> "Not malignant"
            lesions with mixed or uncharacterized suspicion    -> "not_mentioned"
4. Further_Exam_required: 
    "Y" only if the report explicitly recommends follow-up or further
   characterization
   "N" otherwise.
   Never infer a recommendation that is not written.

"""


def build_user_prompt(report_text: str) -> str:
    return f"CT ABDOMEN REPORT:\n\n{report_text}\n\nReturn the JSON now."


def strip_before_marker(text: str, marker: str, report_id: str = "") -> str:
    """Return everything from the LAST occurrence of `marker` in `text`
    onward, with the marker itself removed. Using the last (not first)
    occurrence matters when the marker also appears earlier as a metadata
    field value (e.g. an exam-name column) before appearing again as the
    real report's own section header. If the marker isn't found at all, the
    original text is returned unchanged and a warning is printed -- this is
    a safety fallback so a report using a different header/exam type never
    gets silently emptied out."""
    if not marker:
        return text
    idx = text.rfind(marker)
    if idx == -1:
        print(f"  WARNING: marker {marker!r} not found in report {report_id!r} -- "
              f"sending the full, unstripped text for this one.")
        return text
    return text[idx + len(marker):].lstrip()


STRUCTURED_FIELDS = [
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


def sanitize_lesion_count(value):
    """Enforce the contract: liver_lesions_count is either an int 0-100, or None."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 0 <= n <= 100 else None


def parse_json_response(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        data["liver_lesions_count"] = sanitize_lesion_count(data.get("liver_lesions_count"))
        return data
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(text[start:end + 1])
                data["liver_lesions_count"] = sanitize_lesion_count(data.get("liver_lesions_count"))
                return data
            except json.JSONDecodeError:
                pass
        fallback = {field: "parse_error" for field in STRUCTURED_FIELDS}
        fallback["liver_lesions_count"] = None
        fallback["evidence"] = f"[UNPARSEABLE RESPONSE] {text[:500]}"
        fallback["confidence"] = 0.0
        return fallback



def call_openai(report_text: str, model: str = "", extra: str = "") -> dict:
    from openai import OpenAI
    client = OpenAI()
    use_reasoning = bool(extra) and extra != "none"
    kwargs = dict(
        model=model,
        max_completion_tokens=8000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(report_text)},
        ],
    )
    if use_reasoning:
        kwargs["reasoning_effort"] = extra
    else:
        kwargs["temperature"] = 0
    resp = client.chat.completions.create(**kwargs)
    return parse_json_response(resp.choices[0].message.content)
 
 
def call_anthropic(report_text: str, model: str = "", extra: str = "") -> dict:
    import anthropic
    client = anthropic.Anthropic()
    use_effort = bool(extra) and extra != "none"
    kwargs = dict(
        model=model,
        max_tokens=8000,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": build_user_prompt(report_text)}],
    )
    if use_effort:
        kwargs["output_config"] = {"effort": extra}
    else:
        kwargs["temperature"] = 0
    resp = client.messages.create(**kwargs)
    text = "".join(block.text for block in resp.content if block.type == "text")
    return parse_json_response(text)


MODEL_FUNCS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
}


_cache_lock = threading.Lock()


def cache_key(provider: str, model: str, report_text: str, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(provider.encode("utf-8"))
    h.update(b"|")
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(extra.encode("utf-8"))
    h.update(b"|")
    h.update(SYSTEM_PROMPT.encode("utf-8"))
    h.update(b"|")
    h.update(report_text.encode("utf-8"))
    return h.hexdigest()


def load_cache(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        print(f"Warning: could not read cache file {path}, starting with an empty cache.")
        return {}


def save_cache(path: str, cache: dict) -> None:
    tmp_path = f"{path}.tmp"
    with _cache_lock:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        os.replace(tmp_path, path)


def load_reports(input_path: str, strip_marker: str = None):
    p = Path(input_path)
    reports = []
    if p.is_dir():
        for f in sorted(p.glob("*.txt")):
            text = f.read_text(encoding="utf-8", errors="ignore")
            if strip_marker:
                text = strip_before_marker(text, strip_marker, report_id=f.stem)
            reports.append((f.stem, text))
    else:
        with open(p, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                text = row["report_text"]
                if strip_marker:
                    text = strip_before_marker(text, strip_marker, report_id=row["report_id"])
                reports.append((row["report_id"], text))
    return reports


def error_result(message: str) -> dict:
    result = {field: "error" for field in STRUCTURED_FIELDS}
    result["liver_lesions_count"] = None
    result["evidence"] = message
    result["confidence"] = 0.0
    return result


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


_provider_semaphores = {}
_semaphore_lock = threading.Lock()


def get_semaphore(provider: str, limit: int) -> threading.Semaphore:
    with _semaphore_lock:
        if provider not in _provider_semaphores:
            _provider_semaphores[provider] = threading.Semaphore(limit)
        return _provider_semaphores[provider]


def process_report(report_id, text, models_config, cache, reasoning_effort="medium",
                    anthropic_effort="low", openai_max_concurrent=2, retries=4):
    provider_extra = {"openai": reasoning_effort, "anthropic": anthropic_effort}
    provider_concurrency = {"openai": openai_max_concurrent}
    row = {"report_id": report_id}
    new_cache_entries = False

    for provider, model_name in models_config.items():
        func = MODEL_FUNCS[provider]
        extra = provider_extra.get(provider, "")
        key = cache_key(provider, model_name, text, extra)

        with _cache_lock:
            result = cache.get(key)

        if result is None:
            sem = get_semaphore(provider, provider_concurrency.get(provider, 999))
            last_err = None
            for attempt in range(retries + 1):
                with sem:
                    try:
                        result = func(text, model_name, extra)
                        break
                    except Exception as e:
                        last_err = e
                if result is not None:
                    break
                delay = (5 * (2 ** attempt)) if is_rate_limit_error(last_err) else (1.5 * (attempt + 1))
                time.sleep(delay)
            if result is None:
                result = error_result(str(last_err))
            else:
                with _cache_lock:
                    cache[key] = result
                new_cache_entries = True

        for field in STRUCTURED_FIELDS:
            row[f"{provider}_{field}"] = result.get(field, "not_mentioned")
        row[f"{provider}_evidence"] = result.get("evidence", "")
        row[f"{provider}_confidence"] = result.get("confidence", 0.0)

    return row, new_cache_entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Folder of .txt reports or a CSV with report_id,report_text")
    ap.add_argument("--output", default="liver_labels.csv")
    ap.add_argument("--providers", default="openai,anthropic",
                     help="Comma-separated subset of: openai,anthropic")
    ap.add_argument("--openai-model", default="")
    ap.add_argument("--anthropic-model", default="")
    ap.add_argument("--openai-reasoning-effort", default="",
                     choices=["none", "low", "medium", "high", "xhigh", "max"],
                     help="Reasoning effort for gpt-5.x models.")
    ap.add_argument("--anthropic-effort", default="",
                     choices=["none", "low", "medium", "high", "xhigh", "max"],
                     help="Thinking effort for Claude 5-generation models.")
    ap.add_argument("--openai-max-concurrent", type=int, default=2,
                     help="Max simultaneous OpenAI requests, independent of --workers. "
                          "Lower this (e.g. to 1) if you hit 429 TPM rate-limit errors.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel reports processed at once")
    ap.add_argument("--cache-file", default="llm_cache.json",
                     help="Path to the local response cache. Reused automatically on re-runs.")
    ap.add_argument("--no-cache", action="store_true",
                     help="Ignore and don't update the cache file for this run.")
    ap.add_argument("--strip-before", default=None,
                     help='Drop everything before the LAST occurrence of this marker string in '
                          'each report (e.g. "VCT64 ABDOMINO-PELVIEN"), so leading metadata/'
                          'patient-identifying info is neither sent to the LLM nor cached. '
                          "Reports without the marker keep their full original text (with a "
                          "warning printed) rather than being emptied out.")
    args = ap.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    models_config = {}
    if "openai" in providers:
        models_config["openai"] = args.openai_model
    if "anthropic" in providers:
        models_config["anthropic"] = args.anthropic_model

    if not models_config:
        raise SystemExit("No valid providers selected. Choose from: openai,anthropic")

    reports = load_reports(args.input, strip_marker=args.strip_before)
    if not reports:
        raise SystemExit(f"No reports found at {args.input}")

    print(f"Loaded {len(reports)} reports. Querying providers: {list(models_config)}")
    if args.strip_before:
        print(f"Stripping everything before the last occurrence of {args.strip_before!r} in each report.")

    cache = {} if args.no_cache else load_cache(args.cache_file)
    if cache:
        print(f"Loaded {len(cache)} cached responses from {args.cache_file}")

    rows_by_id = {}
    any_new_entries = False
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_report, rid, text, models_config, cache,
                      args.openai_reasoning_effort, args.anthropic_effort,
                      args.openai_max_concurrent): rid
            for rid, text in reports
        }
        done = 0
        for fut in as_completed(futures):
            rid = futures[fut]
            row, had_new = fut.result()
            rows_by_id[rid] = row
            any_new_entries = any_new_entries or had_new
            done += 1
            print(f"  [{done}/{len(reports)}] done: {rid}")

    if not args.no_cache and any_new_entries:
        save_cache(args.cache_file, cache)
        print(f"Updated cache: {args.cache_file} ({len(cache)} entries total)")

    fieldnames = ["report_id"]
    for provider in models_config:
        for field in STRUCTURED_FIELDS:
            fieldnames.append(f"{provider}_{field}")
        fieldnames += [f"{provider}_evidence", f"{provider}_confidence"]

    with open(args.output, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rid, _ in reports:
            writer.writerow(rows_by_id[rid])

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
