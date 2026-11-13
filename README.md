# LiverCT-LLM-Benchmark

Code accompanying the research paper:

**“Generative Versus Specialized Probabilistic AI Models for Structured Liver Finding Extraction From French CT Reports: A Comparative Evaluation”**

This repository contains Python scripts for extracting structured liver findings from French abdominal CT reports using six generative large language models (LLMs) and one specialized probabilistic decision model (Jev).

The study compares model performance against radiologist-consensus annotations and evaluates inter-model agreement, positive-class performance, and seven-model unanimity for potential research dataset curation.

## 1. Study Overview

The study evaluated **383 French abdominal CT reports**, divided into:

- **54 reports** for prompt development.
- **329 reports** for independent evaluation.

Two experienced radiologists independently annotated the reports and resolved disagreements by consensus.

Each model was evaluated on **12 predefined liver-related variables**, comprising 11 categorical fields and one numerical liver-lesion-count field.

### Evaluated Models

| Provider | Model |
|---|---|
| OpenAI | GPT-4o |
| OpenAI | GPT-5.6-Luna |
| OpenAI | GPT-5.6-Sol |
| Anthropic | Claude-Haiku-4-5 |
| Anthropic | Claude-Sonnet-5-5 |
| Anthropic | Claude-Opus-5-5 |
| TypeSafe | Jev 1.14 |

The six generative LLMs were evaluated using standardized instructions with structured outputs. Jev was evaluated using equivalent clinical definitions adapted to its probabilistic question-answering API.


## 2. Requirements

The extraction and evaluation workflows use Python.

The original study used **Python 3.11.7**.

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` file should reflect the packages and versions actually used by the extraction scripts and statistical notebook.

## 4. API Configuration

Running the extraction scripts requires valid API credentials for the corresponding provider.

For OpenAI, Anthropic and TypeSafe, configure the standard environment variables:
On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = ""
$env:ANTHROPIC_API_KEY = ""
$env:TYPESAFE_API_KEY = "" 
```


## 5. Input Reports

The extraction scripts accept a directory containing abdominal CT reports.

Example input directory:

```text
ReportsSept2025_Val/
```

**The original clinical reports are not included in this repository** because of patient privacy and institutional restrictions.

## 6. Running Model Extraction

Example: GPT-5.6-Sol with low reasoning effort.

```bash
python extract_liver_labels.py \
  --input ReportsSept2025_Val/ \
  --output CTliverlabelsSept2025_Val_Openai_Sol.csv \
  --providers openai \
  --openai-model gpt-5.6-sol \
  --openai-reasoning-effort low \
  --strip-before "VCT64 ABDOMINO-PELVIEN"
```

Example: Claude-Opus-5-5 with low reasoning effort.

```bash
python extract_liver_labels.py \
  --input ReportsSept2025_Val/ \
  --output CTliverlabelsSept2025_Val_Anthropic_Opus5.csv \
  --providers anthropic \
  --anthropic-model claude-opus-5-5 \
  --anthropic-effort low \
  --strip-before "VCT64 ABDOMINO-PELVIEN"
```

Example: Jev 1.14.

```bash
python extract_liver_labels_jev.py \
  --input ReportsSept2025_Val/ \
  --output jev_liver_labels.csv \
  --model jev-1.14.0 \
  --strip-before "VCT64 ABDOMINO-PELVIEN" \
  --probabilities-output jev_raw_probabilities.csv
```

The Jev extraction script produces:

- `jev_liver_labels.csv` — Extracted structured labels.
- `jev_raw_probabilities.csv` — Raw probability outputs for inspection and reproducibility.

In the study, Jev responses were mapped to the common evaluation schema. Binary decisions were derived using a probability threshold of 0.5, while categorical answers were mapped to predefined labels.

Jev was not included in the generative LLM prompt-development process.

## 7. Statistical Evaluation

Model predictions were compared with radiologist-consensus annotations on the held-out evaluation set of 329 reports.

The statistical analysis includes:

- Balanced accuracy and macro F1 across 11 categorical fields.
- Class-specific precision, recall, and F1.
- Report-level bootstrap confidence intervals.
- Pairwise model comparisons using two-sided paired permutation tests.
- Holm correction for multiple comparisons.
- Cohen's κ and raw inter-model agreement.
- Seven-model unanimity and reference-standard concordance.
- Analysis of positive clinical findings.
- Lesion-count exact agreement and mean absolute error.

### Seven-Model Unanimity

Unanimity is defined at the individual report-field level: all seven models must produce the same categorical prediction for that field.

Unanimous predictions are compared with the radiologist reference standard. Non-unanimous decisions are identified as potential candidates for expert review.

Unanimity is analyzed as an agreement-based selection approach and does not imply superiority
