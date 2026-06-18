# Benchmark Data

This directory holds the four benchmarks used by CultureLens-RC. **None of the raw data is committed** — you need to download it yourself before running any experiments.

## Layout (after you build)

```
data/
├── raw/
│   ├── worldvaluesbench/    (WVS Wave 7 v6.0 CSV)
│   ├── globalopinionqa/     (HuggingFace dataset cache)
│   ├── normad/              (cloned from official repo)
│   └── blend/               (HuggingFace dataset cache)
├── processed/
│   ├── wvb/probe_distributions.parquet
│   ├── wvb/probe_manifest.parquet
│   ├── wvb/probe_question_metadata.json
│   ├── wvb/split_info.json
│   ├── goqa/global_opinions_processed.parquet
│   ├── normad/normad_processed.parquet
│   ├── normad/story_id_split.csv
│   ├── blend/blend_mc.parquet
│   └── blend/blend_short_answer.parquet
├── splits/
│   ├── wvb_split.json
│   ├── goqa_split.json
│   ├── normad_split.json
│   └── blend_split.json
├── evidence/
│   ├── evidence.jsonl        (~46k records, built from train splits)
│   └── evidence_schema.json
└── prototypes/
    ├── wvb/country_prototypes.json
    ├── goqa/country_prototypes.json
    ├── normad/country_prototypes.json
    └── blend/country_prototypes.json
```

## Download Instructions

### 1. WorldValuesBench (WVB)

- WVS Wave 7 v6.0 raw CSV: register at https://www.worldvaluessurvey.org/ (WVSA limited-use license).
- Code + metadata: clone `github.com/Demon702/WorldValuesBench`.
- After download, place under `data/raw/worldvaluesbench/`.

### 2. GlobalOpinionQA (GOQA)

- HuggingFace dataset: `Anthropic/llm_global_opinions`.
- Load via `datasets.load_dataset("Anthropic/llm_global_opinions")` and cache under `data/raw/globalopinionqa/`.

### 3. NormAd-Eti

- Clone official repo: `git clone https://github.com/Akhila-Yerukola/NormAd data/raw/normad/`.

### 4. BLEnD

- HuggingFace dataset: `nayeon212/BLEnD`.
- Load and cache under `data/raw/blend/hf/`.

## Build pipeline

After raw data is in place, run the data preparation:

```bash
# 1. (Optional) Canonicalize country names across benchmarks
python data/scripts/canonicalize_countries.py

# 2. Materialize processed parquet files (will depend on each bench's loader)
python data/scripts/eda_all.py        # also produces analysis/eda_report.md

# 3. Build unified evidence database + per-country prototypes from train splits
python data/scripts/build_evidence_and_prototypes.py
```

This produces `data/processed/*`, `data/evidence/evidence.jsonl`, and `data/prototypes/<bench>/country_prototypes.json`, ready for `scripts/build_evidence_embeddings.py`.

## Splits

We use deterministic 60/20/20 splits keyed by question ID (not by row), so the same question never spans train/test across countries. See each `data/splits/<bench>_split.json` for the question-ID lists and the seed (`42`).

## Leakage Verification

Before running experiments, verify no test/probe content leaked into evidence:

```bash
pytest tests/test_leakage_check.py
```

Or programmatically:

```bash
python -m src.leakage_check --evidence data/evidence/evidence.jsonl --benches all
```

## Schema

Evidence record schema (see `data/evidence/evidence_schema.json` for full JSON-schema):

```json
{
  "evidence_id": "wvb::train::Q4::abc123",
  "source": "worldvaluesbench",
  "split": "train",
  "country_or_region": "Japan",
  "language": "en",
  "task_type": "survey_distribution_ordinal",
  "topic": "Social Values, Norms, Stereotypes",
  "text": "On a scale of 1 to 4, ...",
  "answer_options": {"1": "1", "2": "2", "3": "3", "4": "4"},
  "distribution": {"1": 0.10, "2": 0.20, "3": 0.30, "4": 0.40},
  "metadata": {"wvs_wave": 7, "n_respondents": 700},
  "indexed_at": "2026-06-16T09:04:02Z",
  "license": "WVSA-restricted (WVS Wave 7 v6.0)"
}
```
