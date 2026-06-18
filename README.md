# CultureLens-RC

**Retrieval-Calibrated Cultural Alignment across Global Opinions, Social Norms, and Everyday Knowledge**

CultureLens-RC is an **inference-only** framework that aligns large language models with cultural variation by composing three orthogonal modules: (1) hierarchical (country × topic × task) **filtering** over a unified evidence database, (2) per-country **prototype** summaries built from training-split evidence, and (3) temperature **calibration** fit on a held-out dev split. We evaluate on four heterogeneous benchmarks — WorldValuesBench, GlobalOpinionQA, NormAd-Eti, and BLEnD — covering survey distributions, social-norm judgments, and everyday cultural knowledge.

Our headline finding is that the value of CultureLens-RC is **task-adaptive**: filtering matters most for distribution prediction (WVB), prototypes matter most for retrieval-poor benchmarks (GOQA, BLEnD SAQ), and calibration matters most for over-confident argmax tasks (NormAd). The full pipeline shows its largest gains in **low-resource cultural contexts**: +7.3 pp on GOQA low-resource countries and +16.7 pp on BLEnD-MC low-resource regions.

## Overview

```
┌──────────────────────────────────────────────────────────────┐
│   Question  +  Country  +  Topic                             │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  1. Hierarchical Filter      (country × topic over evidence) │
│  2. Top-k Retrieval          (FAISS dense, MiniLM-L6 384d)   │
│  3. Prototype Card           (per-country aggregate summary) │
│  4. LLM Prompt + Answer      (Azure gpt-* deployment)        │
│  5. Temperature Calibration  (T-scaling, per bench×prompt×K) │
└──────────────────────────────────────────────────────────────┘
```

## Environment Setup

- Python 3.11+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Set up API keys: copy `.env.example` to `.env` and fill in your values.
  - `AZURE_OPENAI_API_KEY` — your Azure AI Foundry deployment key
  - `AZURE_OPENAI_ENDPOINT` — your Azure endpoint URL
  - `AZURE_DEPLOYMENT_NAME` — your model deployment name (e.g. `gpt-4o-mini`)
  - `GEMINI_API_KEY` — optional, only for regenerating pipeline figures

> **Security**: never commit `.env`. The `.gitignore` already excludes it.

## Quick Start (5 steps)

1. **Download benchmark data** — see `data/README.md` for download instructions for WVB, GOQA, NormAd, BLEnD.

2. **Build evidence database + prototypes** (from train splits):
   ```bash
   python data/scripts/build_evidence_and_prototypes.py
   ```

3. **Build FAISS retrieval index** (sentence-transformers/all-MiniLM-L6-v2, ~20 s):
   ```bash
   python scripts/build_evidence_embeddings.py \
     --evidence data/evidence/evidence.jsonl \
     --out-dir /tmp/culturelens_rc/embeddings \
     --embedder sentence_transformers
   ```

4. **Fit temperature calibration** on dev splits (~6 min, ~1300 LLM calls, ~$0.12):
   ```bash
   python -m scripts.step_2_calibration_fit
   ```

5. **Run main experiment** (full table: 6 methods × 4 benches × 3 prompts × 3 seeds, ~6 h, ~$5):
   ```bash
   python -m scripts.step_3_main --main
   ```

   And the ablation (~2 h, ~$2):
   ```bash
   python -m scripts.step_4_ablation
   ```

6. **Aggregate and tabulate**:
   ```bash
   python -m scripts.step_5_aggregate
   python scripts/step_6_paper_tables.py
   ```

## Directory Structure

```
.
├── README.md                       (this file)
├── LICENSE                         (MIT)
├── .env.example                    (template for secrets)
├── .gitignore
├── requirements.txt
├── src/                            ← Core library
│   ├── azure_client.py             ← OpenAI-compatible Azure client (retry / fallback)
│   ├── metrics.py                  ← W1, JS-D, KL, TV, Acc, F1, EM, NLL, Brier, ECE
│   ├── prompts.py                  ← 4 task templates + 4 baselines (§13-14)
│   ├── retrieval.py                ← General + Hierarchical retriever abstractions
│   ├── prototypes.py               ← PrototypeCard / RuleBased builder / warning enforcement
│   ├── calibrate.py                ← T-scaling fit + per-bench × per-prompt × per-K
│   ├── leakage_check.py            ← Test/probe leakage detector
│   ├── io.py                       ← Unified parquet loader for all 4 benches
│   ├── build_retrieval_index.py    ← FAISS IndexFlatIP construction
│   ├── run_predictions.py          ← Main run driver + ThreadPoolExecutor dispatch
│   └── aggregate_results.py        ← Bootstrap CI + paired bootstrap + headroom-norm
├── scripts/                        ← End-to-end pipeline scripts
│   ├── build_evidence_embeddings.py
│   ├── step_2_calibration_fit.py
│   ├── step_3_main.py              (--smoke | --main)
│   ├── step_4_ablation.py
│   ├── step_5_aggregate.py
│   ├── step_6_paper_tables.py
│   ├── make_results_plots.py
│   └── disk_budget.py
├── data/scripts/                   ← Data preparation
│   ├── build_evidence_and_prototypes.py
│   ├── canonicalize_countries.py
│   └── eda_all.py
├── tests/                          ← Offline unit tests (175+, no LLM needed)
└── experiments/
    └── REPRODUCIBILITY.md          ← Full reproducibility guide
```

## Reproducing Paper Results

See `experiments/REPRODUCIBILITY.md` for:
- Exact step-by-step commands
- Dependency versions
- Random seeds (42, 123, 456)
- Hardware specs
- Expected wall time & cost
- Known caveats

## Testing

Run the offline test suite (no LLM calls; ~10 s):

```bash
pytest tests/
```

200+ tests cover metrics, prompts, retrieval, prototypes, calibration, and aggregation.

## Citation

```bibtex
@inproceedings{culturelens_rc_2026,
  title     = {CultureLens-RC: Retrieval-Calibrated Cultural Alignment across
               Global Opinions, Social Norms, and Everyday Knowledge},
  author    = {Anonymous},
  booktitle = {Proceedings of the Annual Meeting of the Association for
               Computational Linguistics (ACL)},
  year      = {2026},
}
```

## License

MIT License — see `LICENSE`.

## Acknowledgements

Built on top of: WorldValuesBench (Zeng et al., 2024), GlobalOpinionQA
(Durmus et al., 2023), NormAd (Rao et al., 2024), BLEnD (Myung et al., 2024),
sentence-transformers (Reimers & Gurevych, 2019), FAISS (Johnson et al.,
2019), and OpenAI's Python SDK.
