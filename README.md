# CultureLens-RC

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/quickstart-5%20step%20pipeline-2ea44f" alt="Quickstart"></a>
  <a href="#intuition"><img src="https://img.shields.io/badge/intuition-cross--country%20retrieval-7c3aed" alt="Intuition"></a>
  <a href="#environment-setup"><img src="https://img.shields.io/badge/setup-Azure%20%2B%20FAISS-f97316" alt="Environment setup"></a>
  <a href="#reproducing-paper-results"><img src="https://img.shields.io/badge/reproduce-paper%20results-2563eb" alt="Reproduce paper results"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/citation-BibTeX-64748b" alt="Citation"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/method-retrieval--calibrated%20alignment-lightgrey" alt="Retrieval-calibrated alignment">
  <img src="https://img.shields.io/badge/benchmarks-WVB%20%7C%20GOQA%20%7C%20NormAd%20%7C%20BLEnD-lightgrey" alt="Benchmarks">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
</p>

**Retrieval-Calibrated Cultural Alignment across Global Opinions, Social Norms, and Everyday Knowledge**

CultureLens-RC is an **inference-only** framework that aligns large language models with cultural variation by composing three orthogonal modules: (1) hierarchical country/topic/task **filtering** over a unified evidence database, (2) per-country **prototype** summaries built from training-split evidence, and (3) temperature **calibration** fit on a held-out dev split.

We evaluate on four heterogeneous benchmarks: WorldValuesBench, GlobalOpinionQA, NormAd-Eti, and BLEnD, covering survey distributions, social-norm judgments, and everyday cultural knowledge.

## Intuition

<p align="center">
  <img src="figures/fig_intuition_cross_country_pollution.png" width="900" alt="Cross-country pollution problem in general semantic retrieval">
</p>

General semantic retrieval can surface evidence from the wrong country or region when questions are semantically similar. CultureLens-RC reduces this cross-country pollution by filtering retrieval through the target country, topic, and task before prompting the model.

## Highlights

- Hierarchical evidence filtering for country-aware and topic-aware retrieval.
- Per-country prototype cards built only from training-split evidence.
- Temperature calibration fit on held-out dev data for over-confident argmax tasks.
- Four benchmark families: survey distributions, global opinions, social norms, and everyday cultural knowledge.
- Largest reported gains appear in low-resource cultural contexts: +7.3 pp on GOQA low-resource countries and +16.7 pp on BLEnD-MC low-resource regions.

## Overview

```text
Question + Country + Topic
        |
        v
+------------------------------------------------------------+
| 1. Hierarchical Filter     country/topic evidence filtering |
| 2. Top-k Retrieval         FAISS dense, MiniLM-L6 384d      |
| 3. Prototype Card          per-country aggregate summary    |
| 4. LLM Prompt + Answer     Azure gpt-* deployment           |
| 5. Temperature Calibration T-scaling per bench/prompt/K     |
+------------------------------------------------------------+
```

## Environment Setup

Python 3.11+ is recommended.

```bash
git clone git@github.com:Hik289/culture_alignment.git
cd culture_alignment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set up API keys:

```bash
cp .env.example .env
$EDITOR .env
```

Required and optional variables:

- `AZURE_OPENAI_API_KEY`: Azure AI Foundry deployment key.
- `AZURE_OPENAI_ENDPOINT`: Azure endpoint URL.
- `AZURE_DEPLOYMENT_NAME`: model deployment name, for example `gpt-4o-mini`.
- `GEMINI_API_KEY`: optional, only for regenerating pipeline figures.

Never commit `.env` or local credential files.

## Quick Start

1. Download benchmark data.
   See [data/README.md](data/README.md) for WorldValuesBench, GlobalOpinionQA, NormAd, and BLEnD download instructions.

2. Build the evidence database and country prototypes from train splits.

   ```bash
   python data/scripts/build_evidence_and_prototypes.py
   ```

3. Build the FAISS retrieval index.

   ```bash
   python scripts/build_evidence_embeddings.py \
     --evidence data/evidence/evidence.jsonl \
     --out-dir /tmp/culturelens_rc/embeddings \
     --embedder sentence_transformers
   ```

4. Fit temperature calibration on dev splits.

   ```bash
   python -m scripts.step_2_calibration_fit
   ```

5. Run the main experiment and ablation.

   ```bash
   python -m scripts.step_3_main --main
   python -m scripts.step_4_ablation
   ```

6. Aggregate and tabulate.

   ```bash
   python -m scripts.step_5_aggregate
   python scripts/step_6_paper_tables.py
   ```

## Directory Structure

```text
.
|-- README.md
|-- LICENSE
|-- .env.example
|-- requirements.txt
|-- figures/
|   `-- fig_intuition_cross_country_pollution.png
|-- src/                            # Core library
|   |-- azure_client.py             # OpenAI-compatible Azure client
|   |-- metrics.py                  # W1, JS-D, KL, TV, Acc, F1, EM, NLL, Brier, ECE
|   |-- prompts.py                  # Task templates and baselines
|   |-- retrieval.py                # General and hierarchical retrievers
|   |-- prototypes.py               # Prototype cards and warning enforcement
|   |-- calibrate.py                # Temperature scaling
|   |-- leakage_check.py            # Test/probe leakage detector
|   |-- io.py                       # Unified parquet loader
|   |-- build_retrieval_index.py    # FAISS IndexFlatIP construction
|   |-- run_predictions.py          # Main run driver
|   `-- aggregate_results.py        # Bootstrap CI and aggregation
|-- scripts/                        # End-to-end pipeline scripts
|-- data/scripts/                   # Data preparation
|-- tests/                          # Offline unit tests
`-- experiments/
    `-- REPRODUCIBILITY.md          # Full reproducibility guide
```

## Reproducing Paper Results

See [experiments/REPRODUCIBILITY.md](experiments/REPRODUCIBILITY.md) for exact commands, dependency versions, random seeds, hardware specs, expected wall time and cost, and known caveats.

## Testing

Run the offline test suite with no LLM calls:

```bash
pytest tests/
```

The tests cover metrics, prompts, retrieval, prototypes, calibration, and aggregation.

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

MIT License. See [LICENSE](LICENSE).

## Acknowledgements

Built on top of WorldValuesBench (Zeng et al., 2024), GlobalOpinionQA (Durmus et al., 2023), NormAd (Rao et al., 2024), BLEnD (Myung et al., 2024), sentence-transformers (Reimers and Gurevych, 2019), FAISS (Johnson et al., 2019), and OpenAI's Python SDK.
