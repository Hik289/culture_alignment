# Artifact Guide

This guide maps the public `culture_alignment` repository to a reviewer-friendly artifact workflow for `CultureLens-RC`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `experiments/`: Experiment drivers, ablations, and benchmark-specific runners.
- `tests/`: Local tests or smoke checks for fresh checkouts.
- `data/`: Small fixtures, schemas, manifests, or data-layout notes; large data should stay outside git.
- `figures/`: README and paper-facing figures.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `.env.example`: Template for local credentials or backend configuration.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
python -m pytest tests -q
```

## Reproduction And Analysis Entry Points

These are the main tracked files to inspect for paper-scale or benchmark-scale reproduction. Some require arguments, credentials, downloaded benchmarks, or local data paths described in the README.

- `python experiments/anchor_3_azure_endpoint/probe.py`
- `python scripts/build_evidence_embeddings.py`
- `python scripts/make_results_plots.py`
- `python scripts/step_2_calibration_fit.py`
- `python scripts/step_3_main.py`
- `python scripts/step_4_ablation.py`
- `python scripts/step_5_aggregate.py`
- `python scripts/step_6_paper_tables.py`

## Data Layout Notes

- `data/README.md`

## Figure Assets

- `figures/culturelens_rc_overview.png`
- `figures/fig_intuition_cross_country_pollution.png`

## Data, Credentials, And Generated Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
