# Artifact Guide

Operational notes for reproducing `CultureLens-RC` from the public `culture_alignment` repository.

## Review Path

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `experiments/`: Experiment drivers, ablations, and benchmark-specific runners.
- `tests/`: Local tests or smoke checks for fresh checkouts.
- `data/`: Small fixtures, schemas, manifests, or data-layout notes; large data should stay outside git.
- `figures/`: README and paper-facing figures.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `.env.example`: Template for local credentials or backend configuration.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
python -m pytest tests -q
```

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

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

## Data And Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
