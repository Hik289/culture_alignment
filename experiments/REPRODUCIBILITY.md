# Reproducibility — CultureLens-RC (RUNNING phase)

**ml_engineer** | 2026-06-17 JST

Complete commands + environment + seeds for reproducing the RUNNING-phase experiments end-to-end.

---

## 1. Environment

**Machine**: `gpu_server` (GPU Server)
- OS: Ubuntu 20.04 (Linux 5.4.0-216-generic)
- CPU: x86_64
- GPU: 4× NVIDIA RTX 2080 Ti (driver 535.230.02; only used for sentence-transformers encode)
- Python: 3.11.0 (conda env `culturelens_rc_env`)

**Activate environment**:
```bash
ssh gpu_server
source ${CONDA_ROOT}/bin/activate culturelens_rc_env
# or directly: python
```

**Key dependencies** (`pip list | grep -iE 'openai|azure|sentence|faiss|numpy|pandas'`):
```
faiss-cpu                 1.14.3
numpy                     1.26.4
openai                    2.38.0
pandas                    2.3.3
pyarrow                   23.0.1
pytest                    9.0.3
sentence-transformers     2.7.0
```

**Azure LLM**:
- Model: `gpt-5.4-mini` via Azure AI Foundry
- Endpoint: `${AZURE_OPENAI_ENDPOINT}`
- Auth: `AZURE_API_KEY` env var (direct api_key, NOT DefaultAzureCredential — researcher-authorized override of idea.md §4)
- ⚠️ API key NEVER committed to git; passed only via env var at process start.

---

## 2. Random Seeds

| Use | Seed(s) |
|-----|---------|
| Sample selection (Step 3, Step 4) | 42, 123, 456 |
| Calibration fit (Step 2) | 42 |
| Bootstrap CI (Step 5) | 42 |
| Embedding (Step 1, IdentityEmbedder fallback only) | hash-based deterministic |
| LLM temperature | 0 (default, not passed to gpt-5 reasoning model per SDK constraint) |

---

## 3. Step-by-step commands

All commands run on `gpu_server` with `culturelens_rc_env` active, in `${EXPERIMENT_ROOT}/`.

### Step 1: Build FAISS retrieval index (Wall: 18s; LLM calls: 0)
```bash
cd ${EXPERIMENT_ROOT}

# Install faiss-cpu first (one-time)
${PYTHON_ENV}/bin/pip install faiss-cpu

# Build embeddings + FAISS index from data_scientist's evidence.jsonl
python -m scripts.build_evidence_embeddings \
  --evidence ${EXPERIMENT_ROOT}/data/evidence/evidence.jsonl \
  --out-dir /tmp/culturelens_rc/embeddings \
  --embedder sentence_transformers \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128

# Outputs:
#   /tmp/culturelens_rc/embeddings/embeddings.npy   (70 MB, float32, 46018×384)
#   /tmp/culturelens_rc/embeddings/general.faiss    (70 MB, IndexFlatIP)
#   /tmp/culturelens_rc/embeddings/items_meta.jsonl (30 MB, evidence metadata)
```

### Step 2: T-scaling fit on dev (Wall: 6.4 min; LLM calls: 1,323; Cost: $0.12)
```bash
export AZURE_API_KEY='<your_key>'

python -m scripts.step_2_calibration_fit

# Outputs:
#   experiments/calibration_fits/calibrations.json     (full per-K per-prompt T*)
#   experiments/calibration_fits/step_2_sanity.json   (sample records)
#   analysis/step_2_calibrations.json                  (compact for downstream)
```

### Step 3: Main table (Wall: 5h 41min; LLM calls: 81,432; Cost: $5.37)
```bash
export AZURE_API_KEY='<your_key>'

# Background run (270 cells × 6 methods × 3 prompts × 3 seeds × 5 bench)
nohup python -m scripts.step_3_main --main \
  > experiments/step_3_main/main.log 2>&1 &

# Outputs (one JSON per cell):
#   experiments/step_3_main/runs/{method}_{bench}_pv{0|1|2}_seed{42|123|456}.json
#   experiments/step_3_main/progress.json (resumable, 6h auto-checkpoint)
```

### Step 4: Ablation (Wall: 2h; LLM calls: ~27,000; Cost: $1.81)
```bash
export AZURE_API_KEY='<your_key>'

nohup python -m scripts.step_4_ablation \
  > experiments/step_4_ablation/main.log 2>&1 &

# Outputs:
#   experiments/step_4_ablation/runs/{combo}_{bench}_pv{0|1|2}_seed{42|123|456}.json
#     combo ∈ {fPC, FpC, FPc}
#   experiments/step_4_ablation/progress.json
```

### Step 5: Aggregate (Wall: < 1s; LLM calls: 0; Cost: $0)
```bash
python -m scripts.step_5_aggregate

# Outputs:
#   analysis/main_results.json       (per-bench × per-method, mean ± 95% CI, paired bootstrap p-value)
#   analysis/ablation_results.json   (ablation + tier_stratified)
#   analysis/running_report.md       (auto-generated detailed tables)
```

### Step 6 (ANALYSIS): Paper tables (Wall: < 1s; Cost: $0)
```bash
python scripts/step_6_paper_tables.py

# Outputs:
#   analysis/paper_tables.md   (Table 1 main + Table 2 ablation + Table 3 tier + Table 4 cost)
#   analysis/paper_tables.tex  (Table 1 + Table 3 LaTeX/booktabs)
```

---

## 4. Data versions

See `analysis/data_versions.json` for full sha256 hashes of raw data sources.

Summary:
- **WVB**: WVS Wave 7 v6.0 official CSV, downloaded 2026-06-15 (sha256 in data_versions.json)
- **GOQA**: HF `Anthropic/llm_global_opinions` dataset
- **NormAd**: official repo `Akhila-Yerukola/NormAd`
- **BLEnD**: official HF dataset `nayeon212/BLEnD`
- **Evidence DB**: built from train splits only (zero test/probe leakage, verified by `analysis/leakage_report.json`)

---

## 5. Method definitions (exact specification)

All methods use the same Azure `gpt-5.4-mini` endpoint, JSON-mode response, max_retries=2.

| Method | Prompt | Retrieval | Prototype | Calibration |
|--------|--------|-----------|-----------|-------------|
| no_culture | §14.1 baseline | none | none | none |
| country | §14.2 baseline | none | none | none |
| demographic | §14.3 baseline (fallback to country if no demo field) | none | none | none |
| prototype | §14.4 baseline | none | data_sci card | none |
| general_semantic | country prompt + top-5 evidence | FAISS top-5 (no filter) | none | none |
| culturelens_rc | prototype prompt + top-5 evidence | Hierarchical (country×topic filter) top-5 | data_sci card | per-bench per-prompt per-K T* |

Ablation combos (Step 4):
- `fPC`: filter OFF (general semantic retrieval) + proto ON + calib ON
- `FpC`: filter ON + proto OFF + calib ON
- `FPc`: filter ON + proto ON + calib OFF
- `FPC`: full = culturelens_rc in main table

---

## 6. Aggregate compute

| Step | LLM Calls | Cost | Wall |
|------|-----------|------|------|
| 1 (build index) | 0 | $0.00 | 18s |
| 2 (calibration fit) | 1,323 | $0.12 | 6.4min |
| 3 (main table 270 cells) | 81,432 | $5.37 | 5h 41min |
| 4 (ablation 84 cells) | ~27,000 | $1.81 | 2h |
| 5 (aggregate) | 0 | $0.00 | < 1s |
| 6 (paper tables) | 0 | $0.00 | < 1s |
| **Total** | **~110,000** | **$7.30** | **~8h** |

**Errors**: 0 / 110,000+ LLM calls (max_retries=2 救回所有 stochastic JSON parse failures).

---

## 7. Hardware-specific notes

- `gpu_server` is a multi-user research server; concurrent users may slow throughput. Our wall times measured during typical usage (CPU load on average 40-60%).
- The 8-concurrent worker default (`ThreadPoolExecutor(max_workers=8)`) was empirically robust against Azure rate limits without explicit RPS throttling.
- FAISS index uses `IndexFlatIP` (exact cosine via dot product on L2-normalized vectors); IVF/PQ not needed for 46k×384 embeddings.

---

## 8. Replay / Resume

All long-running steps (3, 4) write incremental progress JSON. If interrupted:
- The same command **resumes from the last completed cell** (idempotent).
- Each cell's output file presence is the canonical signal for "completed".

To force re-run a single cell:
```bash
rm experiments/step_3_main/runs/{cell_id}.json
# then re-run the step command — only the missing cell will be re-executed
```

---

## 9. Known limitations / caveats

- **gpt-5.4-mini pricing**: Used `gpt-4o-mini` rate ($0.15 in / $0.60 out per 1M tokens) as placeholder. Real prices may differ; update `PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M` constants in `experiments/anchor_3_azure_endpoint/probe.py` and re-run `scripts/step_5_aggregate.py` for accurate cost.
- **Temperature not passed**: gpt-5 reasoning model rejects non-default temperature via OpenAI SDK. `src/azure_client.py` detects model name prefix and skips. Actual sampling temperature controlled by model defaults.
- **WVB W1 high variance**: σ ≈ 0.66 in main table, attributed to ordinal scale mismatch (integer support 1..K vs literature's [0,1] normalization). H0.anchor_4_diag opened by Researcher for future investigation.
- **BLEnD MC tier high-resource**: data_scientist's tier groupings sparse on the high-resource side for BLEnD MC; tier-stratified table reports only low-resource for that bench (high-resource n=0 cells in tier breakdown).

---

## 10. Contact

- ml_engineer (RUNNING orchestration)
- data_scientist (data + evidence + prototypes + tier groupings)
- theorist (insight + hypothesis tree)
- Researcher / Anonymous Researcher (human PI)
