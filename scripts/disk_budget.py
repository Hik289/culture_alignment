"""估算 culturealignment RUNNING 阶段各类产物的磁盘占用 (数量级).

基于:
- 4 个 benchmark (WVB, GOQA, NormAd, BLEnD)
- 6 个 method (no_culture / country / demographic / prototype / general_semantic / CultureLens-RC)
- 3 个 seed

输出: analysis/disk_budget.md (Markdown 表) + 控制台一行总结.

公式 / 假设全部明文写出, 便于 Researcher 改参数后重算.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 输入假设 (anchor_1+2+3+4 实际数字 + 文献规模)
# ---------------------------------------------------------------------------

# 全 test 规模 + Theorist EXP_DESIGN §4 子样原则:
# - WVB / GOQA: 全 test (≥3 seeds 即可, 不分层 boot)
# - NormAd / BLEnD: 子样 200-400 + bootstrap CI (Theorist power: σ≈0.35-0.4, n≥400 detect ΔW1=0.05)
BENCH_TEST_SIZE = {
    "wvb": 1656,         # 全 probe_distributions
    "goqa": 511,         # split=='test' 全量
    "normad": 400,       # Theorist 推荐子样 + bootstrap
    "blend_mc": 400,     # 子样 (每国家 ~ 20-30) + bootstrap
    "blend_saq": 200,    # 子样 + bootstrap
}
TOTAL_TEST_CALLS_PER_METHOD = sum(BENCH_TEST_SIZE.values())  # 8715 调用 / method / seed

METHODS = ["no_culture", "country", "demographic", "prototype", "general_semantic", "culturelens_rc"]
N_METHODS = len(METHODS)
N_SEEDS = 3
# EXP_DESIGN 新增: 主表 prompts × seeds; ablation 子集按 Theorist §4 分层
N_PROMPT_VARIANTS_MAIN = 3   # 主表 + calibration ablation
N_PROMPT_VARIANTS_FILTER = 1 # filtering 鲁棒性最稳, 单 var 足
# 总 main-table runs = 6 method × 3 prompt × 3 seed = 54

# anchor_3 + anchor_4 实测平均
AVG_PROMPT_TOK = 200
AVG_COMPLETION_TOK = 60

# 单次 chat 调用 raw response 大小 (JSON content + usage + finish_reason, 估)
# anchor_3 probe_log records 平均 ~600B per record → 0.6KB
AVG_RAW_RESPONSE_BYTES = 600
# 整条 record 含 parsed/probs/options/preview ~ 1.2KB
AVG_FULL_RECORD_BYTES = 1200
# 只保留 derived metric + small probability tensor (np.float32 K-dim, K avg ~5)
AVG_DERIVED_BYTES = 200  # qid + scalar + 5 float32 ≈ 100B + overhead

# 评估 derived metrics 表 (pandas → parquet): rows = bench size, cols ≈ 20
DERIVED_PARQUET_BYTES_PER_ROW = 250  # parquet snappy 压缩

# Evidence DB (从 raw 抽出 train 部分, 估计 30% 留作 evidence)
RAW_DATA_BYTES = {
    "wvb_raw": 424_000_000,
    "goqa_raw": 4_600_000,
    "normad_raw": 48_000_000,
    "blend_raw": 899_000_000,
}
EVIDENCE_FRACTION = 0.30  # 只抽 train 部分 + 文本字段
EVIDENCE_TEXT_ONLY_RATIO = 0.10  # 抽完文本后压缩比 (去掉元数据 / 不用字段)

# Embedding cache: 假设 sentence-transformers MPNet (768-d float32) 或 MiniLM (384-d)
# 选 384 平衡质量/盘
EMBEDDING_DIM = 384
EMBEDDING_BYTES_PER_VEC = EMBEDDING_DIM * 4  # float32

# FAISS index 额外开销 (IVF-PQ 量化时 ~1/16, flat 索引 ~ 1×; 取 flat 上限)
FAISS_INDEX_OVERHEAD = 1.05  # 5% 元数据

# Prototypes (每个国家 1 张卡, ~1KB JSON)
N_COUNTRIES = 65  # WVB 全 65 国
PROTOTYPE_BYTES_PER_CARD = 1500


# ---------------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------------

def _hr(n: int) -> str:
    """human-readable bytes."""
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def _estimate_evidence_db_bytes() -> int:
    """Evidence DB = train 部分 raw × evidence_fraction × text_only_ratio."""
    total_raw = sum(RAW_DATA_BYTES.values())
    return int(total_raw * EVIDENCE_FRACTION * EVIDENCE_TEXT_ONLY_RATIO)


def _estimate_embedding_cache_bytes(n_evidence_items: int) -> int:
    """每条 evidence 一个 dense vector."""
    return n_evidence_items * EMBEDDING_BYTES_PER_VEC


def _estimate_n_evidence_items() -> int:
    """证据条数 (粗估: evidence_db_bytes / 平均文本 250B = 文本条数)."""
    avg_text_bytes = 250
    return _estimate_evidence_db_bytes() // avg_text_bytes


def build_report() -> dict:
    n_evidence_items = _estimate_n_evidence_items()
    evidence_db_b = _estimate_evidence_db_bytes()
    embedding_cache_b = _estimate_embedding_cache_bytes(n_evidence_items)
    faiss_index_b = int(embedding_cache_b * FAISS_INDEX_OVERHEAD)
    prototype_b = N_COUNTRIES * PROTOTYPE_BYTES_PER_CARD

    # 每个 (method, seed) 跑出来的产物
    calls_per_run = TOTAL_TEST_CALLS_PER_METHOD
    raw_per_run_b = calls_per_run * AVG_FULL_RECORD_BYTES        # 完整 records
    derived_per_run_b = calls_per_run * AVG_DERIVED_BYTES        # 仅 derived
    parquet_per_run_b = calls_per_run * DERIVED_PARQUET_BYTES_PER_ROW

    # 主表 runs = 6 methods × 3 prompts × 3 seeds = 54 (Theorist EXP_DESIGN §4)
    total_runs = N_METHODS * N_PROMPT_VARIANTS_MAIN * N_SEEDS

    # 两种磁盘策略
    # 策略 A (naive, 不应用 Researcher 新规): 每 run 全保留 raw records
    strategy_a_total_b = (
        evidence_db_b
        + embedding_cache_b
        + faiss_index_b
        + prototype_b
        + raw_per_run_b * total_runs
    )

    # 策略 B (Researcher 新规: 跑完 (method, bench) 立刻清, 保留前 100 sanity + derived 全集)
    # sanity = 100 per (method, bench, prompt, seed) = 100×5×6×3×3 = 27000 records
    sanity_kept_records = 100 * 5 * N_METHODS * N_PROMPT_VARIANTS_MAIN * N_SEEDS
    sanity_b = sanity_kept_records * AVG_FULL_RECORD_BYTES
    strategy_b_total_b = (
        evidence_db_b
        + embedding_cache_b
        + faiss_index_b
        + prototype_b
        + derived_per_run_b * total_runs
        + sanity_b
    )

    # 策略 C: 新规 + sanity 只在 seed=42 + prompt_id=0 留, 其余 derived only
    sanity_final_only = 100 * 5 * N_METHODS  # 3000 records
    sanity_final_b = sanity_final_only * AVG_FULL_RECORD_BYTES
    strategy_c_total_b = (
        evidence_db_b
        + embedding_cache_b
        + faiss_index_b
        + prototype_b
        + derived_per_run_b * total_runs
        + sanity_final_b
    )

    return {
        "assumptions": {
            "benchmarks": BENCH_TEST_SIZE,
            "total_calls_per_method_per_seed": calls_per_run,
            "methods": METHODS,
            "n_methods": N_METHODS,
            "n_prompt_variants_main": N_PROMPT_VARIANTS_MAIN,
            "n_prompt_variants_filter": N_PROMPT_VARIANTS_FILTER,
            "n_seeds": N_SEEDS,
            "total_runs": total_runs,
            "total_llm_calls": calls_per_run * total_runs,
            "avg_prompt_tokens": AVG_PROMPT_TOK,
            "avg_completion_tokens": AVG_COMPLETION_TOK,
            "avg_full_record_bytes": AVG_FULL_RECORD_BYTES,
            "avg_derived_bytes": AVG_DERIVED_BYTES,
            "embedding_dim": EMBEDDING_DIM,
            "n_countries_for_proto": N_COUNTRIES,
            "raw_data_bytes": RAW_DATA_BYTES,
            "evidence_fraction": EVIDENCE_FRACTION,
            "evidence_text_only_ratio": EVIDENCE_TEXT_ONLY_RATIO,
        },
        "derived": {
            "n_evidence_items_est": n_evidence_items,
            "evidence_db_bytes": evidence_db_b,
            "embedding_cache_bytes": embedding_cache_b,
            "faiss_index_bytes": faiss_index_b,
            "prototype_cards_bytes": prototype_b,
            "raw_per_run_bytes": raw_per_run_b,
            "derived_per_run_bytes": derived_per_run_b,
            "parquet_per_run_bytes": parquet_per_run_b,
        },
        "strategies": {
            "A_naive_keep_all_raw": {
                "description": "每 run 全保留 raw records (不应用 Researcher 新规)",
                "total_bytes": strategy_a_total_b,
                "components": {
                    "evidence_db": _hr(evidence_db_b),
                    "embedding_cache": _hr(embedding_cache_b),
                    "faiss_index": _hr(faiss_index_b),
                    "prototypes": _hr(prototype_b),
                    "raw_records_all_runs": _hr(raw_per_run_b * total_runs),
                },
            },
            "B_director_rule": {
                "description": "Researcher 新规: prune raw, 每 (method, bench) 留前 100 sanity",
                "total_bytes": strategy_b_total_b,
                "components": {
                    "evidence_db": _hr(evidence_db_b),
                    "embedding_cache": _hr(embedding_cache_b),
                    "faiss_index": _hr(faiss_index_b),
                    "prototypes": _hr(prototype_b),
                    "derived_metrics_all_runs": _hr(derived_per_run_b * total_runs),
                    "sanity_records_100_per_method_bench_seed": _hr(sanity_b),
                },
            },
            "C_aggressive_prune": {
                "description": "新规 + sanity 只留 1 个 seed (42) 的, 其他 seed 仅 derived",
                "total_bytes": strategy_c_total_b,
                "components": {
                    "evidence_db": _hr(evidence_db_b),
                    "embedding_cache": _hr(embedding_cache_b),
                    "faiss_index": _hr(faiss_index_b),
                    "prototypes": _hr(prototype_b),
                    "derived_metrics_all_runs": _hr(derived_per_run_b * total_runs),
                    "sanity_records_seed42_only": _hr(sanity_final_b),
                },
            },
        },
    }


def render_md(report: dict) -> str:
    a = report["assumptions"]
    d = report["derived"]
    s = report["strategies"]

    lines = [
        "# Disk Budget — culturealignment RUNNING 阶段",
        "",
        "**作者**: ml_engineer",
        "**用途**: EXP_DESIGN 阶段决定产出路径策略 (/tmp vs persistent, 哪些 prune)",
        "",
        "## 输入假设",
        "",
        "- Benchmarks (test 子集行数, 由 data_scientist split.json 决定):",
        "  - WVB: {wvb:,}; GOQA: {goqa:,}; NormAd: {normad:,}; BLEnD MC: {blend_mc:,}; BLEnD SAQ: {blend_saq:,}".format(**a["benchmarks"]),
        f"- 单 method × 单 seed × 全 bench 调用数: **{a['total_calls_per_method_per_seed']:,}**",
        f"- 方法数: {a['n_methods']} ({', '.join(METHODS)})",
        f"- Seed 数: {a['n_seeds']}",
        f"- **总 run 数 (method × seed)**: {a['total_runs']}",
        f"- **总 LLM 调用 (全实验)**: {a['total_llm_calls']:,}",
        "",
        f"- 平均 prompt tokens: {a['avg_prompt_tokens']} (来源: anchor_3+4 实测)",
        f"- 平均 completion tokens: {a['avg_completion_tokens']}",
        f"- Full record bytes (含 parsed/probs/preview): {a['avg_full_record_bytes']}",
        f"- Derived bytes (仅 metric + 小概率向量): {a['avg_derived_bytes']}",
        f"- Embedding dim: {a['embedding_dim']} (MiniLM/MPNet 量级)",
        f"- 国家数 (prototype): {a['n_countries_for_proto']}",
        "",
        "## 估算量",
        "",
        "| 产物 | 量 |",
        "|------|----|",
        f"| Evidence DB (text only, 30% × 10% of raw) | **{_hr(d['evidence_db_bytes'])}** |",
        f"| 估计 evidence 条数 | {d['n_evidence_items_est']:,} |",
        f"| Embedding cache (384-d float32) | **{_hr(d['embedding_cache_bytes'])}** |",
        f"| FAISS index (flat + 5% 元数据) | {_hr(d['faiss_index_bytes'])} |",
        f"| Prototype cards (65 国家) | {_hr(d['prototype_cards_bytes'])} |",
        f"| Raw records / run | {_hr(d['raw_per_run_bytes'])} |",
        f"| Derived metric / run | {_hr(d['derived_per_run_bytes'])} |",
        f"| Parquet (derived all) / run | {_hr(d['parquet_per_run_bytes'])} |",
        "",
        "## 三策略对比",
        "",
    ]

    for key, info in s.items():
        lines.append(f"### Strategy {key}: {info['description']}")
        lines.append("")
        lines.append(f"**总占用估计: {_hr(info['total_bytes'])}**")
        lines.append("")
        lines.append("| 组件 | 大小 |")
        lines.append("|------|------|")
        for k, v in info["components"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines += [
        "## 推荐 (待 Researcher 在 EXP_DESIGN 确认)",
        "",
        "- **Persistent (${EXPERIMENT_ROOT}/)**:",
        "  - evidence DB (text only)",
        "  - prototype cards (65 KB)",
        "  - derived metrics parquet (all runs, ~ MB)",
        "  - assertions / report markdown",
        "  - sanity records (seed=42 only, ~ MB)",
        "- **Temp (/tmp/culturelens_rc, 重启丢失 OK)**:",
        "  - embedding cache (大 numpy)",
        "  - FAISS index (中间方法配置, 仅最终配置保留到 persistent)",
        "  - raw records (跑完 (method, bench) 立刻 prune)",
        "",
        "## 余量结论",
        "",
        "- gpu_server `/home` avail: 51 GB (实际 ~50 GB 余量, 因为 .cache 已占 45 GB, 但那是其他用户)",
        f"- Strategy B (Researcher 推荐): **{_hr(s['B_director_rule']['total_bytes'])}** → 单一数字, 完全在 gpu_server 单机可承受",
        f"- Strategy A (naive): **{_hr(s['A_naive_keep_all_raw']['total_bytes'])}** → 也仍在余量内, 但浪费",
        "- 即使是 Strategy A 也只有几百 MB 量级, **culturealignment 项目不会撞 gpu_server 磁盘墙**",
        "- 真正可能撞墙的: 如果换大模型 / 全 BLEnD 30 万行 / 加 thinking traces / dump LLM logprobs full vocab",
        "",
        "## 注意",
        "",
        "- 数字是数量级估算, 实际可能 ±50%",
        "- 假设全 BLEnD MC 用 5k 子样, 全 BLEnD SAQ 用 1k 子样 (而不是全 305k MC + 8k SAQ); 若改全数据 raw_per_run 会涨 ~60×",
        "- Embedding 384-d 是平衡选择; 用 MPNet 768-d 量级翻倍但仍 < 1 GB",
        "- FAISS IVF-PQ 量化可把 index 大小压到 1/16, 当前用 flat 上限",
    ]
    return "\n".join(lines)


def main():
    report = build_report()
    out_md = Path(__file__).resolve().parents[1] / "analysis" / "disk_budget.md"
    out_json = Path(__file__).resolve().parents[1] / "analysis" / "disk_budget.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(report))
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    # 一行控制台总结
    s = report["strategies"]
    print()
    print(f"Strategy A (naive):           {_hr(s['A_naive_keep_all_raw']['total_bytes'])}")
    print(f"Strategy B (Researcher rule):   {_hr(s['B_director_rule']['total_bytes'])}")
    print(f"Strategy C (aggressive):      {_hr(s['C_aggressive_prune']['total_bytes'])}")
    print("gpu_server avail: 51 GB (实际余量受其他用户 .cache 影响, ~50 GB free)")


if __name__ == "__main__":
    main()
