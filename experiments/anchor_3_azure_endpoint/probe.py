"""H_anchor_3: Azure gpt-5.4-mini endpoint 可用性 probe.

debug_move: COMPARE + ASSERT
invariant: 10 个样例 prompt 全部返回符合 JSON schema 的概率分布; sum 容忍 ±1e-3 (归一后 sum=1).
prediction: ≥9/10 直接合规; 剩余在 2 次重试内合规; 总失败率 < 5%.

四类输出 (覆盖 readme §11 任务自适应输出):
  A. 调查分布 (ordinal options 上的概率分布)
  B. 单标签分类 (Accept / Not Accept)
  C. 多类分类 (多 label 概率)
  D. 短答案 (字符串)

每类 2-3 prompt, 共 10 个.

运行 (hpc 上, 已 azure login):
  python -m experiments.anchor_3_azure_endpoint.probe
输出:
  experiments/anchor_3_azure_endpoint/probe_log.json (详细)
  analysis/anchor_3_assertions.json (汇总)
  analysis/anchor_3_cost_estimate.md (成本估算)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model_client import MODEL_NAME, chat_json  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("anchor3.probe")

JST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# 10 个样例 prompt, 覆盖 4 类输出
# ---------------------------------------------------------------------------

SYSTEM = "You are a careful annotator. Always respond with strictly valid JSON matching the requested schema. Probabilities must be non-negative and sum to 1.0 (within 1e-3)."


def _dist_prompt(question: str, options: list[str]) -> dict:
    schema_desc = (
        "Return JSON: {\"probs\": [p_1, ..., p_K]} where p_i corresponds to option i in order, "
        f"K={len(options)}, all p_i >= 0, sum = 1.0."
    )
    user = (
        f"Question: {question}\n"
        f"Options (in order): {options}\n"
        f"{schema_desc}"
    )
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ], "expected_keys": ["probs"], "K": len(options), "type": "distribution"}


def _binary_prompt(scenario: str) -> dict:
    user = (
        f"Scenario: {scenario}\n"
        "Decide if this behavior is socially acceptable in a generic modern context. "
        "Return JSON: {\"label\": \"acceptable\" or \"not_acceptable\", "
        "\"probs\": {\"acceptable\": p1, \"not_acceptable\": p2}} with p1+p2=1."
    )
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ], "expected_keys": ["label", "probs"], "type": "binary"}


def _multiclass_prompt(question: str, options: list[str]) -> dict:
    user = (
        f"Question: {question}\n"
        f"Choose from: {options}\n"
        "Return JSON: {\"answer\": <one of the options>, "
        "\"probs\": {<option>: p, ...}} where probs sums to 1.0."
    )
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ], "expected_keys": ["answer", "probs"], "options": options, "type": "multiclass"}


def _short_answer_prompt(question: str) -> dict:
    user = (
        f"Question: {question}\n"
        "Return JSON: {\"answer\": \"<short string>\"}."
    )
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ], "expected_keys": ["answer"], "type": "short_answer"}


PROMPTS = [
    # === A. 分布 (3 个) ===
    {"id": "A1", **_dist_prompt(
        "On a scale 1 to 4, how important is family in your life? (1=not important, 4=very important)",
        ["1", "2", "3", "4"],
    )},
    {"id": "A2", **_dist_prompt(
        "On a scale 1 to 5, how much do you trust the government? (1=no trust, 5=full trust)",
        ["1", "2", "3", "4", "5"],
    )},
    {"id": "A3", **_dist_prompt(
        "How often do you attend religious services?",
        ["never", "rarely", "monthly", "weekly", "daily"],
    )},
    # === B. 二分类 (3 个) ===
    {"id": "B1", **_binary_prompt(
        "A person eats with their left hand at a formal dinner."
    )},
    {"id": "B2", **_binary_prompt(
        "A guest removes shoes before entering a host's home."
    )},
    {"id": "B3", **_binary_prompt(
        "A person interrupts an elder mid-sentence to correct a fact."
    )},
    # === C. 多类 (2 个) ===
    {"id": "C1", **_multiclass_prompt(
        "Which food is most commonly eaten for breakfast in Japan?",
        ["rice and miso soup", "croissant", "tortilla", "pancakes"],
    )},
    {"id": "C2", **_multiclass_prompt(
        "Which is the official language of Brazil?",
        ["Spanish", "Portuguese", "English", "French"],
    )},
    # === D. 短答案 (2 个) ===
    {"id": "D1", **_short_answer_prompt(
        "What is the capital city of Australia?"
    )},
    {"id": "D2", **_short_answer_prompt(
        "Name a traditional Korean fermented dish."
    )},
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SUM_TOL = 1e-3


def _validate(parsed: dict, prompt: dict) -> tuple[bool, str]:
    """检查解析后的 dict 是否符合期望 schema."""
    if not isinstance(parsed, dict):
        return False, f"parsed 不是 dict: {type(parsed).__name__}"

    for k in prompt["expected_keys"]:
        if k not in parsed:
            return False, f"缺少键 {k!r}"

    t = prompt["type"]
    if t == "distribution":
        probs = parsed["probs"]
        if not isinstance(probs, list):
            return False, f"probs 不是 list: {type(probs).__name__}"
        if len(probs) != prompt["K"]:
            return False, f"probs 长度 {len(probs)} != K={prompt['K']}"
        try:
            probs = [float(x) for x in probs]
        except (TypeError, ValueError) as e:
            return False, f"probs 含非数值: {e}"
        if any(p < -1e-9 for p in probs):
            return False, f"probs 含负数: min={min(probs)}"
        s = sum(probs)
        if abs(s - 1.0) > SUM_TOL:
            return False, f"probs sum={s:.6f}, 偏离 1.0 超过 {SUM_TOL}"
        return True, ""

    if t == "binary":
        label = parsed["label"]
        if label not in ("acceptable", "not_acceptable"):
            return False, f"label={label!r} 不在 {{acceptable, not_acceptable}}"
        probs = parsed["probs"]
        if not isinstance(probs, dict):
            return False, "probs 不是 dict"
        if set(probs.keys()) != {"acceptable", "not_acceptable"}:
            return False, f"probs keys={list(probs.keys())} 不匹配"
        s = float(probs["acceptable"]) + float(probs["not_acceptable"])
        if abs(s - 1.0) > SUM_TOL:
            return False, f"probs sum={s:.6f}, 偏离 1.0 超过 {SUM_TOL}"
        return True, ""

    if t == "multiclass":
        opts = prompt["options"]
        ans = parsed["answer"]
        if ans not in opts:
            return False, f"answer={ans!r} 不在 options={opts}"
        probs = parsed["probs"]
        if not isinstance(probs, dict):
            return False, "probs 不是 dict"
        # 允许 key 与 options 集合相同
        if set(probs.keys()) != set(opts):
            return False, f"probs keys={set(probs.keys())} != options={set(opts)}"
        s = sum(float(v) for v in probs.values())
        if abs(s - 1.0) > SUM_TOL:
            return False, f"probs sum={s:.6f}, 偏离 1.0 超过 {SUM_TOL}"
        return True, ""

    if t == "short_answer":
        ans = parsed["answer"]
        if not isinstance(ans, str) or len(ans.strip()) == 0:
            return False, f"answer 非空字符串失败: {ans!r}"
        return True, ""

    return False, f"未知 type={t}"


# ---------------------------------------------------------------------------
# 成本估算 (gpt-5.4-mini 公开价格未确定; 用 gpt-4o-mini 量级占位, Researcher 知情)
# ---------------------------------------------------------------------------

# 假设价格 (USD per 1M tokens). gpt-5.4-mini 当前没有公开公告价, 用 gpt-4o-mini ($0.15 in / $0.60 out) 作占位.
# 真实价格出来后只需改这两行.
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60


def _cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1e6) * PRICE_INPUT_PER_M + (completion_tokens / 1e6) * PRICE_OUTPUT_PER_M


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "anchor": "H_anchor_3",
        "model": MODEL_NAME,
        "timestamp_jst": datetime.now(JST).isoformat(),
        "endpoint": "${AZURE_OPENAI_ENDPOINT}",
        "auth": "DefaultAzureCredential + bearer token (cognitiveservices scope)",
        "n_prompts": len(PROMPTS),
        "max_retries": 2,
        "sum_tolerance": SUM_TOL,
        "price_assumption": {
            "input_per_M_usd": PRICE_INPUT_PER_M,
            "output_per_M_usd": PRICE_OUTPUT_PER_M,
            "note": "占位价格 (gpt-4o-mini 量级); 待 Azure gpt-5.4-mini 正式价格出来后调整",
        },
        "results": [],
    }

    n_first_try = 0
    n_with_retry = 0
    n_failed = 0
    tot_prompt = 0
    tot_completion = 0
    tot_latency = 0.0

    for p in PROMPTS:
        logger.info("=== prompt %s (type=%s) ===", p["id"], p["type"])
        res = chat_json(
            messages=p["messages"],
            max_retries=2,
            response_format={"type": "json_object"},
        )
        ok_schema, schema_err = (False, "no parsed")
        if res.ok and res.parsed is not None:
            ok_schema, schema_err = _validate(res.parsed, p)

        item = {
            "id": p["id"],
            "type": p["type"],
            "api_ok": bool(res.ok),
            "schema_ok": bool(ok_schema),
            "schema_error": schema_err if not ok_schema else None,
            "n_attempts": res.n_attempts,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.total_tokens,
            "latency_seconds": round(res.latency_seconds, 3),
            "first_try_ok": (res.n_attempts == 1 and res.ok and ok_schema),
            "raw_content_preview": (res.content or "")[:300],
            "parsed": res.parsed if ok_schema else None,
            "error": res.error,
        }
        log["results"].append(item)

        tot_prompt += res.prompt_tokens
        tot_completion += res.completion_tokens
        tot_latency += res.latency_seconds

        if item["first_try_ok"]:
            n_first_try += 1
        if res.ok and ok_schema:
            if res.n_attempts > 1:
                n_with_retry += 1
        if not (res.ok and ok_schema):
            n_failed += 1
            logger.warning("prompt %s 失败: api_ok=%s schema_err=%s err=%s",
                           p["id"], res.ok, schema_err, res.error)

    N = len(PROMPTS)
    log["summary"] = {
        "n_total": N,
        "n_first_try_ok": n_first_try,
        "n_passed_with_retry": n_with_retry,
        "n_failed": n_failed,
        "first_try_pass_rate": n_first_try / N,
        "total_pass_rate": (n_first_try + n_with_retry) / N,
        "failure_rate": n_failed / N,
        "avg_prompt_tokens": tot_prompt / N,
        "avg_completion_tokens": tot_completion / N,
        "avg_latency_seconds": tot_latency / N,
        "total_cost_usd_estimate": _cost_usd(tot_prompt, tot_completion),
    }

    # invariant 检查
    inv = {
        "≥9/10 直接合规": n_first_try >= 9,
        "总失败率 < 5%": (n_failed / N) < 0.05,
        "剩余在 2 次重试内合规": (n_failed == 0),
    }
    log["invariants"] = inv

    probe_path = out_dir / "probe_log.json"
    probe_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    logger.info("写入: %s", probe_path)

    # 小 report → analysis/
    assertions = {
        "anchor": "H_anchor_3",
        "hypothesis": "Azure gpt-5.4-mini endpoint 可用 (DefaultAzureCredential + JSON schema)",
        "timestamp_jst": log["timestamp_jst"],
        "model": MODEL_NAME,
        "n_prompts": N,
        "summary": log["summary"],
        "invariants": inv,
        "verdict": "PASS" if all(inv.values()) else "FAIL",
    }
    (analysis_dir / "anchor_3_assertions.json").write_text(
        json.dumps(assertions, indent=2, ensure_ascii=False)
    )
    logger.info("写入: %s", analysis_dir / "anchor_3_assertions.json")

    cost_md = _build_cost_md(log)
    (analysis_dir / "anchor_3_cost_estimate.md").write_text(cost_md)
    logger.info("写入: %s", analysis_dir / "anchor_3_cost_estimate.md")

    print(json.dumps(log["summary"], indent=2, ensure_ascii=False))
    print("invariants:", json.dumps(inv, ensure_ascii=False))

    # 如果 invariant 全部不满足, 退出非 0 让 shell 端可见
    if not all(inv.values()):
        sys.exit(1)


def _build_cost_md(log: dict) -> str:
    s = log["summary"]
    pa = log["price_assumption"]
    lines = [
        "# H_anchor_3 成本估算",
        "",
        f"- 时间 (JST): {log['timestamp_jst']}",
        f"- 模型: {log['model']}",
        f"- 样本量: {s['n_total']}",
        "",
        "## Token / Latency 平均值",
        "",
        f"- 平均 prompt tokens: {s['avg_prompt_tokens']:.1f}",
        f"- 平均 completion tokens: {s['avg_completion_tokens']:.1f}",
        f"- 平均延迟 (秒): {s['avg_latency_seconds']:.2f}",
        "",
        "## 价格假设",
        "",
        f"- input ${pa['input_per_M_usd']:.4f} / 1M tokens",
        f"- output ${pa['output_per_M_usd']:.4f} / 1M tokens",
        f"- 备注: {pa['note']}",
        "",
        "## 本次 probe 总成本",
        "",
        f"- 总成本估计: ${s['total_cost_usd_estimate']:.6f} USD",
        "",
        "## 假设规模外推",
        "",
        "若以下游 benchmark 规模 (WVB ~10k 调用, GOQA ~5k, NormAd ~3k, BLEnD ~5k = 23k 调用) 计算, ",
        "并假设 token 用量与本次 probe 相同, 单次完整评测 (1 个 seed) 成本约:",
        "",
    ]
    avg_cost_per_call = s["total_cost_usd_estimate"] / s["n_total"] if s["n_total"] else 0
    for n_calls, label in [(23_000, "1 seed 全 benchmark"), (69_000, "3 seeds 全 benchmark")]:
        lines.append(f"- {label} ({n_calls:,} 调用): ${avg_cost_per_call * n_calls:.2f} USD")
    lines.append("")
    lines.append("**注**: 真实价格出来后只需更新 probe.py 中 PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
