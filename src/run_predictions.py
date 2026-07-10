"""统一实验入口骨架 (RUNNING 阶段实施前的接口契约).

不实现 LLM 调用逻辑 (那是 RUNNING 阶段事). 仅:
- 命令行参数 (method / benchmark / prompt_variant / seed / 输出路径)
- 配置加载 (RunConfig)
- 子样选择 (data_scientist split.json 索引)
- 任务调度脚手架 (循环每个 item → 调 method → 拿 prediction → 拿 metric → 累计)
- 产物路径 + Researcher 新规 (sanity 100 + 大文件 prune)

EXP_DESIGN 后接入:
- method handlers (no_culture / country / demographic / prototype / general_semantic / culturelens_rc)
- 实际 LLM 调用 (src.model_client + src.prompts)
- 实际 metric 计算 (src.metrics)
- 实际 split 加载 (src.io.load_split)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# 支持的方法 (与 hypothesis_tree H0.method_* + readme §14 baseline 对齐)
METHODS = (
    "no_culture",        # §14.1 baseline
    "country",           # §14.2 baseline
    "demographic",       # §14.3 baseline (仅 WVB/GOQA 有人口字段)
    "prototype",         # §14.4 baseline + H0.method_prototype 主体
    "general_semantic",  # §14.5 general retrieval (filter off)
    "culturelens_rc",    # §14.6 主方法 = filter + prototype + calibration
)
BENCHMARKS = ("wvb", "goqa", "normad", "blend_mc", "blend_saq")


@dataclass
class RunConfig:
    """单次 run 的全部参数. 写到 results.json 的 'config' 字段保证可复现."""
    method: str
    benchmark: str
    prompt_variant: int = 0        # 0,1,2 → 三个 prompt 措辞变体 (Theorist §4)
    seed: int = 42                 # numpy/random 全局种子
    n_subsample: Optional[int] = None  # None = 全 test; 非 None 时按 seed 子样
    use_calibration: bool = True   # T-scaling on/off (calibration ablation)
    use_filter: bool = True        # hierarchical filter on/off (filtering ablation)
    use_prototype: bool = True     # prototype on/off (prototype ablation)
    top_k: int = 8                 # 检索 top-k
    out_dir: str = "experiments/runs"
    model: str = os.environ.get("LLM_MODEL", "your-model-name")
    # 三模块 on/off 速记 (写入文件名)
    @property
    def config_id(self) -> str:
        f = "F" if self.use_filter else "f"
        p = "P" if self.use_prototype else "p"
        c = "C" if self.use_calibration else "c"
        return f"{self.method}_{self.benchmark}_pv{self.prompt_variant}_seed{self.seed}_{f}{p}{c}"

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {METHODS}")
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {self.benchmark!r}; choose from {BENCHMARKS}")
        if self.prompt_variant not in (0, 1, 2):
            raise ValueError("prompt_variant must be 0/1/2")
        if self.seed < 0:
            raise ValueError("seed >= 0")
        if self.top_k <= 0:
            raise ValueError("top_k > 0")


# ---------------------------------------------------------------------------
# Method registry (placeholder; 真正实现等 RUNNING 阶段)
# ---------------------------------------------------------------------------

MethodHandler = Callable[[dict, RunConfig], dict]
"""(item, config) -> {pred_distribution / pred_label / pred_text + meta}."""

_METHOD_REGISTRY: dict[str, MethodHandler] = {}


def register_method(name: str):
    def deco(fn: MethodHandler):
        if name in _METHOD_REGISTRY:
            raise ValueError(f"method {name!r} already registered")
        _METHOD_REGISTRY[name] = fn
        return fn
    return deco


def get_method(name: str) -> MethodHandler:
    if name not in _METHOD_REGISTRY:
        raise NotImplementedError(
            f"method {name!r} not registered; will be implemented in RUNNING phase"
        )
    return _METHOD_REGISTRY[name]


# ---------------------------------------------------------------------------
# Driver (骨架; 不调 LLM)
# ---------------------------------------------------------------------------

@dataclass
class RunArtifacts:
    """单次 run 产物."""
    config: RunConfig
    n_items: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    sanity_records: list[dict] = field(default_factory=list)  # 前 100 条
    derived_records: list[dict] = field(default_factory=list)  # 小型 derived per item

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "n_items": self.n_items,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "metrics": self.metrics,
            "n_sanity": len(self.sanity_records),
            "n_derived": len(self.derived_records),
        }


def load_items(config: RunConfig) -> list[dict]:
    """加载 benchmark test split 的 items. 骨架: 仅声明 contract.

    EXP_DESIGN 后接 src.io.load_bench + src.io.load_split('test') + 子样.
    返回 list[dict], 每个 dict 至少含:
      - item_id: str
      - bench-specific fields (question, options, gold, country, etc.)
    """
    raise NotImplementedError(
        "load_items 等 RUNNING 阶段接入 src.io.load_bench + split filter"
    )


def run_one(config: RunConfig) -> RunArtifacts:
    """单次 run: 遍历 items, 调 method, 收 metric, 写 sanity.

    骨架: 仅声明流程, 实际 LLM 调用 / metric 算 / sanity prune 留给 RUNNING 阶段填充.
    """
    config.validate()
    arts = RunArtifacts(config=config)
    # items = load_items(config)
    # arts.n_items = len(items)
    # method = get_method(config.method)
    # for i, item in enumerate(items):
    #     pred = method(item, config)
    #     if pred is None:
    #         arts.n_failed += 1; continue
    #     arts.n_succeeded += 1
    #     metric = compute_metrics(pred, item, bench=config.benchmark)
    #     arts.derived_records.append(metric)
    #     if i < 100:
    #         arts.sanity_records.append({**item, **pred, **metric})
    # if config.use_calibration:
    #     arts.metrics = apply_calibration_and_aggregate(...)
    # else:
    #     arts.metrics = aggregate(arts.derived_records)
    raise NotImplementedError("run_one 等 RUNNING 阶段实现")


def dump_artifacts(arts: RunArtifacts, base_dir: str | os.PathLike) -> dict[str, Path]:
    """写产物到 disk. Researcher 新规:
      - {config_id}.json : 配置 + metrics 汇总 (小, persist)
      - {config_id}_sanity.jsonl : 前 100 records (持 seed=42 + prompt_id=0; 其余可省)
      - {config_id}_derived.parquet : per-item metrics + small probs (上层 aggregate 用)
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    cid = arts.config.config_id
    out: dict[str, Path] = {}
    summary_path = base / f"{cid}.json"
    summary_path.write_text(json.dumps(arts.to_dict(), ensure_ascii=False, indent=2))
    out["summary"] = summary_path

    # sanity 只在 seed=42 + prompt_variant=0 时落盘 (Strategy C 的写法; Researcher 决定后切换)
    if arts.config.seed == 42 and arts.config.prompt_variant == 0:
        sanity_path = base / f"{cid}_sanity.jsonl"
        with sanity_path.open("w") as f:
            for r in arts.sanity_records[:100]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out["sanity"] = sanity_path

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(description="culturealignment run driver")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    p.add_argument("--prompt-variant", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-subsample", type=int, default=None)
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--no-prototype", action="store_true")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-dir", default="experiments/runs")
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "your-model-name"))
    a = p.parse_args(argv)
    return RunConfig(
        method=a.method,
        benchmark=a.benchmark,
        prompt_variant=a.prompt_variant,
        seed=a.seed,
        n_subsample=a.n_subsample,
        use_calibration=not a.no_calibration,
        use_filter=not a.no_filter,
        use_prototype=not a.no_prototype,
        top_k=a.top_k,
        out_dir=a.out_dir,
        model=a.model,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = parse_args(argv)
    config.validate()
    logger.info("run config: %s", config.config_id)
    arts = run_one(config)
    paths = dump_artifacts(arts, config.out_dir)
    logger.info("wrote: %s", paths)


# ---------------------------------------------------------------------------
# 并发 batch dispatch (Researcher: ThreadPoolExecutor mock LLM 验证)
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence


def dispatch_items(
    items: Sequence[dict],
    handler: Callable[[dict], dict],
    *,
    max_workers: int = 8,
    on_error: str = "record",  # "record" | "raise" | "skip"
) -> list[dict]:
    """并发对每个 item 调用 handler.

    Args:
        items: 每个 dict 必须含 'item_id' (作 stable key, 用于结果重排)
        handler: item -> result dict (含 'item_id')
        max_workers: 并发数
        on_error: 失败处理
            - record: 在 results 里加 {item_id, ok: False, error: ...}
            - raise: 抛出
            - skip: 静默跳过

    Returns:
        list[dict], 按输入 items 顺序排列 (即使并发完成顺序不同)
    """
    if not items:
        return []

    by_id: dict[str, dict] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for it in items:
            iid = it.get("item_id")
            if iid is None:
                raise ValueError("item must have 'item_id' field")
            fut = ex.submit(handler, it)
            futures[fut] = iid

        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                res = fut.result()
                if "item_id" not in res:
                    res = {**res, "item_id": iid}
                by_id[iid] = res
            except Exception as e:  # noqa: BLE001
                if on_error == "raise":
                    raise
                if on_error == "skip":
                    continue
                by_id[iid] = {
                    "item_id": iid, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }

    # 按输入顺序输出
    return [by_id[it["item_id"]] for it in items if it["item_id"] in by_id]


def dispatch_items_with_rate_limit(
    items: Sequence[dict],
    handler: Callable[[dict], dict],
    *,
    max_workers: int = 8,
    max_per_second: float = 0.0,  # 0 = no limit
    on_error: str = "record",
) -> list[dict]:
    """带 rate limit 的并发 dispatch.

    Model API rate limit 可能是 RPM / TPM, 这里只做简单 RPS 限制.
    max_per_second > 0 时, 每提交一个任务前 sleep 间隔.
    """
    import time

    if max_per_second <= 0:
        return dispatch_items(items, handler, max_workers=max_workers, on_error=on_error)

    if not items:
        return []

    interval = 1.0 / max_per_second
    by_id: dict[str, dict] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for it in items:
            iid = it.get("item_id")
            if iid is None:
                raise ValueError("item must have 'item_id' field")
            fut = ex.submit(handler, it)
            futures[fut] = iid
            time.sleep(interval)

        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                res = fut.result()
                if "item_id" not in res:
                    res = {**res, "item_id": iid}
                by_id[iid] = res
            except Exception as e:  # noqa: BLE001
                if on_error == "raise":
                    raise
                if on_error == "skip":
                    continue
                by_id[iid] = {
                    "item_id": iid, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }

    return [by_id[it["item_id"]] for it in items if it["item_id"] in by_id]


__all__ = [
    "METHODS", "BENCHMARKS",
    "RunConfig", "RunArtifacts",
    "register_method", "get_method",
    "load_items", "run_one", "dump_artifacts",
    "dispatch_items", "dispatch_items_with_rate_limit",
    "parse_args", "main",
]


if __name__ == "__main__":
    main(sys.argv[1:])
