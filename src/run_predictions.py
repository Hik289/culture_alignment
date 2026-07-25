"""Shared configuration, dispatch, and artifact contracts for experiment runs.

Method handlers and benchmark loading are explicit integration points; calling
an unregistered or incomplete path fails instead of producing partial results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Methods correspond to the baselines and main method in README §14.
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
    """Complete configuration persisted with one experiment run."""
    method: str
    benchmark: str
    prompt_variant: int = 0
    seed: int = 42
    n_subsample: int | None = None
    use_calibration: bool = True
    use_filter: bool = True
    use_prototype: bool = True
    top_k: int = 8
    out_dir: str = "experiments/runs"
    model: str = os.environ.get("LLM_MODEL", "your-model-name")
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
            f"method {name!r} has no registered handler"
        )
    return _METHOD_REGISTRY[name]


@dataclass
class RunArtifacts:
    """Artifacts collected for one run."""
    config: RunConfig
    n_items: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    sanity_records: list[dict] = field(default_factory=list)
    derived_records: list[dict] = field(default_factory=list)

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
    """Load test items for ``config``; each item must include ``item_id``."""
    raise NotImplementedError(
        "load_items requires a benchmark adapter using src.io loaders"
    )


def run_one(config: RunConfig) -> RunArtifacts:
    """Run one configuration through its registered benchmark adapter."""
    config.validate()
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
    raise NotImplementedError("run_one requires benchmark and metric adapters")


def dump_artifacts(arts: RunArtifacts, base_dir: str | os.PathLike) -> dict[str, Path]:
    """Persist the summary and a bounded sanity sample for the default run."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    cid = arts.config.config_id
    out: dict[str, Path] = {}
    summary_path = base / f"{cid}.json"
    summary_path.write_text(json.dumps(arts.to_dict(), ensure_ascii=False, indent=2))
    out["summary"] = summary_path

    # Retain a bounded sanity sample only for the canonical seed and prompt.
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


from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed


def dispatch_items(
    items: Sequence[dict],
    handler: Callable[[dict], dict],
    *,
    max_workers: int = 8,
    on_error: str = "record",  # "record" | "raise" | "skip"
) -> list[dict]:
    """Dispatch items concurrently while preserving input order.

    Args:
        items: Each item must contain a stable ``item_id``.
        handler: Maps an item to a result dictionary.
        max_workers: Maximum number of concurrent calls.
        on_error: ``record``, ``raise``, or ``skip``.

    Returns:
        Results in input order.
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
            except Exception as e:
                if on_error == "raise":
                    raise
                if on_error == "skip":
                    continue
                by_id[iid] = {
                    "item_id": iid, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }

    return [by_id[it["item_id"]] for it in items if it["item_id"] in by_id]


def dispatch_items_with_rate_limit(
    items: Sequence[dict],
    handler: Callable[[dict], dict],
    *,
    max_workers: int = 8,
    max_per_second: float = 0.0,  # 0 = no limit
    on_error: str = "record",
) -> list[dict]:
    """Dispatch concurrently with an optional request-per-second limit."""
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
            except Exception as e:
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
    "BENCHMARKS",
    "METHODS",
    "RunArtifacts",
    "RunConfig",
    "dispatch_items",
    "dispatch_items_with_rate_limit",
    "dump_artifacts",
    "get_method",
    "load_items",
    "main",
    "parse_args",
    "register_method",
    "run_one",
]


if __name__ == "__main__":
    main(sys.argv[1:])
