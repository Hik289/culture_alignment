"""统一加载 data/processed/ 下的 parquet (data_scientist 已物化).

设计:
- 单一入口 BENCH_PATHS 字典 + load_bench(name) 函数
- 路径默认相对项目根 (`.../culturealignment/`); 可通过 root_dir 覆盖 (hpc 上传 ${EXPERIMENT_ROOT}/)
- 不依赖 Azure / network, 仅 pandas + pyarrow
- 不修改原始数据, 只读

Benchmarks (data_scientist 输出 schema):
- wvb.probe_distributions : qid, continent, urb_rur, edu, n, distribution, sum, answer_options
- wvb.probe_manifest      : Question, Question Category, Continent, Urban / Rural, Education, D_INTERVIEW
- goqa                    : question_id, question, options_parsed, selections_parsed, n_options, n_countries, source, split
- normad                  : ID, Country, Background, Axis, Subaxis, Value, Rule-of-Thumb, Story, Explanation, Gold Label, story_words, story_id
- blend.mc                : MCQID, ID, country, prompt, choices, choice_countries, answer_idx
- blend.short             : country, qid, question_local, question_en, n_answer_groups, total_respondents
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError as e:  # noqa: F841
    pd = None  # type: ignore


# 项目根 (这个文件的父父级 = .../culturealignment)
_PKG_ROOT = Path(__file__).resolve().parents[1]


def _default_root() -> Path:
    """按优先级返回项目根:
    1. 环境变量 CULTUREALIGNMENT_ROOT
    2. ${EXPERIMENT_ROOT} (hpc 路径)
    3. _PKG_ROOT (本仓库相对路径, GCP 上)
    """
    env = os.environ.get("CULTUREALIGNMENT_ROOT")
    if env:
        return Path(env)
    hpc = Path("${EXPERIMENT_ROOT}")
    if hpc.exists():
        return hpc
    return _PKG_ROOT


# 相对 root 的 parquet 路径
BENCH_PATHS = {
    "wvb.probe_distributions": "data/processed/wvb/probe_distributions.parquet",
    "wvb.probe_manifest":      "data/processed/wvb/probe_manifest.parquet",
    "goqa":                    "data/processed/goqa/global_opinions_processed.parquet",
    "normad":                  "data/processed/normad/normad_processed.parquet",
    "blend.mc":                "data/processed/blend/blend_mc.parquet",
    "blend.short":             "data/processed/blend/blend_short_answer.parquet",
}


@dataclass
class BenchHandle:
    name: str
    path: Path
    exists: bool
    rows: int | None = None
    columns: list[str] | None = None


def list_benchmarks(root_dir: str | os.PathLike | None = None) -> list[BenchHandle]:
    """枚举所有已物化的 benchmark, 返回是否存在 + 行数 + 列名."""
    root = Path(root_dir) if root_dir else _default_root()
    out: list[BenchHandle] = []
    for name, rel in BENCH_PATHS.items():
        p = root / rel
        if p.exists() and pd is not None:
            try:
                # 用 pyarrow 仅读 metadata 不全加载
                import pyarrow.parquet as pq  # noqa: WPS433
                meta = pq.read_metadata(p)
                schema = pq.read_schema(p)
                out.append(BenchHandle(
                    name=name, path=p, exists=True,
                    rows=meta.num_rows, columns=list(schema.names),
                ))
                continue
            except Exception:
                pass
        out.append(BenchHandle(name=name, path=p, exists=p.exists()))
    return out


def load_bench(
    name: str,
    *,
    root_dir: str | os.PathLike | None = None,
    columns: Iterable[str] | None = None,
    n_rows: int | None = None,
):
    """加载指定 benchmark.

    Args:
        name: 见 BENCH_PATHS keys.
        root_dir: 覆盖默认项目根.
        columns: 仅加载子集列 (pyarrow projection, 内存友好).
        n_rows: 仅取前 n 行 (调试用).
    Returns:
        pd.DataFrame.
    """
    if pd is None:
        raise ImportError("需要 pandas; pip install pandas pyarrow")
    if name not in BENCH_PATHS:
        raise KeyError(
            f"未知 benchmark {name!r}; 可选: {sorted(BENCH_PATHS.keys())}"
        )
    root = Path(root_dir) if root_dir else _default_root()
    path = root / BENCH_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"parquet 不存在: {path}")

    kwargs: dict = {}
    if columns is not None:
        kwargs["columns"] = list(columns)
    df = pd.read_parquet(path, **kwargs)
    if n_rows is not None:
        df = df.head(n_rows)
    return df


def load_split(
    bench: str,
    split: str,
    *,
    root_dir: str | os.PathLike | None = None,
) -> "pd.DataFrame":
    """按 splits/ 目录的 split 文件过滤.

    支持的 split 文件:
      - data/splits/{bench}_split.{json,parquet,csv}
    若不存在, 抛 FileNotFoundError. 留待 data_scientist 落 split.
    """
    if pd is None:
        raise ImportError("需要 pandas")
    root = Path(root_dir) if root_dir else _default_root()
    # 优先 json (manifest 风格)
    candidates = [
        root / "data" / "splits" / f"{bench}_split.json",
        root / "data" / "splits" / f"{bench}_split.parquet",
        root / "data" / "splits" / f"{bench}_split.csv",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"split manifest 不存在 (tried: {[str(p) for p in candidates]})"
        )

    # 简单读取: 不在这里做复杂语义, 让上层 decide
    if found.suffix == ".json":
        import json
        data = json.loads(found.read_text())
        # 期望 schema: {split: [id, id, ...]} 或 [{"id":, "split":}, ...]
        if isinstance(data, dict) and split in data:
            ids = list(data[split])
            return pd.DataFrame({"id": ids, "split": split})
        if isinstance(data, list):
            df = pd.DataFrame(data)
            if "split" in df.columns:
                return df[df["split"] == split].reset_index(drop=True)
        raise ValueError(f"split json 格式不符合预期: {found}")

    if found.suffix == ".parquet":
        df = pd.read_parquet(found)
        if "split" in df.columns:
            return df[df["split"] == split].reset_index(drop=True)
        raise ValueError("split parquet 缺少 'split' 列")

    # csv
    df = pd.read_csv(found)
    if "split" in df.columns:
        return df[df["split"] == split].reset_index(drop=True)
    raise ValueError("split csv 缺少 'split' 列")


__all__ = [
    "BENCH_PATHS",
    "BenchHandle",
    "list_benchmarks",
    "load_bench",
    "load_split",
]
