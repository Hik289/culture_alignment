"""Read the materialized benchmark files under ``data/processed``.

The loaders are read-only and work offline. Paths default to the repository
root and may be overridden explicitly or through an environment variable.

Materialized benchmark schemas:
- wvb.probe_distributions : qid, continent, urb_rur, edu, n, distribution, sum, answer_options
- wvb.probe_manifest      : Question, Question Category, Continent, Urban / Rural, Education, D_INTERVIEW
- goqa                    : question_id, question, options_parsed, selections_parsed, n_options, n_countries, source, split
- normad                  : ID, Country, Background, Axis, Subaxis, Value, Rule-of-Thumb, Story, Explanation, Gold Label, story_words, story_id
- blend.mc                : MCQID, ID, country, prompt, choices, choice_countries, answer_idx
- blend.short             : country, qid, question_local, question_en, n_answer_groups, total_respondents
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:  # noqa: F841
    pd = None  # type: ignore


_PKG_ROOT = Path(__file__).resolve().parents[1]


def _default_root() -> Path:
    """Resolve the data root in priority order.

    1. ``CULTUREALIGNMENT_ROOT``
    2. ``EXPERIMENT_ROOT``
    3. The repository root
    """
    culture_root = os.environ.get("CULTUREALIGNMENT_ROOT")
    if culture_root:
        return Path(culture_root)
    experiment_root = os.environ.get("EXPERIMENT_ROOT")
    if experiment_root:
        return Path(experiment_root)
    return _PKG_ROOT


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
    """List benchmark availability and, when possible, Parquet metadata."""
    root = Path(root_dir) if root_dir else _default_root()
    out: list[BenchHandle] = []
    for name, rel in BENCH_PATHS.items():
        p = root / rel
        if p.exists() and pd is not None:
            try:
                # Read only Parquet metadata to avoid loading each dataset.
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
    """Load a materialized benchmark.

    Args:
        name: A key in ``BENCH_PATHS``.
        root_dir: Optional data-root override.
        columns: Optional column projection.
        n_rows: Return at most this many rows.
    Returns:
        pd.DataFrame.
    """
    if name not in BENCH_PATHS:
        raise KeyError(
            f"unknown benchmark {name!r}; choose from {sorted(BENCH_PATHS)}"
        )
    if pd is None:
        raise ImportError("pandas and pyarrow are required to load benchmarks")
    root = Path(root_dir) if root_dir else _default_root()
    path = root / BENCH_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")

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
) -> pd.DataFrame:
    """Load one split from a manifest under ``data/splits``.

    支持的 split 文件:
      - data/splits/{bench}_split.{json,parquet,csv}
    The manifest may be JSON, Parquet, or CSV.
    """
    if pd is None:
        raise ImportError("pandas is required to load split manifests")
    root = Path(root_dir) if root_dir else _default_root()
    # Prefer JSON manifests when several formats are present.
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
        # Accepted forms: {split: [id, ...]} or [{"id": ..., "split": ...}, ...].
        if isinstance(data, dict) and split in data:
            ids = list(data[split])
            return pd.DataFrame({"id": ids, "split": split})
        if isinstance(data, list):
            df = pd.DataFrame(data)
            if "split" in df.columns:
                return df[df["split"] == split].reset_index(drop=True)
        raise ValueError(f"unsupported split JSON schema: {found}")

    if found.suffix == ".parquet":
        df = pd.read_parquet(found)
        if "split" in df.columns:
            return df[df["split"] == split].reset_index(drop=True)
        raise ValueError("split Parquet file is missing the 'split' column")

    df = pd.read_csv(found)
    if "split" in df.columns:
        return df[df["split"] == split].reset_index(drop=True)
    raise ValueError("split CSV file is missing the 'split' column")


__all__ = [
    "BENCH_PATHS",
    "BenchHandle",
    "list_benchmarks",
    "load_bench",
    "load_split",
]
