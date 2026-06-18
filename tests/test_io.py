"""src/io.py 离线测试.

不依赖 Azure; 但**依赖 hpc 上的 data/processed/*.parquet**.
所以只在 hpc 上跑会全部通过; GCP 上 data_scientist 还没物化时部分会 skip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import (  # noqa: E402
    BENCH_PATHS,
    BenchHandle,
    list_benchmarks,
    load_bench,
)


def _root():
    """优先选 hpc 路径 (跑测试在 hpc 上); 否则用本仓库根."""
    hpc = Path("${EXPERIMENT_ROOT}")
    if hpc.exists():
        return hpc
    return ROOT


class TestListBenchmarks:
    def test_returns_all_keys(self):
        handles = list_benchmarks(root_dir=_root())
        names = {h.name for h in handles}
        assert names == set(BENCH_PATHS.keys())

    def test_at_least_one_exists(self):
        """data_scientist 已物化, 至少有一个."""
        handles = list_benchmarks(root_dir=_root())
        existing = [h for h in handles if h.exists]
        if not existing:
            pytest.skip("data_scientist 物化未完成, 跳过")
        assert all(isinstance(h, BenchHandle) for h in existing)
        assert all(h.rows is None or h.rows > 0 for h in existing)


class TestLoadBench:
    @pytest.mark.parametrize("name", list(BENCH_PATHS.keys()))
    def test_load_first_rows(self, name):
        root = _root()
        p = root / BENCH_PATHS[name]
        if not p.exists():
            pytest.skip(f"未物化: {p}")
        df = load_bench(name, root_dir=root, n_rows=5)
        assert len(df) > 0
        assert len(df) <= 5
        assert list(df.columns)  # 非空

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            load_bench("nope")

    def test_columns_projection(self):
        name = "wvb.probe_distributions"
        p = _root() / BENCH_PATHS[name]
        if not p.exists():
            pytest.skip("未物化")
        df = load_bench(name, root_dir=_root(), columns=["qid", "n"], n_rows=3)
        assert set(df.columns) == {"qid", "n"}


class TestBenchSchemas:
    """data_scientist schema 约束 — 改 schema 必更新这里, 防止下游静默 break."""

    def test_wvb_distributions_schema(self):
        name = "wvb.probe_distributions"
        p = _root() / BENCH_PATHS[name]
        if not p.exists():
            pytest.skip()
        df = load_bench(name, root_dir=_root(), n_rows=1)
        required = {"qid", "continent", "urb_rur", "edu", "n", "distribution", "sum", "answer_options"}
        assert required.issubset(set(df.columns)), df.columns

    def test_goqa_schema(self):
        name = "goqa"
        p = _root() / BENCH_PATHS[name]
        if not p.exists():
            pytest.skip()
        df = load_bench(name, root_dir=_root(), n_rows=1)
        required = {"question_id", "question", "options_parsed", "selections_parsed",
                    "n_options", "n_countries", "source", "split"}
        assert required.issubset(set(df.columns)), df.columns

    def test_normad_schema(self):
        name = "normad"
        p = _root() / BENCH_PATHS[name]
        if not p.exists():
            pytest.skip()
        df = load_bench(name, root_dir=_root(), n_rows=1)
        required = {"ID", "Country", "Story", "Gold Label", "story_id"}
        assert required.issubset(set(df.columns)), df.columns

    def test_blend_mc_schema(self):
        name = "blend.mc"
        p = _root() / BENCH_PATHS[name]
        if not p.exists():
            pytest.skip()
        df = load_bench(name, root_dir=_root(), n_rows=1)
        required = {"MCQID", "ID", "country", "prompt", "choices", "answer_idx"}
        assert required.issubset(set(df.columns)), df.columns

    def test_blend_short_schema(self):
        name = "blend.short"
        p = _root() / BENCH_PATHS[name]
        if not p.exists():
            pytest.skip()
        df = load_bench(name, root_dir=_root(), n_rows=1)
        required = {"country", "qid", "question_en", "n_answer_groups", "total_respondents"}
        assert required.issubset(set(df.columns)), df.columns
