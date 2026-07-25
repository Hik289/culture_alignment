"""Offline tests for the materialized benchmark loaders."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import (
    BENCH_PATHS,
    BenchHandle,
    list_benchmarks,
    load_bench,
)


def _root():
    """Use the configured experiment root when available."""
    experiment_root = os.environ.get("EXPERIMENT_ROOT")
    if experiment_root:
        return Path(experiment_root)
    return ROOT


class TestListBenchmarks:
    def test_returns_all_keys(self):
        handles = list_benchmarks(root_dir=_root())
        names = {h.name for h in handles}
        assert names == set(BENCH_PATHS.keys())

    def test_at_least_one_exists(self):
        """Inspect materialized datasets when they are available."""
        handles = list_benchmarks(root_dir=_root())
        existing = [h for h in handles if h.exists]
        if not existing:
            pytest.skip("no materialized datasets are available")
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
    """Guard the schemas consumed by downstream evaluation code."""

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
