"""EXP_DESIGN 阶段 3 个骨架的契约测试 (不调 LLM, 不需要 evidence)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aggregate_results import (
    RunSummary,
    aggregate_runs,
    bootstrap_ci,
    compare_table_markdown,
    compare_to_baseline,
    headroom_normalized,
    load_run_summaries,
    main_table_markdown,
    paired_bootstrap_diff,
)
from src.calibrate import (
    CalibrationResult,
    apply_per_benchmark,
    apply_temperature,
    fit_per_benchmark,
    fit_temperature,
    negative_log_likelihood_from_probs,
)
from src.run_predictions import (
    RunArtifacts,
    RunConfig,
    dump_artifacts,
    get_method,
    parse_args,
    register_method,
)

# ---------------------------------------------------------------------------
# run_predictions
# ---------------------------------------------------------------------------

class TestRunConfig:
    def test_valid(self):
        c = RunConfig(method="no_culture", benchmark="wvb")
        c.validate()
        assert c.seed == 42
        assert "no_culture_wvb_pv0_seed42_FPC" in c.config_id

    def test_unknown_method_raises(self):
        c = RunConfig(method="nope", benchmark="wvb")
        with pytest.raises(ValueError):
            c.validate()

    def test_unknown_benchmark_raises(self):
        c = RunConfig(method="no_culture", benchmark="nope")
        with pytest.raises(ValueError):
            c.validate()

    def test_invalid_prompt_variant(self):
        c = RunConfig(method="no_culture", benchmark="wvb", prompt_variant=5)
        with pytest.raises(ValueError):
            c.validate()

    def test_config_id_encodes_flags(self):
        c = RunConfig(method="culturelens_rc", benchmark="goqa",
                      prompt_variant=2, seed=123,
                      use_calibration=False, use_filter=False, use_prototype=True)
        assert "fPc" in c.config_id  # filter off, proto on, calib off
        assert "pv2" in c.config_id
        assert "seed123" in c.config_id


class TestMethodRegistry:
    def test_register_and_get(self):
        @register_method("test_method_1")
        def handler(item, cfg):
            return {"pred": "x"}
        assert get_method("test_method_1") is handler

    def test_get_unregistered_raises(self):
        with pytest.raises(NotImplementedError):
            get_method("nonexistent_xxx_yyy")

    def test_register_duplicate_raises(self):
        @register_method("dup_test")
        def h1(item, cfg):
            return {}
        with pytest.raises(ValueError):
            @register_method("dup_test")
            def h2(item, cfg):
                return {}


class TestParseArgs:
    def test_minimal(self):
        c = parse_args(["--method", "no_culture", "--benchmark", "wvb"])
        assert c.method == "no_culture"
        assert c.benchmark == "wvb"
        assert c.use_filter and c.use_prototype and c.use_calibration

    def test_ablation_flags(self):
        c = parse_args([
            "--method", "culturelens_rc", "--benchmark", "goqa",
            "--no-filter", "--no-prototype", "--no-calibration",
            "--prompt-variant", "2", "--seed", "7",
        ])
        assert not c.use_filter
        assert not c.use_prototype
        assert not c.use_calibration
        assert c.prompt_variant == 2
        assert c.seed == 7

    def test_invalid_choice_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--method", "bogus", "--benchmark", "wvb"])


class TestDumpArtifacts:
    def test_dumps_summary(self):
        c = RunConfig(method="no_culture", benchmark="wvb")
        arts = RunArtifacts(config=c, n_items=10, n_succeeded=10, n_failed=0,
                            metrics={"W1_mean": 0.3},
                            sanity_records=[{"item_id": "x", "pred": 1}] * 5)
        with tempfile.TemporaryDirectory() as td:
            paths = dump_artifacts(arts, td)
            assert "summary" in paths
            data = json.loads(paths["summary"].read_text())
            assert data["n_items"] == 10
            assert data["metrics"]["W1_mean"] == 0.3
            # seed=42 + pv=0 → sanity 落盘
            assert "sanity" in paths

    def test_no_sanity_for_non_default_seed(self):
        c = RunConfig(method="no_culture", benchmark="wvb", seed=99)
        arts = RunArtifacts(config=c, sanity_records=[{"x": 1}])
        with tempfile.TemporaryDirectory() as td:
            paths = dump_artifacts(arts, td)
            assert "sanity" not in paths


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

class TestApplyTemperature:
    def test_T1_identity(self):
        p = np.array([0.2, 0.3, 0.5])
        out = apply_temperature(p, 1.0)
        np.testing.assert_allclose(out, p, atol=1e-6)

    def test_T_large_smooths(self):
        """T → ∞ 应趋向 uniform."""
        p = np.array([0.9, 0.05, 0.05])
        out = apply_temperature(p, 100.0)
        # 应该非常接近 uniform 1/3
        np.testing.assert_allclose(out, [1/3, 1/3, 1/3], atol=0.01)

    def test_T_small_sharpens(self):
        """T → 0 应趋向 one-hot on argmax."""
        p = np.array([0.4, 0.35, 0.25])
        out = apply_temperature(p, 0.01)
        assert out[0] > 0.99

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError):
            apply_temperature(np.array([0.5, 0.5]), 0.0)

    def test_2d_batch(self):
        p = np.array([[0.5, 0.5], [0.9, 0.1]])
        out = apply_temperature(p, 1.0)
        np.testing.assert_allclose(out.sum(axis=-1), [1.0, 1.0], atol=1e-6)


class TestNLL:
    def test_perfect(self):
        p = np.array([[1.0, 0.0], [0.0, 1.0]])
        nll = negative_log_likelihood_from_probs(p, [0, 1], eps=1e-12)
        assert nll < 1e-6

    def test_uniform(self):
        p = np.array([[0.5, 0.5]] * 4)
        nll = negative_log_likelihood_from_probs(p, [0, 1, 0, 1])
        assert abs(nll - np.log(2)) < 1e-6

    def test_shape_check(self):
        with pytest.raises(ValueError):
            negative_log_likelihood_from_probs(np.array([0.5, 0.5]), [0])


class TestFitTemperature:
    def test_overconfident_finds_T_gt_1(self):
        """构造 over-confident 错预测: argmax 概率高但常错 → 最优 T > 1 平滑."""
        rng = np.random.default_rng(0)
        N, K = 200, 3
        probs = rng.dirichlet(alpha=[0.3] * K, size=N)  # 尖锐分布
        y_true = rng.integers(0, K, size=N)  # 随机 label (与 probs 无关 → over-confident)
        res = fit_temperature(probs, y_true.tolist())
        assert isinstance(res, CalibrationResult)
        assert res.T_star > 1.0  # 应平滑
        assert res.nll_after <= res.nll_before + 1e-6
        assert res.n_dev_samples == N

    def test_calibrated_T_near_1(self):
        """已校准的分布 T* 应在 1 附近.

        构造 well-calibrated 数据: probs 任意, y 按 probs 采样, 不是 argmax.
        当 y 按 probs 分布采样时, raw probs 已经是 calibrated → T*≈1.
        """
        rng = np.random.default_rng(1)
        N, K = 500, 3
        logits = rng.normal(size=(N, K))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        # 关键: 按 probs 采样 y, 不是 argmax (后者使 raw probs over-confident, T*>1)
        y = np.array([rng.choice(K, p=probs[i]) for i in range(N)])
        res = fit_temperature(probs, y.tolist())
        # well-calibrated 数据上 T* 应接近 1 (允许较宽容差因为有限样本)
        assert 0.5 < res.T_star < 2.0, f"T*={res.T_star} 应接近 1"


class TestFitPerBenchmark:
    def test_runs_each_bench(self):
        rng = np.random.default_rng(42)
        dev = {
            "wvb": (rng.dirichlet([0.3] * 4, size=50), rng.integers(0, 4, size=50).tolist()),
            "goqa": (rng.dirichlet([0.5] * 3, size=30), rng.integers(0, 3, size=30).tolist()),
        }
        out = fit_per_benchmark(dev)
        assert set(out.keys()) == {"wvb", "goqa"}
        for r in out.values():
            assert r.T_star > 0
            assert r.n_dev_samples > 0


class TestApplyPerBenchmark:
    def test_pass_through_when_missing(self):
        probs = {"wvb": np.array([[0.5, 0.5]]), "goqa": np.array([[0.1, 0.9]])}
        cal = {"wvb": CalibrationResult(T_star=1.0, nll_before=1.0, nll_after=1.0, n_dev_samples=1)}
        out = apply_per_benchmark(probs, cal)
        np.testing.assert_allclose(out["wvb"], [[0.5, 0.5]], atol=1e-6)
        np.testing.assert_allclose(out["goqa"], [[0.1, 0.9]], atol=1e-6)  # 无 cal 透传


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

def _mk_summary(method, bench, pv=0, seed=42, **metrics) -> RunSummary:
    cfg_id = f"{method}_{bench}_pv{pv}_seed{seed}_FPC"
    return RunSummary(
        method=method, benchmark=bench, prompt_variant=pv, seed=seed,
        use_calibration=True, use_filter=True, use_prototype=True,
        n_items=100, metrics=metrics, config_id=cfg_id,
    )


class TestBootstrapCI:
    def test_single_value(self):
        m, lo, hi = bootstrap_ci([0.5])
        assert m == lo == hi == 0.5

    def test_empty(self):
        m, lo, hi = bootstrap_ci([])
        assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)

    def test_mean_in_ci(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(loc=0.5, scale=0.1, size=50).tolist()
        m, lo, hi = bootstrap_ci(vals, n_boot=500, seed=0)
        assert lo <= m <= hi
        assert abs(m - 0.5) < 0.05  # 大概率


class TestPairedBootstrap:
    def test_zero_diff(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        out = paired_bootstrap_diff(a, a, n_boot=200, seed=0)
        assert abs(out["mean_diff"]) < 1e-9
        # p value 在 0 处应该 = 2 * min(0, 0) = 0, 但我们 mean 全 0 →
        # 这里实际上 boot 全 0, p = 0
        assert 0 <= out["p_value_two_sided"] <= 1

    def test_positive_diff(self):
        a = [0.5, 0.6, 0.7, 0.8, 0.9]
        b = [0.1, 0.2, 0.3, 0.4, 0.5]
        out = paired_bootstrap_diff(a, b, n_boot=500, seed=0)
        assert out["mean_diff"] > 0
        assert out["lo"] > 0  # CI 不跨 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([1.0], [1.0, 2.0])


class TestAggregateRuns:
    def test_groups_by_method_bench_metric(self):
        summaries = [
            _mk_summary("a", "wvb", seed=42, W1_mean=0.3),
            _mk_summary("a", "wvb", seed=123, W1_mean=0.32),
            _mk_summary("a", "wvb", seed=456, W1_mean=0.28),
            _mk_summary("b", "wvb", seed=42, W1_mean=0.4),
        ]
        rows = aggregate_runs(summaries, metric_keys=["W1_mean"], n_boot=200)
        assert len(rows) == 2  # (a,wvb,W1) + (b,wvb,W1)
        a_row = next(r for r in rows if r["method"] == "a")
        assert a_row["n_runs"] == 3
        assert abs(a_row["mean"] - 0.3) < 0.05


class TestHeadroomNormalized:
    def test_lower_is_better(self):
        # baseline 0.4, method 0.3, oracle 0 → headroom 0.4, Δ 0.1, ratio 0.25
        hn = headroom_normalized(0.3, 0.4, "W1_mean")
        assert abs(hn - 0.25) < 1e-6

    def test_higher_is_better(self):
        # baseline 0.6, method 0.7, oracle 1.0 → headroom 0.4, Δ 0.1, ratio 0.25
        hn = headroom_normalized(0.7, 0.6, "Accuracy")
        assert abs(hn - 0.25) < 1e-6

    def test_zero_headroom_returns_none(self):
        hn = headroom_normalized(1.0, 1.0, "Accuracy")
        assert hn is None

    def test_unknown_metric_returns_none(self):
        hn = headroom_normalized(0.5, 0.4, "nope_metric")
        assert hn is None


class TestCompareToBaseline:
    def test_pair_by_prompt_seed(self):
        summaries = [
            _mk_summary("culturelens_rc", "wvb", pv=0, seed=42, W1_mean=0.20),
            _mk_summary("culturelens_rc", "wvb", pv=0, seed=123, W1_mean=0.22),
            _mk_summary("no_culture",     "wvb", pv=0, seed=42, W1_mean=0.35),
            _mk_summary("no_culture",     "wvb", pv=0, seed=123, W1_mean=0.37),
            _mk_summary("no_culture",     "wvb", pv=1, seed=42, W1_mean=0.40),  # 无配对
        ]
        rows = compare_to_baseline(summaries, n_boot=500)
        wvb_w1 = next(r for r in rows if r["benchmark"] == "wvb" and r["metric"] == "W1_mean")
        assert wvb_w1["n_pairs"] == 2  # 只有 (pv=0, seed=42), (pv=0, seed=123) 配对
        # method 0.21 (mean), baseline 0.36 → diff -0.15
        assert wvb_w1["mean_diff"] < 0
        assert wvb_w1["headroom_normalized_improvement"] is not None


class TestLoadAndMarkdown:
    def test_load_run_summaries(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            p.write_text(json.dumps({
                "config": {"method": "a", "benchmark": "wvb", "config_id": "abc"},
                "n_items": 10,
                "metrics": {"W1_mean": 0.3},
            }))
            loaded = load_run_summaries([p])
            assert len(loaded) == 1
            assert loaded[0].method == "a"

    def test_main_table_markdown(self):
        md = main_table_markdown([
            {"method": "a", "benchmark": "wvb", "metric": "W1_mean", "n_runs": 3,
             "mean": 0.3, "ci_lo": 0.28, "ci_hi": 0.32},
        ])
        assert "| method | benchmark |" in md
        assert "0.3000" in md

    def test_compare_table_markdown(self):
        md = compare_table_markdown([{
            "benchmark": "wvb", "metric": "W1_mean", "n_pairs": 3,
            "method_mean": 0.2, "baseline_mean": 0.4,
            "mean_diff": -0.2, "lo": -0.25, "hi": -0.15,
            "p_value_two_sided": 0.01,
            "headroom_normalized_improvement": 0.5,
        }])
        assert "0.5000" in md
        assert "wvb" in md
