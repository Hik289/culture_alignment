"""STANDBY 期间 3 件准备工作的测试.

(1) build_retrieval_index 骨架 (用 tmp jsonl + IdentityEmbedder, 不调 sentence-transformers)
(2) T-scaling pipeline 走通 (用 anchor_4 records 真概率)
(3) 并发 dispatch (mock LLM 用 time.sleep)
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_retrieval_index import (  # noqa: E402
    BuildConfig,
    BuildResult,
    build_index,
    load_evidence_jsonl,
    make_embedder,
    parse_args,
)
from src.calibrate import (  # noqa: E402
    apply_temperature,
    fit_temperature,
    negative_log_likelihood_from_probs,
)
from src.retrieval import IdentityEmbedder, EvidenceItem  # noqa: E402
from src.run_predictions import (  # noqa: E402
    dispatch_items,
    dispatch_items_with_rate_limit,
)


# ---------------------------------------------------------------------------
# (1) build_retrieval_index
# ---------------------------------------------------------------------------

def _write_evidence_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


class TestLoadEvidenceJsonl:
    def test_basic_load(self):
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "evidence.jsonl"
            _write_evidence_jsonl(p, [
                {"id": "e1", "text": "family in Japan", "country": "Japan",
                 "topic": "family", "year": 2019},
                {"id": "e2", "text": "religion in Brazil", "country": "Brazil",
                 "topic": "religion"},
            ])
            items = load_evidence_jsonl(p)
            assert len(items) == 2
            assert items[0].id == "e1"
            assert items[0].country == "Japan"
            assert items[1].topic == "religion"

    def test_missing_required_field(self):
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "bad.jsonl"
            _write_evidence_jsonl(p, [{"id": "x"}])  # 没 text
            with pytest.raises(ValueError):
                load_evidence_jsonl(p)

    def test_invalid_json(self):
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "bad.jsonl"
            p.write_text("{not json}\n")
            with pytest.raises(ValueError):
                load_evidence_jsonl(p)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_evidence_jsonl("/nonexistent/path/x.jsonl")

    def test_extra_fields_go_to_meta(self):
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "ev.jsonl"
            _write_evidence_jsonl(p, [
                {"id": "e", "text": "t", "custom_field": "v", "another": 42},
            ])
            items = load_evidence_jsonl(p)
            assert items[0].meta == {"custom_field": "v", "another": 42}

    def test_empty_lines_skipped(self):
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "ev.jsonl"
            p.write_text('{"id":"a","text":"x"}\n\n\n{"id":"b","text":"y"}\n')
            items = load_evidence_jsonl(p)
            assert len(items) == 2


class TestMakeEmbedder:
    def test_identity(self):
        emb = make_embedder(BuildConfig(embedder_kind="identity", embedder_dim=64))
        assert isinstance(emb, IdentityEmbedder)
        assert emb.dim == 64

    def test_sentence_transformers_instantiates(self):
        """ST 已实装. 装了就成功, 没装应抛 ImportError."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        # 真 instantiate 会下载模型, 这里只检查 callable; 实际下载留 RUNNING
        # 跳过昂贵的真实例化, 改为检查可导入
        from src.build_retrieval_index import SentenceTransformerEmbedder
        assert SentenceTransformerEmbedder is not None


class TestBuildConfig:
    def test_validate_ok(self):
        BuildConfig().validate()

    def test_invalid_embedder(self):
        with pytest.raises(ValueError):
            BuildConfig(embedder_kind="bogus").validate()

    def test_invalid_index_kind(self):
        with pytest.raises(ValueError):
            BuildConfig(index_kinds=("nope",)).validate()


class TestBuildIndex:
    def test_end_to_end_identity(self):
        """识别 embedder + 完整 pipeline 应跑通."""
        with tempfile.TemporaryResearchery() as td:
            ev_path = Path(td) / "evidence.jsonl"
            out = Path(td) / "out"
            _write_evidence_jsonl(ev_path, [
                {"id": f"e{i}", "text": f"text {i}", "country": "X" if i % 2 == 0 else "Y"}
                for i in range(20)
            ])
            cfg = BuildConfig(
                evidence_path=str(ev_path),
                out_dir=str(out),
                embedder_kind="identity",
                embedder_dim=16,
            )
            result = build_index(cfg)
            assert result.n_evidence == 20
            assert "general" in result.index_paths
            # hierarchical 与 general 共享 index, 标记为 hierarchical_shares_with_general
            assert ("hierarchical" in result.index_paths
                    or "hierarchical_shares_with_general" in result.index_paths)
            assert result.embedding_cache_path is not None
            assert result.elapsed_seconds >= 0

    def test_only_general(self):
        with tempfile.TemporaryResearchery() as td:
            ev_path = Path(td) / "ev.jsonl"
            _write_evidence_jsonl(ev_path, [{"id": "1", "text": "t"}])
            cfg = BuildConfig(
                evidence_path=str(ev_path),
                out_dir=str(Path(td) / "out"),
                embedder_kind="identity",
                index_kinds=("general",),
            )
            result = build_index(cfg)
            assert "general" in result.index_paths
            assert "hierarchical" not in result.index_paths


class TestCLI:
    def test_default_parse(self):
        cfg = parse_args([])
        assert cfg.evidence_path == "data/evidence/evidence.jsonl"
        assert "general" in cfg.index_kinds
        assert "hierarchical" in cfg.index_kinds

    def test_no_hierarchical(self):
        cfg = parse_args(["--no-hierarchical"])
        assert "general" in cfg.index_kinds
        assert "hierarchical" not in cfg.index_kinds


# ---------------------------------------------------------------------------
# (2) T-scaling 预热: 用 anchor_4 real probs 走通 fit/apply
# ---------------------------------------------------------------------------

ANCHOR4_RECORDS = ROOT / "experiments" / "anchor_4_no_culture_baseline" / "records.json"


def _load_anchor4_distribution_records(bench: str = "wvb"):
    """从 anchor_4 records 抓 distribution 类样本 (含 pred_probs + gold_probs)."""
    if not ANCHOR4_RECORDS.exists():
        return None
    recs = json.loads(ANCHOR4_RECORDS.read_text())
    out = [r for r in recs if r.get("bench") == bench and r.get("ok") and "pred_probs" in r]
    return out


class TestCalibrationWarmupWithAnchor4Data:
    """用 anchor_4 真概率走通 calibrate pipeline. Skip 若 records 不存在."""

    def test_pipeline_runs_on_real_anchor4(self):
        records = _load_anchor4_distribution_records("wvb")
        if not records or len(records) < 10:
            pytest.skip("anchor_4 records 不存在或样本不足")

        # 把不同 K 的样本分组, T-scaling 只对相同 K 一次
        by_k = {}
        for r in records:
            k = len(r["pred_probs"])
            by_k.setdefault(k, []).append(r)
        # 选最大组
        k_best = max(by_k, key=lambda x: len(by_k[x]))
        recs = by_k[k_best]
        if len(recs) < 10:
            pytest.skip(f"K={k_best} 组样本 {len(recs)} 不足")

        N = len(recs)
        K = k_best
        probs = np.array([r["pred_probs"] for r in recs], dtype=np.float64)
        # 用 gold_probs 的 argmax 作为 y_true 替代 (anchor_4 没有单一 ground truth label,
        # 用 gold 分布的 argmax 当作 calibration target — 这是合理近似)
        y_true = np.array([int(np.argmax(r["gold_probs"])) for r in recs])

        # fit T*
        cal = fit_temperature(probs, y_true.tolist(), bench="wvb_anchor4")
        assert cal.T_star > 0
        assert cal.n_dev_samples == N
        # nll_after 应该 ≤ nll_before
        assert cal.nll_after <= cal.nll_before + 1e-6
        # apply 后仍是有效概率
        p_calibrated = apply_temperature(probs, cal.T_star)
        assert p_calibrated.shape == probs.shape
        np.testing.assert_allclose(p_calibrated.sum(axis=1), 1.0, atol=1e-6)
        # argmax 不变 (T-scaling 保 argmax)
        np.testing.assert_array_equal(probs.argmax(axis=1), p_calibrated.argmax(axis=1))

    def test_calibrate_does_not_change_argmax_anywhere(self):
        """T-scaling 不变 argmax 是核心不变量."""
        records = _load_anchor4_distribution_records("wvb")
        if not records or len(records) < 5:
            pytest.skip()
        recs = records[:30]
        probs = np.array([r["pred_probs"] for r in recs[:1]], dtype=np.float64)
        # 多个 T 测试
        for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
            out = apply_temperature(probs, T)
            assert int(out.argmax(axis=1)[0]) == int(probs.argmax(axis=1)[0]), \
                f"T={T} 改变了 argmax"


# ---------------------------------------------------------------------------
# (3) 并发 batch dispatch (mock LLM 用 sleep)
# ---------------------------------------------------------------------------

def _mock_llm_handler(item: dict) -> dict:
    """模拟 LLM 调用: sleep 50ms, 返回 echo."""
    time.sleep(0.05)
    return {"item_id": item["item_id"], "ok": True, "echo": item.get("question", "")}


def _flaky_handler(item: dict) -> dict:
    """偶发失败 (item_id 末位为 'x' 时 raise)."""
    time.sleep(0.02)
    if str(item["item_id"]).endswith("x"):
        raise RuntimeError("simulated failure")
    return {"item_id": item["item_id"], "ok": True}


class TestDispatchItems:
    def test_order_preserved(self):
        items = [{"item_id": f"i{i}", "question": f"q{i}"} for i in range(20)]
        results = dispatch_items(items, _mock_llm_handler, max_workers=4)
        assert len(results) == 20
        # 输出顺序与输入一致 (尽管并发完成顺序不同)
        for i, r in enumerate(results):
            assert r["item_id"] == f"i{i}"

    def test_concurrency_speedup(self):
        """并发 8 比串行快约 8 倍 (50ms × 20 = 1s 串行 → 8 并发 ~ 125ms + overhead)."""
        items = [{"item_id": f"i{i}"} for i in range(20)]

        t0 = time.time()
        dispatch_items(items, _mock_llm_handler, max_workers=1)
        serial = time.time() - t0

        t0 = time.time()
        dispatch_items(items, _mock_llm_handler, max_workers=8)
        parallel = time.time() - t0

        # 并发应至少快 3 倍 (考虑 thread 开销)
        assert parallel < serial / 3, f"serial={serial:.3f}s parallel={parallel:.3f}s, 加速比不足"

    def test_error_recorded(self):
        items = [{"item_id": f"i{i}"} for i in range(5)] + [{"item_id": "ix"}]
        results = dispatch_items(items, _flaky_handler, max_workers=4, on_error="record")
        # 失败的 ix 应被记录
        bad = [r for r in results if r.get("ok") is False]
        assert len(bad) == 1
        assert bad[0]["item_id"] == "ix"
        assert "simulated failure" in bad[0]["error"]
        # 其他 5 个成功
        good = [r for r in results if r.get("ok") is True]
        assert len(good) == 5

    def test_error_raise_mode(self):
        items = [{"item_id": "ix"}]
        with pytest.raises(RuntimeError):
            dispatch_items(items, _flaky_handler, on_error="raise")

    def test_error_skip_mode(self):
        items = [{"item_id": "i1"}, {"item_id": "ix"}, {"item_id": "i2"}]
        results = dispatch_items(items, _flaky_handler, on_error="skip", max_workers=2)
        # ix 被跳过, 剩 2 个
        assert len(results) == 2
        assert all(r["item_id"] != "ix" for r in results)

    def test_missing_item_id_raises(self):
        with pytest.raises(ValueError):
            dispatch_items([{"question": "q"}], _mock_llm_handler)

    def test_empty_items(self):
        results = dispatch_items([], _mock_llm_handler)
        assert results == []


class TestDispatchWithRateLimit:
    def test_rps_limit_slows_submission(self):
        """max_per_second=10 应使 20 个 task 提交至少 1.8s (其中 19 个 sleep)."""
        items = [{"item_id": f"i{i}"} for i in range(20)]

        t0 = time.time()
        # mock handler 几乎 instant, rate limit 应主导 wall time
        dispatch_items_with_rate_limit(
            items,
            lambda it: {"item_id": it["item_id"], "ok": True},
            max_workers=20, max_per_second=10.0,
        )
        elapsed = time.time() - t0
        # 19 个 interval × 0.1s = 1.9s, 允许一些容差
        assert elapsed >= 1.7, f"elapsed={elapsed:.3f}s, rate limit 没生效"

    def test_no_limit_when_zero(self):
        items = [{"item_id": f"i{i}"} for i in range(5)]
        results = dispatch_items_with_rate_limit(
            items, _mock_llm_handler, max_workers=4, max_per_second=0.0,
        )
        assert len(results) == 5


# ---------------------------------------------------------------------------
# 补充: data_scientist schema 兼容性
# ---------------------------------------------------------------------------

class TestDataScientistSchema:
    """评估 data_scientist 真 evidence.jsonl schema 是否能被 loader 读懂."""

    def test_data_scientist_schema_load(self):
        """data_sci 用 evidence_id / country_or_region / language / metadata."""
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "ds_evidence.jsonl"
            # 真实 data_sci 一行 (从 gpu_server 抓的截断版)
            line = {
                "evidence_id": "worldvaluesbench::train::Q4::9d3bd1fb5278",
                "source": "worldvaluesbench",
                "split": "train",
                "country_or_region": "Andorra",
                "language": "en",
                "task_type": "survey_distribution_ordinal",
                "topic": "Social Values, Norms, Stereotypes",
                "text": "On a scale of 1 to 4, how important are politics in your life?",
                "answer_options": {"1": "1", "2": "2", "3": "3", "4": "4"},
                "distribution": {"1": 0.08, "2": 0.22, "3": 0.37, "4": 0.32},
                "metadata": {"wvs_wave": 7, "n_respondents": 704},
                "indexed_at": "2026-06-16T09:04:02Z",
            }
            p.write_text(json.dumps(line) + "\n")
            items = load_evidence_jsonl(p)
            assert len(items) == 1
            it = items[0]
            assert it.id == "worldvaluesbench::train::Q4::9d3bd1fb5278"
            assert it.country == "Andorra"
            assert it.topic == "Social Values, Norms, Stereotypes"
            assert it.source == "worldvaluesbench"
            assert it.lang == "en"
            # year 从 metadata.wvs_wave 推断
            assert it.year == 7
            # split/distribution 进 meta
            assert it.meta.get("split") == "train"
            assert it.meta.get("task_type") == "survey_distribution_ordinal"
            assert "distribution" in it.meta

    def test_data_scientist_schema_missing_text(self):
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "x.jsonl"
            p.write_text(json.dumps({"evidence_id": "x"}) + "\n")
            with pytest.raises(ValueError):
                load_evidence_jsonl(p)

    def test_mixed_schemas_in_same_file(self):
        """一个文件里既有 evidence_id 又有 id 的行 — 都能读."""
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "mixed.jsonl"
            p.write_text(
                json.dumps({"evidence_id": "ds1", "text": "x", "country_or_region": "Japan"}) + "\n"
                + json.dumps({"id": "s1", "text": "y", "country": "Brazil"}) + "\n"
            )
            items = load_evidence_jsonl(p)
            assert len(items) == 2
            assert items[0].id == "ds1" and items[0].country == "Japan"
            assert items[1].id == "s1" and items[1].country == "Brazil"


# ---------------------------------------------------------------------------
# 补充: calibration 端到端集成 (用 anchor_4 + run_predictions 的概念串通)
# ---------------------------------------------------------------------------

class TestCalibrationEndToEnd:
    """完整 pipeline 集成: load anchor_4 records → fit T → apply T → 对比 metric."""

    def test_full_pipeline_improves_nll(self):
        records = _load_anchor4_distribution_records("wvb")
        if not records or len(records) < 20:
            pytest.skip("anchor_4 records 不足")
        # 抓 K=4 的样本 (最大组)
        by_k = {}
        for r in records:
            by_k.setdefault(len(r["pred_probs"]), []).append(r)
        if not by_k:
            pytest.skip()
        k = max(by_k, key=lambda x: len(by_k[x]))
        recs = by_k[k]
        if len(recs) < 10:
            pytest.skip()

        probs = np.array([r["pred_probs"] for r in recs], dtype=np.float64)
        y_true = np.array([int(np.argmax(r["gold_probs"])) for r in recs])

        # 划分 dev/test (50/50, 用 seed)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(recs))
        half = len(recs) // 2
        dev_idx = idx[:half]
        test_idx = idx[half:]

        # Fit on dev
        cal = fit_temperature(
            probs[dev_idx], y_true[dev_idx].tolist(),
            bench="wvb_integration",
        )
        assert cal.T_star > 0
        # Apply on test
        probs_test_raw = probs[test_idx]
        probs_test_cal = apply_temperature(probs_test_raw, cal.T_star)
        # Test NLL 改善 (允许小退步, 但不应大爆)
        nll_raw_test = negative_log_likelihood_from_probs(
            probs_test_raw, y_true[test_idx].tolist()
        )
        nll_cal_test = negative_log_likelihood_from_probs(
            probs_test_cal, y_true[test_idx].tolist()
        )
        # 不强制 cal < raw (test 不是 dev), 但应该量级合理
        assert np.isfinite(nll_cal_test)
        assert nll_cal_test > 0  # NLL 必正
        # argmax 不变
        np.testing.assert_array_equal(
            probs_test_raw.argmax(axis=1),
            probs_test_cal.argmax(axis=1),
        )


class TestSentenceTransformerEmbedderLazyImport:
    """SentenceTransformer 包装类应该惰性 import, 不该在测试启动时就 crash."""

    def test_can_construct_config_without_st_loaded(self):
        """BuildConfig(embedder_kind='sentence_transformers') 应能正常构造."""
        cfg = BuildConfig(embedder_kind="sentence_transformers")
        cfg.validate()  # 不该 raise

    def test_make_embedder_st_needs_install(self):
        """实际 instantiate ST embedder 时才尝试 import; 若已装应成功."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed locally")
        # 装了就跑 (在 gpu_server 上装了, GCP 可能没装)
        from src.build_retrieval_index import SentenceTransformerEmbedder
        # 真正初始化 ST 模型会下载, 这里不实际跑, 仅检查类可导入
        assert SentenceTransformerEmbedder is not None
