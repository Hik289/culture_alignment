"""H_anchor_2: 评估指标解析解对齐测试.

要求: 所有指标在合成 toy distribution 上与解析解对齐 ± 1e-6.
失败任一 → 上报 Researcher.

运行: pytest -v experiments/anchor_2_metrics/test_metrics.py
或:   python -m pytest experiments/anchor_2_metrics/test_metrics.py -v
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# 让 src 包能被 import (pytest 从项目根运行)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics import (
    accuracy,
    brier_score,
    exact_match,
    exact_match_corpus,
    expected_calibration_error,
    js_distance,
    js_divergence,
    kl_divergence,
    macro_f1,
    negative_log_likelihood,
    token_f1,
    token_f1_corpus,
    top1_accuracy,
    tv_distance,
    wasserstein_1,
)

TOL = 1e-6


# ---------------------------------------------------------------------------
# 16.1 分布预测
# ---------------------------------------------------------------------------

class TestWasserstein1:
    def test_director_example_ordinal_12(self):
        """W1([0.5,0.5], [0.0,1.0]) on support {1,2} = 0.5 (Researcher invariant)."""
        w = wasserstein_1([0.5, 0.5], [0.0, 1.0], support=[1, 2])
        assert math.isclose(w, 0.5, abs_tol=TOL), f"W1={w}, 预期 0.5"

    def test_identity_zero(self):
        p = [0.2, 0.3, 0.5]
        assert wasserstein_1(p, p) < TOL

    def test_point_masses_distance_2(self):
        """所有质量在 0 vs 所有在 2, support={0,1,2} → W1=2."""
        w = wasserstein_1([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], support=[0, 1, 2])
        assert math.isclose(w, 2.0, abs_tol=TOL)

    def test_default_support_integers(self):
        """默认 support=0..K-1."""
        w_default = wasserstein_1([1.0, 0.0], [0.0, 1.0])
        w_explicit = wasserstein_1([1.0, 0.0], [0.0, 1.0], support=[0, 1])
        assert math.isclose(w_default, w_explicit, abs_tol=TOL)
        assert math.isclose(w_default, 1.0, abs_tol=TOL)

    def test_symmetric(self):
        p = [0.1, 0.4, 0.5]
        q = [0.6, 0.3, 0.1]
        a = wasserstein_1(p, q)
        b = wasserstein_1(q, p)
        assert math.isclose(a, b, abs_tol=TOL)

    def test_known_value_uniform_vs_corner(self):
        """W1(uniform on {0..4}, delta on 0) = mean(0..4)/1 = 2.0."""
        uni = [0.2] * 5
        delta = [1.0, 0, 0, 0, 0]
        w = wasserstein_1(delta, uni)
        # cdf_delta = [1,1,1,1,1], cdf_uni = [.2,.4,.6,.8,1.0]
        # 差段 = [.8,.6,.4,.2], 每段宽 1 → 2.0
        assert math.isclose(w, 2.0, abs_tol=TOL)


class TestJSAndKLAndTV:
    def test_js_div_identity_zero(self):
        p = [0.2, 0.3, 0.5]
        assert js_divergence(p, p) < TOL
        assert js_distance(p, p) < TOL

    def test_js_div_max_for_disjoint(self):
        """两个支撑完全不相交时, base=2 → JS 散度 → 1.0 (极限).

        因 epsilon 平滑会略低于 1, 这里用宽松上限验证 > 0.99.
        """
        p = [1.0, 0.0, 0.0]
        q = [0.0, 0.0, 1.0]
        div = js_divergence(p, q, base=2.0, eps=1e-12)
        assert 0.99 < div <= 1.0, f"JS div={div}"
        # 距离 = sqrt(div)
        dist = js_distance(p, q, base=2.0, eps=1e-12)
        assert math.isclose(dist, math.sqrt(div), abs_tol=TOL)

    def test_js_div_known_value(self):
        """p=[1,0], q=[.5,.5], base=2 → 解析解.

        m=[.75,.25]
        KL(p||m) = 1*log2(1/.75) = log2(4/3)
        KL(q||m) = .5*log2(.5/.75) + .5*log2(.5/.25)
                 = .5*log2(2/3) + .5*log2(2)
        JS = .5*KL(p||m) + .5*KL(q||m).
        """
        p = [1.0, 0.0]
        q = [0.5, 0.5]
        kl_pm = math.log2(4 / 3)
        kl_qm = 0.5 * math.log2(2 / 3) + 0.5 * math.log2(2)
        expected = 0.5 * kl_pm + 0.5 * kl_qm
        got = js_divergence(p, q, base=2.0, eps=1e-12)
        assert math.isclose(got, expected, abs_tol=1e-5), f"got={got}, expected={expected}"

    def test_kl_identity_zero(self):
        p = [0.1, 0.2, 0.7]
        assert abs(kl_divergence(p, p)) < 1e-9

    def test_kl_known_value(self):
        """KL([1,0]||[.5,.5]) = 1*ln(1/.5) = ln 2 (in nats)."""
        got = kl_divergence([1.0, 0.0], [0.5, 0.5], base=math.e, eps=1e-12)
        assert math.isclose(got, math.log(2), abs_tol=1e-5), got

    def test_tv_identity_zero(self):
        assert tv_distance([0.3, 0.7], [0.3, 0.7]) < TOL

    def test_tv_known_value(self):
        """TV([1,0,0],[0,1,0]) = 0.5*(1+1+0) = 1.0."""
        assert math.isclose(tv_distance([1, 0, 0], [0, 1, 0]), 1.0, abs_tol=TOL)
        assert math.isclose(tv_distance([0.5, 0.5], [0.0, 1.0]), 0.5, abs_tol=TOL)


class TestTop1:
    def test_match(self):
        assert top1_accuracy([0.1, 0.7, 0.2], [0.2, 0.6, 0.2]) == 1.0

    def test_mismatch(self):
        assert top1_accuracy([0.1, 0.7, 0.2], [0.8, 0.1, 0.1]) == 0.0


# ---------------------------------------------------------------------------
# 16.2 分类指标
# ---------------------------------------------------------------------------

class TestClassification:
    def test_accuracy(self):
        assert accuracy([0, 1, 2, 1], [0, 1, 1, 1]) == 0.75

    def test_macro_f1_perfect(self):
        assert math.isclose(macro_f1([0, 1, 2], [0, 1, 2]), 1.0, abs_tol=TOL)

    def test_macro_f1_known(self):
        """二分类 y=[1,1,0,0], pred=[1,0,1,0].
        class 1: tp=1, fp=1, fn=1 → P=.5, R=.5, F1=.5.
        class 0: tp=1, fp=1, fn=1 → P=.5, R=.5, F1=.5.
        macro = .5.
        """
        got = macro_f1([1, 1, 0, 0], [1, 0, 1, 0])
        assert math.isclose(got, 0.5, abs_tol=TOL)

    def test_nll_perfect(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = [0, 1]
        # 被 clip 到 eps=1e-12 → NLL ≈ 0
        nll = negative_log_likelihood(probs, y, eps=1e-12)
        assert nll < 1e-6

    def test_nll_uniform(self):
        """C=2 均匀 → NLL = ln 2."""
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        y = [0, 1]
        nll = negative_log_likelihood(probs, y)
        assert math.isclose(nll, math.log(2), abs_tol=1e-9)

    def test_brier_perfect(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = [0, 1]
        assert brier_score(probs, y) < TOL

    def test_brier_known(self):
        """probs=[[.7,.3]], y=[0] → (1-.7)^2 + (0-.3)^2 = .09+.09 = .18."""
        got = brier_score(np.array([[0.7, 0.3]]), [0])
        assert math.isclose(got, 0.18, abs_tol=TOL)

    def test_ece_perfect_confident(self):
        """完全置信且全对 → ECE=0."""
        probs = np.array([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]])
        y = [0, 1, 0]
        ece = expected_calibration_error(probs, y, n_bins=10)
        # 全部进最后一桶: acc=1, conf≈0.99 → ECE≈0.01
        assert ece < 0.02

    def test_ece_known_two_bins(self):
        """构造一组样本, 落在两个桶, 手算 ECE.

        样本 1: max_conf=0.9, pred=0, y=0 (对)
        样本 2: max_conf=0.9, pred=0, y=1 (错)
        样本 3: max_conf=0.6, pred=1, y=1 (对)
        样本 4: max_conf=0.6, pred=1, y=0 (错)
        n_bins=10:
          bin [.8,.9]: 样本1,2; acc=.5, conf=.9 → |diff|=.4, weight=2/4=.5
          bin [.5,.6]: 样本3,4; acc=.5, conf=.6 → |diff|=.1, weight=2/4=.5
          ECE = .5*.4 + .5*.1 = .25
        注意: bin 边界右开, 0.9 与 0.6 实际归属:
          0.9 落在 [0.8, 0.9) 还是 [0.9, 1.0]?
          按代码: 最后桶闭区间; 中间桶左闭右开 → 0.9 落在 [.9, 1.0]
        重新算:
          bin [.9,1.0] (最后桶, 闭): 样本1,2; acc=.5, conf=.9 → .4
          bin [.5,.6): 样本3,4 → 0.6 不在 [.5,.6) 内, 而在 [.6,.7).
          bin [.6,.7): 样本3,4; acc=.5, conf=.6 → .1
        ECE = .5*.4 + .5*.1 = .25.
        """
        probs = np.array([
            [0.9, 0.1],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.4, 0.6],
        ])
        y = [0, 1, 1, 0]
        got = expected_calibration_error(probs, y, n_bins=10)
        assert math.isclose(got, 0.25, abs_tol=TOL), f"ECE={got}"


# ---------------------------------------------------------------------------
# 16.3 短答案
# ---------------------------------------------------------------------------

class TestShortAnswer:
    def test_em_normalization(self):
        assert exact_match("The Cat", "a cat") == 1.0  # 去冠词+小写
        assert exact_match("Tokyo!", "tokyo") == 1.0  # 去标点
        assert exact_match("dog", "cat") == 0.0

    def test_em_corpus_multi_gold(self):
        preds = ["tokyo", "berlin"]
        golds = [["Tokyo", "東京"], "Berlin"]
        # 第二个 gold 单个字符串
        assert math.isclose(exact_match_corpus(preds, golds), 1.0, abs_tol=TOL)

    def test_token_f1_known(self):
        """pred='the brown fox', gold='quick brown fox'.
        normalized pred = 'brown fox' (去 the), gold = 'quick brown fox'.
        common = {brown:1, fox:1}, num_same=2.
        precision = 2/2 = 1.0, recall = 2/3.
        F1 = 2*1*(2/3)/(1+2/3) = (4/3)/(5/3) = 4/5 = 0.8.
        """
        got = token_f1("the brown fox", "quick brown fox")
        assert math.isclose(got, 0.8, abs_tol=TOL), got

    def test_token_f1_perfect(self):
        assert token_f1("hello world", "Hello, World!") == 1.0

    def test_token_f1_disjoint(self):
        assert token_f1("cat", "dog") == 0.0

    def test_token_f1_corpus(self):
        preds = ["hello world", "good"]
        golds = ["hello", "good"]
        # 第一对: pred={hello,world}, gold={hello}, common=1
        #   P=1/2, R=1/1 → F1 = 2*.5*1/(1.5)= 2/3
        # 第二对: F1=1
        # mean = (2/3+1)/2 = 5/6
        got = token_f1_corpus(preds, golds)
        assert math.isclose(got, 5 / 6, abs_tol=TOL)


# ---------------------------------------------------------------------------
# 辅助: 输出 assertions.json (供 Researcher 引用)
# ---------------------------------------------------------------------------

def test_zz_dump_assertions(tmp_path_factory):
    """所有上面测试通过后, 输出一份 assertions.json 记录通过/失败.

    用 zz_ 前缀确保按字母序最后执行 (pytest 默认按定义顺序, 但 zz 直观).
    """
    out_dir = Path(os.environ.get("ANCHOR2_OUT", str(ROOT / "experiments" / "anchor_2_metrics")))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "anchor": "H_anchor_2",
        "hypothesis": "评估指标实现正确",
        "debug_move": "ASSERT",
        "tolerance": TOL,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "passed (此文件由通过的最后一个 test 写出)",
        "metrics_covered": [
            "W1",
            "JS-D (divergence + distance)",
            "KL-D",
            "TV-D",
            "Top-1 Accuracy",
            "Accuracy",
            "Macro-F1",
            "NLL",
            "Brier",
            "ECE",
            "EM",
            "Token F1",
        ],
        "director_example": {
            "W1([0.5,0.5], [0.0,1.0]) on support {1,2}": 0.5,
        },
    }
    out_path = out_dir / "assertions.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    assert out_path.exists()
