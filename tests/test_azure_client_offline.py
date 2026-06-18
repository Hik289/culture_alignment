"""离线测试 azure_client 的不依赖网络部分:
- _try_parse_json: 直接 / fence / 平衡子串 fallback
- _classify_exc: 异常分类
- _is_retriable: 是否重试

不调用 Azure, 不依赖凭据.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.azure_client import (  # noqa: E402
    ERR_AUTH,
    ERR_BAD_REQUEST,
    ERR_CONNECTION,
    ERR_JSON,
    ERR_OTHER,
    ERR_RATE_LIMIT,
    ERR_SERVER,
    ERR_TIMEOUT,
    _classify_exc,
    _is_retriable,
    _try_parse_json,
)


class TestJSONFallback:
    def test_plain_json(self):
        parsed, used_fb, err = _try_parse_json('{"a": 1}')
        assert parsed == {"a": 1}
        assert used_fb is False
        assert err is None

    def test_fence_json(self):
        content = 'Sure! Here is the answer:\n```json\n{"answer": "Tokyo"}\n```\n'
        parsed, used_fb, err = _try_parse_json(content)
        assert parsed == {"answer": "Tokyo"}
        assert used_fb is True
        assert err is None

    def test_fence_bare(self):
        content = '```\n{"x": 2}\n```'
        parsed, used_fb, _ = _try_parse_json(content)
        assert parsed == {"x": 2}
        assert used_fb is True

    def test_prefix_then_json(self):
        """LLM 在 JSON 前加自然语言, 应抓取首个平衡 {...}."""
        content = 'The result is: {"probs": [0.1, 0.9]}. Hope this helps!'
        parsed, used_fb, _ = _try_parse_json(content)
        assert parsed == {"probs": [0.1, 0.9]}
        assert used_fb is True

    def test_nested_object(self):
        content = 'Output: {"label": "ok", "probs": {"a": 0.7, "b": 0.3}}'
        parsed, used_fb, _ = _try_parse_json(content)
        assert parsed["label"] == "ok"
        assert parsed["probs"]["a"] == 0.7

    def test_brace_inside_string(self):
        """字符串里的 } 不应误终结对象."""
        content = '{"text": "He said: {hello}", "n": 1}'
        parsed, _, _ = _try_parse_json(content)
        assert parsed == {"text": "He said: {hello}", "n": 1}

    def test_escaped_quote(self):
        content = '{"q": "she said \\"hi\\""}'
        parsed, _, _ = _try_parse_json(content)
        assert parsed == {"q": 'she said "hi"'}

    def test_no_json(self):
        parsed, _, err = _try_parse_json("This has no JSON at all.")
        assert parsed is None
        assert err is not None

    def test_malformed(self):
        parsed, _, err = _try_parse_json('{"a": ')
        assert parsed is None
        assert err is not None

    def test_empty(self):
        parsed, _, err = _try_parse_json("")
        assert parsed is None
        assert err is not None

    def test_none(self):
        parsed, _, err = _try_parse_json(None)  # type: ignore[arg-type]
        assert parsed is None
        assert err is not None


class TestClassifyExc:
    def test_generic_exception_other(self):
        """非 openai 异常归为 other."""
        et, ra = _classify_exc(ValueError("nope"))
        assert et == ERR_OTHER
        assert ra is None

    def test_runtime_other(self):
        et, _ = _classify_exc(RuntimeError("boom"))
        assert et == ERR_OTHER


class TestRetriable:
    def test_retriable_set(self):
        for t in [ERR_AUTH, ERR_RATE_LIMIT, ERR_SERVER, ERR_TIMEOUT, ERR_CONNECTION]:
            assert _is_retriable(t)

    def test_non_retriable(self):
        for t in [ERR_BAD_REQUEST, ERR_JSON, ERR_OTHER]:
            assert not _is_retriable(t)
