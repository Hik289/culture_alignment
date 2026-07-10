"""General model API JSON client for CultureLens-RC experiments.

设计:
- 默认读取 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL, 不硬编码 API key.
- 用 chat-completions-compatible SDK 作为通用传输层.
- 高层接口 chat_json(): temperature=0 + JSON 模式 + 自动重试 + 错误分类.
- 记录 tokens / latency / 调用元数据 / error_type.

错误处理 (Researcher 要求):
- 401 / token expiry → 立刻 refresh token 重试 (不计入 max_retries 的等待退避)
- 429 / rate limit → 解析 Retry-After (若有), 否则指数退避
- 5xx / network → 指数退避重试
- JSON 解析失败 → 尝试 markdown fence 剥离 + 抓首个 {...} 子串作为 fallback, 仍失败再重试

约束:
1. 模型、endpoint、key 都从环境变量或函数参数配置.
2. 仅供本仓库实验使用.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

ENV_API_KEY = "LLM_API_KEY"
EXTRA_ENV_API_KEYS = ("MODEL_API_KEY",)
MODEL_API_BASE_URL = (
    os.environ.get("LLM_BASE_URL")
    or os.environ.get("MODEL_API_BASE_URL")
    or "${LLM_BASE_URL}"
)
MODEL_NAME = (
    os.environ.get("LLM_MODEL")
    or os.environ.get("MODEL_NAME")
    or "your-model-name"
)
NO_TEMPERATURE_PREFIXES = tuple(
    p.strip()
    for p in os.environ.get("LLM_NO_TEMPERATURE_PREFIXES", "").split(",")
    if p.strip()
)

# error_type 标签 (写进 CallResult 用于聚合分析)
ERR_AUTH = "auth"               # 401, 凭据失败, token 过期
ERR_RATE_LIMIT = "rate_limit"   # 429
ERR_SERVER = "server"           # 5xx
ERR_TIMEOUT = "timeout"         # 超时
ERR_CONNECTION = "connection"   # 网络
ERR_BAD_REQUEST = "bad_request" # 400 (schema 错误等, 不重试)
ERR_JSON = "json_parse"         # 内容非 JSON
ERR_EMPTY = "empty_content"     # 返回空内容 (可能 reasoning model finish_reason=length)
ERR_OTHER = "other"


@dataclass
class CallResult:
    """单次 chat 调用结果."""
    ok: bool
    content: Optional[str]
    parsed: Optional[Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    n_attempts: int = 1
    error: Optional[str] = None
    error_type: Optional[str] = None
    finish_reason: Optional[str] = None
    used_fallback_json: bool = False
    attempt_log: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token / Client (惰性导入)
# ---------------------------------------------------------------------------

def _get_token_provider() -> Callable[[], str]:
    """认证 provider 工厂.

    优先级:
    1. 通用环境变量 LLM_API_KEY.
    2. 通用备用环境变量 MODEL_API_KEY.
    """
    api_key = os.environ.get(ENV_API_KEY)
    if api_key:
        logger.info("auth: using LLM_API_KEY env var")
        return lambda: api_key  # noqa: E731

    for env_name in EXTRA_ENV_API_KEYS:
        api_key = os.environ.get(env_name)
        if api_key:
            logger.info("auth: using MODEL_API_KEY env var")
            return lambda: api_key  # noqa: E731

    raise RuntimeError("Set LLM_API_KEY before running model API experiments.")


def _build_client(token_provider: Callable[[], str]):
    from openai import OpenAI as ChatCompletionsClient
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("MODEL_API_BASE_URL")
        or MODEL_API_BASE_URL
    )
    return ChatCompletionsClient(
        base_url=base_url,
        api_key=token_provider(),  # 当前 token / api_key, 后续每次调用前刷新
    )


# ---------------------------------------------------------------------------
# JSON fallback parser
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


def _try_parse_json(content: str) -> tuple[Optional[Any], bool, Optional[str]]:
    """尝试解析 JSON. 返回 (parsed, used_fallback, error_msg).

    Fallback 顺序:
      1. 直接 json.loads
      2. 剥离 markdown ```json ... ``` fence 后再 loads
      3. 抓首个平衡 {...} 子串 (允许 LLM 前后加自然语言), 再 loads
    """
    if not isinstance(content, str) or not content.strip():
        return None, False, "empty content"

    # 1
    try:
        return json.loads(content), False, None
    except json.JSONDecodeError as e:
        err = str(e)

    # 2
    m = _FENCE_RE.search(content)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate), True, None
        except json.JSONDecodeError:
            pass

    # 3 抓首个 {...} 平衡子串
    start = content.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(content)):
            c = content[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[start:i + 1]
                    try:
                        return json.loads(candidate), True, None
                    except json.JSONDecodeError:
                        break
    return None, False, err


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------

def _classify_exc(e: Exception) -> tuple[str, Optional[float]]:
    """根据 chat-completions SDK 抛出的异常分类. 返回 (error_type, retry_after_seconds_hint)."""
    # 惰性导入避免没装 model client SDK 时炸
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            InternalServerError,
            PermissionDeniedError,
            RateLimitError,
        )
    except Exception:
        APIConnectionError = APITimeoutError = AuthenticationError = ()
        BadRequestError = InternalServerError = PermissionDeniedError = RateLimitError = ()

    if isinstance(e, AuthenticationError) or isinstance(e, PermissionDeniedError):
        return ERR_AUTH, None
    if isinstance(e, RateLimitError):
        # 尝试从 response header 拿 Retry-After
        retry_after = None
        try:
            resp = getattr(e, "response", None)
            if resp is not None:
                ra = resp.headers.get("retry-after")
                if ra:
                    retry_after = float(ra)
        except Exception:
            pass
        return ERR_RATE_LIMIT, retry_after
    if isinstance(e, InternalServerError):
        return ERR_SERVER, None
    if isinstance(e, APITimeoutError):
        return ERR_TIMEOUT, None
    if isinstance(e, APIConnectionError):
        return ERR_CONNECTION, None
    if isinstance(e, BadRequestError):
        return ERR_BAD_REQUEST, None
    return ERR_OTHER, None


def _is_retriable(err_type: str) -> bool:
    return err_type in {ERR_AUTH, ERR_RATE_LIMIT, ERR_SERVER, ERR_TIMEOUT, ERR_CONNECTION}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def chat_json(
    messages: list[dict],
    *,
    model: str = MODEL_NAME,
    max_retries: int = 2,
    temperature: float = 0.0,
    response_format: Optional[dict] = None,
    extra_kwargs: Optional[dict] = None,
    base_backoff: float = 1.0,
    max_backoff: float = 30.0,
    timeout: Optional[float] = 60.0,
) -> CallResult:
    """同步调用配置的模型 API, 强制 JSON 输出.

    Args:
        messages: chat messages.
        model: 模型名.
        max_retries: 额外重试次数 (总尝试 = max_retries + 1).
        temperature: 默认 0; 对部分 reasoning model 不传.
        response_format: 默认 {"type": "json_object"}.
        extra_kwargs: 额外参数 (如 max_completion_tokens / seed).
        base_backoff / max_backoff: 指数退避基/上限 (秒).
        timeout: 单次调用超时.

    Returns:
        CallResult, 含 error_type / used_fallback_json / attempt_log.
    """
    if response_format is None:
        response_format = {"type": "json_object"}
    if extra_kwargs is None:
        extra_kwargs = {}

    token_provider = _get_token_provider()
    client = _build_client(token_provider)

    attempt_log: list[dict] = []
    last_err: Optional[str] = None
    last_err_type: Optional[str] = None
    total_pt = total_ct = total_tt = 0

    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            # 每次刷新 token, 避免长任务过期
            client.api_key = token_provider()

            kwargs = {
                "model": model,
                "messages": messages,
                "response_format": response_format,
                **extra_kwargs,
            }
            # 部分 reasoning model 不接受 temperature 不为默认值; 需要时用
            # LLM_NO_TEMPERATURE_PREFIXES 配置这些模型名前缀.
            if temperature is not None and not model.startswith(NO_TEMPERATURE_PREFIXES):
                kwargs["temperature"] = temperature
            if timeout is not None:
                kwargs["timeout"] = timeout

            resp = client.chat.completions.create(**kwargs)
            dt = time.time() - t0

            content = resp.choices[0].message.content if resp.choices else None
            finish = resp.choices[0].finish_reason if resp.choices else None
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) if usage else 0
            ct = getattr(usage, "completion_tokens", 0) if usage else 0
            tt = getattr(usage, "total_tokens", 0) if usage else 0
            total_pt += pt; total_ct += ct; total_tt += tt

            if not content:
                last_err = f"empty content (finish_reason={finish})"
                last_err_type = ERR_EMPTY
                attempt_log.append({
                    "attempt": attempt + 1, "ok": False, "latency": round(dt, 3),
                    "error_type": ERR_EMPTY, "error": last_err, "finish_reason": finish,
                })
                if attempt < max_retries:
                    time.sleep(min(base_backoff * (2 ** attempt) + random.uniform(0, 0.5), max_backoff))
                    continue
                return CallResult(
                    ok=False, content=content, parsed=None,
                    prompt_tokens=total_pt, completion_tokens=total_ct, total_tokens=total_tt,
                    latency_seconds=dt, n_attempts=attempt + 1,
                    error=last_err, error_type=last_err_type, finish_reason=finish,
                    attempt_log=attempt_log,
                )

            parsed, used_fallback, parse_err = _try_parse_json(content)
            if parsed is None:
                last_err = f"JSON 解析失败: {parse_err}"
                last_err_type = ERR_JSON
                attempt_log.append({
                    "attempt": attempt + 1, "ok": False, "latency": round(dt, 3),
                    "error_type": ERR_JSON, "error": last_err,
                    "finish_reason": finish, "content_preview": content[:200],
                })
                if attempt < max_retries:
                    # 不退避: JSON 问题大概率是 stochastic, 直接再来
                    continue
                return CallResult(
                    ok=False, content=content, parsed=None,
                    prompt_tokens=total_pt, completion_tokens=total_ct, total_tokens=total_tt,
                    latency_seconds=dt, n_attempts=attempt + 1,
                    error=last_err, error_type=last_err_type, finish_reason=finish,
                    attempt_log=attempt_log,
                )

            attempt_log.append({
                "attempt": attempt + 1, "ok": True, "latency": round(dt, 3),
                "used_fallback_json": used_fallback, "finish_reason": finish,
            })
            return CallResult(
                ok=True, content=content, parsed=parsed,
                prompt_tokens=total_pt, completion_tokens=total_ct, total_tokens=total_tt,
                latency_seconds=dt, n_attempts=attempt + 1,
                error=None, error_type=None, finish_reason=finish,
                used_fallback_json=used_fallback,
                attempt_log=attempt_log,
            )

        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            err_type, retry_after = _classify_exc(e)
            msg = f"{type(e).__name__}: {e}"
            last_err = msg
            last_err_type = err_type
            attempt_log.append({
                "attempt": attempt + 1, "ok": False, "latency": round(dt, 3),
                "error_type": err_type, "error": msg, "retry_after_hint": retry_after,
            })
            logger.warning("attempt %d failed (%s): %s", attempt + 1, err_type, msg)

            if not _is_retriable(err_type):
                # 比如 BadRequest / other: 立刻返回, 不重试
                return CallResult(
                    ok=False, content=None, parsed=None,
                    prompt_tokens=total_pt, completion_tokens=total_ct, total_tokens=total_tt,
                    latency_seconds=dt, n_attempts=attempt + 1,
                    error=msg, error_type=err_type,
                    attempt_log=attempt_log,
                )

            if attempt < max_retries:
                if err_type == ERR_AUTH:
                    # 立刻刷 token, 几乎不等
                    sleep_s = min(0.5, max_backoff)
                elif retry_after is not None:
                    sleep_s = min(float(retry_after), max_backoff)
                else:
                    sleep_s = min(base_backoff * (2 ** attempt) + random.uniform(0, 0.5), max_backoff)
                time.sleep(sleep_s)
                continue

            return CallResult(
                ok=False, content=None, parsed=None,
                prompt_tokens=total_pt, completion_tokens=total_ct, total_tokens=total_tt,
                latency_seconds=dt, n_attempts=attempt + 1,
                error=msg, error_type=err_type,
                attempt_log=attempt_log,
            )

    # 不应到达
    return CallResult(
        ok=False, content=None, parsed=None,
        prompt_tokens=total_pt, completion_tokens=total_ct, total_tokens=total_tt,
        n_attempts=max_retries + 1,
        error=last_err or "未知", error_type=last_err_type,
        attempt_log=attempt_log,
    )


__all__ = [
    "MODEL_API_BASE_URL",
    "MODEL_NAME",
    "CallResult",
    "chat_json",
    "ERR_AUTH",
    "ERR_RATE_LIMIT",
    "ERR_SERVER",
    "ERR_TIMEOUT",
    "ERR_CONNECTION",
    "ERR_BAD_REQUEST",
    "ERR_JSON",
    "ERR_EMPTY",
    "ERR_OTHER",
]
