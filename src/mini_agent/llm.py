"""OpenAI-compatible chat completions over the standard library."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from mini_agent import __version__
from mini_agent.config import LLMConfig
from mini_agent.types import LLMResponse, ToolCall

# Cloudflare error 1010 rejects urllib's default Python-urllib User-Agent.
USER_AGENT = f"mini-agent/{__version__}"
RETRY_CUSHION_SECONDS = 0.25
MAX_RETRY_SLEEP_SECONDS = 60.0

_TRY_AGAIN = re.compile(
    r"try again in\s+([0-9]*\.?[0-9]+)\s*(milliseconds|millisecond|ms|seconds|second|secs|sec|s|minutes|minute|mins|min|m)\b",
    re.IGNORECASE,
)
_DURATION = re.compile(r"([0-9]*\.?[0-9]+)\s*(ms|s|m|h)(?![A-Za-z])", re.IGNORECASE)
_TOOL_NAME = re.compile(r"attempted to call tool\s+['\"]?([A-Za-z0-9_\-]+)['\"]?", re.IGNORECASE)
_UNIT_SECONDS = {
    "ms": 0.001,
    "milliseconds": 0.001,
    "millisecond": 0.001,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
}


class LLMError(RuntimeError):
    """The chat-completions call failed or returned something we cannot use."""


class RateLimitError(LLMError):
    """The provider asked us to wait. ``retry_after`` is seconds, when known.

    ``ignored_wait`` is a longer reset from a different bucket (for example
    the request window) that was not used to choose ``retry_after``.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None,
        ignored_wait: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.ignored_wait = ignored_wait


class ToolValidationError(LLMError):
    """The provider rejected a tool name that was not in the request."""

    def __init__(self, message: str, *, tool_name: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Return the next model turn. ``tools`` uses the OpenAI function schema."""


Transport = Any
Sleeper = Any
Clock = Any


class OpenAICompatibleClient:
    """POST ``{base_url}/chat/completions`` and parse tool calls.

    Groq, Gemini's OpenAI-compatible endpoint, and OpenAI itself all speak
    this shape. The transport, clock, and sleep function are injectable so
    tests never open a socket or wait on a rate limit.

    A 429 is retried. The wait is Retry-After or the "try again in" line when
    the body has one. A token-limit 429 does not use the request-bucket
    reset, which on Groq can be tens of minutes while the token reset is
    milliseconds. Any single sleep is capped by ``LLM_MAX_WAIT`` (default
    60 seconds). The chosen wait becomes the gap before later calls.

    Some Groq models emit a tool named ``json`` that was not in the request.
    Groq then rejects the whole call with HTTP 400. That turn is retried
    once with a correction naming the registered tools. The caller's message
    list is not modified.
    """

    def __init__(
        self,
        config: LLMConfig,
        transport: Transport | None = None,
        *,
        sleep: Sleeper | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not config.api_key:
            raise LLMError("LLM API key is empty.")
        self.model = config.model
        self._config = config
        self._transport = transport or urllib_transport
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._min_interval = max(0.0, config.min_interval)
        self._max_retries = max(0, config.max_retries)
        self._last_request_at = 0.0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        outgoing = list(messages)
        corrected = False
        rate_attempts = 0
        while True:
            self._wait_for_pace()
            try:
                body = self._post(outgoing, tools)
            except ToolValidationError as exc:
                self._mark_request()
                if corrected:
                    raise LLMError(str(exc)) from exc
                corrected = True
                outgoing = [*outgoing, {"role": "user", "content": _tool_correction(exc.tool_name, tools)}]
                continue
            except RateLimitError as exc:
                self._mark_request()
                delay = retry_delay(exc.retry_after, rate_attempts, max_wait=self._config.max_wait)
                self._min_interval = max(self._min_interval, delay)
                if rate_attempts >= self._max_retries:
                    raise
                rate_attempts += 1
                _log_wait(delay, exc.ignored_wait)
                self._sleep(delay)
                # The retry wait already satisfied the pace for the next try.
                self._last_request_at = self._clock() - self._min_interval
                continue
            self._mark_request()
            return parse_chat_response(body)

    def _post(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        return self._transport(url, headers, payload, self._config.timeout)

    def _wait_for_pace(self) -> None:
        if self._min_interval <= 0 or self._last_request_at <= 0:
            return
        remaining = self._min_interval - (self._clock() - self._last_request_at)
        if remaining > 0.001:
            self._sleep(remaining)

    def _mark_request(self) -> None:
        self._last_request_at = self._clock()


def urllib_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise _http_error(exc.code, detail, getattr(exc, "headers", None), str(exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"LLM request failed: {exc.reason}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMError("LLM response was not a JSON object.")
    if "error" in parsed and "choices" not in parsed:
        raise LLMError(f"LLM request failed: {error_message(parsed) or parsed['error']}")
    return parsed


def parse_chat_response(body: dict[str, Any]) -> LLMResponse:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("LLM response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMError("LLM response did not include a message.")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = str(content)
    tool_calls: list[ToolCall] = []
    for index, raw_call in enumerate(message.get("tool_calls") or []):
        parsed = _parse_tool_call(raw_call, index)
        if parsed is not None:
            tool_calls.append(parsed)
    return LLMResponse(content=content, tool_calls=tool_calls)


def error_message(detail: str | dict[str, Any]) -> str:
    if isinstance(detail, dict):
        error = detail.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error is not None:
            return str(error)
        return ""
    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        return detail.strip()[:500]
    if isinstance(parsed, dict):
        return error_message(parsed)
    return detail.strip()[:500]


class RetryPlan:
    """The wait to use, and a longer unrelated reset that was ignored."""

    def __init__(self, *, wait: float | None, ignored: float | None) -> None:
        self.wait = wait
        self.ignored = ignored


def retry_after_seconds(headers: Any, body: str) -> float | None:
    """Seconds to wait for this 429, or None when the body gives no hint.

    Explicit Retry-After and "try again in" win. Otherwise a token-limit
    response uses ``x-ratelimit-reset-tokens`` and a request-limit response
    uses ``x-ratelimit-reset-requests``. The other bucket is not consulted:
    Groq's request reset can be tens of minutes on the same response whose
    token reset is under a second.
    """

    return retry_plan(headers, body).wait


def retry_plan(headers: Any, body: str) -> "RetryPlan":
    explicit: list[float] = []
    retry_after = _header_delay(headers, "Retry-After")
    if retry_after is not None:
        explicit.append(retry_after)
    try_again = _try_again_seconds(body)
    if try_again is not None:
        explicit.append(try_again)
    resets = {
        "tokens": _header_delay(headers, "x-ratelimit-reset-tokens"),
        "requests": _header_delay(headers, "x-ratelimit-reset-requests"),
        "generic": _header_delay(headers, "x-ratelimit-reset"),
    }
    kind = limit_kind(body)
    if explicit:
        chosen = max(explicit)
        return RetryPlan(wait=chosen, ignored=_ignored_reset(resets, kind, chosen))
    if kind == "tokens" and resets["tokens"] is not None:
        return RetryPlan(wait=resets["tokens"], ignored=_longer(resets["requests"], resets["tokens"]))
    if kind == "requests" and resets["requests"] is not None:
        return RetryPlan(wait=resets["requests"], ignored=_longer(resets["tokens"], resets["requests"]))
    available = [value for value in resets.values() if value is not None]
    if not available:
        return RetryPlan(wait=None, ignored=None)
    chosen = min(available)
    longer = [value for value in available if value > chosen + 1]
    return RetryPlan(wait=chosen, ignored=max(longer) if longer else None)


def retry_delay(retry_after: float | None, attempt: int, max_wait: float = MAX_RETRY_SLEEP_SECONDS) -> float:
    """Seconds to sleep before the next try. Never longer than ``max_wait``."""

    if retry_after is None:
        base = min(2**attempt, 30.0)
    else:
        base = retry_after + RETRY_CUSHION_SECONDS
    cap = max_wait if max_wait > 0 else MAX_RETRY_SLEEP_SECONDS
    return min(cap, max(0.0, base))


def limit_kind(body: str) -> str:
    """Which bucket a 429 is about: ``tokens``, ``requests``, or ``unknown``."""

    text = f"{body or ''}\n{error_message(body or '')}".lower()
    tokens = any(
        marker in text
        for marker in ("tokens per minute", "tokens per day", "itpm", "on tokens", "(tpm)", " tpm")
    )
    requests = any(
        marker in text
        for marker in ("requests per minute", "requests per day", "on requests", "(rpm)", " rpm")
    )
    if tokens and not requests:
        return "tokens"
    if requests and not tokens:
        return "requests"
    if tokens:
        return "tokens"
    return "unknown"


def _header_delay(headers: Any, name: str) -> float | None:
    raw = _header_value(headers, name)
    if not raw:
        return None
    return _parse_delay_token(raw)


def _try_again_seconds(body: str) -> float | None:
    found = [
        float(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]
        for match in _TRY_AGAIN.finditer(body or "")
    ]
    if not found:
        return None
    return max(found)


def _longer(other: float | None, chosen: float) -> float | None:
    if other is None or other <= chosen + 1:
        return None
    return other


def _ignored_reset(resets: dict[str, float | None], kind: str, chosen: float) -> float | None:
    if kind == "tokens":
        return _longer(resets["requests"], chosen)
    if kind == "requests":
        return _longer(resets["tokens"], chosen)
    longer = [value for value in resets.values() if value is not None and value > chosen + 1]
    if not longer:
        return None
    return max(longer)


def _log_wait(delay: float, ignored: float | None) -> None:
    if delay < 0.5:
        return
    message = f"rate limit: waiting {delay:.1f}s"
    if ignored is not None and ignored > delay + 1:
        message += f"; ignored a {ignored:.0f}s reset from another bucket"
    print(message, file=sys.stderr, flush=True)


def is_tool_validation_error(detail: str) -> bool:
    text = error_message(detail).lower()
    return "tool call validation failed" in text or "not in request.tools" in text


def tool_name_from_validation_error(detail: str) -> str | None:
    match = _TOOL_NAME.search(error_message(detail))
    if match is None:
        return None
    return match.group(1)


def _http_error(code: int, detail: str, headers: Any, reason: str) -> LLMError:
    message = error_message(detail) or reason
    if code == 429:
        plan = retry_plan(headers, detail)
        return RateLimitError(
            f"LLM request failed (429): {message}",
            retry_after=plan.wait,
            ignored_wait=plan.ignored,
        )
    if code == 400 and is_tool_validation_error(detail):
        return ToolValidationError(
            f"LLM request failed (400): {message}",
            tool_name=tool_name_from_validation_error(detail) or "unknown",
        )
    return LLMError(f"LLM request failed ({code}): {message}")


def _tool_correction(tool_name: str, tools: list[dict[str, Any]]) -> str:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    available = ", ".join(names) if names else "(none)"
    return (
        f"The provider rejected a call to tool {tool_name!r} because it is not in this request. "
        f"Available tools: {available}. "
        "Call one of those, or answer in text. Do not call a tool named json."
    )


def _header_value(headers: Any, name: str) -> str | None:
    if not headers:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    value = getter(name)
    if value:
        return str(value)
    if isinstance(headers, dict):
        for key, item in headers.items():
            if str(key).lower() == name.lower() and item:
                return str(item)
    return None


def _parse_delay_token(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        return float(text)
    spans = list(_DURATION.finditer(text))
    if spans and "".join(text[span.start() : span.end()] for span in spans) == "".join(text.split()):
        return sum(float(span.group(1)) * _UNIT_SECONDS[span.group(2).lower()] for span in spans)
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, parsed.timestamp() - time.time())


def _parse_tool_call(raw_call: Any, index: int) -> ToolCall | None:
    if not isinstance(raw_call, dict):
        return None
    function = raw_call.get("function") or {}
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments") if "arguments" in function else "{}"
    if isinstance(arguments, str):
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_unparsed": arguments}
    if not isinstance(arguments, dict):
        arguments = {"_unparsed": arguments}
    call_id = raw_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{index}"
    return ToolCall(id=call_id, name=name, arguments=arguments)
