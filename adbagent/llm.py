"""Provider-agnostic LLM access. Fireworks today; OpenAI/Claude/Gemini later.

Hand-rolled rather than via LiteLLM. Three of the four target providers speak
the OpenAI wire protocol, so the abstraction a routing library sells is one
`base_url=` argument -- against 20 transitive dependencies, an `openai<3` pin,
and a documented history of silently rewriting `json_schema` into `json_object`
on the Fireworks path specifically. When correctness depends on getting a schema
back, a 300-line adapter you can read beats a transformation pipeline you cannot.

Adding a provider means adding a `Provider` entry. Anthropic and Gemini, which
do not speak this protocol, get their own subclass of `LLMClient` later; the
agent loop only ever sees `decide()` and `judge()`.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Config

log = logging.getLogger("adbagent.llm")

M = TypeVar("M", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class TruncatedResponse(LLMError):
    """finish_reason == "length": the JSON is cut off and cannot be valid."""


class BudgetExceeded(LLMError):
    pass


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    #: Control-plane catalogue that reports capabilities. The OpenAI-compatible
    #: /models endpoint exists but carries no capability flags, so it is useless
    #: for a model picker.
    catalogue_url: str = ""
    model_prefix: str = ""


PROVIDERS: Dict[str, Provider] = {
    "fireworks": Provider(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        catalogue_url="https://api.fireworks.ai/v1/accounts/fireworks/models",
        model_prefix="accounts/fireworks/models/",
    ),
    "openai": Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
}


@dataclass
class ModelInfo:
    id: str
    display_name: str = ""
    context_length: int = 0
    vision: bool = False
    tools: bool = False
    serverless: bool = True
    deprecated: str = ""

    def row(self) -> str:
        caps = []
        if self.vision:
            caps.append("vision")
        if self.tools:
            caps.append("tools")
        ctx = f"{self.context_length // 1024}k" if self.context_length else "?"
        flag = "  DEPRECATED " + self.deprecated if self.deprecated else ""
        return f"{self.id:<58} {ctx:>6}  {','.join(caps) or '-':<13}{flag}"


def list_models(provider: Provider, api_key: str,
                timeout: float = 30.0) -> List[ModelInfo]:
    """Page the catalogue so the user can pick any model.

    Note the filter grammar takes snake_case field names while the JSON response
    is camelCase.
    """
    if not provider.catalogue_url:
        raise LLMError(f"{provider.name} has no model catalogue endpoint")

    out: List[ModelInfo] = []
    page_token = ""
    with httpx.Client(timeout=timeout) as client:
        while True:
            params = {"pageSize": 200, "filter": "supports_serverless=true"}
            if page_token:
                params["pageToken"] = page_token
            resp = client.get(provider.catalogue_url, params=params,
                              headers={"Authorization": f"Bearer {api_key}"})
            if resp.status_code == 401:
                raise LLMError("model catalogue rejected the API key (401)")
            resp.raise_for_status()
            body = resp.json()
            for entry in body.get("models", []):
                if entry.get("state") not in (None, "READY"):
                    continue
                # conversationConfig present == the chat API is enabled for it.
                if "conversationConfig" not in entry:
                    continue
                name = entry.get("name", "")
                dep = entry.get("deprecationDate") or {}
                out.append(ModelInfo(
                    id=name.split("/models/")[-1] if "/models/" in name else name,
                    display_name=entry.get("displayName", ""),
                    context_length=int(entry.get("contextLength") or 0),
                    vision=bool(entry.get("supportsImageInput")),
                    tools=bool(entry.get("supportsTools")),
                    serverless=bool(entry.get("supportsServerless", True)),
                    deprecated=(f"{dep.get('year')}-{dep.get('month'):02d}-"
                                f"{dep.get('day'):02d}" if dep.get("year") else ""),
                ))
            page_token = body.get("nextPageToken") or ""
            if not page_token:
                break
    out.sort(key=lambda m: m.id)
    return out


def qualify(provider: Provider, model: str) -> str:
    """Fireworks wants fully-qualified ids; accept the short form too."""
    if not model or not provider.model_prefix:
        return model
    if model.startswith(provider.model_prefix) or model.startswith("accounts/"):
        return model
    return provider.model_prefix + model


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

@dataclass
class Call:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_s: float = 0.0
    usd: float = 0.0
    purpose: str = "decide"
    request_id: str = ""


#: USD per 1M tokens, (input, output), keyed by a substring of the model id.
#: Published Fireworks serverless prices. Reporting only -- the provider's own
#: `usage` is the billing truth -- but the budget guard depends on these being
#: non-zero, so an unknown model must NOT fall through to free.
DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    "kimi-k3": (3.00, 15.00),
    "kimi-k2p7": (0.95, 4.00),
    "kimi-k2p6": (0.95, 4.00),
    "kimi-k2": (0.95, 4.00),
    "deepseek-v4-pro": (1.74, 3.48),
    "deepseek-v4-flash": (0.14, 0.28),
    "glm-5": (1.40, 4.40),
    "qwen3p7-plus": (0.40, 1.60),
    "qwen3p6-plus": (0.40, 1.60),
    "minimax-m3": (0.30, 1.20),
    "gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-20b": (0.07, 0.30),
}

#: Used when the model is not in the table. Deliberately pessimistic: a budget
#: that silently never fires is worse than one that stops a little early, and an
#: agent loop left running against an unknown model is exactly how a bill
#: happens. Override by setting `llm.prices` in config.json.
FALLBACK_PRICE: Tuple[float, float] = (1.00, 4.00)


@dataclass
class Ledger:
    """Running cost of a session, and the thing the budget guard reads."""

    prices: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_PRICES))
    fallback: Tuple[float, float] = FALLBACK_PRICE
    calls: List[Call] = field(default_factory=list)
    total_usd: float = 0.0
    #: True when any call was priced by the fallback, so the CLI can say the
    #: figure is an estimate rather than quoting it as fact.
    estimated: bool = False

    def price_for(self, model: str) -> Tuple[float, float]:
        best: Optional[Tuple[str, Tuple[float, float]]] = None
        for key, price in self.prices.items():
            if key in model and (best is None or len(key) > len(best[0])):
                best = (key, price)
        if best is not None:
            return best[1]
        self.estimated = True
        return self.fallback

    def record(self, call: Call) -> Call:
        pin, pout = self.price_for(call.model)
        call.usd = (call.prompt_tokens * pin + call.completion_tokens * pout) / 1e6
        self.total_usd += call.usd
        self.calls.append(call)
        return call

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def tokens(self) -> Tuple[int, int]:
        return (sum(c.prompt_tokens for c in self.calls),
                sum(c.completion_tokens for c in self.calls))


class RateLimiter:
    """Simple client-side throttle. Fireworks free accounts allow 10 RPM."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()


_LIMITERS: Dict[Tuple[str, int], RateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def shared_limiter(provider: str, rpm: int) -> RateLimiter:
    """One throttle per (provider, rpm) for the whole process.

    The account limit is account-wide, so it must not reset every time a new
    client is constructed -- otherwise a repeat loop bursts straight through it.
    """
    key = (provider, rpm)
    with _LIMITER_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = RateLimiter(rpm)
            _LIMITERS[key] = limiter
        return limiter


# ---------------------------------------------------------------------------
# Schema hardening and JSON extraction
# ---------------------------------------------------------------------------

def harden_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Make a pydantic schema safe for constrained decoders.

    Inlines `$defs` (external refs are not resolvable server-side and nested
    definitions are handled inconsistently), and forbids extra properties
    everywhere so the model cannot invent fields.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**target, **merged})
            out = {k: resolve(v) for k, v in node.items()}
            if out.get("type") == "object" and "properties" in out:
                out.setdefault("additionalProperties", False)
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    hardened = resolve(schema)
    hardened.pop("$id", None)
    return hardened


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> str:
    """Pull an object out of a reply that may be fenced or prefaced."""
    if not text:
        raise LLMError("empty response")
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start == -1:
        raise LLMError(f"no JSON object in response: {text[:200]!r}")
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise LLMError("unterminated JSON object in response")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 520})


def image_part(jpeg: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def text_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


class LLMClient:
    """OpenAI-protocol client, pointed at whichever provider is configured."""

    def __init__(self, cfg: Config, run_id: str = ""):
        from openai import OpenAI  # imported here to keep CLI startup fast

        self.cfg = cfg
        self.run_id = run_id
        self.provider = PROVIDERS.get(cfg.llm.provider)
        if self.provider is None:
            raise LLMError(f"unknown provider {cfg.llm.provider!r}; "
                           f"known: {sorted(PROVIDERS)}")
        key_env = cfg.llm.api_key_env or self.provider.api_key_env
        self.api_key = __import__("os").environ.get(key_env, "")
        if not self.api_key:
            raise LLMError(f"${key_env} is not set")

        self.model = qualify(self.provider, cfg.llm.model)
        self.model_small = qualify(self.provider, cfg.llm.small())
        self.model_image = qualify(self.provider, cfg.llm.image())
        self.ledger = Ledger()
        self.limiter = shared_limiter(self.provider.name, cfg.llm.rpm)
        self._client = OpenAI(
            base_url=cfg.llm.base_url or self.provider.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(connect=10.0, read=cfg.llm.read_timeout,
                                  write=60.0, pool=10.0),
            max_retries=0,  # we retry ourselves, so backoff and logging are ours
        )

    # -- transport ---------------------------------------------------------

    def _extra_body(self) -> Dict[str, Any]:
        if self.provider.name != "fireworks":
            return {}
        return {
            # Pin the run to one replica so the static prefix keeps hitting the
            # prompt cache, which is where most of the input discount comes from.
            "prompt_cache_key": self.run_id or "adbagent",
            # Fireworks silently truncates an over-long prompt by default. For an
            # agent that would mean acting on a screen it only half saw.
            "context_length_exceeded_behavior": "error",
            # UI dumps are attacker-controllable text and can contain chat
            # template markers.
            "safe_tokenization": False,
        }

    def _post(self, messages: List[Dict[str, Any]], *, model: str,
              schema: Optional[Dict[str, Any]], max_tokens: int,
              purpose: str) -> Tuple[str, Call]:
        from openai import APIStatusError, APIConnectionError, APITimeoutError

        if self.ledger.total_usd > self.cfg.safety.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.ledger.total_usd:.4f}, budget is "
                f"${self.cfg.safety.budget_usd:.2f}")

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.llm.temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "AgentAction", "schema": schema},
            }
        extra = self._extra_body()
        if extra:
            kwargs["extra_body"] = extra

        last: Optional[Exception] = None
        retries = self.cfg.llm.max_retries
        for attempt in range(retries):
            self.limiter.wait()
            started = time.monotonic()
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except (APIConnectionError, APITimeoutError) as exc:
                last = exc
            except APIStatusError as exc:
                if exc.status_code not in RETRY_STATUS:
                    raise LLMError(
                        f"{exc.status_code} from {self.provider.name}: "
                        f"{getattr(exc, 'message', exc)}") from exc
                last = exc
            else:
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                call = self.ledger.record(Call(
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_s=time.monotonic() - started,
                    purpose=purpose,
                    request_id=getattr(resp, "id", "") or "",
                ))
                if choice.finish_reason == "length":
                    # Not a soft problem: a truncated reply is invalid JSON.
                    raise TruncatedResponse(
                        f"reply hit max_tokens={max_tokens}; raise it")
                return (choice.message.content or ""), call

            delay = min(30.0, 0.8 * (2 ** attempt)) + random.uniform(0, 0.5)
            log.warning("LLM call failed (%s); retrying in %.1fs", last, delay)
            time.sleep(delay)

        raise LLMError(f"giving up after {retries} attempts: {last}")

    # -- structured calls --------------------------------------------------

    def structured(self, messages: List[Dict[str, Any]], model_cls: Type[M], *,
                   model: str = "", max_tokens: int = 0,
                   purpose: str = "decide") -> M:
        """One schema-constrained call, with a bounded repair loop."""
        from .prompts import REPAIR

        schema = harden_schema(model_cls)
        target = model or self.model
        budget = max_tokens or self.cfg.llm.max_tokens
        conversation = list(messages)

        last_error = ""
        for attempt in range(3):
            try:
                raw, _ = self._post(conversation, model=target, schema=schema,
                                    max_tokens=budget, purpose=purpose)
            except TruncatedResponse as exc:
                # Asking again with the same ceiling would truncate again. Give
                # the model more room instead, once.
                max_budget = max(self.cfg.llm.max_tokens * 2, 6000)
                if budget >= max_budget:
                    raise
                budget = min(max_budget, budget * 2)
                log.warning("%s; retrying with max_tokens=%d", exc, budget)
                last_error = str(exc)
                continue
            try:
                return model_cls.model_validate_json(extract_json(raw))
            except (ValidationError, LLMError, json.JSONDecodeError) as exc:
                last_error = str(exc)[:600]
                log.warning("invalid structured reply (attempt %d): %s",
                            attempt + 1, last_error)
                conversation = conversation + [
                    {"role": "assistant", "content": raw[:2000]},
                    {"role": "user", "content": REPAIR.format(error=last_error)},
                ]
        raise LLMError(f"model never produced a valid {model_cls.__name__}: "
                       f"{last_error}")

    # -- the two things the agent loop actually calls -----------------------

    def decide(self, *, goal: str, rendered: str, history: Sequence[str],
               width: int, height: int, package: str = "",
               screenshot: Optional[bytes] = None, note: str = ""):
        from . import prompts
        from .actions import AgentAction

        content: List[Dict[str, Any]] = [
            text_part(prompts.screen_block(rendered, note))]
        if screenshot:
            # Text before image: it measurably improves grounding, and the image
            # is the most volatile block so it must come last for cache reuse.
            content.append(image_part(screenshot))

        messages = [
            {"role": "system",
             "content": prompts.system_prompt(harden_schema(AgentAction))},
            {"role": "user",
             "content": prompts.device_profile(width, height, package)},
            {"role": "user", "content": prompts.goal_block(goal)},
            {"role": "user", "content": prompts.history_block(history)},
            {"role": "user", "content": content},
        ]
        target = self.model_image if screenshot else self.model
        return self.structured(messages, AgentAction, model=target, purpose="decide")

    def judge(self, *, goal: str, rendered: str, history: Sequence[str],
              screenshot: Optional[bytes] = None,
              max_tokens: int = 0) -> "Verdict":
        from . import prompts

        content: List[Dict[str, Any]] = [
            text_part(prompts.judge_user(goal, history, rendered))]
        if screenshot:
            content.append(image_part(screenshot))
        messages = [
            {"role": "system", "content": prompts.JUDGE_SYSTEM},
            {"role": "user", "content": content},
        ]
        target = self.model_image if screenshot else self.model_small
        return self.structured(messages, Verdict, model=target,
                               max_tokens=max_tokens, purpose="judge")


class Verdict(BaseModel):
    """Independent check that the goal is actually satisfied."""

    satisfied: bool
    evidence: str = ""
