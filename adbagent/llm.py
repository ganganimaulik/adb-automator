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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    #: Reasoning tokens, when the provider reports them separately.
    reasoning_tokens: int = 0
    #: Characters of reasoning we actually saw on the wire. Providers do not all
    #: report `reasoning_tokens`, but a reasoning model's thinking still arrives
    #: as `reasoning_content` deltas or inside `<think>` tags -- and on a fast
    #: model the reasoning length *is* the step latency, so it has to be
    #: measurable without depending on the usage block being generous.
    reasoning_chars: int = 0
    latency_s: float = 0.0
    usd: float = 0.0
    purpose: str = "decide"
    request_id: str = ""

    def metrics(self) -> Dict[str, Any]:
        """Compact record for `events.jsonl`."""
        return {
            "purpose": self.purpose,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_chars": self.reasoning_chars,
            "latency_s": round(self.latency_s, 3),
            "usd": round(self.usd, 6),
        }


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
    #: `Prefetch` runs a call on another thread, so two can settle at once and
    #: `total_usd +=` is a read-modify-write. A lost update here understates the
    #: spend the budget guard reads, which is the one number that must not drift.
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

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
        with self._lock:
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

    # -- attribution -------------------------------------------------------
    #
    # One agent step can be several calls (a screenshot turn is an image
    # analysis *then* a decision), so "the last call" is not the cost of the
    # step. Mark before, read after.

    def mark(self) -> int:
        return len(self.calls)

    def since(self, mark: int) -> List[Call]:
        return self.calls[mark:]


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


# ---------------------------------------------------------------------------
# Reasoning depth
# ---------------------------------------------------------------------------
#
# Nothing about this is standard. Two conventions are in use on the OpenAI wire
# protocol and a model that does not know the one you sent either ignores it --
# the worst case, because the bill and the latency say nothing changed while you
# believe you fixed them -- or rejects the whole call.
#
# So the family table below is a *default*, not a detection: `llm.reasoning_style`
# overrides it, an unrecognised model sends nothing at all rather than guessing,
# and `adbagent doctor` prints the exact body that will go out so it can be
# confirmed against the provider's docs before a long run depends on it.

#: Accepted depths. "none" means "do not think", not "provider default" -- that
#: is what an empty string is for.
REASONING_LEVELS = ("none", "low", "medium", "high")

#: Families taking `reasoning_effort: "low" | "medium" | "high"`.
EFFORT_FAMILIES = ("gpt-oss", "gpt-5", "o1", "o3", "o4")
#: Families taking `chat_template_kwargs: {"thinking": bool}` -- a hybrid model
#: with one switch rather than a dial, so any non-"none" depth just turns it on.
#:
#: Version markers are deliberate. Most models do not reason at all, and within a
#: family the split is by version rather than by name: DeepSeek is hybrid from 3.1,
#: GLM from 4.5, Kimi from k2-thinking. An earlier table matched bare "deepseek-v3"
#: and "glm-4", which aimed the flag at three models that do not think.
THINKING_FAMILIES = ("deepseek-v3p1", "deepseek-v3.1", "deepseek-v4",
                     "qwen3", "glm-4p5", "glm-4.5", "glm-4p6", "glm-4.6",
                     "glm-5", "minimax-m", "kimi-k2-thinking", "kimi-k2p7",
                     "kimi-k3")

#: Families that do not reason at all, so there is nothing to cap and nothing
#: wrong. Listed only so `doctor` can say which of the two silences it is in --
#: "this model does not think" needs no action, while "I do not recognise this
#: model" may need `llm.reasoning_style` set by hand.
NON_REASONING_FAMILIES = ("llama", "mixtral", "mistral", "gemma", "phi",
                          "gpt-4o", "gpt-4-", "gpt-3", "qwen2", "qwen1",
                          "deepseek-v2", "kimi-k2p6", "kimi-k2-instruct",
                          "firefunction", "embedding", "whisper")


def reasoning_style_for(model: str) -> str:
    """Which convention `model` takes: "effort", "thinking", or "off".

    A default rather than a detection. Nothing in the OpenAI protocol reports
    whether a model reasons or how to ask it to stop, and most models do not
    reason at all -- so an unrecognised name gets "off" and the request goes out
    unchanged. `llm.reasoning_style` overrides this, and `_post` drops the field
    for good if the provider turns out to reject it.
    """
    name = (model or "").lower()
    if any(family in name for family in EFFORT_FAMILIES):
        return "effort"
    if any(family in name for family in THINKING_FAMILIES):
        return "thinking"
    return "off"


#: Request fields this module adds to control reasoning, and nothing else -- so
#: dropping them cannot take a `prompt_cache_key` or a `service_tier` with it.
REASONING_KEYS = ("reasoning_effort", "chat_template_kwargs")

#: Statuses that mean "the request was malformed" rather than "try again later".
#: A rejected reasoning field lands here, so this is where it is caught.
REJECTED_STATUS = frozenset({400, 422})


def reasoning_fields(extra_body: Optional[Dict[str, Any]]) -> List[str]:
    """Which reasoning fields a request body is carrying, if any."""
    if not extra_body:
        return []
    return [key for key in REASONING_KEYS if key in extra_body]


def known_non_reasoning(model: str) -> bool:
    """True when `model` is a familiar model that simply does not think."""
    name = (model or "").lower()
    if any(family in name for family in NON_REASONING_FAMILIES):
        return True
    return False


def reasoning_body(model: str, effort: str, style: str = "auto") -> Dict[str, Any]:
    """The request fields that ask `model` for `effort` worth of thinking.

    Empty when the depth is unset, or when neither convention is known to apply:
    sending a field the model does not understand is how you get a 400 mid-run, or
    worse, a silent no-op that looks like a fix.
    """
    if effort not in REASONING_LEVELS:
        return {}
    resolved = style if style in ("effort", "thinking", "off") else \
        reasoning_style_for(model)
    if resolved == "effort":
        # No family exposes an "off"; the floor is the closest thing to it.
        return {"reasoning_effort": "low" if effort == "none" else effort}
    if resolved == "thinking":
        return {"chat_template_kwargs": {"thinking": effort != "none"}}
    return {}


def usage_detail(usage: Any, group: str, field_name: str) -> int:
    """Read `usage.<group>.<field>`, whether it arrives typed or as a dict.

    `prompt_tokens_details.cached_tokens` and
    `completion_tokens_details.reasoning_tokens` are the two numbers that say
    whether prompt caching is working and where the latency went. Both are
    optional in the OpenAI protocol and both land in `model_extra` when the
    installed SDK predates the field, so neither can be read directly.
    """
    if usage is None:
        return 0
    details = getattr(usage, group, None)
    if details is None:
        extra = getattr(usage, "model_extra", None)
        if extra:
            details = extra.get(group)
    if details is None:
        return 0
    if isinstance(details, dict):
        return int(details.get(field_name) or 0)
    return int(getattr(details, field_name, 0) or 0)


def image_part(jpeg: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def text_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


class ScreenAnalysis(BaseModel):
    """What the vision model is allowed to say about a screenshot.

    Every field is optional in practice -- an empty string means "does not
    apply" -- so the model is never pushed into inventing a dialog or a reading
    to fill a slot. See :data:`prompts.IMAGE_ANALYSIS_SYSTEM` for why this
    replaced free prose.
    """

    model_config = ConfigDict(extra="forbid")

    reading: str = Field("", description="The fact the goal asks for, read off "
                                        "the image. Empty if not applicable.")
    item_label: str = Field("", description="The app's own caption for the item "
                                            "shown, if visible.")
    blocking_dialog: str = Field("", description="Dialog or prompt covering the "
                                                 "screen, with its buttons.")
    notable: str = Field("", description="Anything else visually important that "
                                         "the accessibility tree omits.")

    def render(self) -> str:
        """The lines worth putting in a prompt -- named, and only if filled."""
        labels = (("reading", "Reading"), ("item_label", "Item label"),
                  ("blocking_dialog", "BLOCKING"), ("notable", "Also visible"))
        lines = []
        for field_name, label in labels:
            value = " ".join(str(getattr(self, field_name) or "").split())
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)


class Prefetch:
    """A call started now and collected later, so it overlaps device work.

    The agent's expensive waits are strictly alternating: it talks to the model,
    then it talks to the phone. A per-item vision read is the one call whose
    input is already complete before the next gesture is sent, so it can run
    while the swipe and the settle happen -- roughly 1.5-2.5s of device time that
    was previously spent idle.

    A failure is swallowed and reported through `result`'s default: a prefetched
    read is an optimisation, and losing one must degrade the reading rather than
    abort the walk that was collecting it. Streaming callbacks are deliberately
    not plumbed through -- two threads writing to the same live terminal panel
    interleave into nonsense.
    """

    def __init__(self, fn: Callable[[], Any]):
        self._value: Any = None
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, args=(fn,),
                                        daemon=True, name="adbagent-prefetch")
        self._thread.start()

    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            self._value = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced via result()
            self._error = exc

    @property
    def failed(self) -> bool:
        return self._error is not None

    def result(self, default: Any = "", timeout: Optional[float] = None) -> Any:
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("prefetched call did not finish within %.1fs", timeout)
            return default
        if self._error is not None:
            log.warning("prefetched call failed: %s", self._error)
            return default
        return default if self._value is None else self._value


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
        self.api_key = cfg.api_key()
        if not self.api_key:
            key_env = cfg.llm.api_key_env or self.provider.api_key_env
            raise LLMError(f"no API key: set llm.api_key in config.json "
                           f"or ${key_env} in the environment")

        self.model = qualify(self.provider, cfg.llm.model)
        self.model_small = qualify(self.provider, cfg.llm.small())
        self.model_image = qualify(self.provider, cfg.llm.image())
        self.ledger = Ledger()
        self.limiter = shared_limiter(self.provider.name, cfg.llm.rpm)
        #: Models the provider has told us do not take a reasoning field. Most
        #: models do not reason, no catalogue says which, and the family table is
        #: a guess -- so the authoritative answer is the one the API gives, and it
        #: is remembered rather than rediscovered on every call.
        self._rejects_reasoning: set = set()
        self._client = OpenAI(
            base_url=cfg.llm.base_url or self.provider.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(connect=10.0, read=cfg.llm.read_timeout,
                                  write=60.0, pool=10.0),
            max_retries=0,  # we retry ourselves, so backoff and logging are ours
        )

    # -- transport ---------------------------------------------------------

    def _extra_body(self, model: str = "", effort: str = "") -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if model in self._rejects_reasoning:
            effort = ""
        if self.provider.name == "fireworks":
            extra.update({
                # Pin the run to one replica so the static prefix keeps hitting the
                # prompt cache, which is where most of the input discount comes from.
                "prompt_cache_key": self.run_id or "adbagent",
                # Fireworks silently truncates an over-long prompt by default. For an
                # agent that would mean acting on a screen it only half saw.
                "context_length_exceeded_behavior": "error",
                # UI dumps are attacker-controllable text and can contain chat
                # template markers.
                "safe_tokenization": False,
            })
        if self.cfg.llm.service_tier:
            extra["service_tier"] = self.cfg.llm.service_tier
        extra.update(reasoning_body(model or self.model, effort,
                                    self.cfg.llm.reasoning_style))
        return extra

    def _post(self, messages: List[Dict[str, Any]], *, model: str,
              schema: Optional[Dict[str, Any]], max_tokens: int,
              purpose: str, effort: str = "",
              on_event: Optional[Callable[..., None]] = None) -> Tuple[str, Call]:
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
            "stream": True,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.get("title") or "AgentAction",
                                "schema": schema},
            }
        # A caller that says nothing gets the depth its purpose implies, so a
        # vision transcription is never billed for a chain of thought just
        # because nobody threaded the argument through.
        extra = self._extra_body(model, effort or self.cfg.llm.effort_for(purpose))
        if extra:
            kwargs["extra_body"] = extra

        last: Optional[Exception] = None
        retries = self.cfg.llm.max_retries
        for attempt in range(retries):
            self.limiter.wait()
            started = time.monotonic()
            try:
                stream_kwargs = dict(kwargs)
                stream_kwargs["stream_options"] = {"include_usage": True}
                try:
                    resp = self._client.chat.completions.create(**stream_kwargs)
                except TypeError:
                    # An SDK old enough not to know `stream_options` rejects it
                    # before the request leaves the process, so retrying without
                    # it costs nothing. `except Exception` here caught the real
                    # API errors too -- a 400 or a 401 was silently reissued, so
                    # every hard failure cost two calls and the retry's exception
                    # was the one that got reported.
                    stream_kwargs.pop("stream_options", None)
                    resp = self._client.chat.completions.create(**stream_kwargs)
            except (APIConnectionError, APITimeoutError) as exc:
                last = exc
            except APIStatusError as exc:
                # Most models do not reason, and asking one of them to think less
                # is a bad request. That must not end a run 90 steps in over an
                # optimisation, so the field is dropped, the model is remembered,
                # and the call is reissued exactly once. If it fails again the
                # reasoning field was not the problem and the error stands.
                if (exc.status_code in REJECTED_STATUS
                        and reasoning_fields(kwargs.get("extra_body"))):
                    dropped = reasoning_fields(kwargs["extra_body"])
                    log.warning("%s rejected %s (%s); continuing without it",
                                model, ", ".join(dropped),
                                getattr(exc, "message", exc))
                    self._rejects_reasoning.add(model)
                    kwargs["extra_body"] = {
                        k: v for k, v in kwargs["extra_body"].items()
                        if k not in REASONING_KEYS} or None
                    if kwargs["extra_body"] is None:
                        kwargs.pop("extra_body")
                    if on_event:
                        on_event("reasoning_unsupported", model=model,
                                 fields=dropped, purpose=purpose)
                    continue
                if exc.status_code not in RETRY_STATUS:
                    raise LLMError(
                        f"{exc.status_code} from {self.provider.name}: "
                        f"{getattr(exc, 'message', exc)}") from exc
                last = exc
            else:
                # Handle non-stream response object if returned (e.g. in mocks)
                if hasattr(resp, "choices") and not hasattr(resp, "__iter__"):
                    choice = resp.choices[0]
                    usage = getattr(resp, "usage", None)
                    raw_text = choice.message.content or ""
                    reasoning = (
                        getattr(choice.message, "reasoning_content", None)
                        or getattr(choice.message, "reasoning", None)
                        or getattr(choice.message, "thinking", None)
                    )
                    if not reasoning and hasattr(choice.message, "model_extra") and choice.message.model_extra:
                        reasoning = (
                            choice.message.model_extra.get("reasoning_content")
                            or choice.message.model_extra.get("reasoning")
                            or choice.message.model_extra.get("thinking")
                        )
                    if reasoning and on_event:
                        on_event("llm_stream", stream_type="thinking", text=reasoning, purpose=purpose)
                    if raw_text and on_event:
                        on_event("llm_stream", stream_type="content", text=raw_text, purpose=purpose)
                    call = self.ledger.record(Call(
                        model=model,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        cached_tokens=usage_detail(usage, "prompt_tokens_details",
                                                   "cached_tokens"),
                        reasoning_tokens=usage_detail(usage, "completion_tokens_details",
                                                      "reasoning_tokens"),
                        reasoning_chars=len(reasoning or ""),
                        latency_s=time.monotonic() - started,
                        purpose=purpose,
                        request_id=getattr(resp, "id", "") or "",
                    ))
                    if choice.finish_reason == "length":
                        raise TruncatedResponse(
                            f"reply hit max_tokens={max_tokens}; raise it")
                    return raw_text, call

                # Handle streaming generator/iterator response
                full_content: List[str] = []
                full_reasoning: List[str] = []
                usage_obj = None
                finish_reason = None
                request_id = ""
                in_think_tag = False

                try:
                    for chunk in resp:
                        if hasattr(chunk, "id") and chunk.id:
                            request_id = chunk.id
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage_obj = chunk.usage

                        if not getattr(chunk, "choices", None):
                            continue

                        choice = chunk.choices[0]
                        if getattr(choice, "finish_reason", None):
                            finish_reason = choice.finish_reason

                        delta = getattr(choice, "delta", None)
                        if not delta:
                            continue

                        # Check for explicit reasoning/thinking field
                        reasoning_chunk = (
                            getattr(delta, "reasoning_content", None)
                            or getattr(delta, "reasoning", None)
                            or getattr(delta, "thinking", None)
                        )
                        if not reasoning_chunk and hasattr(delta, "model_extra") and delta.model_extra:
                            reasoning_chunk = (
                                delta.model_extra.get("reasoning_content")
                                or delta.model_extra.get("reasoning")
                                or delta.model_extra.get("thinking")
                            )

                        if reasoning_chunk:
                            full_reasoning.append(reasoning_chunk)
                            if on_event:
                                on_event("llm_stream", stream_type="thinking", text=reasoning_chunk, purpose=purpose)

                        # Check for content chunk
                        content_chunk = getattr(delta, "content", None)
                        if content_chunk:
                            full_content.append(content_chunk)

                            if not reasoning_chunk:
                                text_to_process = content_chunk
                                while text_to_process:
                                    if not in_think_tag:
                                        if "<think>" in text_to_process:
                                            before, after = text_to_process.split("<think>", 1)
                                            if before and on_event:
                                                on_event("llm_stream", stream_type="content", text=before, purpose=purpose)
                                            in_think_tag = True
                                            text_to_process = after
                                        else:
                                            if on_event:
                                                on_event("llm_stream", stream_type="content", text=text_to_process, purpose=purpose)
                                            text_to_process = ""
                                    else:
                                        if "</think>" in text_to_process:
                                            think_part, after = text_to_process.split("</think>", 1)
                                            if think_part:
                                                full_reasoning.append(think_part)
                                                if on_event:
                                                    on_event("llm_stream", stream_type="thinking", text=think_part, purpose=purpose)
                                            in_think_tag = False
                                            text_to_process = after
                                        else:
                                            full_reasoning.append(text_to_process)
                                            if on_event:
                                                on_event("llm_stream", stream_type="thinking", text=text_to_process, purpose=purpose)
                                            text_to_process = ""
                            else:
                                if on_event:
                                    on_event("llm_stream", stream_type="content", text=content_chunk, purpose=purpose)
                except (APIConnectionError, APITimeoutError) as exc:
                    last = exc
                    delay = min(30.0, 0.8 * (2 ** attempt)) + random.uniform(0, 0.5)
                    log.warning("LLM stream failed (%s); retrying in %.1fs", last, delay)
                    time.sleep(delay)
                    continue

                raw_text = "".join(full_content)
                reasoning_text = "".join(full_reasoning)
                prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
                # The fallback must count the reasoning too: on a reasoning model
                # the thinking is most of the completion, and a `completion_tokens`
                # estimated from the JSON alone understates the call by ~20x.
                comp_tokens = (getattr(usage_obj, "completion_tokens", 0)
                               or max(1, (len(raw_text) + len(reasoning_text)) // 4))

                call = self.ledger.record(Call(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=comp_tokens,
                    cached_tokens=usage_detail(usage_obj, "prompt_tokens_details",
                                               "cached_tokens"),
                    reasoning_tokens=usage_detail(usage_obj, "completion_tokens_details",
                                                  "reasoning_tokens"),
                    reasoning_chars=len(reasoning_text),
                    latency_s=time.monotonic() - started,
                    purpose=purpose,
                    request_id=request_id,
                ))

                if finish_reason == "length":
                    raise TruncatedResponse(
                        f"reply hit max_tokens={max_tokens}; raise it")
                return raw_text, call

            delay = min(30.0, 0.8 * (2 ** attempt)) + random.uniform(0, 0.5)
            log.warning("LLM call failed (%s); retrying in %.1fs", last, delay)
            time.sleep(delay)

        raise LLMError(f"giving up after {retries} attempts: {last}")

    # -- structured calls --------------------------------------------------

    def structured(self, messages: List[Dict[str, Any]], model_cls: Type[M], *,
                   model: str = "", max_tokens: int = 0,
                   purpose: str = "decide", effort: str = "",
                   on_event: Optional[Callable[..., None]] = None) -> M:
        """One schema-constrained call, with a bounded repair loop."""
        from .prompts import REPAIR

        schema = harden_schema(model_cls)
        target = model or self.model
        budget = max_tokens or self.cfg.llm.max_tokens
        conversation = list(messages)

        kw = {}
        if on_event is not None:
            kw["on_event"] = on_event

        last_error = ""
        for attempt in range(3):
            try:
                raw, _ = self._post(conversation, model=target, schema=schema,
                                    max_tokens=budget, purpose=purpose,
                                    effort=effort, **kw)
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
                # A reply that missed the schema is the clearest evidence there
                # is that this turn was harder than the caller assumed, so the
                # repair thinks properly. Retrying at the same shallow depth
                # tends to reproduce the same malformed answer, and three of
                # those cost far more than one deeper call.
                escalated = self.cfg.llm.effort_for(purpose, hard=True)
                if escalated and escalated != effort:
                    log.info("repairing at reasoning effort %r", escalated)
                    effort = escalated
        raise LLMError(f"model never produced a valid {model_cls.__name__}: "
                       f"{last_error}")

    # -- the two things the agent loop actually calls -----------------------

    @property
    def needs_vision_pass(self) -> bool:
        """True when a screenshot has to be described before the decider sees it.

        False means the decider is looking at the image itself, and a separate
        description would be a second round trip spent on prose it does not need.
        """
        return not self.cfg.llm.vision_in_decider

    def analyze_image(self, screenshot: bytes, *, goal: str = "", rendered: str = "",
                      step: int = 0, recorder: Optional[Any] = None,
                      on_event: Optional[Callable[..., None]] = None) -> ScreenAnalysis:
        """Use the vision model (self.model_image) ONLY to analyze the image content."""
        from . import prompts

        prompt_text = prompts.image_analysis_user(goal=goal, rendered=rendered)
        content: List[Dict[str, Any]] = [
            text_part(prompt_text),
            image_part(screenshot),
        ]
        messages = [
            {"role": "system", "content": prompts.IMAGE_ANALYSIS_SYSTEM},
            {"role": "user", "content": content},
        ]
        if recorder is not None:
            recorder.dump_messages(step, messages, purpose="analyze_image")
        log.info("submitting screenshot (%d bytes) to image model (%s) for visual analysis",
                 len(screenshot), self.model_image)
        if on_event:
            on_event("llm_start", step=step, purpose="analyze_image", model=self.model_image, screenshot=True)
        t0 = time.monotonic()
        kw = {}
        if on_event is not None:
            kw["on_event"] = on_event
        try:
            # Budget comes from `llm.max_tokens_image` (falling back to
            # `llm.max_tokens`) and nowhere else. A private ceiling here used to
            # clip long screen descriptions into a TruncatedResponse and a
            # second, doubled call -- paying for the same screenshot twice to
            # save output tokens the config already said were affordable. Keep
            # the prompt short instead of the reply.
            analysis = self.structured(messages, ScreenAnalysis,
                                       model=self.model_image,
                                       max_tokens=self.cfg.llm.image_max_tokens(),
                                       purpose="analyze_image", **kw)
        except LLMError as exc:
            # A vision model that cannot hold the schema is not worth failing the
            # step over: the description is an aid, and the element list the
            # decider actually acts on is unaffected.
            log.warning("image analysis produced no usable answer: %s", exc)
            analysis = ScreenAnalysis()
        elapsed = time.monotonic() - t0
        result = analysis.render()
        last_call = self.ledger.calls[-1] if self.ledger.calls else None
        if on_event:
            on_event("llm_end", step=step, purpose="analyze_image", elapsed=elapsed, call=last_call)
            on_event("image_analysis", step=step, model=self.model_image, elapsed=elapsed, result=result)
        if recorder is not None and hasattr(recorder, "event"):
            recorder.event("image_analysis", step=step, model=self.model_image, result=result)
        return analysis

    def read_item(self, screenshot: bytes, *, goal: str = "", label: str = "",
                  step: int = 0, recorder: Optional[Any] = None,
                  on_event: Optional[Callable[..., None]] = None) -> str:
        """One short line answering the goal for the item in `screenshot`.

        `analyze_image` describes a *screen* -- layout, dialogs, navigation --
        which is the right job when the question is "what am I looking at" and
        the wrong one when the question is "what does this scale read". Its
        answers run ~1,100 characters, most of it restating the Android nav bar,
        against a ledger that keeps 110 per item. This asks for the fact instead.
        """
        from . import prompts

        messages = [
            {"role": "system", "content": prompts.ITEM_READING_SYSTEM},
            {"role": "user", "content": [
                text_part(prompts.item_reading_user(goal=goal, label=label)),
                image_part(screenshot),
            ]},
        ]
        if recorder is not None:
            recorder.dump_messages(step, messages, purpose="read_item")
        kw = {}
        if on_event is not None:
            kw["on_event"] = on_event
        raw, _ = self._post(messages, model=self.model_image, schema=None,
                            max_tokens=self.cfg.llm.image_max_tokens(),
                            purpose="read_item", **kw)
        return " ".join(raw.split())

    def decide(self, *, goal: str, rendered: str, history: Sequence[str],
               width: int, height: int, package: str = "",
               screenshot: Optional[bytes] = None, note: str = "",
               scratchpad: str = "", progress: str = "",
               step: int = 0, recorder: Optional[Any] = None,
               purpose: str = "decide", effort: str = "",
               image_analysis: Optional[str] = None,
               on_event: Optional[Callable[..., None]] = None):
        from . import prompts
        from .actions import AgentAction

        # When the deciding model can see for itself, the image goes to it
        # directly and the separate analysis call disappears -- one round trip
        # per screenshot turn instead of two, on the ~22% of turns that take one.
        # Describing a screenshot in prose and then reasoning over the prose is
        # only worth a whole extra call when the decider is blind.
        inline_image = bool(screenshot) and self.cfg.llm.vision_in_decider
        if screenshot and not image_analysis and not inline_image:
            # The agent normally runs this pass itself, because it wants the
            # structured fields for its own ledgers; this covers the callers that
            # do not -- the judge, replay, a bare `decide`.
            image_analysis = self.analyze_image(
                screenshot, goal=goal, rendered=rendered, step=step,
                recorder=recorder, on_event=on_event).render()

        content: List[Dict[str, Any]] = [
            text_part(prompts.screen_block(rendered, note, image_analysis=image_analysis or ""))]
        if inline_image:
            content.append(image_part(screenshot))  # type: ignore[arg-type]

        state_text = prompts.state_block(scratchpad, progress)
        messages = [
            {"role": "system",
             "content": prompts.system_prompt(harden_schema(AgentAction))},
            {"role": "user",
             "content": prompts.device_profile(width, height)},
            {"role": "user", "content": prompts.goal_block(goal)},
            {"role": "user", "content": prompts.history_only_block(history)},
        ]
        if state_text:
            messages.append({"role": "user", "content": state_text})
        messages.append({"role": "user", "content": content})
        if recorder is not None:
            recorder.dump_messages(step, messages, purpose=purpose)
        # Next action steps are ALWAYS decided by the main model (self.model)!
        target = self.model
        kw = {}
        if on_event is not None:
            kw["on_event"] = on_event
        return self.structured(messages, AgentAction, model=target,
                               purpose=purpose, effort=effort, **kw)

    def judge(self, *, goal: str, rendered: str, history: Sequence[str],
              screenshot: Optional[bytes] = None,
              max_tokens: int = 0, scratchpad: str = "",
              progress: str = "", done_text: str = "",
              step: int = 0, recorder: Optional[Any] = None,
              image_analysis: Optional[str] = None,
              on_event: Optional[Callable[..., None]] = None) -> "Verdict":
        from . import prompts

        if screenshot and not image_analysis:
            # `.render()`, as `decide` does. Without it the prompt carried the
            # model's repr -- `reading='428 g' item_label='' ...` -- and, because
            # a pydantic model is always truthy, an analysis with nothing in it
            # still injected four empty fields into the block that decides
            # whether the run succeeded. `render()` returns "" for that case.
            image_analysis = self.analyze_image(
                screenshot, goal=goal, rendered=rendered, step=step,
                recorder=recorder, on_event=on_event
            ).render()

        content: List[Dict[str, Any]] = [
            text_part(prompts.judge_user(goal, history, rendered, scratchpad,
                                        progress, image_analysis=image_analysis or "",
                                        done_text=done_text))]
        messages = [
            {"role": "system", "content": prompts.JUDGE_SYSTEM},
            {"role": "user", "content": content},
        ]
        if recorder is not None:
            recorder.dump_messages(step, messages, purpose="judge")
        target = self.model_small
        kw = {}
        if on_event is not None:
            kw["on_event"] = on_event
        return self.structured(messages, Verdict, model=target,
                               max_tokens=max_tokens, purpose="judge",
                               effort=self.cfg.llm.effort_for("judge", hard=True),
                               **kw)


class Verdict(BaseModel):
    """Independent check that the goal is actually satisfied."""

    satisfied: bool
    evidence: str = ""
