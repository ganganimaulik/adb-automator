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
import hashlib
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from .background import Prefetch  # re-exported: callers import it from here
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


#: Catalogue `kind`s that do not serve chat completions, so cannot be what any
#: of the `llm.model*` settings points at.
#:
#: A denylist and not an allowlist: the kinds a provider files chat models under
#: grow -- this one catalogue already spreads them over `HF_BASE_MODEL` and
#: `CUSTOM_MODEL` -- and a picker that hides a model the account can really use
#: is worse than one that lists a model it cannot. Add a kind here when one turns
#: up that a chat call cannot be made against.
NON_CHAT_KINDS = frozenset({"EMBEDDING_MODEL"})


def list_models(provider: Provider, api_key: str,
                timeout: float = 30.0) -> List[ModelInfo]:
    """Page the catalogue for the chat models the user can pick from.

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
                # `conversationConfig` is necessary but not sufficient, which is
                # what this filter used to assume on its own: an embedding model
                # carries the chat template it inherited from its base model, so
                # both the embedder and the reranker came through it looking like
                # models to drive a phone with. `kind` is what separates them.
                if "conversationConfig" not in entry:
                    continue
                if entry.get("kind") in NON_CHAT_KINDS:
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
#:
#: The second row is the 2026 open-weight generation, which moved off the chat
#: template and onto this field: Fireworks answers `chat_template_kwargs` with a
#: 400 ("Extra inputs are not permitted") for all three, and takes
#: `reasoning_effort` for all three. Checked against the API, not inferred from
#: the names -- and being wrong about a newer sibling still only costs the one
#: call `_post` spends discovering it.
EFFORT_FAMILIES = ("gpt-oss", "gpt-5", "o1", "o3", "o4",
                   "deepseek-v4", "kimi-k3", "qwen3p7")

#: Effort families whose "none" is a real setting rather than a 400. gpt-oss and
#: the o-series expose no off switch, so for them "none" has to be floored to the
#: lowest real level -- but these take it and honour it (`reasoning_content` comes
#: back empty and the completion roughly halves), and flooring it would quietly
#: buy the chain of thought the config asked not to have. On deepseek-v4-flash
#: that is the whole of the setting: low, medium and high are indistinguishable
#: from sending nothing, and only "none" changes anything.
EFFORT_NONE_FAMILIES = ("deepseek-v4", "kimi-k3", "qwen3p7")

#: Families taking `chat_template_kwargs: {"thinking": bool}` -- a hybrid model
#: with one switch rather than a dial, so any non-"none" depth just turns it on.
#:
#: Version markers are deliberate. Most models do not reason at all, and within a
#: family the split is by version rather than by name: DeepSeek is hybrid from 3.1,
#: GLM from 4.5, Kimi from k2-thinking. An earlier table matched bare "deepseek-v3"
#: and "glm-4", which aimed the flag at three models that do not think. The same
#: split now runs the other way too -- "qwen3" is here while "qwen3p7" is an effort
#: family above, and EFFORT_FAMILIES is matched first so the newer name wins.
THINKING_FAMILIES = ("deepseek-v3p1", "deepseek-v3.1",
                     "qwen3", "glm-4p5", "glm-4.5", "glm-4p6", "glm-4.6",
                     "glm-5", "minimax-m", "kimi-k2-thinking", "kimi-k2p7")

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

#: Generic ways a provider says "you sent a field I do not take" without naming
#: it. Fireworks answers `chat_template_kwargs` with the first of these.
#:
#: Every entry has to be specific enough that it cannot also describe a rejected
#: *image*, which is the error this whole test exists to stop misreading. A bare
#: "not allowed" would match "image inputs are not allowed" and reintroduce the
#: bug in a new provider's wording, so the generic half of the list is phrases,
#: not words.
_REJECTED_FIELD_PHRASES = (
    "extra inputs are not permitted", "additional properties",
    "unrecognized request argument", "unrecognised request argument",
    "unknown field", "unknown parameter", "unexpected keyword",
    "extra fields not permitted",
)


def reasoning_implicated(message: str, fields: Sequence[str] = ()) -> bool:
    """True when a 400's text supports blaming the reasoning field for it.

    Without this test every 400 was blamed on reasoning, because a reasoning
    field was in the body and nothing checked the message. The cost was not the
    wasted retry -- it was `_rejects_reasoning`, which is remembered for the rest
    of the process: one unrelated 400 silently stripped the configured depth from
    every later call to that model. A vision model pointed at a text-only id is
    how this was found. Fireworks answers the image part with "This model does
    not support image inputs", which was logged as "rejected reasoning_effort"
    and then held against the model for the whole run.

    Two ways to qualify: the message names a field this request actually carried
    -- `fields` comes from the body, so this cannot be fooled by a message about
    something we did not send -- or it reports one of the generic shapes above.

    The two failure modes are not symmetric, which is why the generic list errs
    towards matching. Declining to strip when we should ends the call with the
    provider's own message, which a human can read and answer with
    `llm.reasoning_style: off`. Stripping when we should not leaves a run
    mis-configured with nothing on screen to say so.
    """
    text = (message or "").lower()
    for field_name in fields or REASONING_KEYS:
        if field_name.lower() in text:
            return True
        # `chat_template_kwargs` is often reported by the key inside it.
        if field_name == "chat_template_kwargs" and "thinking" in text:
            return True
    return any(phrase in text for phrase in _REJECTED_FIELD_PHRASES)


def prefix_cache_key() -> str:
    """Affinity key naming the *prompt prefix*, not the run.

    The key exists to route requests that share a prefix to the replica that
    already holds it. Keying it on the run id -- which is what this used to do --
    named the wrong thing: every run got a fresh key, so every run started on an
    arbitrary replica with a cold cache and paid full price for a system prompt
    that is byte-identical across every run the agent has ever made. Over the
    traces in ``runs/`` that showed up as an 8.4% cache hit rate against a layout
    that supports about 55%.

    Hashing the system text instead means all runs of one agent version share a
    replica and the prefix stays warm between them, while editing the prompt
    rolls the key -- which is what you want, since the old replica is warm for
    text that no longer gets sent.

    The tradeoff is that one key concentrates load on one replica. That is right
    for a single-tenant agent driving one phone; a multi-tenant deployment wants
    a per-tenant suffix here.
    """
    from . import prompts
    digest = hashlib.sha256(prompts.SYSTEM.encode("utf-8")).hexdigest()
    return f"adbagent-{digest[:16]}"


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


def accepts_effort_none(model: str) -> bool:
    """True when `model` takes `reasoning_effort: "none"` verbatim.

    The alternative is the floor, which is a real cap for a model that has no off
    switch and a silent no-op for one that does -- so it has to be asked per
    family rather than applied to the whole convention.
    """
    name = (model or "").lower()
    return any(family in name for family in EFFORT_NONE_FAMILIES)


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
        if effort == "none" and not accepts_effort_none(model):
            # No off switch on this family; the floor is the closest thing to it.
            # Also where a forced style lands, so a model nobody has checked gets
            # the setting least likely to be a 400.
            return {"reasoning_effort": "low"}
        return {"reasoning_effort": effort}
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

    #: Private, so it stays out of the JSON schema the model is held to. Set when
    #: the call did not come back at all -- see `unavailable`.
    _failed: bool = PrivateAttr(default=False)

    @property
    def unavailable(self) -> bool:
        """True when the vision call failed, as against having nothing to report.

        Every field here is legitimately empty on a screen the tree already
        describes, so "all four blank" cannot distinguish a screenshot that was
        read and held no surprises from one that was never read. The difference
        matters because `needs_screenshot` only asks for an image when the tree
        *cannot* answer the question, and its note tells the decider to rely on
        the image -- so a swallowed failure hands the decider an instruction to
        use evidence it does not have.
        """
        return self._failed

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
        #: Names the prompt prefix rather than the run, so the static system text
        #: stays warm on one replica across runs. Sent both ways: Fireworks
        #: documents the body field in the API reference and the header in the
        #: caching guide, and they cost nothing together.
        self.cache_key = prefix_cache_key()
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
                # Pin to one replica so the static prefix keeps hitting the prompt
                # cache, which is where most of the input discount comes from.
                "prompt_cache_key": self.cache_key,
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
        if self.provider.name == "fireworks":
            # Same key as `prompt_cache_key` above. The API reference documents
            # the body field and the caching guide documents this header; which
            # one the serving path actually reads is not stated, so send both.
            kwargs["extra_headers"] = {"x-session-affinity": self.cache_key}

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
                #
                # Only when the message supports the accusation, though: a 400
                # naming a different cause is that cause. See
                # `reasoning_implicated`.
                message = str(getattr(exc, "message", exc) or "")
                sent = reasoning_fields(kwargs.get("extra_body"))
                if (exc.status_code in REJECTED_STATUS and sent
                        and reasoning_implicated(message, sent)):
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
        return not self.cfg.llm.decider_sees()

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
        shot = ""
        if recorder is not None:
            recorder.dump_messages(step, messages, purpose="analyze_image")
            if hasattr(recorder, "screenshot"):
                # Kept on disk so the UI can show what was submitted beside what
                # came back. A recorder without it (replay's, a test's) simply
                # leaves the panel without an image.
                shot = recorder.screenshot(step, screenshot, "analyze_image")
        log.info("submitting screenshot (%d bytes%s) to image model (%s) for visual analysis",
                 len(screenshot), f", kept as {shot}" if shot else "", self.model_image)
        if on_event:
            on_event("llm_start", step=step, purpose="analyze_image",
                     model=self.model_image, screenshot=True, shot=shot)
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
            # decider actually acts on is unaffected. But the step has to *know*,
            # or it goes on believing it looked -- so the empty analysis is
            # flagged rather than merely logged. See `ScreenAnalysis.unavailable`.
            log.warning("image analysis produced no usable answer: %s", exc)
            analysis = ScreenAnalysis()
            analysis._failed = True
            if on_event:
                on_event("vision_unavailable", step=step, model=self.model_image,
                         error=str(exc))
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
               width: int, height: int, package: str = "", today: str = "",
               screenshot: Optional[bytes] = None, note: str = "",
               scratchpad: str = "", progress: str = "", skill: str = "",
               policy: str = "", budget: str = "",
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
        inline_image = bool(screenshot) and self.cfg.llm.decider_sees()
        # `is None` and not falsiness. An analysis that ran and found nothing
        # worth reporting renders to "", which read as "no analysis was done" and
        # bought a second one -- the same screenshot described twice, on every
        # turn the vision model correctly had nothing to say. Seen in
        # ``runs/9b205cb055b4``: two `analyze_image` calls on step 3, 1,663 prompt
        # tokens each, both answering with four empty fields. None means nobody
        # looked; "" means somebody looked and the screen held no surprises.
        if screenshot and image_analysis is None and not inline_image:
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

        state_text = prompts.state_block(scratchpad, progress, budget=budget)
        messages = [
            {"role": "system",
             "content": prompts.system_prompt(harden_schema(AgentAction))},
            {"role": "user",
             "content": prompts.device_profile(width, height, today=today)},
            {"role": "user", "content": prompts.goal_block(goal)},
        ]
        if policy:
            # Above the skill and the history: for a watch this is the one block
            # that is byte-identical on every turn for days, so it belongs as
            # early in the prefix as the goal itself. See `prompts.policy_block`.
            messages.append({"role": "user",
                             "content": prompts.policy_block(policy)})
        if skill:
            # Above the history on purpose -- see `prompts.skill_block`.
            messages.append({"role": "user", "content": prompts.skill_block(skill)})
        messages.append(
            {"role": "user", "content": prompts.history_only_block(history)})
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

        if screenshot and image_analysis is None:
            # `.render()`, as `decide` does. Without it the prompt carried the
            # model's repr -- `reading='428 g' item_label='' ...` -- and, because
            # a pydantic model is always truthy, an analysis with nothing in it
            # still injected four empty fields into the block that decides
            # whether the run succeeded. `render()` returns "" for that case --
            # which is why the guard tests `is None` rather than falsiness, or
            # that "" would buy a second look at the same frame.
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

    def goal_check(self, *, goal: str, history: Sequence[str] = (),
                   rendered: str = "", scratchpad: str = "", progress: str = "",
                   step: int = 0, recorder: Optional[Any] = None,
                   on_event: Optional[Callable[..., None]] = None) -> "Verdict":
        """Is the goal already satisfied? Asked mid-run, of a model that has not said so.

        Runs on the deciding model rather than `model_small`: this verdict can end
        the run, and the cheap model is already the weakest link in the one other
        place a verdict does that (`judge`). Cost is not the constraint here --
        `Agent` issues this inside the device round trip it was going to wait
        through anyway, so it is bought with latency that was already spent.

        No screenshot. The question is about the record the run has built, not
        about the frame it happens to be on, and a vision pass would put this call
        on the critical path it is specifically designed to stay off.
        """
        from . import prompts

        messages = [
            {"role": "system", "content": prompts.GOAL_CHECK_SYSTEM},
            {"role": "user",
             "content": prompts.goal_check_user(
                 goal, history=history, rendered=rendered,
                 scratchpad=scratchpad, progress=progress, step=step)},
        ]
        if recorder is not None:
            recorder.dump_messages(step, messages, purpose="goal_check")
        kw = {}
        if on_event is not None:
            kw["on_event"] = on_event
        return self.structured(messages, Verdict, model=self.model,
                               purpose="goal_check",
                               effort=self.cfg.llm.effort_for("goal_check"),
                               **kw)

    def replan(self, *, goal: str, rendered: str,
               tried: Sequence[Tuple[str, int]] = (), stalled: int = 0,
               scratchpad: str = "", progress: str = "",
               packages: Sequence[str] = (),
               screenshot: Optional[bytes] = None,
               image_analysis: Optional[str] = None,
               step: int = 0, recorder: Optional[Any] = None,
               on_event: Optional[Callable[..., None]] = None) -> "Strategy":
        """One call for a different approach, made from outside the step history.

        Runs on the deciding model at the hard effort: this is the single
        hardest question the run will ask, and it is asked at most a few times.
        The step history is deliberately *not* in the prompt -- it is a record of
        the approach that is failing, and handing it over is what would argue
        for one more go at it.
        """
        from . import prompts

        inline_image = bool(screenshot) and self.cfg.llm.decider_sees()
        # `is None`, for the reason spelled out in `decide`.
        if screenshot and image_analysis is None and not inline_image:
            image_analysis = self.analyze_image(
                screenshot, goal=goal, rendered=rendered, step=step,
                recorder=recorder, on_event=on_event).render()

        body = prompts.replan_user(goal, rendered=rendered, tried=tried,
                                   stalled=stalled, scratchpad=scratchpad,
                                   progress=progress, packages=packages)
        if image_analysis:
            body += f"\n\nVISUAL SCREEN ANALYSIS (from image model):\n{image_analysis}"
        content: List[Dict[str, Any]] = [text_part(body)]
        if inline_image:
            content.append(image_part(screenshot))  # type: ignore[arg-type]
        messages = [
            {"role": "system", "content": prompts.REPLAN_SYSTEM},
            {"role": "user", "content": content},
        ]
        if recorder is not None:
            recorder.dump_messages(step, messages, purpose="replan")
        kw = {}
        if on_event is not None:
            kw["on_event"] = on_event
        return self.structured(messages, Strategy, model=self.model,
                               purpose="replan",
                               effort=self.cfg.llm.effort_for("decide", hard=True),
                               **kw)


class Verdict(BaseModel):
    """Independent check that the goal is actually satisfied."""

    satisfied: bool
    evidence: str = ""


class Strategy(BaseModel):
    """A way out, asked for once when a run has stopped getting anywhere.

    Deliberately not an action. The decider already answers with actions and is
    the thing that got stuck; what it is missing is not another turn but a
    different plan, and a plan can be carried across turns while an action
    cannot. See :data:`prompts.REPLAN_SYSTEM`.
    """

    model_config = ConfigDict(extra="forbid")

    assessment: str = Field("", description="One sentence: why the current "
                                            "approach is not working.")
    strategy: str = Field("", description="A concretely different approach for "
                                          "the agent to follow.")
    abandon: bool = Field(False, description="True when the goal cannot be "
                                             "reached from here at all.")
