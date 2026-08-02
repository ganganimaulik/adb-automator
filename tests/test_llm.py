"""LLM layer parts that need no network: schema hardening, JSON extraction,
cost accounting, throttling, model-catalogue parsing.
"""

from __future__ import annotations

import time

import pytest

from adbagent.actions import AgentAction
from adbagent.llm import (Call, Ledger, LLMError, ModelInfo, PROVIDERS, RateLimiter,
                          extract_json, harden_schema, image_part, qualify, text_part)


# ---------------------------------------------------------------------------
# Schema hardening
# ---------------------------------------------------------------------------

def test_hardening_inlines_defs():
    """$defs are handled inconsistently by constrained decoders; inline them."""
    schema = harden_schema(AgentAction)
    assert "$defs" not in schema
    assert "$ref" not in repr(schema)
    target = schema["properties"]["target"]
    assert "Target" not in repr(target.get("$ref", ""))


def test_hardening_forbids_extra_properties_everywhere():
    schema = harden_schema(AgentAction)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False, node
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)


def test_hardened_schema_is_json_serialisable():
    import json
    json.dumps(harden_schema(AgentAction))


def test_hardening_keeps_property_order():
    props = list(harden_schema(AgentAction)["properties"])
    assert props[:3] == ["observation", "reasoning", "action"]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    ('Here you go:\n{"a": 1}\nHope that helps.', '{"a": 1}'),
    ('{"a": {"b": [1, 2]}}', '{"a": {"b": [1, 2]}}'),
    (r'{"a": "has } brace"}', r'{"a": "has } brace"}'),
    (r'{"a": "escaped \" quote"}', r'{"a": "escaped \" quote"}'),
])
def test_extract_json(raw, expected):
    assert extract_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "no object here", '{"a": 1'])
def test_extract_json_rejects_junk(raw):
    with pytest.raises(LLMError):
        extract_json(raw)


def test_extract_then_validate_round_trip():
    reply = ('```json\n{"observation":"a list","reasoning":"tap it",'
             '"action":"tap","target":{"index":3},"confidence":"high"}\n```')
    action = AgentAction.model_validate_json(extract_json(reply))
    assert action.action == "tap" and action.target.index == 3


# ---------------------------------------------------------------------------
# Model ids and catalogue
# ---------------------------------------------------------------------------

def test_qualify_fireworks_model_ids():
    fw = PROVIDERS["fireworks"]
    assert qualify(fw, "kimi-k2p6") == "accounts/fireworks/models/kimi-k2p6"
    already = "accounts/fireworks/models/kimi-k2p6"
    assert qualify(fw, already) == already
    assert qualify(PROVIDERS["openai"], "gpt-4o") == "gpt-4o"


def test_model_info_row_shows_capabilities():
    row = ModelInfo(id="kimi-k2p6", context_length=262144, vision=True,
                    tools=True).row()
    assert "kimi-k2p6" in row and "vision" in row and "tools" in row and "256k" in row


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------

def test_ledger_prices_by_substring():
    ledger = Ledger(prices={"kimi-k2p6": (0.95, 4.00)})
    call = ledger.record(Call(model="accounts/fireworks/models/kimi-k2p6",
                              prompt_tokens=1_000_000,
                              completion_tokens=1_000_000))
    assert call.usd == pytest.approx(4.95)
    assert ledger.total_usd == pytest.approx(4.95)
    assert ledger.n_calls == 1
    assert ledger.tokens == (1_000_000, 1_000_000)
    assert not ledger.estimated


def test_known_fireworks_models_are_priced_out_of_the_box():
    """A budget that silently never fires is worse than no budget at all."""
    ledger = Ledger()
    call = ledger.record(Call(model="accounts/fireworks/models/deepseek-v4-flash",
                              prompt_tokens=1_000_000))
    assert call.usd == pytest.approx(0.14)
    assert not ledger.estimated


def test_the_longest_matching_price_key_wins():
    """'kimi-k2' must not shadow 'kimi-k2p6'."""
    ledger = Ledger()
    cheap = ledger.price_for("accounts/fireworks/models/deepseek-v4-flash")
    dear = ledger.price_for("accounts/fireworks/models/deepseek-v4-pro")
    assert cheap != dear


def test_an_unknown_model_still_costs_something():
    """Otherwise --budget-usd would silently do nothing on any new model."""
    ledger = Ledger()
    call = ledger.record(Call(model="brand-new-model-2027",
                              prompt_tokens=1_000_000))
    assert call.usd > 0
    assert ledger.estimated, "the figure must be flagged as an estimate"


def test_shared_limiter_is_not_reset_by_a_new_client():
    """The account limit is account-wide, so a repeat loop must not burst."""
    from adbagent.llm import shared_limiter

    a = shared_limiter("fireworks", 120)
    b = shared_limiter("fireworks", 120)
    assert a is b
    assert shared_limiter("fireworks", 10) is not a


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(rpm=600)  # 100ms apart
    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    assert time.monotonic() - start >= 0.18


def test_rate_limiter_disabled_at_zero():
    limiter = RateLimiter(rpm=0)
    start = time.monotonic()
    for _ in range(10):
        limiter.wait()
    assert time.monotonic() - start < 0.05


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------

def test_image_part_is_a_data_uri():
    part = image_part(b"\xff\xd8\xff fake jpeg")
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_text_part():
    assert text_part("hi") == {"type": "text", "text": "hi"}


def test_llm_config_model_fallbacks():
    from adbagent.config import LLMConfig

    cfg = LLMConfig(model="main-model")
    assert cfg.small() == "main-model"
    assert cfg.image() == "main-model"

    cfg_custom = LLMConfig(model="main-model", model_small="small-model", model_image="vision-model")
    assert cfg_custom.small() == "small-model"
    assert cfg_custom.image() == "vision-model"


def test_judge_uses_config_max_tokens(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    cfg.llm.max_tokens = 3200
    client = LLMClient(cfg)

    recorded_max_tokens = []
    def mock_post(messages, *, model, schema, max_tokens, purpose):
        recorded_max_tokens.append(max_tokens)
        return '{"satisfied": true, "evidence": "all good"}', None

    monkeypatch.setattr(client, "_post", mock_post)

    verdict = client.judge(goal="test", rendered="xml", history=[])
    assert verdict.satisfied is True
    assert recorded_max_tokens == [3200]


def test_decide_messages_cache_friendly_structure(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    client = LLMClient(cfg)

    from adbagent.actions import Target
    captured_messages = []
    def mock_structured(messages, model_cls, model, purpose):
        captured_messages.append(messages)
        return AgentAction(observation="on home", reasoning="tap home", action="tap", target=Target(index=1))

    monkeypatch.setattr(client, "structured", mock_structured)

    client.decide(
        goal="test goal",
        width=720, height=1600, package="com.example",
        history=["1. tap #1"],
        rendered="screen 1",
        scratchpad="collected notes",
        progress="step 1 done"
    )

    msgs = captured_messages[0]
    # Expect 6 messages: system, device, goal, pure history, state (scratchpad+progress), content
    assert len(msgs) == 6
    assert msgs[0]["role"] == "system"
    assert "Device: 720x1600 px" in msgs[1]["content"]
    assert msgs[2]["content"] == "GOAL: test goal"
    assert msgs[3]["content"] == "HISTORY (oldest first):\n1. tap #1"
    assert "YOUR SCRATCHPAD" in msgs[4]["content"]
    assert "YOUR PROGRESS" in msgs[4]["content"]
    assert "CURRENT SCREEN:\nscreen 1" in msgs[5]["content"][0]["text"]

