"""LLM layer parts that need no network: schema hardening, JSON extraction,
cost accounting, throttling, model-catalogue parsing.
"""

from __future__ import annotations

import time

import pytest

from adbagent.actions import AgentAction
from adbagent.llm import (Call, Ledger, LLMError, Location, ModelInfo, PROVIDERS,
                          RateLimiter, ScreenAnalysis, extract_json, harden_schema,
                          image_part, list_models, point_fractions, qualify,
                          text_part)


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


#: Catalogue entries as the provider returns them, trimmed to the fields the
#: parser reads. The embedder and the reranker are the shape of the real thing:
#: `kind` is `EMBEDDING_MODEL`, and both carry the chat template they inherited
#: from their base model -- which is why `conversationConfig` cannot tell them
#: apart from a model that can be driven.
CATALOGUE_ENTRIES = [
    {"name": "accounts/fireworks/models/kimi-k3", "kind": "HF_BASE_MODEL",
     "state": "READY", "contextLength": 1048576, "supportsTools": True,
     "supportsImageInput": True, "conversationConfig": {"style": ""}},
    # A closed model served through the provider is filed under its own kind.
    {"name": "accounts/fireworks/models/qwen3p7-plus", "kind": "CUSTOM_MODEL",
     "state": "READY", "supportsTools": True, "supportsImageInput": True,
     "conversationConfig": {"style": ""}},
    {"name": "accounts/fireworks/models/qwen3-embedding-8b",
     "kind": "EMBEDDING_MODEL", "state": "READY", "contextLength": 40960,
     "conversationConfig": {"style": "jinja",
                            "template": "{%- if tools %}<|im_start|>"}},
    {"name": "accounts/fireworks/models/qwen3-reranker-8b",
     "kind": "EMBEDDING_MODEL", "state": "READY", "contextLength": 40960,
     "conversationConfig": {"style": "jinja",
                            "template": "{%- if tools %}<|im_start|>"}},
    # Still uploading, and one the chat API is not enabled for at all.
    {"name": "accounts/fireworks/models/half-uploaded", "kind": "HF_BASE_MODEL",
     "state": "UPLOADING", "conversationConfig": {"style": ""}},
    {"name": "accounts/fireworks/models/not-a-chat-model", "kind": "HF_BASE_MODEL",
     "state": "READY"},
]


def fake_catalogue(monkeypatch, entries):
    """Stand in for the provider's catalogue endpoint, over two pages.

    Returns the list of query params it was asked with, so the caller can check
    the paging was followed rather than the first page taken as the whole.
    """
    import httpx

    asked = []

    class Resp:
        status_code = 200

        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, params=None, headers=None):
            asked.append(dict(params or {}))
            if "pageToken" not in (params or {}):
                return Resp({"models": entries[:1], "nextPageToken": "page-2"})
            return Resp({"models": entries[1:]})

    monkeypatch.setattr(httpx, "Client", Client)
    return asked


def test_the_catalogue_lists_only_what_a_chat_call_can_be_made_against(monkeypatch):
    """An embedding model is not a model the agent can be pointed at.

    Both of these were offered by `adbagent models` and selectable in the web
    UI's config dropdowns, where picking one only fails once a call is made --
    mid-run, several steps in.
    """
    asked = fake_catalogue(monkeypatch, CATALOGUE_ENTRIES)
    models = list_models(PROVIDERS["fireworks"], "sk-test")

    assert [m.id for m in models] == ["kimi-k3", "qwen3p7-plus"]
    # Both pages were read, the second with the token the first handed back.
    assert len(asked) == 2 and asked[1]["pageToken"] == "page-2"
    kimi = models[0]
    assert (kimi.context_length, kimi.vision, kimi.tools) == (1048576, True, True)


def test_an_unfamiliar_kind_is_still_offered(monkeypatch):
    """The provider's `kind` enum grows, so the filter is a denylist: a picker
    that hides a model the account can really use is the worse failure."""
    entries = [dict(CATALOGUE_ENTRIES[0],
                    name="accounts/fireworks/models/next-thing",
                    kind="KIND_INVENTED_NEXT_QUARTER")]
    fake_catalogue(monkeypatch, entries)
    assert [m.id for m in list_models(PROVIDERS["fireworks"], "sk-test")] == \
        ["next-thing"]


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


# ---------------------------------------------------------------------------
# Locate grounding: whatever space the model answered in, the tap gets
# fractions of the frame
# ---------------------------------------------------------------------------

def test_location_schema_imposes_no_range():
    """A `maximum` in the schema does not make a pixel-native model answer in
    fractions -- the constrained decoder deforms 540 into 1.0 or 0.54 instead,
    and the tap lands somewhere else with no error anywhere."""
    schema = harden_schema(Location)
    for axis in ("x", "y"):
        prop = schema["properties"][axis]
        assert "minimum" not in repr(prop) and "maximum" not in repr(prop)


def test_point_fractions_passes_fractions_through():
    assert point_fractions(0.25, 0.75, 576, 1280) == (0.25, 0.75)


def test_point_fractions_reads_absolute_pixels_of_the_shown_frame():
    """The pixel-native grounding models (Qwen-VL, GLM-V) answer in the
    downscaled capture's space, not the device's."""
    assert point_fractions(288, 640, 576, 1280) == (0.5, 0.5)


def test_point_fractions_reads_a_thousand_grid_when_pixels_cannot_fit():
    """0..1000 answers are only distinguishable from pixels when they overrun
    the frame -- a 700-wide answer on a 576-wide frame cannot be pixels."""
    assert point_fractions(700, 500, 576, 1280) == (0.7, 0.5)


def test_point_fractions_prefers_pixels_when_both_spaces_fit():
    """(500, 800) on a 576x1280 frame reads as pixels or grid. Overrunning 0..1
    at all is the signature of a pixel-trained model, and the providers'
    vision models that do it are pixel-native, so pixels win."""
    assert point_fractions(500, 800, 576, 1280) == (500 / 576, 800 / 1280)


def test_point_fractions_rejects_what_fits_no_space():
    """A point in no known space is a miss the caller reports, not a tap."""
    assert point_fractions(1200, 2500, 576, 1280) is None
    assert point_fractions(-3, 0.5, 576, 1280) is None


def test_point_fractions_without_a_known_frame_size():
    """An unreadable frame still carries fraction answers -- and only those."""
    assert point_fractions(0.25, 0.75, 0, 0) == (0.25, 0.75)
    assert point_fractions(288, 640, 0, 0) == (0.288, 0.64)   # the 0..1000 grid
    assert point_fractions(1200, 2500, 0, 0) is None


def test_point_fractions_propagates_not_visible():
    assert point_fractions(None, None, 576, 1280) is None
    assert point_fractions(0.5, None, 576, 1280) is None


def test_llm_config_model_fallbacks():
    from adbagent.config import LLMConfig

    cfg = LLMConfig(model="main-model")
    assert cfg.small() == "main-model"
    assert cfg.image() == "main-model"

    cfg_custom = LLMConfig(model="main-model", model_small="small-model", model_image="vision-model")
    assert cfg_custom.small() == "small-model"
    assert cfg_custom.image() == "vision-model"


def test_skill_image_falls_back_through_the_vision_model():
    from adbagent.config import LLMConfig

    assert LLMConfig(model="main-model").skill_image() == "main-model"
    assert LLMConfig(model="main-model", model_image="vision-model").skill_image() == "vision-model"
    assert LLMConfig(model="main-model", model_image="vision-model",
                     model_skill_image="skill-vision").skill_image() == "skill-vision"


def test_one_model_named_for_deciding_and_for_pictures_sees_for_itself():
    """Naming the same model twice already says the decider takes images -- the
    frame was going to it either way. Making that a checkbox as well means the
    round trip is only saved by whoever knows the checkbox is there."""
    from adbagent.config import LLMConfig

    assert LLMConfig(model="kimi-k3", model_image="kimi-k3").decider_sees() is True
    # The two accepted forms of one id (`llm.qualify` takes either).
    assert LLMConfig(
        model="kimi-k3",
        model_image="accounts/fireworks/models/kimi-k3").decider_sees() is True

    assert LLMConfig(model="text-only", model_image="seeing").decider_sees() is False
    assert LLMConfig(model="text-only", model_image="seeing",
                     vision_in_decider=True).decider_sees() is True


def test_an_unset_vision_model_is_not_a_matching_pair():
    """The pair is read off `model_image`, not `image()`. `image()` falls back to
    `model`, so reading it there would call every text-only config a seeing
    decider -- and an image part fails the whole call, not just the picture."""
    from adbagent.config import LLMConfig

    cfg = LLMConfig(model="text-only")
    assert cfg.image() == "text-only"        # the fallback, not a decision
    assert cfg.decider_sees() is False


def test_judge_uses_config_max_tokens(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    cfg.llm.max_tokens = 3200
    client = LLMClient(cfg)

    recorded_max_tokens = []
    def mock_post(messages, *, model, schema, max_tokens, purpose, **kw):
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
    def mock_structured(messages, model_cls, model, purpose, **kw):
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
    # The ledger renders its own header, so the block carries it through verbatim.
    assert "collected notes" in msgs[4]["content"]
    assert "YOUR PROGRESS" in msgs[4]["content"]
    assert "CURRENT SCREEN:\nscreen 1" in msgs[5]["content"][0]["text"]


def _capture_decide(monkeypatch, **kwargs):
    """Run one `decide` and hand back the messages it would have sent."""
    from adbagent.config import Config
    from adbagent.llm import LLMClient
    from adbagent.actions import Target

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    client = LLMClient(Config())
    captured = []

    def mock_structured(messages, model_cls, model, purpose, **kw):
        captured.append(messages)
        return AgentAction(observation="on home", reasoning="tap home",
                           action="tap", target=Target(index=1))

    monkeypatch.setattr(client, "structured", mock_structured)
    base = dict(goal="test goal", width=720, height=1600, package="com.example",
                history=["1. tap #1"], rendered="screen 1")
    base.update(kwargs)
    client.decide(**base)
    return captured[0]


def test_skill_sits_above_the_history_not_in_the_screen_note(monkeypatch):
    """A skill is chosen per app, not per turn.

    Carried in the NOTE block it rode at the end of the last message and was
    re-sent uncached every step -- 1,210 tokens median over the runs in
    ``runs/``. Above the history it survives in the prompt cache for as long as
    the run stays in one app, which is the overwhelming majority of turns.
    """
    skill = "APP SKILL & GUIDANCE (WhatsApp):\nWorkflows:\n  - open: tap search"
    msgs = _capture_decide(monkeypatch, skill=skill)

    assert msgs[2]["content"] == "GOAL: test goal"
    assert msgs[3]["content"] == skill
    assert msgs[4]["content"].startswith("HISTORY")
    # And it is not also trailing the screen, where it used to live.
    screen = msgs[-1]["content"][0]["text"]
    assert "APP SKILL" not in screen


def test_no_skill_leaves_the_message_layout_untouched(monkeypatch):
    """The block is absent, not empty: a blank message would be a cache-visible
    difference between a run with a skill and a run without one."""
    msgs = _capture_decide(monkeypatch)
    assert len(msgs) == 5          # system, device, goal, history, screen
    assert msgs[3]["content"].startswith("HISTORY")
    assert not any("APP SKILL" in str(m["content"]) for m in msgs)


def test_cache_key_names_the_prefix_so_it_survives_across_runs(monkeypatch):
    """Keyed on the run id, every run began on a cold replica and re-bought a
    system prompt that is byte-identical across every run ever made."""
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    first = LLMClient(Config(), run_id="run-one")
    second = LLMClient(Config(), run_id="run-two")

    assert first.cache_key == second.cache_key
    assert "run-one" not in first.cache_key
    assert first._extra_body()["prompt_cache_key"] == first.cache_key


def test_cache_key_rolls_when_the_prompt_it_names_changes(monkeypatch):
    """A key naming a prefix that is no longer sent points at a replica warm for
    the wrong bytes."""
    from adbagent import prompts
    from adbagent.llm import prefix_cache_key

    before = prefix_cache_key()
    monkeypatch.setattr(prompts, "SYSTEM", prompts.SYSTEM + "\nnew rule.")
    assert prefix_cache_key() != before


def test_session_affinity_header_carries_the_same_key(monkeypatch):
    """Fireworks documents the body field in the API reference and the header in
    the caching guide, and does not say which the serving path reads."""
    client, sent = _rejecting_client(monkeypatch)
    headers = []
    original = client._client.chat.completions.create

    def spy(**kw):
        headers.append(kw.get("extra_headers"))
        return original(**kw)

    monkeypatch.setattr(client._client.chat.completions, "create", spy)
    client._post([{"role": "user", "content": "hi"}], model=client.model,
                 schema=None, max_tokens=100, purpose="decide")
    assert headers[0] == {"x-session-affinity": client.cache_key}


def test_service_tier_extra_body(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    cfg.llm.service_tier = "priority"
    client = LLMClient(cfg)

    extra = client._extra_body()
    assert extra.get("service_tier") == "priority"
    assert extra.get("prompt_cache_key") is not None


def test_image_model_used_for_image_analysis_and_main_model_decides_action(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient
    from adbagent.actions import Target

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    cfg.llm.model = "main-action-model"
    cfg.llm.model_image = "vision-analysis-model"
    client = LLMClient(cfg)

    structured_models = []
    def mock_structured(messages, model_cls, model="", purpose="decide", **kw):
        structured_models.append((model, purpose))
        if model_cls is ScreenAnalysis:
            return ScreenAnalysis(blocking_dialog="permission popup, Allow button")
        # Ensure visual analysis was passed to main model
        last_msg = messages[-1]["content"][0]["text"]
        assert "VISUAL SCREEN ANALYSIS (from image model)" in last_msg
        assert "Allow button" in last_msg
        return AgentAction(observation="popup", reasoning="tap allow", action="tap", target=Target(index=1))

    monkeypatch.setattr(client, "structured", mock_structured)

    action = client.decide(
        goal="allow permission",
        width=720, height=1600,
        rendered="screen 1",
        history=[],
        screenshot=b"fake-image-bytes"
    )

    # the vision pass goes to the image model, under its own purpose
    assert ("accounts/fireworks/models/vision-analysis-model", "analyze_image") in structured_models
    # structured decision should be called with main-action-model for purpose decide
    assert ("accounts/fireworks/models/main-action-model", "decide") in structured_models
    assert action.action == "tap"


class KeepingRecorder:
    """A recorder that keeps the frames it is handed, like the real one."""

    def __init__(self):
        self.kept = []

    def dump_messages(self, step, messages, purpose="decide"):
        return ""

    def screenshot(self, step, jpeg, purpose):
        self.kept.append((step, jpeg, purpose))
        return f"step_{step:03d}_{purpose}_00c0ffee.jpg"


def _analyze(client, monkeypatch, recorder):
    monkeypatch.setattr(client, "structured",
                        lambda *a, **kw: ScreenAnalysis(reading="428 g"))
    events = []
    client.analyze_image(b"\xff\xd8jpeg", goal="weigh it", step=7,
                         recorder=recorder,
                         on_event=lambda kind, **kw: events.append((kind, kw)))
    return next(kw for kind, kw in events if kind == "llm_start")


def test_a_submitted_frame_is_kept_and_named_on_the_stream(monkeypatch):
    """The web UI shows the screenshot beside the call that was shown it, so the
    frame is kept and the file it landed in travels on `llm_start`."""
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    rec = KeepingRecorder()
    start = _analyze(LLMClient(Config()), monkeypatch, rec)

    assert rec.kept == [(7, b"\xff\xd8jpeg", "analyze_image")]
    assert start["screenshot"] is True
    assert start["shot"] == "step_007_analyze_image_00c0ffee.jpg"


class BareRecorder:
    """A recorder that dumps prompts and keeps no frames."""

    def dump_messages(self, step, messages, purpose="decide"):
        return ""


@pytest.mark.parametrize("recorder", [None, BareRecorder()])
def test_a_recorder_that_keeps_no_frames_still_reads_the_screen(monkeypatch,
                                                                recorder):
    """No recorder at all, or one without a frame store: the analysis is
    unaffected and the panel simply has no image to show."""
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    start = _analyze(LLMClient(Config()), monkeypatch, recorder)
    assert start["shot"] == ""


def test_llm_streaming_reasoning_and_content_events(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient
    from dataclasses import dataclass

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    client = LLMClient(cfg)

    @dataclass
    class MockDelta:
        content: str = None
        reasoning_content: str = None

    @dataclass
    class MockChoice:
        delta: MockDelta
        finish_reason: str = None

    @dataclass
    class MockChunk:
        choices: list
        id: str = "chunk-1"
        usage: object = None

    chunks = [
        MockChunk(choices=[MockChoice(delta=MockDelta(reasoning_content="Thinking step 1... "))]),
        MockChunk(choices=[MockChoice(delta=MockDelta(reasoning_content="Thinking step 2."))]),
        MockChunk(choices=[MockChoice(delta=MockDelta(content='{"observation": "home", "}'), finish_reason=None)]),
        MockChunk(choices=[MockChoice(delta=MockDelta(content='"reasoning": "tap home", "action": "home"}'), finish_reason="stop")]),
    ]

    class MockCompletions:
        def create(self, **kwargs):
            return iter(chunks)

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    monkeypatch.setattr(client, "_client", MockClient())

    events = []
    def on_event(kind, **kw):
        events.append((kind, kw))

    raw, call = client._post([], model="test-model", schema=None, max_tokens=100, purpose="decide", on_event=on_event)
    assert '{"observation": "home"' in raw

    stream_events = [e for e in events if e[0] == "llm_stream"]
    assert len(stream_events) == 4
    assert stream_events[0][1]["stream_type"] == "thinking"
    assert stream_events[0][1]["text"] == "Thinking step 1... "
    assert stream_events[2][1]["stream_type"] == "content"
    assert '{"observation": "home", ' in stream_events[2][1]["text"]


def test_llm_streaming_inline_think_tags(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient
    from dataclasses import dataclass

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    client = LLMClient(cfg)

    @dataclass
    class MockDelta:
        content: str = None

    @dataclass
    class MockChoice:
        delta: MockDelta
        finish_reason: str = None

    @dataclass
    class MockChunk:
        choices: list

    chunks = [
        MockChunk(choices=[MockChoice(delta=MockDelta(content="<think>\nAnalyzing screen...\n"))]),
        MockChunk(choices=[MockChoice(delta=MockDelta(content="Target element is 3.\n</think>\n"))]),
        MockChunk(choices=[MockChoice(delta=MockDelta(content='{"action": "home"}'))]),
    ]

    class MockCompletions:
        def create(self, **kwargs):
            return iter(chunks)

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    monkeypatch.setattr(client, "_client", MockClient())

    events = []
    def on_event(kind, **kw):
        events.append((kind, kw))

    raw, call = client._post([], model="test-model", schema=None, max_tokens=100, purpose="decide", on_event=on_event)

    stream_events = [e for e in events if e[0] == "llm_stream"]
    thinking_text = "".join(e[1]["text"] for e in stream_events if e[1]["stream_type"] == "thinking")
    content_text = "".join(e[1]["text"] for e in stream_events if e[1]["stream_type"] == "content")

    assert "Analyzing screen..." in thinking_text
    assert "Target element is 3." in thinking_text
    assert '{"action": "home"}' in content_text


def test_judge_user_includes_done_text_and_system_prompt_guidelines():
    from adbagent import prompts

    prompt = prompts.judge_user(
        goal="let me know how i can improve my chat",
        history=["1. tap chat"],
        rendered="Screen XML",
        scratchpad="Collected 5 match message histories",
        done_text="Here are 3 tips to improve your chat response rate..."
    )

    assert "GOAL: let me know how i can improve my chat" in prompt
    assert "AGENT DONE SUMMARY / OUTPUT:\nHere are 3 tips to improve your chat response rate..." in prompt
    assert "COLLECTED DATA (agent's scratchpad):\nCollected 5 match message histories" in prompt

    # Verify JUDGE_SYSTEM guidelines prohibit rejecting done solely because output is text-based
    assert "Do NOT reject 'done' simply because output/advice/results appear in text/scratchpad" in prompts.JUDGE_SYSTEM
    assert "information retrieval, advice, recommendations, analysis" in prompts.JUDGE_SYSTEM


def test_judge_passes_done_text_to_prompt(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    client = LLMClient(cfg)

    captured_messages = []
    def mock_post(messages, *, model, schema, max_tokens, purpose, **kw):
        captured_messages.extend(messages)
        return '{"satisfied": true, "evidence": "advice provided"}', None

    monkeypatch.setattr(client, "_post", mock_post)

    verdict = client.judge(
        goal="give me chat advice",
        rendered="Chat List",
        history=[],
        done_text="Your chat advice summary"
    )

    assert verdict.satisfied is True
    user_msg = captured_messages[1]["content"][0]["text"]
    assert "AGENT DONE SUMMARY / OUTPUT:\nYour chat advice summary" in user_msg


def _judge_prompt_with_analysis(monkeypatch, analysis) -> str:
    """The judge's user message when the vision pass returns `analysis`."""
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    client = LLMClient(Config())

    captured = []

    def mock_post(messages, *, model, schema, max_tokens, purpose, **kw):
        captured.extend(messages)
        return '{"satisfied": true, "evidence": "ok"}', None

    monkeypatch.setattr(client, "_post", mock_post)
    monkeypatch.setattr(client, "analyze_image", lambda *a, **kw: analysis)
    client.judge(goal="weigh the chicken", rendered="Photo viewer",
                 history=[], screenshot=b"jpeg")
    return captured[1]["content"][0]["text"]


def test_judge_is_shown_the_rendered_analysis_not_the_object(monkeypatch):
    """`decide` renders it; `judge` used to interpolate the model itself, so the
    prompt that grades the run carried `reading='428 g' item_label=''` instead of
    the named lines."""
    from adbagent.llm import ScreenAnalysis

    prompt = _judge_prompt_with_analysis(
        monkeypatch, ScreenAnalysis(reading="428 g", item_label="9:33 am"))

    assert "Reading: 428 g" in prompt
    assert "Item label: 9:33 am" in prompt
    assert "reading=" not in prompt          # no pydantic repr
    assert "blocking_dialog" not in prompt   # nor its empty fields


def test_an_empty_analysis_adds_no_section_to_the_judge_prompt(monkeypatch):
    """A pydantic model is always truthy, so an analysis that found nothing used
    to inject four empty fields under a heading saying the image was analysed."""
    from adbagent.llm import ScreenAnalysis

    prompt = _judge_prompt_with_analysis(monkeypatch, ScreenAnalysis())

    assert "VISUAL SCREEN ANALYSIS" not in prompt







# ---------------------------------------------------------------------------
# Prompt prefix stability
# ---------------------------------------------------------------------------
#
# Providers cache on an exact token prefix, so the value of the layout is
# entirely in what does *not* change between two consecutive calls. These tests
# assert that property directly rather than asserting the text.

def test_the_device_profile_holds_still_when_the_app_changes():
    """It sits above the goal and the history, so one app switch used to evict
    both. The screen block names the current app in its own header anyway."""
    from adbagent import prompts

    profile = prompts.device_profile(720, 1600)
    assert profile == "Device: 720x1600 px"
    # Passing a package is tolerated and ignored, so an old caller cannot
    # silently reintroduce the churn.
    assert prompts.device_profile(720, 1600, package="com.whatsapp") == profile


def test_the_device_profile_carries_the_phones_date():
    """A goal that says "today" is unreadable without one.

    In ``runs/963a4f4ae96c`` -- "check today and yesterday's messages" -- the
    prompt held no date at all, and the run walked a recency-ordered list past
    the window into Sunday, Saturday and 27 Jul.
    """
    from adbagent import prompts

    profile = prompts.device_profile(720, 1600, today="2026-08-06")
    assert "Today is Thursday 2026-08-06" in profile
    # Yesterday too: it is the common half of the phrase, and the model should
    # not be doing calendar arithmetic in its head to get it.
    assert "yesterday was Wednesday 2026-08-05" in profile
    # Still the same message it was, and still holding still across an app
    # switch -- the date is a fact about the run, not about the foreground app.
    assert profile.startswith("Device: 720x1600 px | ")
    assert prompts.device_profile(720, 1600, today="2026-08-06",
                                 package="com.whatsapp") == profile


def test_the_clock_stays_out_of_the_cached_prefix():
    """Date only, never the time.

    This message sits above the goal, the skill, the history and the screen. At
    minute resolution it would change on every turn and evict all four; the
    status bar carries the time in the screen block, which is rebuilt anyway.
    """
    from adbagent import prompts

    profile = prompts.device_profile(720, 1600, today="2026-08-06")
    assert ":" not in profile.split("|", 1)[1]


@pytest.mark.parametrize("today, expected", [
    # A month boundary, and a year one. Neither is a string edit.
    ("2026-03-01", "yesterday was Saturday 2026-02-28"),
    ("2026-01-01", "yesterday was Wednesday 2025-12-31"),
])
def test_yesterday_survives_a_boundary(today, expected):
    from adbagent import prompts

    assert expected in prompts.date_facts(today)


@pytest.mark.parametrize("today", ["", "   ", "yesterday", "06/08/2026", None])
def test_a_date_the_phone_would_not_give_leaves_the_line_out(today):
    """No date beats a guessed one: the prompt states this as fact and the model
    has nothing on screen to check it against."""
    from adbagent import prompts

    assert prompts.date_facts(today) == ""
    assert prompts.device_profile(720, 1600, today=today) == "Device: 720x1600 px"


def test_the_date_sits_above_the_goal_that_needs_it(monkeypatch):
    msgs = _capture_decide(monkeypatch, today="2026-08-06")
    assert "Today is Thursday 2026-08-06" in msgs[1]["content"]
    assert msgs[2]["content"] == "GOAL: test goal"


def test_the_current_app_is_still_reported_somewhere():
    from adbagent.fingerprint import attach
    from adbagent.screen import parse, render
    from tests import xmlgen as X

    rendered = render(attach(parse(X.settings_screen(), width=X.W, height=X.H)))
    assert "app=com.android.settings" in rendered


def test_the_history_block_only_grows_between_jumps():
    """The point of the whole exercise. A sliding window rewrites the block on
    every turn and caches nothing; this one is append-only for CHUNK turns."""
    from adbagent.prompts import HISTORY_CHUNK, history_only_block

    history = [f"{i}. tap #{i} -> success" for i in range(1, 60)]
    rewrites = 0
    previous = ""
    for n in range(1, len(history) + 1):
        block = history_only_block(history[:n])
        if previous and not block.startswith(previous):
            rewrites += 1
        previous = block
    # One rewrite per jump, not one per turn.
    assert rewrites <= len(history) // HISTORY_CHUNK + 1
    assert rewrites < len(history) / 4


def test_the_rendered_history_stays_bounded():
    from adbagent.prompts import (HISTORY_CHUNK, HISTORY_KEEP,
                                  history_only_block)

    history = [f"{i}. tap #{i} -> success" for i in range(1, 400)]
    for n in (1, 5, 11, 50, 399):
        shown = [l for l in history_only_block(history[:n]).splitlines()
                 if l[:1].isdigit()]
        assert len(shown) <= HISTORY_KEEP + HISTORY_CHUNK, n
        # The newest step is never the one dropped.
        assert shown[-1] == history[n - 1], n


def test_the_omission_marker_states_a_count_not_a_step_number():
    """"since step N" would be one more thing changing every turn, inside the
    part of the prompt whose whole job is to hold still."""
    from adbagent.prompts import history_only_block

    block = history_only_block([f"{i}. tap" for i in range(1, 40)])
    assert "earlier step(s) omitted" in block
    assert "since step" not in block


def test_an_empty_history_says_so():
    from adbagent.prompts import history_only_block

    assert "nothing yet" in history_only_block([])


def test_the_window_is_quantised_but_can_be_turned_off():
    from adbagent.prompts import history_window

    assert history_window(5, keep=10, chunk=6) == 0        # nothing to drop yet
    # Quantised: the start only moves in multiples of the chunk.
    assert history_window(17, keep=10, chunk=6) == 6
    assert history_window(21, keep=10, chunk=6) == 6
    assert history_window(23, keep=10, chunk=6) == 12
    # chunk=0 is a plain sliding window, for a call with nothing after it.
    assert history_window(23, keep=10, chunk=0) == 13


def test_the_judge_sees_far_more_of_the_run_than_a_decide_turn():
    """A verdict on "did this run collect what was asked for", reached from the
    tail of it, is a verdict reached from the wrong evidence. The decide window
    is tuned for what one turn needs next; the judge gets the whole run."""
    from adbagent import prompts

    history = [f"{i}. tap #{i} -> success" for i in range(1, 71)]
    judged = prompts.judge_user("g", history, "screen")
    decided = prompts.history_only_block(history)
    # Short enough to fit JUDGE_HISTORY_KEEP whole, long enough that a decide
    # turn has had to drop the start of it.
    assert judged.count("-> success") == len(history)
    assert judged.count("-> success") > decided.count("-> success") * 2
    assert "1. tap #1 -> success" in judged


def test_the_judges_view_is_still_capped():
    from adbagent import prompts

    history = [f"{i}. tap #{i}" for i in range(1, 500)]
    judged = prompts.judge_user("g", history, "screen")
    assert judged.count(". tap #") <= prompts.JUDGE_HISTORY_KEEP
    assert "earlier step(s) omitted" in judged


# ---------------------------------------------------------------------------
# Vision without a second round trip
# ---------------------------------------------------------------------------

def _client(monkeypatch, **llm_cfg):
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    cfg.llm.model = "main-model"
    cfg.llm.model_image = "vision-model"
    for key, value in llm_cfg.items():
        setattr(cfg.llm, key, value)
    return LLMClient(cfg)


def _stub_decide(monkeypatch, client):
    """Capture the messages `decide` builds, and count analyze_image calls."""
    from adbagent.actions import Target

    seen = {"messages": None, "analyses": 0}

    def structured(messages, model_cls, model="", purpose="decide", **kw):
        seen["messages"] = messages
        return AgentAction(observation="o", reasoning="r", action="tap",
                           target=Target(index=1))

    def analyze_image(screenshot, **kw):
        seen["analyses"] += 1
        return ScreenAnalysis(reading="428 g on the scale",
                              notable="a dimmed Save button")

    monkeypatch.setattr(client, "structured", structured)
    monkeypatch.setattr(client, "analyze_image", analyze_image)
    return seen


def _image_parts(messages):
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            out += [p for p in content if p.get("type") == "image_url"]
    return out


def test_a_blind_decider_gets_prose_from_the_vision_model(monkeypatch):
    client = _client(monkeypatch)
    seen = _stub_decide(monkeypatch, client)

    client.decide(goal="g", rendered="screen", history=[], width=720,
                  height=1600, screenshot=b"jpeg-bytes")

    assert seen["analyses"] == 1                      # the extra round trip
    assert _image_parts(seen["messages"]) == []       # it never sees the image
    assert "VISUAL SCREEN ANALYSIS" in seen["messages"][-1]["content"][0]["text"]


def test_a_seeing_decider_gets_the_image_and_skips_the_extra_call(monkeypatch):
    """One round trip per screenshot turn instead of two. Describing an image in
    prose and then reasoning over the prose is only worth a whole extra call
    when the decider cannot see."""
    client = _client(monkeypatch, vision_in_decider=True)
    seen = _stub_decide(monkeypatch, client)

    client.decide(goal="g", rendered="screen", history=[], width=720,
                  height=1600, screenshot=b"jpeg-bytes")

    assert seen["analyses"] == 0
    parts = _image_parts(seen["messages"])
    assert len(parts) == 1
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "VISUAL SCREEN ANALYSIS" not in seen["messages"][-1]["content"][0]["text"]


def test_a_decider_that_is_its_own_vision_model_needs_no_flag(monkeypatch):
    """`model` and `model_image` naming one model makes the same statement
    `vision_in_decider` does, so it buys the same round trip: without this the
    config describes the screenshot to the model that is about to be shown it."""
    client = _client(monkeypatch, model="kimi-k3", model_image="kimi-k3")
    seen = _stub_decide(monkeypatch, client)

    client.decide(goal="g", rendered="screen", history=[], width=720,
                  height=1600, screenshot=b"jpeg-bytes")

    assert client.cfg.llm.vision_in_decider is False   # nobody set it
    assert seen["analyses"] == 0
    assert len(_image_parts(seen["messages"])) == 1
    assert client.needs_vision_pass is False


def test_no_screenshot_means_no_image_and_no_analysis(monkeypatch):
    for flag in (False, True):
        client = _client(monkeypatch, vision_in_decider=flag)
        seen = _stub_decide(monkeypatch, client)
        client.decide(goal="g", rendered="screen", history=[], width=720,
                      height=1600)
        assert seen["analyses"] == 0
        assert _image_parts(seen["messages"]) == []


def test_a_supplied_analysis_is_never_recomputed(monkeypatch):
    client = _client(monkeypatch)
    seen = _stub_decide(monkeypatch, client)
    client.decide(goal="g", rendered="screen", history=[], width=720,
                  height=1600, screenshot=b"jpeg",
                  image_analysis="already analysed")
    assert seen["analyses"] == 0
    assert "already analysed" in seen["messages"][-1]["content"][0]["text"]


def test_the_vision_pass_spends_the_configured_budget(monkeypatch):
    """A private ceiling here truncated long screen descriptions, and the repair
    paid for the same screenshot a second time at double the budget."""
    client = _client(monkeypatch, max_tokens=9000, max_tokens_image=9000)
    captured = {}

    def post(messages, *, model, schema, max_tokens, purpose, **kw):
        captured.update(max_tokens=max_tokens, purpose=purpose)
        return '{"reading":"428 g","item_label":"","blocking_dialog":"","notable":""}', Call(model=model)

    monkeypatch.setattr(client, "_post", post)
    analysis = client.analyze_image(b"jpeg", goal="read the weight")

    assert analysis.reading == "428 g"
    assert captured["purpose"] == "analyze_image"
    assert captured["max_tokens"] == 9000               # config, not a literal


def test_image_max_tokens_falls_back_to_max_tokens():
    from adbagent.config import LLMConfig
    cfg = LLMConfig()
    cfg.max_tokens = 5000
    cfg.max_tokens_image = 0
    assert cfg.image_max_tokens() == 5000


def test_image_max_tokens_used_when_set():
    from adbagent.config import LLMConfig
    cfg = LLMConfig()
    cfg.max_tokens = 5000
    cfg.max_tokens_image = 1500
    assert cfg.image_max_tokens() == 1500


def test_analyze_image_uses_image_max_tokens(monkeypatch):
    """max_tokens_image, when set, overrides max_tokens for the vision pass."""
    client = _client(monkeypatch, max_tokens=9000, max_tokens_image=2000)
    captured = {}

    def post(messages, *, model, schema, max_tokens, purpose, **kw):
        captured.update(max_tokens=max_tokens, purpose=purpose)
        return '{"reading":"428 g","item_label":"","blocking_dialog":"","notable":""}', Call(model=model)

    monkeypatch.setattr(client, "_post", post)
    client.analyze_image(b"jpeg", goal="read the weight")

    assert captured["max_tokens"] == 2000               # image-specific, not 9000


def test_reading_one_item_asks_for_a_fact_not_a_screen_description(monkeypatch):
    client = _client(monkeypatch, max_tokens=2222, max_tokens_image=2222)
    captured = {}

    def post(messages, *, model, schema, max_tokens, purpose, **kw):
        captured.update(messages=messages, model=model, purpose=purpose,
                        max_tokens=max_tokens)
        return "  chicken breast on scale, 428 g\n", Call(model=model)

    monkeypatch.setattr(client, "_post", post)
    reading = client.read_item(b"jpeg", goal="read the weight",
                               label="Today, 9:52 am")

    assert reading == "chicken breast on scale, 428 g"   # collapsed, stripped
    assert captured["model"] == client.model_image       # fully qualified
    assert captured["model"].endswith("vision-model")
    assert captured["purpose"] == "read_item"
    assert captured["max_tokens"] == 2222                # config, not a literal
    assert "read the weight" in captured["messages"][1]["content"][0]["text"]
    assert "Today, 9:52 am" in captured["messages"][1]["content"][0]["text"]
    assert len(_image_parts(captured["messages"])) == 1


def test_read_item_uses_image_max_tokens(monkeypatch):
    """max_tokens_image, when set, overrides max_tokens for read_item too."""
    client = _client(monkeypatch, max_tokens=2222, max_tokens_image=800)
    captured = {}

    def post(messages, *, model, schema, max_tokens, purpose, **kw):
        captured.update(max_tokens=max_tokens, purpose=purpose)
        return "  chicken breast on scale, 428 g\n", Call(model=model)

    monkeypatch.setattr(client, "_post", post)
    client.read_item(b"jpeg", goal="read the weight", label="Today, 9:52 am")

    assert captured["max_tokens"] == 800                # image-specific, not 2222


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------

def test_a_prefetched_call_runs_while_the_caller_works():
    import threading
    from adbagent.llm import Prefetch

    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return "the reading"

    task = Prefetch(slow)
    assert started.wait(5), "the call did not start until it was collected"
    # The caller is free to do its device work here; the call is already running.
    release.set()
    assert task.result() == "the reading"


def test_a_failed_prefetch_degrades_the_reading_instead_of_the_run():
    """An item reading is an optimisation. Losing one must cost the reading, not
    the walk that was collecting it."""
    from adbagent.llm import Prefetch

    task = Prefetch(lambda: (_ for _ in ()).throw(RuntimeError("502 from vision")))
    assert task.result(default="") == ""
    assert task.failed


def test_a_prefetch_returning_nothing_falls_back_to_the_default():
    from adbagent.llm import Prefetch

    assert Prefetch(lambda: None).result(default="nothing read") == "nothing read"


def test_the_ledger_survives_two_calls_settling_at_once():
    """`total_usd +=` is a read-modify-write, and it is the number the budget
    guard reads."""
    import threading

    ledger = Ledger()
    ledger.prices = {"m": (1000.0, 1000.0)}

    def record():
        for _ in range(200):
            ledger.record(Call(model="m", prompt_tokens=1, completion_tokens=1))

    threads = [threading.Thread(target=record) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ledger.n_calls == 800
    assert ledger.total_usd == pytest.approx(800 * 2 / 1e3)


# ---------------------------------------------------------------------------
# The vision contract
# ---------------------------------------------------------------------------
#
# Free prose ran 1,143 chars median and 2,284 at worst over the runs in `runs/`,
# half of them spending a sentence on the Android nav bar. Four named fields ask
# for the same facts without the padding.

def test_the_vision_schema_asks_for_four_named_facts():
    schema = harden_schema(ScreenAnalysis)
    assert set(schema["properties"]) == {
        "reading", "item_label", "blocking_dialog", "notable"}
    assert schema["additionalProperties"] is False


def test_an_empty_field_renders_to_nothing():
    """Every field is optional in practice, so the model is never pushed into
    inventing a dialog to fill a slot."""
    assert ScreenAnalysis().render() == ""
    assert ScreenAnalysis(reading="428 g").render() == "Reading: 428 g"


def test_a_filled_analysis_renders_one_short_labelled_line_each():
    rendered = ScreenAnalysis(reading="428 g on the scale",
                              item_label="Today, 9:45 am",
                              blocking_dialog="Allow location? [Deny] [Allow]",
                              notable="Save button dimmed").render()
    assert rendered.splitlines() == [
        "Reading: 428 g on the scale",
        "Item label: Today, 9:45 am",
        "BLOCKING: Allow location? [Deny] [Allow]",
        "Also visible: Save button dimmed",
    ]
    # An order of magnitude under the 1,143-char median it replaces.
    assert len(rendered) < 150


def test_the_vision_prompt_forbids_the_chrome_it_kept_describing():
    """Half the old free-prose answers spent a sentence on the nav bar, which is
    in the element list on every screen and was never what was asked."""
    from adbagent import prompts
    assert "navigation buttons" in prompts.IMAGE_ANALYSIS_SYSTEM
    assert "Never describe" in prompts.IMAGE_ANALYSIS_SYSTEM


@pytest.mark.parametrize("prompt_name", ["IMAGE_ANALYSIS_SYSTEM",
                                         "ITEM_READING_SYSTEM"])
def test_both_vision_prompts_rule_out_the_clock_as_an_answer(prompt_name):
    """Against a real phone the vision model returned "3:51 PM, 71%, 1" as the
    reading and the whole tab bar as the item. Both are Android's, not the app's.
    The tree side of this was fixed by `screen.content_elements`; the pixels kept
    carrying it, and the earlier wording only forbade the chrome in passing,
    after the field descriptions."""
    from adbagent import prompts
    text = getattr(prompts, prompt_name).lower()
    assert "clock" in text
    assert "battery" in text
    assert "never the answer" in text


def test_a_vision_model_that_cannot_hold_the_schema_does_not_fail_the_step(monkeypatch):
    """The description is an aid; the element list the decider acts on is
    unaffected, so a malformed answer degrades rather than aborts."""
    client = _client(monkeypatch)

    def explode(*a, **kw):
        raise LLMError("never produced a valid ScreenAnalysis")

    monkeypatch.setattr(client, "structured", explode)
    assert client.analyze_image(b"jpeg").render() == ""


# ---------------------------------------------------------------------------
# Situational advice, gated
# ---------------------------------------------------------------------------
#
# These three blocks were 36% of the system prompt and irrelevant on most turns.
# They live in the NOTE block now, which is rebuilt every turn anyway, so varying
# them costs no prompt-cache prefix.

def test_the_system_prompt_no_longer_carries_the_situational_blocks():
    from adbagent import prompts
    assert "SCROLLING STRATEGY" not in prompts.SYSTEM
    assert "MULTI-APP NAVIGATION" not in prompts.SYSTEM
    # It kept the parts that apply on every single turn.
    assert "THE ACTIONS" in prompts.SYSTEM
    assert "SECURITY" in prompts.SYSTEM
    # A creep guard, not a law: 9,722 before the gating, 6,935 after it, and
    # 7,488 once the bounded-collection stop rule went in -- 553 characters, or
    # about 140 tokens, in the one block that is byte-identical on every call
    # and so bought once per cache lifetime rather than once per turn. What it
    # buys: `runs/963a4f4ae96c` spent 17 decide turns walking a recency-ordered
    # list past the window its goal asked for, which is nearer 90,000 tokens.
    # Anything wanting the next 500 should have a comparison like that one.
    # `tap_at` cost 300 characters here until it moved to the situational
    # notes: a last resort should be revealed by failure, not advertised on
    # every healthy turn.
    # 7,735 with the `read_each` rule -- 247 characters, about 60 tokens, telling
    # the model it can sweep without the per-item vision read. One 12-item
    # sweep's reads are 12 image calls; skipping them when the in-between
    # content does not matter saves that whole bill.
    assert len(prompts.SYSTEM) < 7800


def test_an_ordinary_turn_gets_no_situational_advice():
    from adbagent.prompts import situational_notes
    assert situational_notes() == ""


def test_scrolling_advice_arrives_once_scrolling_starts():
    from adbagent.prompts import situational_notes
    assert "SCROLLING STRATEGY" not in situational_notes()
    assert "SCROLLING STRATEGY" in situational_notes(scrolls=1)


def test_app_switching_advice_arrives_once_the_run_crosses_apps():
    from adbagent.prompts import situational_notes
    assert "SWITCHING APPS" not in situational_notes()
    assert "SWITCHING APPS" in situational_notes(packages_seen=2)


def test_tap_at_advice_arrives_only_once_the_run_has_struggled():
    from adbagent.prompts import situational_notes
    assert "tap_at" not in situational_notes()
    assert "tap_at" in situational_notes(struggle=1)


def test_the_system_prompt_does_not_advertise_the_escape_hatch():
    """tap_at is the last resort; naming it on a healthy turn invites it."""
    from adbagent import prompts
    assert "tap_at" not in prompts.SYSTEM


def test_the_blocks_stack_when_they_all_apply():
    from adbagent.prompts import situational_notes
    note = situational_notes(scrolls=3, packages_seen=2)
    assert all(block in note for block in
               ("SCROLLING STRATEGY", "SWITCHING APPS"))


def test_advice_is_gated_on_behaviour_not_on_the_goals_wording():
    """These gates used to guess the situation from English substrings.

    Every goal below was handed a page of scrolling strategy for what is a
    one-tap task, because "install" contains "all ", "call" contains "all " and
    "account" contains "count". The signature no longer takes a goal at all, so
    the whole class of mismatch is gone rather than patched word by word -- and
    a goal written in any other language now behaves the same as an English one,
    which it could not before.
    """
    from adbagent.prompts import situational_notes
    import inspect

    assert "goal" not in inspect.signature(situational_notes).parameters
    for scrolls, packages, expected in ((0, 1, ""), (1, 1, "SCROLLING"),
                                        (0, 2, "SWITCHING")):
        note = situational_notes(scrolls=scrolls, packages_seen=packages)
        assert (expected in note) if expected else note == ""


# ---------------------------------------------------------------------------
# Reasoning depth
# ---------------------------------------------------------------------------
#
# Reasoning tokens are the run's wall clock, so this is the largest single lever
# there is. It is also the easiest to get silently wrong: two conventions exist
# on the OpenAI wire protocol, and a model that does not know the one you sent
# ignores it -- which looks exactly like a fix while nothing has changed.

def test_an_unset_depth_sends_nothing():
    """The feature is off by default. Nothing about the request may change."""
    from adbagent.llm import reasoning_body

    for style in ("auto", "effort", "thinking", "off"):
        assert reasoning_body("gpt-oss-120b", "", style) == {}


def test_an_effort_family_gets_reasoning_effort():
    from adbagent.llm import reasoning_body

    assert reasoning_body("accounts/fireworks/models/gpt-oss-120b", "medium") == {
        "reasoning_effort": "medium"}
    assert reasoning_body("gpt-5", "high") == {"reasoning_effort": "high"}


def test_an_effort_family_without_an_off_switch_floors_none():
    """gpt-oss and the o-series expose no "off", so "none" becomes the lowest real
    setting. Sending `reasoning_effort: "none"` would be a 400."""
    from adbagent.llm import reasoning_body

    assert reasoning_body("gpt-oss-120b", "none") == {"reasoning_effort": "low"}
    assert reasoning_body("gpt-5", "none") == {"reasoning_effort": "low"}


def test_an_effort_family_with_an_off_switch_is_actually_switched_off():
    """The floor is a cap for a model that cannot stop and a silent no-op for one
    that can. On deepseek-v4-flash "low" is indistinguishable from sending nothing,
    so flooring "none" would buy back the whole chain of thought the config just
    asked not to have."""
    from adbagent.llm import reasoning_body

    for model in ("accounts/fireworks/models/deepseek-v4-flash-0731",
                  "kimi-k3", "qwen3p7-plus"):
        assert reasoning_body(model, "none") == {"reasoning_effort": "none"}, model
        assert reasoning_body(model, "high") == {"reasoning_effort": "high"}, model


def test_a_hybrid_family_gets_a_thinking_switch():
    """These models have one switch, not a dial, so any real depth turns it on."""
    from adbagent.llm import reasoning_body

    off = {"chat_template_kwargs": {"thinking": False}}
    on = {"chat_template_kwargs": {"thinking": True}}
    assert reasoning_body("accounts/fireworks/models/deepseek-v3p1", "none") == off
    assert reasoning_body("accounts/fireworks/models/deepseek-v3p1", "high") == on
    assert reasoning_body("qwen3-235b-a22b", "low") == on
    assert reasoning_body("glm-4p6", "low") == on


def test_an_unrecognised_model_sends_nothing_rather_than_guessing():
    """The fail-safe. A wrong field either 400s mid-run or is ignored, and being
    ignored is worse: the latency and the bill say nothing changed while the
    config says it was capped."""
    from adbagent.llm import reasoning_body, reasoning_style_for

    assert reasoning_style_for("brand-new-model-2027") == "off"
    assert reasoning_body("brand-new-model-2027", "none") == {}


def test_the_style_can_be_forced_when_the_guess_is_wrong():
    from adbagent.llm import reasoning_body

    unknown = "brand-new-model-2027"
    assert reasoning_body(unknown, "low", "effort") == {"reasoning_effort": "low"}
    assert reasoning_body(unknown, "low", "thinking") == {
        "chat_template_kwargs": {"thinking": True}}
    # And silenced even for a model that would otherwise take one.
    assert reasoning_body("deepseek-v4-flash", "low", "off") == {}


def test_a_nonsense_depth_is_ignored_not_forwarded():
    from adbagent.llm import reasoning_body

    for junk in ("very high", "0.5", "NONE", "maximum"):
        assert reasoning_body("deepseek-v4-flash", junk) == {}, junk


def test_the_depth_reaches_the_request_body(monkeypatch):
    from adbagent.config import Config
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = Config()
    cfg.llm.model = "deepseek-v4-flash"
    cfg.llm.reasoning_effort = "none"
    client = LLMClient(cfg)

    body = client._extra_body(client.model, "none")
    assert body["reasoning_effort"] == "none"
    # And the settings that were already there survive alongside it.
    assert "prompt_cache_key" in body


# ---------------------------------------------------------------------------
# Which call gets which depth
# ---------------------------------------------------------------------------

def cfg_with(effort="low", hard="high"):
    from adbagent.config import Config

    cfg = Config()
    cfg.llm.reasoning_effort = effort
    cfg.llm.reasoning_effort_hard = hard
    return cfg


def test_nothing_is_capped_until_the_feature_is_switched_on():
    from adbagent.config import Config

    cfg = Config()
    for purpose in ("decide", "judge", "analyze_image", "read_item"):
        assert cfg.llm.effort_for(purpose) == ""
        assert cfg.llm.effort_for(purpose, hard=True) == ""


def test_a_routine_turn_gets_the_shallow_depth():
    assert cfg_with().llm.effort_for("decide") == "low"


def test_a_hard_turn_gets_the_deep_one():
    assert cfg_with().llm.effort_for("decide", hard=True) == "high"


def test_the_hard_depth_falls_back_to_the_routine_one():
    assert cfg_with(hard="").llm.effort_for("decide", hard=True) == "low"


def test_vision_calls_never_pay_for_reasoning():
    """"What does this scale read" has no chain of thought worth buying, so a
    transcription is pinned to the floor even when the run is struggling."""
    cfg = cfg_with(effort="high", hard="high")
    for purpose in ("analyze_image", "read_item"):
        assert cfg.llm.effort_for(purpose) == "none"
        assert cfg.llm.effort_for(purpose, hard=True) == "none"


def test_a_vision_call_is_capped_without_the_caller_threading_it(monkeypatch):
    """`_post` derives the depth from the purpose, so a call site that forgets to
    pass one cannot accidentally buy a chain of thought."""
    from adbagent.llm import LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = cfg_with(effort="high")
    cfg.llm.model_image = "deepseek-v4-flash"
    client = LLMClient(cfg)

    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here; the body is what matters")

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    with pytest.raises(Exception):
        client._post([{"role": "user", "content": "hi"}],
                     model=client.model_image, schema=None, max_tokens=100,
                     purpose="read_item")
    assert seen["extra_body"]["reasoning_effort"] == "none"


def test_the_judge_always_thinks_properly(monkeypatch):
    """One call, at the end, and a wrong verdict throws away the whole run."""
    from adbagent.llm import LLMClient, Verdict

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    client = LLMClient(cfg_with(effort="none", hard="high"))

    seen = {}

    def structured(messages, model_cls, **kw):
        seen.update(kw)
        return Verdict(satisfied=True, evidence="fine")

    monkeypatch.setattr(client, "structured", structured)
    client.judge(goal="g", rendered="screen", history=[])
    assert seen["effort"] == "high"


def test_a_schema_violation_escalates_its_own_repair(monkeypatch):
    """Retrying a malformed answer at the same shallow depth tends to reproduce
    it, and three of those cost far more than one deeper call."""
    from adbagent.llm import Call, LLMClient

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    client = LLMClient(cfg_with(effort="none", hard="high"))

    efforts = []

    def post(messages, *, model, schema, max_tokens, purpose, effort="", **kw):
        efforts.append(effort)
        if len(efforts) == 1:
            return "not json at all", Call(model=model)
        return ('{"observation":"o","reasoning":"r","action":"press_key",'
                '"key":"back"}'), Call(model=model)

    monkeypatch.setattr(client, "_post", post)
    action = client.structured([{"role": "user", "content": "go"}], AgentAction,
                               effort="none")
    assert action.action == "press_key"
    assert efforts == ["none", "high"]


# ---------------------------------------------------------------------------
# Models that do not support reasoning
# ---------------------------------------------------------------------------
#
# Most models do not reason, nothing in the catalogue says which do, and the
# family table is a guess about names. So the guess must be cheap to be wrong
# about: the provider's rejection is the authoritative answer, and taking it
# costs one call rather than the run.

@pytest.mark.parametrize("model", [
    "llama-v3p3-70b-instruct", "mixtral-8x22b-instruct", "gpt-4o",
    "qwen2p5-72b-instruct", "deepseek-v2", "kimi-k2p6", "gemma-2-9b",
    "mistral-small", "phi-3", "nomic-embedding-text-v1p5",
])
def test_a_model_that_does_not_reason_is_asked_for_nothing(model):
    from adbagent.llm import known_non_reasoning, reasoning_body

    assert known_non_reasoning(model), model
    assert reasoning_body(model, "none") == {}, model
    assert reasoning_body(model, "high") == {}, model


@pytest.mark.parametrize("model", [
    "deepseek-v3p1", "deepseek-v4-flash", "qwen3-235b-a22b", "glm-4p6",
    "glm-5", "kimi-k3", "kimi-k2-thinking", "minimax-m2", "gpt-oss-120b",
])
def test_a_model_that_does_reason_is_asked(model):
    from adbagent.llm import known_non_reasoning, reasoning_body

    assert not known_non_reasoning(model), model
    assert reasoning_body(model, "high"), model


def test_the_version_that_gained_reasoning_is_respected():
    """Within a family the split is by version, not by name. An earlier table
    matched bare "deepseek-v3" and "glm-4" and aimed the flag at models that do
    not think."""
    from adbagent.llm import reasoning_style_for

    assert reasoning_style_for("deepseek-v3") == "off"      # pre-3.1, not hybrid
    assert reasoning_style_for("deepseek-v3p1") == "thinking"
    assert reasoning_style_for("glm-4-9b") == "off"
    assert reasoning_style_for("glm-4p6") == "thinking"
    assert reasoning_style_for("kimi-k2p6") == "off"


def test_the_version_that_changed_convention_is_respected_too():
    """The 2026 generation moved off the chat template and onto `reasoning_effort`,
    and the split is again by version inside the family -- so the newer name has to
    win over the older prefix it contains."""
    from adbagent.llm import reasoning_style_for

    assert reasoning_style_for("deepseek-v3p1") == "thinking"
    assert reasoning_style_for("deepseek-v4-flash-0731") == "effort"
    assert reasoning_style_for("qwen3-235b-a22b") == "thinking"
    assert reasoning_style_for("qwen3p7-plus") == "effort"   # not shadowed by qwen3
    assert reasoning_style_for("kimi-k2-thinking") == "thinking"
    assert reasoning_style_for("kimi-k3") == "effort"


def test_the_reasoning_fields_are_named_so_dropping_them_takes_nothing_else():
    from adbagent.llm import reasoning_fields

    body = {"prompt_cache_key": "run-1", "service_tier": "priority",
            "chat_template_kwargs": {"thinking": False}}
    assert reasoning_fields(body) == ["chat_template_kwargs"]
    assert reasoning_fields({"reasoning_effort": "low"}) == ["reasoning_effort"]
    assert reasoning_fields({"prompt_cache_key": "x"}) == []
    assert reasoning_fields(None) == []


def _rejecting_client(monkeypatch, status=400, reject_times=1,
                      model="deepseek-v3p1", field="chat_template_kwargs"):
    """A client whose provider rejects any request carrying a reasoning field.

    Parametrised by convention, because either one can be the wrong guess: a
    thinking family that has moved to `reasoning_effort` is how this was found,
    and the drop has to work the same way whichever field went out.
    """
    from adbagent.llm import LLMClient
    from openai import APIStatusError
    import httpx

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = cfg_with(effort="none", hard="high")
    cfg.llm.model = model
    client = LLMClient(cfg)

    sent = []
    state = {"rejections": 0}

    class Chunk:
        def __init__(self):
            self.id = "req-1"
            self.usage = None
            self.choices = [type("C", (), {
                "finish_reason": "stop",
                "delta": type("D", (), {"content": "ok", "model_extra": None})(),
            })()]

    def create(**kwargs):
        sent.append(kwargs.get("extra_body") or {})
        if (field in (kwargs.get("extra_body") or {})
                and state["rejections"] < reject_times):
            state["rejections"] += 1
            raise APIStatusError(
                f"Extra inputs are not permitted, field: '{field}'",
                response=httpx.Response(status, request=httpx.Request("POST", "http://x")),
                body=None)
        return iter([Chunk()])

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    return client, sent


@pytest.mark.parametrize("model,field", [
    ("deepseek-v3p1", "chat_template_kwargs"),
    ("deepseek-v4-flash", "reasoning_effort"),
])
def test_a_rejected_reasoning_field_is_dropped_rather_than_ending_the_run(
        monkeypatch, model, field):
    """A 400 ninety steps into a run, over an optimisation, is not acceptable."""
    client, sent = _rejecting_client(monkeypatch, model=model, field=field)

    raw, call = client._post([{"role": "user", "content": "hi"}],
                             model=client.model, schema=None, max_tokens=100,
                             purpose="decide")
    assert raw == "ok"
    assert len(sent) == 2                              # rejected, then retried
    assert field in sent[0]
    assert field not in sent[1]


def test_dropping_the_field_keeps_the_rest_of_the_body(monkeypatch):
    """`prompt_cache_key` is where most of the input discount comes from; losing
    it to a reasoning retry would be an expensive way to fix a cheap problem."""
    client, sent = _rejecting_client(monkeypatch)
    client._post([{"role": "user", "content": "hi"}], model=client.model,
                 schema=None, max_tokens=100, purpose="decide")
    assert sent[1]["prompt_cache_key"]
    assert sent[1]["context_length_exceeded_behavior"] == "error"


def test_the_rejection_is_remembered_so_it_costs_one_call_not_every_call(monkeypatch):
    client, sent = _rejecting_client(monkeypatch)
    for _ in range(3):
        client._post([{"role": "user", "content": "hi"}], model=client.model,
                     schema=None, max_tokens=100, purpose="decide")
    # One rejection, then three clean calls -- not three rejections.
    assert sum(1 for body in sent if "chat_template_kwargs" in body) == 1
    assert client.model in client._rejects_reasoning


def test_a_rejection_that_was_not_about_reasoning_still_raises(monkeypatch):
    """The drop-and-retry must not swallow a genuine bad request."""
    from adbagent.llm import LLMClient, LLMError
    from openai import APIStatusError
    import httpx

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    client = LLMClient(cfg_with(effort="none"))     # no reasoning field sent

    def create(**kwargs):
        raise APIStatusError(
            "messages: too many images",
            response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
            body=None)

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    with pytest.raises(LLMError, match="400"):
        client._post([{"role": "user", "content": "hi"}], model="llama-v3p3-70b",
                     schema=None, max_tokens=100, purpose="decide")


def test_a_second_rejection_after_dropping_is_not_hidden(monkeypatch):
    """If it fails again the reasoning field was never the problem."""
    from adbagent.llm import LLMError

    client, sent = _rejecting_client(monkeypatch, reject_times=1)

    # Make the *retry* fail too, for an unrelated reason.
    from openai import APIStatusError
    import httpx

    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        raise APIStatusError(
            "something else entirely",
            response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
            body=None)

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    with pytest.raises(LLMError):
        client._post([{"role": "user", "content": "hi"}], model=client.model,
                     schema=None, max_tokens=100, purpose="decide")


def test_the_drop_is_announced_rather_than_silent(monkeypatch):
    """Silently continuing would leave someone believing the depth was capped."""
    client, _ = _rejecting_client(monkeypatch)
    events = []
    client._post([{"role": "user", "content": "hi"}], model=client.model,
                 schema=None, max_tokens=100, purpose="decide",
                 on_event=lambda kind, **kw: events.append((kind, kw)))
    kinds = [kind for kind, _ in events]
    assert "reasoning_unsupported" in kinds
    payload = next(kw for kind, kw in events if kind == "reasoning_unsupported")
    assert payload["fields"] == ["chat_template_kwargs"]
    assert payload["model"] == client.model


# ---------------------------------------------------------------------------
# A 400 is only the reasoning field's fault when it says so
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "Extra inputs are not permitted, field: 'chat_template_kwargs'",
    "Unrecognized request argument supplied: reasoning_effort",
    "unknown field: thinking",
    "Additional properties are not allowed",
])
def test_a_400_naming_a_field_implicates_reasoning(message):
    from adbagent.llm import reasoning_implicated
    assert reasoning_implicated(message)


@pytest.mark.parametrize("message", [
    "This model does not support image inputs",
    "context length exceeded",
    "invalid base64 in image_url",
    "The model accounts/fireworks/models/nope does not exist",
    # The wording that would reintroduce the bug in a new provider's phrasing.
    # A bare "not allowed"/"not permitted" in the hint list matches this.
    "image inputs are not allowed for this model",
    "images are not permitted on this endpoint",
])
def test_a_400_naming_a_different_cause_does_not(message):
    from adbagent.llm import reasoning_implicated
    assert not reasoning_implicated(message)
    assert not reasoning_implicated(message, ["reasoning_effort"])


def test_a_field_we_did_not_send_is_not_ours_to_be_blamed_for():
    """The body says which reasoning fields went out, so a 400 about a different
    field cannot be pinned on one that was never in the request."""
    from adbagent.llm import reasoning_implicated
    # We sent `reasoning_effort`; the provider is complaining about the other one.
    assert not reasoning_implicated("chat_template_kwargs: bad value",
                                    ["reasoning_effort"])
    assert reasoning_implicated("chat_template_kwargs: bad value",
                                ["chat_template_kwargs"])


def test_the_thinking_key_counts_as_its_container():
    """Providers report the `chat_template_kwargs` convention by the key inside
    it as often as by the field itself."""
    from adbagent.llm import reasoning_implicated
    assert reasoning_implicated('"thinking" is not a valid argument',
                                ["chat_template_kwargs"])


def test_an_unrelated_400_is_not_blamed_on_the_reasoning_field(monkeypatch):
    """`model_image` pointed at a text-only model is how this was found.

    Fireworks answers the image part with "This model does not support image
    inputs". That was logged as "rejected reasoning_effort", retried once for
    nothing, and -- the part that outlived the call -- held against the model in
    `_rejects_reasoning`, so every later call silently lost its configured depth.
    """
    from adbagent.llm import LLMClient, LLMError
    from openai import APIStatusError
    import httpx

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = cfg_with(effort="none", hard="high")
    cfg.llm.model = "deepseek-v4-flash"          # does take reasoning_effort
    client = LLMClient(cfg)

    sent = []

    def create(**kwargs):
        sent.append(kwargs.get("extra_body") or {})
        raise APIStatusError(
            "This model does not support image inputs",
            response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
            body=None)

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    with pytest.raises(LLMError, match="does not support image inputs"):
        client._post([{"role": "user", "content": "hi"}], model=client.model,
                     schema=None, max_tokens=100, purpose="analyze_image")

    assert len(sent) == 1                        # no retry bought for nothing
    assert "reasoning_effort" in sent[0]         # the field was never the issue
    assert client.model not in client._rejects_reasoning


def test_a_real_rejection_is_still_dropped_and_remembered(monkeypatch):
    """The narrowing must not cost the behaviour it was narrowing."""
    client, sent = _rejecting_client(monkeypatch)
    raw, _ = client._post([{"role": "user", "content": "hi"}],
                          model=client.model, schema=None, max_tokens=100,
                          purpose="decide")
    assert raw == "ok"
    assert client.model in client._rejects_reasoning


# ---------------------------------------------------------------------------
# A vision pass that failed is not a vision pass that saw nothing
# ---------------------------------------------------------------------------

def test_an_analysis_is_available_by_default():
    assert not ScreenAnalysis().unavailable
    assert not ScreenAnalysis(reading="428 g").unavailable


def test_the_failure_flag_stays_out_of_the_schema_the_model_is_held_to():
    """A private attribute, so the model is never asked to fill it in."""
    schema = harden_schema(ScreenAnalysis)
    assert set(schema["properties"]) == {
        "reading", "item_label", "blocking_dialog", "notable"}
    assert "_failed" not in str(schema)


def test_a_swallowed_vision_failure_is_flagged_not_merely_logged(monkeypatch):
    """All four fields empty is a legitimate answer -- so it cannot be the signal.

    `needs_screenshot` only asks for an image when the tree cannot answer the
    question, and its note tells the decider to rely on the image. A failure that
    reads as "nothing to report" hands the decider that instruction anyway.
    """
    client = _client(monkeypatch)

    def post(*a, **kw):
        raise LLMError("400 from fireworks: This model does not support image inputs")

    monkeypatch.setattr(client, "_post", post)
    events = []
    analysis = client.analyze_image(b"jpeg", goal="read the weight",
                                    on_event=lambda kind, **kw: events.append((kind, kw)))

    assert analysis.unavailable
    assert analysis.render() == ""               # still nothing to show the model
    assert "vision_unavailable" in [kind for kind, _ in events]


def test_a_vision_pass_that_answers_is_not_flagged(monkeypatch):
    client = _client(monkeypatch)

    def post(messages, *, model, schema, max_tokens, purpose, **kw):
        return ('{"reading":"148.50","item_label":"","blocking_dialog":"",'
                '"notable":""}'), Call(model=model)

    monkeypatch.setattr(client, "_post", post)
    analysis = client.analyze_image(b"jpeg", goal="read the price")
    assert not analysis.unavailable
    assert analysis.reading == "148.50"


# ---------------------------------------------------------------------------
# "Nothing to report" is an answer, and must not be paid for twice
# ---------------------------------------------------------------------------

def test_an_analysis_that_found_nothing_is_not_recomputed(monkeypatch):
    """Seen against a real phone in ``runs/9b205cb055b4``: two `analyze_image`
    calls on one step, 1,663 prompt tokens each, both answering with four empty
    fields. The agent runs the vision pass itself and passes `render()` down --
    which is "" whenever the screen held no surprises, and "" read as "nobody
    looked". None means nobody looked; "" means somebody did."""
    client = _client(monkeypatch)
    seen = _stub_decide(monkeypatch, client)

    client.decide(goal="g", rendered="screen", history=[], width=720,
                  height=1600, screenshot=b"jpeg", image_analysis="")

    assert seen["analyses"] == 0
    assert "VISUAL SCREEN ANALYSIS" not in seen["messages"][-1]["content"][0]["text"]


def test_no_analysis_at_all_still_buys_one(monkeypatch):
    """The narrowing must not stop the callers that rely on `decide` doing the
    pass for them -- replay, a bare `decide`."""
    client = _client(monkeypatch)
    seen = _stub_decide(monkeypatch, client)
    client.decide(goal="g", rendered="screen", history=[], width=720,
                  height=1600, screenshot=b"jpeg")
    assert seen["analyses"] == 1


def test_the_judge_does_not_recompute_an_empty_analysis(monkeypatch):
    client = _client(monkeypatch)
    analyses = {"n": 0}

    def analyze_image(screenshot, **kw):
        analyses["n"] += 1
        return ScreenAnalysis()

    def structured(messages, model_cls, **kw):
        return model_cls(satisfied=True, evidence="ok")

    monkeypatch.setattr(client, "analyze_image", analyze_image)
    monkeypatch.setattr(client, "structured", structured)

    client.judge(goal="g", rendered="screen", history=[], screenshot=b"jpeg",
                 image_analysis="")
    assert analyses["n"] == 0

    client.judge(goal="g", rendered="screen", history=[], screenshot=b"jpeg")
    assert analyses["n"] == 1
