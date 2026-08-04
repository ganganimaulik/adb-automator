"""LLM layer parts that need no network: schema hardening, JSON extraction,
cost accounting, throttling, model-catalogue parsing.
"""

from __future__ import annotations

import time

import pytest

from adbagent.actions import AgentAction
from adbagent.llm import (Call, Ledger, LLMError, ModelInfo, PROVIDERS, RateLimiter,
                          ScreenAnalysis, extract_json, harden_schema, image_part,
                          qualify, text_part)


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
    last ten steps, is a verdict reached from the wrong evidence."""
    from adbagent import prompts

    history = [f"{i}. tap #{i} -> success" for i in range(1, 60)]
    judged = prompts.judge_user("g", history, "screen")
    decided = prompts.history_only_block(history)
    assert judged.count("-> success") > decided.count("-> success") * 3
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


def test_reading_one_item_asks_for_a_fact_not_a_screen_description(monkeypatch):
    client = _client(monkeypatch)
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
    assert captured["max_tokens"] <= 400                 # a line, not an essay
    assert "read the weight" in captured["messages"][1]["content"][0]["text"]
    assert "Today, 9:52 am" in captured["messages"][1]["content"][0]["text"]
    assert len(_image_parts(captured["messages"])) == 1


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


def test_the_vision_prompt_forbids_the_nav_bar_it_kept_describing():
    from adbagent import prompts
    assert "navigation bar" in prompts.IMAGE_ANALYSIS_SYSTEM
    assert "Never describe" in prompts.IMAGE_ANALYSIS_SYSTEM


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
    assert "BROWSING A GALLERY" not in prompts.SYSTEM
    assert "MULTI-APP NAVIGATION" not in prompts.SYSTEM
    # It kept the parts that apply on every single turn.
    assert "THE ACTIONS" in prompts.SYSTEM
    assert "SECURITY" in prompts.SYSTEM
    assert len(prompts.SYSTEM) < 7000          # was 9,722


def test_an_ordinary_turn_gets_no_situational_advice():
    from adbagent.prompts import situational_notes
    assert situational_notes(goal="turn on wifi") == ""


def test_a_pager_turn_gets_the_gallery_block_only():
    from adbagent.prompts import situational_notes
    note = situational_notes(goal="turn on wifi", is_pager=True)
    assert "BROWSING A GALLERY" in note
    assert "SCROLLING STRATEGY" not in note


def test_scrolling_advice_arrives_once_scrolling_starts():
    from adbagent.prompts import situational_notes
    assert "SCROLLING STRATEGY" not in situational_notes(goal="turn on wifi")
    assert "SCROLLING STRATEGY" in situational_notes(goal="turn on wifi", scrolls=1)


def test_a_search_goal_gets_scrolling_advice_before_the_first_scroll():
    """Waiting for the first scroll would withhold "start fast, slow down near the
    target" from the turn that chooses the first scroll's size."""
    from adbagent.prompts import situational_notes
    note = situational_notes(goal="find every message about the menu",
                             has_scroller=True)
    assert "SCROLLING STRATEGY" in note


def test_a_search_goal_on_an_unscrollable_screen_gets_nothing():
    from adbagent.prompts import situational_notes
    assert situational_notes(goal="find the wifi setting", has_scroller=False) == ""


def test_app_switching_advice_arrives_once_the_run_crosses_apps():
    from adbagent.prompts import situational_notes
    assert "SWITCHING APPS" not in situational_notes(goal="read my chats",
                                                    packages_seen=1)
    assert "SWITCHING APPS" in situational_notes(goal="read my chats",
                                                packages_seen=2)


def test_a_goal_that_says_it_will_switch_apps_gets_the_advice_up_front():
    from adbagent.prompts import situational_notes
    note = situational_notes(goal="copy the address and paste it into maps")
    assert "SWITCHING APPS" in note


def test_the_blocks_stack_when_they_all_apply():
    from adbagent.prompts import situational_notes
    note = situational_notes(goal="find and share every photo", is_pager=True,
                             scrolls=3, packages_seen=2)
    assert all(block in note for block in
               ("BROWSING A GALLERY", "SCROLLING STRATEGY", "SWITCHING APPS"))


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


def test_an_effort_family_floors_none_rather_than_inventing_a_level():
    """No family exposes an "off", so "none" becomes the lowest real setting.
    Sending `reasoning_effort: "none"` would be a 400."""
    from adbagent.llm import reasoning_body

    assert reasoning_body("gpt-oss-120b", "none") == {"reasoning_effort": "low"}


def test_a_hybrid_family_gets_a_thinking_switch():
    """These models have one switch, not a dial, so any real depth turns it on."""
    from adbagent.llm import reasoning_body

    off = {"chat_template_kwargs": {"thinking": False}}
    on = {"chat_template_kwargs": {"thinking": True}}
    assert reasoning_body("accounts/fireworks/models/deepseek-v4-flash", "none") == off
    assert reasoning_body("accounts/fireworks/models/deepseek-v4-flash", "high") == on
    assert reasoning_body("qwen3p7-plus", "low") == on


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
    assert body["chat_template_kwargs"] == {"thinking": False}
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
    assert seen["extra_body"]["chat_template_kwargs"] == {"thinking": False}


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
    assert reasoning_style_for("kimi-k3") == "thinking"


def test_the_reasoning_fields_are_named_so_dropping_them_takes_nothing_else():
    from adbagent.llm import reasoning_fields

    body = {"prompt_cache_key": "run-1", "service_tier": "priority",
            "chat_template_kwargs": {"thinking": False}}
    assert reasoning_fields(body) == ["chat_template_kwargs"]
    assert reasoning_fields({"reasoning_effort": "low"}) == ["reasoning_effort"]
    assert reasoning_fields({"prompt_cache_key": "x"}) == []
    assert reasoning_fields(None) == []


def _rejecting_client(monkeypatch, status=400, reject_times=1):
    """A client whose provider rejects any request carrying a reasoning field."""
    from adbagent.llm import LLMClient
    from openai import APIStatusError
    import httpx

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    cfg = cfg_with(effort="none", hard="high")
    cfg.llm.model = "deepseek-v4-flash"
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
        if ("chat_template_kwargs" in (kwargs.get("extra_body") or {})
                and state["rejections"] < reject_times):
            state["rejections"] += 1
            raise APIStatusError(
                "chat_template_kwargs is not supported for this model",
                response=httpx.Response(status, request=httpx.Request("POST", "http://x")),
                body=None)
        return iter([Chunk()])

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    return client, sent


def test_a_rejected_reasoning_field_is_dropped_rather_than_ending_the_run(monkeypatch):
    """A 400 ninety steps into a run, over an optimisation, is not acceptable."""
    client, sent = _rejecting_client(monkeypatch)

    raw, call = client._post([{"role": "user", "content": "hi"}],
                             model=client.model, schema=None, max_tokens=100,
                             purpose="decide")
    assert raw == "ok"
    assert len(sent) == 2                              # rejected, then retried
    assert "chat_template_kwargs" in sent[0]
    assert "chat_template_kwargs" not in sent[1]


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
