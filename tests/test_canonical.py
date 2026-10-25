import pytest

from src.cache.canonical import (
    InvalidRequestError,
    bypass_reason,
    canonicalize,
    tenant_id,
)


def req(**overrides):
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "How do I reset my password?"},
        ],
    }
    body.update(overrides)
    return body


# --- Eligibility filter (ADR-0004 §4.2) ---


def test_plain_request_is_eligible():
    assert bypass_reason(req(), allow_tools=True) is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"n": 2}, "n>1"),
        ({"logprobs": True}, "logprobs"),
        ({"modalities": ["text", "audio"]}, "audio"),
        ({"audio": {"voice": "alloy", "format": "wav"}}, "audio"),
    ],
)
def test_bypass_reasons(overrides, reason):
    assert bypass_reason(req(**overrides), allow_tools=True) == reason


def test_tools_bypass_only_when_disallowed():
    body = req(tools=[{"type": "function", "function": {"name": "f"}}])
    assert bypass_reason(body, allow_tools=True) is None
    assert bypass_reason(body, allow_tools=False) == "tools"


@pytest.mark.parametrize("part_type", ["image_url", "input_audio", "file", "some_future_type"])
def test_non_text_content_bypasses(part_type):
    body = req(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is this?"}, {"type": part_type}],
            }
        ]
    )
    assert bypass_reason(body, allow_tools=True) == "non-text-content"


def test_text_parts_do_not_bypass():
    body = req(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert bypass_reason(body, allow_tools=True) is None


# --- Canonicalization (ADR-0004 §4.3) ---


def test_omitted_and_explicit_defaults_hash_identically():
    explicit = req(temperature=1, top_p=1.0, n=1, presence_penalty=0, frequency_penalty=0.0)
    assert canonicalize(req()).exact_hash == canonicalize(explicit).exact_hash


@pytest.mark.parametrize(
    "field",
    [
        "stream",
        "stream_options",
        "user",
        "metadata",
        "store",
        "max_tokens",
        "max_completion_tokens",
    ],
)
def test_denylisted_fields_do_not_change_hash(field):
    values = {
        "stream": True,
        "stream_options": {"include_usage": True},
        "user": "u-123",
        "metadata": {"trace": "abc"},
        "store": True,
        "max_tokens": 50,
        "max_completion_tokens": 50,
    }
    assert canonicalize(req()).exact_hash == canonicalize(req(**{field: values[field]})).exact_hash


def test_temperature_and_top_p_are_rounded():
    a = canonicalize(req(temperature=0.701, top_p=0.904))
    b = canonicalize(req(temperature=0.7, top_p=0.9))
    assert a.exact_hash == b.exact_hash


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0.5},
        {"stop": ["\n"]},
        {"response_format": {"type": "json_object"}},
        {"reasoning_effort": "high"},
        {"model": "gpt-4o"},
        {"tools": [{"type": "function", "function": {"name": "f"}}]},
    ],
)
def test_answer_changing_fields_change_hash(overrides):
    assert canonicalize(req()).exact_hash != canonicalize(req(**overrides)).exact_hash


def test_unknown_field_enters_hash_and_is_reported():
    """Fail-closed: a parameter released after this code was written must cause a miss."""
    result = canonicalize(req(brand_new_param=True))
    assert result.exact_hash != canonicalize(req()).exact_hash
    assert result.param_fp != canonicalize(req()).param_fp
    assert result.unknown_fields == ["brand_new_param"]


def test_seed_ignored_only_for_greedy_decoding():
    greedy = canonicalize(req(temperature=0, seed=1)).exact_hash
    assert greedy == canonicalize(req(temperature=0, seed=2)).exact_hash
    assert greedy == canonicalize(req(temperature=0)).exact_hash

    sampled = canonicalize(req(temperature=0.7, seed=1)).exact_hash
    assert sampled != canonicalize(req(temperature=0.7, seed=2)).exact_hash


def test_message_text_is_not_normalized():
    a = canonicalize(req(messages=[{"role": "user", "content": "Hello"}]))
    b = canonicalize(req(messages=[{"role": "user", "content": "hello "}]))
    assert a.exact_hash != b.exact_hash


def test_key_order_does_not_matter():
    a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.2}
    b = {"temperature": 0.2, "messages": [{"content": "x", "role": "user"}], "model": "m"}
    assert canonicalize(a).exact_hash == canonicalize(b).exact_hash


# --- Layer 2 scope and embedded text (ADR-0004 §4.3, §4.6) ---


def test_paraphrase_shares_scope_but_not_exact_hash():
    a = canonicalize(req())
    b = canonicalize(
        req(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Forgot my password, how do I recover access?"},
            ]
        )
    )
    assert a.exact_hash != b.exact_hash
    assert a.scope == b.scope


def test_different_context_changes_scope():
    follow_up = {"role": "user", "content": "Yes, go ahead"}
    a = canonicalize(
        req(
            messages=[
                {"role": "user", "content": "Delete my account?"},
                {"role": "assistant", "content": "Are you sure?"},
                follow_up,
            ]
        )
    )
    b = canonicalize(
        req(
            messages=[
                {"role": "user", "content": "Book a table for two?"},
                {"role": "assistant", "content": "Shall I confirm?"},
                follow_up,
            ]
        )
    )
    assert a.last_user_text == b.last_user_text == "Yes, go ahead"
    assert a.scope != b.scope


def test_different_params_change_scope():
    assert canonicalize(req()).scope != canonicalize(req(temperature=0.2)).scope


def test_last_user_text_joins_text_parts():
    body = req(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "line one"},
                    {"type": "text", "text": "line two"},
                ],
            }
        ]
    )
    assert canonicalize(body).last_user_text == "line one\nline two"


def test_no_layer2_text_when_last_message_is_not_user():
    body = req(
        messages=[
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "tool_calls": [{"id": "1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "1", "content": "22C"},
        ]
    )
    assert canonicalize(body).last_user_text is None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, None),
        ({"max_tokens": 100}, 100),
        ({"max_completion_tokens": 200}, 200),
        ({"max_tokens": 100, "max_completion_tokens": 200}, 200),
    ],
)
def test_max_tokens_extraction(overrides, expected):
    assert canonicalize(req(**overrides)).max_tokens == expected


# --- Invalid input and tenant ---


@pytest.mark.parametrize(
    "body",
    [
        {"model": "m"},
        {"model": "m", "messages": []},
        {"model": "m", "messages": ["not an object"]},
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": "hot"},
    ],
)
def test_invalid_requests_raise(body):
    with pytest.raises(InvalidRequestError):
        canonicalize(body)


def test_tenant_id_is_stable_short_and_not_the_key():
    tid = tenant_id("sk-secret")
    assert tid == tenant_id("sk-secret")
    assert len(tid) == 16
    assert "secret" not in tid
    assert tid != tenant_id("sk-other")
