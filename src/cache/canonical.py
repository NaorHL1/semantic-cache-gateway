"""Request eligibility, canonicalization and cache hashes (ADR-0004 §4.2-4.4).

The fingerprint is fail-closed: every request field enters the hash except the
fields on EXCLUSION_DENYLIST, so an unknown or newly released parameter causes
a cache miss, never a wrong hit. Changing anything in this module changes what
the hashes mean and requires bumping CACHE_SCHEMA_VERSION.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Provider defaults, filled in so an omitted field and its explicit default hash identically.
# Values are typed so that `0` and `0.0` (or `1` and `1.0`) serialize the same way.
PROVIDER_DEFAULTS: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}
FLOAT_FIELDS = frozenset({"temperature", "top_p", "presence_penalty", "frequency_penalty"})
ROUNDED_FIELDS = frozenset({"temperature", "top_p"})

# Fields known not to change the generated answer. Everything else enters the hash.
EXCLUSION_DENYLIST = frozenset(
    {
        "stream",
        "stream_options",
        "user",
        "metadata",
        "store",
        "max_tokens",
        "max_completion_tokens",
    }
)

# Chat Completions fields we recognize. Fields outside this set still enter the hash;
# they are only logged, so harmless ones can be added to the denylist deliberately.
KNOWN_FIELDS = (
    EXCLUSION_DENYLIST
    | PROVIDER_DEFAULTS.keys()
    | {
        "model",
        "messages",
        "seed",
        "stop",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "reasoning_effort",
        "modalities",
        "audio",
        "prediction",
        "service_tier",
        "verbosity",
        "web_search_options",
    }
)

# Content part types that carry only text. Any other part (image_url, input_audio,
# file, or a type released later) bypasses the cache.
TEXT_PART_TYPES = frozenset({"text", "refusal"})


class InvalidRequestError(ValueError):
    """The body is not a Chat Completions request the cache can reason about."""


@dataclass(frozen=True)
class CanonicalRequest:
    exact_hash: str
    context_hash: str
    param_fp: str
    scope: str
    # Text embedded by Layer 2; None when the last message is not a user text message.
    last_user_text: str | None
    # Read-time limit: a cached entry is served only if completion_tokens <= max_tokens.
    max_tokens: int | None
    unknown_fields: list[str] = field(default_factory=list)


def tenant_id(api_key: str) -> str:
    """Tenant from the caller's API key (ADR-0004 §4.4). Raw keys are never stored."""
    return _sha256(api_key)[:16]


def bypass_reason(body: dict[str, Any], allow_tools: bool) -> str | None:
    """Return why the request must skip both cache layers, or None (ADR-0004 §4.2)."""
    if body.get("n", 1) != 1:
        return "n>1"
    if body.get("logprobs"):
        return "logprobs"
    if "audio" in body or "audio" in (body.get("modalities") or []):
        return "audio"
    if not allow_tools and ("tools" in body or "tool_choice" in body):
        return "tools"
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in TEXT_PART_TYPES:
                    return "non-text-content"
    return None


def canonicalize(body: dict[str, Any]) -> CanonicalRequest:
    """Build the canonical form of a request and derive the cache hashes (ADR-0004 §4.3)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError("messages must be a non-empty list")
    if not all(isinstance(m, dict) for m in messages):
        raise InvalidRequestError("every message must be an object")

    canonical: dict[str, Any] = {k: v for k, v in body.items() if k not in EXCLUSION_DENYLIST}
    for key, default in PROVIDER_DEFAULTS.items():
        canonical.setdefault(key, default)
    for key in FLOAT_FIELDS:
        value = canonical[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise InvalidRequestError(f"{key} must be a number")
        canonical[key] = round(float(value), 2) if key in ROUNDED_FIELDS else float(value)
    # Greedy decoding ignores the seed, so it must not split otherwise identical entries.
    if canonical["temperature"] == 0.0:
        canonical.pop("seed", None)

    unknown = sorted(k for k in body if k not in KNOWN_FIELDS)
    if unknown:
        logger.info("unknown request fields kept in fingerprint: %s", unknown)

    last_user = _last_user_index(messages)
    context = messages[:last_user] if last_user is not None else messages
    params = {k: v for k, v in canonical.items() if k != "messages"}

    exact_hash = _sha256(_serialize(canonical))
    context_hash = _sha256(_serialize(context))
    param_fp = _sha256(_serialize(params))

    return CanonicalRequest(
        exact_hash=exact_hash,
        context_hash=context_hash,
        param_fp=param_fp,
        scope=_sha256(f"{context_hash}|{param_fp}"),
        last_user_text=_last_user_text(messages),
        max_tokens=_max_tokens(body),
        unknown_fields=unknown,
    )


def _serialize(value: Any) -> str:
    # Sorted keys and compact separators; message text itself is never altered.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _last_user_index(messages: list[dict[str, Any]]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def _last_user_text(messages: list[dict[str, Any]]) -> str | None:
    """Text of the final message if it is a user text message, else None (ADR-0004 §4.6)."""
    last = messages[-1]
    if last.get("role") != "user":
        return None
    content = last.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        for p in content
    ):
        return "\n".join(p["text"] for p in content)
    return None


def _max_tokens(body: dict[str, Any]) -> int | None:
    # max_completion_tokens supersedes the deprecated max_tokens in the OpenAI API.
    for key in ("max_completion_tokens", "max_tokens"):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None
