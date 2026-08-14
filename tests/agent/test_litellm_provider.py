"""Behavior contracts for the first-class LiteLLM provider.

LiteLLM is a multiplexer: one /v1 URL fronts many model families (Claude, GPT,
Gemini, Qwen, local). The provider's whole reason for existing is that the wire
format — and therefore prompt-cache eligibility — is a property of the TARGET
MODEL, not the endpoint. These tests pin that invariant end to end:

  * ``_litellm_model_api_mode`` routes Claude → anthropic_messages and every
    other family → chat_completions, honoring an explicit model.api_mode
    override as an escape hatch.
  * The registry entry exists and is an api_key provider keyed on
    LITELLM_API_KEY.
  * ``anthropic_prompt_cache_policy`` grants native-layout caching for a Claude
    model on the Anthropic wire, denies it for a GPT model on the OpenAI wire,
    and still fires the PR #84982 hostname/provider heuristic for a Claude model
    left on the OpenAI wire.
  * The real resolution path (``resolve_runtime_provider`` against a temp
    HERMES_HOME, no api_mode in the provider block — the old config trap)
    produces the correct per-model api_mode.
  * Base-URL normalization strips /v1 only for the Anthropic-wire (Claude)
    routing and leaves it intact for chat_completions.

These are contracts about how the pieces must relate, not snapshots of current
values, per AGENTS.md. No live network; config is injected; nothing touches the
real ~/.hermes.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli import runtime_provider as rp
from agent.agent_runtime_helpers import anthropic_prompt_cache_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakePoolEntry:
    """Minimal stand-in for a PooledCredential.

    ``_resolve_runtime_from_pool_entry`` only reads ``runtime_base_url`` /
    ``base_url`` (for the URL), ``runtime_api_key`` / ``access_token`` (for the
    key), and ``source``. A tiny object keeps the pool-path test free of the
    whole credential-pool machinery.
    """

    def __init__(self, base_url: str, api_key: str = "sk-litellm-test"):
        self.runtime_base_url = base_url
        self.base_url = base_url
        self.runtime_api_key = api_key
        self.access_token = api_key
        self.source = "pool"


def _resolve_pool_entry(*, model: str, base_url: str, model_cfg: dict) -> dict:
    """Drive the pool-entry resolver, which owns the primary litellm branch."""
    return rp._resolve_runtime_from_pool_entry(
        provider="litellm",
        entry=_FakePoolEntry(base_url),
        requested_provider="litellm",
        model_cfg=model_cfg,
        target_model=model,
    )


def _stub_agent():
    """anthropic_prompt_cache_policy reads only agent._cache_disabled when the
    provider/base_url/api_mode/model are all passed as kwargs."""
    return types.SimpleNamespace(_cache_disabled=False)


# ---------------------------------------------------------------------------
# 1. api_mode routing (family -> wire)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-sonnet-4-5", "anthropic_messages"),
        ("anthropic/claude-3-7-sonnet", "anthropic_messages"),
        ("gpt-4o", "chat_completions"),
        ("qwen3-max", "chat_completions"),
        ("gemini-2.5-pro", "chat_completions"),
    ],
)
def test_api_mode_routes_by_model_family(model, expected):
    """The wire is chosen from the TARGET model family, not the endpoint:
    Claude → native Anthropic wire, everything else → OpenAI wire."""
    assert rp._litellm_model_api_mode(model, {}) == expected


def test_explicit_api_mode_override_is_honored():
    """An explicit model.api_mode is the escape hatch and wins over family
    inference — including forcing a non-Claude model onto the Anthropic wire."""
    # Override wins even against the family default it contradicts.
    assert (
        rp._litellm_model_api_mode("gpt-4o", {"api_mode": "anthropic_messages"})
        == "anthropic_messages"
    )
    # A garbage api_mode is ignored (not a valid mode) and inference resumes.
    assert (
        rp._litellm_model_api_mode("claude-sonnet-4-5", {"api_mode": "nonsense"})
        == "anthropic_messages"
    )


# ---------------------------------------------------------------------------
# 2. Cache policy interaction
# ---------------------------------------------------------------------------

def test_cache_policy_grants_native_layout_for_claude_on_anthropic_wire():
    """provider=litellm + anthropic_messages + Claude -> (should_cache, native)
    both True, so the gateway gets the same cost reduction as native Anthropic."""
    should_cache, native = anthropic_prompt_cache_policy(
        _stub_agent(),
        provider="litellm",
        base_url="https://llm.example.com",
        api_mode="anthropic_messages",
        model="claude-sonnet-4-5",
    )
    assert (should_cache, native) == (True, True)


def test_cache_policy_denies_caching_for_gpt_on_openai_wire():
    """A non-Claude family on the OpenAI wire must NOT receive cache_control
    markers (no measured benefit, wrong wire shape) -> (False, False)."""
    should_cache, native = anthropic_prompt_cache_policy(
        _stub_agent(),
        provider="litellm",
        base_url="https://llm.example.com/v1",
        api_mode="chat_completions",
        model="gpt-4o",
    )
    assert (should_cache, native) == (False, False)


def test_cache_policy_pr84982_heuristic_fires_for_litellm_claude_on_openai_wire():
    """Belt-and-suspenders: even if a Claude model is left on the OpenAI wire
    (api_mode=chat_completions), the PR #84982 heuristic still engages caching
    because the provider string / hostname says litellm."""
    should_cache, native = anthropic_prompt_cache_policy(
        _stub_agent(),
        provider="litellm",
        base_url="https://llm.example.com/v1",
        api_mode="chat_completions",
        model="claude-sonnet-4-5",
    )
    assert (should_cache, native) == (True, True)


# ---------------------------------------------------------------------------
# 3. Registry presence
# ---------------------------------------------------------------------------

def test_registry_entry_is_an_api_key_provider_keyed_on_litellm_api_key():
    assert "litellm" in PROVIDER_REGISTRY
    entry = PROVIDER_REGISTRY["litellm"]
    assert entry.auth_type == "api_key"
    assert "LITELLM_API_KEY" in entry.api_key_env_vars
    # No baked-in default: setup must prompt for a URL rather than silently use
    # a wrong one (LiteLLM is always self/team hosted).
    assert entry.inference_base_url == ""


# ---------------------------------------------------------------------------
# 4 + 5. Runtime resolution E2E (no api_mode in the provider block)
# ---------------------------------------------------------------------------

def _litellm_config(default_model: str) -> dict:
    """A first-class litellm config WITHOUT any api_mode anywhere — the exact
    shape that was the old 'config trap' for bare custom providers."""
    return {
        "model": {
            "provider": "litellm",
            "base_url": "https://llm.example.com/v1",
            "default": default_model,
            # api_mode intentionally absent — resolution must derive it.
        },
        "providers": {
            "litellm": {
                "base_url": "https://llm.example.com/v1",
                "key_env": "LITELLM_API_KEY",
                # api_mode intentionally absent (the trap): must not be required.
            }
        },
    }


def _resolve_via_public_api(cfg: dict, monkeypatch, *, target_model: str) -> dict:
    """Exercise the real resolve_runtime_provider path against injected config
    and a temp env, with no credential pool (the common env-credential setup)."""
    monkeypatch.setenv("LITELLM_API_KEY", "sk-litellm-e2e-key")
    # No credential pool -> resolution falls to the generic api_key path, which
    # must apply the same per-model litellm routing as the pool path.
    with (
        patch.object(rp, "load_config", return_value=cfg),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch.object(rp, "load_pool", return_value=None),
    ):
        return rp.resolve_runtime_provider(
            requested="litellm", target_model=target_model
        )


def test_resolution_derives_anthropic_wire_for_claude_without_configured_api_mode(monkeypatch):
    """Config trap fixed: with NO api_mode set anywhere, a Claude default still
    resolves to the Anthropic wire through the public resolver."""
    cfg = _litellm_config("claude-sonnet-4-5")
    runtime = _resolve_via_public_api(cfg, monkeypatch, target_model="claude-sonnet-4-5")
    assert runtime["provider"] == "litellm"
    assert runtime["api_mode"] == "anthropic_messages"
    # base_url comes from the providers.litellm block (a provider setting in
    # config.yaml, not the LITELLM_BASE_URL env var), with /v1 stripped for the
    # Anthropic SDK. An empty base_url here is the 401 bug this fix closes.
    assert runtime["base_url"] == "https://llm.example.com"


def test_resolution_derives_openai_wire_for_gpt_without_configured_api_mode(monkeypatch):
    """Same config, a GPT target model -> OpenAI wire. Proves the routing is
    per-model, not a single endpoint-wide override."""
    cfg = _litellm_config("claude-sonnet-4-5")  # default is Claude...
    runtime = _resolve_via_public_api(cfg, monkeypatch, target_model="gpt-4o")
    # ...but the switched-to target model drives the mode.
    assert runtime["api_mode"] == "chat_completions"
    # chat_completions keeps /v1 — the OpenAI-compatible surface needs it.
    assert runtime["base_url"] == "https://llm.example.com/v1"


# ---------------------------------------------------------------------------
# 7. Base URL sourced from the providers.litellm block (reserved-name shadowing)
# ---------------------------------------------------------------------------

def test_base_url_resolves_from_providers_block_without_env_or_model_base_url(monkeypatch):
    """The canonical home for the proxy URL is the providers.litellm block in
    config.yaml. A config with the block but NO model.base_url and NO
    LITELLM_BASE_URL must still resolve the URL — the reserved provider name
    ``litellm`` shadows _get_named_custom_provider, so the block would be
    silently dropped without the explicit read this test pins."""
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    cfg = {
        "model": {"provider": "litellm", "default": "claude-sonnet-4-5"},
        "providers": {"litellm": {"base_url": "https://proxy.internal/v1"}},
    }
    runtime = _resolve_via_public_api(cfg, monkeypatch, target_model="claude-sonnet-4-5")
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["base_url"] == "https://proxy.internal"


def test_providers_block_base_url_wins_over_litellm_base_url_env(monkeypatch):
    """Resolution order: the providers.litellm block (config.yaml) is canonical
    and beats the LITELLM_BASE_URL env var (back-compat only)."""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://stale-env.example/v1")
    cfg = {
        "model": {"provider": "litellm", "default": "gpt-4o"},
        "providers": {"litellm": {"base_url": "https://canonical.example/v1"}},
    }
    runtime = _resolve_via_public_api(cfg, monkeypatch, target_model="gpt-4o")
    # chat_completions keeps /v1; the block URL wins over the env URL.
    assert runtime["base_url"] == "https://canonical.example/v1"


def test_litellm_base_url_env_used_as_backcompat_fallback(monkeypatch):
    """When no providers.litellm block and no model.base_url exist, the
    LITELLM_BASE_URL env var is still honored (back-compat for the original
    community setup)."""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://env-only.example/v1")
    cfg = {"model": {"provider": "litellm", "default": "gpt-4o"}}
    runtime = _resolve_via_public_api(cfg, monkeypatch, target_model="gpt-4o")
    assert runtime["base_url"] == "https://env-only.example/v1"


def test_disabled_providers_block_is_rejected(monkeypatch):
    """A providers.litellm block with enabled:false must NOT resolve to a
    runtime — the resolver raises so the fallback chain advances to the next
    provider instead of silently using a disabled one. (The disable guard fires
    at the top of resolution, before base_url is even considered.)"""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://fallback.example/v1")
    cfg = {
        "model": {"provider": "litellm", "default": "gpt-4o"},
        "providers": {
            "litellm": {"base_url": "https://disabled.example/v1", "enabled": False}
        },
    }
    with pytest.raises(ValueError, match="disabled"):
        _resolve_via_public_api(cfg, monkeypatch, target_model="gpt-4o")


# ---------------------------------------------------------------------------
# 6. Base URL normalization
# ---------------------------------------------------------------------------

def test_base_url_v1_stripped_only_for_anthropic_wire():
    """The Anthropic SDK appends its own /v1/messages, so a trailing /v1 must be
    stripped for Claude routing — and left intact for chat_completions."""
    model_cfg = {"provider": "litellm", "base_url": "https://llm.example.com/v1"}

    claude = _resolve_pool_entry(
        model="claude-sonnet-4-5",
        base_url="https://llm.example.com/v1",
        model_cfg=model_cfg,
    )
    assert claude["api_mode"] == "anthropic_messages"
    assert claude["base_url"] == "https://llm.example.com"

    gpt = _resolve_pool_entry(
        model="gpt-4o",
        base_url="https://llm.example.com/v1",
        model_cfg=model_cfg,
    )
    assert gpt["api_mode"] == "chat_completions"
    assert gpt["base_url"] == "https://llm.example.com/v1"
