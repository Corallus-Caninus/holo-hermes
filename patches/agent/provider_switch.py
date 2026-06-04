"""Runtime provider switch for Hermes Agent.

Monkey-patches the OpenAI client's chat.completions.create to check
a switch file before each API call. When the launcher's probe thread
detects a recovered higher-priority provider, it writes to this file
and Hermes seamlessly swaps model/provider/base_url mid-session.

Also injects reasoning_effort into every API call via extra_body,
controlled by the REASONING_EFFORT env var (default: "max").
Set to "low" for fast agent tasks, or "high"/"xhigh" for deep reasoning.

Usage: call install() after agent.client is initialized.
"""

import json
import logging
import os
from pathlib import Path

_log = logging.getLogger("provider_switch")

SWITCH_FILE = Path.home() / ".hermes" / "apply-provider-switch.json"

# Provider → (api_mode, base_url) mapping
PROVIDER_CONFIG = {
    "openrouter": {
        "api_mode": "chat_completions",
        "base_url": None,  # Hermes resolves from --provider flag
        "needs_reinit": False,
    },
    "opencode-zen": {
        "api_mode": "chat_completions",
        "base_url": "https://opencode.ai/zen/v1",
        "needs_reinit": False,
    },
    "opencode-go": {
        "api_mode": "chat_completions",
        "base_url": "https://opencode.ai/zen/go/v1",
        "needs_reinit": False,
    },
    "gemini": {
        "api_mode": "chat_completions",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "needs_reinit": False,
    },
}


def _resolve_provider_config(provider: str, model: str) -> dict:
    """Resolve the full provider config for a given provider name and model."""
    cfg = dict(PROVIDER_CONFIG.get(provider, {}))

    if provider == "openrouter":
        cfg["base_url"] = "https://openrouter.ai/api/v1"
    elif provider in ("opencode", "opencode-zen"):
        cfg["base_url"] = "https://opencode.ai/zen/v1"
    elif provider == "opencode-go":
        cfg["base_url"] = "https://opencode.ai/zen/go/v1"
    elif provider == "gemini":
        cfg["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai"

    # Model mapping
    cfg["model"] = model
    cfg["provider"] = provider
    return cfg


def _read_switch_request() -> dict | None:
    """Read and remove the switch file. Returns the switch payload or None."""
    if not SWITCH_FILE.exists():
        return None
    try:
        payload = json.loads(SWITCH_FILE.read_text(encoding="utf-8"))
        SWITCH_FILE.unlink(missing_ok=True)
        return payload
    except (json.JSONDecodeError, OSError) as e:
        _log.warning("Failed to read provider switch file: %s", e)
        try:
            SWITCH_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _apply_switch(payload: dict, agent_module) -> bool:
    """Apply a provider switch to the running agent module.

    Updates agent.model, agent.provider, agent.base_url, agent.api_mode
    and reconfigures the OpenAI client in-place.

    Returns True if the switch was applied successfully.
    """
    import openai

    provider = payload.get("provider", "")
    model = payload.get("model", "")

    if not provider or not model:
        _log.warning("Invalid switch payload: missing provider or model")
        return False

    cfg = _resolve_provider_config(provider, model)

    # Update module-level attributes
    old_provider = getattr(agent_module, "provider", "")
    old_model = getattr(agent_module, "model", "")

    agent_module.model = model
    agent_module.provider = provider
    agent_module.base_url = cfg["base_url"]
    agent_module.api_mode = cfg["api_mode"]

    # Update the OpenAI client in-place
    client = getattr(agent_module, "client", None)
    if client is not None:
        client.base_url = cfg["base_url"]
        # Update API key if needed — load from env or switch payload
        new_key = payload.get("api_key", "")
        if not new_key:
            if provider in ("opencode", "opencode-zen", "opencode-go"):
                new_key = os.environ.get("OPENCODE_API_KEY", "")
            elif provider == "openrouter":
                new_key = os.environ.get("OPENROUTER_API_KEY", "")
            elif provider == "gemini":
                new_key = os.environ.get("GEMINI_API_KEY", "")
        if new_key:
            client.api_key = new_key
            agent_module.api_key = new_key

    # Restart the credential pool if applicable (for OpenRouter routing)
    pool = getattr(agent_module, "_credential_pool", None)
    if pool is not None:
        try:
            pool.clear()
        except Exception:
            pass

    # Update environment so subprocess calls see the new provider
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    if cfg.get("base_url"):
        os.environ["LLM_URL"] = cfg["base_url"]

    _log.info(
        "Provider switch: %s/%s -> %s/%s (base_url=%s)",
        old_provider, old_model, provider, model, cfg["base_url"],
    )
    return True


def _wrap_create(original_create, agent_module):
    """Wrap chat.completions.create to check for provider switches
    and inject reasoning_effort before each call."""
    _reasoning_effort = os.environ.get("REASONING_EFFORT", "max")

    def _switched_create(*args, **kwargs):
        payload = _read_switch_request()
        if payload is not None:
            _apply_switch(payload, agent_module)
        # Inject reasoning_effort via extra_body
        if _reasoning_effort:
            extra = kwargs.pop("extra_body", {}) or {}
            extra["reasoning_effort"] = _reasoning_effort
            kwargs["extra_body"] = extra
        return original_create(*args, **kwargs)
    return _switched_create


def install(agent_module):
    """Install the provider switch hook into the agent's OpenAI client.

    Call this after agent.client has been initialized.
    """
    client = getattr(agent_module, "client", None)
    if client is None:
        _log.warning("agent.client not initialized yet — cannot install switch hook")
        return False

    try:
        original = client.chat.completions.create
        client.chat.completions.create = _wrap_create(original, agent_module)
        _log.info("Provider switch hook installed (checks %s before each API call)", SWITCH_FILE)
        return True
    except AttributeError as e:
        _log.warning("Failed to install switch hook: %s", e)
        return False
