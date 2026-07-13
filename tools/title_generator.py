"""Patched title_generator — checks HERMES_DISABLE_TITLE_GENERATION env var.

When this env var is set to 1/true/yes, maybe_auto_title becomes a no-op.
The applypilot launcher sets this to prevent wasted API calls on
one-shot job application sessions.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

# ruff: noqa: F401, F403 — intentional re-export

# ---------------------------------------------------------------------------
# 1. Find and load the real title_generator module from agent.__path__
#    (skipping our patches directory to avoid circular resolution).
# ---------------------------------------------------------------------------
import agent  # noqa: E402

_real_path: Path | None = None
for _p in agent.__path__:
    if "patches" in str(_p):
        continue
    _candidate = Path(_p) / "title_generator.py"
    if _candidate.exists():
        _real_path = _candidate
        break

if _real_path is None:
    msg = "Could not locate real agent/title_generator.py in agent.__path__"
    raise ImportError(msg)

_spec = importlib.util.spec_from_file_location(
    "agent.title_generator_real", str(_real_path)
)
_real_mod = importlib.util.module_from_spec(_spec)
sys.modules["agent.title_generator_real"] = _real_mod
_spec.loader.exec_module(_real_mod)

# ---------------------------------------------------------------------------
# 2. Re-export everything from the real module into our namespace.
# ---------------------------------------------------------------------------
for _attr in dir(_real_mod):
    if _attr.startswith("_") and _attr not in ("__all__",):
        continue
    globals()[_attr] = getattr(_real_mod, _attr)

# ---------------------------------------------------------------------------
# 3. Get reference to the real maybe_auto_title
# ---------------------------------------------------------------------------
_real_maybe_auto_title = getattr(_real_mod, "maybe_auto_title", None)


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    failure_callback: Optional = None,
    main_runtime: dict = None,
    title_callback: Optional = None,
) -> None:
    """Fire-and-forget title generation after the first exchange.

    Checks HERMES_DISABLE_TITLE_GENERATION before firing. When set,
    silently returns without spawning a thread (saves API calls in
    one-shot applypilot sessions).
    """
    import logging
    _logger = logging.getLogger(__name__)

    _disabled = os.environ.get("HERMES_DISABLE_TITLE_GENERATION", "").lower()
    if _disabled in ("1", "true", "yes"):
        _logger.debug("Title generation disabled via HERMES_DISABLE_TITLE_GENERATION — skipping")
        return

    if _real_maybe_auto_title is None:
        return

    _real_maybe_auto_title(
        session_db,
        session_id,
        user_message,
        assistant_response,
        conversation_history,
        failure_callback=failure_callback,
        main_runtime=main_runtime,
        title_callback=title_callback,
    )


# Export the same names as the real module
__all__ = [n for n in dir() if not n.startswith("_") and n not in ("_real_mod", "_real_path", "_spec")]
