"""Patched context compressor — updates MEMORY.md/USER.md references to
reflect the switch to holographic memory store, overrides
MINIMUM_CONTEXT_LENGTH so the threshold actually works instead of
being floored at 64K, and allows zero-message tail (only system prompt
+ LLM summary survive compression, eliminating post-compression cold
re-prefill).

Loads everything from the real Nix-store context_compressor.py and patches
``SUMMARY_PREFIX`` and the inline compression note to reference the memory
provider instead of MEMORY.md/USER.md.

Additions vs the nix-store original:
  - MINIMUM_CONTEXT_LENGTH lowered from 64K → 16K so compression fires
    at the configured threshold instead of being floored.
  - min_tail in _find_tail_cut_by_tokens lowered from 3 → 0 so the tail
    can be empty after compression — only system prompt + LLM summary
    survive, eliminating the expensive post-compression cold re-prefill.
  - LLM summarization still runs (it produces a useful summary that
    preserves task context).
"""

# ruff: noqa: F401, F403 — intentional re-export

import importlib.util
import inspect
import sys
from pathlib import Path

import agent  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Load the real context_compressor module.
# ---------------------------------------------------------------------------
_real_path: Path | None = None
for _p in agent.__path__:
    if "patches" in str(_p):
        continue
    _candidate = Path(_p) / "context_compressor.py"
    if _candidate.exists():
        _real_path = _candidate
        break

if _real_path is None:
    raise ImportError("Could not locate real agent/context_compressor.py")

_spec = importlib.util.spec_from_file_location(
    "agent.context_compressor_real", str(_real_path)
)
_real_mod = importlib.util.module_from_spec(_spec)
sys.modules["agent.context_compressor_real"] = _real_mod
_spec.loader.exec_module(_real_mod)

# Re-export all names.
for _attr in dir(_real_mod):
    if _attr.startswith("_") and _attr not in ("__all__",):
        continue
    globals()[_attr] = getattr(_real_mod, _attr)

# ---------------------------------------------------------------------------
# 3. Override MINIMUM_CONTEXT_LENGTH so compression fires below 64K.
#    The nix-store constant is 64000, which floors the threshold at 64K and
#    makes our ApplyPilot threshold=0.30 (targeting ~29K at 98K context)
#    ineffective.  Setting to 16000 allows 30% of 98K = ~29K to work.
# ---------------------------------------------------------------------------
_OVERRIDE_MIN_CTX = 16_000
# The real module already loaded with MINIMUM_CONTEXT_LENGTH=64000 in its
# namespace.  Python functions resolve global variables at call time via
# the module's __dict__, so changing it now affects new instantiations.
setattr(_real_mod, "MINIMUM_CONTEXT_LENGTH", _OVERRIDE_MIN_CTX)
setattr(globals().get("_agent_md", None) or __import__("agent.model_metadata"),
        "MINIMUM_CONTEXT_LENGTH", _OVERRIDE_MIN_CTX)

# ---------------------------------------------------------------------------
# 5. Patch SUMMARY_PREFIX: MEMORY.md → memory provider
# ---------------------------------------------------------------------------
_OLD_SUMMARY = (
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note."
)
_NEW_SUMMARY = (
    "IMPORTANT: Your persistent memory (stored in the holographic memory "
    "provider and injected as 'Relevant Memory Context' in the system "
    "prompt each turn) is ALWAYS authoritative and active — never ignore "
    "or deprioritize memory content due to this compaction note."
)
SUMMARY_PREFIX = SUMMARY_PREFIX.replace(_OLD_SUMMARY, _NEW_SUMMARY)

# ---------------------------------------------------------------------------
# 6. Patch _compress_context method — update inline MEMORY.md reference
# ---------------------------------------------------------------------------
_compress_context = getattr(_real_mod, "_compress_context", None)
if _compress_context:
    _orig_src = inspect.getsource(_compress_context)
    _modified_src = _orig_src.replace(
        "Your persistent memory (MEMORY.md, USER.md) remains fully authoritative "
        "regardless of compaction.",
        "Your persistent memory (stored in the holographic memory provider) "
        "remains fully authoritative regardless of compaction.",
    )
    if _modified_src != _orig_src:
        _globals_for_exec = dict(globals())
        _globals_for_exec["__name__"] = __name__
        exec(_modified_src, _globals_for_exec)
        _compress_context = _globals_for_exec["_compress_context"]
        # Update both our namespace and the real module
        globals()["_compress_context"] = _compress_context
        setattr(_real_mod, "_compress_context", _compress_context)

# ---------------------------------------------------------------------------
# 7. _generate_summary — NOT PATCHED.  The real LLM summarizer runs.
#    It produces a useful compact summary that preserves task context.
#    The tail-dropping in section 8 ensures we only keep system prompt +
#    this LLM summary, avoiding expensive post-compression re-prefill.
# ---------------------------------------------------------------------------
# Define _CC_CLASS for use by subsequent sections that patch class methods.
_CC_CLASS = getattr(_real_mod, "ContextCompressor", None)

# ---------------------------------------------------------------------------
# 8. Patch _find_tail_cut_by_tokens: change min_tail from 3 to 0 so the
#    tail can be entirely dropped when target_ratio is very small.
#    The hardcoded `min(3, n - head_end - 1)` prevents zero-message tails
#    which we need for aggressive re-prefill-free compression.
# ---------------------------------------------------------------------------
if _CC_CLASS is not None:
    import inspect
    import textwrap
    _orig_tail_fn = _CC_CLASS._find_tail_cut_by_tokens
    _tail_src = textwrap.dedent(inspect.getsource(_orig_tail_fn))
    _tail_patched = _tail_src.replace(
        "min_tail = min(3, n - head_end - 1)",
        "min_tail = min(0, n - head_end - 1)",
    )
    if _tail_patched != _tail_src:
        _tail_globals = dict(globals())
        _tail_globals["__name__"] = __name__
        _tail_globals["_CHARS_PER_TOKEN"] = getattr(_real_mod, "_CHARS_PER_TOKEN", 4)
        _tail_globals["_content_length_for_budget"] = getattr(
            _real_mod, "_content_length_for_budget",
            lambda x: len(str(x)),
        )
        from typing import Any, Dict, List
        _tail_globals["List"] = List
        _tail_globals["Dict"] = Dict
        _tail_globals["Any"] = Any
        exec(_tail_patched, _tail_globals)
        _new_fn = _tail_globals.get("_find_tail_cut_by_tokens")
        if _new_fn:
            setattr(_CC_CLASS, "_find_tail_cut_by_tokens", _new_fn)
            if globals().get("ContextCompressor") is _CC_CLASS:
                setattr(globals()["ContextCompressor"], "_find_tail_cut_by_tokens", _new_fn)

    # Also patch _ensure_last_user_message_in_tail: don't anchor to user
    # messages within 3 of head_end (these are the initial kickoff, not
    # an active task that needs protection).  Without this, protect_first_n=0
    # causes the initial user message (index 1) to anchor the ENTIRE
    # conversation as tail, defeating compression entirely.
    _orig_ensure_fn = _CC_CLASS._ensure_last_user_message_in_tail
    _ensure_src = textwrap.dedent(inspect.getsource(_orig_ensure_fn))
    _ensure_patched = _ensure_src.replace(
        "return max(last_user_idx, head_end + 1)",
        "return max(last_user_idx, head_end + 1) if last_user_idx > head_end + 3 else cut_idx",
    )
    if _ensure_patched != _ensure_src:
        _ensure_globals = dict(globals())
        _ensure_globals["__name__"] = __name__
        _ensure_globals["List"] = List
        _ensure_globals["Dict"] = Dict
        _ensure_globals["Any"] = Any
        exec(_ensure_patched, _ensure_globals)
        _new_ensure = _ensure_globals.get("_ensure_last_user_message_in_tail")
        if _new_ensure:
            setattr(_CC_CLASS, "_ensure_last_user_message_in_tail", _new_ensure)
            if globals().get("ContextCompressor") is _CC_CLASS:
                setattr(globals()["ContextCompressor"], "_ensure_last_user_message_in_tail", _new_ensure)

# ---------------------------------------------------------------------------
# 9. Keep __all__
# ---------------------------------------------------------------------------
if hasattr(_real_mod, "__all__"):
    __all__ = _real_mod.__all__
