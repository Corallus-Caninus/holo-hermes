"""Patched context compressor — updates MEMORY.md/USER.md references to
reflect the switch to holographic memory store.

Loads everything from the real Nix-store context_compressor.py and patches
``SUMMARY_PREFIX`` and the inline compression note to reference the memory
provider instead of MEMORY.md/USER.md.
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

# ---------------------------------------------------------------------------
# 2. Re-export all names.
# ---------------------------------------------------------------------------
for _attr in dir(_real_mod):
    if _attr.startswith("_") and _attr not in ("__all__",):
        continue
    globals()[_attr] = getattr(_real_mod, _attr)

# ---------------------------------------------------------------------------
# 3. Patch SUMMARY_PREFIX: MEMORY.md → memory provider
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
# 4. Patch _compress_context method — update inline MEMORY.md reference
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
# 5. Keep __all__
# ---------------------------------------------------------------------------
if hasattr(_real_mod, "__all__"):
    __all__ = _real_mod.__all__
