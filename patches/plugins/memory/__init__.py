"""Proxy package for plugins.memory — extends __path__ to include the real
Nix-store plugins/memory directory, AND re-exports module-level names
(load_memory_provider, find_provider_dir, discover_memory_providers, etc.)
so imports like "from plugins.memory import load_memory_provider" work.

Debug: set MEMORY_DEBUG=1 to trace provider resolution.
"""
import importlib.util
import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger("plugins.memory.patch")
_DEBUG_MEM = os.environ.get("MEMORY_DEBUG", "").lower() in ("1", "true", "yes")
def _dbg(m): _log.info("PATCH_MEMORY: " + m) if _DEBUG_MEM else None
_dbgn = lambda *a: _dbg(a[0] % a[1:] if len(a) > 1 else a[0]) if _DEBUG_MEM else None  # noqa: E731

_dbgn("proxy loaded from %s", __file__)

_parent = Path(__file__).parent

# Extend __path__ so submodule imports (plugins.memory.holographic, etc.)
# resolve from the real Nix-store directory.
_real_memory: Path | None = None
for _p in sys.path:
    if "patches" in str(_p):
        continue
    _candidate = Path(_p) / "plugins" / "memory"
    if _candidate.is_dir() and (_candidate / "__init__.py").is_file():
        _rv = _candidate.resolve()
        if _rv != _parent.resolve():
            _real_memory = _rv
            _dbgn("found real plugins.memory at %s", _real_memory)
            break

if _real_memory is not None:
    __path__ = [str(_parent), str(_real_memory)]
    _dbgn("__path__ = %s", __path__)

    # Re-export all module-level names from the real plugins/memory/__init__.py
    _real_init = _real_memory / "__init__.py"
    _real_name = "plugins.memory_real"
    _spec = importlib.util.spec_from_file_location(_real_name, str(_real_init))
    _real_mod = importlib.util.module_from_spec(_spec)
    sys.modules[_real_name] = _real_mod
    _spec.loader.exec_module(_real_mod)
    _dbgn("real module loaded: %d names re-exported",
          sum(1 for _a in dir(_real_mod) if not _a.startswith("_") or _a == "__all__"))
    for _attr in dir(_real_mod):
        if _attr.startswith("_") and _attr != "__all__":
            continue
        globals()[_attr] = getattr(_real_mod, _attr)

    # Override _MEMORY_PLUGINS_DIR to point at our patches dir so
    # find_provider_dir("holographic") returns the patched version.
    old_dir = str(getattr(_real_mod, "_MEMORY_PLUGINS_DIR", "UNSET"))
    _real_mod._MEMORY_PLUGINS_DIR = _parent
    _dbgn("_MEMORY_PLUGINS_DIR: %s -> %s", old_dir, _parent)
    _dbgn("find_provider_dir('holographic') -> %s",
          globals().get("find_provider_dir", "NOT FOUND"))

    # Clear holographic module cache so the patches version gets loaded.
    for _key in list(sys.modules.keys()):
        if "holographic" in _key:
            del sys.modules[_key]
            _dbgn("cleared cached module: %s", _key)

    # Wrap load_memory_provider to clear cached modules right before
    # loading, defeating the boot chain's pre-import of the Nix-store version.
    _orig_lmp = globals().get("load_memory_provider")
    if _orig_lmp:
        def _patched_lmp(name: str):
            for _k in list(sys.modules.keys()):
                if "holographic" in _k:
                    del sys.modules[_k]
            return _orig_lmp(name)
        globals()["load_memory_provider"] = _patched_lmp
        _real_mod.load_memory_provider = _patched_lmp
        _dbgn("load_memory_provider wrapped to clear holographic cache")

    # Wrap find_provider_dir to log what it returns
    _orig_fpd = globals().get("find_provider_dir")
    if _orig_fpd:
        def _patched_fpd(name: str):
            result = _orig_fpd(name)
            _dbgn("find_provider_dir(%r) = %s", name, result)
            return result
        globals()["find_provider_dir"] = _patched_fpd
        # Also update the real module so load_memory_provider uses our wrapper
        _real_mod.find_provider_dir = _patched_fpd
else:
    _dbgn("ERROR: real plugins.memory NOT FOUND")
