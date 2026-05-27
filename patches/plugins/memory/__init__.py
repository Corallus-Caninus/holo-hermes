"""Proxy package for plugins.memory — extends __path__ to include the real
Nix-store plugins/memory directory, AND re-exports module-level names
(load_memory_provider, find_provider_dir, discover_memory_providers, etc.)
so imports like "from plugins.memory import load_memory_provider" work.
"""
import importlib.util
import sys
from pathlib import Path

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
            break

if _real_memory is not None:
    __path__ = [str(_parent), str(_real_memory)]

    # Re-export all module-level names from the real plugins/memory/__init__.py
    _real_init = _real_memory / "__init__.py"
    _real_name = "plugins.memory_real"
    _spec = importlib.util.spec_from_file_location(_real_name, str(_real_init))
    _real_mod = importlib.util.module_from_spec(_spec)
    sys.modules[_real_name] = _real_mod
    _spec.loader.exec_module(_real_mod)
    for _attr in dir(_real_mod):
        if _attr.startswith("_") and _attr != "__all__":
            continue
        globals()[_attr] = getattr(_real_mod, _attr)

    # Override _MEMORY_PLUGINS_DIR to point at our patches dir so
    # find_provider_dir("holographic") returns the patched version.
    _real_mod._MEMORY_PLUGINS_DIR = _parent
