"""Proxy package for plugins — extends __path__ to include the real
Nix-store plugins directory AND re-exports everything from the real
module, allowing us to shadow individual submodules (e.g. holographic)
while keeping the module-level API intact.
"""
import importlib.util
import sys
from pathlib import Path

_parent = Path(__file__).parent

# Extend __path__ so submodule imports (plugins.memory, plugins.browser, etc.)
# resolve from the real Nix-store directory.
_real_plugins: Path | None = None
for _p in sys.path:
    if "patches" in str(_p):
        continue
    _candidate = Path(_p) / "plugins"
    if _candidate.is_dir() and (_candidate / "__init__.py").is_file():
        _rv = _candidate.resolve()
        if _rv != _parent.resolve():
            _real_plugins = _rv
            break

if _real_plugins is not None:
    __path__ = [str(_parent), str(_real_plugins)]

    # Re-export everything from the real module so
    # "from plugins import ..." works.
    _real_init = _real_plugins / "__init__.py"
    _real_name = "plugins_real"
    _spec = importlib.util.spec_from_file_location(_real_name, str(_real_init))
    _real_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_real_mod)
    for _attr in dir(_real_mod):
        if _attr.startswith("_") and _attr != "__all__":
            continue
        globals()[_attr] = getattr(_real_mod, _attr)
