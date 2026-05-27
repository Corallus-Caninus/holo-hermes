"""Proxy package that shadows the Nix-store agent with our patches.
Extends __path__ so unimpatched submodules (agent_init, etc.)
still resolve from the real Nix-store directory.
"""
import sys
from pathlib import Path

# Find the real agent directory in sys.path (skip our patches dir)
_real_agent: Path | None = None
for _p in sys.path:
    _candidate = Path(_p) / "agent"
    if (
        (_candidate / "__init__.py").is_file()
        and _candidate.resolve() != Path(__file__).parent.resolve()
    ):
        _real_agent = _candidate
        break

if _real_agent is not None:
    # Prepend our dir so patched submodules take priority,
    # append the real dir so everything else resolves.
    __path__ = [str(Path(__file__).parent), str(_real_agent)]
