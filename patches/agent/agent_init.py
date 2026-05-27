"""Patched agent_init — strips the old MEMORY.md/USER.md MemoryStore
initialization (dead code since we use holographic fact_store exclusively).
"""
# ruff: noqa: F401, F403 — intentional re-export

import importlib.util
import inspect
import sys
from pathlib import Path

import agent  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Load the real agent_init module.
# ---------------------------------------------------------------------------
_real_path: Path | None = None
for _p in agent.__path__:
    if "patches" in str(_p):
        continue
    _candidate = Path(_p) / "agent_init.py"
    if _candidate.exists():
        _real_path = _candidate
        break

if _real_path is None:
    raise ImportError("Could not locate real agent/agent_init.py")

_spec = importlib.util.spec_from_file_location(
    "agent.agent_init_real", str(_real_path)
)
_real_mod = importlib.util.module_from_spec(_spec)
sys.modules["agent.agent_init_real"] = _real_mod
_spec.loader.exec_module(_real_mod)

# ---------------------------------------------------------------------------
# 2. Re-export all names from the real module.
# ---------------------------------------------------------------------------
for _attr in dir(_real_mod):
    if _attr.startswith("_") and _attr not in ("__all__",):
        continue
    globals()[_attr] = getattr(_real_mod, _attr)

# ---------------------------------------------------------------------------
# 3. Patch aiagent_init to remove the dead MEMORY.md MemoryStore init block.
# ---------------------------------------------------------------------------
_init_agent = getattr(_real_mod, "init_agent", None)
if _init_agent:
    _orig_src = inspect.getsource(_init_agent)

    # Remove the entire MemoryStore initialization block:
    #   # Persistent memory (MEMORY.md + USER.md) -- loaded from disk
    #   agent._memory_store = None
    #   ...
    #   if agent._memory_enabled or agent._user_profile_enabled:
    #       from tools.memory_tool import MemoryStore
    #       ...
    #
    # But keep _memory_nudge_interval and _turns_since_memory which
    # are still used by the background review trigger.
    _OLD_MEMORY_BLOCK = (
        "# Persistent memory (MEMORY.md + USER.md) -- loaded from disk\n"
        "    agent._memory_store = None\n"
        "    agent._memory_enabled = False\n"
        "    agent._user_profile_enabled = False\n"
        "    agent._memory_nudge_interval = 10\n"
        "    agent._turns_since_memory = 0\n"
        "    agent._iters_since_skill = 0\n"
        "    if not skip_memory:\n"
        "        try:\n"
        "            mem_config = _agent_cfg.get(\"memory\", {})\n"
        "            agent._memory_enabled = mem_config.get(\"memory_enabled\", False)\n"
        "            agent._user_profile_enabled = mem_config.get(\"user_profile_enabled\", False)\n"
        "            agent._memory_nudge_interval = int(mem_config.get(\"nudge_interval\", 10))\n"
        "            if agent._memory_enabled or agent._user_profile_enabled:\n"
        "                from tools.memory_tool import MemoryStore\n"
        "                agent._memory_store = MemoryStore(\n"
        "                    memory_char_limit=mem_config.get(\"memory_char_limit\", 2200),\n"
        "                    user_char_limit=mem_config.get(\"user_char_limit\", 1375),\n"
        "                )\n"
        "                agent._memory_store.load_from_disk()\n"
        "        except Exception:\n"
        "            pass  # Memory is optional -- don't break agent init"
    )
    _NEW_MEMORY_BLOCK = (
        "# Holographic fact_store is used instead of MEMORY.md.\n"
        "    agent._memory_store = None\n"
        "    agent._memory_enabled = False\n"
        "    agent._user_profile_enabled = False\n"
        "    agent._memory_nudge_interval = 10\n"
        "    agent._turns_since_memory = 0\n"
        "    agent._iters_since_skill = 0\n"
        "    if not skip_memory:\n"
        "        try:\n"
        "            mem_config = _agent_cfg.get(\"memory\", {})\n"
        "            agent._memory_nudge_interval = int(mem_config.get(\"nudge_interval\", 10))\n"
        "        except Exception:\n"
        "            pass"
    )

    _modified_src = _orig_src.replace(_OLD_MEMORY_BLOCK, _NEW_MEMORY_BLOCK)

    if _modified_src == _orig_src:
        # Fallback: try shorter match for just the MemoryStore creation
        _modified_src = _orig_src.replace(
            "if agent._memory_enabled or agent._user_profile_enabled:\n"
            "                from tools.memory_tool import MemoryStore\n"
            "                agent._memory_store = MemoryStore(\n"
            "                    memory_char_limit=mem_config.get(\"memory_char_limit\", 2200),\n"
            "                    user_char_limit=mem_config.get(\"user_char_limit\", 1375),\n"
            "                )\n"
            "                agent._memory_store.load_from_disk()",
            "# MemoryStore init removed — holographic fact_store is used instead",
        )

    if _modified_src != _orig_src:
        _globals_for_exec = _real_mod.__dict__.copy()
        _globals_for_exec["__name__"] = "agent.agent_init"
        exec(_modified_src, _globals_for_exec)
        # Replace init_agent in both our namespace and the real module
        globals()["init_agent"] = _globals_for_exec.get("init_agent", _init_agent)
        setattr(_real_mod, "init_agent", globals()["init_agent"])

# ---------------------------------------------------------------------------
# 4. Keep __all__
# ---------------------------------------------------------------------------
if hasattr(_real_mod, "__all__"):
    __all__ = _real_mod.__all__
