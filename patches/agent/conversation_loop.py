"""Patched conversation loop — injects holographic memory prefetch
into ephemeral_system_prompt on every turn, fixing the review trigger
gate so the background review actually fires when using fact_store.

This module loads everything from the real Nix-store conversation_loop.py
and patches only the ``run_conversation`` function.
"""

import logging as _patch_log
_patch_log.getLogger("bg_check").warning("conversation_loop patch LOADED")

# ruff: noqa: F401, F403 — we intentionally re-export the real module's names

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import names that the exec'd run_conversation function references
# from the real conversation_loop module's module-level scope.
from agent.process_bootstrap import _install_safe_stdio

# ---------------------------------------------------------------------------
# 1. Find and load the real conversation_loop module from agent.__path__
#    (skipping our patches directory to avoid circular resolution).
# ---------------------------------------------------------------------------
import agent  # noqa: E402 — required for agent.__path__ lookup

_real_path: Path | None = None
for _p in agent.__path__:
    if "patches" in str(_p):
        continue
    _candidate = Path(_p) / "conversation_loop.py"
    if _candidate.exists():
        _real_path = _candidate
        break

if _real_path is None:
    msg = "Could not locate real agent/conversation_loop.py in agent.__path__"
    raise ImportError(msg)

_spec = importlib.util.spec_from_file_location(
    "agent.conversation_loop_real", str(_real_path)
)
_real_mod = importlib.util.module_from_spec(_spec)
sys.modules["agent.conversation_loop_real"] = _real_mod
_spec.loader.exec_module(_real_mod)

# ---------------------------------------------------------------------------
# 2. Re-export everything from the real module into our namespace.
# ---------------------------------------------------------------------------
_imported_names: set = set()
for _attr in dir(_real_mod):
    if _attr.startswith("_") and _attr not in ("__all__",):
        continue
    globals()[_attr] = getattr(_real_mod, _attr)
    _imported_names.add(_attr)

# ---------------------------------------------------------------------------
# 3. Patch run_conversation — fix trigger gate + inject prefetch.
# ---------------------------------------------------------------------------
_orig_src = inspect.getsource(_real_mod.run_conversation)

# -- Replacement A: inject prefetch into ephemeral_system_prompt -----------
_OLD_PREFETCH_LINE = (
    '            _ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""'
)
_NEW_PREFETCH_BLOCK = (
    '            _ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""\n'
    '            # Inject prefetched memory into ephemeral system prompt\n'
    '            if _ext_prefetch_cache and hasattr(agent, "ephemeral_system_prompt"):\n'
    '                _mem_block = _ext_prefetch_cache\n'
    '                _existing = agent.ephemeral_system_prompt or ""\n'
    '                if _existing:\n'
    '                    _mem_block = _mem_block + "\\n\\n" + _existing\n'
    '                agent.ephemeral_system_prompt = _mem_block'
)
_modified_src = _orig_src.replace(_OLD_PREFETCH_LINE, _NEW_PREFETCH_BLOCK)

if _modified_src == _orig_src:
    _modified_src = _modified_src.replace(
        'agent._memory_manager.prefetch_all(_query) or ""',
        'agent._memory_manager.prefetch_all(_query) or ""\n'
        '            # Inject prefetched memory into ephemeral system prompt\n'
        '            if _ext_prefetch_cache and hasattr(agent, "ephemeral_system_prompt"):\n'
        '                _mem_block = _ext_prefetch_cache\n'
        '                _existing = agent.ephemeral_system_prompt or ""\n'
        '                if _existing:\n'
        '                    _mem_block = _mem_block + "\\n\\n" + _existing\n'
        '                agent.ephemeral_system_prompt = _mem_block',
    )

if _modified_src == _orig_src:
    _lines = _orig_src.split("\n")
    for _i, _line in enumerate(_lines):
        if "prefetch_all(_query) or " in _line and "_ext_prefetch_cache" in _line:
            _indent = len(_line) - len(_line.lstrip())
            _spaces = " " * _indent
            _new_line = (
                _line + "\n" + _spaces
                + '# Inject prefetched memory into ephemeral system prompt\n'
                + _spaces + 'if _ext_prefetch_cache and hasattr(agent, "ephemeral_system_prompt"):\n'
                + _spaces + '    _mem_block = _ext_prefetch_cache\n'
                + _spaces + '    _existing = agent.ephemeral_system_prompt or ""\n'
                + _spaces + '    if _existing:\n'
                + _spaces + '        _mem_block = _mem_block + "\\n\\n" + _existing\n'
                + _spaces + '    agent.ephemeral_system_prompt = _mem_block'
            )
            _lines[_i] = _new_line
            _modified_src = "\n".join(_lines)
            break

# -- Replacement B: fix memory review gate from _memory_store to _memory_manager --
_modified_src = _modified_src.replace(
    '            and "memory" in agent.valid_tool_names\n'
    '            and agent._memory_store):',
    '            and (agent._memory_store is not None\n'
    '                 or agent._memory_manager is not None)):\n'
    '        import logging as _bg_l\n'
    '        _bg_l.getLogger("bg_check").info(\n'
    '            "review trigger: nudge=%s memory_store=%s mgr=%s tsm=%s",\n'
    '            agent._memory_nudge_interval,\n'
    '            agent._memory_store is not None,\n'
    '            agent._memory_manager is not None,\n'
    '            agent._turns_since_memory,\n'
    '        )',
)

# -- Replacement C: make background reviews fire every turn, with only
#    the current turn's messages. Uses non-daemon threads so the review
#    completes before process exit (critical for -q/one-shot mode).
_OLD_REVIEW_TRIGGER = (
    '    # Background memory/skill review — runs AFTER the response is delivered\n'
    '    # so it never competes with the user\'s task for model attention.\n'
    '    if final_response and not interrupted and (_should_review_memory or _should_review_skills):\n'
    '        try:\n'
    '            agent._spawn_background_review(\n'
    '                messages_snapshot=list(messages),\n'
    '                review_memory=_should_review_memory,\n'
    '                review_skills=_should_review_skills,\n'
    '            )\n'
    '        except Exception:\n'
    '            pass  # Background review is best-effort\n'
)
_NEW_REVIEW_TRIGGER = (
    '    # Background memory review — runs AFTER the response is delivered\n'
    '    # so it never competes with the user\'s task for model attention.\n'
    '    # Fires every turn with only the current turn\'s messages.\n'
    '    # Uses non-daemon threads so the review completes before process exit.\n'
    '    if final_response and not interrupted:\n'
    '        try:\n'
    '            # Extract only the current turn (last user + assistant msgs)\n'
    '            _current_turn_msgs = []\n'
    '            for _m in reversed(messages):\n'
    '                _current_turn_msgs.insert(0, _m)\n'
    '                if _m.get("role") == "user":\n'
    '                    break\n'
    '            from agent.background_review import spawn_background_review_thread\n'
    '            _bg_target, _ = spawn_background_review_thread(\n'
    '                agent, _current_turn_msgs,\n'
    '                review_memory=True,\n'
    '                review_skills=False,\n'
    '            )\n'
    '            import threading as _thr\n'
    '            _thr.Thread(target=_bg_target, daemon=False, name="bg-review").start()\n'
    '        except Exception:\n'
    '            pass  # Background review is best-effort\n'
    '        # Spawn fact evaluation thread in parallel (if pending data)\n'
    '        try:\n'
    '            import os as _os\n'
    '            from hermes_constants import get_hermes_home\n'
    '            if _os.path.isfile(str(get_hermes_home() / ".pending_fact_eval.json")):\n'
    '                from agent.background_review import spawn_fact_evaluation_thread\n'
    '                _fact_target, _ = spawn_fact_evaluation_thread(agent, _current_turn_msgs)\n'
    '                import threading as _thr2\n'
    '                _thr2.Thread(target=_fact_target, daemon=False, name="bg-fact-eval").start()\n'
    '        except Exception:\n'
    '            pass  # Fact evaluation is best-effort\n'
)
_modified_src = _modified_src.replace(_OLD_REVIEW_TRIGGER, _NEW_REVIEW_TRIGGER)

# -- Replacement D: include last assistant response in prefetch query -------
_modified_src = _modified_src.replace(
    '            _query = original_user_message if isinstance(original_user_message, str) else ""',
    '            _query = original_user_message if isinstance(original_user_message, str) else ""\n'
            '            # Include last assistant response for multi-turn context\n'
            '            if messages and len(messages) >= 2:\n'
            '                _last_asst_msg = messages[-2]\n'
            '                if isinstance(_last_asst_msg, dict) and _last_asst_msg.get("role") == "assistant":\n'
            '                    _asst_text = _last_asst_msg.get("content", "")\n'
            '                    _asst_reasoning = _last_asst_msg.get("reasoning", "")\n'
            '                    if isinstance(_asst_reasoning, str) and _asst_reasoning.strip():\n'
            '                        _asst_text = _asst_reasoning.strip() + "\\n\\n" + (_asst_text or "")\n'
            '                    if isinstance(_asst_text, str) and _asst_text.strip():\n'
            '                        _query = _asst_text.strip() + "\\n" + _query',
)

# Execute the modified function source using the real module's namespace
_globals_for_exec = _real_mod.__dict__.copy()
_globals_for_exec["__name__"] = __name__
try:
    exec(_modified_src, _globals_for_exec)
except Exception as _exec_err:
    import logging as _exec_log
    _exec_log.getLogger("bg_check").error(
        "conversation_loop exec failed: %s", _exec_err, exc_info=True
    )
    raise

# Replace run_conversation in our namespace with the patched version
run_conversation = _globals_for_exec.get("run_conversation", _real_mod.run_conversation)
_real_mod.run_conversation = run_conversation

# ---------------------------------------------------------------------------
# 5. Keep __all__ from the real module.
# ---------------------------------------------------------------------------
if hasattr(_real_mod, "__all__"):
    __all__ = _real_mod.__all__
else:
    __all__ = [n for n in _imported_names if not n.startswith("_")]
