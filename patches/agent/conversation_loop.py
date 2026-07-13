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

# Disable MCP circuit breaker — consecutive failure cooldown wastes
# API calls on free-tier models for no benefit in this use case.
import tools.mcp_tool as _mcp_tool
_mcp_tool._CIRCUIT_BREAKER_THRESHOLD = 9999

# Pre-import the rich output renderer so it's available at exec time
# (the boot chain may overwrite sys.modules['agent'], making
# from agent.rich_output import ... fail inside the exec'd function).
from agent.rich_output import render_response as _render_response
from agent.rich_output import render_final as _render_final

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

# -- Replacement A: skip broken main replacement, use fallback --
_OLD_PREFETCH_LINE = "THIS_PATTERN_WILL_NEVER_MATCH_IN_THE_REAL_SOURCE_xxxxxxxxxx"
_modified_src = _orig_src

if _modified_src == _orig_src:
    _modified_src = _modified_src.replace(
        'agent._memory_manager.prefetch_all(_query) or ""',
        'agent._memory_manager.prefetch_all(_query) or ""\n'
        '            agent._last_prefetch_query = _query\n'
        '            try:\n'
        '                for _aprov in getattr(getattr(agent, "_memory_manager", None), "providers", []):\n'
        '                    _apn = getattr(_aprov, "_provider_name", "") or ""\n'
        '                    if "holographic" in _apn.lower():\n'
        '                        _override = getattr(_aprov, "_config", {}).get("prefetch_query_override", "")\n'
        '                        if _override:\n'
        '                            agent._last_prefetch_query = _override\n'
        '                        break\n'
        '            except Exception:\n'
        '                pass\n'
        '            agent._last_prefetch_facts = []\n'
        '            _all_provs = getattr(agent._memory_manager, "_providers", None) or getattr(agent._memory_manager, "providers", [])\n'
        '            for _prov in _all_provs:\n'
        '                if hasattr(_prov, "_last_raw_results"):\n'
        '                    agent._last_prefetch_facts = _prov._last_raw_results\n'
        '                    break\n'
        '            # Reformat as FACTS section\n'
        '            if agent._last_prefetch_facts:\n'
        '                _facts_lines = ["== FACTS =="]\n'
        '                for _r in agent._last_prefetch_facts:\n'
        '                    _t = _r.get("trust_score", 0)\n'
        '                    _c = _r.get("content", "")\n'
        '                    _facts_lines.append(f"- [{_t:.1f}] {_c}")\n'
        '                _ext_prefetch_cache = "\\n".join(_facts_lines)\n'
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
                + 'agent._last_prefetch_query = _query\n'
                + _spaces + 'try:\n'
                + _spaces + '    for _aprov in getattr(getattr(agent, "_memory_manager", None), "providers", []):\n'
                + _spaces + '        _apn = getattr(_aprov, "_provider_name", "") or ""\n'
                + _spaces + '        if "holographic" in _apn.lower():\n'
                + _spaces + '            _override = getattr(_aprov, "_config", {}).get("prefetch_query_override", "")\n'
                + _spaces + '            if _override:\n'
                + _spaces + '                agent._last_prefetch_query = _override\n'
                + _spaces + '            break\n'
                + _spaces + 'except Exception:\n'
                + _spaces + '    pass\n'
                + _spaces + 'agent._last_prefetch_facts = []\n'
                + _spaces + '_all_provs = getattr(agent._memory_manager, "_providers", None) or getattr(agent._memory_manager, "providers", [])\n'
                + _spaces + 'for _prov in _all_provs:\n'
                + _spaces + '    if hasattr(_prov, "_last_raw_results"):\n'
                + _spaces + '        agent._last_prefetch_facts = _prov._last_raw_results\n'
                + _spaces + '        break\n'
                + _spaces + '# Reformat as FACTS section\n'
                + _spaces + 'if agent._last_prefetch_facts:\n'
                + _spaces + '    _facts_lines = ["== FACTS =="]\n'
                + _spaces + '    for _r in agent._last_prefetch_facts:\n'
                + _spaces + '        _t = _r.get("trust_score", 0)\n'
                + _spaces + '        _c = _r.get("content", "")\n'
                + _spaces + '        _facts_lines.append(f"- [{_t:.1f}] {_c}")\n'
                + _spaces + '    _ext_prefetch_cache = "\\n".join(_facts_lines)\n'
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
    '    # Fact pipeline — runs AFTER the response is delivered\n'
    '    # Extracts new facts from the turn and scores prefetched facts\n'
    '    # in a single LLM call. Replaces background memory review.\n'
    '    _in_applypilot = bool(os.environ.get("HERMES_APPLYPILOT_MODE"))\n'
    '    if (final_response or _in_applypilot) and not interrupted and (agent._memory_enabled or agent._memory_manager is not None):\n'
    '        try:\n'
    '            # Extract only the current turn (last user + assistant msgs)\n'
    '            _current_turn_msgs = []\n'
    '            for _m in reversed(messages):\n'
    '                _current_turn_msgs.insert(0, _m)\n'
    '                if _m.get("role") == "user":\n'
    '                    break\n'
    '            from agent.background_review import _build_conversation_text, _run_fact_extraction, _run_fact_scoring\n'
    '            if _in_applypilot:\n'
    '                # In ApplyPilot mode the agent replies are tool-only (empty text),\n'
    '                # making final_response falsy. Pass the full conversation after\n'
    '                # the system + first-user setup message so the extractor has\n'
    '                # context about what happened across the whole job.\n'
    '                _conv_text = _build_conversation_text(messages[2:])\n'
    '            else:\n'
    '                _conv_text = _build_conversation_text(_current_turn_msgs)\n'
    '            _extracted = _run_fact_extraction(agent, _conv_text)\n'
    '            if _extracted:\n'
    '                import threading as _thr\n'
    '                _thr.Thread(target=_run_fact_scoring, args=(agent, _conv_text), daemon=False, name="fact-scoring").start()\n'
    '        except Exception:\n'
    '            pass  # Fact pipeline is best-effort\n'
)
# Wire up the review trigger replacement
_modified_src = _modified_src.replace(_OLD_REVIEW_TRIGGER, _NEW_REVIEW_TRIGGER)

# -- Replacement E: compression timeout (600s) — fail gracefully instead of hanging.
# 120s is too short for cloud APIs like Gemini free tier.
_modified_src = _modified_src.replace(
    '                    original_len = len(messages)\n'
    '                    messages, active_system_prompt = agent._compress_context(\n'
    '                        messages, system_message, approx_tokens=approx_tokens,\n'
    '                        task_id=effective_task_id,\n'
    '                    )',
    '                    original_len = len(messages)\n'
    '                    # ── Compression with 600s timeout (runs in background thread) ──\n'
    '                    import threading as _thr\n'
    '                    _compress_result = []\n'
    '                    _compress_error = []\n'
    '                    def _do_compress():\n'
    '                        try:\n'
    '                            _compress_result.append(agent._compress_context(\n'
    '                                messages, system_message,\n'
    '                                approx_tokens=approx_tokens,\n'
    '                                task_id=effective_task_id,\n'
    '                            ))\n'
    '                        except Exception as _ce:\n'
    '                            _compress_error.append(_ce)\n'
    '                    _ct = _thr.Thread(target=_do_compress, daemon=True)\n'
    '                    _ct.start()\n'
    '                    _ct.join(timeout=600)\n'
    '                    if _ct.is_alive():\n'
    '                        agent._vprint(f"{agent.log_prefix}⏱️ Compression timed out (600s) — persisting session for continuation retry", force=True)\n'
    '                        agent._persist_session(messages, conversation_history)\n'
    '                        return {\n'
    '                            "messages": messages,\n'
    '                            "completed": False,\n'
    '                            "api_calls": api_call_count,\n'
    '                            "error": "Compression timed out (600s)",\n'
    '                            "partial": True,\n'
    '                            "failed": True,\n'
    '                        }\n'
    '                    if _compress_error:\n'
    '                        raise _compress_error[0]\n'
    '                    messages, active_system_prompt = _compress_result[0]',
)
# Exponential backoff (2-60s or 5-120s) is wrong for local single-server setups —
# the server is either up or down, and retrying in ~1s is always correct.
# For cloud APIs (Gemini free tier, etc.), respect _retry_after from the API.
_modified_src = _modified_src.replace(
    '                    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)',
    '                    wait_time = 1.0',
)

# Also patch the main API error retry path — keep _retry_after if the API
# provides it (Gemini returns retryDelay on 429), otherwise use 1s.
_modified_src = _modified_src.replace(
    '                wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)',
    '                wait_time = _retry_after if _retry_after else 1.0',
)

# -- # -- Replacement F: parse JSON error body for Google's RetryInfo.retryDelay ---------
# Gemini puts retryDelay in the JSON body (google.rpc.RetryInfo), not HTTP headers.
# Append body parsing after the header-based _retry_after extraction.
_modified_src = _modified_src.replace(
    '                            except (TypeError, ValueError):\n'
    '                                pass\n'
    '                wait_time = _retry_after if _retry_after else',
    '                            except (TypeError, ValueError):\n'
    '                                pass\n'
    '                        # Also check JSON error body for Google\'s RetryInfo format\n'
    '                        if _retry_after is None and is_rate_limited:\n'
    '                            _err_body = getattr(api_error, "body", None)\n'
    '                            if _err_body is not None and "retryDelay" in str(_err_body):\n'
    '                                import json as _json\n'
    '                                try:\n'
    '                                    _err_data = _json.loads(_err_body) if isinstance(_err_body, str) else _err_body\n'
    '                                    for _detail in _err_data.get("error", {}).get("details", []):\n'
    '                                        if _detail.get("@type", "").endswith("RetryInfo"):\n'
    '                                            _rd = _detail.get("retryDelay", "")\n'
    '                                            if _rd.endswith("s"):\n'
    '                                                _retry_after = min(float(_rd[:-1]), 120.0)\n'
    '                                except Exception:\n'
    '                                    pass\n'
    '                wait_time = _retry_after if _retry_after else',
)

# -- Replacement G: render assistant response with Rich Markdown syntax highlighting --
_modified_src = _modified_src.replace(
    '            # Handle assistant response\n'
    '            if assistant_message.content and not agent.quiet_mode:\n'
    '                if agent.verbose_logging:\n'
    '                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content}")\n'
    '                else:\n'
    '                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{\'...\' if len(assistant_message.content) > 100 else \'\'}")\n',
    '            # Handle assistant response\n'
    '            if assistant_message.content and not agent.quiet_mode:\n'
    '                try:\n'
    '                    _render_response(agent, assistant_message.content, log_prefix=agent.log_prefix)\n'
    '                except Exception:\n'
    '                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content}")\n',
)


# -- Replacement H: apply Rich syntax highlighting to final response --
_modified_src = _modified_src.replace(
    '    # Build result with interrupt info if applicable\n'
    '    result = {\n'
    '        "final_response": final_response,\n',
    '    # Apply Rich syntax highlighting to final response\n'
    '    _rich_display = render_final(final_response)\n'
    '    if _rich_display and _rich_display != final_response:\n'
    '        final_response = _rich_display\n'
    '\n'
    '    # Build result with interrupt info if applicable\n'
    '    result = {\n'
    '        "final_response": final_response,\n',
)


# Execute the modified function source using the real module's namespace
_globals_for_exec = _real_mod.__dict__.copy()
# Inject the rich output renderer — imported at module load before boot chain
_globals_for_exec["_render_response"] = _render_response
_globals_for_exec["render_final"] = _render_final
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
_patched_fn = _globals_for_exec.get("run_conversation")
if _patched_fn is None:
    import logging as _cl
    _cl.getLogger("bg_check").error(
        "PATCHED run_conversation NOT FOUND in exec namespace — using ORIGINAL. "
        "One or more .replace() patterns failed to match."
    )
    run_conversation = _real_mod.run_conversation
else:
    run_conversation = _patched_fn
_real_mod.run_conversation = run_conversation

# ---------------------------------------------------------------------------
# 5. Keep __all__ from the real module.
# ---------------------------------------------------------------------------
if hasattr(_real_mod, "__all__"):
    __all__ = _real_mod.__all__
else:
    __all__ = [n for n in _imported_names if not n.startswith("_")]
