"""Background memory review using direct LLM calls (no agent fork).

After each turn, runs TWO sequential lightweight chat completions:

1. **Fact extraction** — extracts new facts from the conversation -> fact_store('add')
2. **Fact scoring** — queries fact_store for recently-updated facts and evaluates
   which were useful -> fact_feedback with trust score adjustments

No AIAgent is forked, no TTY spinner. All output routes through
agent._safe_print() for TUI compatibility (falls back to raw fd write).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

# Auto-delete stale pyc so source changes take effect on next Hermes restart
import pathlib as _pl
_py = _pl.Path(__file__)
_pyc = _pl.Path(_py.parent / "__pycache__" / (_py.stem + ".cpython-312.pyc"))
if _pyc.exists() and _pyc.stat().st_mtime < _py.stat().st_mtime:
    _pyc.unlink()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy prompt names -- still imported by run_agent.py in the Nix store.
# ---------------------------------------------------------------------------
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and save facts to memory.\n"
    "You can only call memory/skill tools.\n"
    "If nothing to save, say 'Nothing to save.' and stop."
)
_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update skills if needed.\n"
    "You can only call memory/skill tools.\n"
    "If nothing to save, say 'Nothing to save.' and stop."
)

# ---------------------------------------------------------------------------
# Prompt templates for the two-stage review
# ---------------------------------------------------------------------------

_FACT_EXTRACTION_PROMPT = (
    "{conversation}\n\n"
    "---\n\n"
    "Extract facts as JSON from the conversation above. "
    "Reasoning and content both count.\n"
    "{\n"
    '  "new_facts": [\n'
    '    {"content": "...", "category": "user_pref|project|tool|general", "tags": "tag1, tag2"}\n'
    "  ]\n"
    "}\n\n"
    "Empty list if nothing useful."
)

_FACT_SCORING_PROMPT = (
    "{conversation}\n\n"
    "---\n\n"
    "### Memory Facts\n"
    "{prefetched}\n\n"
    "Score which facts were useful.\n"
    "{\n"
    '  "fact_evaluations": [\n'
    '    {"fact_id": N, "useful": true|false, "trust_delta": 0.0}\n'
    "  ]\n"
    "}\n\n"
    "trust_delta +0.001 to +0.05 if useful, "
    "-0.001 to -0.05 if not useful, 0.0 if neutral. "
    "Include ALL fact IDs."
)

# DEPRECATED: conversation text appears twice (once per sub-prompt).
# The dispatch functions use individual prompts, not this combined one.
_COMBINED_REVIEW_PROMPT = _FACT_EXTRACTION_PROMPT + "\n\n" + _FACT_SCORING_PROMPT
_FACT_EVAL_BASE_PROMPT = _FACT_SCORING_PROMPT

# ---------------------------------------------------------------------------
# ApplyPilot-specific prompts (activated by HERMES_APPLYPILOT_MODE=1)
# ---------------------------------------------------------------------------

_APPLYPILOT_EXTRACTION_PROMPT = (
    "{conversation}\n\n"
    "---\n\n"
    "{job_context}\n\n"
    "Extract facts as JSON. For each issue the bot encountered,\n"
    "describe what went wrong AND how it was solved in one fact.\n"
    "{\n"
    '  "new_facts": [\n'
    '    {"content": "what went wrong — how it was solved",\n'
    '     "category": "stuck|workaround|element|general", "tags": "tag1, tag2"}\n'
    "  ]\n"
    "}\n\n"
    'Categories: "stuck" (problem+fix), "workaround" (proactive tip),\n'
    '"element" (field+how-to), "general".\n'
    'Example: "phone field rejected +1 — use raw digits instead"\n'
    "Empty list if nothing useful."
)

_APPLYPILOT_SCORING_PROMPT = (
    "{conversation}\n\n"
    "---\n\n"
    "{job_context}\n\n"
    "### Memory Facts\n"
    "{prefetched}\n\n"
    "Score which memory facts helped the bot avoid getting stuck.\n"
    "Check the [thinking] block for explicit fact references.\n"
    "{\n"
    '  "fact_evaluations": [\n'
    '    {"fact_id": N, "useful": true|false, "trust_delta": 0.0}\n'
    "  ]\n"
    "}\n\n"
    "trust_delta: +0.001 to +0.05 (useful), "
    "-0.001 to -0.05 (not), 0.0 (neutral). "
    "Include ALL fact IDs."
)


# ---------------------------------------------------------------------------
# Print helper (TUI-safe)
# ---------------------------------------------------------------------------

def _safe_bg_print(msg: str, agent: Any = None) -> None:
    """Print from background thread, preferring agent._safe_print for TUI.
    
    agent._safe_print() routes through the TUI's _cprint renderer so
    output appears in the scrollable output area. Falls back to os.write(1)
    when agent is not available or _safe_print fails.
    """
    if agent is not None:
        try:
            agent._safe_print(msg)
            return
        except Exception:
            pass
    try:
        os.write(1, (msg + "\n").encode("utf-8", errors="replace"))
    except Exception:
        try:
            import sys as _sys
            _sys.stderr.write(f"{msg}\n")
            _sys.stderr.flush()
        except Exception:
            pass


def _preview(text: str, max_len: int = 60) -> str:
    """First ~max_len chars of text, ellipsized."""
    t = text.strip()
    if len(t) <= max_len:
        return t
    return t[:max_len] + "..."


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_db_path(agent: Any) -> str:
    """Resolve the memory_store.db path from config, falling back to HERMES_HOME.
    
    The holographic provider's store path is configured via:
        plugins.hermes-memory-store.db_path
    in the agent's config. When this is set (e.g. by ApplyPilot which uses
    a separate DB), the background review MUST use the same path so it can
    find and score the stored facts.
    
    Falls back to get_hermes_home()/memory_store.db for backward compat
    with personal Hermes usage.
    """
    try:
        # Try agent._config (set by fully_automatic_holographic / Hermes)
        _cfg = getattr(agent, "_config", None) or {}
        _db = _cfg.get("plugins", {}).get("hermes-memory-store", {}).get("db_path")
        if _db:
            return str(_db)
    except Exception:
        pass
    try:
        # Try agent._memory_manager._config (alternative storage)
        _mm = getattr(agent, "_memory_manager", None)
        if _mm:
            _cfg = getattr(_mm, "_config", None) or {}
            _db = _cfg.get("plugins", {}).get("hermes-memory-store", {}).get("db_path")
            if _db:
                return str(_db)
    except Exception:
        pass
    # Fallback
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "memory_store.db")


def _build_conversation_text(messages_snapshot: List[Dict]) -> str:
    parts = []
    for m in messages_snapshot:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(texts)
        # Include assistant reasoning/thinking — this is where the model
        # explicitly references prefetched facts ("from fact #47: ...")
        reasoning = m.get("reasoning", "")
        if isinstance(reasoning, str) and reasoning.strip():
            reasoning = reasoning.strip()[:1200]
        else:
            reasoning = ""
        if isinstance(content, str) and content.strip():
            full = content.strip()[:800]
            if role == "assistant" and reasoning:
                full = f"[thinking]: {reasoning}\n\n{full}"
            parts.append(f"[{role}]: {full}")
        elif reasoning:
            parts.append(f"[{role}]: [thinking]: {reasoning}")
    return "\n\n".join(parts) if parts else "(empty turn)"


def _get_recently_retrieved_facts(agent: Any) -> List[dict]:
    """Fetch the facts that were prefetched for the current turn.

    Reads from ``agent._last_prefetch_facts`` (set by the conversation_loop
    patch from the holographic provider's ``_last_raw_results`` after
    prefetch runs). Falls back to iterating providers for backward compat.
    Returns empty list if prefetch tracking is unavailable.
    """
    # Collect prefetched fact IDs
    prefetched_ids: set = set()

    # Primary source: agent._last_prefetch_facts (set by conversation_loop)
    try:
        raw = getattr(agent, "_last_prefetch_facts", None)
        if raw:
            for f in raw:
                if isinstance(f, dict) and f.get("fact_id"):
                    prefetched_ids.add(f["fact_id"])
    except Exception:
        pass

    # Secondary source: iterate providers (legacy fallback)
    if not prefetched_ids:
        try:
            for prov in (agent._memory_manager.providers if agent._memory_manager else []):
                try:
                    if hasattr(prov, "_last_raw_results") and prov._last_raw_results:
                        for f in prov._last_raw_results:
                            if isinstance(f, dict) and f.get("fact_id"):
                                prefetched_ids.add(f["fact_id"])
                except Exception:
                    pass
        except Exception:
            pass

    if not prefetched_ids:
        # No prefetched facts tracked — log empty and return empty
        # The fallback (20 random facts) was misleading — better to score nothing
        return []

    # Fetch full data for those IDs from the DB
    facts = []
    try:
        import sqlite3
        from pathlib import Path
        db_path = Path(_get_db_path(agent))
        if not db_path.exists():
            return facts
        placeholders = ",".join("?" * len(prefetched_ids))
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                f"SELECT fact_id, content, retrieval_count, trust_score, updated_at "
                f"FROM facts WHERE fact_id IN ({placeholders})",
                list(prefetched_ids),
            ).fetchall()
            for row in rows:
                facts.append({
                    "fact_id": row[0],
                    "content": row[1],
                    "retrieval_count": row[2],
                    "trust_score": row[3],
                    "updated_at": str(row[4]),
                })
        finally:
            conn.close()
    except Exception:
        pass
    return facts


def _call_llm(
    agent: Any,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Optional[str]:
    try:
        from agent.auxiliary_client import call_llm
        runtime = agent._current_main_runtime()
        # Extract provider info from runtime to prevent OpenRouter fallback
        _provider = runtime.get("provider") if isinstance(runtime, dict) else None
        _base_url = runtime.get("base_url") if isinstance(runtime, dict) else None
        _model = runtime.get("model") if isinstance(runtime, dict) else None
        if _provider == "custom":
            # Pin to the custom endpoint, never fall back to OpenRouter
            response = call_llm(
                messages=[{"role": "user", "content": prompt}],
                provider="custom",
                base_url=_base_url,
                model=_model,
                main_runtime=runtime,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = call_llm(
                messages=[{"role": "user", "content": prompt}],
                main_runtime=runtime,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning("Background review LLM call failed: %s", e)
        return None


def _try_parse_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.debug("Failed to parse review JSON (no JSON object): %s", text[:500])
        return None
    text = text[start:end+1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Failed to parse review JSON: %s", text[:500])
        return None


def _add_fact(agent: Any, content: str, category: str = "general", tags: str = "") -> bool:
    if not agent._memory_manager:
        return False
    try:
        # Memory manager's tool registry doesn't have "fact_store" in
        # ApplyPilot mode (get_tool_schemas returns []).  Call the
        # holographic provider directly — its handle_tool_call works
        # regardless of get_tool_schemas.
        _providers = getattr(agent._memory_manager, "providers", []) or []
        for _p in _providers:
            _pn = getattr(_p, "_provider_name", "") or ""
            if "holographic" in _pn.lower() and hasattr(_p, "handle_tool_call"):
                _p.handle_tool_call("fact_store", {
                    "action": "add", "content": content,
                    "category": category, "tags": tags,
                })
                return True
        # Fallback for non-ApplyPilot mode where tools are registered
        agent._memory_manager.handle_tool_call("fact_store", {
            "action": "add", "content": content, "category": category, "tags": tags,
        })
        return True
    except Exception as e:
        logger.debug("Failed to add fact: %s", e)
        return False


def _apply_feedback(agent: Any, fact_id: int, useful: bool, trust_delta: float = 0.0,
                    conn: Any = None) -> bool:
    """Update trust score and helpful_count directly in SQLite.

    Bypasses the Nix store's handle_tool_call which applies hardcoded
    deltas (±0.05/-0.10) and ignores trust_delta entirely.  Writes
    the exact delta requested, clamped to ±0.05.

    If a SQLite Connection is passed (reused from _run_fact_scoring), use that
    to avoid "database is locked" errors from opening multiple connections.
    Otherwise opens a new connection (legacy path).

    Raises RuntimeError on failure — no silent fallback.
    """
    import sqlite3
    from pathlib import Path
    db_path = Path(_get_db_path(agent))
    delta = max(-0.05, min(0.05, trust_delta))
    if conn is None:
        conn = sqlite3.connect(str(db_path), timeout=8.0)
        _owns_conn = True
    else:
        _owns_conn = False
    try:
        conn.execute(
            "UPDATE facts SET "
            "trust_score = MIN(1.0, MAX(0.0, trust_score + ?)), "
            "helpful_count = helpful_count + ?, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE fact_id = ?",
            (delta, 1 if useful else 0, fact_id),
        )
        if conn.total_changes == 0:
            raise RuntimeError(f"fact_id {fact_id} not found")
        conn.commit()
    except (sqlite3.Error, RuntimeError):
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if _owns_conn:
            conn.close()
    return True


# ---------------------------------------------------------------------------
# Stage 1: Fact extraction
# ---------------------------------------------------------------------------

def _run_fact_extraction(
    agent: Any,
    conversation_text: str,
) -> list[dict]:
    try:
        # ApplyPilot mode: use alternate prompt + inject job context
        _is_ap = os.environ.get("HERMES_APPLYPILOT_MODE") == "1"
        if _is_ap:
            _jc = getattr(agent, "_last_prefetch_query", "") or ""
            _prompt_name = "_APPLYPILOT_EXTRACTION_PROMPT"
            filled = _APPLYPILOT_EXTRACTION_PROMPT.replace(
                "{conversation}", conversation_text
            ).replace("{job_context}", _jc)
        else:
            _prompt_name = "_FACT_EXTRACTION_PROMPT"
            filled = _FACT_EXTRACTION_PROMPT.replace("{conversation}", conversation_text)
        try:
            _plog = os.path.expanduser("~/.hermes/holographic_debug.log")
            with open(_plog, "a", encoding="utf-8") as _pf:
                _pf.write(json.dumps({
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "event": "fact_extraction_prompt_selected",
                    "prompt_name": _prompt_name,
                    "applypilot_mode": _is_ap,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
    except Exception as e:
        logger.warning("Fact extraction prompt format failed: %s", e)
        return []

    raw = _call_llm(agent, filled)
    if not raw:
        _safe_bg_print("  \u2757 Fact extraction: LLM call failed", agent)
        return []

    result = _try_parse_json(raw)
    if not result:
        _safe_bg_print("  \u2757 Fact extraction: couldn't parse response", agent)
        return []

    new_facts = result.get("new_facts", [])
    if not new_facts:
        _safe_bg_print("  \U0001f4be Fact extraction: nothing to save", agent)
        return []

    saved_facts: list[dict] = []
    saved_previews = []
    # Build company prefix from job context to auto-prepend to every fact
    _fact_prefix = ""
    if _is_ap and _jc:
        _fact_prefix = f"[{_jc}] "
    for f in new_facts:
        content = f.get("content", "").strip()
        if len(content) < 10:
            continue
        if _fact_prefix:
            content = _fact_prefix + content
        entry = {
            "content": content,
            "category": f.get("category", "general"),
            "tags": f.get("tags", ""),
        }
        if _add_fact(agent, entry["content"], entry["category"], entry["tags"]):
            saved_facts.append(entry)
            saved_previews.append(_preview(content))

    n_saved = len(saved_facts)
    if saved_previews:
        items = " \u00b7 ".join(f'"{p}"' for p in saved_previews[:3])
        if len(saved_previews) > 3:
            items += f" \u00b7 +{len(saved_previews) - 3} more"
        _safe_bg_print(f"  \U0001f4be Fact extraction: saved {n_saved} fact(s): {items}", agent)
    elif n_saved > 0:
        _safe_bg_print(
            f"  \u26a0\ufe0f Fact extraction: saved {n_saved}/{len(new_facts)}"
            f" ({len(new_facts) - n_saved} failed)", agent
        )
    else:
        _safe_bg_print(
            f"  \u26a0\ufe0f Fact extraction: couldn't save {len(new_facts)} fact(s)", agent
        )
    return saved_facts


# ---------------------------------------------------------------------------
# Stage 2: Fact scoring
# ---------------------------------------------------------------------------

def _run_fact_scoring(
    agent: Any,
    conversation_text: str,
) -> list[dict]:
    try:
        facts = _get_recently_retrieved_facts(agent)
    except Exception as e:
        _safe_bg_print(f"  \u2757 Fact scoring: get_facts failed: {e}", agent)
        return []
    if not facts:
        _safe_bg_print("  \U0001f4be Fact scoring: no facts to score", agent)
        return []

    try:
        prefetched_text = "\n".join(
            f"  [{f.get('fact_id', '?')}] (trust={float(f.get('trust_score', 0.5) or 0.5):.2f}) {f.get('content', '')}"
            for f in facts
        )

        _is_ap = os.environ.get("HERMES_APPLYPILOT_MODE") == "1"
        if _is_ap:
            _jc = ""
            try:
                _jc = agent._config.get("plugins", {}).get("hermes-memory-store", {}).get("prefetch_query_override", "")
            except Exception:
                pass
            _prompt_name = "_APPLYPILOT_SCORING_PROMPT"
            filled = _APPLYPILOT_SCORING_PROMPT.replace(
                "{conversation}", conversation_text
            ).replace("{prefetched}", prefetched_text
            ).replace("{job_context}", _jc)
        else:
            _prompt_name = "_FACT_SCORING_PROMPT"
            filled = _FACT_SCORING_PROMPT.replace(
                "{conversation}", conversation_text
            ).replace("{prefetched}", prefetched_text)
        try:
            _plog = os.path.expanduser("~/.hermes/holographic_debug.log")
            with open(_plog, "a", encoding="utf-8") as _pf:
                _pf.write(json.dumps({
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "event": "fact_scoring_prompt_selected",
                    "prompt_name": _prompt_name,
                    "applypilot_mode": _is_ap,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
    except Exception as e:
        _safe_bg_print(f"  \u2757 Fact scoring: prompt build failed: {e}", agent)
        return []

    raw = _call_llm(agent, filled)
    if not raw:
        _safe_bg_print("  \u2757 Fact scoring: LLM call failed", agent)
        return []

    result = _try_parse_json(raw)
    if not result:
        _safe_bg_print("  \u2757 Fact scoring: couldn't parse response", agent)
        return []

    evaluations = result.get("fact_evaluations", [])
    if not evaluations:
        _safe_bg_print("  \U0001f4be Fact scoring: nothing to evaluate", agent)
        return []

    # ── Single SQLite connection for all scoring operations ──────────────
    # Using one connection avoids "database is locked" errors from multiple
    # concurrent SQLite connections (one per _apply_feedback call).
    import sqlite3 as _sq
    from pathlib import Path
    _sc_conn = None
    try:
        _sc_conn = _sq.connect(str(Path(_get_db_path(agent))), timeout=10.0)

        # Increment retrieval_count for all facts that were evaluated
        all_ids = [f["fact_id"] for f in facts if isinstance(f, dict) and f.get("fact_id")]
        if all_ids:
            _sc_conn.execute(
                f"UPDATE facts SET retrieval_count = retrieval_count + 1, "
                f"updated_at = CURRENT_TIMESTAMP "
                f"WHERE fact_id IN ({','.join('?' * len(all_ids))})",
                all_ids,
            )
            _sc_conn.commit()

        scored_results: list[dict] = []
        scored_lines = []
        for ev in evaluations:
            fact_id = ev.get("fact_id")
            useful = ev.get("useful", False)
            if not isinstance(fact_id, int):
                continue
            trust_delta = float(ev.get("trust_delta", 0.0) or 0.0)
            fact_content = ""
            prev_trust = 0.5
            for f in facts:
                if isinstance(f, dict) and f.get("fact_id") == fact_id:
                    fact_content = f.get("content", "")
                    prev_trust = float(f.get("trust_score", 0.5) or 0.5)
                    break
            if _apply_feedback(agent, fact_id, useful, trust_delta, conn=_sc_conn):
                clamped = max(-0.05, min(0.05, trust_delta))
                new_trust = min(1.0, max(0.0, prev_trust + clamped))
                scored_results.append({
                    "fact_id": fact_id,
                    "useful": useful,
                    "trust_delta": clamped,
                    "prev_trust": round(prev_trust, 4),
                    "new_trust": round(new_trust, 4),
                    "content": fact_content,
                })
                direction = "\u2191" if clamped > 0 else "\u2193" if clamped < 0 else "\u2192"
                scored_lines.append(
                    f"#{fact_id} {direction} {prev_trust:.2f}\u2192{new_trust:.2f}"
                    f" ({_preview(fact_content, 40)})"
                )
    finally:
        if _sc_conn is not None:
            try:
                _sc_conn.close()
            except Exception:
                pass

    n_scored = len(scored_results)
    if scored_lines:
        items = " \u00b7 ".join(scored_lines[:5])
        if len(scored_lines) > 5:
            items += f" \u00b7 +{len(scored_lines) - 5} more"
        _safe_bg_print(f"  \U0001f4be Fact scoring: {n_scored} fact(s): {items}", agent)
    elif n_scored > 0:
        _safe_bg_print(
            f"  \u26a0\ufe0f Fact scoring: scored {n_scored}/{len(evaluations)}"
            f" ({len(evaluations) - n_scored} failed)", agent
        )
    else:
        _safe_bg_print(
            f"  \u26a0\ufe0f Fact scoring: couldn't score {len(evaluations)} fact(s)", agent
        )

    # ── Log training datum to separate training.db ───────────────────
    prefetch_query = getattr(agent, "_last_prefetch_query", "") or conversation_text
    session_id = getattr(agent, "session_id", "")
    try:
        import sqlite3
        from pathlib import Path
        train_conn = sqlite3.connect(
            str(Path.home() / ".hermes" / "training.db"), timeout=5.0
        )
        train_conn.execute(
            "CREATE TABLE IF NOT EXISTS training_events ("
            "event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "query_text TEXT NOT NULL, "
            "fact_id INTEGER NOT NULL, "
            "trust_delta REAL NOT NULL, "
            "session_id TEXT DEFAULT '', "
            "logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        for ev in evaluations:
            fid = ev.get("fact_id")
            if not isinstance(fid, int):
                continue
            td = round(float(ev.get("trust_delta", 0.0) or 0.0), 5)
            train_conn.execute(
                "INSERT INTO training_events (query_text, fact_id, trust_delta, session_id) "
                "VALUES (?, ?, ?, ?)",
                (prefetch_query, fid, td, session_id),
            )
        train_conn.commit()
        train_conn.close()
        _safe_bg_print(
            f"  \U0001f4be Training database: logged {len(evaluations)} fact evaluation(s)", agent
        )
    except Exception:
        pass  # non-critical logging

    return scored_results


# ── Turn debug log ─────────────────────────────────────────────────────────────

def _append_turn_log(
    agent: Any = None,
    conversation_text: str = "",
    extracted_facts: list | None = None,
    scored_facts: list | None = None,
) -> None:
    """Append a structured turn record to ~/.hermes/holographic_debug.log.

    JSON-lines format — one line per turn. Captures:
    - The prefetch query (previous thinking + response + user message)
    - ALL facts injected (full content, not just top-5 preview)
    - Extracted facts that were saved
    - Scored facts with trust deltas applied
    """
    import json
    from datetime import datetime

    prefetched = getattr(agent, "_last_prefetch_facts", []) if agent else []
    # Fallback: check provider's _last_raw_results directly
    if not prefetched and agent:
        try:
            for _p in getattr(agent._memory_manager, "providers", []):
                if hasattr(_p, "_last_raw_results") and _p._last_raw_results:
                    prefetched = _p._last_raw_results
                    break
        except Exception:
            pass

    record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": getattr(agent, "session_id", "") if agent else "",
        "prefetch_query": getattr(agent, "_last_prefetch_query", "") if agent else "",
        "conversation": conversation_text,
        "prefetched_facts": prefetched,
        "extracted_facts": extracted_facts or [],
        "scored_facts": scored_facts or [],
    }

    log_path = os.path.expanduser("~/.hermes/holographic_debug.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.debug("Failed to write turn log: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _run_review_in_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    prompt: str = "",
    spinner_message: str = "memory review",
) -> None:
    try:
        if not agent._memory_manager:
            return

        conversation_text = _build_conversation_text(messages_snapshot)

        # Run extraction and scoring in parallel
        import threading as _t
        _results: dict = {"extracted_facts": [], "scored_facts": []}

        def _do_extract():
            _safe_bg_print("  [tool] extracting facts...", agent)
            _results["extracted_facts"] = _run_fact_extraction(agent, conversation_text)

        def _do_score():
            _safe_bg_print("  [tool] scoring facts...", agent)
            _results["scored_facts"] = _run_fact_scoring(agent, conversation_text)

        _t1 = _t.Thread(target=_do_extract, daemon=False)
        _t2 = _t.Thread(target=_do_score, daemon=False)
        _t1.start()
        _t2.start()
        _t1.join()
        _t2.join()

        # Log the complete turn after both threads finish
        _append_turn_log(
            agent=agent,
            conversation_text=conversation_text,
            extracted_facts=_results.get("extracted_facts", []),
            scored_facts=_results.get("scored_facts", []),
        )

    except Exception as e:
        logger.error("Background review thread crashed: %s", e, exc_info=True)
        _safe_bg_print(f"  \u2757 Self-improvement review: crashed ({e})", agent)

    # Flush stdout to ensure any buffered output is written
    try:
        os.write(1, b"")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spawn_background_review_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    review_memory: bool = False,
    review_skills: bool = False,
) -> tuple:
    def _target() -> None:
        _run_review_in_thread(agent, messages_snapshot)
    return _target, _FACT_EXTRACTION_PROMPT


def spawn_fact_evaluation_thread(
    agent: Any,
    messages_snapshot: List[Dict],
) -> tuple:
    def _target() -> None:
        _run_review_in_thread(agent, messages_snapshot)
    return _target, _FACT_SCORING_PROMPT


def summarize_background_review_actions(
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
) -> List[str]:
    return []


def build_memory_write_metadata(
    agent: Any,
    *,
    write_origin: Optional[str] = None,
    execution_context: Optional[str] = None,
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "write_origin": write_origin or getattr(agent, "_memory_write_origin", "assistant_tool"),
        "execution_context": (
            execution_context
            or getattr(agent, "_memory_write_context", "foreground")
        ),
        "session_id": agent.session_id or "",
        "parent_session_id": agent._parent_session_id or "",
        "platform": agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
        "tool_name": "memory",
    }
    if task_id:
        metadata["task_id"] = task_id
    if tool_call_id:
        metadata["tool_call_id"] = tool_call_id
    return {k: v for k, v in metadata.items() if v not in {None, ""}}


__all__ = [
    "_MEMORY_REVIEW_PROMPT",
    "_SKILL_REVIEW_PROMPT",
    "_COMBINED_REVIEW_PROMPT",
    "_FACT_EVAL_BASE_PROMPT",
    "spawn_background_review_thread",
    "spawn_fact_evaluation_thread",
    "summarize_background_review_actions",
    "build_memory_write_metadata",
]
