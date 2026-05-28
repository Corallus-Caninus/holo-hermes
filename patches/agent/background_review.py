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
    "You are a memory-extraction assistant for an AI agent.\n\n"
    "Analyze the conversation turn below and extract durable facts.\n\n"
    "### Conversation Turn\n"
    "{conversation}\n\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    '  "new_facts": [\n'
    '    {"content": "fact text", "category": "user_pref|project|tool|general", "tags": "tag1, tag2"}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Check the assistant's [thinking] block too — that often contains\n"
    "  user preferences, project decisions, and environment details that\n"
    "  didn't make it into the final response.\n"
    "- Extract any: user preferences, project decisions, environment details,\n"
    "  tool info, architecture choices, or other durable information.\n"
    "- Empty list if nothing to save (rare -- even small signals count).\n"
    "- No markdown, no explanation -- just the JSON object."
)

_FACT_SCORING_PROMPT = (
    "You are a fact-scoring assistant for an AI agent.\n\n"
    "Evaluate which of the memory facts below were useful for this turn.\n\n"
    "### Conversation Turn\n"
    "{conversation}\n\n"
    "### Memory Facts\n"
    "{prefetched}\n\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    '  "fact_evaluations": [\n'
    '    {"fact_id": N, "useful": true|false, "trust_delta": 0.0}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Look in the assistant's [thinking] block for explicit mentions of\n"
    "  specific facts (e.g. 'from fact #47: ...', 'per holographic memory: ...').\n"
    "  Those facts were clearly useful.\n"
    "- Also check whether the assistant's response draws on a fact's content\n"
    "  even if the fact ID isn't explicitly named.\n"
    "- A fact is useful if it informed the assistant's response or was contextually relevant.\n"
    "- A fact is not useful if it was irrelevant to the current turn.\n"
    "- `trust_delta` is how much the trust score should change:\n"
    "  * Positive if useful (+0.001 to +0.05) — higher = more useful\n"
    "  * Negative if not useful (-0.001 to -0.05) — lower = more irrelevant\n"
    "  * 0.0 if neutral / no strong signal\n"
    "  * Any granularity is fine (0.005, 0.01, 0.03, 0.05, etc.)\n"
    "  * Max magnitude is 0.05 in either direction\n"
    "- Include ALL fact IDs listed above.\n"
    "- No markdown, no explanation -- just the JSON object."
)

_COMBINED_REVIEW_PROMPT = _FACT_EXTRACTION_PROMPT + "\n\n" + _FACT_SCORING_PROMPT
_FACT_EVAL_BASE_PROMPT = _FACT_SCORING_PROMPT


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

    Gets fact IDs from the holographic provider's _last_prefetch_facts
    (set by prefetch() during the conversation turn), then queries the
    SQLite DB for their full content and scores.
    Returns empty list if prefetch tracking is unavailable.
    """
    # Collect prefetched fact IDs from the memory provider
    prefetched_ids: set = set()
    try:
        for prov in (agent._memory_manager.providers if agent._memory_manager else []):
            try:
                if hasattr(prov, "_last_prefetch_facts") and prov._last_prefetch_facts:
                    for f in prov._last_prefetch_facts:
                        if isinstance(f, dict) and f.get("fact_id"):
                            prefetched_ids.add(f["fact_id"])
            except Exception:
                pass
    except Exception:
        pass

    if not prefetched_ids:
        # Fallback: no prefetch tracking available (Nix store plugin).
        # Score the 20 most recently updated facts as best-effort.
        facts = []
        try:
            from hermes_constants import get_hermes_home
            import sqlite3
            db_path = get_hermes_home() / "memory_store.db"
            if db_path.exists():
                conn2 = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                try:
                    rows2 = conn2.execute(
                        "SELECT fact_id, content, retrieval_count, trust_score, updated_at "
                        "FROM facts ORDER BY updated_at DESC LIMIT 20"
                    ).fetchall()
                    for row in rows2:
                        facts.append({
                            "fact_id": row[0], "content": row[1],
                            "retrieval_count": row[2], "trust_score": row[3],
                            "updated_at": str(row[4]),
                        })
                finally:
                    conn2.close()
        except Exception:
            pass
        return facts

    # Fetch full data for those IDs from the DB
    facts = []
    try:
        from hermes_constants import get_hermes_home
        import sqlite3
        db_path = get_hermes_home() / "memory_store.db"
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
    for prefix in ["```json\n", "```\n", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Failed to parse review JSON: %s", text[:500])
        return None


def _add_fact(agent: Any, content: str, category: str = "general", tags: str = "") -> bool:
    if not agent._memory_manager:
        return False
    try:
        agent._memory_manager.handle_tool_call("fact_store", {
            "action": "add", "content": content, "category": category, "tags": tags,
        })
        return True
    except Exception as e:
        logger.debug("Failed to add fact: %s", e)
        return False


def _apply_feedback(agent: Any, fact_id: int, useful: bool, trust_delta: float = 0.0) -> bool:
    """Update trust score and helpful_count directly in SQLite.

    Bypasses the Nix store's handle_tool_call which applies hardcoded
    deltas (±0.05/-0.10) and ignores trust_delta entirely.  Writes
    the exact delta requested, clamped to ±0.05.

    Raises RuntimeError on failure — no silent fallback.
    """
    import sqlite3
    from hermes_constants import get_hermes_home
    db_path = get_hermes_home() / "memory_store.db"
    delta = max(-0.05, min(0.05, trust_delta))
    conn = sqlite3.connect(str(db_path), timeout=8.0)
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
        conn.rollback()
        raise
    finally:
        conn.close()
    return True


# ---------------------------------------------------------------------------
# Stage 1: Fact extraction
# ---------------------------------------------------------------------------

def _run_fact_extraction(
    agent: Any,
    conversation_text: str,
) -> int:
    try:
        filled = _FACT_EXTRACTION_PROMPT.replace("{conversation}", conversation_text)
    except Exception as e:
        logger.warning("Fact extraction prompt format failed: %s", e)
        return 0

    raw = _call_llm(agent, filled)
    if not raw:
        _safe_bg_print("  \u2757 Fact extraction: LLM call failed", agent)
        return 0

    result = _try_parse_json(raw)
    if not result:
        _safe_bg_print("  \u2757 Fact extraction: couldn't parse response", agent)
        return 0

    new_facts = result.get("new_facts", [])
    if not new_facts:
        _safe_bg_print("  \U0001f4be Fact extraction: nothing to save", agent)
        return 0

    ok = 0
    saved_previews = []
    for f in new_facts:
        content = f.get("content", "").strip()
        if len(content) < 10:
            continue
        if _add_fact(agent, content, f.get("category", "general"), f.get("tags", "")):
            ok += 1
            saved_previews.append(_preview(content))

    if saved_previews:
        items = " \u00b7 ".join(f'"{p}"' for p in saved_previews[:3])
        if len(saved_previews) > 3:
            items += f" \u00b7 +{len(saved_previews) - 3} more"
        _safe_bg_print(f"  \U0001f4be Fact extraction: saved {ok} fact(s): {items}", agent)
    elif ok > 0:
        _safe_bg_print(
            f"  \u26a0\ufe0f Fact extraction: saved {ok}/{len(new_facts)}"
            f" ({len(new_facts) - ok} failed)", agent
        )
    else:
        _safe_bg_print(
            f"  \u26a0\ufe0f Fact extraction: couldn't save {len(new_facts)} fact(s)", agent
        )
    return ok


# ---------------------------------------------------------------------------
# Stage 2: Fact scoring
# ---------------------------------------------------------------------------

def _run_fact_scoring(
    agent: Any,
    conversation_text: str,
) -> int:
    try:
        facts = _get_recently_retrieved_facts(agent)
    except Exception as e:
        _safe_bg_print(f"  \u2757 Fact scoring: get_facts failed: {e}", agent)
        return 0
    if not facts:
        _safe_bg_print("  \U0001f4be Fact scoring: no facts to score", agent)
        return 0

    try:
        prefetched_text = "\n".join(
            f"  [{f.get('fact_id', '?')}] (trust={float(f.get('trust_score', 0.5) or 0.5):.2f}) {f.get('content', '')}"
            for f in facts
        )

        filled = _FACT_SCORING_PROMPT.replace(
            "{conversation}", conversation_text
        ).replace("{prefetched}", prefetched_text)
    except Exception as e:
        _safe_bg_print(f"  \u2757 Fact scoring: prompt build failed: {e}", agent)
        return 0

    raw = _call_llm(agent, filled)
    if not raw:
        _safe_bg_print("  \u2757 Fact scoring: LLM call failed", agent)
        return 0

    result = _try_parse_json(raw)
    if not result:
        _safe_bg_print("  \u2757 Fact scoring: couldn't parse response", agent)
        return 0

    evaluations = result.get("fact_evaluations", [])
    if not evaluations:
        _safe_bg_print("  \U0001f4be Fact scoring: nothing to evaluate", agent)
        return 0

    # Increment retrieval_count for all facts that were evaluated,
    # regardless of whether the LLM chose to score each one.
    try:
        all_ids = [f["fact_id"] for f in facts if isinstance(f, dict) and f.get("fact_id")]
        if all_ids:
            import sqlite3 as _sq
            from hermes_constants import get_hermes_home
            _rc = _sq.connect(str(get_hermes_home() / "memory_store.db"))
            try:
                _rc.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1, "
                    f"updated_at = CURRENT_TIMESTAMP "
                    f"WHERE fact_id IN ({','.join('?' * len(all_ids))})",
                    all_ids,
                )
                _rc.commit()
            finally:
                _rc.close()
    except Exception:
        pass

    try:
        ok = 0
        scored_lines = []
        for ev in evaluations:
            fact_id = ev.get("fact_id")
            useful = ev.get("useful", False)
            if not isinstance(fact_id, int):
                continue
            # Extract trust_delta from LLM response, default 0.0
            trust_delta = float(ev.get("trust_delta", 0.0) or 0.0)
            fact_content = ""
            prev_trust = 0.5
            for f in facts:
                if isinstance(f, dict) and f.get("fact_id") == fact_id:
                    fact_content = f.get("content", "")
                    prev_trust = float(f.get("trust_score", 0.5) or 0.5)
                    break
            if _apply_feedback(agent, fact_id, useful, trust_delta):
                ok += 1
                clamped = max(-0.05, min(0.05, trust_delta))
                new_trust = min(1.0, max(0.0, prev_trust + clamped))
                direction = "\u2191" if clamped > 0 else "\u2193" if clamped < 0 else "\u2192"
                scored_lines.append(
                    f"#{fact_id} {direction} {prev_trust:.2f}\u2192{new_trust:.2f}"
                    f" ({_preview(fact_content, 40)})"
                )
    except Exception as e:
        _safe_bg_print(f"  \u2757 Fact scoring: apply failed: {e}", agent)
        return ok or 0

    if scored_lines:
        items = " \u00b7 ".join(scored_lines[:5])
        if len(scored_lines) > 5:
            items += f" \u00b7 +{len(scored_lines) - 5} more"
        _safe_bg_print(f"  \U0001f4be Fact scoring: {ok} fact(s): {items}", agent)
    elif ok > 0:
        _safe_bg_print(
            f"  \u26a0\ufe0f Fact scoring: scored {ok}/{len(evaluations)}"
            f" ({len(evaluations) - ok} failed)", agent
        )
    else:
        _safe_bg_print(
            f"  \u26a0\ufe0f Fact scoring: couldn't score {len(evaluations)} fact(s)", agent
        )
    return ok


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
        _results = {"extraction": 0, "scoring": 0}

        def _do_extract():
            _safe_bg_print("  [tool] extracting facts...", agent)
            _results["extraction"] = _run_fact_extraction(agent, conversation_text)

        def _do_score():
            _safe_bg_print("  [tool] scoring facts...", agent)
            _results["scoring"] = _run_fact_scoring(agent, conversation_text)

        _t1 = _t.Thread(target=_do_extract, daemon=False)
        _t2 = _t.Thread(target=_do_score, daemon=False)
        _t1.start()
        _t2.start()
        _t1.join()
        _t2.join()

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
