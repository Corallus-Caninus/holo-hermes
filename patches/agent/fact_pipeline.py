"""Combined fact extraction and scoring pipeline for applypilot.

Runs after each successful turn (via conversation_loop patch).
Makes ONE LLM call to:

1. Extract up to N new facts from the current conversation about the
   company's application system (form quirks, login requirements, etc.)
2. Score the facts that were prefetched into this turn (helpful/unhelpful)

Uses the same provider/model as the main conversation via the parent
agent's runtime. If the main model was rate-limited or unavailable,
this pipeline does NOT fire (gated on ``final_response``).
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FACT_PIPELINE_PROMPT = """{conversation_text}

{prefetched_facts}

TASK (read conversation above, then respond):
You are a fact extraction and scoring system for a job application bot.

1. EXTRACT up to 5 new facts from the conversation about the company's
   application system. Focus on: form field quirks, login/credential
   requirements, job portal type (Greenhouse/Lever/Workday/Ashby etc.),
   SSO requirements, reCAPTCHA presence, auto-fill behavior, and any
   blockers encountered.

2. SCORE each prefetched fact above as HELPFUL or NOT HELPFUL based on
   whether it was relevant to this application. A fact is HELPFUL if it
   accurately describes the form, portal, or process the bot encountered.
   A fact is NOT HELPFUL if it was irrelevant, misleading, or for a
   different company/site entirely.

Respond in JSON format only — no markdown, no explanation:
{{
  "new_facts": [
    {{"content": "Company X uses Greenhouse with Workday-style comboboxes for location. Form requires phone country code dropdown.", "tags": "greenhouse,companyname", "category": "general"}},
    {{"content": "Company X requires account creation with email verification before applying. Password policy: 8+ chars, symbol required.", "tags": "account,companyname", "category": "general"}}
  ],
  "fact_scores": [
    {{"fact_id": 1, "helpful": true, "trust_delta": 0.1}},
    {{"fact_id": 2, "helpful": false, "trust_delta": -0.1}}
  ]
}}
"""


def build_fact_pipeline_target(
    agent: Any,
    current_turn_msgs: List[Dict[str, Any]],
) -> Any:
    """Build a target function for the fact pipeline thread.

    Returns a callable (no return value) that runs the pipeline:
    1. Formats the conversation + prefetched facts
    2. Calls the LLM (same provider/model as main turn)
    3. Persists extracted facts via ``fact_store(action='add')``
    4. Scores prefetched facts via ``fact_feedback(helpful/unhelpful)``
    """
    from agent.auxiliary_client import call_llm

    def _target() -> None:
        try:
            # ── 1. Format conversation text ────────────────────────────────
            conv_lines = []
            for m in current_turn_msgs:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                reasoning = m.get("reasoning", "")
                if reasoning:
                    content = f"[Reasoning] {reasoning}\n\n{content}"
                if content:
                    conv_lines.append(f"[{role.upper()}]\n{content}")
            conversation_text = "\n\n".join(conv_lines)

            # ── 2. Get prefetched facts from agent cache ───────────────────
            prefetched = getattr(agent, "_last_prefetch_facts", [])
            prefetched_text = json.dumps(prefetched, indent=2) if prefetched else "None"

            # ── 3. Build prompt and call LLM ────────────────────────────────
            # Send the full current turn. The model's context limit handles
            # any truncation — no need to pre-truncate here.
            prompt = _FACT_PIPELINE_PROMPT.format(
                conversation_text=conversation_text,
                prefetched_facts=prefetched_text,
            )

            runtime = agent._current_main_runtime()
            response = call_llm(
                task="fact_pipeline",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=0.0,
                timeout=60,
                main_runtime=runtime,
                provider=runtime.get("provider"),
                model=runtime.get("model"),
            )

            raw = response.choices[0].message.content
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                raw = raw.rsplit("\n", 1)[0] if "\n" in raw else raw
                raw = raw.strip("`").strip()
            result = json.loads(raw)

            # ── 4. Store new facts ─────────────────────────────────────────
            mm = getattr(agent, "_memory_manager", None)
            if mm is None:
                logger.debug("Fact pipeline: no memory manager, skipping persistence")
                return

            new_facts_saved: list[dict] = []
            for fact in result.get("new_facts", []):
                try:
                    mm.handle_tool_call("fact_store", {
                        "action": "add",
                        "content": fact.get("content", ""),
                        "tags": fact.get("tags", ""),
                        "category": fact.get("category", "general"),
                    })
                    new_facts_saved.append({
                        "content": fact.get("content", ""),
                        "tags": fact.get("tags", ""),
                        "category": fact.get("category", "general"),
                    })
                except Exception as e:
                    logger.warning("Fact pipeline: failed to store fact: %s", e)

            # ── 5. Score prefetched facts ──────────────────────────────────
            fact_scores_applied: list[dict] = []
            for score in result.get("fact_scores", []):
                try:
                    fid = score.get("fact_id")
                    if fid is None:
                        continue
                    helpful = score.get("helpful", True)
                    delta = abs(score.get("trust_delta", 0.1))
                    action = "helpful" if helpful else "unhelpful"
                    mm.handle_tool_call("fact_feedback", {
                        "action": action,
                        "fact_id": fid,
                        "trust_delta": delta,
                    })
                    fact_scores_applied.append({
                        "fact_id": fid,
                        "useful": helpful,
                        "trust_delta": delta,
                    })
                except Exception as e:
                    logger.warning(
                        "Fact pipeline: failed to score fact %s: %s",
                        score.get("fact_id"), e,
                    )

            # ── 6. Log the complete turn ───────────────────────────────────
            try:
                from agent.background_review import _append_turn_log
            except ImportError:
                _append_turn_log = lambda **kw: None  # noqa: E731
            _append_turn_log(
                agent=agent,
                conversation_text=conversation_text,
                extracted_facts=new_facts_saved,
                scored_facts=fact_scores_applied,
            )

        except Exception as e:
            logger.warning("Fact pipeline failed: %s", e)
            logger.debug("Fact pipeline traceback", exc_info=True)

    return _target
