"""Combined fact extraction and scoring pipeline for applypilot.

Runs after each successful turn (via conversation_loop patch).
Makes TWO sequential LLM calls per turn:

1. Extract up to N new facts from the current turn about the
   company's application system (form quirks, login requirements, etc.)
2. Score the facts that were prefetched into this turn (helpful/unhelpful)

The two calls are separate so the model never confuses injected
prefetched facts for conversation content to extract from.

Uses the same provider/model as the main conversation via the parent
agent's runtime. If the main model was rate-limited or unavailable,
this pipeline does NOT fire (gated on ``final_response``).
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FACT_EXTRACTION_PROMPT = """{conversation_text}

TASK:
You are a fact extraction system for a job application bot.

Extract up to 5 new facts from the conversation above about the company's
application system. Focus on: form field quirks, login/credential
requirements, job portal type (Greenhouse/Lever/Workday/Ashby etc.),
SSO requirements, reCAPTCHA presence, auto-fill behavior, and any
blockers encountered.

Respond in JSON format only — no markdown, no explanation:
{
  "new_facts": [
    {"content": "...", "tags": "tag1, tag2", "category": "general"}
  ]
}
"""

_FACT_SCORING_PROMPT = """{conversation_text}

{prefetched_facts}

TASK:
You are a fact scoring system for a job application bot.

Score each prefetched fact above as HELPFUL or NOT HELPFUL based on
whether it was relevant to this application. A fact is HELPFUL if it
accurately describes the form, portal, or process the bot encountered.
A fact is NOT HELPFUL if it was irrelevant, misleading, or for a
different company/site entirely.

Respond in JSON format only — no markdown, no explanation:
{
  "fact_scores": [
    {"fact_id": 1, "helpful": true, "trust_delta": 0.1}
  ]
}
"""


def _parse_llm_response(raw: Optional[str]) -> Optional[dict]:
    """Parse JSON from LLM output, finding {…} boundaries.

    Handles extra text around the JSON (prefix like "Here is the JSON:",
    suffix like "This is all I found", and markdown ```json fences).
    """
    if not raw:
        return None
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    text = text[start:end+1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_fact_pipeline_target(
    agent: Any,
    current_turn_msgs: List[Dict[str, Any]],
) -> Any:
    """Build a target function for the fact pipeline thread.

    Makes two sequential LLM calls:
      1. Fact extraction (only conversation, no prefetched facts shown)
      2. Fact scoring (conversation + prefetched facts for evaluation)

    Returns a callable (no return value) that runs the pipeline.
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

            runtime = agent._current_main_runtime()

            # Use the model's configured temperature/top_p when available
            # (e.g. Nemotron 3 Super uses temp=0.6, top_p=0.95).
            # Default to 0.0 for deterministic extraction/scoring when no
            # model-specific override is set.
            model_temperature = runtime.get("temperature", 0.0)
            model_top_p = runtime.get("top_p")
            extra_body = {"top_p": model_top_p} if model_top_p is not None else None

            # ── 2. EXTRACTION: one LLM call, conversation only ─────────────
            extract_prompt = _FACT_EXTRACTION_PROMPT.format(
                conversation_text=conversation_text,
            )
            extract_response = call_llm(
                task="fact_extraction",
                messages=[{"role": "user", "content": extract_prompt}],
                max_tokens=4096,
                temperature=model_temperature,
                extra_body=extra_body,
                timeout=60,
                main_runtime=runtime,
                provider=runtime.get("provider"),
                model=runtime.get("model"),
            )
            extract_raw = extract_response.choices[0].message.content
            extract_result = _parse_llm_response(extract_raw)

            # ── 3. Store new facts ─────────────────────────────────────────
            mm = getattr(agent, "_memory_manager", None)
            new_facts_saved: list[dict] = []
            if mm and extract_result:
                for fact in extract_result.get("new_facts", []):
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

            # ── 4. Get prefetched facts from agent cache ───────────────────
            prefetched = getattr(agent, "_last_prefetch_facts", [])
            if not prefetched:
                logger.debug("Fact pipeline: no prefetched facts to score")
                # Still log extraction results
                try:
                    from agent.background_review import _append_turn_log
                except ImportError:
                    _append_turn_log = lambda **kw: None
                _append_turn_log(
                    agent=agent,
                    conversation_text=conversation_text,
                    extracted_facts=new_facts_saved,
                    scored_facts=[],
                )
                return

            prefetched_text = json.dumps(prefetched, indent=2)

            # ── 5. SCORING: separate LLM call with prefetched facts ───────
            score_prompt = _FACT_SCORING_PROMPT.format(
                conversation_text=conversation_text,
                prefetched_facts=prefetched_text,
            )
            score_response = call_llm(
                task="fact_scoring",
                messages=[{"role": "user", "content": score_prompt}],
                max_tokens=4096,
                temperature=model_temperature,
                extra_body=extra_body,
                timeout=60,
                main_runtime=runtime,
                provider=runtime.get("provider"),
                model=runtime.get("model"),
            )
            score_raw = score_response.choices[0].message.content
            score_result = _parse_llm_response(score_raw)

            # ── 6. Apply fact feedback scores ──────────────────────────────
            fact_scores_applied: list[dict] = []
            if mm and score_result:
                for score in score_result.get("fact_scores", []):
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

            # ── 7. Log the complete turn ──────────────────────────────────
            try:
                from agent.background_review import _append_turn_log
            except ImportError:
                _append_turn_log = lambda **kw: None
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
