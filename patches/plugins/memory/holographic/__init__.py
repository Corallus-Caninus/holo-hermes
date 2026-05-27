"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

PATCHED: system_prompt_block() now returns top-5 highest-trust facts inline
in the cached system prompt, replacing the old MEMORY.md built-in store.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Extend __path__ to include the real Nix-store holographic plugin dir so
# relative imports (from .store, from .retrieval) resolve correctly.
for _p in sys.path:
    if "patches" in str(_p):
        continue
    _candidate = Path(_p) / "plugins" / "memory" / "holographic"
    if _candidate.is_dir() and (_candidate / "__init__.py").is_file():
        _rv = _candidate.resolve()
        if _rv != Path(__file__).parent.resolve():
            __path__ = [str(Path(__file__).parent), str(_rv)]
            break

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "WHEN TO SEARCH (call this proactively, don't wait for your prefetched memory):\n"
        "• The user mentions a project, tool, or topic from a past session — "
        "search before assuming you remember correctly.\n"
        "• You're about to make a decision or recommendation — check if the "
        "user has already chosen a direction or expressed a preference.\n"
        "• The user gives vague or incomplete instructions — past context may fill gaps.\n"
        "• A task is stalling or failing — the answer may be in a past decision.\n"
        "• The user seems frustrated or corrects your approach — check what "
        "preferences or decisions exist about the current task.\n"
        "• You recognize an entity (person, project, system name) — probe it.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink.\n\n"
        "Optionally specify ``trust_delta`` (-0.20 to 0.20) to control how much the trust "
        "score changes — the sign determines direction (positive = increase, negative = "
        "decrease). Useful for nuanced ratings. If omitted, defaults to ±0.10."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"],
                       "description": "'helpful' increases helpful_count, 'unhelpful' decreases helpful_count"},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
            "trust_delta": {"type": "number",
                            "description": "Optional: trust change (-0.20 to 0.20). "
                                           "Positive = increase trust, negative = decrease. "
                                           "Magnitude clamped to 0.20. Defaults to ±0.10."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))
        # Auto-feedback: tracks last prefetched facts for automatic scoring
        self._last_prefetch_facts: List[Dict[str, Any]] = []

    # Trust adjustment constants for fact_feedback tool
    _FEEDBACK_TRUST_DELTA = 0.10  # default magnitude when model doesn't specify one
    _FEEDBACK_TRUST_MAX_DELTA = 0.20  # maximum allowed magnitude (model can choose -0.20 to 0.20)

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        """Return memory status banner for the cached system prompt.

        No facts are listed inline — all retrieval happens per-turn via
        the prefetch mechanism (up to 20 relevant facts injected into
        ephemeral_system_prompt based on the current user query).
        """
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0

        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )

        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Score-thresholded prefetch with a char budget — best scoring facts,
        then fill up to ~1k chars with lower-scoring valid facts if needed.

        Keeps the ``_min_score`` relevance filter but removes the strict
        count cap: if the top 20 scoring facts don't add up to ~1000 chars,
        we keep pulling more valid facts until the budget is met.  This
        guarantees a minimum amount of context is injected each turn.

        Also tracks fact_ids for auto-feedback (sync_turn) and increments
        retrieval_count for all returned facts.
        """
        _MIN_FACTS = 20
        _TARGET_CHARS = 5000
        _FETCH_LIMIT = 300
        _MIN_SCORE = 0.15

        # Clear previous tracking
        self._last_prefetch_facts = []
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=_FETCH_LIMIT)
            if not results:
                return ""

            # Score threshold: keep only relevant facts
            filtered = [r for r in results if r.get("score", 0) >= _MIN_SCORE]
            if not filtered:
                return ""

            # Accumulate by char budget — walk score-descending until we've
            # filled ~5k chars or run out of valid facts.
            selected_facts = []
            running_chars = 0
            for r in filtered:
                trust = r.get("trust_score", r.get("trust", 0))
                line = f"- [{trust:.1f}] {r.get('content', '')}"
                line_chars = len(line)
                # +1 for the \n separator before this line (if not first)
                sep = 1 if selected_facts else 0
                # Always take at least _MIN_FACTS, then enforce char budget
                if len(selected_facts) >= _MIN_FACTS and running_chars + sep + line_chars > _TARGET_CHARS:
                    break  # budget exhausted, the floor of _MIN_FACTS is met
                selected_facts.append((r, line))
                running_chars += sep + line_chars

            if not selected_facts:
                return ""

            # Track fact_ids for auto-feedback
            self._last_prefetch_facts = [
                {"fact_id": r["fact_id"], "content": r.get("content", "")}
                for r, _ in selected_facts if r.get("fact_id")
            ]

            # Increment retrieval_count for all returned facts
            if self._last_prefetch_facts and self._store:
                try:
                    ids = [f["fact_id"] for f in self._last_prefetch_facts]
                    placeholders = ",".join("?" * len(ids))
                    self._store._conn.execute(
                        f"UPDATE facts SET retrieval_count = retrieval_count + 1, updated_at = CURRENT_TIMESTAMP WHERE fact_id IN ({placeholders})",
                        ids,
                    )
                    self._store._conn.commit()
                except Exception:
                    pass  # Non-critical — don't break the prompt

            lines = [line for _, line in selected_facts]
            return (
                "## Relevant Memory Context\n"
                f"(showing {len(selected_facts)} relevant facts — review before responding; "
                "use fact_store(probe/search) if you need deeper details on any of these)\n"
                + "\n".join(lines)
            )
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Save prefetched fact data for background review evaluation.

        Writes a JSON evaluation context to ``.pending_fact_eval.json`` in
        HERMES_HOME so the background review process (which runs as a separate
        agent) can evaluate which facts were useful and adjust trust scores.

        Includes the user message and assistant response from this turn so the
        review agent can judge fact relevance against the actual conversation.

        The background review agent uses ``fact_feedback()`` to apply results.
        retrieval_count is already handled by prefetch() automatically.
        """
        if not self._last_prefetch_facts or not self._store:
            return

        try:
            from hermes_constants import get_hermes_home
            eval_path = get_hermes_home() / ".pending_fact_eval.json"
            data = {
                "session_id": session_id or getattr(self, "_session_id", ""),
                "user_content": user_content or "",
                "assistant_content": assistant_content or "",
                "prefetched_facts": [
                    {"fact_id": f["fact_id"], "content": f["content"]}
                    for f in self._last_prefetch_facts
                    if f.get("fact_id") and f.get("content")
                ],
            }
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass  # Non-critical — evaluation is optional

        self._last_prefetch_facts = []

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._config.get("auto_extract", False):
            return
        if not self._store or not messages:
            return
        self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            # Lazy-init guard: if initialize() failed or was never called,
            # try again on first use so transient failures (e.g. database
            # lock at boot) don't permanently break the session.
            if self._store is None:
                logger.warning(
                    "fact_store called but _store is None — "
                    "attempting lazy initialize (session_id=%s)",
                    getattr(self, "_session_id", "unknown"),
                )
                try:
                    self.initialize(
                        session_id=getattr(self, "_session_id", "unknown"),
                    )
                except Exception as _lazy_err:
                    logger.error("lazy fact_store initialize failed: %s", _lazy_err)
                    return tool_error(f"fact_store backend not available: {_lazy_err}")
                if self._store is None:
                    return tool_error("fact_store backend failed to initialize")
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                # Include a content preview so background review summaries
                # show what was stored, not just the ID.
                _preview = args["content"].strip()[:80]
                if len(args["content"].strip()) > 80:
                    _preview += "..."
                return json.dumps({
                    "fact_id": fact_id,
                    "status": "added",
                    "preview": _preview,
                })

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        """Record fact feedback with configurable trust delta and helpful_count.

        ``trust_delta`` can be any value from -_FEEDBACK_TRUST_MAX_DELTA to
        +_FEEDBACK_TRUST_MAX_DELTA (default: ±0.10). The sign determines the
        direction — positive increases trust, negative decreases it.
        Magnitude is clamped to _FEEDBACK_TRUST_MAX_DELTA.

        ``action`` only controls helpful_count:
          helpful=True  → helpful_count += 1
          helpful=False → helpful_count -= 1
        """
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"

            row = self._store._conn.execute(
                "SELECT trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                return tool_error(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]

            # Resolve trust_delta: model can specify -MAX to +MAX.
            # Clamp magnitude, preserve sign.
            raw_delta = args.get("trust_delta")
            if raw_delta is not None:
                raw = float(raw_delta)
                magnitude = min(abs(raw), self._FEEDBACK_TRUST_MAX_DELTA)
                delta = magnitude if raw >= 0 else -magnitude
            else:
                delta = self._FEEDBACK_TRUST_DELTA if helpful else -self._FEEDBACK_TRUST_DELTA

            new_trust = max(0.0, min(1.0, old_trust + delta))

            helpful_delta = 1 if helpful else -1
            new_helpful = max(0, row["helpful_count"] + helpful_delta)

            self._store._conn.execute(
                "UPDATE facts SET trust_score=?, helpful_count=?, updated_at=CURRENT_TIMESTAMP WHERE fact_id=?",
                (new_trust, new_helpful, fact_id),
            )
            self._store._conn.commit()

            return json.dumps({
                "fact_id": fact_id,
                "status": "updated",
                "old_trust": round(old_trust, 2),
                "new_trust": round(new_trust, 2),
                "trust_delta_applied": round(delta, 2),
                "helpful_count": new_helpful,
            })
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="user_pref")
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="project")
                        extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


# ---------------------------------------------------------------------------
# Module-level cleanup: deregister the dead built-in memory tool
# The holographic patch strips the MemoryStore init (agent._memory_store = None),
# so calling the `memory` tool always errors. Remove it from the registry
# entirely so the model never sees it in its tool definitions. Also clean up
# _AGENT_LOOP_TOOLS in model_tools to keep dispatch tables consistent.
# ---------------------------------------------------------------------------
try:
    from tools.registry import registry
    registry.deregister("memory")
    import model_tools
    model_tools._AGENT_LOOP_TOOLS.discard("memory")
except Exception:
    pass  # Non-critical — best-effort cleanup

# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
