"""Patched HolographicMemoryProvider — FAISS + HRR + compositional retrieval.
Augments the holographic provider with:
- **FAISS** vector index for semantic search across all facts (bypasses FTS5 keyword bottleneck)
- **Compositional** retrieval via probe/reason for entity-linked facts
- Entity extraction from query text to drive probe/reason
- Merged three-axis prefetch: FAISS ∪ FTS5 ∪ compositional → scored → top 5
- Backfill HRR vectors for existing facts (one-time at init)
- Incremental FAISS updates on new facts
Debug logging: set HOLOGRAPHIC_DEBUG=1 env var.
"""
import importlib.util
import json
import logging
import os as _os
import sys
import threading as _threading
from pathlib import Path
from typing import Any
# ── BERT embedding for FAISS semantic search ──────────────────────────────────
from . import bert_embed as _bert

# Current bert_vec encoding version. Bump when the instruct prefix changes.
# v5: IndexFlatIP exact search (more robust than HNSW for noisy queries)
# v4: shared "Query:" prefix for both encode_queries and encode_passages
# v3: different "Query:" vs "Passage:" prefixes (BROKEN — subspace mismatch)
_BERT_VEC_VERSION = 5
# ── Debug logging (toggle via HOLOGRAPHIC_DEBUG=1) ────────────────────────────
_log = logging.getLogger("plugins.memory.holographic.patch")
_DEBUG = _os.environ.get("HOLOGRAPHIC_DEBUG", "").lower() in ("1", "true", "yes")
if _DEBUG:
    _log.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, force=True)
def _dbg(msg: str, *args) -> None:
    if _DEBUG:
        _log.info("PATCH: " + msg, *args)
def _warn(msg: str, *args) -> None:
    _log.warning("PATCH: " + msg, *args)
def _dbgn(msg, *args):
    """Debug logging with automatic % formatting — no-op when _DEBUG is False."""
    if _DEBUG:
        _log.info("PATCH: " + (msg % args if args else msg))
# ═══════════════════════════════════════════════════════════════════════════════
# 0. Make numpy + faiss available before loading the real module
# ═══════════════════════════════════════════════════════════════════════════════
import os as _os
import site as _site
_dbg("Step 0: making numpy + faiss available")
# Set LD_LIBRARY_PATH so pip-installed numpy/faiss/torch C extensions can find
# Nix system libraries (zlib, libstdc++, zstd).
_ZLIB_LIB = "/nix/store/vpg96mfr1jw5arlqg831i69g29v0sdb3-zlib-1.3.1/lib"
_GCC_LIB = "/nix/store/mhd0rk497xm0xnip7262xdw9bylvzh99-gcc-13.3.0-lib/lib"
_ZSTD_LIB = "/nix/store/6cf2yj12gf51jn5vdbdw01gmgvyj431s-zstd-1.5.6/lib"
_os.environ.setdefault("LD_LIBRARY_PATH", "")
for _lp in [_ZLIB_LIB, _GCC_LIB, _ZSTD_LIB]:
    if _os.path.isdir(_lp) and _lp not in _os.environ["LD_LIBRARY_PATH"]:
        _os.environ["LD_LIBRARY_PATH"] = _lp + ":" + _os.environ["LD_LIBRARY_PATH"]
# Add faiss-venv Python 3.12 path + Nix numpy (NOT applypilot venv — its
# 3.11 packages like regex conflict with transformers imports).
_faiss_path = "/tmp/faiss-venv/lib/python3.12/site-packages"
# IMPORTANT: add Nix numpy path AFTER faiss-venv below so the Nix one wins
_numpy_path = "/nix/store/d6x7mb4fbhms6mshya1x0qp9s37wv08q-python3.12-numpy-2.4.4/lib/python3.12/site-packages"
# First add faiss-venv (so faiss is findable), then OVERRIDE with Nix numpy
# so numpy._core._multiarray_umath resolves from the Nix store.
for _p in [_faiss_path, _numpy_path]:
    if _os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        _dbg("added path: %s", _p)
    elif _os.path.isdir(_p):
        # Already in path — move it to front by removing & re-inserting
        sys.path.remove(_p)
        sys.path.insert(0, _p)
        _dbg("moved to front: %s", _p)
    else:
        _dbg("path not found: %s", _p)
# Try to import numpy and faiss now — the real module checks _HAS_NUMPY at import time
_HAS_FAISS = False
try:
    import numpy as _np  # noqa: F401
    _dbg("numpy %s imported OK from %s", _np.__version__, _np.__file__)
    # Now try faiss once numpy is available
    try:
        import faiss as _faiss  # noqa: F401
        _HAS_FAISS = True
        # Use all available CPU cores for FAISS vector operations
        # (brute-force IndexFlatIP inner product parallelizes via OpenMP)
        _ncpu = _os.cpu_count() or 4
        _faiss.omp_set_num_threads(_ncpu)
        _dbg("faiss imported OK, omp_set_num_threads=%d", _ncpu)
    except ImportError as _e:
        _dbg("faiss not available: %s", _e)
except ImportError as _e:
    _warn("numpy not available: %s", _e)
except Exception as _e:
    _warn("numpy import failed: %s", _e)
_dbg("_HAS_NUMPY=%s, _HAS_FAISS=%s", 
     str(globals().get("_HAS_NUMPY", "UNSET")), _HAS_FAISS)
# ═══════════════════════════════════════════════════════════════════════════════
# 1. Find and load the real module
# ═══════════════════════════════════════════════════════════════════════════════
_dbg("Step 1: finding real module")
_real_path: Path | None = None
_dbgn("sys.path entries:")
for _p in sys.path:
    _dbgn("  %s", _p)
    _candidate = Path(_p) / "plugins" / "memory" / "holographic" / "__init__.py"
    if _candidate.exists() and "patches" not in str(_candidate):
        _real_path = _candidate
        _dbgn("  -> found at %s", _candidate)
        break
if _real_path is None:
    msg = "Could not locate real plugins/memory/holographic/__init__.py"
    _warn(msg)
    raise ImportError(msg)
_dbg("real module found at: %s", _real_path)
_spec = importlib.util.spec_from_file_location(
    "plugins.memory.holographic_real", str(_real_path)
)
_real_mod = importlib.util.module_from_spec(_spec)
sys.modules["plugins.memory.holographic_real"] = _real_mod
# Extend __path__ to include the REAL holographic module's directory so
# submodule imports (holographic.py, store.py, retrieval.py) resolve from
# the Nix store. Without this, ``from plugins.memory.holographic.holographic
# import HolographicMemoryProvider`` inside the real __init__.py fails
# because our patch directory doesn't contain those submodules.
_our_dir = Path(__file__).parent
_real_holographic_dir = _real_path.parent
__path__ = [str(_our_dir), str(_real_holographic_dir)]
_dbg("__path__ set to: %s", __path__)
_spec.loader.exec_module(_real_mod)
_dbg("real module loaded OK, HolographicMemoryProvider=%s",
     hasattr(_real_mod, "HolographicMemoryProvider"))
# ═══════════════════════════════════════════════════════════════════════════════
# 2. Re-export everything
# ═══════════════════════════════════════════════════════════════════════════════
for _attr in dir(_real_mod):
    if _attr.startswith("_") and _attr not in ("__all__",):
        continue
    globals()[_attr] = getattr(_real_mod, _attr)
_dbg("step 2: re-exported %d names from real module",
     sum(1 for _a in dir(_real_mod) if not _a.startswith("_") or _a in ("__all__",)))
# ── Imports from holographic submodules ───────────────────────────────────────
# Import from the real Nix store path via spec for reliability.
_dbg("step 2b: importing HRR functions")
_hrr_path = str(Path(__file__).parent / "holographic.py")
if not _os.path.isfile(_hrr_path):
    _hrr_path = str(_real_holographic_dir / "holographic.py")
_dbg("loading HRR module from: %s", _hrr_path)
_hrr_spec = importlib.util.spec_from_file_location("plugins.memory.holographic._hrr_mod", _hrr_path)
_hrr_mod = importlib.util.module_from_spec(_hrr_spec)
sys.modules["plugins.memory.holographic._hrr_mod"] = _hrr_mod
_hrr_spec.loader.exec_module(_hrr_mod)
encode_text = _hrr_mod.encode_text
similarity = _hrr_mod.similarity
bytes_to_phases = _hrr_mod.bytes_to_phases
phases_to_bytes = _hrr_mod.phases_to_bytes
encode_atom = _hrr_mod.encode_atom
bind = _hrr_mod.bind
unbind = _hrr_mod.unbind
_dbg("HRR functions imported: _HAS_NUMPY=%s", getattr(_hrr_mod, "_HAS_NUMPY", "UNSET"))
# ═══════════════════════════════════════════════════════════════════════════════
# 3. FAISS index wrapper
# ═══════════════════════════════════════════════════════════════════════════════
class FaissIndex:
    """Thin FAISS wrapper for cosine-similarity vector search over facts.
    Uses IndexFlatIP (exact brute-force inner product) for 100% recall.
    At 22K x 1536 dims with OpenMP parallelization, search is ~1-2ms
    on a Ryzen 3600 — fast enough to justify exact search over HNSW
    approximation, especially for noisy multi-turn queries.
    """
    def __init__(self, dim: int = 1024, m: int = 32, ef_construction: int = 200, ef_search: int = 200):
        import numpy as np
        self.dim = dim
        self.index = None
        self.id_map: list[int] = []
        self._np = np
    def _make_flatip(self) -> "faiss.IndexFlatIP":
        """Create an exact brute-force inner-product index.
        FlatIP + L2-normalized vectors = cosine similarity, 100% recall."""
        import faiss
        return faiss.IndexFlatIP(self.dim)
    def build(self, vectors: list, fact_ids: list[int]) -> None:
        """Build index from list of vector arrays and matching fact_ids.
        Vectors can be numpy arrays (float32) or None entries (skipped).
        Validates each vector before adding to the index.
        """
        import faiss
        import numpy as _np2
        valid = []
        skipped = 0
        for fid, v in zip(fact_ids, vectors):
            if v is None:
                skipped += 1
                continue
            if not isinstance(v, _np2.ndarray):
                skipped += 1
                continue
            if v.ndim != 1 or v.shape[0] != self.dim:
                skipped += 1
                continue
            valid.append((fid, v))
        if not valid:
            _warn("FaissIndex.build: no valid vectors (%d skipped)", skipped)
            self.index = None
            self.id_map = []
            return
        self.id_map = [v[0] for v in valid]
        arr = self._np.array([v[1] for v in valid], dtype=self._np.float32)
        faiss.normalize_L2(arr)
        self.index = self._make_flatip()
        self.index.add(arr)
        _dbg("FaissIndex.build: %d vectors (%d skipped)", len(valid), skipped)
    def search(self, query_vec, k: int = 10) -> list[tuple[int, float]]:
        """Return top-K (fact_id, faiss_distance) pairs for a query phase vector.
        
        FAISS returns inner-product distances (cosine sim after L2 norm, 
        range -1 to 1). Higher = more similar.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        import faiss
        q = self._np.asarray(query_vec, dtype=self._np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, min(k, self.index.ntotal))
        return [(self.id_map[i], float(distances[0][j])) 
                for j, i in enumerate(indices[0])]
    def add(self, vector, fact_id: int) -> None:
        """Incrementally add a single new vector."""
        import faiss
        if self.index is None:
            if vector is not None:
                arr = self._np.array([vector], dtype=self._np.float32)
                faiss.normalize_L2(arr)
                self.index = self._make_flatip()
                self.index.add(arr)
                self.id_map = [fact_id]
            return
        if vector is None:
            return
        arr = self._np.array([vector], dtype=self._np.float32)
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.id_map.append(fact_id)
# ═══════════════════════════════════════════════════════════════════════════════
# 4. Patched HolographicMemoryProvider
# ═══════════════════════════════════════════════════════════════════════════════
_HolographicMemoryProvider = getattr(_real_mod, "HolographicMemoryProvider", None)
if _HolographicMemoryProvider is not None:
    _dbg("HolographicMemoryProvider found, applying patches")
    _orig_init = _HolographicMemoryProvider.__init__
    _orig_initialize = _HolographicMemoryProvider.initialize
    _orig_prefetch = _HolographicMemoryProvider.prefetch
    def _patched_init(self, config: dict | None = None) -> None:
        """Initialize provider — store is created later in initialize()."""
        _orig_init(self, config)
        _dbg("_patched_init called, config keys: %s", list(self._config.keys()) if self._config else "NONE")
        # Log config keys unconditionally for ApplyPilot debugging
        try:
            import json as _ic
            _ilog = _os.path.expanduser("~/.hermes/holographic_debug.log")
            with open(_ilog, "a", encoding="utf-8") as _if:
                _if.write(_ic.dumps({
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "event": "patched_init_config",
                    "config_keys": list(self._config.keys()) if self._config else "NONE",
                    "has_prefetch_override": "prefetch_query_override" in (self._config or {}),
                    "has_memory_section": "memory" in (self._config or {}),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # ── FAISS index ────────────────────────────────────────────────────
        self._faiss_index = None
        self._has_faiss = _HAS_FAISS
        self._faiss_dim = _bert.get_dim()  # BERT embedding dim (384)
        _dbg("_has_faiss=%s, _faiss_dim=%s", self._has_faiss, self._faiss_dim)
    def _patched_initialize(self, session_id: str, **kwargs) -> None:
        """Initialize store, backfill HRR vectors, build FAISS index."""
        _orig_initialize(self, session_id, **kwargs)
        _dbg("_patched_initialize called, store=%s _hrr_available=%s",
             self._store is not None,
             getattr(getattr(self, "_store", None), "_hrr_available", "N/A"))
        # ── Enable WAL mode for concurrent reads/writes ──────────────────────
        # Without WAL, background threads (fact scoring, extraction) that open
        # their own connections get "database is locked" errors when the main
        # thread holds a write transaction.
        if self._store:
            try:
                self._store._conn.execute("PRAGMA journal_mode=WAL")
                self._store._conn.commit()
                _dbg("WAL mode enabled on memory_store.db")
            except Exception as _e:
                _warn("Failed to enable WAL mode: %s", _e)
        # ── Ensure row factory for column-name access ────────────────────────
        # All our patches access rows by name (e.g. row["bert_vec"]).
        # Without this, bare except: blocks silently swallow TypeError from
        # tuple indexing with strings, causing empty FAISS rebuilds and
        # zero facts being prefetched.
        if self._store:
            try:
                import sqlite3 as _sq3
                self._store._conn.row_factory = _sq3.Row
            except Exception as _e:
                _warn("Failed to set row factory: %s", _e)
        # ── Backfill HRR vectors for facts that lack them ───────────────────
        if self._store and getattr(self._store, "_hrr_available", False):
            try:
                null_count = self._store._conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE hrr_vector IS NULL"
                ).fetchone()[0]
                _dbg("backfill: %d facts with NULL hrr_vector", null_count)
                if null_count > 0:
                    rows = self._store._conn.execute(
                        "SELECT fact_id, content FROM facts WHERE hrr_vector IS NULL"
                    ).fetchall()
                    _dbg("backfill: processing %d facts", len(rows))
                    for i, row in enumerate(rows):
                        self._store._compute_hrr_vector(row["fact_id"], row["content"])
                        if _DEBUG and i % 1000 == 0 and i > 0:
                            _dbg("backfill: %d/%d done", i, len(rows))
                    # Rebuild memory banks after backfill
                    cats = self._store._conn.execute(
                        "SELECT DISTINCT category FROM facts"
                    ).fetchall()
                    for (cat,) in cats:
                        self._store._rebuild_bank(cat)
                    _dbg("backfill complete: %d vectors computed", null_count)
            except Exception as _e:
                _warn("backfill failed: %s", _e)
        # ── Backfill content_vec for facts that lack it ──────────────────────
        if self._store:
            try:
                self._store._conn.execute(
                    "ALTER TABLE facts ADD COLUMN content_vec BLOB"
                )
                self._store._conn.commit()
            except Exception:
                pass  # column already exists
            try:
                cv_null = self._store._conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE content_vec IS NULL"
                ).fetchone()[0]
                _dbg("content_vec backfill: %d facts NULL", cv_null)
                if cv_null > 0:
                    rows = self._store._conn.execute(
                        "SELECT fact_id, content FROM facts WHERE content_vec IS NULL"
                    ).fetchall()
                    _dbg("content_vec: processing %d facts", len(rows))
                    _hrr_d = getattr(self._store, "hrr_dim", 1024)
                    for i, row in enumerate(rows):
                        cv = phases_to_bytes(encode_text(row["content"], _hrr_d))
                        self._store._conn.execute(
                            "UPDATE facts SET content_vec = ? WHERE fact_id = ?",
                            (cv, row["fact_id"]),
                        )
                        if _DEBUG and i % 1000 == 0 and i > 0:
                            _dbg("content_vec: %d/%d done", i, len(rows))
                    self._store._conn.commit()
                    _dbg("content_vec backfill complete: %d vectors computed", cv_null)
            except Exception as _e:
                _warn("content_vec backfill failed: %s", _e)
        # ── Backfill bert_vec for FAISS semantic search ───────────────────────
        if self._has_faiss and self._store:
            # Log diagnostic
            try:
                import json as _dj
                with open(_os.path.expanduser("~/.hermes/holographic_debug.log"), "a") as _df:
                    _df.write(_dj.dumps({
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "event": "faiss_init",
                        "_has_faiss": self._has_faiss,
                        "has_store": self._store is not None,
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
            try:
                self._store._conn.execute(
                    "ALTER TABLE facts ADD COLUMN bert_vec BLOB"
                )
                self._store._conn.commit()
            except Exception:
                pass  # column already exists
            # ── Stale vector detection: version mismatch ───────────────────
            # BERT_VEC_VERSION = 4 means "shared Query: prefix for both
            # queries and passages" (encode_passages now uses Query: just
            # like encode_queries). Older vectors used "Passage:" prefix
            # which put stored facts in a different subspace — FAISS
            # comparisons between queries and passages gave cos~0.0.
            # When the version in faiss_state is < _BERT_VEC_VERSION, ALL existing
            # bert_vec are stale and must be cleared before backfill.
            try:
                _old_ver = self._store._conn.execute(
                    "SELECT COALESCE(version, 0) FROM faiss_state WHERE id = 1"
                ).fetchone()
                _stale_ver = _old_ver is not None and _old_ver[0] < _BERT_VEC_VERSION
            except Exception:
                _stale_ver = False
            if _stale_ver:
                _dbg("bert_vec version %d < %d — stale, rebuilding FAISS index only", _old_ver[0] if _old_ver else 0, _BERT_VEC_VERSION)
                # Only clear the FAISS cache — NOT bert_vec!
                # The bert vectors are pure embeddings valid for any index type.
                # Clearing them forces a full 22K re-encode which takes 20+ min
                # and silently fails (OOM/interrupt).
                self._store._conn.execute("DELETE FROM faiss_state WHERE id = 1")
                self._store._conn.commit()
            # Also ensure the version column exists (migration from v3 -> v4)
            try:
                self._store._conn.execute("ALTER TABLE faiss_state ADD COLUMN bert_vec_version INTEGER DEFAULT 0")
                self._store._conn.commit()
            except Exception:
                pass
            try:
                bv_null = self._store._conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE bert_vec IS NULL"
                    " OR LENGTH(bert_vec) < 6000"  # old 384-dim vecs are 1536 bytes
                ).fetchone()[0]
                _dbg("bert_vec backfill: %d facts NULL", bv_null)
                if bv_null > 0:
                    rows = self._store._conn.execute(
                        "SELECT fact_id, content FROM facts WHERE bert_vec IS NULL"
                    ).fetchall()
                    _dbg("bert_vec: processing %d facts in batches", len(rows))
                    _batch_size = 100
                    _offset = 0
                    while _offset < len(rows):
                        batch = rows[_offset:_offset + _batch_size]
                        texts = [r["content"] for r in batch]
                        try:
                            vectors = _bert.encode_passages(texts)
                        except Exception as _e:
                            err_str = str(_e)
                            if "HIP out of memory" in err_str or "CUDA out of memory" in err_str:
                                # OOM — reduce batch and retry on CPU
                                _batch_size = max(1, _batch_size // 2)
                                _dbg("bert_vec: OOM at batch_size=%d, retrying on CPU",
                                     100 if _batch_size == 50 else _batch_size * 2)
                                # Fall back to CPU for this batch
                                import os as _bert_os
                                _bert_os.environ['CUDA_VISIBLE_DEVICES'] = ''
                                _bert._device = __import__('torch').device('cpu')
                                if _bert._model is not None:
                                    _bert._model = _bert._model.to('cpu')
                                continue
                            _warn("bert_vec: encode batch failed: %s", _e)
                            _offset += len(batch)
                            continue
                        for j, r in enumerate(batch):
                            self._store._conn.execute(
                                "UPDATE facts SET bert_vec = ? WHERE fact_id = ?",
                                (vectors[j].tobytes(), r["fact_id"]),
                            )
                        self._store._conn.commit()
                        _offset += len(batch)
                        if _DEBUG:
                            _dbg("bert_vec: %d/%d done", _offset, bv_null)
                    _dbg("bert_vec backfill complete: %d vectors", bv_null)
            except Exception as _e:
                _warn("bert_vec backfill failed: %s", _e)
        # ── Build FAISS index from BERT vectors ────────────────────────────
        if self._has_faiss and self._store:
            _dbg("rebuild_faiss_index: _has_faiss=True, calling...")
            try:
                self._rebuild_faiss_index()
                _dbg("FAISS index built: %d vectors in index",
                     self._faiss_index.index.ntotal if self._faiss_index and self._faiss_index.index else 0)
            except Exception as _e:
                _warn("FAISS index build failed: %s", _e)
                self._has_faiss = False
        else:
            _dbg("FAISS init SKIPPED: _has_faiss=%s, _store=%s",
                 self._has_faiss, self._store is not None)
        # ── Pre-load BERT embedding model on MI25 ───────────────────────────
        # Load at startup so the first turn doesn't pay the ~7s cold-start cost.
        if self._has_faiss and self._store:
            _dbg("Attempting BERT model load...")
            # Write diagnostic before loading
            try:
                import json as _jj
                with open(_os.path.expanduser("~/.hermes/holographic_debug.log"), "a") as _f:
                    _f.write(_jj.dumps({"timestamp": __import__("datetime").datetime.now().isoformat(),
                                        "event": "bert_load_start",
                                        "faiss_index_ok": self._faiss_index is not None and self._faiss_index.index is not None}) + "\n")
            except Exception:
                pass
            try:
                _bert._load()
                _dbg("BERT model loaded at startup")
                import json as _jj
                with open(_os.path.expanduser("~/.hermes/holographic_debug.log"), "a") as _f:
                    _f.write(_jj.dumps({"timestamp": __import__("datetime").datetime.now().isoformat(),
                                        "event": "bert_startup_load", "ok": True}) + "\n")
            except Exception as _e:
                _warn("BERT model load at startup failed: %s — will retry lazily", _e)
                # Log the actual error to debug log
                try:
                    import json as _jj
                    with open(_os.path.expanduser("~/.hermes/holographic_debug.log"), "a") as _f:
                        _f.write(_jj.dumps({"timestamp": __import__("datetime").datetime.now().isoformat(),
                                            "event": "bert_load_failed",
                                            "error": str(_e)[:200]}) + "\n")
                except Exception:
                    pass
        # ── Monkey-patch retriever.probe to use content_vec ──────────────
        if self._retriever and self._has_faiss:
            try:
                import types as _types
                _orig_probe = self._retriever.probe
                def _probe_with_cache(self_, entity, category=None, limit=10):
                    import time as _time
                    _t0 = _time.time()
                    conn = self_.store._conn
                    dim = self_.hrr_dim
                    role_entity = encode_atom("__hrr_role_entity__", dim)
                    entity_vec = encode_atom(entity.lower(), dim)
                    probe_key = bind(entity_vec, role_entity)
                    role_content = encode_atom("__hrr_role_content__", dim)
                    if category:
                        bank_row = conn.execute(
                            "SELECT vector FROM memory_banks WHERE bank_name = ?",
                            (f"cat:{category}",),
                        ).fetchone()
                        if bank_row:
                            bank_vec = bytes_to_phases(bank_row["vector"])
                            extracted = unbind(bank_vec, probe_key)
                            res = self_._score_facts_by_vector(
                                extracted, category=category, limit=limit
                            )
                            if _DEBUG:
                                _dbg("probe(%s) bank=%.1fms res=%d",
                                     entity, (_time.time() - _t0) * 1000, len(res))
                            return res
                    cache_key = category or "__all__"
                    _hit = hasattr(self_, "_probe_cache") and self_._probe_cache.get("key") == cache_key
                    if _hit:
                        # ── Incremental: append new facts since _max_fact_id ──
                        cache = self_._probe_cache
                        max_id = cache.get("_max_fact_id", 0)
                        where_extra = " AND hrr_vector IS NOT NULL AND content_vec IS NOT NULL"
                        extra_params: list = []
                        if category:
                            where_extra += " AND category = ?"
                            extra_params.append(category)
                        new_rows = conn.execute(
                            f"SELECT fact_id, trust_score, hrr_vector, content_vec "
                            f"FROM facts WHERE fact_id > ?{where_extra}",
                            [max_id] + extra_params
                        ).fetchall()
                        if new_rows:
                            import numpy as _np
                            n_new = len(new_rows)
                            hv_new = _np.empty((n_new, dim), dtype=_np.float64)
                            cv_new = _np.empty((n_new, dim), dtype=_np.float64)
                            ts_new = _np.empty(n_new, dtype=_np.float64)
                            fid_new = _np.empty(n_new, dtype=_np.int64)
                            new_max = max_id
                            for i, r in enumerate(new_rows):
                                hv_new[i] = bytes_to_phases(r["hrr_vector"])
                                cv_new[i] = bytes_to_phases(r["content_vec"])
                                ts_new[i] = r["trust_score"]
                                fid_new[i] = r["fact_id"]
                                if r["fact_id"] > new_max:
                                    new_max = r["fact_id"]
                            self_._probe_cache["hv"] = _np.concatenate([cache["hv"], hv_new])
                            self_._probe_cache["cv"] = _np.concatenate([cache["cv"], cv_new])
                            self_._probe_cache["ts"] = _np.concatenate([cache["ts"], ts_new])
                            self_._probe_cache["fid"] = _np.concatenate([cache["fid"], fid_new])
                            self_._probe_cache["n"] = cache["n"] + n_new
                            self_._probe_cache["_max_fact_id"] = new_max
                            if _DEBUG:
                                _dbg("probe(%s) incremental: +%d facts (total %d)",
                                     entity, n_new, self_._probe_cache["n"])
                    else:
                        if _DEBUG:
                            _dbg("probe(%s) BUILDING cache (had=%s key=%s)",
                                 entity, hasattr(self_, "_probe_cache"), cache_key)
                        where = "WHERE hrr_vector IS NOT NULL AND content_vec IS NOT NULL"
                        params = []
                        if category:
                            where += " AND category = ?"
                            params.append(category)
                        rows = conn.execute(
                            f"SELECT fact_id, trust_score, hrr_vector, content_vec "
                            f"FROM facts {where}", params
                        ).fetchall()
                        if not rows:
                            self_._probe_cache = {"key": cache_key, "n": 0, "_max_fact_id": 0}
                        else:
                            import numpy as _np
                            n = len(rows)
                            hv = _np.empty((n, dim), dtype=_np.float64)
                            cv = _np.empty((n, dim), dtype=_np.float64)
                            ts = _np.empty(n, dtype=_np.float64)
                            fid = _np.empty(n, dtype=_np.int64)
                            max_fid = 0
                            for i, r in enumerate(rows):
                                hv[i] = bytes_to_phases(r["hrr_vector"])
                                cv[i] = bytes_to_phases(r["content_vec"])
                                ts[i] = r["trust_score"]
                                fid[i] = r["fact_id"]
                                if r["fact_id"] > max_fid:
                                    max_fid = r["fact_id"]
                            self_._probe_cache = {
                                "key": cache_key, "n": n, "hv": hv, "cv": cv,
                                "ts": ts, "fid": fid, "_max_fact_id": max_fid,
                            }
                    cache = self_._probe_cache
                    if _DEBUG and _hit:
                        _dbg("probe(%s) CACHE HIT n=%d", entity, cache.get("n", 0))
                    n = cache.get("n", 0)
                    if n == 0:
                        res = self_.search(entity, category=category, limit=limit)
                        if _DEBUG:
                            _dbg("probe(%s) empty=%.1fms", entity, (_time.time() - _t0) * 1000)
                        return res
                    hv = cache["hv"]
                    cv = cache["cv"]
                    ts = cache["ts"]
                    fid = cache["fid"]
                    import numpy as _np
                    sims = _np.mean(
                        _np.cos(hv - cv - probe_key - role_content), axis=1
                    )
                    scores = (sims + 1.0) / 2.0 * ts
                    idx = _np.argsort(scores)[-limit:][::-1]
                    top_ids = tuple(int(fid[i]) for i in idx)
                    placeholders = ",".join("?" * len(top_ids))
                    rows = conn.execute(
                        f"SELECT * FROM facts WHERE fact_id IN ({placeholders})",
                        top_ids,
                    ).fetchall()
                    id_order = {fid: i for i, fid in enumerate(top_ids)}
                    rows.sort(key=lambda r: id_order.get(r["fact_id"], 9999))
                    res = []
                    for row, i in zip(rows, idx):
                        f = dict(row)
                        f["score"] = float(scores[i])
                        res.append(f)
                    if _DEBUG:
                        _dbg("probe(%s) brute=%.1fms res=%d/%d",
                             entity, (_time.time() - _t0) * 1000, len(res), n)
                    return res
                self._retriever.probe = _types.MethodType(
                    _probe_with_cache, self._retriever
                )
                _dbg("retriever.probe patched for content_vec")
                # ── Also patch retriever.search: hoist encode_text out of per-fact loop ──
                # The original calls encode_text(query) inside the for loop,
                # causing 30× redundant SHA-256 hashing of the full reasoning block.
                def _search_with_hoisted_encode(self_, query, category=None, min_trust=0.3, limit=10):
                    """Wrapper that encodes query once instead of once-per-fact."""
                    candidates = self_._fts_candidates(query, category, min_trust, limit * 3)
                    if not candidates:
                        return []
                    query_tokens = self_._tokenize(query)
                    query_vec = encode_text(query, self_.hrr_dim)  # ← once
                    scored = []
                    for fact in candidates:
                        content_tokens = self_._tokenize(fact["content"])
                        tag_tokens = self_._tokenize(fact.get("tags", ""))
                        all_tokens = content_tokens | tag_tokens
                        jaccard = self_._jaccard_similarity(query_tokens, all_tokens)
                        fts_score = fact.get("fts_rank", 0.0)
                        if self_.hrr_weight > 0 and fact.get("hrr_vector"):
                            fact_vec = bytes_to_phases(fact["hrr_vector"])
                            hrr_sim = (similarity(query_vec, fact_vec) + 1.0) / 2.0
                        else:
                            hrr_sim = 0.5
                        relevance = (self_.fts_weight * fts_score
                                    + self_.jaccard_weight * jaccard
                                    + self_.hrr_weight * hrr_sim)
                        score = relevance * fact["trust_score"]
                        if self_.half_life > 0:
                            score *= self_._temporal_decay(fact.get("updated_at") or fact.get("created_at"))
                        fact["score"] = score
                        scored.append(fact)
                    scored.sort(key=lambda x: x["score"], reverse=True)
                    results = scored[:limit]
                    for fact in results:
                        fact.pop("hrr_vector", None)
                    return results
                self._retriever.search = _types.MethodType(
                    _search_with_hoisted_encode, self._retriever
                )
                _dbg("retriever.search patched to hoist encode_text")
            except Exception as _e:
                _warn("retriever.probe or search patch failed: %s", _e)
    def _rebuild_faiss_index(self) -> None:
        """Load or build FAISS index, persisted in the faiss_state table."""
        if not self._store:
            return
        import faiss as _faiss
        # Ensure the cache table exists
        self._store._conn.execute(
            "CREATE TABLE IF NOT EXISTS faiss_state ("
            "id INTEGER PRIMARY KEY, "
            "index_blob BLOB, "
            "id_map_blob TEXT, "
            "fact_count INTEGER, "
            "version INTEGER DEFAULT 3, "
            "updated_at TEXT DEFAULT (datetime('now'))"
            ")"
        )
        self._store._conn.commit()
        # Add version column if table existed before (safe if already exists)
        try:
            self._store._conn.execute("ALTER TABLE faiss_state ADD COLUMN version INTEGER DEFAULT 2")
            self._store._conn.commit()
        except Exception:
            pass  # column already exists
        # Wipe any stale pre-v6 cache (version < 6, old HNSW)
        self._store._conn.execute(
"DELETE FROM faiss_state WHERE id = 1 AND (version IS NULL OR version < 6"
" OR LENGTH(index_blob) < 100)"
        )
        self._store._conn.commit()
        total_vectors = self._store._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE bert_vec IS NOT NULL"
        ).fetchone()[0]
        # Try cache
        cached = self._store._conn.execute(
            "SELECT index_blob, id_map_blob FROM faiss_state "
            "WHERE id = 1 AND fact_count = ?",
            (total_vectors,)
        ).fetchone()
        if cached:
            try:
                id_map = json.loads(cached["id_map_blob"])
                import numpy as _np
                self._faiss_index = FaissIndex(self._faiss_dim)
                self._faiss_index.index = _faiss.deserialize_index(
                    _np.frombuffer(cached["index_blob"], dtype=_np.uint8)
                )
                self._faiss_index.id_map = id_map
                _dbg("FAISS index restored from cache: %d vectors", len(id_map))
                # Log success
                try:
                    import json as _rj
                    with open(_os.path.expanduser("~/.hermes/holographic_debug.log"), "a") as _rf:
                        _rf.write(_rj.dumps({"timestamp": __import__("datetime").datetime.now().isoformat(),
                                             "event": "faiss_cache_load", "status": "ok",
                                             "ntotal": self._faiss_index.index.ntotal}) + "\n")
                except Exception:
                    pass
                return
            except Exception as _e:
                _dbg("FAISS cache load failed: %s — rebuilding", _e)
                try:
                    import json as _rj
                    with open(_os.path.expanduser("~/.hermes/holographic_debug.log"), "a") as _rf:
                        _rf.write(_rj.dumps({"timestamp": __import__("datetime").datetime.now().isoformat(),
                                             "event": "faiss_cache_load", "status": "fail",
                                             "error": str(_e)[:100]}) + "\n")
                except Exception:
                    pass
        # Rebuild from SQLite
        import numpy as _np
        # Use the provider's own store connection (points to the correct DB
        # path, including custom paths like ApplyPilot's separate DB).
        # Row factory was already set in _patched_initialize.
        try:
            rows = self._store._conn.execute(
                "SELECT fact_id, bert_vec FROM facts WHERE bert_vec IS NOT NULL"
            ).fetchall()
            _dbg("rebuild_faiss_index: %d rows with BERT vectors", len(rows))
            if not rows:
                self._faiss_index = FaissIndex(self._faiss_dim)
                return
            fact_ids = []
            vectors = []
            for row in rows:
                try:
                    vec = _np.frombuffer(row[1], dtype=_np.float32)
                    fact_ids.append(row[0])
                    vectors.append(vec)
                except Exception as _e:
                    _dbg("skipping BERT vector decode: %s", _e)
                    continue
        except Exception as _e:
            _dbg("rebuild_faiss_index: store connection failed: %s", _e)
            self._faiss_index = FaissIndex(self._faiss_dim)
            return
        self._faiss_index = FaissIndex(self._faiss_dim)
        self._faiss_index.build(vectors, fact_ids)
        # Log build result
        try:
            import json as _rj
            with open(_os.path.expanduser("~/.hermes/holographic_debug.log"), "a") as _rf:
                _rf.write(_rj.dumps({"timestamp": __import__("datetime").datetime.now().isoformat(),
                                     "event": "faiss_build_result",
                                     "n_vectors": len(vectors),
                                     "n_fact_ids": len(fact_ids),
                                     "index_is_none": self._faiss_index is None,
                                     "index_dot_index_is_none": self._faiss_index is not None and self._faiss_index.index is None}) + "\n")
        except Exception:
            pass
        # Persist to cache
        if self._faiss_index is not None and self._faiss_index.index is not None:
            try:
                blob = _faiss.serialize_index(self._faiss_index.index)
                id_map_json = json.dumps(self._faiss_index.id_map)
                self._store._conn.execute(
                    "INSERT OR REPLACE INTO faiss_state "
                    "(id, index_blob, id_map_blob, fact_count, version, updated_at) "
                    "VALUES (1, ?, ?, ?, 6, datetime('now'))",  # v6 = last-token pool + asymmetric prefix
                    (blob, id_map_json, total_vectors)
                )
                self._store._conn.commit()
                _dbg("FAISS index cached to DB: %d vectors", total_vectors)
            except Exception as _e:
                _dbg("FAISS cache write failed: %s", _e)
        else:
            _warn("FAISS index is None — skipping cache write")
    def _entities_from_query(self, query: str) -> list[str]:
        """Find known entities appearing in the query text."""
        if not self._store or not query:
            return []
        try:
            cur = self._store._conn.execute(
                "SELECT name, aliases FROM entities"
            )
            matches = []
            q_lower = query.lower()
            for name, aliases_json in cur:
                aliases = json.loads(aliases_json) if aliases_json else []
                all_names = [name] + (aliases if isinstance(aliases, list) else [])
                if any(n.lower() in q_lower for n in all_names if n):
                    matches.append(name)
            if _DEBUG and matches:
                _dbg("entities_from_query: found %s", matches)
            return matches
        except Exception:
            return []
    def _compositional_search(self, entities: list[str], limit: int = 5) -> list[dict]:
        """Probe/reason for matched entities, return deduplicated facts."""
        if not self._retriever or not entities:
            return []
        from concurrent.futures import ThreadPoolExecutor
        seen: set[int] = set()
        results: list[dict] = []
        all_fact_lists: list[list[dict]] = []
        # Probe cache is incrementally updated in _probe_with_cache — no need to clear
        try:
            first_result = self._retriever.probe(entities[0], limit=limit)
            all_fact_lists.append(first_result)
        except Exception:
            pass
        # Parallel probe for remaining entities
        remaining = entities[1:] if len(entities) > 1 else []
        if remaining:
            with ThreadPoolExecutor(max_workers=None) as pool:
                futures = [pool.submit(
                    lambda e=entity: self._retriever.probe(e, limit=limit)
                ) for entity in remaining]
                for f in futures:
                    try:
                        all_fact_lists.append(f.result())
                    except Exception:
                        all_fact_lists.append([])
        for facts in all_fact_lists:
            for f in facts:
                fid = f.get("fact_id")
                if fid is not None and fid not in seen:
                    seen.add(fid)
                    results.append(f)
        if len(entities) >= 2:
            try:
                facts = self._retriever.reason(entities, limit=limit)
                for f in facts:
                    fid = f.get("fact_id")
                    if fid is not None and fid not in seen:
                        seen.add(fid)
                        results.append(f)
            except Exception:
                pass
        if _DEBUG and results:
            _dbg("compositional_search: %d results from %d entities",
                 len(results), len(entities))
        return results
    def _faiss_search(self, query: str, k: int = 20) -> list[dict]:
        """FAISS semantic vector search via gte-Qwen2-1.5B on MI25.
        
        Uses learned embeddings for real semantic similarity (not HRR random
        projections). 1536-dim vectors with 32K context window.
        Returns facts with _faiss_distance set (cosine similarity, -1 to 1).
        """
        print(f"[holo] _faiss_search called: query={query[:80]!r} k={k}", flush=True)
        if not self._has_faiss or not self._faiss_index or not self._store:
            print(f"[holo] _faiss_search SKIPPED: _has_faiss={self._has_faiss} index={self._faiss_index is not None} store={self._store is not None}", flush=True)
            return []
            # Log why FAISS search was skipped
            try:
                import json as _jj
                _p = _os.path.expanduser("~/.hermes/holographic_debug.log")
                with open(_p, "a") as _f:
                    _f.write(_jj.dumps({"timestamp": __import__("datetime").datetime.now().isoformat(),
                                        "event": "faiss_search_skipped",
                                        "_has_faiss": self._has_faiss,
                                        "has_index": self._faiss_index is not None,
                                        "has_store": self._store is not None}) + "\n")
            except Exception:
                pass
            return []
        try:
            query_vec = _bert.encode_queries(query)  # "query: " prefix
            # Log which embedding instruction was used
            try:
                import json as _ei
                _elog = _os.path.expanduser("~/.hermes/holographic_debug.log")
                with open(_elog, "a", encoding="utf-8") as _ef:
                    _ef.write(_ei.dumps({
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "event": "embed_instruction_used",
                        "instruction": getattr(_bert, "_INSTRUCTION", ""),
                        "env_value": _os.environ.get("HERMES_EMBED_INSTRUCTION", ""),
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
            # Validate query vector — GPU memory fragmentation can produce
            # NaN/Inf/zero vectors that corrupt FAISS search.
            import numpy as _npv
            print(f"[holo] query_vec stats: shape={query_vec.shape} min={query_vec.min():.6f} max={query_vec.max():.6f} mean={query_vec.mean():.6f} nan={_npv.any(_npv.isnan(query_vec))} inf={_npv.any(_npv.isinf(query_vec))} zero={_npv.all(query_vec == 0)}", flush=True)
            # Embedding quality check: query vs same text as passage
            try:
                _qv_passage = _bert.encode_passages(query)
                _cos_same = float(_npv.dot(query_vec[0], _qv_passage[0]))
                # Also check determinism: encode twice
                _qv2 = _bert.encode_queries(query)
                _cos_det = float(_npv.dot(query_vec[0], _qv2[0]))
                print(f"[holo] embed quality: query_vs_passage(same_text)={_cos_same:.4f} deterministic={_cos_det:.4f}", flush=True)
            except Exception as _eq:
                print(f"[holo] embed quality check failed: {_eq}", flush=True)
            if _npv.any(_npv.isnan(query_vec)) or _npv.any(_npv.isinf(query_vec)) or _npv.all(query_vec == 0):
                _dbg("faiss_search: bad query vector (NaN/Inf/zero) — skipping")
                return []
            results = self._faiss_index.search(query_vec, k=k)
            print(f"[holo] FAISS search returned {len(results) if results else 0} results", flush=True)
            if not results:
                print(f"[holo] FAISS search empty — skipping post-processing", flush=True)
                return []
            fact_ids = [r[0] for r in results]
            distances = [r[1] for r in results]
            # Write marker: FAISS search returned results
            try:
                with open("/tmp/fact_injection_ok", "a") as _mf:
                    _mf.write(f"[{__import__('datetime').datetime.now().isoformat()[:19]}] "
                              f"FAISS_SEARCH_OK n_results={len(results)} query={query[:60]!r}\n")
            except Exception:
                pass
            placeholders = ",".join("?" for _ in fact_ids)
            rows = self._store._conn.execute(
                f"SELECT * FROM facts WHERE fact_id IN ({placeholders})",
                fact_ids,
            ).fetchall()
            id_order = {fid: i for i, fid in enumerate(fact_ids)}
            rows.sort(key=lambda r: id_order.get(r["fact_id"], 9999))
            facts = []
            for row, dist in zip(rows, distances):
                f = dict(row)
                f["_faiss_distance"] = dist
                facts.append(f)
            return facts
        except Exception as _faiss_err:
            import traceback as _ftb
            _emsg = f"FAISS SEARCH ERROR: {_faiss_err}\n{''.join(_ftb.format_exception(type(_faiss_err), _faiss_err, _faiss_err.__traceback__))}"
            print(f"[holo] {_emsg}", flush=True)
            try:
                with open("/tmp/fact_injection_ok", "a") as _mf:
                    _mf.write(f"[{__import__('datetime').datetime.now().isoformat()[:19]}] FAISS_ERROR: {_faiss_err}\n{''.join(_ftb.format_exception(type(_faiss_err), _faiss_err, _faiss_err.__traceback__))}\n")
            except Exception:
                pass
            return []
    def _patched_prefetch(self, query: str, *, session_id: str = "") -> str:
        """Three-axis prefetch: FAISS ∪ FTS5 ∪ compositional → scored → top 5.
        Merges candidates from all three retrievers, scores via the existing
        hybrid pipeline (Jaccard + HRR cos + trust), and stores raw results
        for the end-of-turn fact pipeline.
        """
        # ApplyPilot mode: use the prepackaged job-scoped query instead
        # of the raw user message (which is the entire system prompt).
        # The key is set under plugins.hermes-memory-store in the Hermes
        # config, which flows through to the provider's self._config.
        # Fall back to nested memory. path for backward compat.
        try:
            _override = self._config.get("prefetch_query_override", "")
            if not _override:
                _override = self._config.get("memory", {}).get("prefetch_query_override", "")
            if _override:
                query = _override
            # ── Log override diagnostic ──────────────────────────────────
            try:
                import json as _oj
                _olog = _os.path.expanduser("~/.hermes/holographic_debug.log")
                with open(_olog, "a", encoding="utf-8") as _of:
                    _of.write(_oj.dumps({
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "event": "prefetch_override_check",
                        "config_has_key": "prefetch_query_override" in (self._config or {}),
                        "config_has_memory_key": "prefetch_query_override" in (self._config or {}).get("memory", {}),
                        "config_keys": list(self._config.keys()) if self._config else "NONE",
                        "override_found": bool(_override),
                        "override_preview": (_override[:80] if _override else ""),
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
        except Exception as _oe:
            try:
                import json as _oj2
                _olog2 = _os.path.expanduser("~/.hermes/holographic_debug.log")
                with open(_olog2, "a", encoding="utf-8") as _of2:
                    _of2.write(_oj2.dumps({
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "event": "prefetch_override_error",
                        "error": str(_oe),
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
        # Also write to the marker file for quick inspection
        try:
            with open("/tmp/fact_injection_ok", "a") as _mf:
                _ts = __import__("datetime").datetime.now().isoformat()[:19]
                _match = "USING_OVERRIDE" if _override else "NO_OVERRIDE"
                _qv = query[:80].replace("\n", "\\n")
                _mf.write(f"[{_ts}] {_match} query_preview={_qv!r}\n")
        except Exception:
            pass
        # WRITE MARKER: prefetch was entered
        try:
            _ts = __import__("datetime").datetime.now().isoformat()[:19]
            with open("/tmp/fact_injection_ok", "w") as _mf:
                _mf.write(f"[{_ts}] PREFETCH_ENTERED query_preview={query[:100]!r}\n")
        except Exception:
            pass
        if not self._retriever:
            # Log diagnostic: why prefetch can't run
            try:
                import json as _dj
                _dlog = _os.path.expanduser("~/.hermes/holographic_debug.log")
                with open(_dlog, "a", encoding="utf-8") as _df:
                    _df.write(_dj.dumps({
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "event": "prefetch_skipped",
                        "reason": "no_retriever",
                        "has_store": self._store is not None,
                        "_min_trust": getattr(self, "_min_trust", "N/A"),
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
            return ""
        try:
            import time as _time
            _t0 = _time.time()
            # ── Axis 1: FAISS semantic vector search ──────────────────────────
            faiss_facts = self._faiss_search(query, k=20)
            _t1 = _time.time()
            # ── Axis 2: FTS5 keyword search (DISABLED) ────────────────────────
            fts5_facts: list = []
            # ── Axis 3: Compositional entity probe/reason (DISABLED) ─────────
            comp_facts: list = []
            _t2 = _time.time()
            # ── Merge + dedup by fact_id, tagging source ──────────────────────
            seen: set[int] = set()
            all_candidates: list[dict] = []
            for fact in faiss_facts:
                fid = fact.get("fact_id")
                if fid is not None and fid not in seen:
                    seen.add(fid)
                    fact["_source"] = "faiss"
                    all_candidates.append(fact)
            if _DEBUG:
                _dbg("prefetch: FAISS=%d merged=%d %.1fms",
                     len(faiss_facts), len(all_candidates),
                     (_t2 - _t0) * 1000)
            if not all_candidates:
                # Log diagnostic: all 3 axes returned nothing
                try:
                    import json as _dj
                    _dlog = _os.path.expanduser("~/.hermes/holographic_debug.log")
                    with open(_dlog, "a", encoding="utf-8") as _df:
                        _df.write(_dj.dumps({
                            "timestamp": __import__("datetime").datetime.now().isoformat(),
                            "event": "prefetch_empty",
                            "faiss_count": len(faiss_facts),
                            "fts5_count": len(fts5_facts),
                            "comp_count": len(comp_facts),
                            "entity_count": len(entities),
                            "_has_faiss": getattr(self, "_has_faiss", False),
                            "_faiss_index_ntotal": getattr(getattr(self, "_faiss_index", None), "index", None) is not None and getattr(self._faiss_index.index, "ntotal", 0) or 0,
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                self._last_raw_results = []
                return ""
            # ── Score candidates — source-aware ──────────────────────────────
            # FAISS facts: use FAISS cosine distance directly (already IS HRR
            # similarity), skip Jaccard/FTS5 which penalize vector matches.
            # FTS5 and compositional facts: full Jaccard + FTS5 rank + HRR cos.
            # All paths: trust_score + temporal_decay applied.
            query_tokens = self._retriever._tokenize(query)
            _cached_query_vec = encode_text(query, self._retriever.hrr_dim)
            scored = []
            for fact in all_candidates:
                trust = fact.get("trust_score", 0.5)
                source = fact.get("_source", "")
                if source == "faiss":
                    # FAISS inner product after L2 norm = cosine similarity, -1..1
                    relevance = fact.get("_faiss_distance", 0.0)
                    # Clamp to [0, 1] so it's comparable with other axes
                    relevance = max(0.0, min(1.0, (relevance + 1.0) / 2.0))
                else:
                    # Existing pipeline: Jaccard + FTS5 rank + HRR cos
                    content_tokens = self._retriever._tokenize(fact.get("content", ""))
                    tag_tokens = self._retriever._tokenize(fact.get("tags", ""))
                    all_tokens = content_tokens | tag_tokens
                    jaccard = self._retriever._jaccard_similarity(query_tokens, all_tokens)
                    fts_score = fact.get("fts_rank", 0.0)
                    if fact.get("hrr_vector"):
                        try:
                            fact_vec = bytes_to_phases(fact["hrr_vector"])
                            hrr_sim = (similarity(_cached_query_vec, fact_vec) + 1.0) / 2.0
                        except Exception:
                            hrr_sim = 0.5
                    else:
                        hrr_sim = 0.5
                    relevance = (
                        self._retriever.fts_weight * fts_score
                        + self._retriever.jaccard_weight * jaccard
                        + self._retriever.hrr_weight * hrr_sim
                    )
                score = relevance * trust
                if self._retriever.half_life > 0:
                    score *= self._retriever._temporal_decay(
                        fact.get("updated_at") or fact.get("created_at")
                    )
                fact["score"] = score
                scored.append(fact)
            scored.sort(key=lambda x: x["score"], reverse=True)
            scored = [s for s in scored if s.get("_faiss_distance", 0) > 0.1]
            top5 = scored[:5]
            _t4 = _time.time()
            # ── Log top-5 source breakdown ───────────────────────────────────
            _src_counts = {"faiss": 0, "fts5": 0, "comp": 0}
            for _f in top5:
                _s = _f.get("_source", "?")
                if _s in _src_counts:
                    _src_counts[_s] += 1
            _dbg("top5 sources: FAISS=%d | "
                "retrieve=%.1fms score=%.1fms total=%.1fms",
                _src_counts.get("faiss", 0),
                (_t2 - _t0) * 1000, (_t4 - _t2) * 1000, (_t4 - _t0) * 1000)
            # ── Store raw results for end-of-turn scoring ───────────────────
            self._last_raw_results = []
            source_breakdown = {"faiss": 0, "fts5": 0, "comp": 0}
            for r in top5:
                src = r.get("_source", "")
                if src in source_breakdown:
                    source_breakdown[src] += 1
                self._last_raw_results.append({
                    "fact_id": r.get("fact_id"),
                    "content": r.get("content", ""),
                    "trust_score": r.get("trust_score", 0.5),
                    "category": r.get("category", ""),
                    "tags": r.get("tags", ""),
                    "score": round(r.get("score", 0.0), 4),
                    "faiss_cos": round(r.get("_faiss_distance", 0.0), 4),
                    "source": src,
                })
            _dbg("sources: FAISS=%d",
                 source_breakdown.get("faiss", 0))
            # ── Format as == FACTS == section ───────────────────────────────
            lines = ["== FACTS =="]
            for r in top5:
                t = r.get("trust_score", 0.5)
                c = r.get("content", "")[:]
                lines.append(f"- [{t:.1f}] {c}")
            # Log retrieved facts for debugging
            try:
                import json as _fj
                _flog = _os.path.expanduser("~/.hermes/holographic_debug.log")
                with open(_flog, "a", encoding="utf-8") as _ff:
                    _ff.write(_fj.dumps({
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "event": "prefetch_facts_retrieved",
                        "n_facts": len(top5),
                        "query_preview": query[:200],
                        "facts": [{
                            "fact_id": r.get("fact_id"),
                            "content": r.get("content", "")[:120],
                            "faiss_cos": round(r.get("_faiss_distance", 0.0), 4),
                            "score": round(r.get("score", 0.0), 4),
                            "trust": r.get("trust_score", 0.5),
                        } for r in top5],
                    }, ensure_ascii=False) + "\n")
                # Also print to stderr so it's visible in Hermes output
                print(f"[holo] FACTS INJECTED: {len(top5)} facts", flush=True)
                for _fi in top5:
                    _fc = _fi.get("content", "")[:100]
                    _fcos = _fi.get("_faiss_distance", 0)
                    print(f"[holo]   cos={_fcos:.4f} {_fc}", flush=True)
            except Exception:
                pass
            # WRITE MARKER: facts were retrieved
            try:
                with open("/tmp/fact_injection_ok", "a") as _mf:
                    _mf.write(f"[{__import__('datetime').datetime.now().isoformat()[:19]}] "
                              f"FACTS_RETRIEVED n={len(top5)} query={query[:80]!r}\n")
                    for _r in top5:
                        _c = _r.get("content", "")[:80]
                        _cos = round(_r.get("_faiss_distance", 0.0), 4)
                        _mf.write(f"  cos={_cos:.4f} {_c}\n")
            except Exception:
                pass
            return "\n".join(lines)
        except Exception as _e:
            _warn("prefetch failed: %s", _e)
            # Log the failure to the turn debug log too
            try:
                import json, os as _dbg_os
                from datetime import datetime
                _dbg_log = _dbg_os.path.expanduser("~/.hermes/holographic_debug.log")
                with open(_dbg_log, "a", encoding="utf-8") as _df:
                    _df.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "event": "prefetch_error",
                        "error": str(_e),
                        "has_retriever": self._retriever is not None,
                        "has_faiss": getattr(self, "_has_faiss", False),
                        "has_faiss_index": getattr(self, "_faiss_index", None) is not None,
                        "has_store": self._store is not None,
                        "_min_trust": getattr(self, "_min_trust", "N/A"),
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
            self._last_raw_results = []
            return ""
    def _patched_handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Handle tool calls, updating FAISS index when facts are added."""
        result = _orig_handle_tool_call(self, tool_name, args, **kwargs)
        if tool_name == "fact_store" and args.get("action") == "add":
            try:
                if self._store:
                    result_data = json.loads(result)
                    fid_val = result_data.get("fact_id")
                    if fid_val is not None:
                        content = args.get("content", "")
                        # Update content_vec (HRR-based, for probe axis)
                        _hrr_d = getattr(self._store, "hrr_dim", 1024)
                        cv = phases_to_bytes(encode_text(content, _hrr_d))
                        self._store._conn.execute(
                            "UPDATE facts SET content_vec = ? WHERE fact_id = ?",
                            (cv, fid_val),
                        )
                        # Update bert_vec (BERT-based, for FAISS axis)
                        try:
                            bv = _bert.encode_passages(content).tobytes()
                            self._store._conn.execute(
                                "UPDATE facts SET bert_vec = ? WHERE fact_id = ?",
                                (bv, fid_val),
                            )
                        except Exception:
                            pass
                        self._store._conn.commit()
                        # Update FAISS index with BERT vector
                        if self._faiss_index is not None:
                            try:
                                bv_arr = _bert.encode_passages(content)
                                self._faiss_index.add(bv_arr[0], fid_val)
                                _dbg("added fact_id=%s to FAISS (BERT)", fid_val)
                            except Exception:
                                pass
            except Exception:
                pass
        return result
    def _patched_get_tool_schemas(self):
        """Hide fact_store/fact_feedback from the model in ApplyPilot mode.
        
        The model cannot successfully call fact_store (bytes serialization bug
        in search/probe) and fact_feedback should not be invoked by the model
        directly (scoring is handled by the background review pipeline).
        
        In ApplyPilot (identified by HERMES_HOME containing '.applypilot'),
        return no schemas so the model doesn't see broken tools. The internal
        injection, extraction, and scoring still work because they call
        handle_tool_call directly, not through get_tool_schemas().
        
        In personal Hermes mode, expose the tools for interactive use.
        """
        # Check if running under ApplyPilot
        _hermes_home = _os.environ.get("HERMES_HOME", "")
        if ".applypilot" in _hermes_home:
            return []
        # Personal Hermes: expose tools normally
        return _orig_get_tool_schemas(self)
    # ── Apply patched methods ────────────────────────────────────────────────
    _HolographicMemoryProvider.__init__ = _patched_init
    _HolographicMemoryProvider.prefetch = _patched_prefetch
    _HolographicMemoryProvider.initialize = _patched_initialize
    _HolographicMemoryProvider._rebuild_faiss_index = _rebuild_faiss_index
    _HolographicMemoryProvider._entities_from_query = _entities_from_query
    _HolographicMemoryProvider._compositional_search = _compositional_search
    _HolographicMemoryProvider._faiss_search = _faiss_search
    _orig_handle_tool_call = _HolographicMemoryProvider.handle_tool_call
    _HolographicMemoryProvider.handle_tool_call = _patched_handle_tool_call
    _orig_get_tool_schemas = _HolographicMemoryProvider.get_tool_schemas
    _HolographicMemoryProvider.get_tool_schemas = _patched_get_tool_schemas
    _dbg("patches applied successfully")
    # ── .pyc staleness check ───────────────────────────────────────────────
    # Log the .py source vs .pyc mtimes so we can detect stale cached bytecode.
    try:
        _py_path = __file__
        _pyc_path = __file__ + "c"  # .pyc alongside .py
        import time as _tc
        _py_mtime = _os.path.getmtime(_py_path) if _os.path.exists(_py_path) else 0
        _pyc_mtime = _os.path.getmtime(_pyc_path) if _os.path.exists(_pyc_path) else 0
        _stale = _pyc_mtime > 0 and _py_mtime > _pyc_mtime
        _ilog2 = _os.path.expanduser("~/.hermes/holographic_debug.log")
        with open(_ilog2, "a", encoding="utf-8") as _if2:
            _if2.write(json.dumps({
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "event": "pyc_staleness_check",
                "py_path": str(_py_path),
                "py_mtime": _py_mtime,
                "pyc_path": str(_pyc_path),
                "pyc_mtime": _pyc_mtime,
                "stale": _stale,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
else:
    _warn("HolographicMemoryProvider NOT FOUND in real module — patches not applied!")
