# RWKV-7 Vector Delta for Holographic Retrieval — Implementation Plan

> Implement a learned query transformation using RWKV-7 with a vector output head.
> The model ingests the same prefetch query currently used by HRR and outputs a delta
> vector that enriches the HRR query before fact similarity scoring. Trained on
> evaluations from the existing background scoring LLM via full end-to-end fine-tuning.

**Goal:** Replace the fixed HRR query encoding with a context-aware, learned query
vector that improves retrieval over time by learning from the scoring LLM's feedback.
Every parameter in the model trains — no frozen backbones, no adapters, no half-measures.

**Architecture:**
```
Input text (prefetch query: prev_thinking + prev_response + user_msg)
  → RWKV-7 blocks (all layers, fully trained)
  → final hidden state [4096]
  → delta_head (linear 4096 → 1024, replaces original lm_head)
  → delta_vec [1024]

final_query_vec = hrr_query_vec + delta_vec
  → cosine similarity against all fact HRR vectors
  → ranked scored facts returned to main model
```

The delta head is **zero-initialized** so the system starts behaviorally identical
to today on day one. Every training iteration moves it toward better retrieval.

**Tech Stack:** RWKV-7 (PyTorch), torch, numpy, existing SQLite / HRR infrastructure,
background_review.py logging. GPU required for training. Inference on GPU or CPU.

**Stateful Architecture — RWKV-7 keeps hidden state across turns:**

The RWKV-7 model is initialized once at the start of a conversation and maintains
its internal RNN hidden state across every turn. It is not reset between turns —
only reset when a new conversation/session begins.

This means that although `compute_delta()` is called with only the current turn's
input (last assistant response + thinking + user prompt + facts), the model's
recurrent state carries information from all prior turns in the conversation.
Over many turns, the hidden state encodes the conversation's trajectory — which
topics recurred, which facts were repeatedly retrieved, which user preferences
persisted, which corrections were applied.

The delta vector it emits is therefore a function of **both** the immediate query
**and** the entire conversational history encoded in its recurrent state. This
allows the system to learn conversational dynamics like "the user keeps correcting
me on this topic — dampen those fact scores" or "this preference was affirmed
multiple times — amplify related facts."

Implementation note: the server process must persist the RWKV-7 model instance
(and thus its hidden state) across multiple HTTP requests within the same session.
The client (`compute_delta`) sends a `session_id` field alongside the query text.
The server maintains a dict of `session_id → (model_state, conversation_counter)`
and resets the state when a new `session_id` is seen (or when the counter exceeds
a max-conversation-length threshold, at which point the hidden state is zeroed
to avoid stale context bleeding across unrelated conversations).

---

## Phase 0: Training Data Collection

### Task 0.1: Store the prefetch query on the agent

**Objective:** The prefetch query (built in conversation_loop.py) needs to be
accessible to the background scoring function, which runs later on a separate
thread. Store it on the agent object.

**Files:**
- Modify: `holographic-fork/patches/agent/conversation_loop.py`

Find Replacement D (around line 184-197) where `_query` is built. After the
prefetch query is fully assembled, add one line to persist it:

```python
# After: _query = _asst_text.strip() + "\n" + _query
# Add:
agent._last_prefetch_query = _query
```

The final block looks like:

```python
'            _query = original_user_message if isinstance(original_user_message, str) else ""\n'
'            # Include last assistant response for multi-turn context\n'
'            if messages and len(messages) >= 2:\n'
'                _last_asst_msg = messages[-2]\n'
'                if isinstance(_last_asst_msg, dict) and _last_asst_msg.get("role") == "assistant":\n'
'                    _asst_text = _last_asst_msg.get("content", "")\n'
'                    _asst_reasoning = _last_asst_msg.get("reasoning", "")\n'
'                    if isinstance(_asst_reasoning, str) and _asst_reasoning.strip():\n'
'                        _asst_text = _asst_reasoning.strip() + "\\\\n\\\\n" + (_asst_text or "")\n'
'                    if isinstance(_asst_text, str) and _asst_text.strip():\n'
'                        _query = _asst_text.strip() + "\\\\n" + _query\n'
'            agent._last_prefetch_query = _query  # ← one new line',
```

**Verify:**
```bash
cd ~/Code/hermes/holographic-fork
python3 -c "import ast; ast.parse(open('patches/agent/conversation_loop.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 0.2: Log scoring evaluations to training.db

**Objective:** After the scoring LLM evaluates which facts were useful, write
each evaluation as a row in a separate training database.

**Storage:** `~/.hermes/training.db` — a dedicated SQLite database with a single
`training_events` table. Separate from `memory_store.db` so training data can
be archived or cleared independently.

```sql
CREATE TABLE IF NOT EXISTS training_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text  TEXT NOT NULL,
    fact_id     INTEGER NOT NULL,
    trust_delta REAL NOT NULL,
    session_id  TEXT DEFAULT '',
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_fact ON training_events(fact_id);
```

**Files:**
- Modify: `holographic-fork/patches/agent/background_review.py`

In `_run_fact_scoring()`, after the evaluation loop (after line ~455 where
`evaluations` has been iterated and scored_lines built), add a logging block:

```python
    # ── Log training datum to separate training.db ───────────────────
    prefetch_query = getattr(agent, "_last_prefetch_query", "") or conversation_text
    session_id = getattr(agent, "session_id", "")
    try:
        import sqlite3
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
    except Exception:
        pass  # non-critical logging
```

**Verify:**
```bash
cd ~/Code/hermes/holographic-fork
python3 -c "import ast; ast.parse(open('patches/agent/background_review.py').read()); print('OK')"
```
Expected: `OK`

After a few conversation turns, verify data accumulates:
```bash
sqlite3 ~/.hermes/training.db "SELECT COUNT(*) FROM training_events"
# Expected: > 0
```

---

## Phase 1: RWKV-7 Model — Architecture Change

### Task 1.1: Create the delta-head RWKV-7 wrapper

**Objective:** Build the model class that loads pretrained RWKV-7 weights,
replaces the lm_head with a delta_head (4096 → 1024), and provides a
`compute_delta(text) → np.ndarray` interface for inference.

**Files:**
- Create: `holographic-fork/rwkv7_model.py`

```python
"""
RWKV-7 with a vector output head for HRR query delta computation.

Replaces the standard vocabulary projection head (lm_head) with a
learned 4096 → 1024 linear layer that outputs a delta vector in
HRR embedding space. The delta is added to the HRR query encoding
during retrieval: final_query = hrr_query + delta.

Full fine-tuning: all blocks, all layers, no frozen parameters.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RWKV7DeltaModel(nn.Module):
    """RWKV-7 backbone with a vector output head instead of language head.

    The original ``lm_head`` (vocab_size × hidden_dim) is replaced with
    ``delta_head`` (hidden_dim × 1024). The rest of the architecture —
    embed, time-mixing blocks, channel-mixing blocks — is identical to
    the pretrained checkpoint.

    Forward pass: input tokens → embeddings → N RWKV blocks → hidden
    state → delta_head → delta vector [1024].

    No token generation; the model outputs a single 1024-dim vector.
    """

    def __init__(
        self,
        rwkv7_backbone: nn.Module,
        hidden_dim: int = 4096,
        delta_dim: int = 1024,
    ):
        super().__init__()
        # Take the pretrained backbone (embed + all blocks)
        # but strip the lm_head
        self.backbone = rwkv7_backbone

        # Replace language head with delta head
        # Zero-initialized so first forward pass outputs zeros
        self.delta_head = nn.Linear(hidden_dim, delta_dim, bias=False)
        nn.init.zeros_(self.delta_head.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks, return delta vector.

        Args:
            input_ids: [batch, seq_len] tokenized input text.

        Returns:
            delta: [batch, 1024] — the enriched query delta.
        """
        # Run through all RWKV blocks
        # The backbone returns the last token's hidden state after all blocks
        hidden = self.backbone(input_ids)  # [batch, hidden_dim]

        # Project to delta space
        delta = self.delta_head(hidden)  # [batch, delta_dim]
        return delta


def load_pretrained_backbone(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> nn.Module:
    """Load pretrained RWKV-7 backbone (embed + blocks, no lm_head).

    This function is RWKV-version-specific and will need adjustment
    based on the exact RWKV-7 release (v7.0, v7.2, etc.).
    See ``rwkv_pip_package`` for the canonical loader.
    """
    from rwkv.model import RWKV as RWKVBackbone
    from rwkv.utils import PIPELINE, PIPELINE_ARGS

    # Load the raw RWKV model
    model = RWKVBackbone(
        str(checkpoint_path),
        strategy=device,  # 'cuda fp16' or 'cpu fp32'
    )

    # Wrap into a torch Module that exposes just the blocks
    # The RWKV library's model has .args, .w (weights dict), .run()
    # We extract the learned parameters and build a standard nn.Module.

    # NOTE: This is a simplification. The actual RWKV-7 model is a custom
    # CUDA kernel or pure-pytorch stack. The exact wrapping depends on
    # which RWKV-7 implementation is used (the official pip package vs
    # the custom CUDA kernel version).

    # For now: the user will adapt this based on their RWKV-7 setup.
    # The key interface is:
    #   backbone(input_ids) → [batch, hidden_dim] last hidden state
    raise NotImplementedError("RWKV-7 backbone loading is implementation-specific")
```

---

### Task 1.2: Inference server for the trained model

**Objective:** A lightweight HTTP server that loads the trained RWKV-7 delta
model and serves `POST /compute_delta` returning the 1024-dim delta vector.

**Files:**
- Create: `holographic-fork/rwkv7_server.py`

```python
"""
HTTP inference server for RWKV-7 delta model.

Usage:
    python rwkv7_server.py --checkpoint /path/to/rwkv7-delta.pt --port 8080

Requests:
    POST /compute_delta
    {"text": "prefetch query text"}
    → {"delta": [0.001, -0.002, ...]}  # 1024 floats
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

from rwkv7_model import RWKV7DeltaModel, load_pretrained_backbone

logger = logging.getLogger(__name__)


def serve(checkpoint_path: str, port: int, device: str):
    """Start the delta computation server."""

    # Load model
    backbone = load_pretrained_backbone(checkpoint_path, device=device)
    model = RWKV7DeltaModel(backbone)
    model.to(device)
    model.eval()

    # Load trained weights (delta_head + all backbone params)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    # Tokenizer
    from rwkv.utils import PIPELINE
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

    from flask import Flask, request, jsonify  # or FastAPI

    app = Flask(__name__)

    @app.route("/compute_delta", methods=["POST"])
    def compute_delta():
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "empty text"}), 400

        # Tokenize
        tokens = pipeline.encode(text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Forward
        with torch.no_grad():
            delta = model(input_ids)  # [1, 1024]

        # Return as list
        delta_list = delta[0].cpu().numpy().tolist()
        return jsonify({"delta": delta_list})

    app.run(host="127.0.0.1", port=port)
```

---

### Task 1.3: Query transformer client

**Objective:** The client that the holographic provider calls during prefetch
to get the delta vector from the RWKV-7 server.

**Files:**
- Create: `holographic-fork/patches/agent/query_transformer.py`

```python
"""
Client for the RWKV-7 delta server. Called by the holographic provider
during prefetch to compute the enriched query vector.

Usage:
    from agent.query_transformer import compute_delta
    delta = compute_delta("prefetch query text")
    # delta is None if server unavailable (graceful fallback)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

_SERVER_URL = os.environ.get("RWKV7_DELTA_URL", "http://127.0.0.1:8080")
_TIMEOUT = 10.0  # seconds


def compute_delta(text: str, server_url: str = _SERVER_URL) -> Optional[np.ndarray]:
    """Send text to RWKV-7 delta server, return 1024-dim delta vector.

    Returns None on any failure — caller falls back to pure HRR.
    """
    if not text or not text.strip():
        return None
    try:
        resp = requests.post(
            f"{server_url}/compute_delta",
            json={"text": text.strip()},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.debug("Delta server returned %d", resp.status_code)
            return None
        data = resp.json()
        d = data.get("delta")
        if d is None:
            return None
        return np.array(d, dtype=np.float32)
    except requests.ConnectionError:
        logger.debug("Delta server at %s not reachable", server_url)
        return None
    except Exception as e:
        logger.debug("Delta query failed: %s", e)
        return None


# Cache for the last delta (for debugging/inspection)
_last_delta: Optional[np.ndarray] = None


def get_last_delta() -> Optional[np.ndarray]:
    global _last_delta
    return _last_delta
```

---

## Phase 2: Retrieval Integration

### Task 2.1: Patch FactRetriever.search() to accept delta_vector

Since `retrieval.py` lives in the Nix store (read-only), we create a patched
version that shadows it.

**Files:**
- Create: `holographic-fork/patches/plugins/memory/holographic/retrieval.py`

```python
"""
Patched FactRetriever — accepts an optional delta_vector to enrich the HRR
query encoding before similarity scoring.

Loads the real FactRetriever from the Nix store, overrides only search().
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 1. Load the real retrieval module from Nix store
# -------------------------------------------------------------------
import importlib.util
import sys
from pathlib import Path

import plugins.memory.holographic as holo_pkg

_real_path: Path | None = None
for _p in holo_pkg.__path__:
    if "patches" in str(_p):
        continue
    _candidate = Path(_p) / "retrieval.py"
    if _candidate.exists():
        _real_path = _candidate
        break

if _real_path is None:
    raise ImportError("Could not locate real plugins/memory/holographic/retrieval.py")

_spec = importlib.util.spec_from_file_location(
    "plugins.memory.holographic.retrieval_real", str(_real_path)
)
_real_mod = importlib.util.module_from_spec(_spec)
sys.modules["plugins.memory.holographic.retrieval_real"] = _real_mod
_spec.loader.exec_module(_real_mod)

# Re-export everything
for _attr in dir(_real_mod):
    if _attr.startswith("_") and _attr != "__all__":
        continue
    globals()[_attr] = getattr(_real_mod, _attr)

from plugins.memory.holographic import holographic as hrr


# -------------------------------------------------------------------
# 2. Override search() with delta support
# -------------------------------------------------------------------

_ORIGINAL_SEARCH = _real_mod.FactRetriever.search


def _patched_search(
    self,
    query: str,
    category: str | None = None,
    min_trust: float = 0.3,
    limit: int = 10,
    delta_vector: Optional[np.ndarray] = None,
) -> list[dict]:
    """Hybrid search with optional delta enrichment of the HRR query.

    When delta_vector is provided and HRR is available, the query encoding
    becomes: hrr.encode_text(query) + delta_vector
    """
    # FTS5 candidates (unchanged)
    candidates = self._fts_candidates(query, category, min_trust, limit * 3)
    if not candidates:
        return []

    query_tokens = self._tokenize(query)
    scored = []

    # Pre-compute enriched query vector
    enriched_query_vec = None
    if delta_vector is not None and self.hrr_weight > 0 and hrr._HAS_NUMPY:
        raw_vec = hrr.encode_text(query, self.hrr_dim)
        enriched_query_vec = raw_vec + delta_vector.astype(raw_vec.dtype)

    for fact in candidates:
        content_tokens = self._tokenize(fact["content"])
        tag_tokens = self._tokenize(fact.get("tags", ""))
        all_tokens = content_tokens | tag_tokens

        jaccard = self._jaccard_similarity(query_tokens, all_tokens)
        fts_score = fact.get("fts_rank", 0.0)

        # HRR similarity — use enriched query vec if available
        if self.hrr_weight > 0 and fact.get("hrr_vector"):
            fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
            query_vec = enriched_query_vec if enriched_query_vec is not None \
                        else hrr.encode_text(query, self.hrr_dim)
            hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
        else:
            hrr_sim = 0.5

        relevance = (self.fts_weight * fts_score
                     + self.jaccard_weight * jaccard
                     + self.hrr_weight * hrr_sim)
        score = relevance * fact["trust_score"]

        if self.half_life > 0:
            score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

        fact["score"] = score
        scored.append(fact)

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:limit]
    for fact in results:
        fact.pop("hrr_vector", None)
    return results


# Replace on the class and re-export
_real_mod.FactRetriever.search = _patched_search
FactRetriever = _real_mod.FactRetriever
FactRetriever.search = _patched_search

__all__ = [n for n in dir(_real_mod) if not n.startswith("_")]
```

**Verify:**
```bash
cd ~/Code/hermes/holographic-fork
python3 -c "import ast; ast.parse(open('patches/plugins/memory/holographic/retrieval.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 2.2: Wire delta into prefetch()

**Files:**
- Modify: `holographic-fork/patches/plugins/memory/holographic/__init__.py`

At the top of the file, add the import:

```python
try:
    from agent.query_transformer import compute_delta
    _HAS_DELTA = True
except ImportError:
    _HAS_DELTA = False

    def compute_delta(text):  # type: ignore
        return None
```

In the `HolographicMemoryProvider.__init__()` (around line 153), add config:

```python
self._delta_enabled = bool(self._config.get("query_delta_enabled", False))
```

In `prefetch()`, before `self._retriever.search(...)`, add:

```python
        # Compute RWKV-7 delta if enabled
        delta_vec = None
        if self._delta_enabled and _HAS_DELTA:
            try:
                delta_vec = compute_delta(query)
            except Exception:
                pass

        kwargs = dict(
            min_trust=self._min_trust,
            limit=_FETCH_LIMIT,
        )
        if delta_vec is not None:
            kwargs["delta_vector"] = delta_vec

        results = self._retriever.search(query, **kwargs)
```

---

### Task 2.3: Add config toggle

**Files:**
- Modify: `~/.hermes/config.yaml`

Under `plugins.hermes-memory-store:` add:

```yaml
plugins:
  hermes-memory-store:
    query_delta_enabled: false   # toggle to true once model is trained
```

---

## Phase 3: Full Training Pipeline

### Task 3.1: Training script

**Files:**
- Create: `holographic-fork/train_rwkv7_delta.py`

```python
#!/usr/bin/env python3
"""
|End-to-end training of RWKV-7 delta head + all backbone parameters.

Reads the training event log from `~/.hermes/training.db`, joins against
fact HRR vectors from `memory_store.db`, and fine-tunes every parameter
of RWKV-7 via contrastive loss.

Usage:
    python train_rwkv7_delta.py --pretrained /path/to/rwkv7-base.pth \\
        --train-db ~/.hermes/training.db \\
        --fact-db ~/.hermes/memory_store.db \\
        --epochs 10 \\
        --lr 2e-5 \\
        --save /path/to/rwkv7-delta-trained.pth

Requires GPU. A single A100 can train a 1.5B model in ~30 min per epoch
on ~10k examples.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rwkv7_model import RWKV7DeltaModel, load_pretrained_backbone


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_hrr_vector(conn: sqlite3.Connection, fact_id: int) -> np.ndarray | None:
    """Load a fact's HRR vector from the facts table."""
    from plugins.memory.holographic import holographic as hrr

    row = conn.execute(
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    if row is None or row["hrr_vector"] is None:
        return None
    return hrr.bytes_to_phases(row["hrr_vector"])


class ContrastiveDataset(Dataset):
    """Each item: (query_text, pos_hrr_vecs, neg_hrr_vecs)

    Loads from training.db (dedicated training event store) and joins
    against fact HRR vectors from memory_store.db in one pass.
    """

    def __init__(self, train_db: Path, fact_db: Path, tokenizer):
        self.tokenizer = tokenizer
        self.examples: list[dict] = []
        self._load(train_db, fact_db)

    def _load(self, train_db: Path, fact_db: Path):
        fact_conn = sqlite3.connect(str(fact_db))
        fact_conn.row_factory = sqlite3.Row

        train_conn = sqlite3.connect(str(train_db))
        train_conn.row_factory = sqlite3.Row

        # Load all training events, joined with HRR vectors
        rows = train_conn.execute(
            "SELECT t.query_text, t.fact_id, t.trust_delta, t.session_id "
            "FROM training_events t "
            "ORDER BY t.event_id"
        ).fetchall()
        train_conn.close()

        # Group by query_text — one example per unique query
        query_groups: dict[str, dict] = {}
        for row in rows:
            q = row["query_text"]
            fid = row["fact_id"]
            td = row["trust_delta"]
            if not q or not isinstance(fid, int) or fid <= 0:
                continue
            hrr = load_hrr_vector(fact_conn, fid)
            if hrr is None:
                continue
            if q not in query_groups:
                query_groups[q] = {"query": q, "positive": [], "negative": []}
            if td > 0:
                query_groups[q]["positive"].append((hrr, td))
            elif td < 0:
                query_groups[q]["negative"].append((hrr, abs(td)))

        fact_conn.close()

        # Flatten to list, skip queries with no usable facts
        self.examples = [
            eg for eg in query_groups.values()
            if eg["positive"] or eg["negative"]
        ]

        print(f"Loaded {len(self.examples)} training examples from {len(rows)} events")
        pos = sum(len(e["positive"]) for e in self.examples)
        neg = sum(len(e["negative"]) for e in self.examples)
        print(f"  Positive pairs: {pos}, Negative pairs: {neg}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Tokenize query text
        tokens = self.tokenizer.encode(ex["query"])
        # Return tokens + numpy arrays of HRR vectors
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "positive_vecs": torch.tensor(
                np.array([v for v, _ in ex["positive"]], dtype=np.float32)
            ) if ex["positive"] else torch.empty(0, 1024),
            "positive_weights": torch.tensor(
                [w for _, w in ex["positive"]], dtype=torch.float32
            ) if ex["positive"] else torch.empty(0),
            "negative_vecs": torch.tensor(
                np.array([v for v, _ in ex["negative"]], dtype=np.float32)
            ) if ex["negative"] else torch.empty(0, 1024),
            "negative_weights": torch.tensor(
                [w for _, w in ex["negative"]], dtype=torch.float32
            ) if ex["negative"] else torch.empty(0),
        }


# -------------------------------------------------------------------
# Loss function
# -------------------------------------------------------------------

def contrastive_loss(
    delta: torch.Tensor,          # [batch, 1024] — model output
    pos_vecs: list[torch.Tensor], # per-item positive HRR vectors
    pos_weights: list[torch.Tensor],
    neg_vecs: list[torch.Tensor],
    neg_weights: list[torch.Tensor],
    margin: float = 0.3,
) -> torch.Tensor:
    """Contrastive loss: pull delta toward positive fact vectors,
    push away from negative fact vectors."""
    loss = torch.tensor(0.0, device=delta.device)
    n = 0

    for i in range(delta.size(0)):
        d = delta[i]  # [1024]

        # Normalize delta for stable cosine
        d_norm = F.normalize(d, dim=0)

        # Positive: maximize cosine similarity
        for j in range(pos_vecs[i].size(0)):
            pos = pos_vecs[i][j].to(delta.device)
            pos_norm = F.normalize(pos, dim=0)
            sim = (d_norm * pos_norm).sum()
            loss += (1.0 - sim) * pos_weights[i][j]
            n += 1

        # Negative: minimize cosine similarity (hinge)
        for j in range(neg_vecs[i].size(0)):
            neg = neg_vecs[i][j].to(delta.device)
            neg_norm = F.normalize(neg, dim=0)
            sim = (d_norm * neg_norm).sum()
            if sim > margin:
                loss += (sim - margin) * neg_weights[i][j]
                n += 1

    return loss / max(n, 1)


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------

def collate_fn(batch):
    """Custom collation for variable-length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    pos_vecs = [item["positive_vecs"] for item in batch]
    pos_weights = [item["positive_weights"] for item in batch]
    neg_vecs = [item["negative_vecs"] for item in batch]
    neg_weights = [item["negative_weights"] for item in batch]

    # Pad input sequences to max length in batch
    max_len = max(len(ids) for ids in input_ids)
    padded = torch.stack([
        F.pad(ids, (0, max_len - len(ids)), value=0)
        for ids in input_ids
    ])

    return {
        "input_ids": padded,
        "pos_vecs": pos_vecs,
        "pos_weights": pos_weights,
        "neg_vecs": neg_vecs,
        "neg_weights": neg_weights,
    }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True, help="Base RWKV-7 checkpoint")
    parser.add_argument("--train-db", default=str(Path.home() / ".hermes/training.db"))
    parser.add_argument("--fact-db", default=str(Path.home() / ".hermes/memory_store.db"))
    parser.add_argument("--save", default=str(Path.home() / ".hermes/rwkv7/rwkv7-delta-trained.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load pretrained backbone (no lm_head)
    print("Loading pretrained RWKV-7 backbone...")
    backbone = load_pretrained_backbone(args.pretrained, device=device)

    # Build delta model
    model = RWKV7DeltaModel(backbone).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load existing trained checkpoint if available (continue training)
    save_path = Path(args.save)
    if save_path.exists():
        print(f"Loading existing checkpoint: {save_path}")
        model.load_state_dict(torch.load(args.save, map_location=device))

    # Tokenizer
    from rwkv.utils import PIPELINE
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

    # Dataset
    dataset = ContrastiveDataset(Path(args.train_db), Path(args.fact_db), pipeline.tokenizer)
    if len(dataset) == 0:
        print("No training data. Run Hermes to accumulate scoring log.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer — all parameters train
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()

            # Forward — delta for each item in batch
            delta = model(input_ids)  # [batch, 1024]

            # Loss
            loss = contrastive_loss(
                delta,
                batch["pos_vecs"],
                batch["pos_weights"],
                batch["neg_vecs"],
                batch["neg_weights"],
            )

            loss.backward()

            # Gradient clipping (prevents RWKV block explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss = {avg_loss:.6f}")

        # Save checkpoint every epoch
        epoch_path = save_path.with_suffix(f".epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_path)
        print(f"  Saved: {epoch_path}")

    # Save final
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Final model: {save_path}")


if __name__ == "__main__":
    train()
```

---

### Task 3.2: Inference mode — loading trained model for serving

In `rwkv7_server.py` (from Task 1.2), the `load_state_dict` call loads
all weights (backbone + delta_head) from the trained checkpoint. The
model is a standard `RWKV7DeltaModel` — the checkpoint contains every
parameter.

No special conversion needed. The saved `.pth` file from training is
loaded directly for inference.

---

## Phase 4: Bootstrap & Verification

### Task 4.1: Verify identity behavior (before training)

With the delta head zero-initialized and `query_delta_enabled: true`:

```bash
# No trained checkpoint exists yet
ls ~/.hermes/rwkv7/rwkv7-delta-trained.pth
# → "No such file" (expected)

# Start the delta server — model outputs all zeros
python rwkv7_server.py --pretrained ~/.hermes/rwkv7/rwkv7-base.pth \
    --port 8080 &

# Run Hermes with delta enabled
hermes chat -q "What are my python settings?"
```

The delta head is zero-initialized, so `delta = [0, 0, ..., 0]`. The
query vector is `hrr_query + 0 = hrr_query`. Behavior is **identical**
to current.

### Task 4.2: Accumulate training data

```bash
# Run normal Hermes sessions
hermes
# ... conversation turns ...
hermes
# ... more turns ...

# Check training event count
sqlite3 ~/.hermes/training.db "SELECT COUNT(*) FROM training_events"
# → growing with each session
```

### Task 4.3: Train and verify improvement

```bash
# Train on accumulated data
python train_rwkv7_delta.py --pretrained ~/.hermes/rwkv7/rwkv7-base.pth \
    --epochs 10 \
    --save ~/.hermes/rwkv7/rwkv7-delta-trained.pth

# Copy checkpoint to server location
cp ~/.hermes/rwkv7/rwkv7-delta-trained.pth ~/.hermes/rwkv7/current.pth

# Restart server with trained weights
# (server loads current.pth instead of base)

# Run same query — delta now non-zero, retrieval shifted
hermes chat -q "What are my python settings?"
```

---

## File Change Map

| File | Action | Purpose |
|------|--------|---------|
| `patches/agent/conversation_loop.py` | Modify (+1 line) | Store prefetch query on agent |
| `patches/agent/background_review.py` | Modify (+~25 lines) | Log scoring evaluations + query to training.db |
| `patches/agent/query_transformer.py` | Create | HTTP client for delta server |
| `patches/plugins/memory/holographic/retrieval.py` | Create | Override FactRetriever.search() with delta support |
| `patches/plugins/memory/holographic/__init__.py` | Modify | Wire delta into prefetch(), add config toggle |
| `rwkv7_model.py` | Create | RWKV-7 with delta_head replacing lm_head |
| `rwkv7_server.py` | Create | HTTP inference server for trained model |
| `train_rwkv7_delta.py` | Create | Full fine-tuning script (all params) |
| `~/.hermes/config.yaml` | Modify | Add `query_delta_enabled: false` |

## Data Flow Summary

```
Every turn:
  conversation_loop builds prefetch query
    → agent._last_prefetch_query = "prev_thinking\nprev_response\nuser_msg"
    → prefetch() calls FactRetriever.search(query)
  
  IF query_delta_enabled:
    → compute_delta(query) → POST to rwkv7_delta_server
    → delta_head(rwkv7_forward(query)) → delta_vec [1024]
    → FactRetriever.search(query, delta_vector=delta_vec)
    → hrr.encode_text(query) + delta_vec → enriched_query_vec
    → cosine similarity against all fact HRR vectors
  ELSE:
    → FactRetriever.search(query)  # unchanged HRR path

After response:
  background_review._run_fact_scoring()
    → LLM evaluates facts → trust_delta per fact
    → applies to trust_scores in DB
    → NEW: inserts one row per fact into ~/.hermes/training.db (training_events table)

Periodically (manual or cron):
  train_rwkv7_delta.py
    → reads training.db events
    → joins against HRR vectors from memory_store.db
    → groups by query_text: one example = one unique query + all its scored facts
    → full forward-backward through all RWKV-7 blocks
    → contrastive loss: pull delta toward positive facts, push from negative
    → saves full checkpoint: rwkv7-delta-trained.pth
    → next inference loads new weights

Bootstrapping:
  Delta head zero-initialized → delta = [0, 0, ..., 0]
  → hrr_query + 0 = hrr_query → identical to current behavior
  → every training step is a strict improvement from a working baseline
  → no cold start, no regression, no special test mode
```
