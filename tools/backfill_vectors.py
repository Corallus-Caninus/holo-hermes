#!/nix/store/37mly3rqq15i0axhcx3258bl34l51psp-hermes-agent-env/bin/python3.12
"""Manual backfill: re-encode all bert_vec with correct last-token pooling + asymmetric prefix on MI25.
Run this with Hermes STOPPED (so GPU is free), then restart Hermes after.

Usage:
    kill $(pgrep -f fully_automatic_holographic) 2>/dev/null; sleep 2
    python3 ~/Code/hermes/backfill_vectors.py
"""
import sys, os, time, sqlite3, json, numpy as np

# ── Paths ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPT_DIR)
for p in ["/tmp/faiss-venv/lib/python3.12/site-packages",
           "/nix/store/d6x7mb4fbhms6mshya1x0qp9s37wv08q-python3.12-numpy-2.4.4/lib/python3.12/site-packages",
           os.path.join(_REPO_DIR, "patches/plugins/memory/holographic")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Load model (auto-detects MI25 via ROCm) ─────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location("bert_embed",
    os.path.join(_REPO_DIR, "patches/plugins/memory/holographic/bert_embed.py"))
bert = importlib.util.module_from_spec(spec)
sys.modules["bert_embed"] = bert
spec.loader.exec_module(bert)

print(f"Loading gte-Qwen2-1.5B-instruct on MI25...", flush=True)
t0 = time.time()
bert._load()
print(f"Model loaded in {time.time()-t0:.1f}s on {bert._device}", flush=True)
print(f"Instruction: '{bert._INSTRUCTION}'", flush=True)

# ── Verify asymmetric encoding ──────────────────────────────────────────
# Queries get "Instruct: ...\nQuery: " prefix; passages are raw text.
# They SHOULD differ for identical text (cos < 0.99 is correct).
t = "Test fact: FAISS cache is populated and loads successfully."
qv = bert.encode_queries(t)
pv = bert.encode_passages(t)
sim_qp = float(np.dot(qv[0]/np.linalg.norm(qv[0]), pv[0]/np.linalg.norm(pv[0])))
print(f"Query vs Passage (same text): cos={sim_qp:.4f}  (expected < 0.9 — asymmetric encoding)", flush=True)
print("  (query gets 'Instruct: ... Query:' prefix, passage is raw text)", flush=True)

# ── DB ──────────────────────────────────────────────────────────────────
DB = os.path.expanduser("~/.hermes/memory_store.db")
db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row

total = db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
null_before = db.execute("SELECT COUNT(*) FROM facts WHERE bert_vec IS NULL").fetchone()[0]
print(f"\nFacts: {total} total, {null_before} NULL bert_vec", flush=True)

# ── Backfill all NULL vectors ────────────────────────────────────────────
rows = db.execute("SELECT fact_id, content FROM facts WHERE bert_vec IS NULL").fetchall()
if not rows:
    print("No NULL vectors found. Checking if existing vectors are stale...", flush=True)
    sample = db.execute("SELECT fact_id, content, bert_vec FROM facts LIMIT 1").fetchone()
    stored = np.frombuffer(sample["bert_vec"], dtype=np.float32)
    fresh = bert.encode_passages(sample["content"])
    cos = float(np.dot(stored/np.linalg.norm(stored), fresh[0]/np.linalg.norm(fresh[0])))
    print(f"Sample [{sample['fact_id']}]: stored vs fresh cos={cos:.4f}", flush=True)
    if cos < 0.9:
        print(f"Stale! Force-clearing all {total} bert_vec...", flush=True)
        db.execute("UPDATE facts SET bert_vec = NULL")
        db.commit()
        rows = db.execute("SELECT fact_id, content FROM facts WHERE bert_vec IS NULL").fetchall()
    else:
        print("All vectors are correct. Nothing to do.", flush=True)
        db.close()
        sys.exit(0)

print(f"\nEncoding {len(rows)} facts on MI25 at batch_size=100...", flush=True)

# Start with batch_size=100. If OOM, we'll catch it.
BATCH = 100
ok = 0
fail = 0
t_start = time.time()

while ok < len(rows):
    batch = rows[ok:ok + BATCH]
    if not batch:
        break
    texts = [r["content"] for r in batch]
    try:
        tb = time.time()
        vectors = bert.encode_passages(texts)
        for j, r in enumerate(batch):
            db.execute("UPDATE facts SET bert_vec = ? WHERE fact_id = ?",
                       (vectors[j].tobytes(), r["fact_id"]))
        ok += len(batch)
        
        if ok % 500 == 0 or ok == len(rows):
            db.commit()
            elapsed = time.time() - t_start
            rate = ok / elapsed
            eta = (len(rows) - ok) / rate if rate > 0 else 0
            print(f"  {ok}/{len(rows)} ({ok/len(rows)*100:.0f}%) rate={rate:.0f}/s eta={eta:.0f}s", flush=True)
    except Exception as e:
        print(f"  OOM at batch_size={BATCH}? {e}", flush=True)
        if BATCH > 5:
            # Halve batch size and retry
            BATCH = max(5, BATCH // 2)
            print(f"  Retrying with batch_size={BATCH}...", flush=True)
            continue
        else:
            fail += len(batch)
            print(f"  FAILED at batch_size={BATCH}: {e}", flush=True)
            break

db.commit()
elapsed = time.time() - t_start
print(f"\nBackfill: {ok} encoded, {fail} failed in {elapsed:.0f}s", flush=True)

# ── Verify ──────────────────────────────────────────────────────────────
final_vec = db.execute("SELECT COUNT(*) FROM facts WHERE bert_vec IS NOT NULL").fetchone()[0]
final_null = db.execute("SELECT COUNT(*) FROM facts WHERE bert_vec IS NULL").fetchone()[0]
print(f"Final: {final_vec} vectors, {final_null} NULL", flush=True)

# Stored vs fresh verification
sample = db.execute("SELECT fact_id, content, bert_vec FROM facts WHERE bert_vec IS NOT NULL LIMIT 1").fetchone()
stored = np.frombuffer(sample["bert_vec"], dtype=np.float32)
fresh = bert.encode_passages(sample["content"])
cos = float(np.dot(stored/np.linalg.norm(stored), fresh[0]/np.linalg.norm(fresh[0])))
print(f"VERIFICATION: stored vs fresh passage cos={cos:.4f}  (must be >0.99!)", flush=True)
if cos < 0.99:
    print("FAILED: vectors still don't match!", flush=True)
    sys.exit(1)

# ── Build FAISS HNSW index ──────────────────────────────────────────────
print(f"\nBuilding FAISS HNSW index (efSearch=200)...", flush=True)
import faiss

all_rows = db.execute("SELECT fact_id, bert_vec FROM facts WHERE bert_vec IS NOT NULL").fetchall()
vectors = np.array([np.frombuffer(r["bert_vec"], dtype=np.float32) for r in all_rows], dtype=np.float32)
fids = [r["fact_id"] for r in all_rows]

dim = 1536
idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
idx.hnsw.efConstruction = 200
idx.hnsw.efSearch = 200
faiss.normalize_L2(vectors)
idx.add(vectors)
print(f"HNSW index: ntotal={idx.ntotal}", flush=True)

# Persist to DB
db.execute("DELETE FROM faiss_state WHERE id = 1")
blob = faiss.serialize_index(idx)
id_map_json = json.dumps(fids)
db.execute(
    "INSERT OR REPLACE INTO faiss_state "
    "(id, index_blob, id_map_blob, fact_count, version, updated_at, bert_vec_version) "
    "VALUES (1, ?, ?, ?, 6, datetime('now'), 6)",
    (blob, id_map_json, len(all_rows))
)
db.commit()
print(f"FAISS cache saved: version=6, {len(all_rows)} vectors, blob={len(blob)//1024//1024}MB", flush=True)

# ── Search test ──────────────────────────────────────────────────────────
print(f"\n=== SEARCH TEST ===", flush=True)
queries = [
    "What tool does the agent use for browser automation?",
    "How is the FAISS index configured?",
    "What provider is the user subscribed to?",
    "Where is the ApplyPilot project located?",
]
for q_text in queries:
    qv = bert.encode_queries(q_text)
    qn = qv[0] / np.linalg.norm(qv[0])
    dists, idxs = idx.search(qn.reshape(1, -1).astype(np.float32), 3)
    top_cos = float(dists[0][0])
    top_fid = fids[idxs[0][0]]
    top_row = db.execute("SELECT content, category FROM facts WHERE fact_id = ?", (top_fid,)).fetchone()
    print(f"  '{q_text}'", flush=True)
    print(f"    top cos={top_cos:.4f}  [{top_row['category']}]  {top_row['content'][:70]}", flush=True)

# Random baseline
rng = np.random.RandomState(42)
rand_cos = []
for _ in range(10):
    rv = rng.randn(1536).astype(np.float32)
    rv = rv / np.linalg.norm(rv)
    d, _ = idx.search(rv.reshape(1, -1), 5)
    rand_cos.append(d[0][0])
print(f"  Random query baseline: top cos mean={np.mean(rand_cos):.4f}", flush=True)

print(f"\n{'='*50}", flush=True)
print(f"ALL DONE — ready to restart Hermes", flush=True)
print(f"{'='*50}", flush=True)

db.close()
