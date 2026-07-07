"""Embedding via gte-Qwen2-1.5B-instruct on MI25 (ROCm) with 32K context.

Uses PyTorch with ROCm for GPU acceleration, falls back to CPU if GPU unavailable.
Produces 1536-dim normalized embeddings — replaces the old 384-dim ONNX MiniLM.

Usage:
    from agent.bert_embed import encode, get_dim
    vec = encode("your query text")  # -> np.ndarray[1536]
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DIM = 1536
_MAX_SEQ_LEN = 32768
_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
_INSTRUCTION = os.environ.get("HERMES_EMBED_INSTRUCTION",
                               "Retrieve relevant information for the current conversation")

_model = None
_tokenizer = None
_device = None


def _ensure_libs() -> None:
    """Pre-load Nix system libraries needed by torch C extensions.

    Setting LD_LIBRARY_PATH via os.environ has no effect after the process
    has started — the dynamic linker reads it at process launch.  Instead
    we use ctypes.CDLL to load each required .so file explicitly before
    torch tries to find them.
    """
    _libs = [
        "/nix/store/6cf2yj12gf51jn5vdbdw01gmgvyj431s-zstd-1.5.6/lib/libzstd.so.1",
        "/nix/store/vpg96mfr1jw5arlqg831i69g29v0sdb3-zlib-1.3.1/lib/libz.so.1",
    ]
    import ctypes
    for _lib in _libs:
        if os.path.exists(_lib):
            try:
                ctypes.CDLL(_lib, ctypes.RTLD_GLOBAL)
            except Exception:
                pass  # might already be loaded


def _last_token_pool(last_hidden_states, attention_mask):
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _load() -> None:
    """Load model + tokenizer lazily. Runs on MI25 via ROCm if available."""
    global _model, _tokenizer, _device

    if _model is not None:
        return

    # Ensure LD_LIBRARY_PATH has Nix system libs (zstd for torch C extensions)
    _ensure_libs()

    # Ensure faiss-venv is in sys.path (Hermes boot chain may clear it)
    _venv_site = "/tmp/faiss-venv/lib/python3.12/site-packages"
    import sys as _sys
    if _venv_site not in _sys.path:
        _sys.path.insert(0, _venv_site)

    # Enable expandable segments before torch import to avoid HIP OOM
    # from memory fragmentation on AMD MI25 (gfx900).
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

    import torch
    # Try to clear stale GPU memory from previous crashed processes
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    _embed_device = os.environ.get("HERMES_EMBED_DEVICE", "").lower()
    if _embed_device == "cpu":
        _device = torch.device("cpu")
        logger.info("Forced CPU via HERMES_EMBED_DEVICE=cpu")
    else:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logger.info("Using MI25 via ROCm (will fall back to CPU if NaN detected)")

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    config.rope_theta = 1000000.0  # compatibility with transformers >= 5.x
    config.use_cache = False       # disable KV cache (not needed for embeddings)
    config.torch_dtype = "float32"  # MI25/gfx900 does NOT support bfloat16

    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    # Decoder models (Qwen2) need LEFT padding for last-token pooling.
    # Default HuggingFace padding_side is "right", which puts pad tokens at the
    # end — _last_token_pool would pick up pad tokens instead of content.
    _tokenizer.padding_side = "left"
    try:
        _model = AutoModel.from_pretrained(
            _MODEL_NAME,
            config=config,
            torch_dtype=torch.float32,
        ).to(_device)
    except (torch.OutOfMemoryError, RuntimeError) as _oom_e:
        err_str = str(_oom_e)
        if "HIP out of memory" in err_str or "CUDA out of memory" in err_str:
            logger.warning(
                "GPU OOM loading BERT model (%s) — falling back to CPU. "
                "Embedding will be slower but functional.",
                err_str[:120],
            )
            _device = torch.device("cpu")
            _model = AutoModel.from_pretrained(
                _MODEL_NAME,
                config=config,
                torch_dtype=torch.float32,
            ).to("cpu")
        else:
            raise

    if _device.type == "cpu":
        _model = _model.to(_device)
    _model.eval()

    logger.info(
        "gte-Qwen2-1.5B-instruct loaded on %s (%d-dim, %d context)",
        _device, _DIM, _MAX_SEQ_LEN,
    )
    # Write load diagnostic to debug log
    try:
        _log_path = os.path.expanduser("~/.hermes/holographic_debug.log")
        with open(_log_path, "a") as _f:
            import json as _jj
            _f.write(_jj.dumps({"timestamp": __import__("datetime").datetime.now().isoformat(),
                                "event": "bert_load", "device": str(_device), "ok": True}) + "\n")
    except Exception:
        pass


def encode(text: str | list[str]) -> np.ndarray:
    """Encode text(s) into normalized 1536-dim embedding vectors.

    Args:
        text: Single string or list of strings.

    Returns:
        np.ndarray of shape (1, 1536) for single input,
        or (N, 1536) for a list of N inputs.
    """
    _load()
    assert _model is not None and _tokenizer is not None

    import torch

    texts = [text] if isinstance(text, str) else list(text)
    if not texts:
        return np.zeros((0, _DIM), dtype=np.float32)

    # gte-Qwen2-1.5B-instruct expects a task prefix
    # Use "query: " for search queries, "passage: " for stored documents.
    # Since we encode both queries AND documents through this function,
    # we detect which one it is by caller context.  For simplicity,
    # queries use "query: " prefix, backfill/docs use "passage: ".
    # Callers can add the prefix themselves if needed.
    prefixed = texts

    # Free GPU cache before encoding to avoid OOM from fragmented memory
    if _device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    inputs = _tokenizer(
        prefixed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_MAX_SEQ_LEN,
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        # Last-token pooling (per model card — gte-Qwen2 is decoder-based)
        embeddings = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def encode_queries(texts: str | list[str]) -> np.ndarray:
    """Encode with instruct + Query prefix for retrieval queries."""
    t = [texts] if isinstance(texts, str) else texts
    return encode([f"Instruct: {_INSTRUCTION}\nQuery: {t}" for t in t])


def encode_passages(texts: str | list[str]) -> np.ndarray:
    """Encode stored passages as raw text (no prefix, per model card).
    
    gte-Qwen2 uses asymmetric prefixes: queries get 'Instruct: ... Query:',
    passages get raw text. The contrastive training aligns them into a shared
    embedding space despite different prefixes.
    """
    t = [texts] if isinstance(texts, str) else texts
    return encode(t)  # raw text, no prefix


def get_dim() -> int:
    """Return embedding dimension (1536)."""
    return _DIM
