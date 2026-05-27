# Holographic Memory Fork — Hermes Agent Patches

Replaces the flat-file MEMORY.md/USER.md store with the holographic
`fact_store` (SQLite + FTS5 + HRR vector retrieval). Background review
writes to fact_store every turn. Injects relevant facts per turn into
the ephemeral system prompt via a char-budget prefetch (min 20 facts,
~5000 char target, score-gated).

## Files

| Path | What it does |
|------|-------------|
| `patches/agent/__init__.py` | Proxy package: extends `__path__` so patched submodules shadow Nix-store originals |
| `patches/agent/agent_init.py` | Strips dead MemoryStore init block; keeps `nudge_interval` from config |
| `patches/agent/background_review.py` | Cursor-safe TTY spinner; detaches `_memory_manager` before shutdown to avoid destroying parent's provider |
| `patches/agent/_bg_review_spinner.py` | Custom spinner writing to `/dev/tty` with ANSI cursor-save/restore (doesn't overwrite input box) |
| `patches/agent/context_compressor.py` | Updated compression notes referencing holographic store |
| `patches/agent/conversation_loop.py` | Injects prefetched facts into `ephemeral_system_prompt` each turn; fixed review trigger gate and exec globals; prefetch query includes previous assistant response + reasoning/thinking |
| `patches/agent/system_prompt.py` | Replaced `MEMORY_GUIDANCE` with `FACT_STORE_GUIDANCE` |
| `patches/plugins/__init__.py` | (empty) |
| `patches/plugins/memory/__init__.py` | Proxy package: extends `__path__`, re-exports from real module, overrides `_MEMORY_PLUGINS_DIR` |
| `patches/plugins/memory/holographic/__init__.py` | Char-budget prefetch (min 20 facts, ~5000 char target, score >= 0.15); lazy-init guard if `_store` is None; top-5 in cached system prompt |
| `fully_automatic_holographic` | Launcher script: proactive import hack to load patches before Nix-store boot chain |

## Config changes (config.yaml)

```yaml
memory:
  memory_enabled: false
  user_profile_enabled: false
  provider: holographic
  nudge_interval: 1

plugins:
  hermes-memory-store:
    auto_extract: true
    min_trust_threshold: 0.1
```

## Usage

```bash
~/Code/hermes/fully_automatic_holographic
```

## Bugs fixed along the way

1. **`_install_safe_stdio` NameError** — exec globals used `dict(globals())` which
   skipped underscore-prefixed names. Fixed: `_real_mod.__dict__.copy()`.
2. **Throbber overwrites input box** — `\r` animation landed on the current
   cursor line (inside the prompt_toolkit input area). Fixed: custom spinner
   writes to `/dev/tty` with `\033[s` / `\033[u` cursor-save/restore.
3. **fact_store breaks after background review** — review fork shared the
   parent's `_memory_manager`, then called `shutdown_memory_provider()` →
   `shutdown_all()` → `provider.shutdown()` → set `_store = None`.
   Fixed: detach `_memory_manager` before review cleanup.
4. **Background review spinner `NameError: name 'time' is not defined`** —
   extracted spinner class was missing imports. Fixed: added `os`, `threading`, `time`.
