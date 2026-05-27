# Holographic Memory Fork — Hermes Agent Patches

> **Platform: NixOS / Nix Hermes only.** These patches override Hermes modules
> installed via Nix (read-only store). For pip/Docker Hermes installs, you can
> apply the patches directly — see [Non-Nix Hermes](#non-nix-hermes) below.

Replaces the flat-file `MEMORY.md`/`USER.md` store with the holographic
`fact_store` (SQLite + FTS5 + HRR vector retrieval). Background review
writes to `fact_store` every turn. Injects relevant facts per turn into
the ephemeral system prompt via a char-budget prefetch (min 20 facts,
~5000 char target, score-gated).

## Quick Install

### Prerequisites

- A working [Hermes Agent](https://github.com/NousResearch/hermes-agent) Nix install:
  ```bash
  nix profile install github:NousResearch/hermes-agent
  ```
- `gh` authenticated or SSH key set up for GitHub
- `~/.local/bin` should be early in your `PATH` (add `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` or `~/.profile` if not already there)

### Install

```bash
# Clone the repo
git clone https://github.com/Corallus-Caninus/holo-hermes.git ~/Code/hermes/holo-hermes

# Run the installer
~/Code/hermes/holo-hermes/install.sh
```

The installer will:

1. **Detect your Nix Hermes installation** — auto-discovers store paths
2. **Copy patches** to `~/.hermes/patches/`
3. **Create entry point wrappers** at `~/.local/bin/hermes` and `~/.local/bin/hermes-agent`
4. **Auto-configure `~/.hermes/config.yaml`** — merges the required `memory:` and `plugins:` settings using the Hermes Python's PyYAML (preserves all your existing settings)

### Config changes (auto-applied by install.sh)

The installer automatically merges these settings into `~/.hermes/config.yaml`
using the Hermes Python's PyYAML — no manual editing needed:

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

Existing settings in your config are preserved — only these specific keys
are added or updated.

### Usage

```bash
hermes
```

That's it — the `~/.local/bin/hermes` wrapper takes precedence over the Nix store
version, injects the patches, and launches Hermes normally.

---

## How It Works

### Architecture

```
                    ~/.local/bin/hermes (bash wrapper)
                    │
                    │  sets HERMES_* env vars from Nix store paths
                    │  execs ~/.local/bin/hermes-agent
                    ▼
         ~/.local/bin/hermes-agent (Python entry point)
                    │
                    │  inserts ~/.hermes/patches into sys.path
                    │  calls hermes_cli.main:main()
                    ▼
              Hermes boots normally
                    │
                    │  Python import order:
                    │  sys.path[0] = ~/.hermes/patches  ← wins over Nix store
                    ▼
         agent.background_review  ✓ patched
         agent.conversation_loop  ✓ patched
         agent.system_prompt      ✓ patched
         plugins.memory.holographic  ✓ patched
```

### Why entry point wrappers?

Nix Hermes is installed in the read-only Nix store. The `~/.local/bin/hermes`
wrapper intercepts the launch and inserts our patched modules into Python's
import path *before* the boot chain loads the originals.

The `~/.hermes/patches/` directory contains proxy packages that shadow the
Nix-store originals via `__path__` extension and direct replacement of key
modules (background_review, conversation_loop, system_prompt, holographic
memory plugin).

### Manual install (if install.sh doesn't cut it)

If `install.sh` doesn't work for your setup, here's what it does manually:

**1. Copy patches:**
```bash
cp -r ~/Code/hermes/holo-hermes/patches ~/.hermes/patches
rm -rf ~/.hermes/patches/*/__pycache__
```

**2. Find your Nix Hermes Python:**
```bash
grep HERMES_PYTHON "$(which hermes)"
# Should output something like:
# export HERMES_PYTHON='/nix/store/xxx...xx-hermes-agent-env/bin/python3'
```

**3. Create `~/.local/bin/hermes-agent`** (Python entry point):
```python
#!/usr/bin/env python3
import os, sys
patch_dir = os.path.expanduser("~/.hermes/patches")
if patch_dir not in sys.path:
    sys.path.insert(0, patch_dir)
from hermes_cli.main import main
if __name__ == "__main__":
    sys.exit(main())
```

**4. Create `~/.local/bin/hermes`** (bash wrapper):
```bash
#!/usr/bin/env bash
# Copy the export lines from your Nix hermes wrapper:
#   $(which hermes) | head -30
# Then add:
PATCHED_ENTRY="$HOME/.local/bin/hermes-agent"
exec "$HERMES_PYTHON" "$PATCHED_ENTRY" "$@"
```

---

## Files

| Path | What it does |
|------|-------------|
| `install.sh` | Auto-installer — detects Nix paths, copies patches, creates wrappers |
| `fully_automatic_holographic` | Standalone launcher (alternative to bash wrapper approach) |
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

## Non-Nix Hermes

If you installed Hermes via pip, Docker, or another method (not Nix), the
patches still work — you just don't need the entry-point wrapper approach.
Instead, copy the patches into your Hermes Python environment directly:

```bash
# Find where Hermes is installed
python3 -c "import agent; print(agent.__file__)"
# e.g. /usr/lib/python3.12/site-packages/agent/__init__.py

# Clone the repo
git clone https://github.com/Corallus-Caninus/holo-hermes.git ~/Code/hermes/holo-hermes

# Copy patches over the originals (overwrites specific files)
cp ~/Code/hermes/holo-hermes/patches/agent/background_review.py /path/to/site-packages/agent/
cp ~/Code/hermes/holo-hermes/patches/agent/conversation_loop.py /path/to/site-packages/agent/
cp ~/Code/hermes/holo-hermes/patches/agent/system_prompt.py /path/to/site-packages/agent/
cp ~/Code/hermes/holo-hermes/patches/agent/agent_init.py /path/to/site-packages/agent/
cp ~/Code/hermes/holo-hermes/patches/plugins/memory/holographic/__init__.py /path/to/site-packages/plugins/memory/holographic/
```

Then apply the [config changes](#config-changes-auto-applied-by-installsh) to
`~/.hermes/config.yaml` manually and run `hermes` normally.

> Note: the auto-config step in `install.sh` only works for Nix Hermes (it
> uses the Hermes Python's PyYAML). For non-Nix installs, just copy-paste
> the YAML snippet from the config section above into your `~/.hermes/config.yaml`.

> Note: the proxy package files (`patches/agent/__init__.py`,
> `patches/plugins/__init__.py`, `patches/plugins/memory/__init__.py`) are only
> needed for the Nix sys.path-override mechanism. For pip installs you can
> skip them and replace files directly.
