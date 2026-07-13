#!/usr/bin/env bash
# Print holographic fact system debug log entries as pretty-printed JSON.
# Usage:
#   ./print_fact_machine_turns.sh           # last 5 turns
#   ./print_fact_machine_turns.sh 20        # last 20 turns
#   ./print_fact_machine_turns.sh -f        # full log (all turns)
#   ./print_fact_machine_turns.sh -s        # summary mode (compact one-liners)

set -euo pipefail

LOG=~/.hermes/holographic_debug.log

if [ ! -f "$LOG" ]; then
    echo "No log file found at $LOG" >&2
    exit 1
fi

MODE="tail"
COUNT=5

# Parse args
for arg in "$@"; do
    case "$arg" in
        -f|--full) MODE="full" ;;
        -s|--summary) MODE="summary" ;;
        *) COUNT="$arg" ;;
    esac
done

case "$MODE" in
    full)
        INPUT="cat $LOG"
        ;;
    summary)
        # Compact per-turn summary with source breakdown
        INPUT="cat $LOG"
        python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    ts = r['timestamp'][:19]
    n_pre = len(r['prefetched_facts'])
    n_ext = len(r['extracted_facts'])
    n_sco = len(r['scored_facts'])
    sources = {}
    for f in r['prefetched_facts']:
        s = f.get('source', '?')
        sources[s] = sources.get(s, 0) + 1
    src_str = ', '.join(f'{k}={v}' for k, v in sorted(sources.items()))
    conv = r.get('conversation', '')[:100].replace(chr(10), ' ')
    sid = r.get('session_id', '')[:12]
    print(f'{ts}  [{sid}]  [{src_str}]  pre={n_pre} ext={n_ext} sco={n_sco}')
    print(f'  {conv}')
    print()
" < "$LOG"
        exit 0
        ;;
    tail|*)
        INPUT="tail -n $COUNT $LOG"
        ;;
esac

# Pretty-print mode
eval "$INPUT" | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        print('[skipping invalid JSON line]')
        continue
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print('---')
"
