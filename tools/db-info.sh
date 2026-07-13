#!/usr/bin/env bash
# Hermes SQLite Database Info
# Shows row counts and sizes for all Hermes SQLite databases.
set -euo pipefail

DB_DIR="${HERMES_HOME:-$HOME/.hermes}"

echo "▌ Hermes Database Info"
echo "▌─────────────────────"
echo ""

python3 -c "
import sqlite3, os

db_dir = os.path.expanduser('$DB_DIR')
found = 0

db_labels = {
    'state.db':       'Session State',
    'memory_store.db':'Holographic Memory',
    'training.db':    'Training Data',
}

for db_name in sorted(db_labels.keys()):
    db_path = os.path.join(db_dir, db_name)
    if not os.path.exists(db_path):
        continue
    found += 1
    size_bytes = os.path.getsize(db_path)
    size_str = f'{size_bytes/1024:.0f}K' if size_bytes < 1024*1024 else f'{size_bytes/(1024*1024):.1f}M'
    label = db_labels.get(db_name, db_name)
    conn = sqlite3.connect(db_path)
    conn.text_factory = str
    cur = conn.cursor()
    try:
        cur.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%_fts%' AND name NOT LIKE '%_config' AND name NOT LIKE '%_content' AND name NOT LIKE '%_data' AND name NOT LIKE '%_docsize' AND name NOT LIKE '%_idx' AND name NOT LIKE 'sqlite_%' ORDER BY name\")
        tables = cur.fetchall()
    except sqlite3.OperationalError:
        tables = []
    conn.close()

    print(f'  📦  {label}  ({size_str})')
    print(f'      File: {db_name}')
    if not tables:
        print('      (no tables)')
    else:
        print(f'      Tables:')
        for t in tables:
            tname = t[0]
            try:
                c = sqlite3.connect(db_path)
                c.text_factory = str
                cnt = c.execute(f'SELECT COUNT(*) FROM \"{tname}\"').fetchone()[0]
                c.close()
                print(f'        \u2022 {tname:30s}  {cnt:>6,} rows')
            except Exception:
                print(f'        \u2022 {tname:30s}      ? rows')
    print('')

if found == 0:
    print(f'  No Hermes databases found in {db_dir}')
" 2>&1
