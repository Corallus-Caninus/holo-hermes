#!/usr/bin/env bash
set -euo pipefail

# belt_fed - Run a prompt through Hermes non-interactively, verify completion,
# and retry until the task is confirmed done.
#
# Usage:
#   ./belt_fed.sh "<your goal/task prompt>"
#
# Self-contained within the holo-hermes repo.  Resolves the launcher as a
# sibling (fully_automatic_holographic) by default.  Override with the
# HERMES_LAUNCHER env var or set HERMES_QUIET=-Q to control chat mode.
#
# The script:
#   1. Sends the prompt to Hermes (non-interactive, quiet mode)
#   2. Sends a verification prompt that asks if the task goal was completed
#   3. The verification model MUST respond with ONLY "YES" or "NO"
#   4. If NO → go back to step 1
#   5. If YES → exit success

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"<goal/task prompt>\""
  exit 1
fi

PROMPT="$1"
QUIET="${HERMES_QUIET:--Q}"

# Resolve the repo root so belt_fed works from any working directory
# (including when symlinked / copied elsewhere).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the repo's patched holographic launcher by default.
# Prerequisite: ~/.hermes/patches must be a symlink to patches/ in this repo.
HERMES_CMD="${HERMES_LAUNCHER:-$SCRIPT_DIR/fully_automatic_holographic}"

# Strip leading/trailing quotes if user passed them as part of the string
PROMPT="${PROMPT#\"}"
PROMPT="${PROMPT%\"}"
PROMPT="${PROMPT#\'}"
PROMPT="${PROMPT%\'}"

VERIFY_PROMPT="I gave you this goal: '${PROMPT}'. Was it completed successfully in the previous run? Answer with ONLY YES or NO. No other text, no punctuation, no explanation."

echo "[belt_fed] Goal: ${PROMPT}"
echo "[belt_fed] Starting loop..."

while true; do
  echo ""
  echo "========================================"
  echo "  EXECUTION ATTEMPT"
  echo "========================================"

  # Step 1: Run the goal prompt through Hermes
  OUTPUT=$(${HERMES_CMD} chat ${QUIET} -q "${PROMPT}" 2>&1)
  EXIT_CODE=$?
  echo "[belt_fed] Hermes exit code: ${EXIT_CODE}"
  echo ""
  echo "${OUTPUT}"

  # Step 2: Run the verification prompt
  echo ""
  echo "--- Verification ---"
  RESULT=$(${HERMES_CMD} chat ${QUIET} -q "${VERIFY_PROMPT}" 2>&1)
  echo "${RESULT}"

  # Step 3: Parse the verification output for YES or NO
  # Strip whitespace and match whole-word YES/NO
  TRIMMED=$(echo "${RESULT}" | head -1 | tr -d '[:space:]')
  if echo "${TRIMMED}" | grep -qiE '^YES$'; then
    echo ""
    echo "========================================"
    echo "  ✓ TASK COMPLETED SUCCESSFULLY"
    echo "========================================"
    exit 0
  elif echo "${TRIMMED}" | grep -qiE '^NO$'; then
    echo ""
    echo "========================================"
    echo "  ✗ Task not yet complete — retrying..."
    echo "========================================"
    sleep 1
    continue
  else
    # Fallback: search the whole output for YES/NO (the model may add extra text)
    if echo "${RESULT}" | grep -qiE '\bYES\b'; then
      echo ""
      echo "========================================"
      echo "  ✓ TASK COMPLETED SUCCESSFULLY"
      echo "========================================"
      exit 0
    elif echo "${RESULT}" | grep -qiE '\bNO\b'; then
      echo ""
      echo "========================================"
      echo "  ✗ Task not yet complete — retrying..."
      echo "========================================"
      sleep 1
      continue
    else
      echo "[belt_fed] WARNING: Could not parse YES/NO from verification response."
      echo "[belt_fed] Raw verification output shown above. Retrying..."
      sleep 2
      continue
    fi
  fi
done
