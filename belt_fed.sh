#!/usr/bin/env bash
set -euo pipefail

# belt_fed - Run a prompt through Hermes non-interactively, verify completion,
# and retry until the task is confirmed done.
#
# Usage:
#   ./belt_fed.sh "<goal/task prompt>"
#
# For large prompts, pipe via stdin (avoids shell arg limits):
#   cat prompt.txt | ./belt_fed.sh
#
# Environment overrides:
#   HERMES_LAUNCHER    — Hermes command
#                        (default: sibling fully_automatic_holographic)
#   HERMES_QUIET       -Q for quiet (default), empty for verbose
#   BELT_FED_TIMEOUT   — per-attempt timeout in seconds (default: 600 = 10m)
#
# The script:
#   1. Sends the prompt to Hermes (non-interactive, quiet mode)
#   2. Sends a verification prompt that asks if the task goal was completed
#   3. The verification model MUST respond with ONLY "YES" or "NO"
#   4. If NO  → go back to step 1
#   5. If YES → exit success

# ---------------------------------------------------------------------------
# Resolve the repo root so belt_fed works from any working directory
# (including when symlinked / copied elsewhere).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HERMES_CMD="${HERMES_LAUNCHER:-$SCRIPT_DIR/fully_automatic_holographic}"
# Quiet mode: -Q by default. Set HERMES_QUIET=0 to see full output.
case "${HERMES_QUIET:-1}" in 0|false|no) QUIET="" ;; *) QUIET="-Q" ;; esac
TIMEOUT="${BELT_FED_TIMEOUT:-600}"   # 10 minutes default

# ---------------------------------------------------------------------------
# Read the goal prompt: from first argument, or stdin if piped
# ---------------------------------------------------------------------------
PROMPT=""
if [ $# -ge 1 ] && [ -n "$1" ]; then
  PROMPT="$1"
  # Strip leading/trailing quotes if user passed them as part of the string
  PROMPT="${PROMPT#\"}"
  PROMPT="${PROMPT%\"}"
  PROMPT="${PROMPT#\'}"
  PROMPT="${PROMPT%\'}"
elif [ ! -t 0 ]; then
  PROMPT="$(cat)"
else
  echo "Usage: $0 \"<goal/task prompt>\""
  echo "   or: cat prompt.txt | $0"
  exit 1
fi

if [ -z "$PROMPT" ]; then
  echo "Error: empty prompt"
  exit 1
fi

# ---------------------------------------------------------------------------
# Short summary (first 200 chars) for the verification prompt
# ---------------------------------------------------------------------------
GOAL_SUMMARY="$(printf '%s' "$PROMPT" | head -c 200)"
echo "[belt_fed] Goal: ${GOAL_SUMMARY}..."
echo "[belt_fed] Timeout: ${TIMEOUT}s per attempt"
echo "[belt_fed] Starting loop..."

while true; do
  echo ""
  echo "========================================"
  echo "  EXECUTION ATTEMPT"
  echo "========================================"

  # Step 1: Run the goal prompt through Hermes (with timeout).
  # Pass via -q with the prompt as a quoted argument. Stderr to /dev/null
  # to suppress shutdown-traceback noise from timeout kills.
  echo "[belt_fed] Running Hermes (up to ${TIMEOUT}s)..."
  OUTPUT=$(timeout "$TIMEOUT" "${HERMES_CMD}" chat ${QUIET} -q "${PROMPT}" 2>/dev/null || true)
  EXIT_CODE=$?
  echo "[belt_fed] Exit code: ${EXIT_CODE}"
  echo ""
  echo "${OUTPUT}"

  # Step 2: Build a short verification prompt and run it
  VERIFY_PROMPT="I gave you this goal: '${GOAL_SUMMARY}'. Was it completed successfully? Answer ONLY YES or NO."
  echo ""
  echo "--- Verification ---"
  RESULT=$(timeout "$TIMEOUT" "${HERMES_CMD}" chat ${QUIET} -q "${VERIFY_PROMPT}" 2>/dev/null || true)
  echo "${RESULT}"

  # Step 3: Parse for YES or NO
  TRIMMED=$(printf '%s' "${RESULT}" | head -1 | tr -d '[:space:]')
  if printf '%s' "${TRIMMED}" | grep -qiE '^YES$'; then
    echo ""
    echo "========================================"
    echo "  ✓ TASK COMPLETED SUCCESSFULLY"
    echo "========================================"
    exit 0
  elif printf '%s' "${TRIMMED}" | grep -qiE '^NO$'; then
    echo ""
    echo "========================================"
    echo "  ✗ Not yet complete — retrying..."
    echo "========================================"
    sleep 1
    continue
  else
    # Fallback: grep whole output for YES/NO
    if printf '%s' "${RESULT}" | grep -qiE '\bYES\b'; then
      echo ""
      echo "========================================"
      echo "  ✓ TASK COMPLETED SUCCESSFULLY"
      echo "========================================"
      exit 0
    elif printf '%s' "${RESULT}" | grep -qiE '\bNO\b'; then
      echo ""
      echo "========================================"
      echo "  ✗ Not yet complete — retrying..."
      echo "========================================"
      sleep 1
      continue
    else
      echo "[belt_fed] WARNING: Could not parse YES/NO from verification."
      echo "[belt_fed] Raw output shown above. Retrying..."
      sleep 2
      continue
    fi
  fi
done
