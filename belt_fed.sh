#!/usr/bin/env bash
set -euo pipefail

# belt_fed - Run a prompt through Hermes non-interactively, verify completion,
# and retry until the task is confirmed done.
#
# Usage:
#   ./belt_fed.sh "<goal/task prompt>"
#   cat prompt.txt | ./belt_fed.sh
#
# Environment overrides:
#   HERMES_LAUNCHER    — Hermes command
#                        (default: sibling fully_automatic_holographic)
#   BELT_FED_TIMEOUT   — per-attempt timeout in seconds (default: 600 = 10m)
#
# The script:
#   1. Sends the prompt to Hermes (non-interactive, quiet mode)
#   2. Sends a short verification prompt ("was it done? answer YES/NO")
#   3. If NO  → go back to step 1
#   4. If YES → exit success

# ---------------------------------------------------------------------------
# Resolve the repo root so belt_fed works from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HERMES_CMD="${HERMES_LAUNCHER:-}"
if [ -z "$HERMES_CMD" ]; then
  if [ -x "$SCRIPT_DIR/fully_automatic_holographic" ]; then
    HERMES_CMD="$SCRIPT_DIR/fully_automatic_holographic"
  else
    HERMES_CMD="hermes"
  fi
fi
TIMEOUT="${BELT_FED_TIMEOUT:-600}"   # 10 minutes default
QUIET="-Q"

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
# Short summary (first line, max 200 chars) for the verification prompt
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
  # In -Q mode, hermes suppresses all tool-call progress output.
  # Only the final model response goes to stdout. Stderr has the
  # session_id banner.  We capture both and print stdout so the
  # user can see what the model ultimately produced.
  echo "[belt_fed] Running Hermes (up to ${TIMEOUT}s)..."
  set +e
  HERMES_OUTPUT=$(timeout "$TIMEOUT" "${HERMES_CMD}" chat ${QUIET} -q "${PROMPT}" 2>&1)
  EXIT_CODE=$?
  set -e
  OUTPUT="${HERMES_OUTPUT:-}"
  # Strip Python tracebacks from timeout kills — they're noise, not errors.
  OUTPUT="$(printf '%s' "${OUTPUT}" | sed '/^Traceback.*/,/^KeyboardInterrupt/D' 2>/dev/null || printf '%s' "${OUTPUT}")"
  echo "[belt_fed] Exit code: ${EXIT_CODE}"
  echo ""
  echo "${OUTPUT}"

  # The execution step did its work via tool calls (file edits, etc.).
  # Now verify whether the goal was met using a short prompt.

  # Step 2: Short verification — runs quickly since the prompt is just
  # the first 200 chars of the goal.  The verification agent evaluates
  # based on filesystem state (side effects from step 1).
  VERIFY_PROMPT="I gave you this goal: '${GOAL_SUMMARY}'. Was it completed? Answer ONLY YES or NO."
  echo ""
  echo "--- Verification ---"
  set +e
  VERIFY_OUTPUT=$(timeout "$TIMEOUT" "${HERMES_CMD}" chat ${QUIET} -q "${VERIFY_PROMPT}" 2>&1)
  VEXIT_CODE=$?
  set -e
  RESULT="${VERIFY_OUTPUT:-}"
  # Strip same traceback noise from verification output
  RESULT="$(printf '%s' "${RESULT}" | sed -r '/^Exception ignored/,/^KeyboardInterrupt/d; /^Traceback/,/^KeyboardInterrupt/d' 2>/dev/null || printf '%s' "${RESULT}")"
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
