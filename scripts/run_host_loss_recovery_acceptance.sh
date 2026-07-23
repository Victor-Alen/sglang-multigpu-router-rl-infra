#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-host-loss-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
FIRST_ITERATION="${FIRST_ITERATION:-2}"
FINAL_ITERATION="${FINAL_ITERATION:-4}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/results/slime_qwen3_4b_4gpu/host_loss/$RUN_ID/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/host_loss/$RUN_ID}"
PHASE1_ID="$RUN_ID-interrupted"
PHASE2_ID="$RUN_ID-recovered"
PHASE1_LOG="$OUTPUT_DIR/phase1_interrupted.log"
PHASE2_LOG="$ROOT/results/slime_qwen3_4b_4gpu/strict_sync/$PHASE2_ID/train.log"
mkdir -p "$OUTPUT_DIR"

if pgrep -u "$(id -u)" -f '(^|/)raylet([[:space:]]|$)' >/dev/null; then
  echo "a Ray cluster already exists for this user; refusing host-loss injection" >&2
  exit 2
fi

common=(
  TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,2}"
  ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-3,4}"
  CHECKPOINT_DIR="$CHECKPOINT_ROOT"
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}"
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
  ROLLOUT_SHUFFLE=0
  ENABLE_DUMP_DETAILS=0
)

set +e
env "${common[@]}" \
  SAVE_INTERVAL="$FIRST_ITERATION" \
  NUM_ROLLOUTS="${PHASE1_NUM_ROLLOUTS:-20}" \
  RL_ROUTER_RUN_ID="$PHASE1_ID" \
  bash "$ROOT/scripts/run_slime_qwen3_4b_4xa6000.sh" >"$PHASE1_LOG" 2>&1 &
phase1_pid=$!
set -e

deadline=$((SECONDS + ${CHECKPOINT_WAIT_SECONDS:-900}))
while [[ ! -s "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]] ||
      [[ "$(<"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")" -lt "$FIRST_ITERATION" ]]; do
  if ! kill -0 "$phase1_pid" >/dev/null 2>&1; then
    wait "$phase1_pid" || phase1_rc=$?
    echo "phase 1 exited before checkpoint $FIRST_ITERATION (rc=${phase1_rc:-0})" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    "$PYTHON_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
    echo "timed out waiting for checkpoint $FIRST_ITERATION" >&2
    exit 1
  fi
  sleep 2
done

interrupted_ns="$(date +%s%N)"
"$PYTHON_ENV/bin/ray" stop --force
set +e
wait "$phase1_pid"
phase1_rc=$?
set -e
if [[ "$phase1_rc" -eq 0 ]]; then
  echo "phase 1 unexpectedly succeeded after host-loss injection" >&2
  exit 1
fi

env "${common[@]}" \
  SAVE_INTERVAL="$FINAL_ITERATION" \
  NUM_ROLLOUTS="$FINAL_ITERATION" \
  EXPECTED_COMPLETED_ROLLOUTS="$((FINAL_ITERATION - FIRST_ITERATION))" \
  MIN_POLICY_VERSIONS="$((FINAL_ITERATION - FIRST_ITERATION))" \
  RL_ROUTER_RUN_ID="$PHASE2_ID" \
  bash "$ROOT/scripts/run_strict_sync_acceptance.sh"
recovered_ns="$(date +%s%N)"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_checkpoint_resume.py" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --first-iteration "$FIRST_ITERATION" \
  --final-iteration "$FINAL_ITERATION" \
  --resume-log "$PHASE2_LOG" \
  --output "$OUTPUT_DIR/checkpoint_acceptance.json"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_host_loss_recovery.py" \
  --checkpoint-acceptance "$OUTPUT_DIR/checkpoint_acceptance.json" \
  --phase1-exit-code "$phase1_rc" \
  --interrupted-ns "$interrupted_ns" \
  --recovered-ns "$recovered_ns" \
  --max-recovery-seconds "${MAX_HOST_RECOVERY_SECONDS:-900}" \
  --output "$OUTPUT_DIR/host_loss_acceptance.json"

echo "Private single-node host-loss recovery passed: $OUTPUT_DIR/host_loss_acceptance.json"
