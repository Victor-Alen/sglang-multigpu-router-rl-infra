#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-checkpoint-resume-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,2}"
ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-3,4}"
FIRST_ITERATION="${FIRST_ITERATION:-2}"
FINAL_ITERATION="${FINAL_ITERATION:-4}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/results/slime_qwen3_4b_4gpu/checkpoint_resume/$RUN_ID/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/checkpoint_resume/$RUN_ID}"
PHASE1_ID="$RUN_ID-phase1"
PHASE2_ID="$RUN_ID-phase2"
mkdir -p "$OUTPUT_DIR"

common=(
  TRAINER_GPU_IDS="$TRAINER_GPU_IDS"
  ROLLOUT_GPU_IDS="$ROLLOUT_GPU_IDS"
  CHECKPOINT_DIR="$CHECKPOINT_ROOT"
  SAVE_INTERVAL="$FIRST_ITERATION"
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}"
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
  ROLLOUT_SHUFFLE=0
  ENABLE_DUMP_DETAILS=0
)

env "${common[@]}" \
  NUM_ROLLOUTS="$FIRST_ITERATION" \
  EXPECTED_COMPLETED_ROLLOUTS="$FIRST_ITERATION" \
  RL_ROUTER_RUN_ID="$PHASE1_ID" \
  bash "$ROOT/scripts/run_strict_sync_acceptance.sh"

test "$(<"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")" = "$FIRST_ITERATION" || {
  echo "phase 1 did not publish checkpoint $FIRST_ITERATION" >&2
  exit 1
}

env "${common[@]}" \
  SAVE_INTERVAL="$FINAL_ITERATION" \
  NUM_ROLLOUTS="$FINAL_ITERATION" \
  EXPECTED_COMPLETED_ROLLOUTS="$((FINAL_ITERATION - FIRST_ITERATION))" \
  MIN_POLICY_VERSIONS="$((FINAL_ITERATION - FIRST_ITERATION))" \
  RL_ROUTER_RUN_ID="$PHASE2_ID" \
  bash "$ROOT/scripts/run_strict_sync_acceptance.sh"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_checkpoint_resume.py" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --first-iteration "$FIRST_ITERATION" \
  --final-iteration "$FINAL_ITERATION" \
  --resume-log "$ROOT/results/slime_qwen3_4b_4gpu/strict_sync/$PHASE2_ID/train.log" \
  --output "$OUTPUT_DIR/acceptance.json"

echo "Checkpoint resume acceptance passed: $OUTPUT_DIR/acceptance.json"
