#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-maturity-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,2}"
ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-3,4}"
SOAK_ROLLOUTS="${SOAK_ROLLOUTS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/maturity/$RUN_ID}"
CHECKPOINT_RUN_ID="$RUN_ID-checkpoint"
FAULT_RUN_ID="$RUN_ID-fault"
SOAK_RUN_ID="$RUN_ID-soak"
mkdir -p "$OUTPUT_DIR"

"$PYTHON_ENV/bin/python" -m pytest -q 2>&1 | tee "$OUTPUT_DIR/pytest.log"
AUTOMATED_TESTS="$(grep -oE '[0-9]+ passed' "$OUTPUT_DIR/pytest.log" | tail -1 | awk '{print $1}')"
test -n "$AUTOMATED_TESTS" || { echo "could not parse pytest count" >&2; exit 1; }

bash "$ROOT/scripts/verify_slime_qwen3_4b.sh"
git -C "$SLIME_ROOT" apply --reverse --check "$ROOT/patches/slime_v021_fsdp_optional_ring_flash_attn.patch"
git -C "$SLIME_ROOT" apply --reverse --check "$ROOT/patches/slime_v021_bounded_async_weight_versions.patch"

TRAINER_GPU_IDS="$TRAINER_GPU_IDS" \
ROLLOUT_GPU_IDS="$ROLLOUT_GPU_IDS" \
RL_ROUTER_RUN_ID="$CHECKPOINT_RUN_ID" \
  bash "$ROOT/scripts/run_checkpoint_resume_acceptance.sh"

TRAINER_GPU_IDS="$TRAINER_GPU_IDS" \
ROLLOUT_GPU_IDS="$ROLLOUT_GPU_IDS" \
RL_ROUTER_RUN_ID="$FAULT_RUN_ID" \
  bash "$ROOT/scripts/run_prefetch_fault_acceptance.sh"

SOAK_DIR="$OUTPUT_DIR/soak"
TRAINER_GPU_IDS="$TRAINER_GPU_IDS" \
ROLLOUT_GPU_IDS="$ROLLOUT_GPU_IDS" \
RL_ROUTER_RUN_ID="$SOAK_RUN_ID" \
LOG_DIR="$SOAK_DIR" \
NUM_ROLLOUTS="$SOAK_ROLLOUTS" \
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}" \
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}" \
SAVE_INTERVAL=1000 \
ENABLE_DUMP_DETAILS=0 \
ROLLOUT_SHUFFLE=0 \
  bash "$ROOT/scripts/run_bounded_async_acceptance.sh"

CHECKPOINT_ACCEPTANCE="$ROOT/results/slime_qwen3_4b_4gpu/checkpoint_resume/$CHECKPOINT_RUN_ID/acceptance.json"
FAULT_ACCEPTANCE="$ROOT/results/slime_qwen3_4b_4gpu/fault_injection/$FAULT_RUN_ID/fault_acceptance.json"
"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_maturity_gate.py" \
  --config "$ROOT/configs/maturity_slo.yaml" \
  --automated-tests "$AUTOMATED_TESTS" \
  --checkpoint "$CHECKPOINT_ACCEPTANCE" \
  --fault "$FAULT_ACCEPTANCE" \
  --soak "$SOAK_DIR/acceptance.json" \
  --soak-router "$SOAK_DIR/live_router_acceptance.json" \
  --output "$OUTPUT_DIR/release_gate.json"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/write_run_manifest.py" \
  --project-root "$ROOT" \
  --run-id "$RUN_ID" \
  --status accepted \
  --trainer-gpus "$TRAINER_GPU_IDS" \
  --rollout-gpus "$ROLLOUT_GPU_IDS" \
  --automated-tests "$AUTOMATED_TESTS" \
  --artifact "$CHECKPOINT_ACCEPTANCE" \
  --artifact "$FAULT_ACCEPTANCE" \
  --artifact "$SOAK_DIR/acceptance.json" \
  --artifact "$SOAK_DIR/live_router_acceptance.json" \
  --artifact "$OUTPUT_DIR/release_gate.json" \
  --artifact "$ROOT/configs/maturity_slo.yaml" \
  --output "$OUTPUT_DIR/manifest.json"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_run_manifest.py" \
  --manifest "$OUTPUT_DIR/manifest.json"

echo "Maturity acceptance passed: $OUTPUT_DIR"
