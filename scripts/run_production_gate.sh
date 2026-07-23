#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

ACTOR_RUN_ID="${ACTOR_RUN_ID:-actor-death-prod3}"
NETWORK_RUN_ID="${NETWORK_RUN_ID:-network-partition-prod4}"
HOST_RUN_ID="${HOST_RUN_ID:-host-loss-prod1}"
SOAK_RUN_ID="${SOAK_RUN_ID:-long-soak-prod1}"
OUTPUT="${OUTPUT:-$ROOT/results/slime_qwen3_4b_4gpu/production_gate/$SOAK_RUN_ID/production_gate.json}"
OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"

test_output="$("$PYTHON_ENV/bin/python" -m pytest -q "$ROOT/tests" 2>&1 | tee "$OUTPUT_DIR/pytest.log")"
printf '%s\n' "$test_output"
automated_tests="$(
  printf '%s\n' "$test_output" |
    grep -Eo '[0-9]+ passed' |
    tail -n 1 |
    awk '{print $1}'
)"
if [[ -z "$automated_tests" ]]; then
  echo "Could not determine passing test count" >&2
  exit 1
fi

bash "$ROOT/scripts/verify_slime_qwen3_4b.sh" >/dev/null
"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_production_gate.py" \
  --config "$ROOT/configs/production_slo.yaml" \
  --automated-tests "$automated_tests" \
  --actor-death "$ROOT/results/slime_qwen3_4b_4gpu/chaos/$ACTOR_RUN_ID/actor_death_acceptance.json" \
  --actor-trace "$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state/$ACTOR_RUN_ID/bounded_async_trace.jsonl" \
  --network-partition "$ROOT/results/slime_qwen3_4b_4gpu/chaos/$NETWORK_RUN_ID/network_partition_acceptance.json" \
  --network-trace "$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state/$NETWORK_RUN_ID/bounded_async_trace.jsonl" \
  --host-loss "$ROOT/results/slime_qwen3_4b_4gpu/host_loss/$HOST_RUN_ID/host_loss_acceptance.json" \
  --long-soak "$ROOT/results/slime_qwen3_4b_4gpu/long_soak/$SOAK_RUN_ID/long_soak_acceptance.json" \
  --output "$OUTPUT"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/write_run_manifest.py" \
  --project-root "$ROOT" \
  --run-id "$SOAK_RUN_ID-production" \
  --status accepted \
  --trainer-gpus "${TRAINER_GPU_IDS:-1,2}" \
  --rollout-gpus "${ROLLOUT_GPU_IDS:-3,4}" \
  --automated-tests "$automated_tests" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/chaos/$ACTOR_RUN_ID/actor_death_acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state/$ACTOR_RUN_ID/bounded_async_trace.jsonl" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/chaos/$ACTOR_RUN_ID/live_router_acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/chaos/$NETWORK_RUN_ID/network_partition_acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state/$NETWORK_RUN_ID/bounded_async_trace.jsonl" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/chaos/$NETWORK_RUN_ID/live_router_acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/host_loss/$HOST_RUN_ID/host_loss_acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/host_loss/$HOST_RUN_ID/checkpoint_acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/long_soak/$SOAK_RUN_ID/long_soak_acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/long_soak/$SOAK_RUN_ID/acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/long_soak/$SOAK_RUN_ID/live_router_acceptance.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/long_soak/$SOAK_RUN_ID/timing.json" \
  --artifact "$ROOT/results/slime_qwen3_4b_4gpu/long_soak/$SOAK_RUN_ID/gpu_telemetry.csv" \
  --artifact "$ROOT/configs/production_slo.yaml" \
  --artifact "$OUTPUT" \
  --output "$OUTPUT_DIR/manifest.json"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_run_manifest.py" \
  --manifest "$OUTPUT_DIR/manifest.json"

echo "Production evidence gate passed: $OUTPUT"
