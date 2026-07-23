#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"
test -r "$LOCK_FILE" || { echo "Lock file missing: $LOCK_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$LOCK_FILE"

TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,3}"
ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-5,7}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/checkpoints}"
LOAD_CHECKPOINT_DIR="${LOAD_CHECKPOINT_DIR:-$CHECKPOINT_DIR}"
SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-$CHECKPOINT_DIR}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
DUMP_DIR="${DUMP_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/dump_details}"
ENABLE_DUMP_DETAILS="${ENABLE_DUMP_DETAILS:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-24000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-24199}"
RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-25001}"
RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-25002}"
RAY_NODE_IP="${RAY_NODE_IP:-127.0.0.1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RL_ROUTER_POLICY="${RL_ROUTER_POLICY:-adaptive-group}"
RL_ROUTER_RUN_ID="${RL_ROUTER_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RL_ROUTER_STATE_DIR="${RL_ROUTER_STATE_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state}"
RL_CHAT_TEMPLATE_HASH="${RL_CHAT_TEMPLATE_HASH:-$(sha256sum "$MODEL_PATH/tokenizer_config.json" | awk '{print $1}')}"
RL_ROUTER_DISCOVERY_RETRIES="${RL_ROUTER_DISCOVERY_RETRIES:-15}"
RL_ROUTER_DISCOVERY_RETRY_SECONDS="${RL_ROUTER_DISCOVERY_RETRY_SECONDS:-1}"
BOUNDED_ASYNC="${BOUNDED_ASYNC:-0}"
RL_MAX_PREFETCHED_BATCHES="${RL_MAX_PREFETCHED_BATCHES:-1}"
RL_MAX_POLICY_LAG="${RL_MAX_POLICY_LAG:-1}"
RL_ASYNC_STRICT_FALLBACK="${RL_ASYNC_STRICT_FALLBACK:-1}"
RL_FAULT_INJECT_PREFETCH_ROLLOUT_ID="${RL_FAULT_INJECT_PREFETCH_ROLLOUT_ID:--1}"
RL_ENABLE_DESTRUCTIVE_CHAOS="${RL_ENABLE_DESTRUCTIVE_CHAOS:-0}"
RL_CHAOS_ENGINE_DEATH_ROLLOUT_ID="${RL_CHAOS_ENGINE_DEATH_ROLLOUT_ID:--1}"
RL_CHAOS_ENGINE_INDEX="${RL_CHAOS_ENGINE_INDEX:-0}"
RL_CHAOS_NETWORK_ROLLOUT_ID="${RL_CHAOS_NETWORK_ROLLOUT_ID:--1}"
RL_CHAOS_NETWORK_HOST="${RL_CHAOS_NETWORK_HOST:-$RAY_NODE_IP}"
RL_CHAOS_NETWORK_PORT="${RL_CHAOS_NETWORK_PORT:-15000}"
RL_CHAOS_NETWORK_SECONDS="${RL_CHAOS_NETWORK_SECONDS:-10}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-30}"
ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-300}"
ROLLOUT_SHUFFLE="${ROLLOUT_SHUFFLE:-1}"
ROLLOUT_SHUFFLE_ARGS=()
if [[ "$ROLLOUT_SHUFFLE" == "1" ]]; then
  ROLLOUT_SHUFFLE_ARGS+=(--rollout-shuffle)
elif [[ "$ROLLOUT_SHUFFLE" != "0" ]]; then
  echo "ROLLOUT_SHUFFLE must be 0 or 1" >&2
  exit 2
fi
if [[ "$BOUNDED_ASYNC" == "1" ]]; then
  TRAIN_ENTRYPOINT="$ROOT/integrations/slime_bounded_train.py"
else
  TRAIN_ENTRYPOINT="$SLIME_ROOT/train.py"
fi
if ! [[ "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAVE_INTERVAL must be a positive integer" >&2
  exit 2
fi
DUMP_DETAILS_ARGS=()
if [[ "$ENABLE_DUMP_DETAILS" == "1" ]]; then
  DUMP_DETAILS_ARGS+=(--dump-details "$DUMP_DIR")
elif [[ "$ENABLE_DUMP_DETAILS" != "0" ]]; then
  echo "ENABLE_DUMP_DETAILS must be 0 or 1" >&2
  exit 2
fi

"$PYTHON_ENV/bin/python" "$ROOT/scripts/check_gpu_layout.py" \
  --trainer-gpu-ids "$TRAINER_GPU_IDS" \
  --rollout-gpu-ids "$ROLLOUT_GPU_IDS" \
  --expected-total 4 >/dev/null
"$PYTHON_ENV/bin/python" "$ROOT/scripts/check_gpu_availability.py" \
  --gpu-ids "$TRAINER_GPU_IDS,$ROLLOUT_GPU_IDS" \
  --max-used-mib "${MAX_USED_MIB:-512}"

test -x "$PYTHON_ENV/bin/python" || { echo "Python environment missing: $PYTHON_ENV" >&2; exit 2; }
test -f "$MODEL_PATH/config.json" || { echo "Model incomplete: $MODEL_PATH" >&2; exit 2; }
test -f "$PROMPT_DATA" || { echo "Dataset missing: $PROMPT_DATA" >&2; exit 2; }
test "$(git -C "$SLIME_ROOT" rev-parse HEAD)" = "$SLIME_COMMIT" || {
  echo "slime commit does not match lock file" >&2; exit 2;
}
test "$(git -C "$SGLANG_ROOT" rev-parse HEAD)" = "$SGLANG_COMMIT" || {
  echo "SGLang base commit does not match lock file" >&2; exit 2;
}

if pgrep -u "$(id -u)" -f '(^|/)raylet([[:space:]]|$)' >/dev/null; then
  echo "A Ray cluster is already running for this user; refusing to disturb it." >&2
  exit 2
fi
while read -r address; do
  port="${address##*:}"
  if [[ "$port" =~ ^[0-9]+$ ]] &&
     (( port == RAY_DASHBOARD_PORT || port == RAY_GCS_PORT ||
        port == RAY_NODE_MANAGER_PORT || port == RAY_OBJECT_MANAGER_PORT ||
        (port >= RAY_MIN_WORKER_PORT && port <= RAY_MAX_WORKER_PORT) )); then
    echo "Required Ray port is already in use: $port" >&2
    exit 2
  fi
done < <(ss -ltnH | awk '{print $4}')

mkdir -p "$LOAD_CHECKPOINT_DIR" "$SAVE_CHECKPOINT_DIR"
if [[ "$ENABLE_DUMP_DETAILS" == "1" ]]; then
  mkdir -p "$DUMP_DIR"
fi
export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$SGLANG_ROOT/python:$SLIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Ray logical IDs 0,1 map to physical Trainer GPUs; logical IDs 2,3 map to Rollout GPUs.
export CUDA_VISIBLE_DEVICES="$TRAINER_GPU_IDS,$ROLLOUT_GPU_IDS"
export PYTHONBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NCCL_NVLS_ENABLE=0
export RL_ROUTER_POLICY RL_ROUTER_RUN_ID RL_ROUTER_STATE_DIR RL_CHAT_TEMPLATE_HASH
export RL_ROUTER_DISCOVERY_RETRIES RL_ROUTER_DISCOVERY_RETRY_SECONDS
export BOUNDED_ASYNC RL_MAX_PREFETCHED_BATCHES RL_MAX_POLICY_LAG RL_ASYNC_STRICT_FALLBACK
export RL_FAULT_INJECT_PREFETCH_ROLLOUT_ID
export RL_ENABLE_DESTRUCTIVE_CHAOS RL_CHAOS_ENGINE_DEATH_ROLLOUT_ID RL_CHAOS_ENGINE_INDEX
export RL_CHAOS_NETWORK_ROLLOUT_ID RL_CHAOS_NETWORK_HOST RL_CHAOS_NETWORK_PORT RL_CHAOS_NETWORK_SECONDS
export RL_TOKENIZER_REVISION="$MODEL_REVISION"

ray_started=0
ownership_guard_pid=""
cleanup() {
  if [[ -n "$ownership_guard_pid" ]]; then
    kill "$ownership_guard_pid" >/dev/null 2>&1 || true
    wait "$ownership_guard_pid" >/dev/null 2>&1 || true
  fi
  if [[ "$ray_started" -eq 1 ]]; then
    "$PYTHON_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
  "$PYTHON_ENV/bin/python" "$ROOT/scripts/cleanup_owned_gpu_processes.py" \
    --gpu-ids "$TRAINER_GPU_IDS,$ROLLOUT_GPU_IDS" \
    --run-id "$RL_ROUTER_RUN_ID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

"$PYTHON_ENV/bin/ray" start --head \
  --node-ip-address "$RAY_NODE_IP" \
  --num-cpus "$RAY_NUM_CPUS" \
  --port "$RAY_GCS_PORT" \
  --min-worker-port "$RAY_MIN_WORKER_PORT" \
  --max-worker-port "$RAY_MAX_WORKER_PORT" \
  --node-manager-port "$RAY_NODE_MANAGER_PORT" \
  --object-manager-port "$RAY_OBJECT_MANAGER_PORT" \
  --num-gpus 4 \
  --disable-usage-stats \
  --dashboard-host 127.0.0.1 \
  --dashboard-port "$RAY_DASHBOARD_PORT"
ray_started=1

"$PYTHON_ENV/bin/python" "$ROOT/scripts/guard_gpu_ownership.py" \
  --gpu-ids "$TRAINER_GPU_IDS,$ROLLOUT_GPU_IDS" \
  --ray-bin "$PYTHON_ENV/bin/ray" \
  --poll-seconds "${GPU_OWNERSHIP_POLL_SECONDS:-2}" &
ownership_guard_pid=$!

RUNTIME_ENV_JSON=$(printf \
  '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"0","PYTORCH_ALLOC_CONF":"expandable_segments:True","RL_ROUTER_POLICY":"%s","RL_ROUTER_RUN_ID":"%s","RL_ROUTER_STATE_DIR":"%s","RL_TOKENIZER_REVISION":"%s","RL_CHAT_TEMPLATE_HASH":"%s","RL_ROUTER_DISCOVERY_RETRIES":"%s","RL_ROUTER_DISCOVERY_RETRY_SECONDS":"%s","RL_MAX_PREFETCHED_BATCHES":"%s","RL_MAX_POLICY_LAG":"%s","RL_ASYNC_STRICT_FALLBACK":"%s","RL_FAULT_INJECT_PREFETCH_ROLLOUT_ID":"%s","RL_ENABLE_DESTRUCTIVE_CHAOS":"%s","RL_CHAOS_ENGINE_DEATH_ROLLOUT_ID":"%s","RL_CHAOS_ENGINE_INDEX":"%s","RL_CHAOS_NETWORK_ROLLOUT_ID":"%s","RL_CHAOS_NETWORK_HOST":"%s","RL_CHAOS_NETWORK_PORT":"%s","RL_CHAOS_NETWORK_SECONDS":"%s"}}' \
  "$PYTHONPATH" "$RL_ROUTER_POLICY" "$RL_ROUTER_RUN_ID" "$RL_ROUTER_STATE_DIR" \
  "$RL_TOKENIZER_REVISION" "$RL_CHAT_TEMPLATE_HASH" "$RL_ROUTER_DISCOVERY_RETRIES" \
  "$RL_ROUTER_DISCOVERY_RETRY_SECONDS" "$RL_MAX_PREFETCHED_BATCHES" \
  "$RL_MAX_POLICY_LAG" "$RL_ASYNC_STRICT_FALLBACK" "$RL_FAULT_INJECT_PREFETCH_ROLLOUT_ID" \
  "$RL_ENABLE_DESTRUCTIVE_CHAOS" "$RL_CHAOS_ENGINE_DEATH_ROLLOUT_ID" "$RL_CHAOS_ENGINE_INDEX" \
  "$RL_CHAOS_NETWORK_ROLLOUT_ID" "$RL_CHAOS_NETWORK_HOST" "$RL_CHAOS_NETWORK_PORT" \
  "$RL_CHAOS_NETWORK_SECONDS")

cd "$SLIME_ROOT"
"$PYTHON_ENV/bin/ray" job submit \
  --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- "$PYTHON_ENV/bin/python" "$TRAIN_ENTRYPOINT" \
  --train-backend fsdp \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  --rollout-num-gpus 2 \
  --num-gpus-per-node 4 \
  --rollout-num-gpus-per-engine 1 \
  --hf-checkpoint "$MODEL_PATH" \
  --load "$LOAD_CHECKPOINT_DIR" \
  --save "$SAVE_CHECKPOINT_DIR" \
  --save-interval "$SAVE_INTERVAL" \
  --prompt-data "$PROMPT_DATA" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  "${ROLLOUT_SHUFFLE_ARGS[@]}" \
  --balance-data \
  --rm-type deepscaler \
  --num-rollout "$NUM_ROLLOUTS" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --n-samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --rollout-max-response-len "$MAX_RESPONSE_LEN" \
  --rollout-temperature 0.8 \
  --custom-generate-function-path integrations.slime_adapter.generate_with_rl_router \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --advantage-estimator grpo \
  --kl-coef 0.0 \
  --kl-loss-coef 0.0 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --gradient-checkpointing \
  --attn-implementation flash_attention_2 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
  --update-weight-buffer-size 536870912 \
  --sglang-mem-fraction-static 0.75 \
  --sglang-chunked-prefill-size 2048 \
  --use-slime-router \
  --use-fault-tolerance \
  --rollout-health-check-interval "$ROLLOUT_HEALTH_CHECK_INTERVAL" \
  --rollout-health-check-timeout "$ROLLOUT_HEALTH_CHECK_TIMEOUT" \
  --rollout-health-check-first-wait "$ROLLOUT_HEALTH_CHECK_FIRST_WAIT" \
  "${DUMP_DETAILS_ARGS[@]}"
