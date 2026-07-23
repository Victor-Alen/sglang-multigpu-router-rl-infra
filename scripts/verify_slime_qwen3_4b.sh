#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

test "$(git -C "$SLIME_ROOT" rev-parse HEAD)" = "$SLIME_COMMIT"
test "$(git -C "$SGLANG_ROOT" rev-parse HEAD)" = "$SGLANG_COMMIT"
git -C "$SLIME_ROOT" apply --reverse --check "$ROOT/$SLIME_LOCAL_PATCH"
git -C "$SLIME_ROOT" apply --reverse --check "$ROOT/$SLIME_BOUNDED_ASYNC_PATCH"
git -C "$SLIME_ROOT" apply --reverse --check "$ROOT/$SLIME_CHAOS_PATCH"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$PROMPT_DATA"
if find "$MODEL_PATH" -type f -name '*.incomplete' -print -quit | grep -q .; then
  echo "Incomplete model downloads remain under $MODEL_PATH" >&2
  exit 2
fi

PYTHONPATH="$SGLANG_ROOT/python:$SLIME_ROOT" "$PYTHON_ENV/bin/python" - <<'PY'
import importlib.metadata
import json
import torch
import ray
import sglang
import slime
import flash_attn
from transformers import AutoConfig, AutoTokenizer
import os

model_path = os.environ.get("MODEL_PATH", "/home/thjiang/models/Qwen3-4B")
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "ray": ray.__version__,
    "sglang": getattr(sglang, "__version__", "unknown"),
    "sglang_router": importlib.metadata.version("sglang-router"),
    "flash_attn": flash_attn.__version__,
    "slime": getattr(slime, "__version__", "editable-source"),
    "model_type": config.model_type,
    "architectures": config.architectures,
    "hidden_layers": config.num_hidden_layers,
    "vocab_size": tokenizer.vocab_size,
}, indent=2))
PY

PYTHONPATH="$SGLANG_ROOT/python:$SLIME_ROOT" "$PYTHON_ENV/bin/python" - <<'PY'
import sys
from slime.utils.arguments import parse_args

sys.argv = [
    "train.py",
    "--train-backend", "fsdp",
    "--actor-num-nodes", "1",
    "--actor-num-gpus-per-node", "2",
    "--rollout-num-gpus", "2",
    "--num-gpus-per-node", "4",
    "--rollout-num-gpus-per-engine", "1",
    "--hf-checkpoint", "/home/thjiang/models/Qwen3-4B",
    "--prompt-data", "/home/thjiang/datasets/dapo-math-17k/dapo-math-17k.jsonl",
    "--input-key", "prompt",
    "--label-key", "label",
    "--num-rollout", "1",
    "--rollout-batch-size", "2",
    "--n-samples-per-prompt", "2",
    "--global-batch-size", "4",
    "--rollout-max-response-len", "1024",
    "--gradient-checkpointing",
    "--attn-implementation", "flash_attention_2",
    "--use-dynamic-batch-size",
    "--max-tokens-per-gpu", "2048",
    "--use-slime-router",
]
args = parse_args()
assert args.train_backend == "fsdp"
assert args.actor_num_gpus_per_node == 2
assert args.rollout_num_gpus == 2
assert args.world_size == 2
assert args.attn_implementation == "flash_attention_2"
print("4-GPU slime argument parsing passed.")
PY
echo "slime/Qwen3-4B installation verification passed."
