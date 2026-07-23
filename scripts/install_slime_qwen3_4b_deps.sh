#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

test -f "$PYTHON_ENV/conda-meta/history" || {
  echo "Conda environment is not complete: $PYTHON_ENV" >&2
  exit 2
}
test "$(git -C "$SLIME_ROOT" rev-parse HEAD)" = "$SLIME_COMMIT"
test "$(git -C "$SGLANG_ROOT" rev-parse HEAD)" = "$SGLANG_COMMIT"
for patch in "$SLIME_LOCAL_PATCH" "$SLIME_BOUNDED_ASYNC_PATCH"; do
  if git -C "$SLIME_ROOT" apply --reverse --check "$ROOT/$patch" >/dev/null 2>&1; then
    : # already applied
  elif git -C "$SLIME_ROOT" apply --check "$ROOT/$patch" >/dev/null 2>&1; then
    git -C "$SLIME_ROOT" apply "$ROOT/$patch"
  else
    echo "The local slime patch cannot be applied cleanly: $patch" >&2
    exit 2
  fi
done
git -C "$SGLANG_ROOT" apply --reverse --check \
  "$SLIME_ROOT/docker/patch/v0.5.6/sglang.patch" >/dev/null || {
  echo "The official slime v0.2.1 SGLang patch is not applied cleanly." >&2
  exit 2
}

PY="$PYTHON_ENV/bin/python"
# The cloned base environment contains SGLang 0.5.8/vLLM pins. They are not
# compatible with slime v0.2.1's patched SGLang 0.5.6 and are isolated here.
"$PY" -m pip uninstall -y sglang slime vllm quack-kernels mamba-ssm >/dev/null 2>&1 || true
"$PY" -m pip install -e "$SGLANG_ROOT/python"
"$PY" -m pip install "sglang-router==$SGLANG_ROUTER_VERSION"
"$PY" -m pip install -e "$SLIME_ROOT[fsdp]"
"$PY" -m pip install "outlines-core==0.1.26" "pluggy>=1.5,<2"
# Repair mixed conda/pip files inherited from the base prefix before checking.
"$PY" -m pip uninstall -y charset-normalizer >/dev/null 2>&1 || true
"$PY" -m pip uninstall -y charset-normalizer >/dev/null 2>&1 || true
"$PY" -m pip install --force-reinstall --no-deps \
  "requests==2.32.3" "charset-normalizer==3.3.2"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}" MAX_JOBS="${MAX_JOBS:-8}" \
  "$PY" -m pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation
"$PY" -m pip check
"$PY" -m pip freeze >"$ROOT/configs/slime_qwen3_4b_4xa6000.pip-freeze.txt"

echo "Installed slime, patched SGLang, and FSDP dependencies in $PYTHON_ENV"
