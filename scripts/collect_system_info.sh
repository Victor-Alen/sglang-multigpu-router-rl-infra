#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-results/system_info/hardware_topology.txt}"
mkdir -p "$(dirname "$OUTPUT")"
{
  date --iso-8601=seconds
  uname -a
  cat /etc/os-release
  nvidia-smi -L
  nvidia-smi topo -m
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,pci.bus_id,driver_version --format=csv
  lscpu
  if command -v numactl >/dev/null 2>&1; then numactl --hardware; fi
  df -h /home
  if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
  if command -v dpkg-query >/dev/null 2>&1; then
    dpkg-query -W 'libnccl*' 2>/dev/null || true
  fi
  git -C /home/thjiang/sglang rev-parse HEAD 2>/dev/null || true
  git rev-parse HEAD 2>/dev/null || true
} >"$OUTPUT"
echo "wrote $OUTPUT"
