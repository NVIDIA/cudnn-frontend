#!/usr/bin/env bash
# Run benchmark/dsa configs and land results in results/<config>/<arch>/.
# Usage:
#   run_all.sh <arch_label> [gpu] [config ...]
#     gpu    = CUDA_VISIBLE_DEVICES value (index or GPU-<uuid>), "" to skip
#     config = subset of configs (default: all)
# e.g. run_all.sh b200 GPU-4a901f61-...   |   run_all.sh gb300 "" deepseek_v4
set -uo pipefail

ARCH=${1:?usage: run_all.sh <arch_label> [gpu] [config ...]}
if [ -n "${2:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$2
fi
shift; [ $# -gt 0 ] && shift
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(deepseek_v4 deepseek_v41 glm53)
cd "$(dirname "$0")/../.." || exit 1

mkdir -p benchmark/dsa/results
status=0
for cfg in "${CONFIGS[@]}"; do
    echo "=== $cfg -> results/$cfg/$ARCH ==="
    if ! python -m benchmark.dsa.runner --config "$cfg" \
        --output-dir "benchmark/dsa/results/$cfg/$ARCH" \
        2>&1 | tee "benchmark/dsa/results/${cfg}_${ARCH}.log"; then
        echo "!! $cfg failed"
        status=1
    fi
done
exit "$status"
