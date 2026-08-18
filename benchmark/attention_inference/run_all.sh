#!/usr/bin/env bash
# Run attention_inference configs and land results in results/<config>/<arch>/.
# Usage:
#   run_all.sh <arch_label> [gpu] [config ...]
#     gpu    = CUDA_VISIBLE_DEVICES value (index or GPU-<uuid>), "" to skip
#     config = subset of configs (default: all six)
# e.g. run_all.sh rtx_pro_6000 GPU-41233e81-...   |   run_all.sh b300 "" llama
set -uo pipefail

ARCH=${1:?usage: run_all.sh <arch_label> [gpu] [config ...]}
if [ -n "${2:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$2
fi
shift; [ $# -gt 0 ] && shift
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(llama qwen35 gpt_oss deepseek_v4 kimi_k3 auto_regressive_dit)
cd "$(dirname "$0")/../.." || exit 1

declare -A NAMES=(
    [llama]=llama3.1
    [qwen35]=qwen35
    [gpt_oss]=gpt_oss
    [deepseek_v4]=deepseek_v4
    [kimi_k3]=kimi_k3
    [auto_regressive_dit]=auto_regressive_dit
)
mkdir -p benchmark/attention_inference/results
for cfg in "${CONFIGS[@]}"; do
    if [ -z "${NAMES[$cfg]:-}" ]; then
        echo "!! unknown config '$cfg' (known: ${!NAMES[*]})"
        continue
    fi
    echo "=== $cfg -> results/${NAMES[$cfg]}/$ARCH ==="
    python -m benchmark.attention_inference.runner --config "$cfg" \
        --output-dir "benchmark/attention_inference/results/${NAMES[$cfg]}/$ARCH" \
        2>&1 | tee "benchmark/attention_inference/results/${NAMES[$cfg]}_${ARCH}.log" || echo "!! $cfg failed"
done
