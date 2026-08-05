#!/usr/bin/env bash
# Run the moe_ep_training test suite (needs GPUs; not a login node).
#   ./run_tests.sh          # reference pytest + torchrun -n4 parity
#   NPROC=2 ./run_tests.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== single-process reference tests (CPU ok) =="
python -m pytest -q test_reference.py

echo "== single-GPU fp4/fp8 QAT numerics =="
python test_fp4_qat_numerics.py

echo "== single-GPU quantizer conformance vs cutedsl kernels =="
python test_quant_vs_kernel.py

NPROC="${NPROC:-4}"
echo "== ${NPROC}-rank EP fwd+bwd parity =="
torchrun --standalone --nproc-per-node="${NPROC}" parity_ep_vs_reference.py

echo "== ${NPROC}-rank fp4 EP fwd+bwd parity =="
torchrun --standalone --nproc-per-node="${NPROC}" parity_ep_vs_reference_fp4.py
