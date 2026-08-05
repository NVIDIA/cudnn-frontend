#!/bin/bash
# PR validation for the MegaMoE backend bundled inside cuDNN
# (python/cudnn/moe_ep/_megamoe_backend).  Runs inside the flashinfer-ep
# container on a GB200 node.
set -x
export CLONE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)

echo "=== 1. PR pytest suite (reference/API tests) ==="
SITE=$(python -c "import cudnn,os;print(os.path.dirname(cudnn.__file__))")
cp -r $CLONE/python/cudnn/moe_ep $SITE/
grep -q moe_ep $SITE/__init__.py || \
  echo "from .moe_ep import BlockScaledTensor, MoeEp, MoeFormat, MoeTensor" >> $SITE/__init__.py
cd $CLONE/test/python
NVIDIA_TF32_OVERRIDE=0 python -m pytest fe_api/moe_ep/test_moe_ep.py -q || exit 1

echo "=== 2. Vendored kernel imports (fp4 forward + fp8 mega backward) ==="
MEGA_NO_DIST=1 python - <<'EOF' || exit 1
import sys, os
sys.path.insert(0, os.path.join(os.environ["CLONE"], "python", "cudnn", "moe_ep", "_megamoe_backend"))
import megamoe.repo_path
from megamoe.forward import MegaMoeMxfp8Forward
from megamoe.forward_nvfp4 import MegaMoeNvfp4Forward
from megamoe.bwd_kernel.backward import MegaMoeMxfp8Backward, mega_backward
import megamoe.repo_path as rp
assert "_megamoe_backend" in rp.REPO_ROOT, rp.REPO_ROOT
print("vendored imports OK, kernel repo root:", rp.REPO_ROOT)
EOF

echo "=== 3. Single-rank megamoe backend parity (compiles kernels, ~5 min) ==="
cd $CLONE/test/python
CUDNN_MOE_EP_BACKEND=megamoe MEGA_NO_DIST=1 \
  python fe_api/moe_ep/megamoe_backend_parity.py || exit 1
echo "ALL PR TESTS PASSED"
