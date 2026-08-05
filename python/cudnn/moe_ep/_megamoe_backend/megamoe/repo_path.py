# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Make the sibling cutedsl_megamoe clone importable.

The kernel repo is not pip-installed; its own modules assume the repo root
(and moe_nvfp4_swapab, for the mxfp8 package's cross-imports) are on
sys.path.  Import this module before importing anything from the repo.

Override the clone location with MEGAMOE_REPO=/path/to/cutedsl_megamoe.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINING_ROOT = os.path.dirname(_HERE)

REPO_ROOT = os.environ.get(
    "MEGAMOE_REPO", os.path.join(_TRAINING_ROOT, "cutedsl_megamoe")
)
if not os.path.isdir(os.path.join(REPO_ROOT, "moe_mxfp8_glu")):
    raise ImportError(
        f"cutedsl_megamoe clone not found at {REPO_ROOT!r}; clone "
        "https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe there or "
        "set MEGAMOE_REPO."
    )

for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "moe_nvfp4_swapab"), _TRAINING_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
