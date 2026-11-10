# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vendored GVR self-sampling top-k (TensorRT-LLM PR#18501 + PR#18410
hint-free + PR#18625 +inf-bracket fix, Apache-2.0).

INTERNAL vendored snapshot -- an exact single-pass top-K (exactness by
count-crossing invariants). PR#18410 makes the varlen path HINT-FREE: the
threshold bracket comes from the current row's self-sample (register
families fold the first k row values; streaming families never consume a
temporal hint), so ``run_varlen`` no longer takes ``pre_idx`` -- matching
our prefill use, which has no prior top-k. Used as an alternative unfused
top-k backend (topk_impl="gvr") and the filtered-fusion exact fallback,
where it replaces the top-k's REREAD-bound large-K cost (7.6x at
256K/K=2048).
"""

from . import gvr_topk_decode_self_sampling as gvr_kernel  # noqa: F401
from . import gvr_topk_decode_self_sampling_host as gvr_host  # noqa: F401
