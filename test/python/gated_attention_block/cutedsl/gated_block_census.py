# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CUDA-kernel census helper the gated-attention-block test modules share (``test_block_serving_decode.py``,
``test_block_qsa_decode.py``): one call under ``torch.profiler`` (CUPTI) after one warm call, the kernel names returned with the
memsets / memcpys split out.  A non-test module so that no test module imports another test module for it (an import cycle
otherwise: ``test_block_serving_decode`` imports ``test_block_qsa_decode`` for its decline rows)."""

import torch


def cupti_kernels(fn):
    """The CUDA kernel names of one call under torch.profiler (CUPTI) after one warm call, memsets / memcpys split out; ``(None, None)``
    when the node records nothing."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        return None, None
    kernels = [n for n in names if "memset" not in n.lower() and "memcpy" not in n.lower()]
    return kernels, [n for n in names if n not in kernels]
