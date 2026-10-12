# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default-stream PARK of every stream-ordering probe in the gated-block suite -- ONE copy, three callers
(``test_block_end_to_end.py``, ``test_proj_gemm_bwd.py``, ``test_block_backward.py``; it was hand-rolled in each).

The probe: enqueue a long spin on torch's CURRENT (default) stream, then run the block on a SIDE stream with its inputs
written on that side stream right before and the workspace zeroed right after.  Anything a stage wrongly launches on
the default stream runs LATE -- after the side stream is long done -- so a late producer leaves zeros for its consumers
and a late consumer reads the zeros written over the workspace: the result differs from the default-stream run, which
it must equal BITWISE (every block here is deterministic).  ``seconds`` is the park length: ``torch.cuda._sleep``
counts cycles at ~2 GHz; the matmul fallback (no ``_sleep``) is a few hundred ms on any part.  A fix here (a longer
park for a faster part) reaches all three probes at once.
"""

import torch


def park_the_default_stream(seconds: float = 0.5) -> None:
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(int(seconds * 2.0e9))  # cycles at ~2 GHz
        return
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(16):
        x = x @ x
