# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 30: GDP (Gated DeltaProduct) forward context.

With ``output_context=True`` the forward also returns, as its last value, the uint8 context it leaves for the backward of the
same tokens.  Under autograd nothing else changes: the backward receives the context and reads the forward's intermediates
instead of recomputing them.  A caller that runs the backward op itself passes the buffer as ``context`` with
``context_from_forward=True``.  The gradients of both routes are checked bitwise against the plain call.  The batch holds one
long sequence, which the forward cuts into chained pieces; a forward that leaves nothing returns a zero-size context.
"""

from __future__ import annotations

import torch

from cudnn.linear_attention import ops as la_ops


def main(seq_lens=(24576, 512), H: int = 2, D: int = 128) -> None:
    if not torch.cuda.is_available():
        print("[30] SKIP  needs a CUDA GPU")
        return
    major, minor = torch.cuda.get_device_capability()
    if not 100 <= major * 10 + minor < 120:
        print("[30] SKIP  needs a data-center Blackwell-family GPU (100 <= SM < 120)")
        return

    torch.manual_seed(0)
    device = "cuda"
    total = sum(seq_lens)
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    n = 3  # Householder updates per token: k, v and beta live on the expanded timeline of total * n rows
    q = torch.nn.functional.normalize(torch.randn(total, H, D, device=device), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(total * n, H, D, device=device), dim=-1).bfloat16()
    v = (torch.randn(total * n, H, D, device=device) * 0.5).bfloat16()
    gate = torch.empty(total, H, device=device).uniform_(0.9, 1.0).log()
    beta = torch.rand(total * n, H, device=device)
    names, leaves = ("q", "k", "v", "g", "beta"), (q, k, v, gate, beta)
    scale = D**-0.5
    do = (torch.randn(total, H, D, device=device) * 0.1).bfloat16()

    # ---- the plain call under autograd: the reference ------------------------------------------------------------
    plain = [x.clone().requires_grad_(True) for x in leaves]
    o_plain = la_ops.gated_delta_product(*plain, cu, n, scale)[0]
    grads_plain = torch.autograd.grad([o_plain], plain, [do])

    # ---- autograd: the forward returns its context and the backward receives it ---------------------------------
    kept = [x.clone().requires_grad_(True) for x in leaves]
    outputs = la_ops.gated_delta_product(*kept, cu, n, scale, output_context=True)
    o, context = outputs[0], outputs[-1]
    assert context.dtype == torch.uint8 and context.numel() > 0, "the forward left no context"
    assert torch.equal(o, o_plain), "o differs from the plain call"
    grads_autograd = torch.autograd.grad([o], kept, [do])

    # ---- the backward op called directly: the context goes in with context_from_forward -------------------------
    grads_direct = torch.ops.cudnn.gated_delta_product_bwd(do, *leaves, cu, n, scale, context=context, context_from_forward=True)[: len(leaves)]

    for name, ref, autograd, direct in zip(names, grads_plain, grads_autograd, grads_direct):
        assert torch.equal(autograd, ref), f"d{name} under autograd differs from the plain call"
        assert torch.equal(direct, ref), f"d{name} of the direct backward differs from the plain call"
    print(f"[30] PASS  gdp forward context      seq_lens={list(seq_lens)} H={H} (context {context.numel()} bytes, gradients bitwise equal to the plain call)")


if __name__ == "__main__":
    main()
