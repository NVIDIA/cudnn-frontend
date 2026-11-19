# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned public forward/backward bytes for each packed execution path."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from test_causal_conv1d_packed import _dsl, reference

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("path", ["tiled-silu", "tiled-identity", "decode-silu", "decode-public-state", "scalar-silu"])
def test_public_packed_golden_all_outputs_and_gradients(dtype, path):
    _dsl()
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")
    from cutlass.base_dsl.version_info import CUDA_VERSION

    lengths = [1] * 17 if path.startswith("decode") else [0, 1, 7, 130, 0]
    hidden, width = 24, 4
    silu = path != "tiled-identity"
    generator = torch.Generator().manual_seed(7713)

    def fixed(shape):
        return (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.125).to(device="cuda", dtype=dtype)

    x = fixed((sum(lengths), hidden))
    if path == "scalar-silu":
        x = x.T.contiguous().T
    weight, bias = fixed((hidden, width)), fixed((hidden,))
    initial = fixed((len(lengths), width - 1, hidden))
    dy, dfinal = fixed(x.shape), fixed(initial.shape)
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    inputs = dict(x=x, weight=weight, bias=bias, initial=initial, cu=cu, dy=dy, dfinal=dfinal)
    before = snapshot(inputs)

    def run():
        xx, ww, bb, ii = (v.detach().clone().requires_grad_() for v in (x, weight, bias, initial))
        out = torch.empty((len(lengths), hidden, width - 1), device=x.device, dtype=dtype) if path == "decode-public-state" else None
        y, final = causal_conv1d(
            xx.T[None],
            ww,
            bb,
            activation="silu" if silu else None,
            cu_seqlens=cu,
            initial_states=ii.transpose(1, 2),
            return_final_states=True,
            final_states_out=out,
        )
        y, final = y[0].T, final.transpose(1, 2)
        dx, dw, db, di = torch.autograd.grad((y, final), (xx, ww, bb, ii), (dy, dfinal))
        return dict(y=y, final=final, dx=dx, dw=dw, db=db, di=di)

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        actual = assert_replays(run, repeats=3)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    xx, ww, bb, ii = (v.double().detach().requires_grad_() for v in (x, weight, bias, initial))
    y, final = reference(xx, ww, bb, ii, lengths, silu)
    gradients = torch.autograd.grad((y, final), (xx, ww, bb, ii), (dy.double(), dfinal.double()))
    atol, rtol = {torch.float16: (0.003, 0.002), torch.bfloat16: (0.03, 0.02), torch.float32: (2e-5, 2e-5)}[dtype]
    for name, expected in zip(("y", "final", "dx", "dw", "db", "di"), (y, final, *gradients)):
        torch.testing.assert_close(actual[name], expected.to(dtype), atol=atol, rtol=rtol)
    assert_bitwise({"final": actual["final"]}, {"final": final.to(dtype)}, "golden history reference")
    assert_bitwise(inputs, before, "golden inputs mutated")
    compiler = f"cuda{CUDA_VERSION.major}_{CUDA_VERSION.minor}"
    name = f"causal_conv1d_packed_{str(dtype).split('.')[-1]}_{path}_{compiler}"
    assert_golden(name, inputs, actual)
