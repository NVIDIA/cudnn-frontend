# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The shared split combine's half form: f16 / bf16 partials (every SM120 split, the SM100 MXFP8 d512 split,
the cc 10.7 flavors without an fp32 partial slot) at every S_q and split count.  The LSE passes stride the
split axis across the warp's lanes, 32 splits a round, and the O pass loads each split's four columns as one
8-byte vector, ``_SPLIT_BATCH`` splits in flight (``d % 4 == 0``; ``d = 64 / 192`` leave lanes past ``d_v``,
which read the row's last columns and store nothing) or element by element (``d = 66``).  Split
counts sit on and off both strides; dead splits (NaN payloads) fall inside a batch, on split 0 and on the
first split of the second round; the sink entries fold the sink once and the Q-length entry keeps padded
rows dead."""

import math

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry

from frost_test_utils import requires_dsl, requires_split_combine

pytestmark = [pytest.mark.L0, requires_dsl, requires_split_combine]

_SINKS = (-120.0, 0.7, 4.0)


def _half_partials(b, h, sq, d, splits, dtype, *, sink=False, q_lens=None):
    """Half split partials and the float64 reference of the merged O / LSE.

    Every split is dead on row 0 of a multi-row launch (a keyless row: O := 0 and LSE := -inf, or LSE := sink with
    a sink); every seventh split and, past one round, split 32 are dead on the last row.  With ``q_lens`` the rows
    at or past a batch's length are dead in every split and the reference keeps them at O := 0 / LSE := -inf."""
    generator = torch.Generator().manual_seed(20261015 + splits + sq)
    o = (torch.randn(splits, b, sq, h, d, generator=generator) * 0.4).to(dtype).double()
    lse = torch.randn(splits, b, h, sq, generator=generator).double() * 2.0 + 0.7
    if sq > 1:
        lse[:, :, :, 0] = -torch.inf
    lse[::7, :, :, sq - 1] = -torch.inf
    if splits > 32:
        lse[32, :, :, sq - 1] = -torch.inf
    padded = torch.zeros(b, sq, dtype=torch.bool)
    if q_lens is not None:
        for i, n in enumerate(q_lens):
            padded[i, n:] = True
            lse[:, i, :, n:] = -torch.inf
    dead = torch.isneginf(lse)
    sink_col = torch.tensor(_SINKS, dtype=torch.float64).view(1, h, 1).expand(b, h, sq) if sink else torch.full((b, h, sq), -torch.inf, dtype=torch.float64)
    if q_lens is not None:
        sink_col = sink_col.masked_fill(padded.view(b, 1, sq).expand(b, h, sq), -torch.inf)
    maximum = torch.maximum(lse.amax(0), sink_col)
    safe_max = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weights = torch.where(dead, torch.zeros_like(lse), (lse - safe_max).exp())
    denominator = weights.sum(0) + torch.where(torch.isfinite(sink_col), (sink_col - safe_max).exp(), torch.zeros_like(sink_col))
    safe_o = o.masked_fill(dead.permute(0, 1, 3, 2).unsqueeze(-1), 0)
    ref_o = (weights.permute(0, 1, 3, 2).unsqueeze(-1) * safe_o).sum(0) / denominator.clamp_min(1e-300).permute(0, 2, 1).unsqueeze(-1)
    ref_lse = torch.where(denominator > 0, safe_max + denominator.clamp_min(1e-300).log(), torch.full_like(denominator, -torch.inf))
    o = o.masked_fill(dead.permute(0, 1, 3, 2).unsqueeze(-1), torch.nan)  # a dead payload must never enter the arithmetic
    return o.to(dtype).reshape(splits * b, sq, h, d).cuda(), lse.float().reshape(splits * b, h, sq).cuda(), ref_o, ref_lse


@pytest.mark.parametrize("variant", ["plain", "sink", "sink_q_lens"])
# d64 / d192: lanes past d_v (the clamped vector read, no store), d192 also a second 128-column round.
@pytest.mark.parametrize("d", [128, 64, 192, 66], ids=["vector_d128", "vector_d64", "vector_d192", "elementwise_d66"])
@pytest.mark.parametrize("splits", [2, 8, 9, 33, 100], ids=lambda s: f"split{s}")
@pytest.mark.parametrize("sq", [1, 2, 5], ids=lambda s: f"sq{s}")
@pytest.mark.parametrize("partial", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
def test_half_form_matches_the_reference(partial, sq, splits, d, variant):
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h = 3, 3
    sink = variant != "plain"
    q_lens = [sq, max(0, sq - 1), 0] if variant == "sink_q_lens" else None
    op, lp, ref_o, ref_lse = _half_partials(b, h, sq, d, splits, partial, sink=sink, q_lens=q_lens)
    o = torch.full((b, sq, h, d), -31.0, device="cuda", dtype=torch.bfloat16)
    lse = torch.full((b, h, sq), -31.0, device="cuda", dtype=torch.float32)
    tag = {torch.float16: "f16", torch.bfloat16: "bf16"}[partial]
    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", dtype_partial=tag, has_lse=True, has_sink=sink, has_q_lens=q_lens is not None))
    assert fn is not None
    tail = ()
    if sink:
        sinks = torch.tensor(_SINKS, device="cuda")
        tail = (sinks.data_ptr(),)
    if q_lens is not None:
        q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device="cuda")
        tail += (q_lens_t.data_ptr(),)
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o.data_ptr(),
        lse.data_ptr(),
        (b, h, sq, d),
        splits,
        o.stride(),
        lse.stride(),
        *tail,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    o_cpu, lse_cpu = o.cpu().double(), lse.cpu().double()
    torch.testing.assert_close(o_cpu, ref_o, atol=0.004, rtol=0.004)
    torch.testing.assert_close(lse_cpu, ref_lse, atol=2e-5, rtol=2e-5)
    keyless = torch.isneginf(lp.view(splits, b, h, sq).cpu().amax(0))
    assert torch.all(o_cpu.permute(0, 2, 1, 3)[keyless] == 0.0), "a keyless or padded row must write O := 0"


@pytest.mark.parametrize("dtype,tag", [(torch.float8_e4m3fn, "e4m3"), (torch.float8_e5m2, "e5m2")])
def test_half_form_quantized_output_keeps_scale_and_amax(dtype, tag):
    """An SM120 quantized-O split writes half partials; the half form applies the output scale at the single cast
    and reports the amax of the merged O."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 2, 3, 5, 128, 9
    op, lp, ref_o, ref_lse = _half_partials(b, h, sq, d, splits, torch.float16)
    o = torch.empty((b, sq, h, d), device="cuda", dtype=dtype)
    lse = torch.full((b, h, sq), torch.nan, device="cuda")
    amax = torch.zeros(1, dtype=torch.float32, device="cuda")
    scale = torch.tensor([1.75], dtype=torch.float32, device="cuda")
    fn = positional_entry(
        comb.compile_ptr(
            dtype_o=tag, has_lse=True, has_amax=True, dtype_partial="f16", has_scale_o=True, stats_log2=True, quantized=True, has_scale_o_input=True
        )
    )
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o.data_ptr(),
        lse.data_ptr(),
        (b, h, sq, d),
        splits,
        o.stride(),
        lse.stride(),
        amax.data_ptr(),
        scale.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    expected = (ref_o * 1.75).float().to(dtype).float()
    # A last-bit pre-quant reduction difference may cross one quantization midpoint.
    torch.testing.assert_close(o.cpu().float(), expected, atol=0.03125, rtol=0.01)
    torch.testing.assert_close(amax.cpu(), ref_o.abs().amax().float().view(1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(lse.cpu().double(), ref_lse * math.log2(math.e), atol=2e-5, rtol=2e-5)
