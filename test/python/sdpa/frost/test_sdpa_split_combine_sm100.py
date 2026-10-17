# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The combine's final outputs honor strides independently of compact partials."""

import math

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _partials(b, h, sq, d, splits=3):
    generator = torch.Generator().manual_seed(20260919)
    o = torch.randn(splits, b, sq, h, d, generator=generator) * 0.4
    lse = torch.randn(splits, b, h, sq, generator=generator) * 2.0 + 0.7
    lse[:, :, :, 0] = -torch.inf  # Every split dead on one row.
    if sq > 1:
        lse[1, :, :, 1] = -torch.inf  # A dead split next to two live splits (a single-row launch has only the all-dead row).
    dead = torch.isneginf(lse).permute(0, 1, 3, 2).unsqueeze(-1)
    o.masked_fill_(dead, torch.nan)  # Dead payloads must never enter arithmetic.
    scores = lse.double()
    maximum = scores.amax(0)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = (scores - safe_maximum).exp()
    denominator = weights.sum(0)
    safe_o = o.double().masked_fill(dead, 0)
    ref_o = (weights.permute(0, 1, 3, 2).unsqueeze(-1) * safe_o).sum(0) / denominator.clamp_min(1).permute(0, 2, 1).unsqueeze(-1)
    ref_lse = safe_maximum + denominator.log()
    return o.reshape(splits * b, sq, h, d).cuda(), lse.reshape(splits * b, h, sq).cuda(), ref_o, ref_lse


def _output(shape, strides, dtype):
    span = 1 + sum((n - 1) * stride for n, stride in zip(shape, strides))
    storage = torch.full((span + 32,), -31.0, device="cuda", dtype=torch.float32).to(dtype)
    view = storage.as_strided(shape, strides, storage_offset=16)
    offsets = torch.zeros(shape, dtype=torch.int64)
    for axis, (n, stride) in enumerate(zip(shape, strides)):
        dims = [1] * len(shape)
        dims[axis] = n
        offsets += torch.arange(n).view(dims) * stride
    occupied = torch.zeros(storage.numel(), dtype=torch.bool)
    occupied[offsets.flatten() + 16] = True
    return view, storage, occupied


def _strides(b, h, sq, d, layout):
    if layout == "compact":
        return (sq * h * d, h * d, d, 1), (h * sq, sq, 1)
    head = d * 2 + 8
    token = h * head + 16
    batch = sq * token + 32
    lse_head = sq * 2 + 3
    lse_batch = h * lse_head + 5
    if layout == "int64_singleton":
        assert b == 1
        batch, lse_batch = 2**33 + 17, 2**34 + 19
    return (batch, token, head, 2), (lse_batch, lse_head, 2)


def _check(o, lse, o_storage, o_used, lse_storage, lse_used, ref_o, ref_lse, stats_log2):
    torch.cuda.synchronize()
    torch.testing.assert_close(o.cpu().float(), ref_o.float(), atol=0.004, rtol=0.004)
    assert torch.count_nonzero(o[:, 0]).item() == 0
    assert torch.all(o_storage.float().cpu()[~o_used] == -31.0)
    if lse is not None:
        if stats_log2:
            ref_lse = ref_lse * math.log2(math.e)
        torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)
        assert torch.all(torch.isneginf(lse[:, :, 0]))
        assert torch.all(lse_storage.cpu()[~lse_used] == -31.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
@pytest.mark.parametrize("layout", ["compact", "strided", "int64_singleton"])
def test_pointer_combine_strided_outputs_and_dead_splits(dtype, stats, layout):
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = (1 if layout == "int64_singleton" else 2), 3, 5, 160, 3
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    ostride, lstride = _strides(b, h, sq, d, layout)
    o, ostorage, oused = _output((b, sq, h, d), ostride, dtype)
    lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32) if stats != "none" else (None, None, None)
    owner = comb.compile_ptr(dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=lse is not None, stats_log2=stats == "log2")
    fn = positional_entry(owner)
    assert fn is not None
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o.data_ptr(),
        lse.data_ptr() if lse is not None else None,
        (b, h, sq, d),
        splits,
        ostride,
        lstride,
        torch.cuda.current_stream().cuda_stream,
    )
    _check(o, lse, ostorage, oused, lstorage, lused, ref_o, ref_lse, stats == "log2")


@pytest.mark.parametrize("layout", ["compact", "strided"])
@pytest.mark.parametrize("d", [5, 66, 130], ids=lambda d: f"d{d}")
def test_pointer_combine_head_dim_not_multiple_of_four(d, layout):
    """The pointer entry takes a runtime d_v: a lane whose four columns run past
    the row must neither read nor write the next row (the strided layout's
    untouched storage would show an overwrite)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, splits = 2, 3, 5, 3
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    ostride, lstride = _strides(b, h, sq, d, layout)
    o, ostorage, oused = _output((b, sq, h, d), ostride, torch.bfloat16)
    lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32)
    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True))
    assert fn is not None
    fn(op.data_ptr(), lp.data_ptr(), o.data_ptr(), lse.data_ptr(), (b, h, sq, d), splits, ostride, lstride, torch.cuda.current_stream().cuda_stream)
    _check(o, lse, ostorage, oused, lstorage, lused, ref_o, ref_lse, False)


def test_pointer_combine_reuses_artifact_for_new_shapes():
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    owner = comb.compile_ptr(dtype_o="bf16", has_lse=True)
    fn = positional_entry(owner)
    assert fn is not None
    for b, h, sq, d, splits in [(2, 3, 5, 160, 3), (1, 2, 3, 256, 5)]:
        op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
        ostride, lstride = _strides(b, h, sq, d, "strided")
        o, ostorage, oused = _output((b, sq, h, d), ostride, torch.bfloat16)
        lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32)
        fn(op.data_ptr(), lp.data_ptr(), o.data_ptr(), lse.data_ptr(), (b, h, sq, d), splits, ostride, lstride, torch.cuda.current_stream().cuda_stream)
        _check(o, lse, ostorage, oused, lstorage, lused, ref_o, ref_lse, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
def test_packed_combine_live_total_and_runtime_split_boundary(dtype, stats):
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    owner = comb.compile_ptr(dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=stats != "none", stats_log2=stats == "log2", packed=True)
    fn = positional_entry(owner)
    assert fn is not None
    b, h, sq, d = 1, 3, 7, 160
    ostride, lstride = _strides(b, h, sq, d, "int64_singleton")
    for splits in (2, 8, 32, 33):
        op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
        o, ostorage, oused = _output((b, sq, h, d), ostride, dtype)
        lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32) if stats != "none" else (None, None, None)
        total = torch.tensor([sq], device="cuda", dtype=torch.int32)

        def run():
            fn(
                op.data_ptr(),
                lp.data_ptr(),
                o.data_ptr(),
                lse.data_ptr() if lse is not None else None,
                (b, h, sq, d),
                splits,
                ostride,
                lstride,
                total.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
                None,  # This direct packed entry has no sink.
            )

        def check(live):
            expected = torch.full_like(ref_o, -31)
            expected[:, :live] = ref_o[:, :live]
            torch.testing.assert_close(o.cpu().float(), expected.float(), atol=0.004, rtol=0.004)
            assert torch.all(ostorage.cpu()[~oused] == -31)
            if lse is not None:
                expected_lse = torch.full_like(ref_lse, -31)
                expected_lse[:, :, :live] = ref_lse[:, :, :live] * (math.log2(math.e) if stats == "log2" else 1)
                torch.testing.assert_close(lse.cpu().double(), expected_lse, atol=2e-5, rtol=2e-5)
                assert torch.all(lstorage.cpu()[~lused] == -31)

        run()
        check(sq)
        captured = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(captured):
                run()
            for live in (3, 0):
                # The captured launch has seven rows of capacity. Device total
                # changes must prevent both partial-tail reads and final stores.
                total.fill_(live)
                op[:, live:].fill_(torch.nan)
                lp[:, :, live:].fill_(torch.nan)
                ostorage.fill_(-31)
                if lstorage is not None:
                    lstorage.fill_(-31)
                captured.replay()
                check(live)
        finally:
            captured.reset()


@pytest.mark.parametrize("layout", ["compact", "strided", "int64_singleton"])
def test_pointer_combine_fp16_strided_output(layout):
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = (1 if layout == "int64_singleton" else 2), 3, 5, 160, 3
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    ostride, lstride = _strides(b, h, sq, d, layout)
    o, ostorage, oused = _output((b, sq, h, d), ostride, torch.float16)
    lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32)
    fn = positional_entry(comb.compile_ptr(has_lse=True, dtype_partial="f32"))
    fn(op.data_ptr(), lp.data_ptr(), o.data_ptr(), lse.data_ptr(), (b, h, sq, d), splits, o.stride(), lse.stride(), torch.cuda.current_stream().cuda_stream)
    _check(o, lse, ostorage, oused, lstorage, lused, ref_o, ref_lse, False)


@pytest.mark.parametrize("dtype,tag", [(torch.float8_e4m3fn, "e4m3"), (torch.float8_e5m2, "e5m2")])
def test_quantized_pointer_combine_keeps_scale_and_amax(dtype, tag):
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 2, 3, 5, 160, 3
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    # Quantized outputs bind current pointers and physical output strides.
    o = torch.empty((b, sq, h, d), device="cuda", dtype=dtype)
    lse = torch.full((b, h, sq), torch.nan, device="cuda")
    amax = torch.zeros(1, dtype=torch.float32, device="cuda")
    scale = torch.tensor([1.75], dtype=torch.float32, device="cuda")
    fn = positional_entry(
        comb.compile_ptr(
            dtype_o=tag, has_lse=True, has_amax=True, dtype_partial="f32", has_scale_o=True, stats_log2=True, quantized=True, has_scale_o_input=True
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
    assert torch.count_nonzero(o[:, 0].float()).item() == 0


def test_pointer_combine_folds_the_sink_once():
    """The sink entries: M = max(max_s lse_s, sink), den = sum_s exp(lse_s - M) +
    exp(sink - M), O = sum_s w_s O_s / den (the sink column has no V row) and
    LSE = M + log(den).  Row 0 has every split dead, so its whole mass is the
    sink: O := 0 and LSE := sink exactly (not -inf); the three heads carry a
    sink far below, inside and above the split LSE range."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 2, 3, 5, 128, 3
    op, lp, _, _ = _partials(b, h, sq, d, splits)
    sinks = torch.tensor([-120.0, 0.7, 4.0], device="cuda")
    lse = lp.view(splits, b, h, sq).double().cpu()
    o = op.view(splits, b, sq, h, d).double().cpu()
    dead = torch.isneginf(lse)
    sink_col = sinks.double().cpu().view(1, h, 1).expand(b, h, sq)
    maximum = torch.maximum(lse.amax(0), sink_col)
    weights = torch.where(dead, torch.zeros_like(lse), (lse - maximum).exp())
    denominator = weights.sum(0) + (sink_col - maximum).exp()
    safe_o = o.masked_fill(dead.permute(0, 1, 3, 2).unsqueeze(-1), 0)
    ref_o = (weights.permute(0, 1, 3, 2).unsqueeze(-1) * safe_o).sum(0) / denominator.permute(0, 2, 1).unsqueeze(-1)
    ref_lse = maximum + denominator.log()
    o_out = torch.empty(b, sq, h, d, device="cuda", dtype=torch.bfloat16)
    lse_out = torch.empty(b, h, sq, device="cuda", dtype=torch.float32)
    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, has_sink=True))
    assert fn is not None
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o_out.data_ptr(),
        lse_out.data_ptr(),
        (b, h, sq, d),
        splits,
        o_out.stride(),
        lse_out.stride(),
        sinks.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(o_out.cpu().double(), ref_o, atol=0.004, rtol=0.004)
    torch.testing.assert_close(lse_out.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(o_out[:, 0]).item() == 0
    torch.testing.assert_close(lse_out[:, :, 0].cpu(), sinks.cpu().view(1, h).expand(b, h), atol=1e-6, rtol=0)


@pytest.mark.parametrize("sink", [False, True], ids=["plain", "sink"])
@pytest.mark.parametrize("d", [128, 66], ids=["fp32_rows_d128", "elementwise_d66"])
@pytest.mark.parametrize("splits", [2, 4, 5, 8, 9, 12, 16, 17, 21, 37, 64, 100, 128], ids=lambda s: f"split{s}")
@pytest.mark.parametrize("sq", [1, 2, 5], ids=lambda s: f"sq{s}")
def test_pointer_combine_walks_many_splits(sq, splits, d, sink):
    """Both forms of the one-warp-per-row kernel across the split axis.  The lean form (aligned fp32 rows, d % 4 == 0) serves
    n_splits <= G_S: every split's LSE and partial-O row is loaded up front through a clamped index, in a group of G_LEAN_SMALL (8) for the common few
    splits and of G_S (16) beyond that.  Past G_S -- and always for d = 66, whose rows take bounded element loads -- the serial form walks the splits in
    two legs, sixteen deep and then four deep (``_serial_legs`` and the ``_accumulate_splits`` helper).  2 to 17 splits stay below a full serial group
    (tail leg only; 2 to 8 are the narrow lean group for d = 128, 9 to 16 the wide one), 21 takes one group plus a remainder, 37 two; 64 to 128 are the long walks of a b = 1 long-KV decode (the decode form's O pass keeps eight splits in flight per warp); row 0 has every
    split dead and split 1 is dead on row 1, so the dead-split select runs inside every form.  The sink arm seeds the denominator with the sink column
    and lands the keyless row on ``O := 0 / LSE := sink``.
    S_q = 1 and 2 are the decode rows: past G_S splits they take the decode form (the block's warps share the split axis), within G_S the
    one-warp-per-row kernel, so both sides of that dispatch run against the same reference."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h = 2, 3
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    sinks = torch.tensor([-120.0, 0.7, 4.0], device="cuda")
    if sink:
        lse = lp.view(splits, b, h, sq).double().cpu()
        o = op.view(splits, b, sq, h, d).double().cpu()
        dead = torch.isneginf(lse)
        sink_col = sinks.double().cpu().view(1, h, 1).expand(b, h, sq)
        maximum = torch.maximum(lse.amax(0), sink_col)
        weights = torch.where(dead, torch.zeros_like(lse), (lse - maximum).exp())
        denominator = weights.sum(0) + (sink_col - maximum).exp()
        safe_o = o.masked_fill(dead.permute(0, 1, 3, 2).unsqueeze(-1), 0)
        ref_o = (weights.permute(0, 1, 3, 2).unsqueeze(-1) * safe_o).sum(0) / denominator.permute(0, 2, 1).unsqueeze(-1)
        ref_lse = maximum + denominator.log()
    ostride, lstride = _strides(b, h, sq, d, "strided")
    o_out, ostorage, oused = _output((b, sq, h, d), ostride, torch.bfloat16)
    lse_out, lstorage, lused = _output((b, h, sq), lstride, torch.float32)
    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, has_sink=sink))
    assert fn is not None
    tail = (sinks.data_ptr(),) if sink else ()
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o_out.data_ptr(),
        lse_out.data_ptr(),
        (b, h, sq, d),
        splits,
        ostride,
        lstride,
        *tail,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    if not sink:
        _check(o_out, lse_out, ostorage, oused, lstorage, lused, ref_o, ref_lse, False)
        return
    torch.testing.assert_close(o_out.cpu().double(), ref_o, atol=0.004, rtol=0.004)
    torch.testing.assert_close(lse_out.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(o_out[:, 0]).item() == 0
    torch.testing.assert_close(lse_out[:, :, 0].cpu(), sinks.cpu().view(1, h).expand(b, h), atol=1e-6, rtol=0)
    assert torch.all(ostorage.float().cpu()[~oused] == -31.0)
    assert torch.all(lstorage.cpu()[~lused] == -31.0)


def _sink_partials_with_padding(b, h, sq, d, splits, q_lens, sinks):
    """Split partials of a sink launch whose batch ``i`` has only ``q_lens[i]`` live query rows: a PADDED row (r >= q_lens[i]) carries every split dead,
    exactly as the split kernels write it, and so does row 0 of every batch (a valid row with no key in range for q_lens[i] > 0).  Returns the device
    partials and the float64 reference of the unsplit contract: valid rows fold the sink once (a keyless one lands on O = 0 / LSE = sink), padded rows
    stay O = 0 / LSE = -inf whatever the sink."""
    generator = torch.Generator().manual_seed(20261015 + splits)
    o = torch.randn(splits, b, sq, h, d, generator=generator) * 0.4
    lse = torch.randn(splits, b, h, sq, generator=generator) * 2.0 + 0.7
    lse[:, :, :, 0] = -torch.inf
    if sq > 1:
        lse[1, :, :, 1] = -torch.inf  # a dead split beside live ones
    padded = torch.zeros(b, sq, dtype=torch.bool)
    for i, n in enumerate(q_lens):
        padded[i, n:] = True
        lse[:, i, :, n:] = -torch.inf
    dead = torch.isneginf(lse).permute(0, 1, 3, 2).unsqueeze(-1)
    o.masked_fill_(dead, torch.nan)  # a dead payload must never enter the arithmetic
    scores = lse.double()
    sink_col = sinks.double().cpu().view(1, h, 1).expand(b, h, sq)
    maximum = torch.maximum(scores.amax(0), sink_col)
    weights = torch.where(torch.isneginf(scores), torch.zeros_like(scores), (scores - maximum).exp())
    denominator = weights.sum(0) + (sink_col - maximum).exp()
    safe_o = o.double().masked_fill(dead, 0)
    ref_o = (weights.permute(0, 1, 3, 2).unsqueeze(-1) * safe_o).sum(0) / denominator.permute(0, 2, 1).unsqueeze(-1)
    ref_lse = maximum + denominator.log()
    pad_o = padded.view(b, sq, 1, 1).expand(b, sq, h, d)
    pad_lse = padded.view(b, 1, sq).expand(b, h, sq)
    ref_o = ref_o.masked_fill(pad_o, 0.0)
    ref_lse = ref_lse.masked_fill(pad_lse, -torch.inf)
    return o.reshape(splits * b, sq, h, d).cuda(), lse.reshape(splits * b, h, sq).cuda(), ref_o, ref_lse, pad_o, pad_lse


@pytest.mark.parametrize(
    "sq, splits, d, q_lens",
    [
        (1, 4, 128, [1, 0, 1]),
        (1, 12, 128, [0, 1, 1]),
        (1, 64, 128, [1, 1, 0]),
        (2, 33, 128, [2, 1, 0]),
        (5, 4, 128, [5, 2, 0]),
        (5, 21, 128, [3, 0, 5]),
        (1, 33, 66, [0, 1, 1]),
        (5, 21, 66, [2, 5, 0]),
    ],
    ids=["lean_narrow", "lean_wide", "decode_64", "decode_33_sq2", "lean_sq5", "serial_21", "decode_d66", "serial_d66"],
)
def test_pointer_combine_sink_keeps_padded_query_rows_dead(sq, splits, d, q_lens):
    """A padded query row (past its batch's Q length: the dense padded-Q trim) and a valid row whose every key is masked reach the combine with the
    same partials (every split dead).  Folding the sink unconditionally gave both LSE = sink; the sink entry that takes the per-batch Q lengths
    drops the sink for the padded rows (O := 0 / LSE := -inf) and keeps the unsplit contract for the valid keyless one (O := 0 / LSE := sink) --
    in the lean, serial and decode forms alike, with the fp32 rows and the element-load rows."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h = 3, 3
    sinks = torch.tensor([-120.0, 0.7, 4.0], device="cuda")
    op, lp, ref_o, ref_lse, pad_o, pad_lse = _sink_partials_with_padding(b, h, sq, d, splits, q_lens, sinks)
    q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device="cuda")
    o_out = torch.full((b, sq, h, d), -31.0, device="cuda", dtype=torch.bfloat16)
    lse_out = torch.full((b, h, sq), -31.0, device="cuda", dtype=torch.float32)
    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, has_sink=True, has_q_lens=True))
    assert fn is not None
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o_out.data_ptr(),
        lse_out.data_ptr(),
        (b, h, sq, d),
        splits,
        o_out.stride(),
        lse_out.stride(),
        sinks.data_ptr(),
        q_lens_t.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    o_cpu, lse_cpu = o_out.cpu().double(), lse_out.cpu().double()
    torch.testing.assert_close(o_cpu, ref_o, atol=0.004, rtol=0.004)
    assert torch.all(o_cpu[pad_o] == 0.0), "a padded row must write O := 0"
    assert torch.all(torch.isneginf(lse_cpu[pad_lse])), "a padded row must write LSE := -inf whatever the sink"
    torch.testing.assert_close(lse_cpu[~pad_lse], ref_lse[~pad_lse], atol=2e-5, rtol=2e-5)
    keyless = ~pad_lse & torch.isneginf(lp.view(splits, b, h, sq).cpu().amax(0))
    assert keyless.any(), "the case needs a valid keyless row"
    torch.testing.assert_close(lse_cpu[keyless], sinks.cpu().double().view(1, h, 1).expand(b, h, sq)[keyless], atol=1e-6, rtol=0)


def test_pointer_combine_q_lens_is_a_dense_sink_specialization():
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    with pytest.raises(ValueError, match="sink"):
        comb.compile_ptr(dtype_o="bf16", has_lse=True, has_q_lens=True)
    with pytest.raises(ValueError, match="dense"):
        comb.compile_ptr(dtype_o="bf16", has_lse=True, has_sink=True, has_q_lens=True, ragged=True)
