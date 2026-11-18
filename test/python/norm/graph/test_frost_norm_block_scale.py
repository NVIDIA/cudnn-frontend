# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Norm block quantization: exact quantizer bytes and graph reference checks."""

import pytest
import torch
import cudnn

from cudnn.norm._lazy import dsl_unavailable

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def _supported(monkeypatch):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("norm MXFP8 requires SM100, SM103, or SM107")
    reason = dsl_unavailable("FROST norm block scale")
    if reason:
        pytest.skip(reason)
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def mxfp8_reference(x, dtype, axis):
    """Integer exponent oracle; avoids log2 rounding at power-of-two boundaries."""
    values = x.float().movedim(axis, -1).contiguous()
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    raw = (blocks.abs().amax(-1) / torch.finfo(dtype).max).clamp_min(torch.finfo(torch.float32).tiny)
    bits = raw.view(torch.int32)
    exponent = (bits >> 23) & 255
    byte = (exponent + ((bits & 0x7FFFFF) != 0)).clamp_max(254).to(torch.uint8)
    scale = (byte.to(torch.int32) << 23).view(torch.float32)
    scale = torch.where(torch.isinf(raw), raw, scale)
    codes = (blocks / scale[..., None]).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max).to(dtype)
    return codes.reshape(values.shape).movedim(-1, axis).contiguous().view(torch.uint8), byte.movedim(-1, axis).contiguous()


def _compile_quant(x, row, sf, col, csf, transpose=False):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor
    from cudnn.frost.compiled_cache import compile_cached
    from cudnn.norm.block_scale_kernels import mxfp8
    from cudnn.norm.utils import fake_stream

    dtype = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
    args = [
        None if t is None else make_fake_tensor(dtype.get(t.dtype, cutlass.Uint8), tuple(t.shape), tuple(t.stride()), assumed_align=16)
        for t in (x, row, sf, col, csf)
    ]
    n = x.shape[-1]
    e5m2 = row.dtype == torch.float8_e5m2
    key = repr(([(tuple(t.shape), tuple(t.stride()), t.dtype) if t is not None else None for t in (x, row, sf, col, csf)]))
    return compile_cached(mxfp8, *args, e5m2, fake_stream(), cache_key="norm_block_scale_test:" + key, options="--enable-tvm-ffi")


def _launch_quant(fn, tensors):
    from cutlass.cute.runtime import from_dlpack
    from cudnn.norm.utils import cur_stream

    args = [
        (
            None
            if t is None
            else from_dlpack(t.view(torch.uint8) if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else t, assumed_align=16, enable_tvm_ffi=True)
        )
        for t in tensors
    ]
    fn(*args, cur_stream())


@pytest.mark.parametrize("shape", [(2, 64, 256), (1, 64, 96), (1, 96, 192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("transpose", [False, True])
def test_frost_norm_mxfp8_quantizer(shape, dtype, out_dtype, transpose):
    torch.manual_seed(2026)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    # A zero block must emit 0x01 (FLT_MIN), including in the column direction.
    x[:, :32, :32] = 0
    row = torch.empty_like(x, dtype=out_dtype)
    sf = torch.empty((*shape[:-1], shape[-1] // 32), device="cuda", dtype=torch.uint8)
    col_shape = (shape[0], shape[2], shape[1]) if transpose else shape
    csf_shape = (shape[0], shape[2], shape[1] // 32) if transpose else (shape[0], shape[1] // 32, shape[2])
    col = torch.empty(col_shape, device="cuda", dtype=out_dtype)
    csf = torch.empty(csf_shape, device="cuda", dtype=torch.uint8)
    if transpose:
        col, csf = col.transpose(1, 2), csf.transpose(1, 2)
    for t in (row, sf, col, csf):
        t.view(torch.uint8).fill_(255)
    fn = _compile_quant(x, row, sf, col, csf)
    _launch_quant(fn, (x, row, sf, col, csf))
    ref_row, ref_sf = mxfp8_reference(x, out_dtype, 2)
    ref_col, ref_csf = mxfp8_reference(x, out_dtype, 1)
    for got, expected in ((row.view(torch.uint8), ref_row), (sf, ref_sf), (col.view(torch.uint8), ref_col), (csf, ref_csf)):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    first = [t.view(torch.uint8).clone() for t in (row, sf, col, csf)]
    for t in (row, sf, col, csf):
        t.view(torch.uint8).fill_(255)
    _launch_quant(fn, (x, row, sf, col, csf))
    for got, expected in zip((row, sf, col, csf), first):
        torch.testing.assert_close(got.view(torch.uint8), expected, rtol=0, atol=0)


def _quantized_graph(
    handle,
    kind,
    x,
    gamma,
    beta,
    phase,
    intermediate_dtype,
    out_dtype,
    *,
    adaptive=False,
    zero_centered_gamma=False,
    dy=None,
    mean=None,
    rstd=None,
    transpose=False,
):
    from test_frost_norm_affine import _graph

    g, vp, results = _graph(handle, kind, x, gamma, beta, phase=phase, adaptive=adaptive, zero_centered_gamma=zero_centered_gamma, dy=dy, mean=mean, rstd=rstd)
    role = "y" if phase == "fprop" else "dx"
    source_buffer = results.pop(role)
    source = next(t for t, buf in vp.items() if buf is source_buffer)
    del vp[source]
    source.set_output(False).set_data_type(intermediate_dtype)
    outputs = []
    for axis in (2, 1):
        yt, st = g.block_scale_quantize(input=source, block_size=32, axis=axis, transpose=transpose and axis == 1)
        shape = list(x.shape)
        sf_shape = shape.copy()
        sf_shape[axis] //= 32
        if transpose and axis == 1:
            # C++ transpose rotates the packed axis through the full layout:
            # [B,M,N] becomes physical [N,B,M], with unchanged logical dims.
            y = torch.empty((shape[2], shape[0], shape[1]), device=x.device, dtype=out_dtype).permute(1, 2, 0)
            sf = torch.empty((sf_shape[2], sf_shape[0], sf_shape[1]), device=x.device, dtype=torch.float8_e8m0fnu).permute(1, 2, 0)
        else:
            y = torch.empty(shape, device=x.device, dtype=out_dtype)
            sf = torch.empty(sf_shape, device=x.device, dtype=torch.float8_e8m0fnu)
        for t, buf in ((yt, y), (st, sf)):
            t.set_output(True).set_data_type(buf.dtype).set_dim(buf.shape).set_stride(buf.stride())
            vp[t] = buf
        y.view(torch.uint8).fill_(255)
        sf.view(torch.uint8).fill_(255)
        outputs.append((y, sf))
    return g, vp, results, outputs, source


def _build(g, phase):
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    candidates = [i for i, p in enumerate(g.plans) if p.engine_id >= 20_000]
    assert candidates, "FROST must offer a block-scaled norm plan"
    g.select_plan(candidates[0])
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == f"norm_{phase}_sm100"
    return torch.full((g.get_workspace_size(),), 255, device="cuda", dtype=torch.uint8)


def _check_quantized(staged, outputs, dtype):
    for axis, (y, sf) in zip((2, 1), outputs):
        ref_y, ref_sf = mxfp8_reference(staged, dtype, axis)
        torch.testing.assert_close(y.view(torch.uint8), ref_y, rtol=0, atol=0)
        torch.testing.assert_close(sf.view(torch.uint8), ref_sf, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind,layout,has_bias",
    [
        pytest.param("ln", "shared", True, id="ln"),
        pytest.param("rms", "shared", False, id="rms"),
        pytest.param("ln", "batch", True, id="adaln"),
        pytest.param("ln", "sequence", True, id="adaln_sequence"),
        pytest.param("rms", "batch", False, id="adarms_batch_scale"),
        pytest.param("rms", "sequence", False, id="adarms_sequence_scale"),
        pytest.param("rms", "batch", True, id="adarms_batch_scale_shift"),
        pytest.param("rms", "sequence", True, id="adarms_sequence_scale_shift"),
    ],
)
@pytest.mark.parametrize(
    "dtype,intermediate_dtype", [(x, y) for x in (torch.bfloat16, torch.float16, torch.float32) for y in (torch.bfloat16, torch.float16, torch.float32)]
)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_frost_norm_block_scale_forward_backward(cudnn_handle, kind, layout, has_bias, dtype, intermediate_dtype, zero_centered_gamma, out_dtype):
    from test_frost_norm_affine import _reference

    torch.manual_seed(10)
    x = torch.randn((2, 32, 96), device="cuda", dtype=dtype).requires_grad_()
    adaptive = layout != "shared"
    scale_shape = {"shared": (1, 1, 96), "batch": (2, 1, 96), "sequence": (1, 32, 96)}[layout]
    gamma = torch.randn(scale_shape, device="cuda", dtype=torch.float32).requires_grad_()
    beta = torch.randn_like(gamma).requires_grad_() if has_bias else None
    yref, mean, rstd, leaves = _reference(kind, x, gamma, beta, zero_centered_gamma, torch.float32)
    g, vp, stats, outputs, source = _quantized_graph(
        cudnn_handle, kind, x, gamma, beta, "fprop", intermediate_dtype, out_dtype, adaptive=adaptive, zero_centered_gamma=zero_centered_gamma
    )
    from cudnn.norm.graph_analyzer import analyze

    assert analyze(g).adaptive == adaptive
    ws = _build(g, "fprop")
    original_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        g.execute(vp, ws, handle=cudnn_handle)
        staged = ws.view(intermediate_dtype).reshape(x.shape)
        torch.testing.assert_close(staged, yref.to(intermediate_dtype))
        if kind == "ln":
            torch.testing.assert_close(stats["mean"], mean)
        torch.testing.assert_close(stats["rstd"], rstd)
        _check_quantized(staged, outputs, out_dtype)
        # Same notebook loss construction; use the same rounded upstream
        # gradient in both backward implementations.
        loss = torch.nn.functional.mse_loss(yref, torch.randn_like(x).float(), reduction="mean")
        dy = torch.autograd.grad(loss, yref, retain_graph=True)[0].to(dtype)
        refs = torch.autograd.grad(yref, tuple(t for t in leaves if t is not None), dy.float())
        bg, bvp, grads, bouts, _ = _quantized_graph(
            cudnn_handle,
            kind,
            x,
            gamma,
            beta,
            "bprop",
            intermediate_dtype,
            out_dtype,
            adaptive=adaptive,
            zero_centered_gamma=zero_centered_gamma,
            dy=dy,
            mean=stats.get("mean"),
            rstd=stats["rstd"],
        )
        assert analyze(bg).adaptive == adaptive
        bws = _build(bg, "bprop")
        bg.execute(bvp, bws, handle=cudnn_handle)
        dx = bws.view(intermediate_dtype).reshape(x.shape)
        torch.testing.assert_close(dx, refs[0].to(intermediate_dtype))
        torch.testing.assert_close(grads["dgamma"], refs[1])
        if beta is not None:
            torch.testing.assert_close(grads["dbeta"], refs[2])
        _check_quantized(dx, bouts, out_dtype)
    finally:
        cudnn.set_stream(cudnn_handle, original_stream)


def nvfp4_reference(x):
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    raw = (blocks.abs().amax(-1) / 6).clamp_min(torch.finfo(torch.float32).tiny)
    scaled = blocks / raw[..., None]
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(scaled.abs().contiguous(), boundaries, right=True)
    tie = (code > 0) & (scaled.abs() == boundaries[(code - 1).clamp_min(0)])
    code -= (tie & (code % 2 == 1)).to(code.dtype)
    code = (code.to(torch.uint8) | (torch.signbit(scaled).to(torch.uint8) << 3)).reshape(x.shape)
    packed = code[..., ::2] | (code[..., 1::2] << 4)
    return packed, raw.clamp_max(448).to(torch.float8_e4m3fn).view(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "kind,layout,has_bias",
    [
        pytest.param("ln", "shared", True, id="ln"),
        pytest.param("rms", "shared", False, id="rms"),
        pytest.param("ln", "batch", True, id="adaln"),
        pytest.param("ln", "sequence", True, id="adaln_sequence"),
        pytest.param("rms", "batch", False, id="adarms_batch_scale"),
        pytest.param("rms", "sequence", False, id="adarms_sequence_scale"),
        pytest.param("rms", "batch", True, id="adarms_batch_scale_shift"),
        pytest.param("rms", "sequence", True, id="adarms_sequence_scale_shift"),
    ],
)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_frost_norm_nvfp4_inference(cudnn_handle, dtype, kind, layout, has_bias, zero_centered_gamma):
    from test_frost_norm_affine import _graph, _reference

    torch.manual_seed(40)
    x = torch.randn((2, 7, 96), device="cuda", dtype=dtype)
    scale_shape = {"shared": (1, 1, 96), "batch": (2, 1, 96), "sequence": (1, 7, 96)}[layout]
    gamma = torch.randn(scale_shape, device="cuda", dtype=torch.float32)
    beta = torch.randn_like(gamma) if has_bias else None
    g, vp, result = _graph(cudnn_handle, kind, x, gamma, beta, adaptive=layout != "shared", zero_centered_gamma=zero_centered_gamma, inference=True)
    source = next(t for t, buf in vp.items() if buf is result["y"])
    del vp[source]
    source.set_output(False).set_data_type(torch.float32)
    yt, st = g.block_scale_quantize(input=source, block_size=16, axis=2)
    y = torch.full((2, 7, 48), 255, device="cuda", dtype=torch.uint8)
    sf = torch.full((2, 7, 6), float("nan"), device="cuda", dtype=torch.float8_e4m3fn)
    yt.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1).set_dim(x.shape).set_stride(x.stride())
    st.set_output(True).set_data_type(sf.dtype).set_dim(sf.shape).set_stride(sf.stride())
    vp.update({yt: y, st: sf})
    ws = _build(g, "fprop")
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        g.execute(vp, ws, handle=cudnn_handle)
        staged = ws.view(torch.float32).reshape(x.shape)
        ref, _, _, _ = _reference(kind, x, gamma, beta, zero_centered_gamma, torch.float32)
        torch.testing.assert_close(staged, ref)
        code, scale = nvfp4_reference(staged)
        torch.testing.assert_close(y, code, rtol=0, atol=0)
        torch.testing.assert_close(sf.view(torch.uint8), scale, rtol=0, atol=0)
    finally:
        cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize("explicit_stride", [True, False])
def test_frost_norm_block_scale_transposed_forward(cudnn_handle, explicit_stride):
    x = torch.randn((2, 64, 96), device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((1, 1, 96), device="cuda", dtype=torch.float32)
    g, vp, _, outputs, _ = _quantized_graph(cudnn_handle, "rms", x, gamma, None, "fprop", torch.float32, torch.float8_e4m3fn, transpose=True)
    if not explicit_stride:
        for t in g.nodes[-1].outputs.values():
            t.stride_assigned = False
            t.stride = []
    ws = _build(g, "fprop")
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        g.execute(vp, ws, handle=cudnn_handle)
        _check_quantized(ws.view(torch.float32).reshape(x.shape), outputs, torch.float8_e4m3fn)
    finally:
        cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("columns", [False, True])
def test_frost_norm_quantizer_boundaries(out_dtype, columns):
    # Include exact exponent transitions and their adjacent FP32 values,
    # the FLT_MIN clamp, subnormals, signed zero, and large finite values.
    maximum = torch.finfo(out_dtype).max
    boundary = torch.ldexp(torch.full((245,), maximum, device="cuda"), torch.arange(-126, 119, device="cuda"))
    boundary = torch.cat(
        (
            torch.nextafter(boundary, torch.zeros_like(boundary)),
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, float("inf"))),
            torch.tensor([0.0, -0.0, torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).max], device="cuda"),
        )
    )
    boundary = boundary[torch.isfinite(boundary)]
    # A block is one row; all 32 rows in each column block share its maximum.
    m = ((boundary.numel() + 31) // 32) * 32 if columns else boundary.numel()
    x = torch.zeros((1, m, 32), device="cuda")
    x[0, : boundary.numel(), :] = boundary[:, None]
    x[..., 1::2].neg_()
    x[..., 2] *= 0.5
    row = torch.full(x.shape, float("nan"), device="cuda", dtype=out_dtype)
    sf = torch.full((1, m, 1), 255, device="cuda", dtype=torch.uint8)
    col = torch.full_like(row, float("nan")) if columns else None
    csf = torch.full((1, m // 32, 32), 255, device="cuda", dtype=torch.uint8) if columns else None
    fn = _compile_quant(x, row, sf, col, csf)
    _launch_quant(fn, (x, row, sf, col, csf))
    ry, rs = mxfp8_reference(x, out_dtype, 2)
    torch.testing.assert_close(row.view(torch.uint8), ry, rtol=0, atol=0)
    torch.testing.assert_close(sf, rs, rtol=0, atol=0)
    if columns:
        cy, cs = mxfp8_reference(x, out_dtype, 1)
        torch.testing.assert_close(col.view(torch.uint8), cy, rtol=0, atol=0)
        torch.testing.assert_close(csf, cs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "row_only_backward",
        "transpose_backward",
        "group_offset",
        "compute_half",
        "reordered",
        "bound_intermediate",
        "virtual_payload",
        "mismatched_payloads",
        "wrong_axis",
        "wrong_stride",
        "malformed_adaptive_scale",
    ],
)
def test_frost_norm_block_scale_declines(cudnn_handle, monkeypatch, case):
    from cudnn.norm.bprop.engine import FrostNormBwdEngine
    from cudnn.norm.bprop.engines import ENGINE_SPECS
    from cudnn.norm.graph_analyzer import analyze
    import cutlass.cute as cute

    x = torch.randn((2, 32, 96), device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones((1, 1, 96), device="cuda", dtype=torch.float32)
    g, _, _, _, source = _quantized_graph(
        cudnn_handle,
        "ln",
        x,
        gamma,
        gamma,
        "bprop",
        torch.float32,
        torch.float8_e4m3fn,
        dy=x,
        mean=torch.zeros((2, 32, 1), device="cuda"),
        rstd=torch.ones((2, 32, 1), device="cuda"),
    )
    engine = FrostNormBwdEngine(ENGINE_SPECS[0], 20_000)
    assert engine._decline_reason(g, None) is None, "negative tests require an accepted control graph"
    norm, row, col = g.nodes
    if case == "row_only_backward":
        g._nodes.remove(col)
    elif case == "transpose_backward":
        col.params["transpose"] = True
    elif case == "group_offset":
        row.inputs["group_offset"] = g.tensor_like(torch.zeros(1, device="cuda", dtype=torch.int32))
    elif case == "compute_half":
        row.compute_data_type = cudnn.data_type.HALF
    elif case == "reordered":
        row.outputs["scale"].set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    elif case == "bound_intermediate":
        source.set_output(True)
    elif case == "virtual_payload":
        row.outputs["Y"].set_output(False)
    elif case == "mismatched_payloads":
        col.outputs["Y"].set_data_type(cudnn.data_type.FP8_E5M2)
    elif case == "wrong_axis":
        col.params["axis"] = 0
    elif case == "wrong_stride":
        row.outputs["Y"].set_stride((3072, 1, 32))
    elif case == "malformed_adaptive_scale":
        norm.params["adaptive"] = True
        norm.node_type = cudnn.graph_types.NodeType.RMSNORM_BWD
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("a declined declaration must not compile"))
    reason = engine._decline_reason(g, None)
    expected = {
        "row_only_backward": "backward requires row and column",
        "transpose_backward": "only forward columnwise",
        "group_offset": "without group offsets",
        "compute_half": "fp32 computation",
        "reordered": "plain device",
        "bound_intermediate": "virtual Y/DX",
        "virtual_payload": "bound outputs",
        "mismatched_payloads": "same payload dtype",
        "wrong_axis": "distinct rowwise/columnwise",
        "wrong_stride": "shape/strides",
        "malformed_adaptive_scale": "adaptive scale",
    }
    assert reason is not None and expected[case] in reason
    if case == "malformed_adaptive_scale":
        assert "adaptive scale" in analyze(g).invalid


@pytest.mark.parametrize("phase", ["fprop", "bprop"])
def test_frost_norm_block_scale_rebind_capture_and_serdes(cudnn_handle, monkeypatch, phase):
    import cutlass.cute as cute
    from test_frost_norm_no_alloc import _AllocWatch

    torch.manual_seed(42)
    x = torch.randn((2, 32, 96), device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((2, 1, 96), device="cuda")
    mean = x.float().mean(-1, keepdim=True)
    rstd = (x.float().var(-1, unbiased=False, keepdim=True) + 1e-3).rsqrt()
    g, vp, _, outputs, _ = _quantized_graph(
        cudnn_handle,
        "ln",
        x,
        gamma,
        gamma,
        phase,
        torch.float32,
        torch.float8_e4m3fn,
        adaptive=True,
        dy=x * 0.01,
        mean=mean,
        rstd=rstd,
        zero_centered_gamma=True,
    )
    ws = _build(g, phase)
    saved_stream = cudnn.get_stream(cudnn_handle)
    capture = None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            cudnn.set_stream(cudnn_handle, stream.cuda_stream)
            g.execute(vp, ws, handle=cudnn_handle)
            _check_quantized(ws.view(torch.float32).reshape(x.shape), outputs, torch.float8_e4m3fn)
            # New allocations and UID keys exercise rebinding, including a
            # serialized plan. All compilation must remain at plan construction.
            blob = g.serialize(as_bytes=True)
            clone = cudnn.pygraph(handle=cudnn_handle)
            clone.deserialize(blob)
            vp2 = {t.get_uid(): v.clone() for t, v in vp.items()}
            output_uids = [t.get_uid() for t in vp if not t.get_is_virtual() and t in [v for n in g.nodes for v in n.outputs.values()]]
            for uid in output_uids:
                vp2[uid].view(torch.uint8).fill_(255)
            ws2 = torch.full_like(ws, 255)
            monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute compiled a kernel"))
            with _AllocWatch() as watch:
                clone.execute(vp2, ws2, handle=cudnn_handle)
            assert not watch.calls
            torch.testing.assert_close(ws2, ws, rtol=0, atol=0)
            clone_outputs = [tuple(vp2[t.get_uid()] for t in q.outputs.values()) for q in g.nodes if q.node_type.name == "BLOCK_SCALE_QUANTIZE"]
            _check_quantized(ws2.view(torch.float32).reshape(x.shape), clone_outputs, torch.float8_e4m3fn)
            for t in vp:
                if t.get_uid() in output_uids and vp[t].dtype == torch.float32:
                    torch.testing.assert_close(vp2[t.get_uid()], vp[t])
            # Capture must contain work, then replay with changed inputs.
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                g.execute(vp, ws, handle=cudnn_handle)
            x[..., ::3].add_(0.5)
            for y, sf in outputs:
                y.view(torch.uint8).fill_(255)
                sf.view(torch.uint8).fill_(255)
            ws.fill_(255)
            capture.replay()
            _check_quantized(ws.view(torch.float32).reshape(x.shape), outputs, torch.float8_e4m3fn)
            captured = ws.clone()
            g.execute(vp, ws, handle=cudnn_handle)
            torch.testing.assert_close(captured, ws, rtol=0, atol=0)
            _check_quantized(ws.view(torch.float32).reshape(x.shape), outputs, torch.float8_e4m3fn)
    finally:
        stream.synchronize()
        if capture is not None:
            capture.reset()
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize(
    "invalid",
    [
        "short_workspace",
        "workspace_alias",
        "output_alias",
        "missing_scale",
        "scale_dtype",
        "misaligned_workspace",
        "output_device",
        "output_stride",
        "workspace_type",
    ],
)
def test_frost_norm_block_scale_rejects_before_writes(cudnn_handle, monkeypatch, invalid):
    x = torch.randn((2, 32, 96), device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones((1, 1, 96), device="cuda")
    g, vp, _, outputs, _ = _quantized_graph(cudnn_handle, "rms", x, gamma, None, "fprop", torch.bfloat16, torch.float8_e4m3fn)
    from cudnn.norm import affine

    def unexpected_launch(*args, **kwargs):
        pytest.fail("invalid binding reached the norm launch")

    monkeypatch.setattr(affine, "lower", lambda *args, **kwargs: unexpected_launch)
    ws = _build(g, "fprop")
    row, sf = g.nodes[-2].outputs.values()
    if invalid == "misaligned_workspace":
        ws = torch.empty(ws.numel() + 1, device="cuda", dtype=torch.uint8)[1:]
    elif invalid == "output_device":
        vp[row] = vp[row].cpu()
    elif invalid == "output_stride":
        vp[row] = torch.empty((2, 96, 32), device="cuda", dtype=vp[row].dtype).transpose(1, 2)
    elif invalid == "workspace_type":
        ws = object()
    elif invalid == "short_workspace":
        ws = ws[:-16]
    elif invalid == "workspace_alias":
        ws = x.view(torch.uint8).reshape(-1)
    elif invalid == "output_alias":
        vp[row] = x.view(torch.uint8).reshape(-1)[: x.numel()].view(torch.float8_e4m3fn).reshape(x.shape)
    elif invalid == "missing_scale":
        del vp[sf]
    elif invalid == "scale_dtype":
        vp[sf] = vp[sf].view(torch.int8)
    buffers = [v for t, v in vp.items() if not t.get_is_pass_by_value() and v.device.type == "cuda"]
    before = [v.view(torch.uint8).clone() for v in buffers]
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        with pytest.raises((ValueError, RuntimeError), match="workspace|overlapping|storage|binding|variant|dtype|missing"):
            g.execute(vp, ws, handle=cudnn_handle)
        for old, current in zip(before, buffers):
            torch.testing.assert_close(old, current.view(torch.uint8), rtol=0, atol=0)
    finally:
        cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize("shape", [(37, 64), (2, 1, 37, 96), (2, 1, 32, 128)])
@pytest.mark.parametrize("byte_storage", [False, True])
@pytest.mark.parametrize("canonical_declaration", [False, True])
def test_frost_norm_block_scale_singletons_and_byte_offsets(cudnn_handle, shape, byte_storage, canonical_declaration):
    from test_frost_norm_affine import _graph, _reference

    torch.manual_seed(2027)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((1,) * (len(shape) - 1) + (shape[-1],), device="cuda")
    g, vp, results = _graph(cudnn_handle, "rms", x, gamma, None)
    source = next(t for t, buf in vp.items() if buf is results["y"])
    del vp[source]
    source.set_output(False).set_data_type(torch.float32)
    y, sf = g.block_scale_quantize(input=source, block_size=32, axis=len(shape) - 1)
    outs = []
    for t, dim, dtype in ((y, shape, torch.float8_e4m3fn), (sf, (*shape[:-1], shape[-1] // 32), torch.float8_e8m0fnu)):
        # Each output is a disjoint slice with an odd byte offset. Singleton
        # strides need not equal their canonical row-major value.
        n = __import__("math").prod(dim)
        buf = torch.full((n + 1,), 255, device="cuda", dtype=torch.uint8)[1:].view(dtype).reshape(dim)
        if len(dim) == 4:
            buf = buf.as_strided(dim, (dim[-2] * dim[-1], 1, dim[-1], 1))
        t.set_output(True).set_dim(dim).set_stride(buf.stride()).set_data_type(dtype)
        if canonical_declaration:
            from cudnn.norm.block_scale import _strides

            t.set_stride(_strides(dim, len(dim) - 1, False))
        vp[t] = buf.view(torch.uint8) if byte_storage else buf
        outs.append(buf)
    ws = _build(g, "fprop")
    saved = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        g.execute(vp, ws, handle=cudnn_handle)
        staged = ws.view(torch.float32).reshape(shape)
        reference, _, _, _ = _reference("rms", x, gamma, None, False, torch.float32)
        torch.testing.assert_close(staged, reference)
        expected = mxfp8_reference(staged, torch.float8_e4m3fn, len(shape) - 1)
        for buf, ref in zip(outs, expected):
            torch.testing.assert_close(buf.view(torch.uint8), ref, rtol=0, atol=0)
    finally:
        cudnn.set_stream(cudnn_handle, saved)


@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_frost_norm_quantizer_infinite_blocks(out_dtype):
    x = torch.ones((1, 32, 32), device="cuda")
    x[0, 0, 0] = float("inf")
    x[0, 1, 1] = -float("inf")
    row = torch.full(x.shape, float("nan"), device="cuda", dtype=out_dtype)
    sf = torch.full((1, 32, 1), 255, device="cuda", dtype=torch.uint8)
    col, csf = torch.full_like(row, float("nan")), torch.full((1, 1, 32), 255, device="cuda", dtype=torch.uint8)
    fn = _compile_quant(x, row, sf, col, csf)
    _launch_quant(fn, (x, row, sf, col, csf))
    for axis, payload, scale in ((2, row, sf), (1, col, csf)):
        expected, expected_sf = mxfp8_reference(x, out_dtype, axis)
        # PTX NaN sign is unspecified; all finite codes and scale bytes are exact.
        decoded = expected.view(out_dtype).float()
        nan = torch.isnan(decoded)
        assert torch.equal(torch.isnan(payload.float()), nan)
        torch.testing.assert_close(payload.view(torch.uint8)[~nan], expected[~nan], rtol=0, atol=0)
        torch.testing.assert_close(scale, expected_sf, rtol=0, atol=0)


@pytest.mark.parametrize("width", [32, 64, 128])
def test_frost_norm_nvfp4_exact_boundaries(width):
    import cutlass
    from cutlass.cute.runtime import make_fake_tensor
    from cudnn.frost.compiled_cache import compile_cached
    from cudnn.norm.block_scale_kernels import nvfp4
    from cudnn.norm.utils import fake_stream

    ties = torch.tensor([0.0, -0.0, 0.25, -0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -6.0, 1.0, 2.0, 3.0, 4.0], device="cuda")
    # raw=1 pins ties. Tiny raw rounds the emitted E4M3 scale to zero;
    # large raw saturates that scale while the payload still divides by raw.
    x = torch.stack(
        [
            ties.repeat(width // 16),
            ties.repeat(width // 16) * 2**-20,
            ties.repeat(width // 16) * 1024,
            torch.zeros(width, device="cuda"),
            ties.repeat(width // 16) * 1.13,
        ]
    )[None]
    row = torch.full((1, 5, width // 2), 255, device="cuda", dtype=torch.uint8)
    sf = torch.full((1, 5, width // 16), 255, device="cuda", dtype=torch.uint8)
    fake = [
        make_fake_tensor(dt, tuple(t.shape), tuple(t.stride()), assumed_align=16) for t, dt in ((x, cutlass.Float32), (row, cutlass.Uint8), (sf, cutlass.Uint8))
    ]
    fn = compile_cached(nvfp4, *fake, None, None, False, fake_stream(), cache_key=f"norm_nvfp4_exact_boundaries:{width}", options="--enable-tvm-ffi")
    _launch_quant(fn, (x, row, sf, None, None))
    ry, rs = nvfp4_reference(x)
    torch.testing.assert_close(row, ry, rtol=0, atol=0)
    torch.testing.assert_close(sf, rs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case,expected",
    [("transpose_row", "only forward columnwise"), ("compute_half", "fp32 computation"), ("malformed_adaptive_scale", "adaptive scale")],
)
def test_frost_norm_block_scale_forward_declines(cudnn_handle, monkeypatch, case, expected):
    from cudnn.norm.fprop.engine import FrostNormFwdEngine
    from cudnn.norm.fprop.engines import ENGINE_SPECS
    import cutlass.cute as cute

    x = torch.ones((2, 32, 96), device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones((1, 1, 96), device="cuda")
    g, _, _, _, _ = _quantized_graph(cudnn_handle, "rms", x, gamma, None, "fprop", torch.float32, torch.float8_e4m3fn)
    engine = FrostNormFwdEngine(ENGINE_SPECS[0], 21000)
    assert engine._decline_reason(g, None) is None
    row = g.nodes[-2]
    if case == "transpose_row":
        row.params["transpose"] = True
    elif case == "compute_half":
        row.compute_data_type = cudnn.data_type.HALF
    else:
        g.nodes[0].params["adaptive"] = True
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("declined graph compiled"))
    assert expected in engine._decline_reason(g, None)


@pytest.mark.parametrize("invalid", ["dtype", "rank", "stride", "alignment", "device"])
def test_norm_workspace_rejects_invalid_cuda_storage(monkeypatch, invalid):
    from cudnn.norm.utils import Workspace

    buf = torch.empty(64, dtype=torch.uint8, device="cuda")
    assert Workspace(buf).take(4, torch.float32).data_ptr() == buf.data_ptr()
    if invalid == "dtype":
        buf = buf.view(torch.float32)
    elif invalid == "rank":
        buf = buf.reshape(8, 8)
    elif invalid == "stride":
        buf = buf[::2]
    elif invalid == "alignment":
        buf = buf[1:]
    else:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: buf.device.index + 1)
    with pytest.raises(ValueError, match="workspace must"):
        Workspace(buf)


@pytest.mark.parametrize("format", ["e4m3", "e5m2", "nvfp4"])
def test_norm_block_scale_default_fp32_intermediate(cudnn_handle, format):
    """The shared graph contract retains FP32 between norm and quantization."""
    torch.manual_seed(83)
    shape = (2, 32, 96, 1)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gamma = torch.zeros((1, 1, 96, 1), device="cuda")
    beta = torch.ones_like(gamma)
    block_size = 16 if format == "nvfp4" else 32
    maximum, midpoint = {"e4m3": (448.0, 1.0625), "e5m2": (57344.0, 1.125), "nvfp4": (6.0, 1.25)}[format]
    beta[..., ::block_size, 0] = maximum
    beta[..., 1::block_size, 0] = midpoint + 0.001
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=cudnn_handle)
    vp = {}

    def inp(value):
        tensor = g.tensor_like(value)
        vp[tensor] = value
        return tensor

    fp4 = format == "nvfp4"
    y, mean, rstd = g.layernorm(
        input=inp(x),
        scale=inp(gamma),
        bias=inp(beta),
        epsilon=inp(torch.full((1, 1, 1, 1), 1e-3)),
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE if fp4 else cudnn.norm_forward_phase.TRAINING,
    )
    # Do not override Y's dtype: infer it from the graph's FP32 intermediate type.
    for tensor in (mean, rstd):
        if tensor is not None:
            value = torch.empty((2, 32, 1, 1), device="cuda")
            tensor.set_output(True).set_dim(value.shape).set_stride(value.stride()).set_data_type(value.dtype)
            vp[tensor] = value
    outputs = []
    axes = (2,) if fp4 else (2, 1)
    dtype = torch.float8_e5m2 if format == "e5m2" else torch.float8_e4m3fn
    for axis in axes:
        q, s = g.block_scale_quantize(input=y, block_size=16 if fp4 else 32, axis=axis)
        payload = torch.empty((2, 32, 48, 1) if fp4 else shape, device="cuda", dtype=torch.uint8)
        dims = list(shape)
        dims[axis] //= 16 if fp4 else 32
        scale = torch.empty(dims, device="cuda", dtype=torch.uint8)
        q.set_output(True).set_dim(shape).set_stride(x.stride()).set_data_type(cudnn.data_type.FP4_E2M1 if fp4 else dtype)
        sf_dtype = torch.float8_e4m3fn if fp4 else torch.float8_e8m0fnu
        s.set_output(True).set_dim(dims).set_stride(scale.stride()).set_data_type(sf_dtype)
        vp[q] = payload if fp4 else payload.view(dtype)
        vp[s] = scale.view(sf_dtype)
        outputs.append((payload, scale))
    g.validate()
    assert y.get_is_virtual() and y.get_data_type() == cudnn.data_type.FLOAT
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    original_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        for backend in (False, True):
            candidates = [i for i, p in enumerate(g.plans) if (p.engine_id < 20_000) == backend]
            assert candidates, f"missing {'backend' if backend else 'FROST'} norm/block-scale plan"
            for plan_index in candidates:
                g.select_plan(plan_index)
                g.check_support()
                g.build_plans()
                workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
                for payload, scale in outputs:
                    payload.fill_(255)
                    scale.fill_(255)
                g.execute(vp, workspace, handle=cudnn_handle)
                # gamma=0 isolates conversion semantics from norm reduction order.
                for axis, (payload, scale) in zip(axes, outputs):
                    if fp4:
                        expected_y, expected_s = nvfp4_reference(beta.expand(shape).squeeze(-1))
                        expected_y, expected_s = expected_y.unsqueeze(-1), expected_s.unsqueeze(-1)
                    else:
                        expected_y, expected_s = mxfp8_reference(beta.expand(shape), dtype, axis)
                    if axis == 2:
                        rounded = beta.expand(shape).to(x.dtype)
                        if fp4:
                            rounded_y, _ = nvfp4_reference(rounded.squeeze(-1))
                            rounded_y = rounded_y.unsqueeze(-1)
                        else:
                            rounded_y, _ = mxfp8_reference(rounded, dtype, axis)
                        # A BF16 staging bug must fail, even with a finite exact norm result.
                        assert torch.any(expected_y != rounded_y).item()
                    torch.testing.assert_close(payload, expected_y, rtol=0, atol=0)
                    torch.testing.assert_close(scale, expected_s, rtol=0, atol=0)
    finally:
        cudnn.set_stream(cudnn_handle, original_stream)
