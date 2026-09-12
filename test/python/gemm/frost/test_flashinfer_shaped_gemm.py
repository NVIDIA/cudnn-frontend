# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer-shaped GEMM graphs under the frost opt-in: accept means run.

FlashInfer (flashinfer/gemm/gemm_base.py) builds every cuDNN GEMM graph the
same way: it derives rank-3 dims/strides from its torch operands by hand
(``[batch, m, k]`` for A, ``[batch, k, n]`` with a unit stride on k for B,
``[batch, bs_m, bs_k]`` F8_128x4-reordered block scales), declares the graph
with those, and then binds the ORIGINAL buffers -- 2-D matrices, 1-D scale
blobs, 0-d scalars -- by uid. The cuDNN backend reads only the pointer, so it
never notices; a python engine that reads the buffer's own shape does.

Each case here re-declares one FlashInfer graph exactly (dims, strides,
dtypes, reordering, uids) and binds the buffers FlashInfer binds. The
contract under test, for the frost plan:

    check_support() accepted  =>  build_plans() + execute() succeed and the
                                  result matches the cuDNN backend plan

A decline at check_support is a routing decision and is reported as xfail
with the reason; a refusal AFTER acceptance is the defect this file exists to
catch. Cases known to fail today carry ``xfail(strict=True)`` with the owner
finding, so a fix flips them loud.
"""

from __future__ import annotations

import pytest
import torch

import cudnn
from cudnn.engines import is_python_engine
from gemm_test_utils import requires_sm100

pytestmark = [pytest.mark.L0, requires_sm100]

# Known gaps, strict so a fix flips them loud. Both are one defect in
# gemm/frost/recipe.py: frost_gemm reads the BUFFER's own shape (rank, units)
# where the cuDNN contract is the graph declaration plus override_shapes.
_XFAIL_BUFFER_RANK = pytest.mark.xfail(
    strict=True, reason="frost_gemm: refuses after check_support -- reads the buffer's rank instead of the declared [batch, ...] dims"
)
_XFAIL_FP4_K_UNITS = pytest.mark.xfail(
    strict=True, reason="frost_gemm: override_shapes are fp4 ELEMENT dims but kpack=2 is applied again -- K doubles, SF blob 'too small'"
)

# FlashInfer's uid enum (gemm_base.UIDs), kept identical so a graph diff is a graph diff.
A_UID, B_UID, ALPHA_UID, SFA_UID, SFB_UID, A_SCALE_UID, B_SCALE_UID, BIAS_UID, O_UID = range(9)
FROST = "frost_gemm"


# ---------------------------------------------------------------------------
# FlashInfer's shape math, verbatim (gemm_base._calculate_block_scale_dims,
# _get_real_fp4_shape_from_packed_uint8, _get_bf16_3d_shape_stride).
# ---------------------------------------------------------------------------


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _block_scale_dims(m: int, n: int, k: int, block_size: int) -> tuple[int, int, int]:
    return _div_up(m, 128) * 128, _div_up(n, 128) * 128, _div_up(_div_up(k, block_size), 4) * 4


def _shape3(t: torch.Tensor) -> tuple[list[int], list[int]]:
    """FlashInfer's 2-D -> ``[1, ...]`` promotion of a matrix's shape/stride."""
    shape, stride = list(t.shape), list(t.stride())
    if len(shape) == 2:
        shape.insert(0, 1)
        stride.insert(0, t.numel())
    return shape, stride


def _fp4_shape3(packed: torch.Tensor) -> tuple[list[int], list[int]]:
    """Real fp4-element dims of a packed x2 buffer (two elements per slot)."""
    shape, stride = _shape3(packed)
    column_major = packed.stride(-2) == 1
    shape[-2 if column_major else -1] *= 2
    if column_major:
        stride[-1] *= 2
        for i in range(len(stride) - 2):
            stride[i] *= 2
    else:
        for i in range(len(stride) - 1):
            stride[i] *= 2
    return shape, stride


# ---------------------------------------------------------------------------
# Plan walk: the frost entry and the first backend entry of one graph.
# ---------------------------------------------------------------------------


def _plan_indices(g) -> tuple[int | None, int | None]:
    frost = backend = None
    for i in range(g.get_execution_plan_count()):
        name = g.get_plan_name_at_index(i)
        engine_id, _ = g.get_engine_and_knobs_at_index(i)
        if name.split("[")[0] == FROST and frost is None:
            frost = i
        elif backend is None and not is_python_engine(engine_id) and name != "backend_heuristics":
            backend = i
    return frost, backend


def _run(build, pack, *, use_frost: bool, override=None):
    """Build one graph, pin the frost (or first backend) plan, run it.

    Returns ``("ok", out)`` or ``("declined", "check_support", reason)``; any
    exception from build_plans or execute after a successful check_support
    propagates -- that is the contract violation."""
    handle = cudnn.create_handle()
    g, out_t = build(handle)
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost, backend = _plan_indices(g)
    idx = frost if use_frost else backend
    assert idx is not None, f"no {'frost' if use_frost else 'backend'} plan; plans={[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    g.select_plan(idx)
    try:
        g.check_support()
    except (NotImplementedError, cudnn.cudnnGraphNotSupportedError) as exc:
        return ("declined", "check_support", str(exc))
    g.build_plans()  # accepted above: a failure from here on is the finding, not a decline
    out = out_t()
    kwargs = {}
    if override is not None:
        uids, shapes, strides = override
        kwargs = dict(override_uids=uids, override_shapes=shapes, override_strides=strides)
        ws_bytes = g.get_workspace_size(handle, uids, shapes, strides)
    else:
        ws_bytes = g.get_workspace_size()
    ws = torch.empty(max(int(ws_bytes), 1), dtype=torch.uint8, device="cuda")
    g.execute({**pack, O_UID: out}, ws, handle=handle, **kwargs)
    torch.cuda.synchronize()
    return ("ok", out)


def _accept_means_run(build, pack, *, override=None, atol=1e-1, rtol=2e-2):
    ref = _run(build, pack, use_frost=False, override=override)
    assert ref[0] == "ok", f"the cuDNN backend itself declined this FlashInfer graph: {ref}"
    got = _run(build, pack, use_frost=True, override=override)  # an exception here is the finding
    if got[0] == "declined":
        pytest.xfail(f"frost_gemm declined at {got[1]}: {got[2][:200]}")
    torch.testing.assert_close(got[1].float(), ref[1].float(), atol=atol, rtol=rtol)


def _graph(handle, **kw):
    return cudnn.pygraph(
        handle=handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        **kw,
    )


# ---------------------------------------------------------------------------
# mm_bf16: A (M, K) row-major, B (K, N) column-major (a transposed view), 2-D
# buffers bound to a [1, ...] graph.
# ---------------------------------------------------------------------------


def _bf16_case(M: int, N: int, K: int, *, override_cache_m: int | None = None):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).t()  # (K, N), stride (1, K)
    a_shape, a_stride = _shape3(a)
    b_shape, b_stride = _shape3(b)
    if override_cache_m is not None:
        # build at cache_m, run at M (gemm_base.build_cudnn_gemm_bf16_graph_override_shape)
        a_decl = ([1, override_cache_m, K], [override_cache_m * K, K, 1])
        o_decl = ([1, override_cache_m, N], [override_cache_m * N, N, 1])
    else:
        a_decl, o_decl = (a_shape, a_stride), ([1, M, N], [M * N, N, 1])

    def build(handle):
        g = _graph(handle, is_override_shape_enabled=override_cache_m is not None)
        ta = g.tensor(name="a", dim=a_decl[0], stride=a_decl[1], data_type=cudnn.data_type.BFLOAT16)
        tb = g.tensor(name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.BFLOAT16)
        c = g.matmul(name="matmul", A=ta, B=tb, compute_data_type=cudnn.data_type.FLOAT)
        c.set_data_type(cudnn.data_type.FLOAT)
        c.set_name("c_final").set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        c.set_dim(o_decl[0]).set_stride(o_decl[1])
        ta.set_uid(A_UID)
        tb.set_uid(B_UID)
        c.set_uid(O_UID)
        g.validate()
        g.build_operation_graph()
        return g, lambda: torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    override = None
    if override_cache_m is not None:
        override = ([A_UID, B_UID, O_UID], [a_shape, b_shape, [1, M, N]], [a_stride, b_stride, [M * N, N, 1]])
    return build, {A_UID: a, B_UID: b}, override


@_XFAIL_BUFFER_RANK
@pytest.mark.parametrize("mnk", [(256, 512, 256), (13, 256, 128)])
def test_mm_bf16_two_d_buffers(mnk):
    build, pack, _ = _bf16_case(*mnk)
    _accept_means_run(build, pack)


def test_mm_bf16_override_shape_path():
    build, pack, override = _bf16_case(200, 512, 256, override_cache_m=256)
    _accept_means_run(build, pack, override=override)


# ---------------------------------------------------------------------------
# mm_fp4: packed x2 operands, F8_128x4 block scales bound as flat blobs.
# ---------------------------------------------------------------------------

_FP4_X2 = getattr(torch, "float4_e2m1fn_x2", None)


def _fp4_case(M: int, N: int, K: int, *, nvfp4: bool, alpha: bool, override_cache_m: int | None = None):
    torch.manual_seed(0)
    block = 16 if nvfp4 else 32
    a_u8 = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    b_u8 = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8).t()  # (K/2, N) column-major
    a, b = a_u8.view(_FP4_X2), b_u8.view(_FP4_X2)
    a_shape, a_stride = _fp4_shape3(a)  # [1, M, K]
    b_shape, b_stride = _fp4_shape3(b)  # [1, K, N], stride [K*N, 1, K]
    bs_m, bs_n, bs_k = _block_scale_dims(M, N, K, block)
    # FlashInfer binds the quantizer's flat F8_128x4 blob: bs_m * bs_k scale bytes.
    if nvfp4:
        sfa = torch.full((bs_m * bs_k,), 1.0, device="cuda").to(torch.float8_e4m3fn)
        sfb = torch.full((bs_n * bs_k,), 1.0, device="cuda").to(torch.float8_e4m3fn)
        sf_type = cudnn.data_type.FP8_E4M3
    else:
        sfa = torch.full((bs_m * bs_k,), 127, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
        sfb = torch.full((bs_n * bs_k,), 127, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
        sf_type = cudnn.data_type.FP8_E8M0
    decl_m = override_cache_m if override_cache_m is not None else M
    d_bs_m, _, d_bs_k = _block_scale_dims(decl_m, N, K, block)
    sfa_decl = ([1, d_bs_m, d_bs_k], [d_bs_m * d_bs_k, d_bs_k, 1])
    sfb_decl = ([1, bs_k, bs_n], [bs_n * bs_k, 1, bs_k])
    a_decl = ([1, decl_m, K], [decl_m * K, K, 1])
    o_decl = ([1, decl_m, N], [decl_m * N, N, 1])
    alpha_t = torch.tensor([0.5], device="cuda", dtype=torch.float32) if alpha else None

    def build(handle):
        g = _graph(handle, is_override_shape_enabled=override_cache_m is not None)
        ta = g.tensor(name="a", dim=a_decl[0], stride=a_decl[1], data_type=cudnn.data_type.FP4_E2M1)
        tb = g.tensor(name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.FP4_E2M1)
        tsa = g.tensor(name="block_descale_a", dim=sfa_decl[0], stride=sfa_decl[1], data_type=sf_type, reordering_type=cudnn.tensor_reordering.F8_128x4)
        tsb = g.tensor(name="block_descale_b", dim=sfb_decl[0], stride=sfb_decl[1], data_type=sf_type, reordering_type=cudnn.tensor_reordering.F8_128x4)
        da = g.block_scale_dequantize(ta, tsa, block_size=[1, block], name="dequant_a")
        da.set_data_type(cudnn.data_type.FLOAT)
        db = g.block_scale_dequantize(tb, tsb, block_size=[block, 1], name="dequant_b")
        db.set_data_type(cudnn.data_type.FLOAT)
        c = g.matmul(da, db, compute_data_type=cudnn.data_type.FLOAT, name="gemm")
        c.set_data_type(cudnn.data_type.FLOAT)
        final = c
        if alpha:
            tg = g.tensor(name="global_scale", dim=(1, 1, 1), stride=(1, 1, 1), data_type=cudnn.data_type.FLOAT)
            final = g.mul(name="scale_mul", a=c, b=tg, compute_data_type=cudnn.data_type.FLOAT)
            tg.set_uid(ALPHA_UID)
        final.set_name("c_final").set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        final.set_dim(o_decl[0]).set_stride(o_decl[1])
        ta.set_uid(A_UID)
        tb.set_uid(B_UID)
        tsa.set_uid(SFA_UID)
        tsb.set_uid(SFB_UID)
        final.set_uid(O_UID)
        g.validate()
        g.build_operation_graph()
        return g, lambda: torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    pack = {A_UID: a, B_UID: b, SFA_UID: sfa, SFB_UID: sfb}
    if alpha:
        pack[ALPHA_UID] = alpha_t.view(torch.float)
    override = None
    if override_cache_m is not None:
        # gemm_base.execute_cudnn_gemm_fp4_graph_override_shape: real fp4 dims for A/B,
        # the scale dims recomputed for the actual M, the output's 3-D dims.
        override = (
            [A_UID, B_UID, SFA_UID, SFB_UID, O_UID],
            [a_shape, b_shape, [1, bs_m, bs_k], [1, bs_k, bs_n], [1, M, N]],
            [a_stride, b_stride, [bs_m * bs_k, bs_k, 1], [bs_n * bs_k, 1, bs_k], [M * N, N, 1]],
        )
    return build, pack, override


@pytest.mark.skipif(_FP4_X2 is None, reason="torch has no float4_e2m1fn_x2")
@pytest.mark.parametrize("mnk", [(256, 512, 256), (13, 256, 128)], ids=["m256", "m13"])
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4", "mxfp4_alpha"])
@_XFAIL_BUFFER_RANK
def test_mm_fp4_flat_scale_blobs(mnk, kind):
    build, pack, _ = _fp4_case(*mnk, nvfp4=kind == "nvfp4", alpha=kind.endswith("alpha"))
    _accept_means_run(build, pack)


@pytest.mark.skipif(_FP4_X2 is None, reason="torch has no float4_e2m1fn_x2")
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4"])
@_XFAIL_FP4_K_UNITS
def test_mm_fp4_override_shape_path(kind):
    build, pack, override = _fp4_case(200, 512, 256, nvfp4=kind == "nvfp4", alpha=False, override_cache_m=256)
    _accept_means_run(build, pack, override=override)


# ---------------------------------------------------------------------------
# bmm_fp8: per-tensor scales declared (1, 1, 1), bound as 0-d tensors.
# ---------------------------------------------------------------------------


def _fp8_case(B: int, M: int, N: int, K: int, *, scalar_rank: int):
    torch.manual_seed(0)
    a = (torch.randn(B, M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(B, N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn).transpose(-2, -1)  # (B, K, N)
    a_shape, a_stride = list(a.shape), list(a.stride())
    b_shape, b_stride = list(b.shape), list(b.stride())
    a_scale = torch.tensor(0.5, device="cuda", dtype=torch.float32).reshape([1] * scalar_rank)
    b_scale = torch.tensor(2.0, device="cuda", dtype=torch.float32).reshape([1] * scalar_rank)

    def build(handle):
        g = _graph(handle)
        ta = g.tensor(name="a", dim=a_shape, stride=a_stride, data_type=cudnn.data_type.FP8_E4M3)
        tb = g.tensor(name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.FP8_E4M3)
        tsa = g.tensor(name="a_scale", dim=(1, 1, 1), stride=(1, 1, 1), data_type=cudnn.data_type.FLOAT)
        tsb = g.tensor(name="b_scale", dim=(1, 1, 1), stride=(1, 1, 1), data_type=cudnn.data_type.FLOAT)
        c = g.matmul(name="matmul", A=ta, B=tb, compute_data_type=cudnn.data_type.FLOAT)
        c.set_name("c").set_data_type(cudnn.data_type.FLOAT)
        c1 = g.mul(name="scale_mul_a", a=c, b=tsa, compute_data_type=cudnn.data_type.FLOAT)
        c2 = g.mul(name="scale_mul_b", a=c1, b=tsb, compute_data_type=cudnn.data_type.FLOAT)
        c2.set_name("c_final").set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        c2.set_dim([B, M, N]).set_stride([M * N, N, 1])
        ta.set_uid(A_UID)
        tb.set_uid(B_UID)
        tsa.set_uid(A_SCALE_UID)
        tsb.set_uid(B_SCALE_UID)
        c2.set_uid(O_UID)
        g.validate()
        g.build_operation_graph()
        return g, lambda: torch.empty(B, M, N, device="cuda", dtype=torch.bfloat16)

    return build, {A_UID: a, B_UID: b, A_SCALE_UID: a_scale, B_SCALE_UID: b_scale}


@pytest.mark.parametrize("scalar_rank", [0, 1, 3], ids=["scalar_0d", "scalar_1d", "scalar_3d"])
def test_bmm_fp8_per_tensor_scales(scalar_rank):
    build, pack = _fp8_case(2, 256, 512, 256, scalar_rank=scalar_rank)
    _accept_means_run(build, pack)


# ---------------------------------------------------------------------------
# bmm_mxfp8: E8M0 block scales, F8_128x4 reordered, bound as one flat blob per operand.
# ---------------------------------------------------------------------------


def _mxfp8_case(B: int, M: int, N: int, K: int):
    torch.manual_seed(0)
    block = 32
    a = (torch.randn(B, M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(B, N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn).transpose(-2, -1)  # (B, K, N)
    bs_m, bs_n, bs_k = _block_scale_dims(M, N, K, block)
    # mxfp8_quantize(is_sf_swizzled_layout=True) hands back ONE flat blob per operand.
    sfa = torch.full((B * bs_m * bs_k,), 127, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sfb = torch.full((B * bs_n * bs_k,), 127, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)

    def build(handle):
        g = _graph(handle)
        ta = g.tensor(name="a", dim=list(a.shape), stride=list(a.stride()), data_type=cudnn.data_type.FP8_E4M3)
        tb = g.tensor(name="b", dim=list(b.shape), stride=list(b.stride()), data_type=cudnn.data_type.FP8_E4M3)
        tsa = g.tensor(
            name="block_descale_a",
            dim=[B, bs_m, bs_k],
            stride=[bs_m * bs_k, bs_k, 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        tsb = g.tensor(
            name="block_descale_b",
            dim=[B, bs_k, bs_n],
            stride=[bs_n * bs_k, 1, bs_k],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        da = g.block_scale_dequantize(ta, tsa, block_size=[1, block], name="dequant_a")
        da.set_data_type(cudnn.data_type.FLOAT)
        db = g.block_scale_dequantize(tb, tsb, block_size=[block, 1], name="dequant_b")
        db.set_data_type(cudnn.data_type.FLOAT)
        c = g.matmul(da, db, compute_data_type=cudnn.data_type.FLOAT, name="gemm")
        c.set_data_type(cudnn.data_type.FLOAT)
        c.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        c.set_dim([B, M, N]).set_stride([M * N, N, 1])
        ta.set_uid(A_UID)
        tb.set_uid(B_UID)
        tsa.set_uid(SFA_UID)
        tsb.set_uid(SFB_UID)
        c.set_uid(O_UID)
        g.validate()
        g.build_operation_graph()
        return g, lambda: torch.empty(B, M, N, device="cuda", dtype=torch.bfloat16)

    return build, {A_UID: a, B_UID: b, SFA_UID: sfa, SFB_UID: sfb}


@pytest.mark.parametrize("bmnk", [(1, 256, 512, 256), (16, 128, 256, 1024)], ids=["b1", "b16"])
@_XFAIL_BUFFER_RANK
def test_bmm_mxfp8_flat_scale_blobs(bmnk):
    build, pack = _mxfp8_case(*bmnk)
    _accept_means_run(build, pack)
