# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 col (M-axis) block_scale_quantize: every F8_128x4 scale byte lands at its layout offset.

The col-scale index is ``((n // 128) * mcb + mb // 4) * 512 + ...``. ``row``, ``col_j`` and the lane are
Int32, so with an Int32 ``mcb`` (a build-time literal on the dense path, the group extent on the grouped
MoE path) that atom term was evaluated in 32 bits: once one batch's padded scale blob passed 2 GiB it
wrapped, and every scale byte past 2^31 was stored 2^32 bytes below its slot. The ``past2GiB`` cells take
the smallest such shape: N = 16 (F8_128x4 pads N to one 128-column atom) and 2^28 + 64 rows at block 16,
which puts the last four row blocks past 2^31. Their scale tensor starts 2 GiB into a sentinel-filled
carrier, so a wrapped store lands on the carrier's head instead of on memory the test does not own.
"""

from __future__ import annotations

import pytest
import torch

from gemm_test_utils import requires_sm120
from test_matmul import _col_quant_reference

import cudnn
from cudnn.gemm.frost.sm120 import compiler as C120
from cudnn.gemm.frost.tile_config import by_name

pytestmark = [requires_sm120]

_CFG = "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"
_ROWS_PAST_2GIB = 2**28 + 64
_PAST_2GIB = [pytest.mark.L1, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]
_SENTINEL = 0xFF  # E8M0 NaN: never a computed scale


def _graph(kind, offsets, M, N, K, bs):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    if kind == "dense":
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
        c, kwargs = g.matmul(A=A, B=B, name="mm"), {}
    else:
        E = len(offsets) - 1
        tok = g.tensor(name="token", dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
        w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
        fto = g.tensor(name="fto", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        fto.set_alignment_value(4 * bs)
        c = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe")
        kwargs = {"group_offset": fto}
    q, qs = g.block_scale_quantize(input=c, block_size=bs, axis=1, name="q", **kwargs)
    q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    rows, cols = (N + 127) // 128 * 128, (M // bs + 3) // 4 * 4
    qs.set_dim([1, rows, cols]).set_stride([rows * cols, cols, 1])
    qs.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    qs.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    return g


@pytest.mark.parametrize(
    "kind,bs,M,N,K",
    [
        pytest.param("dense", 32, 320, 256, 64, marks=pytest.mark.L0, id="dense"),
        pytest.param("moe_grouped", 32, 384, 256, 64, marks=pytest.mark.L0, id="moe_grouped"),
        pytest.param("dense", 16, _ROWS_PAST_2GIB, 16, 8, marks=_PAST_2GIB, id="dense-past2GiB"),
        pytest.param("moe_grouped", 16, _ROWS_PAST_2GIB, 16, 8, marks=_PAST_2GIB, id="moe_grouped-past2GiB"),
    ],
)
def test_col_quant_f8_128x4_scale_offsets_sm120(kind, bs, M, N, K):
    """Dense and grouped-MoE col quant store each scale byte at its F8_128x4 offset, also past 2 GiB."""
    large = M == _ROWS_PAST_2GIB
    # The grouped layout is one table per group at its 4-block base: two groups on the small shape.
    offsets = (0, M) if kind == "dense" or large else (0, 4 * bs, M)
    rows, cols = (N + 127) // 128 * 128, (M // bs + 3) // 4 * 4
    guard = 2**31 if large else 0
    if large:
        need = 2 * M * K + M * N + guard + rows * cols  # bf16 A, fp8 data, scale carrier
        torch.cuda.empty_cache()
        if need + 2**30 > torch.cuda.mem_get_info()[0]:
            pytest.skip(f"the past-2GiB probe needs {need / 2**30:.1f} GiB of free device memory")
    compiled = C120.jit_from_cudnn_graph(_graph(kind, offsets, M, N, K, bs), config=by_name(_CFG))
    assert compiled.chain.quants[0].grouped_by_moe == (kind == "moe_grouped")

    torch.manual_seed(0)
    w = torch.empty(len(offsets) - 1, N, K, dtype=torch.bfloat16, device="cuda").random_(-2, 3)
    try:
        a = torch.empty(1, M, K, dtype=torch.bfloat16, device="cuda").random_(-2, 3)
        q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
        carrier = torch.full((guard + rows * cols,), _SENTINEL, dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("another allocation consumed the memory of the past-2GiB probe")
    scales = carrier[guard:].view(torch.float8_e8m0fnu).view(1, rows, cols)
    bd = compiled.binding
    vp = {bd.a_operands[0]: a, bd.b_operands[0]: w.transpose(1, 2) if kind == "dense" else w, bd.outputs[0]: q, bd.outputs[1]: scales}
    if kind != "dense":
        vp[bd.first_token_offset] = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    compiled(vp)
    torch.cuda.synchronize()

    def check(group, r0, r1):
        begin, end = offsets[group], offsets[group + 1]
        q_ref, s_ref = _col_quant_reference((a[:, r0:r1].float() @ w[group].float().T), bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
        mcb = ((end - begin) // bs + 3) // 4
        n = torch.arange(N, device="cuda").unsqueeze(1)
        mb = torch.arange((r0 - begin) // bs, (r1 - begin) // bs, device="cuda").unsqueeze(0)
        offs = (begin // (4 * bs)) * (rows // 128) * 512 + ((n // 128) * mcb + mb // 4) * 512 + (n % 32) * 16 + ((n % 128) // 32) * 4 + mb % 4
        where = f"rows [{r0}, {r1}), scale offsets up to {int(offs.max())}"
        torch.testing.assert_close(scales.view(-1)[offs].float(), s_ref[0].T.float(), atol=0, rtol=0, msg=lambda m: f"{where}: {m}")
        torch.testing.assert_close(q[0, r0:r1].float(), q_ref[0].float(), atol=0, rtol=0, msg=lambda m: f"{where}: {m}")

    if large:
        # The first atom, the last one below 2^31 and the one past it.
        for r0 in (0, M - 8 * bs, M - 4 * bs):
            check(0, r0, r0 + 4 * bs)
        assert int(carrier[:guard].min()) == _SENTINEL, "a scale byte was stored below the scale tensor: its offset wrapped at 2^31"
    else:
        for group in range(len(offsets) - 1):
            check(group, offsets[group], offsets[group + 1])
