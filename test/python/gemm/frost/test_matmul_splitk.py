# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-kernel split-K: kernel 1 stores fp32 partials to the caller workspace,
a PDL-chained reducer sums them, applies the store cast, and writes D.

Small-integer inputs keep the fp32 reference exactly representable, so every
parity check below is bit-exact (the reducer's fixed-order sum is deterministic
by construction)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import cudnn
from gemm_test_utils import requires_sm100

pytestmark = [pytest.mark.L0]


def _mk_graph(B, M, N, K, io_dt=cudnn.data_type.BFLOAT16, out_dt=cudnn.data_type.BFLOAT16):
    import cudnn.gemm.frost  # noqa: F401  (installs the op-recording hook)

    g = cudnn.pygraph(io_data_type=io_dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    Bt = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(name="mm", A=A, B=Bt)
    C.set_output(True).set_data_type(out_dt)
    return g, A, Bt, C


def _mk_fused_graph(K):
    import cudnn.gemm.frost  # noqa: F401

    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, 256, K], stride=[256 * K, K, 1])
    Bt = g.tensor(name="B", dim=[1, K, 256], stride=[K * 256, 1, K])
    Bias = g.tensor(name="bias", dim=[1, 1, 256], stride=[256, 256, 1])
    out = g.add(name="add", a=g.matmul(name="mm", A=A, B=Bt), b=Bias)
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g


def _jit(g, S, base_cfg=None):
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import DEFAULT_CONFIG

    return jit_from_cudnn_graph(g, replace(base_cfg or DEFAULT_CONFIG, split_k_slices=S))


def _run(compiled, vp):
    from cudnn.frost.workspace import Workspace

    ws = None
    if compiled.workspace_bytes:
        buf = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")
        ws = Workspace(buf, compiled.workspace_bytes, "test_splitk")
    compiled(vp, workspace=ws)
    torch.cuda.synchronize()


def _data(B, M, N, K, torch_in=torch.bfloat16, torch_out=torch.bfloat16):
    torch.manual_seed(0)
    a = torch.empty(B, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch_in, device="cuda")
    b = torch.empty(B, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch_in, device="cuda")
    c = torch.zeros(B, M, N, dtype=torch_out, device="cuda")
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch_out)
    return a, b, c, ref


# Shapes cover what split-K changes: K tails, M tails, batch, deep K.
_SHAPES = ((1, 256, 256, 4096), (1, 256, 256, 4160), (1, 300, 256, 8192), (3, 256, 512, 4096), (1, 128, 128, 16384))


@requires_sm100
@pytest.mark.parametrize("S", (2, 3, 8))
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_splitk_parity(shape, S):
    # shape x S interact (which slice owns the K tail depends on S); the store
    # dtype does not, so it is swept separately in test_splitk_output_dtypes.
    B, M, N, K = shape
    g, A, Bt, C = _mk_graph(B, M, N, K)
    compiled = _jit(g, S)
    a, b, c, ref = _data(B, M, N, K)
    _run(compiled, {A: a, Bt: b, C: c})
    assert torch.equal(c, ref), f"{(c != ref).sum().item()}/{c.numel()} elements differ"


@requires_sm100
@pytest.mark.parametrize(
    ("io_dt", "out_dt", "torch_in", "torch_out"),
    (
        (cudnn.data_type.HALF, cudnn.data_type.HALF, torch.float16, torch.float16),
        (cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT, torch.bfloat16, torch.float32),
    ),
    ids=("fp16", "bf16_to_fp32"),
)
def test_splitk_output_dtypes(io_dt, out_dt, torch_in, torch_out):
    # The reducer's store cast is the only dtype-sensitive step; one K-tail shape suffices.
    g, A, Bt, C = _mk_graph(1, 256, 256, 4160, io_dt, out_dt)
    compiled = _jit(g, 3)
    a, b, c, ref = _data(1, 256, 256, 4160, torch_in, torch_out)
    _run(compiled, {A: a, Bt: b, C: c})
    assert torch.equal(c, ref)


@requires_sm100
@pytest.mark.parametrize("N", (250, 255), ids=("Nmod4", "Nodd"))
def test_splitk_n_not_multiple_of_4(N):
    # splitk_reduce_elems clamps to divide N (2 for 250, 1 for 255), so reducer
    # groups never cross a workspace row. fp32 output: odd N with a 2-byte dtype
    # is rejected engine-wide (row stride must be 4-byte aligned).
    g, A, Bt, C = _mk_graph(1, 256, N, 4096, out_dt=cudnn.data_type.FLOAT)
    compiled = _jit(g, 8)
    a, b, c, ref = _data(1, 256, N, 4096, torch_out=torch.float32)
    _run(compiled, {A: a, Bt: b, C: c})
    assert torch.equal(c, ref)


@requires_sm100
def test_splitk_cta_group1():
    from cudnn.gemm.frost.tile_config import by_name

    g, A, Bt, C = _mk_graph(1, 256, 256, 8192)
    compiled = _jit(g, 4, base_cfg=by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))
    assert compiled.store_modes == ("tma",)
    assert compiled.use_tma_store
    a, b, c, ref = _data(1, 256, 256, 8192)
    _run(compiled, {A: a, Bt: b, C: c})
    assert torch.equal(c, ref)


@requires_sm100
def test_splitk_deterministic():
    g, A, Bt, C = _mk_graph(1, 256, 256, 8192)
    compiled = _jit(g, 8)
    a, b, c1, _ = _data(1, 256, 256, 8192)
    c2 = torch.zeros_like(c1)
    _run(compiled, {A: a, Bt: b, C: c1})
    _run(compiled, {A: a, Bt: b, C: c2})
    assert torch.equal(c1, c2)


@requires_sm100
def test_splitk_workspace_size_and_missing_workspace():
    B, M, N, K, S = 2, 256, 384, 4096, 4
    g, A, Bt, C = _mk_graph(B, M, N, K)
    compiled = _jit(g, S)
    assert compiled.workspace_bytes == -(-S * B * M * N * 4 // 128) * 128
    a, b, c, _ = _data(B, M, N, K)
    with pytest.raises(ValueError, match="needs a workspace"):
        compiled({A: a, Bt: b, C: c})


@requires_sm100
def test_splitk_more_slices_than_k_tiles():
    # DEFAULT_CONFIG cta_tile_k = 64 bf16 elems -> K=128 has 2 tiles < S=4.
    from cudnn.frost.workspace import Workspace

    g, A, Bt, C = _mk_graph(1, 256, 256, 128)
    compiled = _jit(g, 4)
    a, b, c, _ = _data(1, 256, 256, 128)
    buf = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="exceeds the .* K tile"):
        compiled({A: a, Bt: b, C: c}, workspace=Workspace(buf, compiled.workspace_bytes, "t"))


@requires_sm100
def test_splitk_rejects_more_than_32_slices():
    # The reducer unrolls its accumulation at trace time; the gate bounds S.
    g, A, Bt, C = _mk_graph(1, 256, 256, 65536)
    with pytest.raises(NotImplementedError, match="more than 32 slices"):
        _jit(g, 64)


@requires_sm100
def test_splitk_rejects_epilogue_fusion():
    with pytest.raises(NotImplementedError, match="split_k_slices.*plain matmul"):
        _jit(_mk_fused_graph(K=512), 2)


def test_splitk_auto_select():
    """Pure-function checks of the auto-split heuristic (fixed sm_count so no
    GPU is needed)."""
    from cudnn.gemm.frost.compiler import _auto_split_k
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.tile_config import DEFAULT_CONFIG

    def slices(g, sm=148):
        chain, _ = analyze_with_binding(g)
        return _auto_split_k(chain, DEFAULT_CONFIG, sm_count=sm).split_k_slices

    # Latency-bound + deep K: one full wave (148 // 2 = 74) hits the S=32 cap
    # (the reducer unrolls chained loads only up to S=32).
    assert slices(_mk_graph(1, 256, 256, 16384)[0]) == 32
    # Compute-bound: grid already fills the GPU.
    assert slices(_mk_graph(1, 4096, 4096, 8192)[0]) == 1
    # Shallow K: per-slice floor (2 CTA-K tiles of 64 bf16 elems) wins.
    assert slices(_mk_graph(1, 256, 256, 1024)[0]) == 8
    # K below the 2048-byte enable threshold.
    assert slices(_mk_graph(1, 256, 256, 512)[0]) == 1
    # grid.z cap: batch * S <= 65535.
    assert 60000 * slices(_mk_graph(60000, 128, 128, 65536)[0]) <= 65535
    # Fused graph never auto-splits (gate reused).
    assert slices(_mk_fused_graph(K=8192)) == 1


def test_splitk_config_name_round_trip():
    from cudnn.gemm.frost.tile_config import by_name

    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_splitK4")
    assert cfg.split_k_slices == 4
    assert by_name(cfg.name) == cfg
    assert by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma").split_k_slices == 1
