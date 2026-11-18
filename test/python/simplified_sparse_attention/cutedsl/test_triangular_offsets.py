# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check kernel address expressions without allocating a score workspace."""

import ast
import importlib.util
import textwrap
from pathlib import Path
from types import SimpleNamespace

import cudnn
import numpy as np
import pytest

from test_simplified_sparse_attention import supported_gpu

pytestmark = pytest.mark.L0
KERNELS = Path(cudnn.__file__).resolve().parent / "simplified_sparse_attention/cutedsl/kernels"
ROWS = [0, 127, 128, 65407, 65408, 65471, 65472, 65535, 65536, 131071, 131072, 262143]


def assignment_nodes(path, names):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    return sorted(nodes, key=lambda n: n.lineno)


def assignments(path, names):
    return compile(ast.Module(body=assignment_nodes(path, names), type_ignores=[]), str(path), "exec")


def decoder_nodes():
    tree = ast.parse((KERNELS / "indexer_forward_topk_tcgen05.py").read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and any(isinstance(s, ast.While) for s in n.orelse))
    return branch.orelse[1:]


@pytest.mark.parametrize("row", ROWS)
def test_triangular_producer_consumer_offsets(row):
    # NumPy scalars retain the kernel's signed widths, including intermediates.
    bi, rr = divmod(row, 128)
    expected = bi * (bi + 1) // 2 * 16384 + rr * (bi + 1) * 128
    env = dict(cutlass=SimpleNamespace(Int64=np.int64, Int32=np.int32), I32=np.int32, BLK=128)
    for bj in (0, bi):
        for c4 in (0, 31):
            state = dict(env, bi=np.int32(bi), bj=np.int32(bj), rr=np.int32(rr), c4=np.int32(c4))
            exec(assignments(KERNELS / "indexer_forward_topk_tcgen05.py", {"row_stride", "base_elems", "off"}), state)
            assert state["off"] == expected + bj * 128 + c4 * 4
    state = dict(env, bidx=np.int32(row))
    exec(assignments(KERNELS / "cute_dsl_filter_topk/filtered_top_k_varlen_util.py", {"_bi", "_rr", "_off"}), state)
    assert state["_off"] == expected


@pytest.mark.parametrize("blocks", [1, 512, 1024, 1025, 2048])
def test_triangular_tile_decoder(blocks):
    code = compile(ast.Module(body=decoder_nodes(), type_ignores=[]), "<tile decoder>", "exec")
    for bi in (0, blocks - 1):
        for bj in (0, bi):
            state = dict(
                I32=np.int32,
                cutlass=SimpleNamespace(Int64=np.int64),
                cute=SimpleNamespace(size=lambda *a, **kw: blocks * 128),
                mQ=None,
                BLK=128,
                blk=np.int32(bi * (bi + 1) // 2 + bj),
            )
            exec(code, state)
            assert (state["bi"], state["bj"]) == (bi, bj)


@pytest.fixture
def native_probe(tmp_path):
    import torch

    # Compile the production expressions, retaining DSL integer promotion.
    producer = assignment_nodes(KERNELS / "indexer_forward_topk_tcgen05.py", {"row_stride", "base_elems", "off"})
    consumer = assignment_nodes(KERNELS / "cute_dsl_filter_topk/filtered_top_k_varlen_util.py", {"_bi", "_rr", "_off"})
    body = "\n".join(ast.unparse(n) for n in producer + consumer)
    decoder = "\n".join(ast.unparse(n) for n in decoder_nodes())
    source = """import cutlass
import cutlass.cute as cute
I32 = cutlass.Int32
BLK = 128

@cute.kernel
def offsets(rows: cute.Tensor, output: cute.Tensor):
    idx, _, _ = cute.arch.thread_idx()
    if idx < cute.size(rows):
        bidx = rows[idx]
        bi = bidx // BLK
        rr = bidx % BLK
        bj = bi
        c4 = I32(31)
BODY
        output[idx, 0] = off
        output[idx, 1] = _off

@cute.kernel
def decode(tiles: cute.Tensor, output: cute.Tensor, mQ: cute.Tensor):
    idx, _, _ = cute.arch.thread_idx()
    if idx < cute.size(tiles):
        blk = tiles[idx]
DECODER
        output[idx, 0] = bi
        output[idx, 1] = bj

@cute.jit
def launch(rows, output, mQ, stream, decoder: cutlass.Constexpr):
    if cutlass.const_expr(decoder):
        decode(rows, output, mQ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    else:
        offsets(rows, output).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
"""
    source = source.replace("BODY", textwrap.indent(body, " " * 8)).replace("DECODER", textwrap.indent(decoder, " " * 8))
    path = tmp_path / "triangular_probe.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("triangular_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def run(values, blocks=1, decoder=False):
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        rows = torch.tensor(values, device="cuda", dtype=torch.int32)
        output = torch.full((len(values), 2), -1, device="cuda", dtype=torch.int64)
        shape = torch.empty(blocks * 128, device="cuda", dtype=torch.int8)
        args = tuple(from_dlpack(t) for t in (rows, output, shape))
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(mod.launch, *args, stream, decoder)
        compiled(*args, stream)
        return output.cpu().tolist()

    return run


@pytest.mark.usefixtures("supported_gpu")
def test_native_triangular_offsets(native_probe):
    expected = []
    for row in ROWS:
        bi, rr = divmod(row, 128)
        base = bi * (bi + 1) // 2 * 16384 + rr * (bi + 1) * 128
        expected.append([base + bi * 128 + 124, base])
    assert native_probe(ROWS) == expected


@pytest.mark.usefixtures("supported_gpu")
@pytest.mark.parametrize("blocks", [1, 512, 1024, 1025, 2048])
def test_native_triangular_decoder(native_probe, blocks):
    expected = [[bi, bj] for bi in (0, blocks - 1) for bj in (0, bi)]
    tiles = [bi * (bi + 1) // 2 + bj for bi, bj in expected]
    assert native_probe(tiles, blocks, decoder=True) == expected
