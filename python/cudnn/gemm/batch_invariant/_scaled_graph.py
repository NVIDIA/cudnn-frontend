# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match scaled GEMM graphs whose reduction and rounding are implemented by the batch-invariant engine."""

from dataclasses import dataclass

from cudnn import data_type as dt, tensor_reordering
from cudnn.datatypes import _library_type
from cudnn.graph_types import NodeType, Tensor, storage_geometry

_FP8 = (dt.FP8_E4M3, dt.FP8_E5M2)


@dataclass(frozen=True)
class ScaledMatmul:
    a: Tensor
    b: Tensor
    out: Tensor
    sa: Tensor
    sb: Tensor
    alpha: Tensor | None
    kind: str
    in_fmt: str
    out_fmt: str
    batch: int
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    b_major: str
    a_batch_stride: int
    b_batch_stride: int
    out_batch_stride: int
    sa_strides: tuple
    sb_strides: tuple
    enable_pdl: bool


def _require(condition, message):
    if not condition:
        raise NotImplementedError(f"batch_invariant_gemm requires {message}")


def _tensor(t, *, virtual=False, reordered=False):
    _require(t.is_virtual == virtual and not t.is_pass_by_value and t.ragged_offset is None, "ordinary dense device tensors")
    order = t.get_reordering_type()
    _require(order == tensor_reordering.F8_128x4 if reordered else order in (None, tensor_reordering.NONE), "the scale layout specified by the operation")
    _require(
        bool(t.dim) and len(t.dim) == len(t.stride) and all(d > 0 for d in t.dim) and all(s >= 0 for s in t.stride),
        "positive static extents and nonnegative strides",
    )


def _fp32(node):
    _require(node.compute_data_type is not None and _library_type(node.compute_data_type) == dt.FLOAT, "FP32 computation")


def _scalar(t, virtual=False):
    _tensor(t, virtual=virtual)
    _require(t.get_data_type() == dt.FLOAT and all(d == 1 for d in t.dim), "one FP32 device scalar")


def _virtual_float(t, shape):
    _tensor(t, virtual=True)
    _require(t.get_data_type() == dt.FLOAT and tuple(t.dim) == tuple(shape), "a virtual FP32 intermediate of the declared shape")


def _mul(node):
    _require(node.node_type == NodeType.POINTWISE and node.params == {"mode": "mul"} and set(node.inputs) == {"IN_0", "IN_1"}, "binary FP32 multiplication")
    _fp32(node)
    return node.inputs["IN_0"], node.inputs["IN_1"]


def _other_operand(node, value):
    left, right = _mul(node)
    if left is value:
        return right
    _require(right is value, "the GEMM result as a multiplication operand")
    return left


def _per_tensor(nodes, mm):
    # Preserve (A @ B) * (sa * sb), including both FP32 rounding points.
    _require(len(nodes) == 3 and nodes[-1] is not mm, "a matmul multiplied by the product of two FP32 scalars")
    product = next(node for node in nodes[:-1] if node is not mm)
    sa, sb = _mul(product)
    _scalar(sa)
    _scalar(sb)
    factor = _other_operand(nodes[-1], mm.outputs["C"])
    _require(product.outputs["OUT_0"] is factor, "the product of the scalar scales in the epilogue")
    _scalar(factor, virtual=True)
    a, b = mm.inputs["A"], mm.inputs["B"]
    _require(a.get_data_type() in _FP8 and b.get_data_type() == a.get_data_type(), "matching E4M3 or E5M2 inputs")
    in_fmt = "e4m3" if a.get_data_type() == dt.FP8_E4M3 else "e5m2"
    return a, b, sa, sb, None, nodes[-1].outputs["OUT_0"], "per_tensor_fp8", in_fmt


def _block_size(node):
    dims = tuple(node.params.get("block_size") or ())
    if len(dims) == 3 and dims[0] == 1:
        dims = dims[1:]
    _require(len(dims) == 2 and not node.params.get("is_negative_scale"), "two positive matrix block dimensions and unsigned scaling")
    return dims


def _block_scaled(nodes, mm):
    _require(
        len(nodes) in (3, 4) and nodes[2] is mm and all(node.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE for node in nodes[:2]),
        "two block dequantizations feeding matmul",
    )
    dequants = {node.outputs["OUT_0"]: node for node in nodes[:2]}
    qa, qb = mm.inputs["A"], mm.inputs["B"]
    _require(qa in dequants and qb in dequants and qa is not qb, "separately scaled A and B")
    na, nb = dequants[qa], dequants[qb]
    a, b = na.inputs["input"], nb.inputs["input"]
    sa, sb = na.inputs["descale"], nb.inputs["descale"]
    _fp32(na)
    _fp32(nb)
    _virtual_float(qa, a.dim)
    _virtual_float(qb, b.dim)
    scale_type = sa.get_data_type()
    _require(sb.get_data_type() == scale_type, "matching block scale formats")
    block_a, block_b = _block_size(na), _block_size(nb)
    formats = {
        (dt.FP8_E4M3, dt.FP8_E4M3, dt.FLOAT): ("groupwise_fp8", "e4m3_block128", 128, 128),
        (dt.FP8_E4M3, dt.FP8_E4M3, dt.FP8_E8M0): ("mxfp8", "mxfp8", 32, 1),
        (dt.FP4_E2M1, dt.FP4_E2M1, dt.FP8_E4M3): ("nvfp4", "nvfp4", 16, 1),
        (dt.FP4_E2M1, dt.FP4_E2M1, dt.FP8_E8M0): ("mxfp4", "mxfp4", 32, 1),
        (dt.FP8_E4M3, dt.FP4_E2M1, dt.FP8_E8M0): ("mxfp8_mxfp4", "mxfp4w8a", 32, 1),
    }
    fmt = formats.get((a.get_data_type(), b.get_data_type(), scale_type))
    _require(fmt is not None, "a supported block-scaled GEMM format")
    kind, in_fmt, block_k, block_n = fmt
    _require(block_a == (1, block_k) and block_b == (block_k, block_n), "the block shapes of the declared format")
    _require(len(a.dim) >= 2 and len(b.dim) >= 2, "matrix operands")
    m, k, n = a.dim[-2], a.dim[-1], b.dim[-1]
    groupwise = kind == "groupwise_fp8"
    ks = (k + block_k - 1) // block_k
    if groupwise:
        _require(k % 128 == 0, "complete K=128 scale blocks")
        shape_a, shape_b = (m, ks), (ks, (n + 127) // 128)
    else:
        ks = (ks + 3) // 4 * 4
        shape_a, shape_b = ((m + 127) // 128 * 128, ks), (ks, (n + 127) // 128 * 128)
        _require(tuple(sa.stride[-2:]) == (ks, 1) and tuple(sb.stride[-2:]) == (1, ks), "F8_128x4 storage with contiguous groups along K")
    for scale, shape in ((sa, shape_a), (sb, shape_b)):
        _tensor(scale, reordered=not groupwise)
        _require(tuple(scale.dim) in (shape, (1, *shape)), "scale extents matching the matrix blocks")
    alpha = None
    final = mm.outputs["C"]
    if len(nodes) == 4:
        _require(kind in ("nvfp4", "mxfp4", "mxfp8_mxfp4"), "alpha only for FP4 projections")
        alpha = _other_operand(nodes[-1], final)
        _scalar(alpha)
        final = nodes[-1].outputs["OUT_0"]
    return a, b, sa, sb, alpha, final, kind, in_fmt


def is_scaled(nodes):
    return any(n.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE or (n.node_type == NodeType.MATMUL and n.inputs["A"].get_data_type() in _FP8) for n in nodes)


def match_scaled(nodes):
    mms = [node for node in nodes if node.node_type == NodeType.MATMUL]
    _require(len(mms) == 1, "a single GEMM")
    mm = mms[0]
    _fp32(mm)
    _require(mm.params["padding"] == 0, "zero matmul padding")
    if any(node.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE for node in nodes):
        a, b, sa, sb, alpha, out, kind, in_fmt = _block_scaled(nodes, mm)
    else:
        a, b, sa, sb, alpha, out, kind, in_fmt = _per_tensor(nodes, mm)

    for tensor in (a, b, out):
        _tensor(tensor)
    for t in (a, b):
        if t.get_data_type() == dt.FP4_E2M1:
            _require(storage_geometry(t.dim, t.stride, dt.FP4_E2M1) is not None, "two FP4 elements per storage byte along K")
    _require(len(a.dim) in (2, 3) and len(a.dim) == len(b.dim) and a.dim[:-2] == b.dim[:-2], "rank-2 matrices or equal rank-3 batch extents")
    m, k = a.dim[-2:]
    kb, n = b.dim[-2:]
    batch = a.dim[0] if len(a.dim) == 3 else 1
    matrix_shape = (*a.dim[:-2], m, n)
    _require(k == kb and n % 8 == 0 and max(batch, m, n, k) < 2**31, "matching positive int32 matrix dimensions and N aligned to 8")
    _require(kind == "per_tensor_fp8" or batch == 1, "a single matrix for block-scaled GEMM")
    _require(tuple(out.dim) == matrix_shape and out.get_data_type() in (dt.BFLOAT16, dt.HALF), "BF16 or FP16 output of shape (..., M, N)")
    if mm.outputs["C"] is not out:
        _virtual_float(mm.outputs["C"], matrix_shape)
    _require(out.stride[-1] == 1 and (m == 1 or out.stride[-2] == n), "row-major output")

    a_align = 32 if a.get_data_type() == dt.FP4_E2M1 else 16
    k_align = 16 if kind == "per_tensor_fp8" else 32
    lda = a.stride[-2] if m > 1 else k
    _require(k % k_align == 0 and a.stride[-1] == 1 and lda >= k and lda % a_align == 0, "aligned K and row-major A with 16-byte aligned rows")
    b_outer, b_inner = b.stride[-2:]
    if kind == "per_tensor_fp8" and (b_outer, b_inner) == (n, 1):
        _require(n % 16 == 0, "N aligned to 16 for row-major FP8 B")
        major, ldb = "n", n
    else:
        _require(
            b_outer == 1 and (b_inner == k or (kind == "groupwise_fp8" and b_inner >= k and b_inner % 16 == 0)),
            "a K-contiguous weight with supported row pitch",
        )
        major, ldb = "k", b_inner
    a_batch, b_batch, out_batch = (t.stride[0] if batch > 1 else 0 for t in (a, b, out))
    _require(
        a_batch % 16 == 0 and b_batch % 16 == 0 and out_batch % 8 == 0 and (batch == 1 or out_batch >= m * n),
        "aligned batch strides and disjoint output matrices",
    )
    pdl = mm.params.get("enable_pdl", False)
    _require(not pdl or kind == "mxfp8_mxfp4", "PDL on the mixed MXFP8/MXFP4 kernel")
    return ScaledMatmul(
        a=a,
        b=b,
        out=out,
        sa=sa,
        sb=sb,
        alpha=alpha,
        kind=kind,
        in_fmt=in_fmt,
        out_fmt="bf16" if out.get_data_type() == dt.BFLOAT16 else "fp16",
        batch=batch,
        m=m,
        n=n,
        k=k,
        lda=lda,
        ldb=ldb,
        b_major=major,
        a_batch_stride=a_batch,
        b_batch_stride=b_batch,
        out_batch_stride=out_batch,
        sa_strides=tuple(sa.stride[-2:]),
        # Kernel scale coordinates for both matrices are (row, K block).
        sb_strides=tuple(reversed(sb.stride[-2:])),
        enable_pdl=pdl,
    )
