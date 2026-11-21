# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every block of a model in plain PyTorch: the reference of the tie-out, and the provider of the blocks that have no kernel yet
-- among them every block of a drafter's pass.

The functions keep the quantisation rules and every intermediate cast of a serving chain (a norm output, a SiLU, a gate, the
conv taps and the core output of the linear-attention step are each rounded to bf16 where the chain rounds them), and they
accumulate every multiply and every reduction in ``acc`` -- fp64 by default.  fp64 is deliberate: an fp32 reference changes with
the matmul precision mode of the environment, and a small accumulation error can cross a bf16 rounding midpoint.

A block that runs here is not fast and is not meant to be: one launch per PyTorch operator, temporaries allocated per call.
"""

from __future__ import annotations

import struct

import torch
import torch.nn.functional as F

from ..census import Kernel
from ..sampling import greedy
from ..state import RecurrentSlots, TokenSlots
from ..weights import (
    BLOCK,
    E4M3_MAX,
    BlockScaledTensor,
    Dequantized,
    ScaledTensor,
    _e2m1,
    _e4m3,
    _e4m3_values,
    dequantize,
    quantize_fp8,
    quantize_fp8_per_row,
    quantize_nvfp4,
)
from . import FP8_ROW, PYTORCH_CHAIN, PYTORCH_CHAIN_WITH_MULTIPLY, BlockBase, acc_dtype

BF16 = torch.bfloat16


def quantize(y: torch.Tensor, quant):
    """The quantised operand of a consumer, from a bf16 activation: static 8-bit, 4-bit with a static global scale, or
    8-bit with one scale per row that is formed from the row (``weights.quantize_fp8_per_row``)."""
    if quant is None:
        return None
    if quant.fmt == "fp8":
        return quantize_fp8(y, quant.scale)
    if quant.fmt == "nvfp4":
        return quantize_nvfp4(y, quant.scale)
    if quant.fmt == FP8_ROW:
        return quantize_fp8_per_row(y)
    raise ValueError(f"unknown quantisation format {quant.fmt!r}")


def rms_norm(h: torch.Tensor, weight: torch.Tensor, eps: float, acc) -> torch.Tensor:
    """RMSNorm over the last dimension with the weight stored as an offset (the norm multiplies by ``1 + w``), rounded to bf16 once."""
    v = h.to(acc)
    return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * (1.0 + weight.to(acc))).to(BF16)


def add_norm_quantize(branch, residual, weight, quant, eps: float, acc):
    """Residual add, RMSNorm, quantise.  The norm reads the fp32 sum; the residual stream keeps its bf16 rounding.  Returns
    ``(quantised operand or None, normed bf16, new residual)``; the first layer of a model has no residual yet."""
    if residual is None:
        h, new_residual = branch.float(), branch
    else:
        h = residual.float() + branch.float()
        new_residual = h.to(BF16)
    y = rms_norm(h, weight, eps, acc)
    return quantize(y, quant), y, new_residual


def linear(qx, weight: Dequantized, acc, columns_per_chunk: int = 8192) -> torch.Tensor:
    """A multiply on quantised operands: the code values of both sides multiplied and accumulated in ``acc``, the scales applied
    to the result, one rounding to bf16.  Chunked over the weight's rows so the wide copy is never held twice in ``acc``.
    It reads the weight's bf16 copy: the one a load kept, or -- in the checks of a run that keeps the weight on its stored
    format alone -- the one the checks made for this call (``weights.OnDemand``).  An operand whose rows were each
    quantised at a scale of their own brings one scale per row: element ``[i, j]`` of the result then takes the scale of
    row ``i`` of the operand times the scale of row ``j`` of the weight (or the weight's one scale)."""
    x, copy = dequantize(qx), weight.bf16()
    xa = x.data.to(acc)
    out = torch.empty((xa.shape[0], copy.shape[0]), dtype=acc, device=xa.device)
    for a in range(0, copy.shape[0], columns_per_chunk):
        out[:, a : a + columns_per_chunk] = xa @ copy[a : a + columns_per_chunk].to(acc).T
    if isinstance(x.scale, torch.Tensor) and x.scale.dim() == 1 and x.scale.shape[0] == xa.shape[0]:
        # one scale per row of the operand, beside the weight's own: the two are multiplied in ``acc``, like the sums they scale
        per_weight = weight.scale.to(acc) if isinstance(weight.scale, torch.Tensor) else float(weight.scale)
        scale = x.scale.to(acc).unsqueeze(1) * per_weight
    else:  # one scale for every row of the operand: a float, or a tensor of one element
        scale = weight.scale.to(acc) * x.scale if isinstance(weight.scale, torch.Tensor) else float(weight.scale) * float(x.scale)
    return (out * scale).to(BF16)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """``SiLU(gate) * up`` on the two halves of a merged gate/up output, with the chain's two bf16 roundings: the SiLU is
    formed in fp32 and rounded, and so is its product with ``up``."""
    half = gate_up.shape[-1] // 2
    return (F.silu(gate_up[:, :half].float()).to(BF16).float() * gate_up[:, half:].float()).to(BF16)


def silu_mul_quantize(gate_up: torch.Tensor, quant):
    """:func:`silu_mul`, then quantise."""
    return quantize(silu_mul(gate_up), quant)


def pre_fc_norm_quantize(embedding: torch.Tensor, hidden: torch.Tensor, weight_embedding, weight_hidden, quant, eps: float, acc):
    """The operand of a drafter's first multiply: the embedding rows and the hidden rows, each normed with its own weight and
    rounded to bf16 (:func:`rms_norm`), laid side by side -- the embedding half first -- and quantised as ONE row
    ``[rows, 2 * hidden]``: under a scale per row both halves of a row share it."""
    return quantize(torch.cat((rms_norm(embedding, weight_embedding, eps, acc), rms_norm(hidden, weight_hidden, eps, acc)), dim=-1), quant)


def rope_cos_sin(position: int, rotary_dim: int, theta: float, device):
    """cos and sin of the rotary angles of one position, rounded to bf16 as a serving chain's table is."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    angle = position * inv_freq
    return angle.cos().to(BF16), angle.sin().to(BF16)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding on the first ``2 * len(cos)`` dims of every head (the two halves pair up), in bf16 arithmetic."""
    half = cos.shape[0]
    x1, x2, rest = x[..., :half], x[..., half : 2 * half], x[..., 2 * half :]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1)


def attention_prepare(qkv: torch.Tensor, cache, position: int, p, eps: float, acc):
    """Split the in-projection's row into query, gate, key and value, norm query and key per head, rotate both, and insert key and
    value at ``position`` of the layer's cache (``cache.write``).  The row is ``[q | gate]`` per query head, then the keys, then
    the values.  Returns ``(q [rows, heads, D] bf16, gate [rows, heads * D] bf16)``."""
    rows, H, HK, D = qkv.shape[0], p.heads, p.kv_heads, p.head_dim
    q_gate = qkv[:, : 2 * H * D].view(rows, H, 2 * D)
    q, gate = q_gate[..., :D], q_gate[..., D:]
    k = qkv[:, 2 * H * D : (2 * H + HK) * D].view(rows, HK, D)
    v = qkv[:, (2 * H + HK) * D :].view(rows, HK, D)
    cos, sin = rope_cos_sin(position, p.rotary_dim, p.rope_theta, qkv.device)
    q = rotate(rms_norm(q.float(), p.q_norm, eps, acc), cos, sin)
    k = rotate(rms_norm(k.float(), p.k_norm, eps, acc), cos, sin)
    cache.write(position, k, v)
    return q, gate.reshape(rows, H * D)


def bf16_unit(value: torch.Tensor) -> torch.Tensor:
    """One bf16 unit in the last place at the magnitude of ``value`` (fp64): ``2 ** (e - 7)`` for ``2 ** e <= |value| < 2 **
    (e + 1)``, and not below the unit of the smallest normal fp32 value.  Exact on every device: the power of two is built
    from its bits.  (A power function is not: on a CUDA device ``torch.pow(2.0, k)`` is one fp64 unit low for some whole
    ``k``, and a band with such a unit refuses an element that is exactly one unit away.)"""
    exponent = torch.frexp(value.abs().double()).exponent.clamp(min=-125)
    return ((exponent.to(torch.int64) + (1023 - 8)) << 52).view(torch.float64)  # 2 ** (exponent - 8): the fp64 with that exponent field


def prepare_rows(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float, order: str = "3d"):
    """The norm and the rotation of head rows at the rounding points of a kernel that normalises in fp32, rounds the normalised
    value ONCE to bf16 and each rotated value once more (``cudnn.AttentionPrepareDecode``): ``(out, band)`` for rows ``x``
    ``[rows, heads, D]`` bf16 with one ``weight`` ``[D]`` and per row the cosines and sines of its position ``[rows, 1,
    half]`` bf16.  The twin of that entry's own suite, statement by statement
    (``test/python/attention_prepare/frost/attention_prepare_utils.py``): the mean of squares and its reciprocal root
    are evaluated in fp64 and rounded to fp32 once -- whatever precision the run's reference accumulates in: the
    kernel's contract is stated against that evaluation -- the two products of the norm are fp32 products, and a
    rotated value is one fp32 product and one fused sum (formed exactly, rounded to fp32 and then to bf16).  ``order``
    is the position form of the kernel, which decides which product of the second rotary half is the fused one.
    ``band`` (fp64) is what the two normalised operands of a rotated element may move it by, each one bf16 unit off:
    ``|cos| unit(n1) + |sin| unit(n2)`` with the units taken one unit up; zero for an element that is not rotated."""
    half = cos.shape[-1]
    xf = x.float()
    eps32 = float(torch.tensor(eps, dtype=torch.float32).double())  # the fp32 value the kernel adds
    mean = (xf.double() ** 2).sum(dim=-1, keepdim=True) / x.shape[-1]
    inv = (1.0 / torch.sqrt(mean + eps32)).float()  # the root in fp64, rounded to fp32 once
    normed = ((xf * inv) * (weight.float().reshape(1, 1, -1) + 1.0)).to(BF16)  # fp32 products, ONE rounding to bf16
    n1, n2 = normed[..., :half].float(), normed[..., half : 2 * half].float()
    c, s = cos.float(), sin.float()
    first = (n1.double() * c.double() - (n2 * s).double()).float().to(BF16)  # one fp32 product, the fused sum, one bf16 rounding
    if order == "3d":
        second = (n2.double() * c.double() + (n1 * s).double()).float().to(BF16)
    else:
        second = (n1.double() * s.double() + (n2 * c).double()).float().to(BF16)
    n = normed.double()

    def up(v):  # the unit one unit up: the kernel's operand may be the neighbour
        return bf16_unit(v.abs() + bf16_unit(v))

    spread = c.double().abs() * up(n[..., :half]) + s.double().abs() * up(n[..., half : 2 * half])
    band = torch.cat((spread, spread, torch.zeros_like(n[..., 2 * half :])), dim=-1)
    return torch.cat((first, second, normed[..., 2 * half :]), dim=-1), band


def attention_prepare_rounded(qkv: torch.Tensor, cache, position: int, tokens: int, p, eps: float, order: str = "3d"):
    """:func:`attention_prepare` at the rounding points of ``cudnn.AttentionPrepareDecode``, for a step of ``tokens`` tokens
    per sequence whose rows are sequence by sequence: ``(q, gate, terms)``.  Token ``t`` of every sequence is prepared
    for ``position + t`` -- the host integer the step hands the block, with the cosines and sines the reference chain
    forms for it (:func:`rope_cos_sin`) -- and its key and value rows are inserted there through the cache's own
    accessor (``cache.write``): nothing of this function reads a table of the state.  ``terms`` is what the band of the
    entry is judged with beside the outputs: the band of every query element, the key rows it wrote with their band
    and the value rows, in the step's row order, and the position and the tokens per sequence of the call."""
    from types import SimpleNamespace

    rows, H, HK, D = qkv.shape[0], p.heads, p.kv_heads, p.head_dim
    q_gate = qkv[:, : 2 * H * D].view(rows, H, 2 * D)
    q, gate = q_gate[..., :D], q_gate[..., D:]
    k = qkv[:, 2 * H * D : (2 * H + HK) * D].view(rows, HK, D)
    v = qkv[:, (2 * H + HK) * D :].view(rows, HK, D)
    pairs = [rope_cos_sin(position + token, p.rotary_dim, p.rope_theta, qkv.device) for token in range(tokens)]
    cos = torch.stack([pair[0] for pair in pairs]).repeat(rows // tokens, 1)[:, None]  # row (sequence, token): the position of its token
    sin = torch.stack([pair[1] for pair in pairs]).repeat(rows // tokens, 1)[:, None]
    q_out, q_band = prepare_rows(q, p.q_norm, cos, sin, eps, order)
    k_out, k_band = prepare_rows(k, p.k_norm, cos, sin, eps, order)
    keys, values = k_out.unflatten(0, (-1, tokens)), v.unflatten(0, (-1, tokens))
    for token in range(tokens):
        cache.write(position + token, keys[:, token], values[:, token])
    terms = SimpleNamespace(q_band=q_band, k=k_out, k_band=k_band, v=v, position=int(position), tokens=int(tokens))
    return q_out, gate.reshape(rows, H * D), terms


def attention(q: torch.Tensor, cache, position: int, acc) -> torch.Tensor:
    """Scaled dot-product attention of one new token per sequence over the cached keys and values up to ``position``; every
    query head reads the key/value head of its group.  Returns ``[rows, heads * D]`` bf16.  The cache is read through its
    own accessors (``keys`` / ``values``: ``[rows, kv heads, positions, D]`` in ``acc``), whatever its layout."""
    rows, H, D = q.shape
    k = cache.keys(position + 1, acc)
    v = cache.values(position + 1, acc)
    HK = k.shape[1]
    qg = q.to(acc).view(rows, HK, H // HK, D)
    p = torch.softmax(torch.einsum("bhgd,bhtd->bhgt", qg, k) * D**-0.5, dim=-1)
    return torch.einsum("bhgt,bhtd->bhgd", p, v).reshape(rows, H * D).to(BF16)


def attention_prepare_tokens(qkv: torch.Tensor, cache, position: int, tokens: int, p, eps: float, acc):
    """:func:`attention_prepare` for a step of ``tokens`` tokens per sequence, whose rows are sequence by sequence: token ``t``
    of every sequence is prepared and inserted at ``position + t``, one call per token -- the arithmetic of ``tokens``
    decode steps at their own shapes.  Returns ``(q, gate)`` in the step's row order."""
    rows = qkv.unflatten(0, (-1, tokens))
    parts = [attention_prepare(rows[:, t], cache, position + t, p, eps, acc) for t in range(tokens)]
    return tuple(torch.stack([part[output] for part in parts], dim=1).flatten(0, 1) for output in (0, 1))


def attention_tokens(q: torch.Tensor, cache, position: int, tokens: int, acc) -> torch.Tensor:
    """:func:`attention` for a step of ``tokens`` tokens per sequence: token ``t`` of every sequence attends to the cached
    keys and values up to ``position + t`` (its own included, the later tokens' not), one call per token."""
    rows = q.unflatten(0, (-1, tokens))
    return torch.stack([attention(rows[:, t], cache, position + t, acc) for t in range(tokens)], dim=1).flatten(0, 1)


def attention_bound(q: torch.Tensor, cache, position: int, tokens: int = 1):
    """The reference attention of a step BEFORE its rounding to bf16, and the weighted magnitude of the values it attends to:
    ``(chain, spread)``, each ``[rows, heads * D]`` in fp64.  For token ``t`` of a sequence, a query head and an element
    ``d``, with ``p_j`` the softmax weight of the attended position ``j = 0 .. position + t`` of that sequence (its own
    included, the later tokens' of the step not), formed as :func:`attention` forms it::

        chain  = sum_j p_j * v[j, d]
        spread = sum_j p_j * |v[j, d]|

    The operand terms of the band ``attention_bf16_weights`` (``tie_out``).  Both are formed in fp64 on the bf16 operands
    cast exactly, whatever precision the run's reference accumulates in: the band is stated against this evaluation, and
    the chain is never rounded here.  The cache is read through its own accessors with host integers, on either layout,
    and through no table of the state.  The rows of a step of several tokens stand sequence by sequence."""
    rows = q.unflatten(0, (-1, tokens))
    sequences, _, H, D = rows.shape
    chains, spreads = [], []
    for token in range(tokens):
        count = position + token + 1  # the positions this token attends to: every earlier one and its own
        k = cache.keys(count, torch.float64)
        v = cache.values(count, torch.float64)
        HK = k.shape[1]
        qg = rows[:, token].to(torch.float64).reshape(sequences, HK, H // HK, D)
        p = torch.softmax(torch.einsum("bhgd,bhtd->bhgt", qg, k) * D**-0.5, dim=-1)
        chains.append(torch.einsum("bhgt,bhtd->bhgd", p, v).reshape(sequences, H * D))
        spreads.append(torch.einsum("bhgt,bhtd->bhgd", p, v.abs()).reshape(sequences, H * D))
    return torch.stack(chains, dim=1).flatten(0, 1), torch.stack(spreads, dim=1).flatten(0, 1)


def gate_quantize(o: torch.Tensor, gate: torch.Tensor, quant):
    """``sigmoid(gate) * attention output`` with the chain's bf16 roundings, then the static 8-bit quantisation."""
    return quantize((o.float() * torch.sigmoid(gate.float()).to(BF16).float()).to(BF16), quant)


def gate_row_quantize(o: torch.Tensor, gate: torch.Tensor, quant):
    """``attention output * sigmoid(gate)`` formed in fp32 and rounded to bf16 ONCE, then quantised.  Not
    :func:`gate_quantize`, which rounds the sigmoid to bf16 before the product: another function, and another byte on
    some elements."""
    return quantize((o.float() * torch.sigmoid(gate.float())).to(BF16), quant)


def listed_argmax(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The token of every row of ``logits`` ``[rows, listed]``, the logits of a head cut to listed rows: the column of the
    largest value, the first of equal ones, then the token id of that column (``ids`` ``[listed]``, in the dtype the ids
    are returned in).  The id is read by indexing ``ids``: a gather of a step is its embedding lookup alone."""
    return ids[torch.argmax(logits, dim=-1)]


#: the largest fp32 value: an fp32 exponential beyond it is +inf, and the value it divides is then +-0
FP32_MAX = 3.4028234663852886e38


def _bf16_bits(v: torch.Tensor) -> torch.Tensor:
    return v.view(torch.int16).to(torch.int32) & 0xFFFF


def _bf16_from_bits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int16).view(BF16)


def bf16_bracket(t: torch.Tensor):
    """The bf16 bracket of a value ``t`` held in the reference's accumulation precision: ``(nearest, other, margin)``.
    ``nearest`` is the round-to-nearest-even bf16 of ``t``, decided by comparing ``t`` with the midpoint of the two bf16
    values around it (never by a cast that rounds twice); ``other`` is the bf16 value on the other side of ``t``
    (``nearest`` itself where ``t`` is a bf16 value); ``margin`` is the distance of ``t`` from the midpoint of the two, in
    fp32 units of ``t`` (+inf where it is a bf16 value).  A kernel that forms the same value in fp32 rounds to ``nearest``
    wherever the margin exceeds its own fp32 error, and to one of the two everywhere."""
    acc = t.dtype
    v = t.float().to(BF16)  # one end of the bracket (both roundings are monotone); which end is decided below, in ``acc``
    bits = _bf16_bits(v)
    sign, magnitude = bits & 0x8000, bits & 0x7FFF
    away = _bf16_from_bits(sign | (magnitude + 1).clamp(max=0x7F80))
    toward = _bf16_from_bits(sign | (magnitude - 1).clamp(min=0))
    va = v.to(acc)
    other = torch.where(t.abs() > va.abs(), away, torch.where(t.abs() < va.abs(), toward, v))
    oa = other.to(acc)
    exact = t == va
    to_v, to_other = (t - va).abs(), (t - oa).abs()
    # ``v`` unless the other end is strictly nearer.  A value ON the midpoint is exact in fp32, so the cast above has
    # already rounded it to the even end: at equal distances ``v`` is the nearest-even value
    at_v = exact | (to_v <= to_other)
    nearest = torch.where(at_v, v, other)
    # ... and where the other end is the nearer one, ``v`` stays in the bracket as its second value.  That happens where the
    # value rounds to the midpoint itself in fp32 and the even end is the farther one: ``v`` is then what an fp32 chain
    # rounds to, and a bracket without it would hold the nearest value twice
    second = torch.where(at_v, other, v)
    exponent = torch.frexp(t.abs()).exponent.clamp(min=-125)
    unit = torch.pow(torch.tensor(2.0, dtype=acc, device=t.device), (exponent - 24).to(acc))  # one fp32 unit of the value
    margin = torch.where(exact, torch.full_like(t, float("inf")), (t - (va + oa) * 0.5).abs() / unit)
    return nearest, second, margin


def gate_quotient_bracket(o: torch.Tensor, gate: torch.Tensor, acc):
    """The gated attention output formed as the QUOTIENT ``o / (1 + exp(-gate))``, accumulated in ``acc`` on the bf16 operands,
    with its bf16 bracket (:func:`bf16_bracket`): ``(nearest, other, margin)``.  The exponential is formed as the kernel
    forms it: where it overflows fp32 it is +inf and the gated value +-0, whatever ``acc`` could still represent."""
    exponential = torch.exp(-gate.to(acc))
    exponential = torch.where(exponential > FP32_MAX, torch.full_like(exponential, float("inf")), exponential)
    return bf16_bracket(o.to(acc) / (1.0 + exponential))


#: an NVFP4 kernel scales a block by its largest magnitude times this constant: fp32(1 / 6), by its bits
FP4_SIXTH_BITS = 0x3E2AAAAB
#: the relative bound of the encode scale of ``cudnn.AddNormQuantDecode``'s NVFP4 codes, which goes through two approximate
#: reciprocals: one unit in the last place each, doubled, and one fp32 unit for the two multiplications -- the bound the
#: entry's own suite holds its codes to (``test/python/norm/frost/add_norm_quant_utils.py``)
NVFP4_ENCODE_BOUND = 5 * 2.0**-23


def norm_bracket(branch: torch.Tensor, residual, weight: torch.Tensor, eps: float):
    """The residual add and the RMSNorm at the rounding points of a kernel that normalises in fp32 and rounds ONCE to bf16
    (``cudnn.AddNormQuantDecode``): ``(new residual, nearest, other, margin)``.  The new residual is ``bf16(fp32(branch) +
    fp32(residual))`` (the branch itself where there is no residual yet).  The normalised value is evaluated in fp64 on
    the bf16 operands -- ``v (1 + w) / sqrt(mean(v^2) + eps)`` with ``v`` the exact sum and ``eps`` the fp32 value the kernel
    adds -- whatever precision the run's reference accumulates in: the kernel's contract is stated against that
    evaluation; then its bf16 bracket (:func:`bf16_bracket`)."""
    if residual is None:
        v, new_residual = branch.double(), branch
    else:
        v, new_residual = residual.double() + branch.double(), (residual.float() + branch.float()).to(BF16)
    eps32 = struct.unpack("<f", struct.pack("<f", eps))[0]
    t = v * (1.0 + weight.double()) / torch.sqrt(v.square().mean(-1, keepdim=True) + eps32)
    nearest, other, margin = bf16_bracket(t)
    return new_residual, nearest, other, margin


def nvfp4_global_scale(scale: float, device) -> torch.Tensor:
    """The global scale an NVFP4 kernel takes for an operand whose consumer reads ``code * block_scale * scale``: the fp32
    reciprocal of ``scale``, correctly rounded (the consumer of the kernel's bytes reads ``code * block_scale / global``)."""
    return torch.ones((), dtype=torch.float32, device=device) / torch.tensor(float(scale), dtype=torch.float32, device=device)


def nvfp4_block_scales(largest: torch.Tensor, scale: float) -> torch.Tensor:
    """The E4M3 block-scale bytes of blocks whose largest magnitudes are ``largest`` (fp32), as the kernel forms them:
    ``e4m3(global * (largest * fp32(1 / 6)))`` -- two fp32 multiplications and one nearest-even rounding."""
    sixth = torch.tensor(FP4_SIXTH_BITS, dtype=torch.int32, device=largest.device).view(torch.float32)
    return _e4m3(nvfp4_global_scale(scale, largest.device) * (largest.float() * sixth))


def nvfp4_code_bracket(y: torch.Tensor, scale_bytes: torch.Tensor, scale: float):
    """``(low, high)``: the two E2M1 codes (one nibble per element, the sign kept on a zero) the kernel can write for the bf16
    values ``y`` under the stored block scales ``scale_bytes``.  Its encode scale goes through two approximate
    reciprocals, so with ``q = global / block_scale`` in fp64 a code is ``e2m1(y q (1 - r))`` or ``e2m1(y q (1 + r))``,
    ``r`` = :data:`NVFP4_ENCODE_BOUND`: two adjacent codes at most, and one wherever ``y q`` is not within ``r`` of a
    rounding midpoint.  A block whose scale byte is zero holds the codes of the signed zero."""
    stored = _e4m3_values(scale_bytes, torch.float64).repeat_interleave(BLOCK, dim=-1)
    live = stored != 0
    quotient = torch.where(live, nvfp4_global_scale(scale, y.device).double() / torch.where(live, stored, torch.ones_like(stored)), torch.zeros_like(stored))
    exact = y.double() * quotient  # under a zero block scale: the signed zero of y
    return _e2m1(exact * (1.0 - NVFP4_ENCODE_BOUND), signed_zero=True), _e2m1(exact * (1.0 + NVFP4_ENCODE_BOUND), signed_zero=True)


#: the error bound of ``cudnn.SwigluFp4Decode`` against an fp64 evaluation of its chain, term by term as the entry's own
#: suite derives it (``test/python/gemm/frost/swiglu_decode/swiglu_fp4_decode_utils.py``; the entry's page states the
#: contract): one rounding to bf16 (relative), one fp32 unit per product of the accumulation, SiLU's Lipschitz constant,
#: what the approximate exponential, division and reciprocals add to the second rounding, a floor for a value of zero,
#: and the relative slack of the three fp32 multiplications of a block scale.  Not tolerances of this sample: they move
#: only with that suite
SWIGLU_ROUNDING = 2.0**-8
SWIGLU_SUM_UNIT = 2.0**-23
SWIGLU_SILU_LIPSCHITZ = 1.1
SWIGLU_APPROXIMATE = 2.0**-19
SWIGLU_FLOOR = 2.0**-120
SWIGLU_SCALE_SLACK = 2.0**-21


def fp32_product(a: float, b: float) -> torch.Tensor:
    """``fp32(a) * fp32(b)`` as one fp32 element on the host: the scalar a kernel reads for the product of two static
    scales, formed once per pair of values and never inside a step."""
    return torch.tensor([float(a)], dtype=torch.float32) * torch.tensor([float(b)], dtype=torch.float32)


def e4m3_nearest(x: torch.Tensor) -> torch.Tensor:
    """Non-negative fp64 values -> E4M3 code bytes, round to nearest even, saturating at 448: a search over the 127 finite
    non-negative codes, whose values ascend with the byte.  No cast is involved, so a value is rounded once."""
    values = _e4m3_values(torch.arange(0, 0x7F, dtype=torch.uint8, device=x.device), torch.float64)
    x = x.clamp(max=E4M3_MAX).contiguous()
    high = torch.searchsorted(values, x).clamp(max=0x7E)
    low = (high - 1).clamp(min=0)
    below, above = x - values[low], values[high] - x
    return torch.where((above < below) | ((above == below) & (high % 2 == 0)), high, low).to(torch.uint8)


def gate_up_bound(qx, weight: Dequantized, rows_per_chunk: int = 4096):
    """The fused gate / up step of an MLP evaluated in fp64 on the exactly dequantised operands, with the error bound a
    kernel that accumulates the two products in fp32, rounds each once to bf16, forms ``silu(gate) * up`` with an
    approximate exponential and division and rounds once more stays inside (``cudnn.SwigluFp4Decode``):
    ``(value, error, gate, up)``, each ``[rows, intermediate]`` in fp64.  With ``alpha`` the fp32 product of the two
    scales and ``K`` the input width::

        gate = alpha * sum_k x_k Wg_k          spread_g = |alpha| * sum_k |x_k Wg_k|          (up, spread_u likewise)
        value = silu(gate) * up
        d_g = 2^-8 |gate| + (1 + 2^-8) (K + 1) 2^-23 spread_g                                  (d_u likewise)
        a = 1.1 d_g (|up| + d_u) + |silu(gate)| d_u
        error = a + (2^-8 + 2^-19) (|value| + a) + 2^-120

    The weight is read as its bf16 copy in the checkpoint's row order: every gate row, then every up row."""
    x, copy = dequantize(qx), weight.bf16()
    alpha = float(fp32_product(x.scale, weight.scale))
    xa = x.data.double()
    magnitudes = xa.abs()
    width, half = copy.shape[1], copy.shape[0] // 2
    sums = torch.empty((2, xa.shape[0], copy.shape[0]), dtype=torch.float64, device=xa.device)
    for a in range(0, copy.shape[0], rows_per_chunk):  # chunked: the wide copy is never held twice in fp64
        rows = copy[a : a + rows_per_chunk].double()
        sums[0, :, a : a + rows_per_chunk] = xa @ rows.T
        sums[1, :, a : a + rows_per_chunk] = magnitudes @ rows.abs().T
    gate, up = alpha * sums[0, :, :half], alpha * sums[0, :, half:]
    silu = gate * torch.sigmoid(gate)
    value = silu * up
    unit = (1.0 + SWIGLU_ROUNDING) * (width + 1) * SWIGLU_SUM_UNIT * abs(alpha)
    d_gate = SWIGLU_ROUNDING * gate.abs() + unit * sums[1, :, :half]
    d_up = SWIGLU_ROUNDING * up.abs() + unit * sums[1, :, half:]
    spread = SWIGLU_SILU_LIPSCHITZ * d_gate * (up.abs() + d_up) + silu.abs() * d_up
    error = spread + (SWIGLU_ROUNDING + SWIGLU_APPROXIMATE) * (value.abs() + spread) + SWIGLU_FLOOR
    return value, error, gate, up


def gdn_decode_step(x, z, hidden, slot_ids, p, slots, eps: float, acc) -> ScaledTensor:
    """One decode step of a gated delta-rule linear-attention layer, for ``rows`` sequences whose slots are ``slot_ids``:
    the beta / alpha projection of the normed hidden state, the causal conv over the slot's history with SiLU, the scaled query
    and key, the state update ``S <- S * decay + ((v - S k) * beta) k^T``, the gated RMSNorm of ``S q`` and the static 8-bit
    quantisation.  Updates the slot's state and conv history in place and returns the 8-bit output ``[rows, V * D]``."""
    rows, D, NV, NK = x.shape[0], p.head_dim, p.value_heads, p.key_heads
    K0, V0 = NK * D, 2 * NK * D
    idx = slot_ids.long()
    taps = slots.conv[idx].float()
    cw = p.conv_weight.float()
    conv = (taps * cw[:, : taps.shape[-1]]).to(BF16).to(acc).sum(-1) + (x.float() * cw[:, -1]).to(BF16).to(acc)
    c = F.silu(conv).to(BF16).to(acc)
    history = slots.conv_storage
    for tap in range(history.shape[1] - 1):
        history[idx, tap] = history[idx, tap + 1]
    history[idx, -1] = x
    ba = (hidden.to(acc) @ p.ba_weight.to(acc).T).to(BF16).to(acc)
    av = ba[:, NV:] + p.dt_bias.to(acc)
    soft = torch.where(av <= 20, torch.log1p(torch.exp(av.clamp(max=20))), av)
    decay = torch.exp(-torch.exp(p.a_log.to(acc)) * soft)
    beta = torch.sigmoid(ba[:, :NV])
    q, k, v = c[:, :K0].view(rows, NK, D), c[:, K0:V0].view(rows, NK, D), c[:, V0:].view(rows, NV, D)
    q = (q * (D**-0.5 / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6))).repeat_interleave(NV // NK, dim=1)
    k = (k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(NV // NK, dim=1)
    S = slots.state[idx].to(acc) * decay[:, :, None, None]
    inner = (v - torch.einsum("bhij,bhj->bhi", S, k)) * beta[:, :, None]
    S = S + inner[:, :, :, None] * k[:, :, None, :]
    slots.state[idx] = S.float()
    o = torch.einsum("bhij,bhj->bhi", S, q).float().to(BF16).to(acc)
    inv = torch.rsqrt(o.square().mean(-1, keepdim=True) + eps)
    gate = z.to(acc)
    out = (((o * inv) * p.norm_weight.to(acc)) * (gate / (1 + torch.exp(-gate)))).to(BF16)
    codes = _e4m3(out.float() / p.out_scale.float())
    return ScaledTensor(codes.reshape(rows, NV * D), p.out_scale_value)


def gdn_verify_step(x, z, hidden, table: TokenSlots, p, slots, eps: float, acc, anchor=None) -> ScaledTensor:
    """One verify step of a gated delta-rule linear-attention layer: ``tokens`` decode steps (:func:`gdn_decode_step`) per
    sequence in one call, on rows that are sequence by sequence, each token's state kept in a slot of its own.  A sequence
    starts from the slot of the token the last step kept (``table.accept``) and from the three history columns that
    token left; the state after token ``t`` is written to the slot of (sequence, ``t``), and the history becomes ``[the
    last two taps, x of token 0, .., x of token T - 1]``.  So the state a step leaves for token ``t`` is the state of
    ``t + 1`` decode steps, and a run that keeps ``t`` tokens behind the first goes on from exactly that.  Returns the
    8-bit output ``[rows, V * D]`` in the step's row order.

    ``anchor`` is for the check of a kernel that makes this step in one launch: the state pool as that kernel left it.
    Every token after the first then starts from the state the KERNEL wrote for the token before it, so each token is
    one decode step compared on its own -- the comparison the decode step's band is stated for, in which a rounding
    that falls on the other side of a midpoint is seen at the token where it happens and not again at every later one."""
    sequences, tokens = table.sequences, table.tokens
    slot, kept = table.indices.long(), table.accept
    home = slot[:, 0]  # a sequence's history lives in its plain slot
    taps = slots.conv_storage.shape[1] - tokens + 1
    work = RecurrentSlots(torch.zeros_like(slots.state[: sequences + 1]), torch.zeros_like(slots.conv_storage[: sequences + 1, :taps]))
    work.state[1:] = slots.state[slot[:, kept]]
    work.conv_storage[1:] = slots.conv_storage[home, kept : kept + taps]
    last = work.conv_storage[1:, 1:].clone()  # the taps that stay in front of the step's own rows
    ids = torch.arange(1, sequences + 1, dtype=torch.int32, device=x.device)
    xs, zs, hs = (rows.unflatten(0, (sequences, tokens)) for rows in (x, z, hidden))
    codes = []
    for t in range(tokens):
        if anchor is not None and t:
            work.state[1:] = anchor[slot[:, t - 1]]
        codes.append(gdn_decode_step(xs[:, t], zs[:, t], hs[:, t], ids, p, work, eps, acc).codes)
        slots.state[slot[:, t]] = work.state[1:]
    slots.conv_storage[home] = torch.cat((last, xs), dim=1)
    return ScaledTensor(torch.stack(codes, dim=1).flatten(0, 1), p.out_scale_value)


class _Reference(BlockBase):
    """A block served by this module.  ``eps`` is the model's norm epsilon; ``acc`` the accumulation dtype of the run.  Its
    census entry is the family of PyTorch's own kernels; a chain that multiplies (``matmul``, ``einsum``) adds cuBLAS's."""

    kernels = PYTORCH_CHAIN

    def __init__(self, row, model, facts, options):
        super().__init__(row, model, facts, options)
        self.acc = acc_dtype(options)
        self.eps = float(getattr(model, "norm_eps", 1e-6))


class Embedding(_Reference):
    """One gather per call: an exact kernel, known by its name.  Its launch shape is PyTorch's and is declared nowhere."""

    kernels = (Kernel("embedding_gather", "PyTorch", "exact", launches=1),)

    def __call__(self, ids, table):
        return table.index_select(0, ids)


class TokenChoice(_Reference):
    """The sample's host path: what it launches is listed by the kernel check and compared with nothing."""

    kernels = (Kernel("pytorch", "PyTorch", "host"),)

    def __call__(self, logits):
        return greedy(logits)


class ListedTokenChoice(_Reference):
    """The token choice of a head cut to listed rows, with the lookup of the chosen row's token id: a block of a drafter's
    pass, which a kernel is to serve -- a reference chain, not the sample's host path."""

    def __call__(self, logits, ids):
        return listed_argmax(logits, ids)


class PreFcNormQuantize(_Reference):
    per_row = (0, 1)

    def __call__(self, embedding, hidden, weight_embedding, weight_hidden, quant):
        return pre_fc_norm_quantize(embedding, hidden, weight_embedding, weight_hidden, quant, self.eps, self.acc)


class Quantize(_Reference):
    def __call__(self, rows, quant):
        return quantize(rows, quant)


class SiluMul(_Reference):
    def __call__(self, gate_up):
        return silu_mul(gate_up)


class AddNormQuantize(_Reference):
    per_row = (0, 1)

    def __call__(self, branch, residual, weight, quant, site=""):  # the chain is one function at every call site
        return add_norm_quantize(branch, residual, weight, quant, self.eps, self.acc)


class AddNormQuantizeBracket(_Reference):
    """The residual add + RMSNorm + quantise at the rounding points of a kernel that normalises in fp32, rounds the normed
    value ONCE to bf16 and forms its codes from that bf16 value (``cudnn.AddNormQuantDecode``).  The reference chain
    (``add_norm_quantize``) is the same function with its one rounding taken from fp64; the kernel's fp32 value may fall
    on the other side of a bf16 midpoint, and its 4-bit codes follow its own block scales and reciprocals.  This block is
    the twin of that entry's adapter, not a row of any table: it writes what the form of the call site writes (``forms``:
    site -> the fields of the adapter's row there) from the NEAREST bf16 value, and keeps of the last call what the band
    of the entry is judged with (``tie_out``: ``add_norm_quant``): the names of the outputs it compares (``outputs``),
    ``bracket`` = the nearest bf16 value, the other one around the normalised value and that value's distance from their
    midpoint, and the quantisation of the call (``quant``)."""

    entry = "torch_ref.add_norm_quantize at the kernel's rounding points"
    per_row = (0, 1)
    forms: dict = {}
    bracket = None
    outputs: tuple = ()
    quant = None

    def __call__(self, branch, residual, weight, quant, site=""):
        form = self.forms[site]
        new_residual, nearest, other, margin = norm_bracket(branch, residual, weight, self.eps)
        self.bracket, self.quant = (nearest, other, margin), quant
        q, names = None, []
        if form.get("QUANT") == "e4m3_static":
            q, names = quantize_fp8(nearest, quant.scale), ["codes"]
        elif form.get("QUANT") == "nvfp4":
            scales = nvfp4_block_scales(nearest.float().abs().unflatten(-1, (-1, BLOCK)).amax(-1), quant.scale)
            low, _ = nvfp4_code_bracket(nearest, scales, quant.scale)
            q, names = BlockScaledTensor(low[..., 0::2] | (low[..., 1::2] << 4), scales, float(quant.scale)), ["codes", "scales"]
        normed = nearest if form.get("BF16_OUT") else None
        self.outputs = (*names, *(["normed"] if normed is not None else []), "residual")
        return q, normed, new_residual


class Linear(_Reference):
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY

    def __call__(self, qx, weight, site=""):  # the chain is one function at every call site
        return linear(qx, weight, self.acc)


class GateUp(_Reference):
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY

    def __call__(self, qx, weight, quant):
        self.product = linear(qx, weight, self.acc)
        return silu_mul_quantize(self.product, quant)

    def compared(self, out):
        return self.product


class GateUpFp4Bound(_Reference):
    """The gate / up multiply, ``SiLU(gate) * up`` and the 4-bit quantisation as the contract of a kernel that fuses the
    three is stated (``cudnn.SwigluFp4Decode``): against an fp64 evaluation on the exactly dequantised operands, with a
    propagated error bound.  The kernel accumulates both products in fp32 over the whole input width, rounds each once
    to bf16, forms the SiLU with an approximate exponential and division, rounds the product to bf16 and writes 4-bit
    codes under block scales it forms from that value; the reference pair of this block (``linear``, then
    ``silu_mul_quantize``) rounds at the same points from fp64, and no bracket of two values holds the distance of an
    fp32 sum of thousands of products.  This block is the twin of that entry's adapter, not a row of any table.  It
    returns the codes and block scales of the fp64 chain at the kernel's rounding points and keeps of the last call
    what the band of the entry is judged with (``tie_out``: ``swiglu_fp4``): ``bound`` = the fp64 value of every
    element and its error bound, ``quant`` = the quantisation of the call."""

    entry = "torch_ref.linear + torch_ref.silu_mul_quantize at the kernel's rounding points"
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY
    bound = None
    quant = None

    def __call__(self, qx, weight, quant):
        if quant is None or quant.fmt != "nvfp4":
            raise ValueError("the fused gate / up kernel's twin quantises to 4-bit codes")
        value, error, gate, up = gate_up_bound(qx, weight)
        self.bound, self.quant = (value, error), quant
        gate, up = bf16_bracket(gate)[0].double(), bf16_bracket(up)[0].double()  # each product rounded once to bf16
        nearest = bf16_bracket(gate * torch.sigmoid(gate) * up)[0]  # SiLU x up, the second rounding
        scales = nvfp4_block_scales(nearest.float().abs().unflatten(-1, (-1, BLOCK)).amax(-1), quant.scale)
        low, _ = nvfp4_code_bracket(nearest, scales, quant.scale)
        return BlockScaledTensor(low[..., 0::2] | (low[..., 1::2] << 4), scales, float(quant.scale))


class AttentionPrepare(_Reference):
    per_row = (0, 1)
    state_args = (1,)

    def __call__(self, qkv, cache, position, params):
        tokens = getattr(cache, "tokens", 1)  # of the steps that write this cache: a step of several inserts that many positions
        if tokens > 1:
            return attention_prepare_tokens(qkv, cache, position, tokens, params, self.eps, self.acc)
        return attention_prepare(qkv, cache, position, params, self.eps, self.acc)


class AttentionPrepareRounded(_Reference):
    """The attention preparation at the rounding points of a kernel that normalises in fp32, rounds the normalised value ONCE
    to bf16 and each rotated value once more, and writes the keys and values of a step's rows into the cache in one
    launch (``cudnn.AttentionPrepareDecode``).  The reference chain (``attention_prepare``) rounds a rotated value
    three times and takes its norm from the run's accumulation precision: another function.  This block is the twin of
    that entry's adapter, not a row of any table.  It works from the host integer the step hands it and writes through
    the cache's accessor, on either layout of the cache; it reads no table of the state, so a wrong position or slot row
    cannot move it.  ``order`` is the position form of the adapter's row; ``terms`` keeps of the last call what the
    band of the entry is judged with (``tie_out``: ``attention_prepare``)."""

    entry = "torch_ref.attention_prepare at the kernel's rounding points"
    per_row = (0, 1)
    state_args = (1,)
    order = "3d"
    terms = None

    def __call__(self, qkv, cache, position, params):
        q, gate, self.terms = attention_prepare_rounded(qkv, cache, position, getattr(cache, "tokens", 1), params, self.eps, self.order)
        return q, gate


class Attention(_Reference):
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY
    per_row = (0, 1)

    def __call__(self, q, cache, position):
        tokens = getattr(cache, "tokens", 1)
        if tokens > 1:
            return attention_tokens(q, cache, position, tokens, self.acc)
        return attention(q, cache, position, self.acc)


class AttentionBound(_Reference):
    """The attention core as the reference chain itself, which also keeps the operand terms of a bound: the twin of an
    adapter to a library attention kernel that documents no numerics contract and whose rounding points are not
    established.  No reference block can be written at rounding points nobody has stated, so this block returns the
    bytes of the reference chain (``attention`` / ``attention_tokens``, in the run's accumulation precision) and keeps
    of the last call what the band of such a kernel is judged with (``tie_out``: ``attention_bf16_weights``): ``bound`` =
    the chain of every element before its rounding to bf16 and the weighted magnitude of the values it attends to, both
    in fp64 (:func:`attention_bound`).  It works from the host integer the step hands it and reads the cache through
    its accessors, on either layout; it reads no table of the state, so a wrong length or page-table row cannot move
    it.  It writes nothing.  Not a row of any table."""

    entry = "torch_ref.attention with the operand terms of its bound"
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY
    per_row = (0, 1)
    bound = None

    def __call__(self, q, cache, position):
        tokens = getattr(cache, "tokens", 1)
        self.bound = attention_bound(q, cache, position, tokens)
        if tokens > 1:
            return attention_tokens(q, cache, position, tokens, self.acc)
        return attention(q, cache, position, self.acc)


class GateQuantize(_Reference):
    per_row = (0, 1)

    def __call__(self, o, gate, quant):
        return gate_quantize(o, gate, quant)


class GateRowQuantize(_Reference):
    per_row = (0, 1)

    def __call__(self, o, gate, quant):
        return gate_row_quantize(o, gate, quant)


class GateQuantizeQuotient(_Reference):
    """The gate + quantise at the rounding points of a kernel that forms the gated value as a quotient
    (``cudnn.AttentionGateQuantDecode``): ``code = e4m3(bf16(o / (1 + exp(-gate))) * (1 / scale))``, the quotient accumulated
    in ``acc`` and rounded ONCE to bf16.  The reference chain (``gate_quantize``) rounds the sigmoid to bf16 first and then
    the product: another function, one code apart on rare elements.  This block is the twin of that entry's adapter, not a
    row of any table; ``bracket`` keeps, of the last call, the codes of the other bf16 value around each quotient and the
    quotient's distance from the midpoint, which the band of the entry is judged with (``tie_out``: ``gate_fp8``)."""

    entry = "torch_ref.gate_quantize at the kernel's rounding points"
    per_row = (0, 1)
    bracket = None

    def __call__(self, o, gate, quant):
        if quant is None or quant.fmt != "fp8":
            raise ValueError("the gate's twin quantises to static 8-bit codes")
        nearest, other, margin = gate_quotient_bracket(o, gate, self.acc)
        self.bracket = (quantize_fp8(other, quant.scale).codes, margin)
        return quantize_fp8(nearest, quant.scale)


class GdnDecodeStep(_Reference):
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY
    per_row = (0, 1, 2, 3)
    state_args = (5,)

    tokens = 1  # per sequence, of the last call: a check names the state slots of a step of several by token and sequence
    #: set by the block check for ONE call, where this block is compared with a kernel: what that kernel's call left in
    #: the state the call updates (per state operand its tensors).  A step of several tokens then follows the kernel
    #: token by token (``gdn_verify_step(anchor=)``).  ``None`` everywhere else: the reference stack runs its own chain
    anchor = None

    def __call__(self, x, z, hidden, slot_ids, params, slots):
        anchor, self.anchor = self.anchor, None
        if isinstance(slot_ids, TokenSlots):  # a step of several tokens per sequence hands its slot table over: the verify step
            self.tokens = slot_ids.tokens
            pool = None if anchor is None else anchor[0][0]  # of the one state operand, its recurrent states
            return gdn_verify_step(x, z, hidden, slot_ids, params, slots, self.eps, self.acc, pool)
        self.tokens = 1
        return gdn_decode_step(x, z, hidden, slot_ids, params, slots, self.eps, self.acc)
