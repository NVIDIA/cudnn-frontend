# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every block of a model in plain PyTorch: the reference of the tie-out, and the provider of the blocks that have no kernel yet.

The functions keep the quantisation rules and every intermediate cast of a serving chain (a norm output, a SiLU, a gate, the
conv taps and the core output of the linear-attention step are each rounded to bf16 where the chain rounds them), and they
accumulate every multiply and every reduction in ``acc`` -- fp64 by default.  fp64 is deliberate: an fp32 reference changes with
the matmul precision mode of the environment, and a small accumulation error can cross a bf16 rounding midpoint.

A block that runs here is not fast and is not meant to be: one launch per PyTorch operator, temporaries allocated per call.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..census import Kernel
from ..sampling import greedy
from ..weights import Dequantized, ScaledTensor, _e4m3, dequantize, quantize_fp8, quantize_nvfp4
from . import PYTORCH_CHAIN, PYTORCH_CHAIN_WITH_MULTIPLY, BlockBase, acc_dtype

BF16 = torch.bfloat16


def quantize(y: torch.Tensor, quant):
    """The quantised operand of a consumer, from a bf16 activation: static 8-bit or 4-bit with a static global scale."""
    if quant is None:
        return None
    if quant.fmt == "fp8":
        return quantize_fp8(y, quant.scale)
    if quant.fmt == "nvfp4":
        return quantize_nvfp4(y, quant.scale)
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
    to the result, one rounding to bf16.  Chunked over the weight's rows so the wide copy is never held twice in ``acc``."""
    x = dequantize(qx)
    xa = x.data.to(acc)
    out = torch.empty((xa.shape[0], weight.data.shape[0]), dtype=acc, device=xa.device)
    for a in range(0, weight.data.shape[0], columns_per_chunk):
        out[:, a : a + columns_per_chunk] = xa @ weight.data[a : a + columns_per_chunk].to(acc).T
    scale = weight.scale.to(acc) * x.scale if isinstance(weight.scale, torch.Tensor) else float(weight.scale) * float(x.scale)
    return (out * scale).to(BF16)


def silu_mul_quantize(gate_up: torch.Tensor, quant):
    """``SiLU(gate) * up`` on the two halves of a merged gate/up output, with the chain's two bf16 roundings, then quantise."""
    half = gate_up.shape[-1] // 2
    act = (F.silu(gate_up[:, :half].float()).to(BF16).float() * gate_up[:, half:].float()).to(BF16)
    return quantize(act, quant)


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
    value at ``position`` of the layer's cache.  The row is ``[q | gate]`` per query head, then the keys, then the values.
    Returns ``(q [rows, heads, D] bf16, gate [rows, heads * D] bf16)``."""
    rows, H, HK, D = qkv.shape[0], p.heads, p.kv_heads, p.head_dim
    q_gate = qkv[:, : 2 * H * D].view(rows, H, 2 * D)
    q, gate = q_gate[..., :D], q_gate[..., D:]
    k = qkv[:, 2 * H * D : (2 * H + HK) * D].view(rows, HK, D)
    v = qkv[:, (2 * H + HK) * D :].view(rows, HK, D)
    cos, sin = rope_cos_sin(position, p.rotary_dim, p.rope_theta, qkv.device)
    q = rotate(rms_norm(q.float(), p.q_norm, eps, acc), cos, sin)
    k = rotate(rms_norm(k.float(), p.k_norm, eps, acc), cos, sin)
    cache.k[:, :, position] = k
    cache.v[:, :, position] = v
    return q, gate.reshape(rows, H * D)


def attention(q: torch.Tensor, cache, position: int, acc) -> torch.Tensor:
    """Scaled dot-product attention of one new token per sequence over the cached keys and values up to ``position``; every
    query head reads the key/value head of its group.  Returns ``[rows, heads * D]`` bf16."""
    rows, H, D = q.shape
    HK = cache.k.shape[1]
    k = cache.k[:, :, : position + 1].to(acc)
    v = cache.v[:, :, : position + 1].to(acc)
    qg = q.to(acc).view(rows, HK, H // HK, D)
    p = torch.softmax(torch.einsum("bhgd,bhtd->bhgt", qg, k) * D**-0.5, dim=-1)
    return torch.einsum("bhgt,bhtd->bhgd", p, v).reshape(rows, H * D).to(BF16)


def gate_quantize(o: torch.Tensor, gate: torch.Tensor, quant):
    """``sigmoid(gate) * attention output`` with the chain's bf16 roundings, then the static 8-bit quantisation."""
    return quantize((o.float() * torch.sigmoid(gate.float()).to(BF16).float()).to(BF16), quant)


#: the largest fp32 value: an fp32 exponential beyond it is +inf, and the value it divides is then +-0
FP32_MAX = 3.4028234663852886e38


def _bf16_bits(v: torch.Tensor) -> torch.Tensor:
    return v.view(torch.int16).to(torch.int32) & 0xFFFF


def _bf16_from_bits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int16).view(BF16)


def gate_quotient_bracket(o: torch.Tensor, gate: torch.Tensor, acc):
    """The gated attention output formed as the QUOTIENT ``o / (1 + exp(-gate))``, accumulated in ``acc`` on the bf16 operands,
    with its bf16 bracket: ``(nearest, other, margin)``.  ``nearest`` is the round-to-nearest-even bf16 of the quotient,
    decided by comparing the quotient with the midpoint of the two bf16 values around it (never by a cast that rounds
    twice); ``other`` is the bf16 value on the other side of the quotient (``nearest`` itself where the quotient is a bf16
    value); ``margin`` is the distance of the quotient from the midpoint of the two, in fp32 units of the quotient (+inf
    where it is a bf16 value).  A kernel that forms the same quotient in fp32 rounds to ``nearest`` wherever the margin
    exceeds its own fp32 error, and to one of the two everywhere.  The exponential is formed as the kernel forms it: where
    it overflows fp32 it is +inf and the gated value +-0, whatever ``acc`` could still represent."""
    exponential = torch.exp(-gate.to(acc))
    exponential = torch.where(exponential > FP32_MAX, torch.full_like(exponential, float("inf")), exponential)
    t = o.to(acc) / (1.0 + exponential)
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
    # ``v`` unless the other end is strictly nearer.  A quotient ON the midpoint is exact in fp32, so the cast above has
    # already rounded it to the even end: at equal distances ``v`` is the nearest-even value
    nearest = torch.where(exact | (to_v <= to_other), v, other)
    exponent = torch.frexp(t.abs()).exponent.clamp(min=-125)
    unit = torch.pow(torch.tensor(2.0, dtype=acc, device=t.device), (exponent - 24).to(acc))  # one fp32 unit of the quotient
    margin = torch.where(exact, torch.full_like(t, float("inf")), (t - (va + oa) * 0.5).abs() / unit)
    return nearest, other, margin


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


class AddNormQuantize(_Reference):
    per_row = (0, 1)

    def __call__(self, branch, residual, weight, quant):
        return add_norm_quantize(branch, residual, weight, quant, self.eps, self.acc)


class Linear(_Reference):
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY

    def __call__(self, qx, weight):
        return linear(qx, weight, self.acc)


class GateUp(_Reference):
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY

    def __call__(self, qx, weight, quant):
        self.product = linear(qx, weight, self.acc)
        return silu_mul_quantize(self.product, quant)

    def compared(self, out):
        return self.product


class AttentionPrepare(_Reference):
    per_row = (0, 1)
    state_args = (1,)

    def __call__(self, qkv, cache, position, params):
        return attention_prepare(qkv, cache, position, params, self.eps, self.acc)


class Attention(_Reference):
    kernels = PYTORCH_CHAIN_WITH_MULTIPLY
    per_row = (0, 1)

    def __call__(self, q, cache, position):
        return attention(q, cache, position, self.acc)


class GateQuantize(_Reference):
    per_row = (0, 1)

    def __call__(self, o, gate, quant):
        return gate_quantize(o, gate, quant)


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

    def __call__(self, x, z, hidden, slot_ids, params, slots):
        return gdn_decode_step(x, z, hidden, slot_ids, params, slots, self.eps, self.acc)
