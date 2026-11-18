# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Providers of model blocks: adapters to cuDNN Frontend entries (``fe``), to library kernels (``library``) and the plain-PyTorch
reference of every block (``torch_ref``).

``KEYS`` is the closed list of provider keys a provider table may name; it changes only together with a kernel, or with the
reference chain of a block that a kernel is to serve.  The operand containers below are what the blocks of a family pass to
each other.  Nothing here imports a provider module, PyTorch or cuDNN Frontend until :func:`factories` is called.

Every adapter declares its census entries (``kernels``, :class:`model_samples.census.Kernel`): the kernels one call launches.
An adapter to an entry that states its launch also answers ``launch_facts(rows)`` and says what served its last call
(``serving()``: the entry instance, and the fields of the row's form as that instance states them), which a traced step writes
into its call log.  An adapter names the block the tie-out compares it with (``twin``).  An adapter whose row names several
call sites (``ProviderRow.sites``) holds one entry instance per form and serves each call with the instance of ITS site: the
family's step hands the site over as the block's last operand, and the adapter decides nothing.  An adapter that multiplies
on the stored format of its weight says so (``stored_weights``): the loader then keeps that format, and it keeps the bf16
copy of that weight only where another block of the run's plans reads one.  An adapter whose entry reads that weight in a
layout of its own also says how the layout is made (``stored_layout(merged)``, its entry's own load-time helper): the
loader then keeps the weight in that layout and in no other stored form.  An adapter to an entry of a step that feeds
several tokens per sequence takes the step's slot table where the decode step's adapter takes slot ids
(``state.TokenSlots``), and its row names the tokens per sequence as a field of the form (``TOKENS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from ..census import Kernel

#: provider key -> what is called, as the served-by table prints it
ENTRY_NAMES: Dict[str, str] = {
    "fe.gdn_decode_fused": "cudnn.GdnDecodeFused",
    "fe.gdn_verify_fused": "cudnn.GdnVerifyFused",
    "fe.attention_gate_quant_decode": "cudnn.AttentionGateQuantDecode",
    "fe.add_norm_quant_decode": "cudnn.AddNormQuantDecode",
    "fe.fp8_projection_decode": "cudnn.Fp8ProjectionDecode",
    "fe.swiglu_fp4_decode": "cudnn.SwigluFp4Decode",
    "cublas.bf16_dequantized": "torch.matmul over a bf16 copy",
    "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": "torch.matmul over a bf16 copy + torch_ref.silu_mul_quantize",
    "torch.index_select": "torch.index_select",
    "torch.argmax": "torch.argmax",
    "torch.argmax+lookup": "torch.argmax + the token id of the chosen row",
    "torch_ref.add_norm_quantize": "torch_ref.add_norm_quantize",
    "torch_ref.pre_fc_norm_quantize": "torch_ref.pre_fc_norm_quantize",
    "torch_ref.attention_prepare": "torch_ref.attention_prepare",
    "torch_ref.attention": "torch_ref.attention",
    "torch_ref.gate_quantize": "torch_ref.gate_quantize",
    "torch_ref.gate_row_quantize": "torch_ref.gate_row_quantize",
    "torch_ref.silu_mul": "torch_ref.silu_mul",
    "torch_ref.quantize": "torch_ref.quantize",
    "torch_ref.linear": "torch_ref.linear",
    "torch_ref.linear+torch_ref.silu_mul_quantize": "torch_ref.linear + torch_ref.silu_mul_quantize",
    "torch_ref.gdn_decode_step": "torch_ref.gdn_decode_step",
}
KEYS: Tuple[str, ...] = tuple(ENTRY_NAMES)


#: the format of a :class:`Quant` whose scale is formed per call: 8-bit codes with one scale per row of the call, each
#: formed from its row (``weights.quantize_fp8_per_row``)
FP8_ROW = "fp8_row"


@dataclass(frozen=True)
class Quant:
    """How a block quantises its output for its consumer: ``"fp8"`` (static scale), ``"nvfp4"`` (static global scale) or
    ``"fp8_row"`` (:data:`FP8_ROW`: 8-bit codes with one scale per row, formed from the rows of the call).  A static
    format states its scale; the row format has none to state.  ``ValueError`` for either without what it stands for."""

    fmt: str
    scale: Optional[float] = None

    def __post_init__(self):
        if (self.scale is None) != (self.fmt == FP8_ROW):
            has = "states no scale" if self.scale is None else f"states the scale {self.scale!r}"
            raise ValueError(f'a quantisation format with a static scale states it, and "{FP8_ROW}" forms its scale per call; "{self.fmt}" {has}')


#: the quantisation of an operand whose rows are each quantised at a scale of their own
ROW_QUANT = Quant(FP8_ROW)


@dataclass
class GdnParams:
    """The per-layer operands of a linear-attention (Gated DeltaNet) decode step, in the dtypes its kernel takes."""

    key_heads: int
    value_heads: int
    head_dim: int
    conv_weight: object  # [2 K D + V D, taps + 1] bf16
    a_log: object  # [V] fp32
    dt_bias: object  # [V] bf16
    norm_weight: object  # [D] bf16
    ba_weight: object  # [2 V, hidden] bf16: the beta rows first, then the alpha rows
    out_scale: object  # [1] fp32: the static 8-bit scale of the step's output (the out-projection's input scale)
    out_scale_value: float = 0.0  # the same scale as a Python float, so that no block reads the device for it

    @property
    def qkv_channels(self) -> int:
        return (2 * self.key_heads + self.value_heads) * self.head_dim


@dataclass
class AttentionParams:
    """The per-layer operands of an attention layer's preparation and core."""

    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    q_norm: object  # [D] bf16, stored as an offset: the norm multiplies by 1 + w
    k_norm: object  # [D] bf16


def split_qkvz(slab, qkv_channels: int, value_heads: int, head_dim: int):
    """The two operands a linear-attention step reads from its in-projection's output row ``[rows, qkv_channels + V D]``:
    ``x`` ``[rows, qkv_channels]`` and ``z`` ``[rows, V, D]``, both views that keep the slab's row stride."""
    return slab[:, :qkv_channels], slab[:, qkv_channels:].unflatten(1, (value_heads, head_dim))


#: the census entries of a chain of PyTorch operators, and of one that also multiplies through cuBLAS: fallback families
PYTORCH_CHAIN = (Kernel("pytorch", "PyTorch"),)
PYTORCH_CHAIN_WITH_MULTIPLY = (Kernel("pytorch", "PyTorch"), Kernel("cublas", "cuBLAS"))


class BlockBase:
    """Defaults of a block: stateless, one leading row per sequence on its first operand, nothing to build or reserve, no
    kernel declared and no entry that states its launch."""

    entry = ""
    provider = "PyTorch"
    kernels: Tuple[Kernel, ...] = ()
    per_row: Tuple[int, ...] = (0,)
    state_args: Tuple[int, ...] = ()
    #: whether the block multiplies on the stored format of its weight (8-bit codes with their scale; packed 4-bit codes with
    #: their block scales and global scale).  A session's load then keeps that format for the block
    #: (``weights.Dequantized.stored``) -- and no bf16 copy of the weight where every plan of the run has such a block for
    #: it -- and its memory sentence counts what is kept.  A block that does not say so reads the bf16 copy
    stored_weights = False
    #: for a block that multiplies on the stored format in a layout of its entry's own: ``stored_layout(merged)`` returns the
    #: merged stored tensor in that layout, built once per weight at load; the loader keeps what it returns.  ``None``: the
    #: block multiplies on the merged tensor as the checkpoint stores it
    stored_layout = None
    #: for a block whose entry states that one sequence alone is NOT byte-equal to the same sequence among several: that
    #: statement, as a run prints it beside the batch-invariance line and wherever it compares such a block across row
    #: counts.  Empty: the entry states no such difference
    batch_note = ""

    def __init__(self, row, model, facts, options):
        self.row, self.model, self.facts, self.options = row, model, facts, options
        self.entry = self.entry or ENTRY_NAMES[row.key]

    def check_support(self, sequences: int, tokens: int) -> None:
        return None

    def first_use(self, sequences: int, tokens: int) -> None:
        return None

    def compile(self, sequences: int, tokens: int) -> None:
        return None

    def reserve_bytes(self) -> int:
        return 0

    def first_launch(self) -> dict:
        """Rows -> device bytes the first launch of that form took outside PyTorch's allocator in this process, for a block
        that measures it; empty otherwise."""
        return {}

    def compared(self, out):
        """What the tie-out compares of a call's result against the reference block: the result itself, unless the block
        ends in a chain it shares with the reference (then the tensor that enters that chain)."""
        return out

    def twin(self, reference):
        """The block the tie-out compares this block with, given the reference stack's block of the same name: that block,
        unless the provider's kernel rounds at other points than the reference chain -- then a reference block written at
        the kernel's rounding points.  The reference is matched to the kernel's documented arithmetic; a band is never
        widened to cover a chain that rounds elsewhere.  The reference stack itself never asks."""
        return reference

    def serving(self) -> Tuple[str, tuple]:
        """``(entry instance, form)`` of what served the last call of this block, for the call log of a traced step.  This
        block has no entry: no instance, and nothing was passed.  (So a row that names a form for it fails the kernel
        check: nothing carries that form.)  An adapter to an entry answers from the object it called."""
        return "", ()

    def launch_facts(self, rows: int, site: str = ""):
        """What the block's entry states about its launch at ``rows`` (at the call site ``site``, for a row whose entry is
        called in several forms); ``None``: this block has no entry that states one."""
        return None


def factories() -> Dict[str, Callable]:
    """Provider key -> ``factory(row, model, facts, options) -> Block``.  Imports the provider modules (and with them PyTorch);
    cuDNN Frontend is imported only when a cuDNN Frontend block is asked whether it serves or is compiled."""
    from . import fe, library, torch_ref

    return {
        "fe.gdn_decode_fused": fe.GdnDecodeFusedBlock,
        "fe.gdn_verify_fused": fe.GdnVerifyFusedBlock,
        "fe.attention_gate_quant_decode": fe.AttentionGateQuantDecodeBlock,
        "fe.add_norm_quant_decode": fe.AddNormQuantDecodeBlock,
        "fe.fp8_projection_decode": fe.Fp8ProjectionDecodeBlock,
        "fe.swiglu_fp4_decode": fe.SwigluFp4DecodeBlock,
        "cublas.bf16_dequantized": library.CublasLinear,
        "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": library.CublasGateUp,
        "torch.index_select": torch_ref.Embedding,
        "torch.argmax": torch_ref.TokenChoice,
        "torch.argmax+lookup": torch_ref.ListedTokenChoice,
        "torch_ref.add_norm_quantize": torch_ref.AddNormQuantize,
        "torch_ref.pre_fc_norm_quantize": torch_ref.PreFcNormQuantize,
        "torch_ref.attention_prepare": torch_ref.AttentionPrepare,
        "torch_ref.attention": torch_ref.Attention,
        "torch_ref.gate_quantize": torch_ref.GateQuantize,
        "torch_ref.gate_row_quantize": torch_ref.GateRowQuantize,
        "torch_ref.silu_mul": torch_ref.SiluMul,
        "torch_ref.quantize": torch_ref.Quantize,
        "torch_ref.linear": torch_ref.Linear,
        "torch_ref.linear+torch_ref.silu_mul_quantize": torch_ref.GateUp,
        "torch_ref.gdn_decode_step": torch_ref.GdnDecodeStep,
    }


def acc_dtype(options: Optional[dict]):
    """The accumulation dtype of the reference blocks: fp64 unless ``reference_precision`` is ``"ieee-fp32"``."""
    import torch

    precision = (options or {}).get("reference_precision", "fp64")
    if precision not in ("fp64", "ieee-fp32"):
        raise ValueError(f'reference precision must be "fp64" or "ieee-fp32"; got {precision!r}')
    return torch.float64 if precision == "fp64" else torch.float32
