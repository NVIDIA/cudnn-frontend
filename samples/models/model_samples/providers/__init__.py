# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Providers of model blocks: adapters to cuDNN Frontend entries (``fe``), to library kernels (``library``) and the plain-PyTorch
reference of every block (``torch_ref``).

``KEYS`` is the closed list of provider keys a provider table may name; it changes only together with a kernel.  The operand
containers below are what the blocks of a family pass to each other.  Nothing here imports a provider module, PyTorch or
cuDNN Frontend until :func:`factories` is called.

Every adapter declares its census entries (``kernels``, :class:`model_samples.census.Kernel`): the kernels one call launches.
An adapter to an entry that states its launch also answers ``launch_facts(rows)`` and says what served its last call
(``serving()``: the entry instance, and the fields of the row's form as that instance states them), which a traced step writes
into its call log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from ..census import Kernel

#: provider key -> what is called, as the served-by table prints it
ENTRY_NAMES: Dict[str, str] = {
    "fe.gdn_decode_fused": "cudnn.GdnDecodeFused",
    "cublas.bf16_dequantized": "torch.matmul over a bf16 copy",
    "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": "torch.matmul over a bf16 copy + torch_ref.silu_mul_quantize",
    "torch.index_select": "torch.index_select",
    "torch.argmax": "torch.argmax",
    "torch_ref.add_norm_quantize": "torch_ref.add_norm_quantize",
    "torch_ref.attention_prepare": "torch_ref.attention_prepare",
    "torch_ref.attention": "torch_ref.attention",
    "torch_ref.gate_quantize": "torch_ref.gate_quantize",
    "torch_ref.linear": "torch_ref.linear",
    "torch_ref.linear+torch_ref.silu_mul_quantize": "torch_ref.linear + torch_ref.silu_mul_quantize",
    "torch_ref.gdn_decode_step": "torch_ref.gdn_decode_step",
}
KEYS: Tuple[str, ...] = tuple(ENTRY_NAMES)


@dataclass(frozen=True)
class Quant:
    """How a block quantises its output for its consumer: ``"fp8"`` (static scale) or ``"nvfp4"`` (static global scale)."""

    fmt: str
    scale: float


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

    def __init__(self, row, model, facts, options):
        self.row, self.model, self.facts, self.options = row, model, facts, options
        self.entry = self.entry or ENTRY_NAMES[row.key]

    def check_support(self, sequences: int, tokens: int) -> None:
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

    def serving(self) -> Tuple[str, tuple]:
        """``(entry instance, form)`` of what served the last call of this block, for the call log of a traced step.  This
        block has no entry: no instance, and nothing was passed.  (So a row that names a form for it fails the kernel
        check: nothing carries that form.)  An adapter to an entry answers from the object it called."""
        return "", ()

    def launch_facts(self, rows: int):
        """What the block's entry states about its launch at ``rows``; ``None``: this block has no entry that states one."""
        return None


def factories() -> Dict[str, Callable]:
    """Provider key -> ``factory(row, model, facts, options) -> Block``.  Imports the provider modules (and with them PyTorch);
    cuDNN Frontend is imported only when a cuDNN Frontend block is asked whether it serves or is compiled."""
    from . import fe, library, torch_ref

    return {
        "fe.gdn_decode_fused": fe.GdnDecodeFusedBlock,
        "cublas.bf16_dequantized": library.CublasLinear,
        "cublas.bf16_dequantized+torch_ref.silu_mul_quantize": library.CublasGateUp,
        "torch.index_select": torch_ref.Embedding,
        "torch.argmax": torch_ref.TokenChoice,
        "torch_ref.add_norm_quantize": torch_ref.AddNormQuantize,
        "torch_ref.attention_prepare": torch_ref.AttentionPrepare,
        "torch_ref.attention": torch_ref.Attention,
        "torch_ref.gate_quantize": torch_ref.GateQuantize,
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
