# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one compiled gemm needs per call, settled when it compiles.

Operand roles, majors, packing factors, alignment requirements, output shape
rules and which outputs are reductions are all fixed by the time cute hands back
a launchable. A call carries M, N, K, the strides and the pointers, and nothing
else. This module writes the first set down, so that neither the interpreted
path nor the straight line lowered from it re-derives them per call.

The two consumers read the same recipe but do not share a body: :func:`gate`
interprets it, and ``CompiledFusedGemm._lower`` emits a straight line with the
constants inlined. That is a compiler beside its interpreter, kept honest the
way those always are -- ``test_execute_recipe.py`` runs both over the same
accepts and rejects and requires the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dtypes import DTYPE_BYTES, _aux_align_reqs, _output_align_reqs, _pow2_floor, tensor_alignment
from .fusion_ir import FusionChain

# An operand's three axes, named by role rather than by position.
AX_BATCH, AX_MN, AX_K = 0, 1, 2

# cuDNN's matmul ABI declares A as [b, m, k] and B as [b, k, n], while a caller
# allocates B the way the kernel reads it -- (b, n, k). Both describe the same
# memory, so an operand arrives in one of exactly two axis orders and its
# strides alone do not say which: a graph-order K-major B and a caller-order
# N-major B carry stride 1 on the same axis. Its SHAPE does say, since only a
# buffer described from the graph's own declaration carries the declared dims.
_DECLARED_AXES = {"a": (0, 1, 2), "b": (0, 2, 1)}
KERNEL_AXES = (0, 1, 2)

# How an output axis follows from the problem size.
CONST, FROM_M, FROM_N = 0, 1, 2

REDUCTION_INIT_VALUE = {
    "fp32": {
        "add": 0.0,
        "amax": 0.0,
        "max": -float("inf"),
        "min": float("inf"),
        "avg": 0.0,
        "norm1": 0.0,
        "norm2": 0.0,
        "mul": 1.0,
        "mul_no_zeros": 1.0,
    },
    "int32": {
        "add": 0,
        "amax": 0,
        "max": -(2**31),
        "min": 2**31 - 1,
        "norm1": 0,
    },
}


def contiguous_modulus(dtype: str, contiguous_is_k: bool) -> tuple:
    """``(stored_modulus, packing)`` for one operand's TMA 16-byte rule.

    TMA encodes the contiguous dimension in 16-byte units, so the LOGICAL extent
    must divide ``128 // bits``. A shape carries the STORED count, and fp4 packs
    two elements per slot along K -- so a buffer is checked against the quotient
    while a user wrote their shape against the product. Both the graph-time gate
    and the per-call one read this, which is the one number that could drift.
    """
    bits = 4 if dtype == "fp4_e2m1" else DTYPE_BYTES[dtype] * 8
    pack = 2 if (dtype == "fp4_e2m1" and contiguous_is_k) else 1
    return (128 // bits) // pack, pack


def expected_shape(rule, m: int, n: int) -> tuple:
    """One output's runtime shape, from the rule the build recorded."""
    return tuple(v if s == CONST else (m if s == FROM_M else n // v) for s, v in rule)


def _output_rule(spec, chain: FusionChain) -> tuple:
    """How one output's shape follows from (M, N), as a per-axis rule.

    The single answer to a question that used to be asked in two places and
    disagreed: an fp4 dense output is (batch, M, N/2), and a hand-written launch
    path that pinned N instead rejected a legal call.
    """
    batch = chain.matmul.batch
    if spec.is_quant_scale:
        return tuple((CONST, int(d)) for d in spec.dim)
    if not spec.is_reduction:
        return ((CONST, batch), (FROM_M, 0), (FROM_N, 2 if spec.dtype == "fp4_e2m1" else 1))
    red_idx = int(spec.source.rsplit("_", 1)[1])
    if chain.reductions[red_idx].grouped_by_moe:
        return tuple((CONST, int(d)) for d in spec.dim)
    # A reduced axis collapses to 1; the rest follow the problem size.
    return (
        (CONST, 1 if spec.dim[0] == 1 else batch),
        (CONST, 1) if spec.dim[1] == 1 else (FROM_M, 0),
        (CONST, 1) if spec.dim[2] == 1 else (FROM_N, 1),
    )


@dataclass(frozen=True)
class Operand:
    """One A or B operand: what a call must satisfy, with the rest baked."""

    view: int  # position in the view list the engine hands over
    role: str  # names this operand in a rejection message
    major: str
    kpack: int  # 2 when fp4 stores two elements per slot
    batch: int  # the extent the graph declares on the batch axis
    is_b: bool  # its non-K extent is N rather than M
    modulus: int  # the contiguous STORED extent must divide this
    pack: int  # ... and multiplying by this recovers the logical extent
    contiguous_role: int  # AX_K for a k-major operand, else AX_MN
    declared: tuple  # (batch, mn, k) axis positions, graph order
    declared_dim: tuple  # the extents that order comes with
    dc: int  # where stride 1 lands in the graph's order
    kc: int  # ... and in the caller's

    def axes(self, shape, stride) -> "tuple | None":
        """Which axis holds which role, or None if neither order fits."""
        if stride[self.kc] == 1:
            return KERNEL_AXES
        if stride[self.dc] == 1 and tuple(shape) == self.declared_dim:
            return self.declared
        return None


@dataclass(frozen=True)
class Output:
    view: int
    role: str
    rule: tuple
    align: int
    raw: bool  # the kernel takes only its pointer
    init: Any = None  # reduction identity, seeded before the kernel runs
    sqrt: bool = False  # norm2 takes a square root after


@dataclass(frozen=True)
class Aux:
    view: int
    role: str
    align: int
    ref: Any  # TensorRef, for the fake-shape reshape


@dataclass(frozen=True)
class ScaleFactor:
    view: int
    role: str
    is_a: bool
    operand_at: int  # the A or B operand this one scales, as an index into inputs


@dataclass(frozen=True)
class GemmRecipe:
    inputs: tuple  # every A operand then every B operand, in kernel slot order
    outputs: tuple
    aux: tuple
    sf: tuple
    roles: tuple  # what occupies each view position, for a missing-buffer message
    a_at: int  # M and K are read off inputs[a_at]
    b_at: int  # N off inputs[b_at]
    has_output_specs: bool
    block_size: "int | None"
    workspace_bytes: int
    multi_gemm: bool

    @property
    def a(self) -> Operand:
        return self.inputs[self.a_at]

    @property
    def b(self) -> Operand:
        return self.inputs[self.b_at]

    def problem(self, views) -> tuple:
        """``((M, N, K), axes-per-input)``, located rather than assumed.

        Reading M off axis 1 assumes the caller laid the buffer out the way the
        kernel reads it, which a bare device address does not: the pack
        describes that one from the graph's declaration, which orders B the
        other way round.
        """
        axes, bad = [], []
        for op in self.inputs:
            v = views[op.view]
            ax = op.axes(v.shape, v.stride())
            if ax is None:
                bad.append(
                    f"{op.role}: graph declares {op.major}-major (dim {op.kc} contiguous) but the buffer has shape={tuple(v.shape)}, stride={tuple(v.stride())}"
                )
                ax = KERNEL_AXES
            axes.append(ax)
        if bad:
            raise ValueError("runtime operand layout does not match the layout the kernel was compiled for: " + "; ".join(bad))
        a, b = self.a, self.b
        a_ax, b_ax = axes[self.a_at], axes[self.b_at]
        a_shape, b_shape = views[a.view].shape, views[b.view].shape
        return (a_shape[a_ax[AX_MN]], b_shape[b_ax[AX_MN]], a_shape[a_ax[AX_K]] * a.kpack), tuple(axes)


def _tma_reject(recipe: GemmRecipe, views, axes) -> "str | None":
    """TMA encodes the contiguous dimension in 16-byte units; a misaligned
    extent silently mis-strides every row past the first."""
    bad = []
    for op, ax in zip(recipe.inputs, axes):
        role = op.contiguous_role
        extent = views[op.view].shape[ax[role]]
        if extent % op.modulus:
            # Report the LOGICAL extent and modulus: fp4 stores two elements per
            # slot, so "K % 32" is the rule a user wrote their shape against.
            name = "K" if role == AX_K else ("N" if op.is_b else "M")
            bad.append(f"{op.role} ({op.major}-major) requires {name} % {op.modulus * op.pack} == 0, got {name}={extent * op.pack}")
    if bad:
        return "TMA input contiguous dimensions must be 16-byte aligned: " + "; ".join(bad)
    return None


def _shape_reject(recipe: GemmRecipe, views, axes, mnk) -> "str | None":
    """Every operand must agree with the M/N/K read off the first A and B --
    the kernel walks the inferred K on all of them."""
    m, n, k = mnk
    bad = []
    for op, ax in zip(recipe.inputs, axes):
        shape = views[op.view].shape
        want = (op.batch, n if op.is_b else m, k // op.kpack)
        got = (shape[ax[AX_BATCH]], shape[ax[AX_MN]], shape[ax[AX_K]])
        if got != want:
            bad.append(f"{op.role}: expected (batch, M|N, K) = {want}, got {got}")
    if bad:
        return f"runtime operand shapes disagree with the inferred problem size (M={m}, N={n}, K={k}): " + "; ".join(bad)
    return None


def _output_shape_reject(recipe: GemmRecipe, views, mnk) -> "str | None":
    m, n, _ = mnk
    bad = []
    for out in recipe.outputs:
        shape = tuple(views[out.view].shape)
        want = expected_shape(out.rule, m, n)
        if shape != want:
            bad.append(f"{out.role}: expected {want}, got {shape}")
    if bad:
        return "runtime tensors must be rank-3 with shapes matching the graph: " + "; ".join(bad)
    return None


def _align_reject(recipe: GemmRecipe, views) -> "str | None":
    """Every buffer's alignment must meet the width its role's access uses.

    Inputs and scale factors are TMA-loaded, so only the base pointer is at
    stake; outputs and aux are stored and loaded directly, so their strides and
    contiguous extent bound the vector too.
    """
    bad = []
    for item in recipe.inputs + recipe.sf:
        ptr = int(views[item.view].data_ptr())
        align = _pow2_floor(ptr)
        if align < 16:
            bad.append(f"{item.role}: alignment {align}B < required 16B (ptr=0x{ptr:x})")
    for item in recipe.outputs + recipe.aux:
        v = views[item.view]
        ptr = int(v.data_ptr())
        align = tensor_alignment(tuple(v.shape), tuple(v.stride()), v.element_size(), ptr=ptr)
        if align < item.align:
            bad.append(f"{item.role}: alignment {align}B < required {item.align}B (ptr=0x{ptr:x})")
    if bad:
        return "runtime tensor alignment is below the kernel's compiled requirement: " + "; ".join(bad)
    return None


def _sf_blob_reject(recipe: GemmRecipe, views, axes, mnk) -> "str | None":
    """A block-scale SF reaches the kernel as a base pointer plus a layout the
    template re-synthesizes from M/N/K, so a blob that is not one dense byte run
    of at least the required size is read out of bounds with no fault."""
    m, n, k = mnk
    k4 = ((k // recipe.block_size) + 3) // 4
    bad = []
    for sf in recipe.sf:
        v = views[sf.view]
        op = recipe.inputs[sf.operand_at]
        rows = m if sf.is_a else n
        batch = views[op.view].shape[axes[sf.operand_at][AX_BATCH]]
        required = 512 * k4 * ((rows + 127) // 128) * int(batch)
        span = 1 + sum((int(s) - 1) * int(st) for s, st in zip(v.shape, v.stride()))
        if int(v.numel()) != span:
            bad.append(f"{sf.role} shape {tuple(v.shape)} stride {tuple(v.stride())} is not a dense byte run")
            continue
        have = int(v.numel()) * v.element_size()
        if have < required:
            bad.append(
                f"{sf.role} is {have}B but the kernel reads {required}B ({required // 512} atoms of 128 rows x 4 SF-K) — was it produced by to_blocked()?"
            )
    if bad:
        return "block-scale F8_128x4 scale factors must be a packed blob: " + "; ".join(bad)
    return None


def gate(recipe: GemmRecipe, views, mnk, axes) -> None:
    """Raise on anything about this call the compiled kernel cannot serve."""
    reasons = [
        _tma_reject(recipe, views, axes),
        _shape_reject(recipe, views, axes, mnk),
        _output_shape_reject(recipe, views, mnk),
        _align_reject(recipe, views),
    ]
    if recipe.block_size:
        reasons.append(_sf_blob_reject(recipe, views, axes, mnk))
    for reason in reasons:
        if reason is not None:
            raise ValueError(reason)


def _declared_dim(tensor) -> tuple:
    """The dims the graph declared, or ``()`` when the tensor cannot say.

    An empty tuple never matches a runtime shape, so an operand that cannot
    report its declaration is simply read in the caller's order -- which is
    what every path did before this table existed.
    """
    try:
        return tuple(int(d) for d in tensor.get_dim())
    except Exception:  # noqa: BLE001 — an analyzer-synthesized ref has no dims
        return ()


def _operand(view: int, role: str, tensor, *, major: str, dtype: str, batch: int, is_b: bool) -> Operand:
    declared = _DECLARED_AXES["b" if is_b else "a"]
    contiguous = AX_K if major == "k" else AX_MN
    modulus, pack = contiguous_modulus(dtype, contiguous == AX_K)
    return Operand(
        view=view,
        role=role,
        major=major,
        kpack=2 if dtype == "fp4_e2m1" else 1,
        batch=batch,
        is_b=is_b,
        modulus=modulus,
        pack=pack,
        contiguous_role=contiguous,
        declared=declared,
        declared_dim=_declared_dim(tensor),
        dc=declared[contiguous],
        kc=KERNEL_AXES[contiguous],
    )


def build(compiled) -> GemmRecipe:
    """Read one compiled gemm into the table its call path runs off."""
    chain: FusionChain = compiled.chain
    mm = chain.matmul
    binding = compiled.binding
    order = {}
    for i, t in enumerate(binding.bound_tensors()):
        order.setdefault(id(t), i)

    inputs = [
        _operand(order[id(t)], f"A operand[{i}]", t, major=mm.a_major, dtype=mm.a_dtype, batch=mm.a_batch, is_b=False) for i, t in enumerate(binding.a_operands)
    ] + [
        _operand(order[id(t)], f"B operand[{i}]", t, major=mm.b_major, dtype=mm.b_dtype, batch=mm.b_batch, is_b=True) for i, t in enumerate(binding.b_operands)
    ]

    out_reqs = _output_align_reqs(chain, compiled.use_tma_store, vec_bytes=compiled.vec_bytes_epi)
    aux_reqs = _aux_align_reqs(chain, vec_bytes=compiled.vec_bytes_epi)
    outputs = []
    for i, (spec, t) in enumerate(zip(chain.outputs, binding.outputs)):
        init, sqrt = None, False
        if spec.is_reduction:
            red = chain.reductions[int(spec.source.rsplit("_", 1)[1])]
            init = REDUCTION_INIT_VALUE[red.compute_dtype][red.mode]
            sqrt = red.mode == "norm2"
        outputs.append(
            Output(
                view=order[id(t)],
                role=spec.source,
                rule=_output_rule(spec, chain),
                align=out_reqs[i],
                raw=bool(spec.is_reduction or spec.is_quant_scale),
                init=init,
                sqrt=sqrt,
            )
        )

    aux = tuple(
        Aux(view=order[id(t)], role=f"aux {compiled.aux_names[i]!r}", align=aux_reqs[compiled.aux_names[i]], ref=ref)
        for i, (t, ref) in enumerate(zip(binding.aux, chain.aux_tensors))
    )
    na = len(binding.a_operands)
    sf = tuple(
        [ScaleFactor(view=order[id(t)], role=f"SFA[{i}]", is_a=True, operand_at=i) for i, t in enumerate(binding.sfa_operands)]
        + [ScaleFactor(view=order[id(t)], role=f"SFB[{j}]", is_a=False, operand_at=na + j) for j, t in enumerate(binding.sfb_operands)]
    )

    roles = [f"bound tensor {i}" for i in range(len(binding.bound_tensors()))]
    for item in (*inputs, *outputs, *aux, *sf):
        roles[item.view] = item.role

    return GemmRecipe(
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        aux=aux,
        sf=sf,
        roles=tuple(roles),
        a_at=0,
        b_at=na,
        has_output_specs=bool(chain.output_specs),
        block_size=chain.block_scale.block_size if compiled.block_scale else None,
        workspace_bytes=int(getattr(compiled, "workspace_bytes", 0) or 0),
        multi_gemm=bool(chain.is_multi_gemm),
    )
