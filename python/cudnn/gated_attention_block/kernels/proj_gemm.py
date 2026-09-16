# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages (1) and (6) of the gated attention block: the two dense projections.

**This writes no kernel.** Both projections are ordinary dense ``nn.Linear``-shaped
GEMMs, and this repo already ships a production FROST one —
``gemm/frost/kernel_templates/sm100_matmul.py``: persistent, double-TMEM, CLC
dynamic scheduler, ``cta_group`` 1 or 2, a tuned tile-config catalog and a
codegen'd epilogue, behind an engine that competes in the graph's ranked plan
list. Writing a second tcgen05 GEMM to do the same thing would be weeks of work
and a second thing to maintain.

So this module is a thin driver: build the two-operand matmul graph, pin the
``frost_gemm`` plan, and hand it the block's buffers.

    stage (1)   h        [M, d_model]  @ W_qkvg^T  ->  [M, N_qkvg]
    stage (6)   O_gated  [M, H_q*D]    @ W_o^T     ->  [M, d_model]

with ``M = B*S``. One code path serves both: they are the same op at different
shapes, which is why there is one ``_Projection`` stage in ``api.py`` and not
two kernels.

Operand layout follows the graph API's matmul convention, and it happens to be
exactly what a checkpoint already holds: ``A`` is ``[1, M, K]`` row-major, and
``B`` is declared ``[1, K, N]`` with stride ``[K*N, 1, K]`` — i.e. the weight
stored ``[N, K]`` row-major (``nn.Linear``), read transposed. No repacking, at
plan time or execute time.

**When this gets FORKED — and the fork already has prior art in this repo.**
``sdpa/bwd/kernels/bprop_matmul_sm100.py`` forked this same template to give its
stage-3 gradient GEMMs a 2-D ``(batch, head)`` batch: it takes a **rendered
dense-bf16 expansion** of the template and widens the TMA descriptors to 4-D
``[k, m, h, b]``, keeping the mainloop, the CLC scheduler, the TMEM pipeline and
the epilogue identical. It carries a bidirectional "apply fixes both ways" note.

**Start our fork from that one, not from the template and not from scratch**, for
two things the unforked engine cannot express:

* **the four-output-buffer epilogue** (§ 1 of ``api.py``): stage (1)'s N axis is
  ``[Q | GATE | K | V]`` and each block wants its own compact ``[B, S, H, D]``
  destination. That is a per-N-tile choice among several output TMA descriptors
  — the same *kind* of descriptor surgery the backward's fork already did, and
  its 4-D output descriptors are close to the shape Q/K/V want.
* **the quantization epilogue** (FP8/MXFP8): Q/K rowwise, V columnwise, GATE
  passthrough — three scale-factor layouts off one GEMM. Note V's MXFP8 scale
  factors are D-PLANE-major while Q/K's are per-tile contiguous
  (``mma-tma-matrix.md`` § 7); reusing one SF descriptor builder for all three
  is a silent wrong answer that only shows up at ``d > 128``.

Two process rules that apply the moment the fork starts, both learned the hard
way here: **diff the two RENDERINGS, not your edit** (a constants swap that
misses half a descriptor pair produces ``cos ~= 0.006``, not a crash —
``frost-tile-dsl.md`` § 5), and add the matching cross-reference note in BOTH
files so a fix in one is not silently lost in the other.

Until then the unforked engine is the honest baseline, and any fork starts by
measuring against it — which is what this module exists to make cheap.

**Rubin (sm_107) perf note.** The tile catalog already sizes its K-pipeline from
the LIVE device budget (``tile_config._sm_smem_budget_bytes_of`` takes
``max(optin, oversized)``), and on Rubin that is **327 KiB, not SM100's 227** —
so deeper staging is available to the unforked engine today, capped at 16
stages. Two things to check before assuming it is taken: whether the chosen
config is SMEM-limited or already at the 16-stage cap, and — the landmine — that
**no MMA-operand SMEM buffer starts at or past 262144 bytes**. These templates
carry NO ``desc_version`` handling, and a version-0 tcgen05 descriptor's
``start_address`` is 14 bits: past 256 KiB it wraps and the accumulator comes out
EXACTLY ZERO, with no crash (``mma-tma-matrix.md`` § 6). That is why
``test_proj_gemm.py::test_output_is_not_silently_zero`` exists and asserts
non-zero explicitly rather than only a cosine, which would be NaN against a zero
tensor.

**The FROST GEMM engine is opt-in**: ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``
must be set BEFORE ``import cudnn`` or the graph silently runs a cuDNN backend
plan instead. :func:`build_proj_gemm` checks for the plan by name and raises
rather than letting that happen quietly.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Optional

import torch

_FROST_GEMM_PLAN = "frost_gemm"


def _frost_plan_index(names) -> Optional[int]:
    """Index of the top-ranked FROST GEMM plan in ``names``, or None.

    A plan name is the engine name alone (older builds) or the engine name
    followed by its knob bracket, ``frost_gemm[CTA_GROUP=2, ...]`` (the shared
    backend/python knob vocabulary).  Both spell the same engine.
    """
    for i, n in enumerate(names):
        if n == _FROST_GEMM_PLAN or n.startswith(_FROST_GEMM_PLAN + "["):
            return i
    return None


# ---------------------------------------------------------------------------
# The FORK: stage (1) with stages (2)+(3) fused into its epilogue
# ---------------------------------------------------------------------------
#
# ``kernels/proj_gemm_norm_rope.py`` is the rendered dense-bf16 expansion of the
# shipped template for the block's tile config, plus a norm+RoPE arm in the
# epilogue for the Q and K tiles.  It is a FROST template: the loader executes
# it once per ``NormRopeFusionParams`` with the params injected as
# ``FROST_TEMPLATE_PARAMS`` (engine contract S6), so geometries coexist in one
# process and nothing is read from the environment.  The kernel's own docstring
# carries the design; this module carries the params and the launch.
#
# ``kernels/proj_gemm_norm_rope_fp8.py`` is the SAME fusion on the FP8 pipeline
# (``NormRopeFusionParams(quant_fp8=True)``): the rendered e4m3 x e4m3 -> fp32
# GEMM with the per-tensor descale ``alpha``, whose epilogue also QUANTIZES --
# Q/K normed + rotated then written as e4m3 into COMPACT per-tensor outputs
# ``q8 [T, h_q*d]`` / ``k8 [T, h_kv*d]``, V as e4m3 into ``v8 [T, h_kv*d]``, the
# GATE as bf16 into ``gate16 [T, h_q*d]`` (each == compact BSHD, exactly what
# the unfused path hands the FP8 SDPA; there is no ``[T, N_QKV]`` slab any
# more -- round 2 of the FP8 fusion design).  It has its own runner
# (``run_fused_proj_gemm_fp8``): four outputs and a ``[4]`` fp32 scale vector
# (``[alpha, scale_q, scale_k, scale_v]``, read IN-KERNEL) replace the bf16
# runner's single slab, and the bf16 runner refuses an FP8 plan.


@dataclass(frozen=True)
class NormRopeFusionParams:
    """Compile-time geometry + knobs of the fused stage-(1) kernel.

    Defaults are the Qwen3.5-397B geometry so a plain import of the template
    runs standalone.  ``offsets`` / ``n_qkvg`` follow ``api.py`` S1 exactly
    (Q | GATE | K | V, each ``heads * d_head`` wide) -- the fused epilogue
    classifies a tile by comparing its first column against these.

    ``norm_source``:
      ``"ldg"``   per-lane global loads of cos/sin (before the TMEM pass) and of
                  the norm weight (per subtile), issued in the epilogue arm.
                  Kept as the A/B reference for ``ldg_early``.
      ``"ldg_early"`` DEFAULT.  The cos/sin loads issued BEFORE the epilogue's wait on the
                  accumulator barrier, so they land while the MMA drains the
                  tile (non-norm tiles read the tables' row 0 -- one L1 line);
                  the weight stays per subtile.  Hoisting the weight too
                  (+32 live words across the sum-of-squares pass) SPILLED at
                  ``epi_reg_count=232`` -- 44 STL / 88 LDL -- do not retry.
      ``"const"`` the SAME arithmetic with every load deleted -- the diagnostic
                  floor (is the cost latency, or ALU/registers?).  NOT correct.
      ``"const_w"`` / ``"const_cs"`` FP8 fork only: ``const`` split by load class.
                  ``const_w`` = cos/sin loaded exactly as ``ldg_early``, the per-subtile
                  norm WEIGHT replaced by ``const``'s substitute; ``const_cs`` = the
                  weight loaded exactly as ``ldg_early``, cos/sin replaced by ``const``'s
                  substitutes.  Diagnostic floors (which load class costs?).  NOT correct.
      ``"off"``   the epilogue arm is not traced at all: the fork degenerates to
                  the rendered GEMM.  The A/B control.  bf16 only.

    ``quant_fp8`` (LAST field, default False so every bf16 key is unchanged)
    selects the FP8 fork ``proj_gemm_norm_rope_fp8.py``: e4m3 inputs, the
    descale ``alpha`` and the quant scales read in-kernel from a ``[4]`` fp32
    tensor, Q/K/V written as COMPACT e4m3 tensors (``q8 [T, h_q*d]``,
    ``k8 / v8 [T, h_kv*d]``) and the GATE as bf16 (``[T, n_gate]``).
    Inference only: ``want_rstd`` must be False, and
    ``norm_source="off"`` does not exist there (there is no single bf16 slab to
    degenerate to; the loads-deleted floor is ``"const"``).

    ``qk_norm`` (appended after ``quant_fp8``; default True so every existing key
    is spelled identically) selects the RoPE-ONLY epilogue when False: no pass A
    (sum of squares), no ``rsqrt``, no norm-weight loads, no ``rstd`` -- the Q/K
    tiles are rotated on the fp32 accumulator (FP8 fork: descaled + scaled, then
    quantized) and stored.  The kernel switch is ``_QK_NORM`` (``const_expr``) on
    both forks, and the weights are ``None`` at the ABI (the artifact's fakes are
    None too, so a norm-on artifact can never be handed None weights: the key
    differs).  ``want_rstd`` must be False (nothing to emit) and ``norm_source``
    ``"const_w"`` has no meaning (there are no weight loads to delete;
    ``"const_cs"`` == ``"const"`` there).
    """

    d_head: int = 256
    rope_dim: int = 64
    h_q: int = 32
    h_kv: int = 2
    eps: float = 1e-6
    want_rstd: bool = False
    norm_source: str = "ldg_early"
    quant_fp8: bool = False  # the FP8 fork (e4m3 in, e4m3 Q/K/V + bf16 GATE out); appended, bf16 keys unchanged
    qk_norm: bool = True  # False: RoPE-only epilogue (no pass A / rsqrt / weight loads / rstd); appended, keys unchanged

    @property
    def offsets(self) -> tuple[int, int, int, int]:
        q = 0
        g = self.h_q * self.d_head
        k = 2 * self.h_q * self.d_head
        v = k + self.h_kv * self.d_head
        return (q, g, k, v)

    @property
    def n_qkvg(self) -> int:
        return (2 * self.h_q + 2 * self.h_kv) * self.d_head

    @property
    def n_gate(self) -> int:
        """Width of the bf16 GATE slab of the FP8 fork (= ``h_q * d_head``)."""
        return self.h_q * self.d_head

    @property
    def n_qkv(self) -> int:
        """``n_qkvg - n_gate`` = the Q + K + V columns.  Geometry bookkeeping only: since
        round 2 the FP8 fork writes q8 / k8 / v8 as separate compact tensors (no slab)."""
        return self.n_qkvg - self.n_gate

    @property
    def n_q(self) -> int:
        """Width of the compact e4m3 ``q8`` output of the FP8 fork (= ``h_q * d_head``)."""
        return self.h_q * self.d_head

    @property
    def n_kv(self) -> int:
        """Width of the compact e4m3 ``k8`` / ``v8`` outputs of the FP8 fork (= ``h_kv * d_head``)."""
        return self.h_kv * self.d_head


def validate_norm_rope_params(p: NormRopeFusionParams) -> None:
    """Raise ``ValueError`` on anything the fused epilogue cannot express.

    The tile-config-dependent facts (per-CTA N tile == d_head, subtile width
    vs rope_dim) are checked by the template itself against its rendered
    constants; this is the geometry-only half, usable without the DSL.
    """
    if p.d_head <= 0 or p.d_head & (p.d_head - 1):
        raise ValueError(f"d_head must be a power of two (the head index is a shift), got {p.d_head}")
    if p.h_q <= 0 or p.h_kv <= 0 or p.h_q % p.h_kv:
        raise ValueError(f"h_q={p.h_q} must be a positive multiple of h_kv={p.h_kv}")
    if not 0 < p.rope_dim < p.d_head or p.rope_dim % 2:
        raise ValueError(f"rope_dim must be even and in (0, d_head={p.d_head}), got {p.rope_dim}")
    if not p.eps > 0.0:
        raise ValueError(f"eps must be > 0, got {p.eps}")
    if p.norm_source not in ("ldg", "ldg_early", "const", "const_w", "const_cs", "off"):
        raise ValueError(f"norm_source must be 'ldg', 'ldg_early', 'const', 'const_w', 'const_cs' or 'off', got {p.norm_source!r}")
    if p.norm_source in ("const_w", "const_cs") and not p.quant_fp8:
        raise ValueError(f"norm_source={p.norm_source!r} is a diagnostic arm of the FP8 fork only (quant_fp8=True); the bf16 fork has 'const'")
    if p.quant_fp8:
        if p.want_rstd:
            raise ValueError("quant_fp8: the FP8 fused projection is inference-only and emits no rstd; want_rstd must be False")
        if p.norm_source == "off":
            raise ValueError(
                "quant_fp8: norm_source='off' has no meaning on the FP8 fork (no single bf16 slab to degenerate to); use 'const' for the loads-deleted floor"
            )
    if not p.qk_norm:
        if p.want_rstd:
            raise ValueError("qk_norm=False emits no rstd; want_rstd must be False")
        if p.norm_source == "const_w":
            raise ValueError("qk_norm=False has no weight loads: const_w == ldg_early; use 'const_cs' (== 'const') for the floor")


_FUSED_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proj_gemm_norm_rope.py")
_FUSED_TEMPLATE_FP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proj_gemm_norm_rope_fp8.py")


@dataclass
class FusedProjGemmPlan:
    """One compiled fused projection.  Plan-time only; ``run_fused_proj_gemm`` launches."""

    params: NormRopeFusionParams
    module: Any
    launch: Any  # the cute-compiled ``_host``
    fp8: bool = False  # True: the FP8 fork; launch with ``run_fused_proj_gemm_fp8`` (q8/k8/v8/gate16 + qscal)

    @property
    def workspace_bytes(self) -> int:
        return 0  # no split-K, no scratch: the kernel writes the slab directly


def build_fused_proj_gemm(params: NormRopeFusionParams) -> FusedProjGemmPlan:
    """Load the fork specialized for ``params`` and cute-compile it.

    No shape here: M/N/K are runtime (symbolic in the artifact), so one plan
    serves every batch/sequence of a geometry.  N is checked at launch against
    ``params.n_qkvg`` -- the tile classification is only meaningful for the
    slab the params describe.
    """
    from cudnn.frost.template_loader import load_template

    validate_norm_rope_params(params)
    if params.quant_fp8:
        mod = load_template(_FUSED_TEMPLATE_FP8, params, tag="proj_gemm_norm_rope_fp8")
        return FusedProjGemmPlan(params=params, module=mod, launch=mod.compile(), fp8=True)
    mod = load_template(_FUSED_TEMPLATE, params, tag="proj_gemm_norm_rope")
    return FusedProjGemmPlan(params=params, module=mod, launch=mod.compile())


def _check_norm_weights(p: NormRopeFusionParams, w_q_norm, w_k_norm, dtype, dtype_name: str) -> None:
    """Norm weights present iff ``params.qk_norm`` -- BOTH directions, before any tensor is touched.

    The artifact was compiled with the weights as ``None`` under ``qk_norm=False``
    (and as ``[D]`` fakes otherwise), so a mismatch here would be a tvm-ffi ABI
    error at launch rather than a typed decline; check it on the host first."""
    if p.qk_norm:
        for name, t_ in (("w_q_norm", w_q_norm), ("w_k_norm", w_k_norm)):
            if t_ is None:
                raise ValueError(f"this artifact was compiled with qk_norm=True; {name} must be a [{p.d_head}] {dtype_name} tensor, got None")
            if tuple(t_.shape) != (p.d_head,) or t_.dtype != dtype:
                raise ValueError(f"{name} must be [{p.d_head}] {dtype_name}, got {tuple(t_.shape)} {t_.dtype}")
    elif w_q_norm is not None or w_k_norm is not None:
        raise ValueError("this artifact was compiled with qk_norm=False (RoPE only, no weight loads); w_q_norm / w_k_norm must both be None")


def _rope_table_2d(t: torch.Tensor, name: str, m: int, rope_dim: int, dtype: torch.dtype, dtype_name: str) -> torch.Tensor:
    """The kernel's ``[M, ROPE_DIM]`` view of a RoPE table -- a VIEW, never a copy.

    A ``[B, S, R]`` table is flattened with ``.view``: ``reshape`` would silently COPY a non-viewable table
    and the contiguity check below would then pass on the copy -- a hidden per-execute allocation (engine
    contract rule 10).  The result must be a contiguous ``[M, R]`` ``dtype`` table (the same contract the
    unfused stage applies with ``cos.view(t, rope_dim)``)."""
    if t.ndim == 3:
        try:
            t = t.view(-1, rope_dim)
        except RuntimeError as e:
            raise ValueError(
                f"{name} must be a contiguous [M={m}, {rope_dim}] {dtype_name} table; a [B, S, R] table is flattened as a VIEW (never copied) "
                f"and {tuple(t.shape)} with strides {tuple(t.stride())} is not viewable"
            ) from e
    if tuple(t.shape) != (m, rope_dim) or not t.is_contiguous() or t.dtype != dtype:
        raise ValueError(f"{name} must be a contiguous [M={m}, {rope_dim}] {dtype_name} table, got {tuple(t.shape)} {t.dtype}")
    return t


def run_fused_proj_gemm(
    plan: FusedProjGemmPlan,
    a: torch.Tensor,  # [M, K]   bf16
    w: torch.Tensor,  # [N, K]   bf16 (checkpoint layout, read transposed)
    out: torch.Tensor,  # [M, N]   bf16 -- Q/K columns land normed + rotated (qk_norm=False: rotated only)
    w_q_norm: Optional[torch.Tensor],  # [D]; None iff params.qk_norm is False
    w_k_norm: Optional[torch.Tensor],  # [D]
    cos: torch.Tensor,  # [M, ROPE_DIM] (or [B, S, ROPE_DIM] contiguous)
    sin: torch.Tensor,
    rstd_q: Optional[torch.Tensor] = None,  # [M, H_q]  fp32, required iff params.want_rstd
    rstd_k: Optional[torch.Tensor] = None,  # [M, H_kv] fp32
    *,
    stream,
) -> None:
    """Launch.  No allocation, no conversion; every check is cheap host arithmetic.

    The problem tuple mirrors the shipped ``compiler._lower`` for the plain
    flavor: ``(m, n, k, batch, a strides (m, k, l), b strides (n, k, l), out
    strides (m, n, l))`` in ELEMENTS, taken from rank-3 ``[1, M, K]``-shaped views.
    """
    from cuda.bindings import driver as cuda

    if plan.fp8:
        raise ValueError("this plan is the FP8 fork (q8/k8/v8/gate16 outputs + qscal); launch it with run_fused_proj_gemm_fp8")
    p = plan.params
    a3, w3, o3 = _rank3(a, "a"), _rank3(w, "w"), _rank3(out, "out")
    m, k = int(a3.shape[1]), int(a3.shape[2])
    n = int(w3.shape[1])
    if n != p.n_qkvg:
        raise ValueError(f"the fused projection classifies tiles for N={p.n_qkvg} (Q|GATE|K|V at {p.offsets}); the weight has N={n}")
    if tuple(o3.shape) != (1, m, n) or int(w3.shape[2]) != k:
        raise ValueError(f"shape mismatch: a {tuple(a3.shape)}, w {tuple(w3.shape)}, out {tuple(o3.shape)}")
    cos2 = _rope_table_2d(cos, "cos", m, p.rope_dim, a.dtype, str(a.dtype))
    sin2 = _rope_table_2d(sin, "sin", m, p.rope_dim, a.dtype, str(a.dtype))
    _check_norm_weights(p, w_q_norm, w_k_norm, a.dtype, str(a.dtype).removeprefix("torch."))
    if p.want_rstd:
        if rstd_q is None or rstd_k is None:
            raise ValueError("this artifact was compiled with rstd outputs; both must be bound (Rule 1: no silent fallback)")
        if tuple(rstd_q.shape) != (m, p.h_q) or tuple(rstd_k.shape) != (m, p.h_kv) or rstd_q.dtype != torch.float32 or rstd_k.dtype != torch.float32:
            raise ValueError("rstd_q / rstd_k must be fp32 [M, h_q] / [M, h_kv]")
    else:
        rstd_q = rstd_k = None

    def _st(t3):
        st = t3.stride()
        return (st[1], st[2], st[0])

    problem = (m, n, k, 1, *_st(a3), *_st(w3), *_st(o3))
    # The artifact's fakes are (M|N, K, L) / (M, N, L): the same axis relabel the
    # shipped lowering applies (`compiler._lower`: `v.permute(1, 2, 0)`).
    plan.launch(
        problem,
        a3.permute(1, 2, 0),
        w3.permute(1, 2, 0),
        o3.permute(1, 2, 0),
        w_q_norm,
        w_k_norm,
        cos2,
        sin2,
        rstd_q,
        rstd_k,
        cuda.CUstream(int(stream)),
    )


def run_fused_proj_gemm_fp8(
    plan: FusedProjGemmPlan,
    a: torch.Tensor,  # [M, K]        e4m3 (rank-2, or a rank-3 [1, M, K] view)
    w: torch.Tensor,  # [N_qkvg, K]   e4m3 (checkpoint layout, read transposed)
    out_q8: torch.Tensor,  # [M, h_q*d]   e4m3 contiguous (== compact BSHD [B, S, h_q, d])
    out_k8: torch.Tensor,  # [M, h_kv*d]  e4m3 contiguous
    out_v8: torch.Tensor,  # [M, h_kv*d]  e4m3 contiguous
    out_gate16: torch.Tensor,  # [M, h_q*d] bf16 contiguous
    w_q_norm: Optional[torch.Tensor],  # [D] bf16; None iff params.qk_norm is False
    w_k_norm: Optional[torch.Tensor],  # [D] bf16
    cos: torch.Tensor,  # [M, ROPE_DIM] bf16 (or [B, S, ROPE_DIM] contiguous)
    sin: torch.Tensor,
    qscal: torch.Tensor,  # [4] fp32 CUDA: [alpha, scale_q, scale_k, scale_v], read in-kernel
    *,
    stream,
) -> None:
    """Launch the FP8 fork.  No allocation, no conversion; every check is cheap host arithmetic.

    Problem tuple: ``(m, n, k, batch, a strides (m, k, l), b strides (n, k, l),
    q8 strides (m, n, l), k8 strides, v8 strides, gate16 strides)`` in ELEMENTS,
    from rank-3 ``[1, M, K]``-shaped views -- the bf16 runner's tuple plus three
    more output triples.  The cos/sin and norm-weight dtype checks compare
    against bf16 (the ACTIVATION dtype of the tables), not against ``a.dtype``
    (e4m3).  Every output is a TMA-store target: contiguous and 16-B aligned.
    """
    from cuda.bindings import driver as cuda

    if not plan.fp8:
        raise ValueError("this plan is the bf16 fork (one bf16 slab); launch it with run_fused_proj_gemm")
    p = plan.params
    a3, w3 = _rank3(a, "a"), _rank3(w, "w")
    q3, k3, v3, g3 = _rank3(out_q8, "out_q8"), _rank3(out_k8, "out_k8"), _rank3(out_v8, "out_v8"), _rank3(out_gate16, "out_gate16")
    m, k = int(a3.shape[1]), int(a3.shape[2])
    n = int(w3.shape[1])
    if _FP8_E4M3 is None or a.dtype != _FP8_E4M3 or w.dtype != _FP8_E4M3:
        raise ValueError(f"the FP8 fused projection takes e4m3 operands, got a {a.dtype}, w {w.dtype}")
    if n != p.n_qkvg:
        raise ValueError(f"the fused projection classifies tiles for N={p.n_qkvg} (Q|GATE|K|V at {p.offsets}); the weight has N={n}")
    if int(w3.shape[2]) != k:
        raise ValueError(f"shape mismatch: a {tuple(a3.shape)}, w {tuple(w3.shape)}")
    for name, t3, t_, width, dt, dt_name in (
        ("out_q8", q3, out_q8, p.n_q, _FP8_E4M3, "e4m3"),
        ("out_k8", k3, out_k8, p.n_kv, _FP8_E4M3, "e4m3"),
        ("out_v8", v3, out_v8, p.n_kv, _FP8_E4M3, "e4m3"),
        ("out_gate16", g3, out_gate16, p.n_gate, torch.bfloat16, "bf16"),
    ):
        if tuple(t3.shape) != (1, m, width) or t3.dtype != dt or not t3.is_contiguous() or t3.data_ptr() % 16:
            raise ValueError(f"{name} must be a contiguous, 16-B-aligned {dt_name} [M={m}, {width}] tensor, got {tuple(t_.shape)} {t_.dtype}")
    cos2 = _rope_table_2d(cos, "cos", m, p.rope_dim, torch.bfloat16, "bf16")
    sin2 = _rope_table_2d(sin, "sin", m, p.rope_dim, torch.bfloat16, "bf16")
    _check_norm_weights(p, w_q_norm, w_k_norm, torch.bfloat16, "bf16")
    if tuple(qscal.shape) != (4,) or qscal.dtype != torch.float32 or not qscal.is_cuda or not qscal.is_contiguous() or qscal.data_ptr() % 16:
        raise ValueError(
            f"qscal must be a contiguous, 16-B-aligned fp32 CUDA tensor [4] = [alpha, scale_q, scale_k, scale_v], got {tuple(qscal.shape)} {qscal.dtype}"
        )

    def _st(t3):
        st = t3.stride()
        return (st[1], st[2], st[0])

    problem = (m, n, k, 1, *_st(a3), *_st(w3), *_st(q3), *_st(k3), *_st(v3), *_st(g3))
    # The artifact's fakes are (M|N, K, L) / (M, N, L): the same axis relabel the
    # shipped lowering applies (`compiler._lower`: `v.permute(1, 2, 0)`).
    plan.launch(
        problem,
        a3.permute(1, 2, 0),
        w3.permute(1, 2, 0),
        q3.permute(1, 2, 0),
        k3.permute(1, 2, 0),
        v3.permute(1, 2, 0),
        g3.permute(1, 2, 0),
        w_q_norm,
        w_k_norm,
        cos2,
        sin2,
        qscal,
        cuda.CUstream(int(stream)),
    )


def _why_no_frost_plan() -> str:
    """The two reasons the plan is absent, checked in the order that costs least.

    Worth spelling out because the two look identical from the call site and one
    of them is a real capability limit, not a setup mistake.
    """
    import os

    reasons = []
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        arch = cc[0] * 10 + cc[1]
        from cudnn.gemm.frost.kernel_registry import PIPELINE_ARCH_RANGES

        spans = PIPELINE_ARCH_RANGES.get("sm100", ())
        if spans and not any(lo <= arch < hi for lo, hi in spans):
            reasons.append(
                f"the sm100 GEMM template family runs on "
                + " or ".join(f"sm_{lo}..sm_{hi - 1}" for lo, hi in spans)
                + f", and this device is sm_{arch} -- so FROST cannot serve this graph here at all"
            )
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0").strip().lower() not in ("1", "true", "yes", "on"):
        reasons.append("CUDNN_FRONTEND_ENABLE_FROST_ENGINES is not set -- it is an opt-in engine and the flag must be set BEFORE 'import cudnn'")
    if not reasons:
        reasons.append("the engine declined this shape/dtype; run with CUDNN_FRONTEND_LOG_INFO=1 CUDNN_FRONTEND_LOG_FILE=stderr for its reason")
    return (
        " AND ".join(reasons)
        + ". Refusing to fall back: an unpinned graph runs a cuDNN backend plan, and any measurement off it would be of that kernel, not FROST's."
    )


_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)  # None on a torch without fp8 storage types


def _is_fp8(dtype: torch.dtype) -> bool:
    return _FP8_E4M3 is not None and dtype == _FP8_E4M3


def _cudnn_dtype(dtype: torch.dtype):
    import cudnn

    table = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF}
    if _FP8_E4M3 is not None:
        table[_FP8_E4M3] = cudnn.data_type.FP8_E4M3
    try:
        return table[dtype]
    except KeyError:
        raise ValueError(f"proj_gemm serves bf16/f16/fp8-e4m3 only, got {dtype}") from None


_LOG = logging.getLogger(__name__)


# cuDNN handles for the GRAPH route of :func:`run_proj_gemm`, one per
# (device, stream).  The graph API binds a plan's launch stream to a HANDLE
# (``cudnn.set_stream``), not to an execute argument, so threading a stream
# through ``graph.execute`` means owning a handle bound to it.  Created on first
# use and kept for the process -- ``cudnnCreate`` is not free and Rule 1 bans
# per-execute resource creation (same shape as ``sdpa/fwd/torch_op.py``'s
# per-device cache).  Keyed by stream too, and ``set_stream`` is never re-issued
# on a cached handle, so two streams driving a block concurrently never share
# one (the ``cudnn.set_stream`` docstring's "own handle per stream" caveat).
# Bounded by the number of distinct launch streams a caller uses.
_GRAPH_HANDLES: dict = {}


def handle_for_stream(device, stream) -> Any:
    """The cached ``cudnn.Handle`` bound to ``stream`` on ``device`` (created on first use).

    ``stream`` is a raw ``CUstream`` int (``torch.cuda.current_stream(dev).cuda_stream``;
    0 = the legacy default stream).  Only the graph route of :func:`run_proj_gemm`
    needs this -- the JIT route takes ``stream=`` directly.
    """
    import cudnn

    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (idx, int(stream))
    h = _GRAPH_HANDLES.get(key)
    if h is None:
        # create_handle() binds to the CURRENT device -- pin it so a tensor on
        # cuda:1 never gets a handle created against cuda:0.
        with torch.cuda.device(idx):
            h = cudnn.create_handle()
        cudnn.set_stream(handle=h, stream=int(stream))
        _GRAPH_HANDLES[key] = h
    return h


@dataclass
class ProjGemmPlan:
    """One compiled projection. Built at plan time, called per execute."""

    graph: Any
    a: Any
    b: Any
    c: Any
    m: int
    k: int
    n: int
    label: str
    jit: Any = None  # set only when a tile config was forced; see _forced_tile_config
    jit_binding: Any = None
    # FP8 / epilogue additions (append-only). `alpha` is the graph's scalar
    # multiplier tensor when the plan was built with `alpha=True`, else None.
    dtype: Any = None  # input torch dtype
    out_dtype: Any = None  # output torch dtype (bf16 by default for fp8 inputs)
    alpha: Any = None
    tile_config_name: Optional[str] = None  # the JIT config that actually runs, or "heuristic (graph engine)"
    mma_tile_k_bytes: Optional[int] = None  # 32 = the K=32 MMA form (half the Rubin 2xFP8 rate); 64 = K=64

    @property
    def has_alpha(self) -> bool:
        return self.alpha is not None

    @property
    def workspace_bytes(self) -> int:
        return max(int(self.graph.get_workspace_size()), 1)

    def flops(self) -> int:
        """``2*M*N*K`` — the denominator for an MMA SOL number."""
        return 2 * self.m * self.n * self.k


# The shipped tile scorer has a REAL BUG, and this is the workaround for it.
#
# `gemm/frost/tile_config._tile_score` multiplies an efficiency factor
# `n_eff = N / (n_tiles * cta_n)` by `total_ctas`, which contains `n_tiles`.
# The product `n_tiles * n_eff` is `N / cta_n` -- independent of the tiling --
# so the raggedness penalty CANCELS EXACTLY and the score reduces to
# `1 / (sqrt(cta_n) * waves)`. At equal wave count it therefore always prefers
# the NARROWEST tile. Verified numerically: score(224)/score(256) is
# sqrt(256/224) to six digits at the out_proj shape.
#
# Cost here: `out_proj` is M x 4096 x 8192, and the scorer picks a 224-wide N
# tile against N = 4096. 256 divides 4096 into 16 tiles; 224 gives 18 full
# tiles plus a ragged 64, so the last tile does a full-width MMA for a quarter
# of a tile of useful work. Measured on Rubin: 2189 -> 2937 TFLOP/s at M=4096
# (+34%), and 256 also wins at M=8192 (3402 -> 3573) and M=32768 (3706 -> 3775).
#
# Fixing the scorer is the right long-term answer, but it is SHIPPED code used
# by every GEMM caller and the corrected score changes the pick on ~80% of a
# (M, N, K) grid -- that needs a measured sweep, not a drive-by edit. So the
# block names a config for the shape it owns, and the upstream fix is filed
# separately.
def _forced_tile_config(n: int) -> Optional[str]:
    """A catalog config whose N tile DIVIDES ``n``, or None to take the heuristic.

    Only claims the case it measured: a 256-wide N tile when 256 divides N.
    Anything else falls through to the scorer rather than guessing.
    """
    if n % 256 == 0:
        return "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
    return None


def build_proj_gemm(
    *,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    label: str,
    pin_frost: bool = True,
    tile_config: Optional[str] = "auto",
    out_dtype: Optional[torch.dtype] = None,
    alpha: bool = False,
) -> ProjGemmPlan:
    """Compile one projection GEMM and pin the FROST plan.

    Plan-time only: shapes and dtypes come from the block's declaration, never
    from a runtime value (AGENTS.md Rule 4), and nothing here allocates a data
    tensor.

    ``pin_frost=False`` selects whatever the heuristic ranks first — used by the
    tests to get a native-cuDNN reference through the identical graph.

    **FP8 (E4M3) inputs.** ``dtype=torch.float8_e4m3fn`` builds the same graph
    with FP8 A/B, fp32 accumulate, and ``out_dtype`` output (default bf16 for
    fp8 inputs; for bf16/f16 inputs the output keeps the input dtype unless
    ``out_dtype`` says otherwise). ``alpha=True`` adds the per-tensor descale as
    a SCALAR-multiply epilogue -- ``c = matmul(A, B) * alpha`` with ``alpha`` a
    ``[1, 1, 1]`` fp32 graph tensor bound at execute (``run_proj_gemm(...,
    alpha=)``); the FROST engine recognises the ``[1,1,1]`` broadcast as its
    "scalar" fusion mode, so it is one fused epilogue, not a second kernel.
    The dense FP8 path runs the K=32 MMA form on Rubin (``mma_tile_k_bytes=32``,
    the ``preferred_mma_tile_k_bytes`` default for non-block-scale graphs) --
    half the Rubin 2xFP8 rate -- which is why the plan records the resolved tile
    config and its ``mma_tile_k_bytes`` for the perf table to state.
    """
    import cudnn

    for lbl, v in (("m", m), ("k", k), ("n", n)):
        if v <= 0:
            raise ValueError(f"{label}: {lbl} must be > 0, got {v}")
    io_dt = _cudnn_dtype(dtype)
    fp8 = _is_fp8(dtype)
    if fp8 and k % 16:
        # TMA's 16-byte contiguous-extent rule at 1 B/elem (compiler._tma_alignment_reject).
        raise ValueError(f"{label}: FP8 operands need K % 16 == 0 (16-byte TMA rule at 1 B/elem), got K={k}")
    if out_dtype is None:
        out_dtype = torch.bfloat16 if fp8 else dtype
    out_dt = _cudnn_dtype(out_dtype)

    g = cudnn.pygraph(
        io_data_type=io_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a = g.tensor(name="A", dim=[1, m, k], stride=[m * k, k, 1])
    # B is the weight as a checkpoint stores it -- [N, K] row-major -- declared
    # transposed. stride[1] == 1 is what makes it the K-contiguous operand.
    b = g.tensor(name="B", dim=[1, k, n], stride=[k * n, 1, k])
    mm = g.matmul(A=a, B=b, name=label)
    alpha_t = None
    if alpha:
        # dim [1,1,1] == graph_analyzer "scalar" broadcast mode -> fused epilogue multiply.
        # The graph tensor / op names land in the GENERATED kernel source as
        # Python identifiers, so a label with a space or a dash ("1 qkv_gate_proj")
        # is a SyntaxError at JIT time -- sanitise to an identifier.
        _ident = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(label)) or "proj"
        if _ident[0].isdigit():
            _ident = "p_" + _ident
        alpha_t = g.tensor(name=f"{_ident}_alpha", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
        c = g.mul(a=mm, b=alpha_t, name=f"{_ident}_scale")
    else:
        c = mm
    c.set_output(True).set_data_type(out_dt)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    if pin_frost:
        # Plan names carry the plan's knobs since the shared knob vocabulary
        # landed ("frost_gemm[CTA_GROUP=2, MMA_TILE_K=32, ...]"), so the pin
        # matches the engine NAME (exact, or followed by its knob bracket) and
        # takes the top-ranked FROST configuration; deselect_engines already
        # matches by substring (the engine name is the user-visible API).
        frost_idx = _frost_plan_index(names)
        if frost_idx is None:
            raise RuntimeError(f"{label}: the {_FROST_GEMM_PLAN!r} plan is not in the ranked list {names}. {_why_no_frost_plan()}")
        g.select_plan(frost_idx)
    else:
        g.deselect_engines([_FROST_GEMM_PLAN])
    g.check_support()
    g.build_plans()
    plan = ProjGemmPlan(graph=g, a=a, b=b, c=c, m=m, k=k, n=n, label=label, dtype=dtype, out_dtype=out_dtype, alpha=alpha_t)
    plan.tile_config_name = "heuristic (graph engine)"

    # Forced tile: JIT the SAME graph at a named config. The graph engine picks
    # its config internally with no override hook (graph_analyzer.py:571-574),
    # so taking a specific tile means going through the JIT entry point. The
    # graph is still built above -- it is what gets analyzed, and it keeps
    # `workspace_bytes` and the plan-name pin honest.
    name = _forced_tile_config(n) if tile_config == "auto" else tile_config
    if name and pin_frost:
        from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
        from cudnn.gemm.frost.tile_config import by_name

        try:
            compiled = jit_from_cudnn_graph(g, config=by_name(name))
        except Exception as exc:  # a config this shape cannot take is a FALLBACK, not a failure
            if tile_config != "auto":
                raise
            compiled = None
            _LOG.debug("%s: forced tile %s rejected (%s); falling back to the heuristic", label, name, type(exc).__name__)
        if compiled is not None:
            plan.jit, plan.jit_binding = compiled, compiled.binding
            plan.tile_config_name = getattr(compiled.config, "name", name)
            plan.mma_tile_k_bytes = getattr(compiled.config, "mma_tile_k_bytes", None)
    return plan


def run_proj_gemm(
    plan: ProjGemmPlan,
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    alpha: Optional[torch.Tensor] = None,
    *,
    stream=None,
) -> None:
    """Launch. No allocation, no conversion — the caller owns every buffer.

    ``a`` is ``[M, K]``-shaped storage, ``w`` is ``[N, K]``, ``out`` is
    ``[M, N]``; any leading batch dim of 1 is accepted since the graph carries
    the layout and only the pointer is bound.

    **The launch stream (Rule 5).** ``stream`` is a raw ``CUstream`` int (or a
    ``cuda.CUstream``) -- the same value the block's CuTe-DSL stages take.  It
    reaches BOTH routes: the forced-tile JIT plan's own ``stream=``, and, on the
    graph route, a cached per-(device, stream) cuDNN handle bound to it
    (:func:`handle_for_stream`; the graph API carries a stream on a HANDLE, not
    on an execute argument).  ``None`` means torch's current stream on
    ``out.device`` -- so a standalone call under ``with torch.cuda.stream(s):``
    stays ordered with the caller's torch work, exactly as every other stage of
    the block does.  Before this the two GEMMs launched on the default stream
    regardless, and a block run on a side stream had its SDPA read a slab the
    projection had not written yet (all-zero output on SM107).

    ``handle`` remains the graph API's classic way of naming the stream: given
    alone, its stream is the launch stream; given WITH ``stream``, the two must
    agree -- a handle bound to another stream would run this GEMM off the stream
    the caller ordered its other work on, which is the race this guards, so it
    is a ``ValueError`` rather than a silent pick.
    """
    # `alpha`: the per-tensor descale for an fp8 plan built with alpha=True --
    # a 1-element fp32 CUDA tensor, bound as the [1,1,1] scalar aux (a VIEW).
    if plan.has_alpha:
        if alpha is None:
            raise ValueError(f"{plan.label}: this plan was built with alpha=True; pass alpha= (1-element fp32 CUDA tensor). No silent 1.0 (Rule 1).")
        if alpha.numel() != 1 or alpha.dtype != torch.float32 or not alpha.is_cuda:
            raise ValueError(f"{plan.label}: alpha must be a 1-element fp32 CUDA tensor, got {tuple(alpha.shape)} {alpha.dtype} on {alpha.device}")
        alpha3 = alpha.reshape(1, 1, 1)  # a view: 1 element is always contiguous
    elif alpha is not None:
        raise ValueError(f"{plan.label}: this plan has no alpha epilogue (built with alpha=False); refusing to drop the value silently")
    # Resolve the launch stream FIRST (Rule 5), before any route is taken.
    if handle is not None:
        import cudnn

        handle_stream = int(cudnn.get_stream(handle) or 0)
        if stream is not None and int(stream) != handle_stream:
            raise ValueError(
                f"{plan.label}: handle is bound to stream {handle_stream:#x} but stream={int(stream):#x} was requested; "
                "a GEMM off the launch stream races the block's other stages (Rule 5). Pass one or make them agree (cudnn.set_stream)."
            )
        stream = handle_stream
    elif stream is None:
        stream = torch.cuda.current_stream(out.device).cuda_stream
    stream = int(stream)
    if plan.jit is not None:
        bd = plan.jit_binding
        vp = {bd.a_operands[0]: _rank3(a, "a"), bd.b_operands[0]: _rank3(w, "w"), bd.outputs[0]: _rank3(out, "out")}
        if plan.has_alpha:
            # GemmBinding.aux is in FusionChain.aux_tensors order; alpha is the only aux here.
            if len(bd.aux) != 1:
                raise RuntimeError(f"{plan.label}: expected exactly one aux operand (alpha) in the JIT binding, found {len(bd.aux)}")
            vp[bd.aux[0]] = alpha3
        plan.jit(vp, stream=stream)
        return
    vp = {plan.a: _rank3(a, "a"), plan.b: _rank3(w, "w"), plan.c: _rank3(out, "out")}
    if plan.has_alpha:
        vp[plan.alpha] = alpha3
    # The graph route names its stream through the handle: the plan's engine
    # reads `ExecutionContext.stream` off it (`_pygraph.execute` -> `cudnn.get_stream`).
    if handle is None:
        handle = handle_for_stream(out.device, stream)
    plan.graph.execute(vp, workspace, handle)


def _rank3(t: torch.Tensor, name: str) -> torch.Tensor:
    """Bind the operand as the rank-3 buffer the graph declared.

    The kernel reads three axes off every operand, so a rank-2 ``[M, K]`` bind
    raises ``"expected a rank-3 buffer"``. ``unsqueeze(0)`` is always a VIEW —
    never a ``reshape`` here, which may copy, silently allocate per execute and
    (for the OUTPUT) swallow the kernel's write entirely (Rule 1).
    """
    if t.ndim == 3:
        return t
    if t.ndim == 2:
        return t.unsqueeze(0)
    raise ValueError(f"{name} must be rank 2 or 3, got shape {tuple(t.shape)}")
