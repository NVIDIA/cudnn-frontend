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
``sdpa/bwd/kernels/bprop_matmul_blackwell.py`` forked this same template to give its
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

**The BACKWARD's four projection GEMMs -- transposed operands as VIEWS.** The
``nn.Linear`` orientation (``api_bwd.py``: ``dW = dY^T @ X``, ``dX = dY @ W``) needs
an M-major A and an N-major B, which the shipped engine already renders (it
infers each operand's major from the graph's strides). :func:`build_proj_gemm`
takes them as the appended ``a_major`` / ``b_major`` kwargs -- the dims stay
``[1, m, k]`` / ``[1, k, n]``, only the strides move -- and the two drivers bind
views, no repack:

    build                          A bind (view)                      B bind (view)                 C
    a_major="k", b_major="k"       X [T, K] row-major (the forward)   W [N, K] row-major, read ^T   [T, N]
    a_major="m", b_major="n"       dY[T, rows]^T  -> run_wgrad_gemm   X [T, cols] row-major         dW [rows, cols]
    a_major="k", b_major="n"       dY[T, K] row-major -> run_dgrad_gemm   W [K, N] row-major (un-transposed)   dX [T, N]

    B1  dW_o    = dY^T    @ O_gated   ("m", "n"; m=dm, k=T, n=HD)     B2  dO_gated = dY    @ W_o     ("k", "n"; m=T, k=dm, n=HD)
    B7  dW_qkvg = dQKVG^T @ h         ("m", "n"; m=N,  k=T, n=dm)     B8  dh       = dQKVG @ W_qkvg  ("k", "n"; m=T, k=N,  n=dm)

A view binds ONLY against a plan declared with its majors, and ONLY with exactly
the declared strides (size-1 axes aside): the graph fallback binds pointers
against the DECLARED strides with no check (a padded view is a silent
reinterpretation), and the JIT route re-labels B from graph order into kernel
order only when the runtime ``(shape, stride)`` EQUALS the declaration
(``sm100/compiler.py`` ``_lower``: a padded N-major B trips its generic
``gave_up["input layout"]``, a padded M-major A would be accepted) -- so
:func:`run_wgrad_gemm` / :func:`run_dgrad_gemm` assert the whole declared layout
of every view before binding, on both routes (a typed ``ValueError`` naming the
operand and both strides). All four backward GEMMs take
the FORCED ``..._cluster2x1_2ctamma`` tile (every ``n`` is ``d_model`` or
``h_q*d_head``, both ``% 256 == 0``); its MN-major rendering on cc 10.7 is pinned
by ``test_proj_gemm_bwd.py::test_forced_tile_renders_mn_major_on_cc107``. The
appended ``split_k`` kwarg (0 / 1 / S >= 2) is documented on :func:`build_proj_gemm`.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Optional

import torch

_FROST_GEMM_PLAN = "frost_gemm"


class _NoFrostPlan(RuntimeError):
    """``frost_gemm`` is not in the graph's ranked plan list (opt-in flag unset, arch outside
    the template family, or the engine declined the shape).  A ``RuntimeError`` so a dense
    caller sees it as before; a DISTINCT type so the block-scale fallback can catch exactly
    this and the typed backend decline, and let a genuine backend failure propagate."""


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
#
# ``kernels/proj_gemm_norm_rope_mxfp8.py`` is the SAME fusion on the MXFP8 pipeline
# (``NormRopeFusionParams(quant_mxfp8=True)``): the rendered BLOCK-SCALE GEMM (e4m3
# codes + one E8M0 scale per 32-element K block on both operands, dequantized IN the
# MMA -- no alpha) whose epilogue [norms +] rotates Q/K and BLOCK-quantizes Q/K
# rowwise and V columnwise, writing the compact e4m3 ``q8 / k8 / v8`` + bf16
# ``gate16`` AND the three F8_128x4 E8M0 scale-factor blobs ``sf_q / sf_k / sf_v``
# exactly as ``kernels/quantize_mxfp8.py`` lays them down for
# ``sm107/prefill_d256_mxfp8.py``.  Runner ``run_fused_proj_gemm_mxfp8`` (PR-B
# section 3.1 ABI): it takes the block-scale GEMM's own SF blobs (``sf_a`` /
# ``sf_w``, PADDED F8_128x4 -- ``sf_padded_dims``) and ``batch`` / ``seq_len`` (the
# kernel decodes ``(b, s_tile)`` per 128-row tile), and DECLINES (typed
# ``NotImplementedError``) ``seq_len % 128 != 0 and batch > 1`` -- a GEMM tile would
# straddle two sequences; the unfused path serves that shape.


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

    ``quant_mxfp8`` (appended after ``qk_norm``) selects the MXFP8 GEMM fork twin
    ``proj_gemm_norm_rope_mxfp8.py``: the BLOCK-SCALE rendering (e4m3 codes + F8_128x4
    E8M0 scale factors on both operands, no alpha), norm optional + RoPE + ROWWISE
    (Q/K) / COLUMNWISE (V) block quantization with the F8_128x4 E8M0 scale factors
    written in the epilogue; outputs the compact e4m3 ``q8 / k8 / v8``, the bf16
    ``gate16`` and the three ``sf_q / sf_k / sf_v`` blobs.  Inference only
    (``want_rstd`` False), no ``"off"`` arm, exclusive with ``quant_fp8``; launched
    with :func:`run_fused_proj_gemm_mxfp8`.
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
    quant_mxfp8: bool = False  # the MXFP8 fork twin (block-scale rendering, PR-B section 3.1); appended, keys unchanged

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
    if p.norm_source in ("const_w", "const_cs") and not (p.quant_fp8 or p.quant_mxfp8):
        raise ValueError(
            f"norm_source={p.norm_source!r} is a diagnostic arm of the quantizing forks only (quant_fp8=True or quant_mxfp8=True); the bf16 fork has 'const'"
        )
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
    if p.quant_mxfp8:
        if p.quant_fp8:
            raise ValueError("quant_fp8 and quant_mxfp8 name two different forks; set at most one")
        if p.want_rstd:
            raise ValueError("quant_mxfp8: the MXFP8 fused projection is inference-only and emits no rstd; want_rstd must be False")
        if p.norm_source == "off":
            raise ValueError(
                "quant_mxfp8: norm_source='off' has no meaning on the MXFP8 fork (no single bf16 slab to degenerate to); use 'const' for the floor"
            )


_FUSED_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proj_gemm_norm_rope.py")
_FUSED_TEMPLATE_FP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proj_gemm_norm_rope_fp8.py")
_FUSED_TEMPLATE_MXFP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proj_gemm_norm_rope_mxfp8.py")


@dataclass
class FusedProjGemmPlan:
    """One compiled fused projection.  Plan-time only; ``run_fused_proj_gemm`` launches."""

    params: NormRopeFusionParams
    module: Any
    launch: Any  # the cute-compiled ``_host``
    fp8: bool = False  # True: the FP8 fork; launch with ``run_fused_proj_gemm_fp8`` (q8/k8/v8/gate16 + qscal)
    mxfp8: bool = False  # True: the MXFP8 fork twin; launch with ``run_fused_proj_gemm_mxfp8`` (q8/k8/v8/gate16 + sf_q/sf_k/sf_v)

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
    if params.quant_mxfp8:
        mod = load_template(_FUSED_TEMPLATE_MXFP8, params, tag="proj_gemm_norm_rope_mxfp8")
        return FusedProjGemmPlan(params=params, module=mod, launch=mod.compile(), mxfp8=True)
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
    if plan.mxfp8:
        raise ValueError("this plan is the MXFP8 fork (q8/k8/v8/gate16 + sf_q/sf_k/sf_v outputs); launch it with run_fused_proj_gemm_mxfp8")
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

    if plan.mxfp8:
        raise ValueError("this plan is the MXFP8 fork (q8/k8/v8/gate16 + sf_q/sf_k/sf_v outputs); launch it with run_fused_proj_gemm_mxfp8")
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


def _check_fused_outputs(m: int, outs) -> None:
    """Every fused output is a TMA-store target: contiguous, 16-B aligned, its OWN width and dtype."""
    for name, t3, t_, width, dt, dt_name in outs:
        if tuple(t3.shape) != (1, m, width) or t3.dtype != dt or not t3.is_contiguous() or t3.data_ptr() % 16:
            raise ValueError(f"{name} must be a contiguous, 16-B-aligned {dt_name} [M={m}, {width}] tensor, got {tuple(t_.shape)} {t_.dtype}")


def _check_sf_out(name: str, sf: torch.Tensor, need: int, *, batch: int, heads: int, seq_len: int, d: int) -> torch.Tensor:
    """One F8_128x4 E8M0 scale-factor OUTPUT blob of the MXFP8 fork: uint8, contiguous, 16-B aligned, CUDA,
    ``B*H*ceil(S/128)*4*D`` bytes (the SDPA adapter's ``_reshape_sf`` count) -- returned flat."""
    if sf.dtype != torch.uint8:
        raise ValueError(f"{name} must be torch.uint8 (E8M0 bytes in F8_128x4 order), got {sf.dtype}")
    if not sf.is_cuda or not sf.is_contiguous() or sf.data_ptr() % 16:
        raise ValueError(f"{name} must be a contiguous, 16-B-aligned CUDA tensor")
    if sf.numel() != need:
        raise ValueError(f"{name} must hold B*H*ceil(S/128)*{4 * d} = {need} bytes for batch={batch} H={heads} S={seq_len} D={d}, got {sf.numel()}")
    return sf.view(-1)


def run_fused_proj_gemm_mxfp8(
    plan: FusedProjGemmPlan,
    a8: torch.Tensor,  # [M, K]        e4m3 codes (rank-2, or a rank-3 [1, M, K] view), M == batch * seq_len
    sf_a: torch.Tensor,  # F8_128x4 E8M0 blob of a8 over its M rows: sf_blob_bytes(M, K) bytes, uint8 / float8_e8m0fnu
    w8: torch.Tensor,  # [N_qkvg, K]   e4m3 codes (checkpoint layout, read transposed)
    sf_w: torch.Tensor,  # F8_128x4 E8M0 blob of w8 over its N rows: sf_blob_bytes(N, K) bytes
    out_q8: torch.Tensor,  # [M, h_q*d]   e4m3 contiguous (== compact BSHD [B, S, h_q, d])
    out_k8: torch.Tensor,  # [M, h_kv*d]  e4m3 contiguous
    out_v8: torch.Tensor,  # [M, h_kv*d]  e4m3 contiguous
    out_gate16: torch.Tensor,  # [M, h_q*d] bf16 contiguous
    out_sf_q: torch.Tensor,  # uint8, B*h_q*ceil(S/128)*4*d bytes: the SDPA's rowwise Q scale factors
    out_sf_k: torch.Tensor,  # uint8, B*h_kv*ceil(S/128)*4*d bytes: rowwise K
    out_sf_v: torch.Tensor,  # uint8, B*h_kv*ceil(S/128)*4*d bytes: COLUMNWISE V (D-plane-major)
    w_q_norm: Optional[torch.Tensor],  # [D] bf16; None iff params.qk_norm is False
    w_k_norm: Optional[torch.Tensor],  # [D] bf16
    cos: torch.Tensor,  # [M, ROPE_DIM] bf16 (or [B, S, ROPE_DIM] contiguous)
    sin: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    stream,
) -> None:
    """Launch the MXFP8 fork.  No allocation, no conversion; every check is cheap host arithmetic.

    Problem tuple: the block-scale rendering's ``(m, n, k, batch, a strides (m, k, l), b strides
    (n, k, l), q8 strides (m, n, l))`` plus the k8 / v8 / gate16 stride triples, in ELEMENTS,
    from rank-3 ``[1, M, K]``-shaped views.  ``sf_a`` / ``sf_w`` are the block-scale GEMM's OWN
    operand scale factors (what ``run_proj_gemm(..., sf_a=, sf_w=)`` binds): PADDED F8_128x4
    blobs (``sf_blob_bytes``), bound as the ``(1, rows_pad, 4*k4)`` e8m0 view -- the kernel
    reads a base pointer and rebuilds the layout from the problem size.  The three SF OUTPUTS
    are the SDPA's: ``B*H*ceil(S/128)*4*D`` bytes each, written in full (pad rows -> 0x00).

    **Typed decline**: ``seq_len % 128 != 0 and batch > 1`` raises ``NotImplementedError`` -- a
    128-row GEMM tile would straddle two sequences and the kernel's once-per-tile
    ``(b, s_tile)`` decode would be wrong for part of it.  The unfused block-scale path
    (``build_proj_gemm(block_scale=True)`` + ``quantize_mxfp8``) serves that shape.
    """
    from cuda.bindings import driver as cuda

    from .quantize_mxfp8 import n_sf_tiles, sf_bytes

    if not plan.mxfp8:
        raise ValueError("this plan is not the MXFP8 fork; launch the bf16 fork with run_fused_proj_gemm and the FP8 fork with run_fused_proj_gemm_fp8")
    p = plan.params
    if batch <= 0 or seq_len <= 0:
        raise ValueError(f"batch and seq_len must be positive, got batch={batch} seq_len={seq_len}")
    if seq_len % 128 and batch > 1:
        raise NotImplementedError(
            f"the fused MXFP8 projection decodes (b, s_tile) once per 128-row GEMM tile, so at batch={batch} > 1 the sequence length must be a "
            f"multiple of 128 (got {seq_len}: a tile would straddle two sequences); use the unfused block-scale path for this shape"
        )
    a3, w3 = _rank3(a8, "a8"), _rank3(w8, "w8")
    q3, k3, v3, g3 = _rank3(out_q8, "out_q8"), _rank3(out_k8, "out_k8"), _rank3(out_v8, "out_v8"), _rank3(out_gate16, "out_gate16")
    m, k = int(a3.shape[1]), int(a3.shape[2])
    n = int(w3.shape[1])
    if _FP8_E4M3 is None or a8.dtype != _FP8_E4M3 or w8.dtype != _FP8_E4M3:
        raise ValueError(f"the MXFP8 fused projection takes e4m3 codes, got a8 {a8.dtype}, w8 {w8.dtype}")
    if m != batch * seq_len:
        raise ValueError(f"M must equal batch*seq_len: a8 has M={m}, batch*seq_len={batch * seq_len}")
    if n != p.n_qkvg:
        raise ValueError(f"the fused projection classifies tiles for N={p.n_qkvg} (Q|GATE|K|V at {p.offsets}); the weight has N={n}")
    if int(w3.shape[2]) != k:
        raise ValueError(f"shape mismatch: a8 {tuple(a3.shape)}, w8 {tuple(w3.shape)}")
    if k % 32:
        raise ValueError(f"the block-scale GEMM needs K % 32 == 0 (one E8M0 scale per 32-element block), got K={k}")
    _check_fused_outputs(
        m,
        (
            ("out_q8", q3, out_q8, p.n_q, _FP8_E4M3, "e4m3"),
            ("out_k8", k3, out_k8, p.n_kv, _FP8_E4M3, "e4m3"),
            ("out_v8", v3, out_v8, p.n_kv, _FP8_E4M3, "e4m3"),
            ("out_gate16", g3, out_gate16, p.n_gate, torch.bfloat16, "bf16"),
        ),
    )
    cos2 = _rope_table_2d(cos, "cos", m, p.rope_dim, torch.bfloat16, "bf16")
    sin2 = _rope_table_2d(sin, "sin", m, p.rope_dim, torch.bfloat16, "bf16")
    _check_norm_weights(p, w_q_norm, w_k_norm, torch.bfloat16, "bf16")
    # The GEMM's own operand scale factors: the same padded F8_128x4 blobs / e8m0 views the unfused driver binds.
    sfa3 = _sf_view_of(sf_a, "sf_a", m, k, "fused_mxfp8")
    sfw3 = _sf_view_of(sf_w, "sf_w", n, k, "fused_mxfp8")
    n_tiles = n_sf_tiles(seq_len)
    sfq = _check_sf_out("out_sf_q", out_sf_q, sf_bytes(batch, p.h_q, seq_len, p.d_head), batch=batch, heads=p.h_q, seq_len=seq_len, d=p.d_head)
    sfk = _check_sf_out("out_sf_k", out_sf_k, sf_bytes(batch, p.h_kv, seq_len, p.d_head), batch=batch, heads=p.h_kv, seq_len=seq_len, d=p.d_head)
    sfv = _check_sf_out("out_sf_v", out_sf_v, sf_bytes(batch, p.h_kv, seq_len, p.d_head), batch=batch, heads=p.h_kv, seq_len=seq_len, d=p.d_head)

    def _st(t3):
        st = t3.stride()
        return (st[1], st[2], st[0])

    problem = (m, n, k, 1, *_st(a3), *_st(w3), *_st(q3), *_st(k3), *_st(v3), *_st(g3))
    # The artifact's fakes are (M|N, K, L) / (M, N, L): the same axis relabel the
    # shipped lowering applies (`compiler._lower`: `v.permute(1, 2, 0)`).  The SF
    # operands reach the kernel as base pointers only (fully symbolic fakes).
    plan.launch(
        problem,
        a3.permute(1, 2, 0),
        w3.permute(1, 2, 0),
        sfa3,
        sfw3,
        q3.permute(1, 2, 0),
        k3.permute(1, 2, 0),
        v3.permute(1, 2, 0),
        g3.permute(1, 2, 0),
        w_q_norm,
        w_k_norm,
        cos2,
        sin2,
        sfq,
        sfk,
        sfv,
        _cutlass_int32(seq_len),
        _cutlass_int32(batch * p.h_kv * n_tiles),
        cuda.CUstream(int(stream)),
    )


def _cutlass_int32(v: int):
    """A ``cutlass.Int32`` runtime scalar for the fused-kernel ABI (imported lazily: ``cutlass`` is a DSL import)."""
    import cutlass

    return cutlass.Int32(int(v))


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
_FP4_X2 = getattr(torch, "float4_e2m1fn_x2", None)  # e2m1 codes, TWO per byte along the contiguous (K) axis
_E8M0 = getattr(torch, "float8_e8m0fnu", None)


def _is_fp8(dtype: torch.dtype) -> bool:
    return _FP8_E4M3 is not None and dtype == _FP8_E4M3


def _is_fp4(dtype: torch.dtype) -> bool:
    """``torch.float4_e2m1fn_x2`` -- the ONLY fp4 storage dtype the driver accepts.  A uint8 blob
    holding packed e2m1 codes is declined with the ``.view(torch.float4_e2m1fn_x2)`` hint (torch
    2.13 can VIEW but not cast to fp4), so the dtype carries the packing through every check."""
    return _FP4_X2 is not None and dtype == _FP4_X2


def _cudnn_dtype(dtype: torch.dtype):
    import cudnn

    table = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF}
    if _FP8_E4M3 is not None:
        table[_FP8_E4M3] = cudnn.data_type.FP8_E4M3
    if _FP4_X2 is not None:
        table[_FP4_X2] = cudnn.data_type.FP4_E2M1
    try:
        return table[dtype]
    except KeyError:
        raise ValueError(f"proj_gemm serves bf16/f16/fp8-e4m3/fp4-e2m1 (float4_e2m1fn_x2) only, got {dtype}") from None


def _dtype_word(dtype: torch.dtype) -> str:
    return "e2m1" if _is_fp4(dtype) else "e4m3" if _is_fp8(dtype) else str(dtype).replace("torch.", "")


# The block-scale operand PAIRS this driver claims -- one row each, exactly as the FROST GEMM
# catalog spells them (``gemm/frost/kernel_registry.py`` ``_BLOCK_SCALE_CASES``; every row shares
# reorder F8_128x4, fp32 dequant, fp32 accumulate):
#
#   A e4m3  x  W e4m3   with E8M0 scales per 32   -- MXFP8 x MXFP8, today's stage (1)
#   A e4m3  x  W e2m1   with E8M0 scales per 32   -- the MIXED row: MXFP8 activations x MXFP4 weight
#   A e2m1  x  W e4m3   with E8M0 scales per 32   -- the mixed row the other way round
#   A e2m1  x  W e2m1   with E4M3 scales per 16   -- NVFP4 x NVFP4
#   A e2m1  x  W e2m1   with E8M0 scales per 32   -- MXFP4 x MXFP4
#
# Nothing else the catalog lists (e5m2 sides, E5M3 scales, e4m3 scales at block 32 -- SM 10.7-only)
# is claimed: the block never quantizes to it, and an unlisted pair is a typed error, never a
# fallback.  The pairing is a RULE and not a default because the DECLARED scale dtype selects the
# MMA's scale format (``gemm/frost/fusion_ir.py`` ``sf_scale_format``): an e4m3 blob declared E8M0
# does not fail, it miscomputes.
SF_BLOCK_SIZES = (16, 32)


def _sf_torch_dtype(sf_dtype) -> torch.dtype:
    """The torch STORAGE dtype of a scale-factor blob declared as ``sf_dtype`` (a ``cudnn.data_type``;
    ``None`` = the E8M0 default) -- the dtype ``_sf_view_of`` re-views a uint8 blob as."""
    import cudnn

    if sf_dtype is None or sf_dtype == cudnn.data_type.FP8_E8M0:
        return _E8M0
    if sf_dtype == cudnn.data_type.FP8_E4M3:
        return _FP8_E4M3
    raise ValueError(f"block-scale scale factors are FP8_E8M0 or FP8_E4M3 bytes (cudnn.data_type), got {sf_dtype}")


def check_sf_torch_dtype(sf_dtype) -> torch.dtype:
    """:func:`_sf_torch_dtype` plus the torch-version check: a typed ``ValueError`` at PLAN construction
    (``block_scale_pairing`` calls it) rather than at the first ``_sf_view`` of a launch, so a plan whose
    scale dtype this torch cannot even name never exists."""
    dt = _sf_torch_dtype(sf_dtype)
    if dt is None:
        raise ValueError(f"this torch ({torch.__version__}) has no storage dtype for {sf_dtype} scale factors (needs torch.float8_e8m0fnu / float8_e4m3fn)")
    return dt


def block_scale_pairing(*, dtype: torch.dtype, w_dtype: torch.dtype, sf_dtype, block_size: int, label: str):
    """Resolve and check the ``(sf_dtype, block_size)`` of a block-scale plan over ``A: dtype`` x ``W: w_dtype``
    against the claimed pairs above; returns the resolved pair.

    ``sf_dtype=None`` resolves from the operands and the block: E4M3 for two e2m1 sides at block 16,
    E8M0 otherwise -- every claimed pair has exactly one scale dtype per block size, so the default is
    derived, never guessed.  ``block_size`` is never defaulted here (the caller's 32 is the MX contract).
    Raises ``ValueError`` naming the pair; ``_Projection.check_support`` re-raises it as the block's
    typed ``NotImplementedError``, ``build_proj_gemm`` lets it propagate.
    """
    import cudnn

    e8m0, e4m3 = cudnn.data_type.FP8_E8M0, cudnn.data_type.FP8_E4M3
    for side, dt in (("dtype (A)", dtype), ("w_dtype (W)", w_dtype)):
        if not (_is_fp8(dt) or _is_fp4(dt)):
            raise ValueError(
                f"{label}: block_scale=True needs e4m3 or e2m1 codes on each operand (torch.float8_e4m3fn | torch.float4_e2m1fn_x2); {side} is {dt}"
            )
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size not in SF_BLOCK_SIZES:
        # An int, not merely ``== 16``: ``16.0`` would flow float dims into ``sf_padded_dims`` and the graph.
        raise ValueError(f"{label}: block_size must be one of {SF_BLOCK_SIZES} (an int; one scale per block along K), got {block_size!r}")
    pair = f"A {_dtype_word(dtype)} x W {_dtype_word(w_dtype)}"
    if _is_fp4(dtype) and _is_fp4(w_dtype):
        served = {(e4m3, 16): "NVFP4 x NVFP4", (e8m0, 32): "MXFP4 x MXFP4"}
        if sf_dtype is None:
            sf_dtype = e4m3 if block_size == 16 else e8m0
    else:
        served = {(e8m0, 32): "MXFP8 x MXFP8" if _is_fp8(dtype) and _is_fp8(w_dtype) else "the mixed MXFP8 x MXFP4 row"}
        if sf_dtype is None:
            sf_dtype = e8m0
    if (sf_dtype, block_size) not in served:
        menu = "; ".join(f"{name}: {sf} per {blk}" for (sf, blk), name in served.items())
        raise ValueError(
            f"{label}: no block-scale GEMM row for {pair} with {sf_dtype} scales per {block_size} elements "
            f"(kernel_registry._BLOCK_SCALE_CASES); for this operand pair the driver serves {menu}"
        )
    check_sf_torch_dtype(sf_dtype)  # the blob's storage dtype exists on this torch -- decline here, not at the first launch
    return sf_dtype, block_size


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
    # Block-scale (MXFP8) additions (append-only). `sfa` / `sfb` are the graph's
    # F8_128x4 scale-factor tensors when the plan was built with `block_scale=True`,
    # else None; `route` records which path serves the launch:
    #   "graph+jit"  the backend admitted the graph, `frost_gemm` is pinned in its
    #                ranked list (workspace / plan-name honesty) AND the JIT at the
    #                resolved tile config is what `run_proj_gemm` launches;
    #   "jit-only"   the backend declined the graph (a typed cudnnGraphNotSupportedError,
    #                or no `frost_gemm` in its ranked list -- `_NoFrostPlan`); the JIT
    #                never touches the backend, so the kernel still runs.  Any OTHER
    #                backend error propagates (it is a bug: e.g. the unpadded F8_128x4
    #                declaration this driver used to make, which read as "declined" until
    #                the catch was narrowed);
    #   "graph"      dense, no forced tile: the graph engine's own plan runs.
    block_scale: bool = False
    sfa: Any = None
    sfb: Any = None
    sf_dtype: Any = None  # cudnn.data_type of the scale factors (FP8_E8M0 | FP8_E4M3)
    route: Optional[str] = None
    # fp4 additions (append-only). `w_dtype` is W's torch dtype (== `dtype` unless the plan was built
    # with a per-weight `w_dtype`; None on a hand-built plan means "same as dtype"); `block_size` is the
    # scale block along K (32 for every E8M0 pair, 16 for NVFP4) -- what `_sf_view` sizes the blobs by.
    w_dtype: Any = None
    block_size: int = 32
    # Backward-driver additions (append-only; the defaults spell the forward's declaration).
    # `a_major` / `b_major` name the axis the graph DECLARED contiguous for A / B:
    #   A "k": dim [1, m, k] stride [m*k, k, 1]  (a row-major [m, k])
    #     "m": dim [1, m, k] stride [m*k, 1, m]  (the TRANSPOSE of a row-major [k, m] -- dY^T of a wgrad)
    #   B "k": dim [1, k, n] stride [k*n, 1, k]  (a row-major [n, k] weight, read transposed -- the forward)
    #     "n": dim [1, k, n] stride [k*n, n, 1]  (a row-major [k, n] -- X of a wgrad, W of a dgrad)
    # `split_k` is the REQUESTED knob (0 = the driver's pick, `build_proj_gemm`); the slices
    # that run are `plan.jit.config.split_k_slices` on the JIT route, and `tile_config_name`
    # spells them the catalog's way (`..._cluster2x1_2ctamma_splitK2` for S=2; no suffix at 1).
    a_major: str = "k"
    b_major: str = "k"
    split_k: int = 0

    @property
    def has_alpha(self) -> bool:
        return self.alpha is not None

    @property
    def workspace_bytes(self) -> int:
        # The JIT artifact's own scratch (split-K fp32 partials) counts on EVERY route that has
        # one: `run_proj_gemm` launches `plan.jit` whenever it exists, and the graph's number is
        # the backend plan's, which knows nothing of a `split_k=` the driver pinned on the JIT
        # config.  The max of both keeps `gemm_scratch = max(plan.workspace_bytes)` honest under
        # either route (a forgotten reducer workspace does not fail the launch).
        jit_ws = int(getattr(self.jit, "workspace_bytes", 0) or 0) if self.jit is not None else 0
        if self.route == "jit-only":
            # No backend plan exists to ask; the JIT artifact knows its own (split-K only).
            return max(jit_ws, 1)
        return max(int(self.graph.get_workspace_size()), jit_ws, 1)

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
    block_scale: bool = False,
    sf_dtype: Any = None,
    mma_tile_k_bytes: Optional[int] = None,
    w_dtype: Optional[torch.dtype] = None,
    block_size: int = 32,
    a_major: str = "k",
    b_major: str = "k",
    split_k: int = 0,
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

    **Block-scale (MXFP8) inputs.** ``block_scale=True`` (e4m3 ``dtype`` required,
    ``alpha`` must be False -- the E8M0 dequant is exact and happens IN the MMA,
    so there is nothing to descale -- and ``K % 32 == 0``) declares two
    per-32-block E8M0 scale-factor tensors in cuDNN's ``F8_128x4`` order --
    declared at the PADDED atom extents ``(1, ceil(rows/128)*128, ceil(K/32/4)*4)``
    (:func:`sf_padded_dims`), which the backend requires for that reorder -- and
    routes A / B through ``block_scale_dequantize`` before the matmul, as
    ``test/python/gemm/frost/test_block_scale_matmul.py`` builds it.  The
    caller binds the same PADDED blobs at execute (``run_proj_gemm(..., sf_a=, sf_w=)``).
    ``sf_dtype`` defaults to ``cudnn.data_type.FP8_E8M0`` (spelled ``None`` here so
    ``cudnn`` stays a lazy import).  The graph goes to the backend under
    ``[heur_mode.A, heur_mode.FALLBACK]``; if the backend declines it (a typed
    ``cudnnGraphNotSupportedError``, or no ``frost_gemm`` in the ranked list) the
    plan takes the JIT-ONLY route -- ``jit_from_cudnn_graph`` never touches the
    backend -- and says so in ``plan.route``.  Under ``pin_frost`` the block-scale
    kernel is ALWAYS the JIT at the resolved tile config (the forced 256-wide
    tile when 256 divides N, else the auto pick), so ``plan.tile_config_name`` /
    ``plan.mma_tile_k_bytes`` name what runs.

    Without ``mma_tile_k_bytes`` a FORCED block-scale tile follows the engine's
    ``preferred_mma_tile_k_bytes`` (64 on SM 10.7, else the named config's 32).
    ``mma_tile_k_bytes`` (block-scale only; 32 or 64) re-targets the resolved
    config's MMA-instruction K width through ``tile_config.as_mma_tile_k`` --
    PR-B decision D15: the width is A/B'd, never assumed.  ``None`` keeps the
    config's own width (the named 256-wide config is the K=32 form).

    **fp4 (E2M1) operands -- block-scale only.** ``w_dtype`` (default: ``dtype``)
    names W's dtype separately from A's, so stage (1) can run the catalog's MIXED
    row -- e4m3 ``h`` against a ``torch.float4_e2m1fn_x2`` weight -- and the fp4
    out projection can run the both-fp4 rows; ``block_size`` (32 | 16) is the scale
    block along K.  The served pairs are exactly :func:`block_scale_pairing`'s table
    (``kernel_registry._BLOCK_SCALE_CASES``): both e4m3 or exactly one e2m1 side ->
    E8M0 per 32; both e2m1 -> E4M3 per 16 (NVFP4) or E8M0 per 32 (MXFP4); any other
    ``(sf_dtype, block_size)`` is a ``ValueError`` -- the declared scale dtype selects
    the MMA's scale format, so a wrong one miscomputes rather than fails.  The graph
    keeps LOGICAL dims (``A [1, M, K]``, ``B [1, K, N]``) and declares the fp4 side's
    ``data_type``; the caller binds the PACKED storage (``[M, K/2]`` / ``[N, K/2]``,
    two codes per byte, low nibble = even k), which the kernel accepts as
    ``shape[-1] * kpack == K``.  An e4m3 scale blob is re-viewed as
    ``torch.float8_e4m3fn`` at bind (``_sf_view``).  ``alpha`` stays forbidden with
    ``block_scale`` for every pair, and an fp4 side without ``block_scale`` is a
    ``ValueError`` (there is no dense fp4 MMA).

    **Operand majors -- the backward's transposed operands (appended kwargs).**
    ``a_major`` (``"k"`` | ``"m"``) and ``b_major`` (``"k"`` | ``"n"``) pick the graph
    STRIDES of A / B; the dims stay ``[1, m, k]`` / ``[1, k, n]`` and the FROST GEMM
    engine infers each operand's major from them (``graph_analyzer._infer_a_major`` /
    ``_infer_b_major``) and renders its M-major-A / N-major-B template arms.  The
    defaults are the forward's declaration, byte-identical graph.  ``nn.Linear``
    backward (``api_bwd.py``): ``dW = dY^T @ X`` is ``a_major="m", b_major="n"`` with
    ``m=rows, k=T, n=cols`` and ``dX = dY @ W`` is ``a_major="k", b_major="n"`` with the
    UN-transposed row-major weight -- :func:`run_wgrad_gemm` / :func:`run_dgrad_gemm`
    build the views and refuse a mismatch.  The TMA 16-byte contiguous-extent rule
    now falls on the MN extent (``M % (16/BPE)`` for an M-major A, ``N % (16/BPE)``
    for an N-major B) and is a typed ``ValueError`` HERE, before any graph exists.
    Served on the dense bf16 / f16 path only: the block-scale rows lay their
    scale-factor blobs over K-major operand rows and are refused with a non-K major
    (``ValueError``), and an fp8 (e4m3) operand with a non-K major is a typed
    ``NotImplementedError`` until the quantized backward (its own fp8 GEMM drivers)
    measures that rendering on cc 10.7 -- the forward's fp8 plans keep the K-major
    defaults and are untouched.

    **``split_k`` -- three values.** ``0`` (default) is the driver's pick:
    the FORCED tile's catalog ``split_k_slices=1`` on the JIT route -- the heuristic's
    ``_auto_split_k`` (``compiler.py``) is reached ONLY if the forced compile fell back
    to the graph route.  ``1`` PINS ``replace(cfg, split_k_slices=1)`` on the JIT config
    and REFUSES the graph fallback (:class:`SplitKPinRefused`): the recompute plans
    of the backward need this for their bit-identical claim, since a fallback would
    change config, route and slice count.  ``S >= 2`` is ``replace(cfg,
    split_k_slices=S)`` on the JIT config -- the two-kernel split with the FIXED-ORDER
    reducer (``kernel_templates/split_k_reduction_epilogue_fusion.py``; deterministic)
    and fp32 partials of ``S*M*N*4`` bytes, reported through ``plan.workspace_bytes``
    and handed to the JIT by :func:`run_proj_gemm` (``compiler._check_splitk_supported``
    gates S).  A pinned split without a forced name takes the engine's auto pick as
    the base geometry (``_auto_tile_config``), so it is always a JIT plan -- which is
    why ``split_k >= 1`` needs ``pin_frost=True``: with the FROST engine deselected there
    is no JIT to pin, and the combination is a ``ValueError`` (a knob is honoured or
    refused, never silently dropped).
    """
    import cudnn

    if w_dtype is None:
        w_dtype = dtype
    if a_major not in _A_MAJORS:
        raise ValueError(f"{label}: a_major must be one of {_A_MAJORS} (the axis of A [1, m, k] the graph declares contiguous), got {a_major!r}")
    if b_major not in _B_MAJORS:
        raise ValueError(f"{label}: b_major must be one of {_B_MAJORS} (the axis of B [1, k, n] the graph declares contiguous), got {b_major!r}")
    if isinstance(split_k, bool) or not isinstance(split_k, int) or split_k < 0:
        raise ValueError(
            f"{label}: split_k must be an int >= 0 (0 = the driver's pick, 1 = pin one slice on the JIT, S >= 2 = split-K with S slices), got {split_k!r}"
        )
    if split_k >= 1 and not pin_frost:
        raise ValueError(
            f"{label}: split_k={split_k} pins a FROST JIT config and cannot be combined with pin_frost=False (the deselected-FROST graph route "
            "has no JIT to pin, so the knob would be silently dropped)"
        )
    if (a_major != "k" or b_major != "k") and (block_scale or _is_fp4(dtype) or _is_fp4(w_dtype)):
        raise ValueError(
            f"{label}: M-major A / N-major B are served on the dense path only (got a_major={a_major!r}, b_major={b_major!r} with "
            f"block_scale={block_scale}, dtype={dtype}, w_dtype={w_dtype}); the block-scale rows' F8_128x4 scale-factor blobs are laid out over K-major operand rows"
        )
    if (a_major != "k" or b_major != "k") and (_is_fp8(dtype) or _is_fp8(w_dtype)):
        raise NotImplementedError(
            f"{label}: an fp8 (e4m3) operand with a_major={a_major!r}, b_major={b_major!r} is not served by this driver yet (dtype={dtype}, "
            f"w_dtype={w_dtype}): the MN-major fp8 rendering is unmeasured on cc 10.7 and the quantized backward lands its own fp8 GEMM drivers; "
            "use bf16 / f16 operands here, or the K-major defaults"
        )
    if split_k and block_scale:
        raise ValueError(
            f"{label}: split_k={split_k} on the block-scale GEMM is not served by this driver (its reducer is verified on the dense path only); use split_k=0"
        )
    _check_mn_major_tma_rule(label, "A", a_major, m, "M", dtype)
    _check_mn_major_tma_rule(label, "B", b_major, n, "N", w_dtype)
    if block_scale:
        sf_dtype, block_size = block_scale_pairing(dtype=dtype, w_dtype=w_dtype, sf_dtype=sf_dtype, block_size=block_size, label=label)
        if alpha:
            raise ValueError(f"{label}: block_scale=True carries its descale in the per-block scale factors; alpha must be False")
        if k % 32:
            # Covers both the scale blocks (32, or 2 x 16) and TMA's 16-byte contiguous-extent rule
            # for a packed e2m1 operand (16 bytes = 32 codes; `gemm/frost/recipe.py` contiguous_modulus).
            raise ValueError(
                f"{label}: block_scale=True needs K % 32 == 0 (one E8M0 scale per 32-element block, or two E4M3 ones; the fp4 TMA rule), got K={k}"
            )
        if mma_tile_k_bytes not in (None, 32, 64):
            raise ValueError(f"{label}: mma_tile_k_bytes must be None, 32 or 64 (the tcgen05 MMA K widths), got {mma_tile_k_bytes!r}")
        if isinstance(tile_config, str) and tile_config != "auto" and not tile_config.startswith("CONFIG_sm100_"):
            # The sm103 block-scale template declares its rings A, B, SFA, SFB, so on Rubin's 327 KiB
            # budget its scale-factor roots cross the 256 KiB tcgen05-descriptor line and the kernel
            # returns NaN (pre-existing; `mma-tma-matrix.md` section 6).  A block-scale plan takes the
            # sm100 pipeline's catalog only -- refused HERE, before any backend or JIT sees the graph.
            raise ValueError(
                f"{label}: a block-scale plan takes an sm100-pipeline tile config (CONFIG_sm100_*), got {tile_config!r}; "
                "the sm103 block-scale template's SF rings sit past the 256 KiB descriptor line on Rubin and return NaN"
            )
    else:
        if _is_fp4(dtype) or _is_fp4(w_dtype):
            raise ValueError(
                f"{label}: fp4 (e2m1) operands ride the block-scale GEMM only -- pass block_scale=True with their scale factors; got dtype={dtype}, w_dtype={w_dtype}"
            )
        if w_dtype != dtype:
            raise ValueError(f"{label}: the dense GEMM takes one operand dtype (A {dtype} vs W {w_dtype}); mixed dtypes exist only as block-scale rows")
        if mma_tile_k_bytes is not None:
            raise ValueError(f"{label}: mma_tile_k_bytes is a block-scale knob (D15); the dense path keeps the engine's preferred width")
        if sf_dtype is None:
            sf_dtype = cudnn.data_type.FP8_E8M0
    # The op-recording hook the JIT-only route reads the graph through is installed by this
    # import; the graph engine imports it too, but a backend that declines the graph early
    # would otherwise leave the JIT nothing to analyze.
    import cudnn.gemm.frost  # noqa: F401

    for lbl, v in (("m", m), ("k", k), ("n", n)):
        if v <= 0:
            raise ValueError(f"{label}: {lbl} must be > 0, got {v}")
    fp8 = _is_fp8(dtype)
    fp4_a, fp4_w = _is_fp4(dtype), _is_fp4(w_dtype)
    if fp8 and k % 16:
        # TMA's 16-byte contiguous-extent rule at 1 B/elem (compiler._tma_alignment_reject).
        raise ValueError(f"{label}: FP8 operands need K % 16 == 0 (16-byte TMA rule at 1 B/elem), got K={k}")
    if out_dtype is None:
        out_dtype = torch.bfloat16 if (fp8 or fp4_a) else dtype
    out_dt = _cudnn_dtype(out_dtype)
    # The graph's io dtype is what A / B take when they carry no explicit `data_type=`.  An fp4 side
    # ALWAYS declares its own (below), so with an fp4 A the io dtype names no tensor at all and the
    # output's dtype stands in; every non-fp4 plan keeps the io dtype it always had (byte-identical graph).
    io_dt = out_dt if fp4_a else _cudnn_dtype(dtype)

    g = cudnn.pygraph(
        io_data_type=io_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # Dims are LOGICAL for an fp4 side too (K codes, not K/2 bytes): the analyzer reads the
    # element type off `data_type=` and the binding's `shape[-1] * kpack == K` check accepts the
    # packed [.., K/2] storage -- exactly how test_block_scale_matmul.py declares its fp4 operands.
    # A is a row-major [M, K] (K-major, the forward) or the transpose of a row-major [K, M]
    # (M-major: stride[1] == 1, a wgrad's dY^T); the engine reads the major off the stride.
    a = g.tensor(name="A", dim=[1, m, k], stride=[m * k, k, 1] if a_major == "k" else [m * k, 1, m], **({"data_type": _cudnn_dtype(dtype)} if fp4_a else {}))
    # B is the weight as a checkpoint stores it -- [N, K] row-major -- declared
    # transposed. stride[1] == 1 is what makes it the K-contiguous operand (the forward);
    # an N-major B (stride[2] == 1) is a row-major [K, N] -- a wgrad's X or a dgrad's
    # un-transposed W.  It names its dtype only when it differs from the graph's io dtype
    # (an fp4 W, or W of the other side's dtype).
    b = g.tensor(
        name="B",
        dim=[1, k, n],
        stride=[k * n, 1, k] if b_major == "k" else [k * n, n, 1],
        **({"data_type": _cudnn_dtype(w_dtype)} if (fp4_w or w_dtype != dtype) else {}),
    )
    sfa_t = sfb_t = None
    if block_scale:
        # One E8M0 scale per 32-element K block, in cuDNN's F8_128x4 (128 rows x 4 blocks
        # = 512 B atoms, row-major) order: SFA over A's rows, SFB over B's N (the weight's
        # rows).  Declared at the PADDED atom extents -- rows to 128, K-blocks to 4 -- which
        # is (i) what the backend's block_scale_dequantize finalize REQUIRES for F8_128x4
        # ("the scale tensor ... must be (1, 1024, 128) but found (1, 1000, 128)",
        # CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH -- a bare RuntimeError, not a typed decline;
        # measured 2026-09-15 on the dev node at M=1000, K=544 (sf_k 17 -> 20) and N=392), and
        # (ii) exactly the padded blob `run_proj_gemm` binds (`sf_blob_bytes`).  The FROST JIT
        # never reads these dims (the analyzer takes the SF dtype / reorder; the lowering
        # derives ceil(K/32/4) x ceil(rows/128) from A / B), so the declaration is inert there.
        # test_block_scale_matmul.py:81-112 declares the UNPADDED [1, M, K/32] -- fine for its
        # JIT-only path, rejected by the backend the moment M % 128 != 0.
        m_pad, sf_k = sf_padded_dims(m, k, block_size)
        n_pad, _ = sf_padded_dims(n, k, block_size)
        sfa_t = g.tensor(name="SFA", dim=[1, m_pad, sf_k], stride=[m_pad * sf_k, sf_k, 1], data_type=sf_dtype, reordering_type=cudnn.tensor_reordering.F8_128x4)
        sfb_t = g.tensor(name="SFB", dim=[1, sf_k, n_pad], stride=[sf_k * n_pad, 1, sf_k], data_type=sf_dtype, reordering_type=cudnn.tensor_reordering.F8_128x4)
        ad = g.block_scale_dequantize(input=a, descale=sfa_t, block_size=[1, block_size])
        bd = g.block_scale_dequantize(input=b, descale=sfb_t, block_size=[block_size, 1])
        mm = g.matmul(A=ad, B=bd, name=label)
    else:
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

    plan = ProjGemmPlan(
        graph=g,
        a=a,
        b=b,
        c=c,
        m=m,
        k=k,
        n=n,
        label=label,
        dtype=dtype,
        out_dtype=out_dtype,
        alpha=alpha_t,
        block_scale=block_scale,
        sfa=sfa_t,
        sfb=sfb_t,
        w_dtype=w_dtype,
        block_size=block_size,
        a_major=a_major,
        b_major=b_major,
        split_k=split_k,
    )
    plan.sf_dtype = sf_dtype if block_scale else None
    plan.tile_config_name = "heuristic (graph engine)"
    plan.route = "graph"

    def _backend_pin() -> None:
        """validate -> build -> rank -> pin (or deselect) -> check_support -> build_plans on the backend."""
        g.validate()
        g.build_operation_graph()
        # The block-scale graph is ranked under [A, FALLBACK] (as the FROST GEMM suite's own
        # `_pin_frost` does); the dense graph keeps the A-only list it always had.
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK] if block_scale else [cudnn.heur_mode.A])
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        if pin_frost:
            # Plan names carry the plan's knobs since the shared knob vocabulary
            # landed ("frost_gemm[CTA_GROUP=2, MMA_TILE_K=32, ...]"), so the pin
            # matches the engine NAME (exact, or followed by its knob bracket) and
            # takes the top-ranked FROST configuration; deselect_engines already
            # matches by substring (the engine name is the user-visible API).
            frost_idx = _frost_plan_index(names)
            if frost_idx is None:
                raise _NoFrostPlan(f"{label}: the {_FROST_GEMM_PLAN!r} plan is not in the ranked list {names}. {_why_no_frost_plan()}")
            g.select_plan(frost_idx)
        else:
            g.deselect_engines([_FROST_GEMM_PLAN])
        g.check_support()
        g.build_plans()

    if block_scale and pin_frost:
        # A backend that does not admit the e4m3 + E8M0 two-dequant graph is a typed decline
        # (cudnnGraphNotSupportedError), and "no frost_gemm in the list" is the `_NoFrostPlan`
        # above; both leave the JIT-only route, which never asks the backend.  Anything else --
        # a backend failure inside check_support / build_plans, a bare RuntimeError -- is a
        # bug and PROPAGATES (engine contract rule 2); the catch is deliberately this narrow.
        # WARNING, not INFO: on this route the plan-name pin is absent and `workspace_bytes`
        # is the JIT artifact's, which a caller reading the perf table should see without -v.
        try:
            _backend_pin()
        except (cudnn.cudnnGraphNotSupportedError, _NoFrostPlan) as exc:
            plan.route = "jit-only"
            _LOG.warning("%s: backend declined the block-scale graph (%s: %s); taking the JIT-only route", label, type(exc).__name__, str(exc)[:200])
    else:
        _backend_pin()

    # Forced tile: JIT the SAME graph at a named config. The graph engine picks
    # its config internally with no override hook (graph_analyzer.py:571-574),
    # so taking a specific tile means going through the JIT entry point. The
    # graph is still built above -- it is what gets analyzed, and it keeps
    # `workspace_bytes` and the plan-name pin honest.  A block-scale plan under
    # `pin_frost` ALWAYS takes the JIT (forced name, else the auto pick) so one
    # binding path (`bd.sfa_operands` / `bd.sfb_operands`) serves every shape.
    name = _forced_tile_config(n) if tile_config == "auto" else tile_config
    pinned_split = split_k >= 1  # split_k=1 pins ONE slice (and refuses the fallback); S >= 2 splits
    if pin_frost and (name or block_scale or pinned_split):
        from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
        from cudnn.gemm.frost.tile_config import as_mma_tile_k, by_name

        cfg = by_name(name) if name else _auto_tile_config(g)
        if mma_tile_k_bytes is not None:
            cfg = as_mma_tile_k(cfg, mma_tile_k_bytes)
            if cfg.mma_tile_k_bytes != mma_tile_k_bytes:
                raise ValueError(f"{label}: config {cfg.name!r} cannot issue mma_tile_k_bytes={mma_tile_k_bytes} (it stays at {cfg.mma_tile_k_bytes})")
        elif block_scale and name:
            # A FORCED name carries the catalog's K=32 spelling, which would silently pin the block-scale GEMM to the
            # narrow MMA form the engine's own auto path abandons on Rubin (`preferred_mma_tile_k_bytes`: 64 wherever the
            # active GPU issues the 64-byte block-scale MMA, 32 elsewhere -- so SM100 is unchanged).  Measured on the
            # 397B geometry, perf node c09 @2376 MHz, S=16K: MXFP8 qkv+gate proj 0.344 -> 0.298 ms (82 -> 95 % of the fp8
            # MMA peak), the mixed fp8 x fp4 row 0.332 -> 0.268 ms, the fp4 x fp4 out proj 0.114 -> 0.107 (NVFP4) /
            # 0.113 -> 0.102 ms (MXFP4); every S in {4K, 16K, 32K} and every mode moved the same way, so the forced tile
            # follows the engine's preference and an explicit `mma_tile_k_bytes` stays the override.
            from cudnn.gemm.frost.graph_analyzer import analyze
            from cudnn.gemm.frost.kernel_registry import preferred_mma_tile_k_bytes

            cfg = as_mma_tile_k(cfg, preferred_mma_tile_k_bytes(analyze(g)))
        if pinned_split:
            # The catalog entry carries split_k_slices=1 and `jit_from_cudnn_graph` never calls
            # `_auto_split_k`, so this replace is the ONLY way a slice count reaches the JIT.
            from dataclasses import replace

            cfg = replace(cfg, split_k_slices=split_k)
        try:
            compiled = jit_from_cudnn_graph(g, config=cfg)
        except Exception as exc:  # a config this shape cannot take is a FALLBACK, not a failure
            if tile_config != "auto" or block_scale or mma_tile_k_bytes is not None:
                raise  # explicit requests (and the block-scale path, which has no graph fallback) surface their reason
            if pinned_split:
                # The graph heuristic would change the config, the route and the slice count -- exactly
                # what a pinned split forbids -- so the compiler's own decline is surfaced, typed, instead.
                raise SplitKPinRefused(
                    f"{label}: split_k={split_k} pins the JIT at {cfg.name!r}; its compile was refused ({type(exc).__name__}: {str(exc)[:300]}) "
                    "and the graph-heuristic fallback is refused too (it would change the tile config, the route and the slice count)"
                ) from exc
            compiled = None
            # WARNING, not DEBUG (like the block-scale twin above): the fallback changes the tile config, the
            # route and possibly the slice count (`_auto_split_k`), under a different plan name -- a perf table
            # or a bit-identical claim built on it must see the switch without -v.
            _LOG.warning(
                "%s: forced tile %s rejected (%s: %s); falling back to the graph heuristic (tile config, route and slice count change)",
                label,
                name,
                type(exc).__name__,
                str(exc)[:200],
            )
        if compiled is not None:
            plan.jit, plan.jit_binding = compiled, compiled.binding
            plan.tile_config_name = getattr(compiled.config, "name", name)
            plan.mma_tile_k_bytes = getattr(compiled.config, "mma_tile_k_bytes", None)
            if plan.route == "graph":
                plan.route = "graph+jit"
    return plan


def _auto_tile_config(g):
    """The tile config the FROST graph engine's automatic strategy would pick for ``g``
    (``select_config`` + ``preferred_strategy``), for a JIT with no forced name."""
    from cudnn.gemm.frost.compiler import _graph_dynamic_shapes, plan_config
    from cudnn.gemm.frost.graph_analyzer import analyze

    return plan_config(analyze(g), dynamic_shapes=_graph_dynamic_shapes(g))


def run_proj_gemm(
    plan: ProjGemmPlan,
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    alpha: Optional[torch.Tensor] = None,
    sf_a: Optional[torch.Tensor] = None,
    sf_w: Optional[torch.Tensor] = None,
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

    ``sf_a`` / ``sf_w`` (block-scale plans only, both required): the F8_128x4
    E8M0 scale-factor blobs of ``a`` (over its M rows) and ``w`` (over its N
    rows), uint8 or ``float8_e8m0fnu``, contiguous, PADDED to whole atoms --
    ``ceil(rows/128)*128 * ceil(K/32/4)*4`` bytes, exactly what
    ``swizzle_sf_rowwise`` / ``to_blocked`` of the padded logical ``[rows, K/32]``
    scale matrix produces.  They are bound as rank-3 ``(1, rows_pad, 4*k4)``
    VIEWS (a uint8 blob is re-viewed as e8m0, no copy) -- the kernel reads a base
    pointer and rebuilds the layout from the problem size.

    Both routes take the SAME variant pack keyed by the plan's graph tensors
    (``plan.a`` / ``plan.b`` / ``plan.c`` / ``plan.sfa`` / ``plan.sfb`` / ``plan.alpha``):
    the JIT resolves by tensor identity as the graph does, and graph tensors are
    invariant under a ``swap_ab`` tile config (which swaps the binding's LISTS).

    **Operand dtype and fp4 extent are CHECKED here** (``a.dtype == plan.dtype``,
    ``w.dtype == plan.w_dtype``, ``out.dtype == plan.out_dtype``; an e2m1 side's storage
    ``shape[-1] * 2 == K``).  The graph carries the dtypes and the kernel binds a pointer,
    so before this a bf16 ``w`` handed to an fp8 plan was silently REINTERPRETED
    byte-for-byte -- the one deliberate tightening the fp4 widening brings, because a
    ``[N, K]`` fp4 tensor (logical shape) and a ``[N, K/2]`` one (storage) are one wrong
    bind apart.  The output gate is the same rule on C: its stores are typed by the plan,
    so a same-size buffer of another dtype would hold wrong bit patterns and a narrower
    one would be overrun -- refused by name before any route, for the forward and both
    backward drivers alike.
    """
    _check_operand(plan, a, "a", plan.dtype)
    _check_operand(plan, w, "w", plan.w_dtype if plan.w_dtype is not None else plan.dtype)
    # The OUTPUT too: the graph carries C's dtype and the kernel's stores use it, so a same-size f16
    # buffer on a bf16 plan would hold bf16 bit patterns (wrong values, no error) and a narrower one
    # (an e4m3 slab) would be overrun by 2-byte stores.  Shared by the forward and both backward
    # drivers; a hand-built plan with no out_dtype (the routing probes) checks nothing, as for A / W.
    if out is not None and plan.out_dtype is not None and out.dtype != plan.out_dtype:
        raise ValueError(f"{plan.label}: out is {out.dtype} but this plan was built for {plan.out_dtype}; refusing to reinterpret the bytes")
    if plan.block_scale:
        sf_a3, sf_w3 = _sf_view(plan, sf_a, "sf_a", plan.m), _sf_view(plan, sf_w, "sf_w", plan.n)
    elif sf_a is not None or sf_w is not None:
        raise ValueError(f"{plan.label}: this plan has no block-scale dequant (built with block_scale=False); refusing to drop the scale factors silently")
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
    # ONE variant pack, keyed by the GRAPH tensors, for both routes.  The JIT resolves keys
    # by tensor identity too (`resolve_variant_pack`), and the graph tensors are the
    # swap-INVARIANT spelling: `jit_from_cudnn_graph` applies `swap_ab_binding` under a
    # `swap_ab` tile config, which trades the binding's a/b (and sfa/sfb) LISTS while
    # keeping the tensor objects -- so keying by `binding.a_operands[0]` would hand A's
    # buffer to B under any `_swapAB` config a caller passes as `tile_config=`.
    vp = {plan.a: _rank3(a, "a"), plan.b: _rank3(w, "w"), plan.c: _rank3(out, "out")}
    if plan.block_scale:
        vp[plan.sfa] = sf_a3
        vp[plan.sfb] = sf_w3
    if plan.has_alpha:
        vp[plan.alpha] = alpha3
    if plan.jit is not None:
        need = int(getattr(plan.jit, "workspace_bytes", 0) or 0)
        if need:
            # Split-K partials: the JIT carves them out of the CALLER's workspace (never allocates).
            # `Workspace` validates presence, contiguity, size and 128-B alignment and raises with
            # the required size -- the same carver the graph engine's execute path uses.  `device=`
            # pins it to the launch device: without it a host (CPU) buffer of the right size and
            # alignment passes every other check and reaches the launch boundary as a bogus pointer.
            from cudnn.frost.workspace import Workspace

            plan.jit(vp, stream=stream, workspace=Workspace(workspace, need, plan.label, device=out.device.index))
        else:
            plan.jit(vp, stream=stream)
        return
    if plan.route == "jit-only":
        raise RuntimeError(f"{plan.label}: the backend declined this graph (route=jit-only) and no JIT artifact was built -- nothing can launch it")
    # The graph route names its stream through the handle: the plan's engine
    # reads `ExecutionContext.stream` off it (`_pygraph.execute` -> `cudnn.get_stream`).
    if handle is None:
        handle = handle_for_stream(out.device, stream)
    plan.graph.execute(vp, workspace, handle)


def _check_operand(plan: ProjGemmPlan, t: Optional[torch.Tensor], name: str, expect: Any) -> None:
    """``t`` carries the dtype the plan was built for, and an e2m1 side the PACKED K extent
    (``shape[-1] * 2 == K``).  A hand-built plan with no dtype (``ProjGemmPlan(graph=None, ...)``
    in the routing tests) checks nothing; so does a missing tensor -- the launch refuses those itself."""
    if t is None or expect is None:
        return
    if t.dtype != expect:
        hint = (
            " -- e2m1 codes travel as torch.float4_e2m1fn_x2: .view(torch.float4_e2m1fn_x2) the uint8 storage"
            if _is_fp4(expect) and t.dtype == torch.uint8
            else ""
        )
        raise ValueError(f"{plan.label}: {name} is {t.dtype} but this plan was built for {expect}; refusing to reinterpret the bytes{hint}")
    if _is_fp4(expect) and int(t.shape[-1]) * 2 != plan.k:
        raise ValueError(
            f"{plan.label}: {name} is fp4 storage {tuple(t.shape)} -- two e2m1 codes per byte along K -- so its last extent must be K/2 = {plan.k // 2}, "
            f"not {int(t.shape[-1])} (a LOGICAL [.., K] fp4 tensor holds twice the data the graph declared)"
        )


def sf_padded_dims(rows: int, k: int, block: int = 32) -> tuple[int, int]:
    """``(rows_pad, sf_k_pad)`` of one F8_128x4 scale-factor tensor over ``rows`` rows and ``k``
    contraction elements at ``block`` (32, or 16 for NVFP4): whole 128-row x 4-block (512 B) atoms, i.e.
    ``(ceil(rows/128)*128, ceil((k/block)/4)*4)``.  ONE source for the graph declaration
    (``build_proj_gemm``), the blob size (``sf_blob_bytes``) and the bound view (``_sf_view``):
    the backend rejects an unpadded F8_128x4 declaration outright (``CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH``),
    and the JIT refuses a smaller blob (``compiler._lower``: ``count >= 512 * k4 * ceil(rows/128)``)."""
    if block not in SF_BLOCK_SIZES:
        raise ValueError(f"block must be one of {SF_BLOCK_SIZES}, got {block!r}")
    if rows <= 0:
        raise ValueError(f"rows must be > 0, got {rows}")
    if k <= 0 or k % block:
        raise ValueError(f"K={k} is not a positive multiple of the {block}-element block")
    return -(-rows // 128) * 128, -(-(k // block) // 4) * 4


def sf_blob_bytes(rows: int, k: int, block: int = 32) -> int:
    """Bytes of one F8_128x4 scale-factor blob = ``rows_pad * sf_k_pad`` of :func:`sf_padded_dims`
    (one E8M0 / E4M3 byte per scale, pad rows / pad blocks ``0x00``)."""
    rows_pad, sf_k_pad = sf_padded_dims(rows, k, block)
    return rows_pad * sf_k_pad


def _sf_view(plan: ProjGemmPlan, sf: Optional[torch.Tensor], name: str, rows: int) -> torch.Tensor:
    """The padded F8_128x4 blob as the rank-3 view (e8m0, or e4m3 for NVFP4) the kernel binds -- checked, never copied."""
    if sf is None:
        raise ValueError(f"{plan.label}: this plan was built with block_scale=True; pass {name}= (the F8_128x4 scale-factor blob). No silent 1.0 (Rule 1).")
    return _sf_view_of(sf, name, rows, plan.k, plan.label, block=plan.block_size, sf_torch_dtype=_sf_torch_dtype(plan.sf_dtype))


def _sf_view_of(sf: torch.Tensor, name: str, rows: int, k: int, label: str, *, block: int = 32, sf_torch_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``_sf_view`` without the plan: the same checks and the same ``(1, rows_pad, 4*k4)`` view, shared by
    the unfused block-scale driver and the fused MXFP8 runner (which keeps the E8M0 / 32 defaults).

    ``sf`` is ``uint8`` or ``sf_torch_dtype`` (``float8_e8m0fnu`` by default; ``float8_e4m3fn`` for the
    NVFP4 pair) and is RE-VIEWED as ``sf_torch_dtype`` -- the JIT binds a pointer and a byte count, so the
    view's dtype only has to be one byte wide, but it is kept honest because the graph tensor it feeds
    was DECLARED in that dtype and that declaration is what picks the MMA scale format."""
    if sf_torch_dtype is None:
        sf_torch_dtype = _E8M0
    sf_word = str(sf_torch_dtype).replace("torch.", "")
    if sf is None:
        raise ValueError(f"{label}: pass {name}= (the F8_128x4 {sf_word} scale-factor blob of the operand). No silent 1.0 (Rule 1).")
    if sf.dtype not in (torch.uint8, sf_torch_dtype):
        raise ValueError(f"{label}: {name} must be uint8 or {sf_word} ({sf_word} bytes), got {sf.dtype}")
    if not sf.is_cuda or not sf.is_contiguous() or sf.data_ptr() % 16:
        raise ValueError(f"{label}: {name} must be a contiguous, 16-B-aligned CUDA tensor (TMA-fed)")
    need = sf_blob_bytes(rows, k, block)
    if sf.numel() != need:
        raise ValueError(
            f"{label}: {name} has {sf.numel()} bytes; the F8_128x4 blob over {rows} rows x K={k} at block {block} is "
            f"{need} = ceil({rows}/128)*128 * ceil({k // block}/4)*4 (whole 512-B atoms, pad rows/blocks 0x00)"
        )
    rows_pad, sf_k_pad = sf_padded_dims(rows, k, block)
    v = sf.view(sf_torch_dtype) if (sf.dtype == torch.uint8 and sf_torch_dtype is not None) else sf  # a re-view, no copy
    return v.view(1, rows_pad, sf_k_pad)  # == the graph's declared SFA / SFB dims


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


# ---------------------------------------------------------------------------
# The BACKWARD's projection GEMMs: transposed operands as VIEWS (api_bwd.py B1 / B2 / B7 / B8)
# ---------------------------------------------------------------------------

_A_MAJORS = ("k", "m")
_B_MAJORS = ("k", "n")


class SplitKPinRefused(RuntimeError):
    """``build_proj_gemm(split_k >= 1)``: the pinned JIT config was refused by the compiler and
    the graph-heuristic fallback -- which would change the tile config, the route and the slice
    count -- is refused too.  The compiler's own decline is the ``__cause__``."""


def _check_mn_major_tma_rule(label: str, operand: str, major: str, extent: int, axis: str, dtype: torch.dtype) -> None:
    """TMA encodes an operand's contiguous extent in 16-byte units (``compiler._tma_alignment_reject``).
    A K-major operand's K is checked where it always was (fp8 ``k % 16``; bf16 by the engine); an
    MN-major operand moves the rule onto M / N, and the driver names it BEFORE the graph exists --
    the engine's own decline would surface only as 'no frost_gemm plan' after the build."""
    if major == "k":
        return
    bpe = torch.empty((), dtype=dtype).element_size()
    elems16 = 16 // bpe
    if extent % elems16:
        raise ValueError(
            f"{label}: {operand} is {major}-major, so TMA reads its {axis} axis contiguously and needs {axis} % {elems16} == 0 "
            f"(the 16-byte contiguous-extent rule at {bpe} B/elem), got {axis}={extent}"
        )


def _declared_layout(plan: ProjGemmPlan, operand: str) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    """``(dims, strides, contiguous axis)`` the plan DECLARED for ``operand`` (``"A"`` | ``"B"``)."""
    m, k, n = plan.m, plan.k, plan.n
    if operand == "A":
        return ((1, m, k), (m * k, k, 1), 2) if plan.a_major == "k" else ((1, m, k), (m * k, 1, m), 1)
    return ((1, k, n), (k * n, 1, k), 1) if plan.b_major == "k" else ((1, k, n), (k * n, n, 1), 2)


def _check_view_against_declaration(plan: ProjGemmPlan, operand: str, what: str, view: torch.Tensor) -> None:
    """The rank-3 VIEW about to be bound as ``operand`` has the declared dims and its stride-1 axis
    IS the declared major -- else a typed ``ValueError`` naming the operand and both strides.

    Why the DRIVER checks, and why the WHOLE stride tuple on BOTH routes: the graph
    fallback binds pointers against the DECLARED strides with no check -- a wrong or padded view
    there is a silent reinterpretation of the bytes.  The JIT route is not uniform either: its
    lowered call (``sm100/compiler.py`` ``_lower``, the ``renorm`` loop) re-labels B from graph
    order ``[1, K, N]`` into kernel order ONLY when the runtime ``(shape, stride)`` EQUALS the
    declared layout (``graph_order`` is None on the ``__call__`` path), so a padded N-major B is
    refused by its generic ``gave_up["input layout"]`` message, while A -- already in kernel
    order -- takes any leading dimension.  One rule the block can rely on, exactly the declared
    layout, and one typed message naming the operand and both strides.  A size-1 axis' stride is
    arbitrary in torch and is never compared (the drivers build the views from rank-2 inputs, so
    the batch stride follows from the checked axes)."""
    dims, declared, contig = _declared_layout(plan, operand)
    shape, stride = tuple(int(x) for x in view.shape), tuple(int(x) for x in view.stride())
    if shape != dims:
        raise ValueError(f"{plan.label}: {operand} ({what}) view has shape {shape}; the plan declared {operand} as dims {dims}")
    if stride[contig] != 1:
        raise ValueError(
            f"{plan.label}: {operand} ({what}) view has strides {stride}, but the plan declared {operand} "
            f"{plan.a_major if operand == 'A' else plan.b_major}-major with strides {declared}: the view's stride-1 axis is not the declared "
            f"contiguous axis {contig}. Build the plan with the matching a_major / b_major, or hand the driver the un-transposed storage"
        )
    if any(s != d for s, d, e in zip(stride, declared, shape, strict=True) if e != 1):  # all rank 3: shape == dims was checked above
        raise ValueError(
            f"{plan.label}: {operand} ({what}) view has strides {stride} but the plan declared {declared}; the drivers bind exactly the declared "
            "layout on every route -- the graph route would read a padded / strided view as the declared layout, and the JIT re-labels B into "
            "kernel order only on an exact match. Hand the driver a contiguous [rows, cols] storage (or its plain transpose), not a slice of a wider slab"
        )


def _check_output_view(plan: ProjGemmPlan, what: str, view: torch.Tensor) -> None:
    dims = (1, plan.m, plan.n)
    shape = tuple(int(x) for x in view.shape)
    if shape != dims:
        raise ValueError(f"{plan.label}: {what} view has shape {shape}; the plan declared C as dims {dims}")
    if not view.is_contiguous():
        raise ValueError(
            f"{plan.label}: {what} must be a contiguous row-major [{plan.m}, {plan.n}] (the graph declared C with strides {(plan.m * plan.n, plan.n, 1)}), got strides {tuple(view.stride())}"
        )


def _rank2(t: torch.Tensor, label: str, name: str) -> torch.Tensor:
    if t.ndim != 2:
        raise ValueError(f"{label}: {name} must be a rank-2 matrix, got shape {tuple(t.shape)}")
    return t


def run_wgrad_gemm(
    plan: ProjGemmPlan,
    dy_like: torch.Tensor,
    x: torch.Tensor,
    dw: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    *,
    stream=None,
) -> None:
    """``dW[rows, cols] = dy_like[T, rows]^T @ x[T, cols]`` -- ``nn.Linear``'s weight gradient
    (B1 ``dW_o = dY^T @ O_gated``, B7 ``dW_qkvg = dQKVG^T @ h``).

    ``plan`` was built with ``a_major="m", b_major="n", m=rows, k=T, n=cols``.  Binds
    ``A = dy_like.unsqueeze(0).transpose(1, 2)`` (== the declared ``[1, rows, T]`` stride
    ``[T*rows, 1, rows]`` VIEW), ``B = x.unsqueeze(0)`` (== ``[1, T, cols]`` stride
    ``[T*cols, cols, 1]``) and ``C = dw.unsqueeze(0)`` -- all views, no copy, no allocation;
    :func:`run_proj_gemm`'s ``_rank3`` passes rank-3 through.  BEFORE binding it asserts
    ``plan.a_major == "m"`` and ``plan.b_major == "n"`` and that every view carries EXACTLY
    the plan's declared strides, size-1 axes aside (a typed ``ValueError`` naming the operand
    and both strides -- the graph fallback would read the declared strides with no check, and
    the JIT re-labels B into kernel order only on an exact match; a column slice of a
    wider slab is refused, not reinterpreted).
    ``stream`` / ``handle`` as :func:`run_proj_gemm` (Rule 5: one launch stream)."""
    if (plan.a_major, plan.b_major) != ("m", "n"):
        raise ValueError(
            f"{plan.label}: run_wgrad_gemm binds A = dy_like^T (M-major) and B = x (N-major), but this plan was built with "
            f"a_major={plan.a_major!r}, b_major={plan.b_major!r}; build it with a_major='m', b_major='n' (m=rows, k=T, n=cols)"
        )
    a3 = _rank2(dy_like, plan.label, "dy_like").unsqueeze(0).transpose(1, 2)  # [1, rows, T], stride [T*rows, 1, rows] for a row-major dy_like
    b3 = _rank2(x, plan.label, "x").unsqueeze(0)  # [1, T, cols]
    c3 = _rank2(dw, plan.label, "dw").unsqueeze(0)  # [1, rows, cols]
    _check_view_against_declaration(plan, "A", "dy_like^T", a3)
    _check_view_against_declaration(plan, "B", "x", b3)
    _check_output_view(plan, "dw", c3)
    run_proj_gemm(plan, a3, b3, c3, workspace, handle, stream=stream)


def run_dgrad_gemm(
    plan: ProjGemmPlan,
    dy_like: torch.Tensor,
    w: torch.Tensor,
    dx: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    *,
    stream=None,
) -> None:
    """``dX[T, N] = dy_like[T, K] @ w[K, N]`` with ``w`` the UN-transposed row-major weight --
    ``nn.Linear``'s input gradient (B2 ``dO_gated = dY @ W_o``, B8 ``dh = dQKVG @ W_qkvg``).

    ``plan`` was built with ``a_major="k", b_major="n", m=T, k=K, n=N``.  Binds
    ``A = dy_like.unsqueeze(0)`` (``[1, T, K]`` row-major), ``B = w.unsqueeze(0)`` (== the
    declared ``[1, K, N]`` stride ``[K*N, N, 1]``: the weight exactly as the checkpoint holds
    it, ``[out_features, in_features]`` for the forward's ``x @ W^T``) and ``C = dx.unsqueeze(0)``
    -- views only, with the same declaration check as :func:`run_wgrad_gemm`."""
    if (plan.a_major, plan.b_major) != ("k", "n"):
        raise ValueError(
            f"{plan.label}: run_dgrad_gemm binds A = dy_like (K-major) and B = w (N-major, the un-transposed row-major weight), but this plan "
            f"was built with a_major={plan.a_major!r}, b_major={plan.b_major!r}; build it with a_major='k', b_major='n' (m=T, k=K, n=N)"
        )
    a3 = _rank2(dy_like, plan.label, "dy_like").unsqueeze(0)  # [1, T, K]
    b3 = _rank2(w, plan.label, "w").unsqueeze(0)  # [1, K, N]
    c3 = _rank2(dx, plan.label, "dx").unsqueeze(0)  # [1, T, N]
    _check_view_against_declaration(plan, "A", "dy_like", a3)
    _check_view_against_declaration(plan, "B", "w", b3)
    _check_output_view(plan, "dx", c3)
    run_proj_gemm(plan, a3, b3, c3, workspace, handle, stream=stream)
