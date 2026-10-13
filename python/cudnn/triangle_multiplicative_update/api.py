# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cudnn.triangle_multiplicative_update`` -- the AlphaFold-class triangle multiplicative update (outgoing / incoming) as three FROST
launches on one stream, behind cuEquivariance's ``triangle_multiplicative_update`` signature (pinned to cuequivariance-ops-torch 0.12.0).

    x [B, N, N, c_z] bf16  --K1-->  a, b  channel-major planes [2, c, B*N, n_pad] bf16   (LayerNorm_in, value || gate projections, sigmoid * mul, mask)
                           --K2-->  y     channel-major        [c, B*N, n_pad] bf16      (the per-channel contraction: ONE batched GEMM, batch = c * B)
                           --K3-->  out   pair-major           [B, N, N, c_z] bf16       (LayerNorm_out(y), LayerNorm_in(x) recomputed, output gate * projection)

The caller adds the residual (as with cuEquivariance).  K1 and K3 are the two arms of the LayerNorm-prologue dual GEMM substrate
(``cudnn.gemm.frost.ln_prologue``), K2 is the FROST GEMM engine's dense template compiled directly for the contraction's tile; the planes
and ``y`` live in ONE caller-provided workspace (:meth:`TriangleMultiplicativeUpdateFwd.get_workspace_size`).  Compulsory traffic:
``18 N^2 C`` bytes per batch entry (K1 reads ``x`` 2 B / pair-channel and writes ``a``, ``b`` 4 B; K2 reads 4 B and writes ``y`` 2 B; K3 reads
``y`` and ``x`` 4 B and writes 2 B) plus the mask (2 B per pair) -- :meth:`TriangleMultiplicativeUpdateFwd.traffic_bytes`.

Two surfaces:

* :class:`TriangleMultiplicativeUpdateFwd` -- the prepared form (``check_support`` / ``get_workspace_size`` / ``compile`` / ``execute``):
  shape, direction and the optional-tensor flags are declared once, the three plans are compiled once, ``execute`` validates and launches
  (no allocation, no conversion, no synchronisation: CUDA-graph capturable).  The weights are prepared ONCE by :func:`prepare_weights`
  into the kernels' layouts (the value / gate rows interleaved per 128-channel block, the biases stacked) -- a plan-time transform.
* :func:`triangle_multiplicative_update` -- the drop-in function with cuEquivariance's parameter names, order and defaults: it caches one
  compiled plan per shape and one prepared weight set per parameter identity, allocates the output and the workspace, and launches.

Served today: Rubin (SM107, compute capability 10.7), bf16, ``c_z == c in {128, 256}``, ``N >= 33``, both directions (the incoming
direction needs ``N % 8 == 0``), the forward only.  ``c_z = 384`` is withdrawn: its sm_107a renderings spill through a 1.7 KB stack per
thread (the LayerNorm row held in registers) and no oracle cell covers them -- it returns with a rendering free of that residency and an
accept test per direction.  Performance knobs (the contraction tile, the projection stage's tile order, the output stage's prologue form,
the scheduler ring depth, the two LayerNorm-prologue stages' row tile ``k_block``, their LayerNorm pass form and epilogue arithmetic form) come from :mod:`.heuristics` -- the measured defaults per geometry -- and may be overridden per plan; every
value of every domain computes the same function.  Everything else is a typed decline (``NotImplementedError`` for a valid-but-unserved
declaration, ``ValueError`` for a malformed one), never a silent slow path -- including autograd: no backward is registered yet, so a call
under grad mode with an input or parameter that ``requires_grad`` raises ``NotImplementedError`` instead of returning a tensor without a
``grad_fn`` (cuEquivariance's function differentiates; a training caller must not lose its gradients silently).

The three launches carry programmatic dependent launch (PDL): every kernel of the chain is rendered with ``USE_PDL = True`` --
``griddepcontrol wait`` before its first global read and ``launch_dependents`` once its mainloop no longer needs the predecessor --
and launched with the CuTe-DSL kernel launch's ``use_pdl`` attribute, so K2's prologue overlaps K1's tail and K3's overlaps K2's (the
K1 / K3 substrate and the FROST GEMM template both ship PDL on).  Measured on the perf-node table in the documentation: PDL on vs the same
three kernels rendered without it is +0.2..+2.5 % on the eager chain; the captured CUDA graph (which keeps the PDL edges) adds another
+0.1..+3.5 % at N <= 512 and nothing above.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Any, Dict, Optional, Tuple

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TensorDesc
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
from cudnn.gemm.frost.ln_prologue import ln_once_tiles_per_item as _ln_once_tiles_per_item

from . import heuristics as th
from .heuristics import TriMulKnobs  # noqa: F401  -- re-exported: the knob record a caller hands to the constructor
from .kernels import contraction as k2
from .kernels import trimul_proj as k1
from .kernels import trimul_tail as k3

#: The cuEquivariance release whose ``triangle_multiplicative_update`` signature (names, order, defaults) this module mirrors.
CUEQ_SIGNATURE_VERSION = "0.12.0"
DIRECTIONS = k2.DIRECTIONS
#: cuEquivariance's admission rule on the pair width (checked first; a malformed declaration).
HIDDEN_DIM_MULTIPLE = 32
#: The pair widths the kernels serve today -- the widths with a measured, tested rendering: the substrate's LayerNorm width is a multiple of
#: 64 streamed in 64-wide stages, its output channel count a multiple of 128, and K3 needs ``c == c_z`` (one K for both GEMMs).  384 fits the
#: substrate's geometry (K3 keeps every K stage of a tile resident, K <= 384) but its renderings spill through a 1.7 KB stack per thread and
#: have no oracle cell, so it is NOT served until a rendering without the register residency lands (then one accept test per direction).
SUPPORTED_CHANNELS = (128, 256)
#: K3 loads the MN-major ``y`` operand in 64-row boxes of one pair row block: ``N >= 33``.
MIN_N_RES = k3.MIN_N_RES_MN_MAJOR
#: K1's tile order: the two projection passes of one row block adjacent in the schedule, so the second pass's ``x`` read is an L2 hit -- the
#: "both passes share one x load" of the byte model (one ``x`` read per call).  With the pass-major order the second pass re-reads ``x`` from
#: HBM once ``x`` outgrows the L2 (measured: K1 moved 1.33x its byte model at N = 1024, 2.0x on the reads).  Bitwise the same planes.  The
#: heuristics module owns the value per geometry; this name is the API's pin of the c = 128 pick below N = 256 (the table picks ``"major"``
#: at c = 256, where the K = 256 rendering is register-bound, and at c = 128 from N >= 256 -- from 512 together with the 1-deep ring, from
#: 256 with the LayerNorm-once work item, which needs the pass inside the row block's item).
K1_PASS_ORDER = th.DEFAULT_K1_PASS_ORDER
_RUBIN = (10, 7)
_WS_ALIGN = 256
_INT32_MAX = 2**31


def _align_up(n: int, a: int = _WS_ALIGN) -> int:
    return -(-int(n) // a) * a


def _shape_dtype_device(t, name: str):
    """``(shape, dtype, device)`` of a torch tensor or a :class:`TensorDesc` declaration."""
    if isinstance(t, TensorDesc):
        return tuple(int(s) for s in t.shape), t.dtype, torch.device(t.device) if not isinstance(t.device, torch.device) else t.device
    if isinstance(t, torch.Tensor):
        return tuple(int(s) for s in t.shape), t.dtype, t.device
    raise TypeError(f"{name} must be a torch.Tensor or a cudnn.api_base.TensorDesc, got {type(t).__name__}")


def _autograd_decline(tensors) -> None:
    """The forward has no registered backward: under grad mode an input / parameter with ``requires_grad`` would be served as a constant
    and the output would carry no ``grad_fn`` -- a silent-wrong-gradient for a caller switching providers from cuEquivariance.  A typed
    decline instead (``torch.no_grad()`` or ``.detach()`` the tensors for inference)."""
    if not torch.is_grad_enabled():
        return
    for name, t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            raise NotImplementedError(
                f"triangle_multiplicative_update has no backward registered yet: {name} requires grad under grad mode and its gradient would be "
                "silently dropped. Run under torch.no_grad() or pass detached tensors (inference); the backward is a later item of this API."
            )


def _stream_handle(stream, device=None) -> int:
    """A raw CUDA stream handle (int) from None / int / ``cuda.CUstream`` / a torch stream.  ``None`` = torch's current stream ON
    ``device`` (the plan's device, not the process's current one: a plan for ``cuda:1`` launched from a ``cuda:0`` context used to pick
    the ``cuda:0`` stream and fail the launch with ``cudaErrorInvalidValue``)."""
    if stream is None:
        return torch.cuda.current_stream(device).cuda_stream
    if isinstance(stream, int):
        return stream
    if isinstance(stream, torch.cuda.Stream):
        return stream.cuda_stream
    try:
        return int(stream)
    except TypeError:
        return int(stream.getPtr())


# ---------------------------------------------------------------------------
# Weights: cuEquivariance's parameter convention -> the kernels' layouts, once
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TriMulWeights:
    """The twelve parameters of the op in the kernels' layouts, prepared ONCE (:func:`prepare_weights`; a plan-time transform).

    ``w_in`` ``[2, 2 c, c_z]`` = cuEquivariance's ``p_in_weight`` / ``g_in_weight`` (rows ``[a; b]``) as the substrate's two passes with the
    value and gate rows interleaved per 128-channel block; ``b_in`` ``[2, 2 c]`` likewise (None when neither input bias was given);
    ``b_out`` ``[2 c_z]`` = ``p_out_bias`` / ``g_out_bias`` interleaved (None when neither was given); the LayerNorm parameters and the two
    output weights contiguous bf16.  A bias given on one side only is paired with zeros for the other -- here, at preparation, never per call.
    """

    c_z: int
    c: int
    w_in: torch.Tensor
    b_in: Optional[torch.Tensor]
    norm_in_weight: torch.Tensor
    norm_in_bias: torch.Tensor
    norm_out_weight: torch.Tensor
    norm_out_bias: torch.Tensor
    p_out_weight: torch.Tensor
    g_out_weight: torch.Tensor
    b_out: Optional[torch.Tensor]

    @property
    def has_in_bias(self) -> bool:
        return self.b_in is not None

    @property
    def has_out_bias(self) -> bool:
        return self.b_out is not None

    @property
    def device(self) -> torch.device:
        return self.w_in.device


def _bf16_contiguous(t: torch.Tensor, name: str, shape: Tuple[int, ...], *, cast: bool) -> torch.Tensor:
    if tuple(t.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(t.shape)}")
    if t.dtype != torch.bfloat16:
        if not cast:
            raise NotImplementedError(f"{name} must be bf16 (got {t.dtype}); pass precision='bfloat16' to cast the parameters like cuEquivariance does")
        t = t.to(torch.bfloat16)
    return t.contiguous()


def prepare_weights(
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    p_in_bias: Optional[torch.Tensor],
    g_in_weight: torch.Tensor,
    g_in_bias: Optional[torch.Tensor],
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    p_out_bias: Optional[torch.Tensor],
    g_out_weight: torch.Tensor,
    g_out_bias: Optional[torch.Tensor],
    *,
    cast_to_bf16: bool = False,
) -> TriMulWeights:
    """cuEquivariance's parameters (``p_in_weight``, ``g_in_weight`` ``[2 c, c_z]`` rows ``[a; b]``; ``p_out_weight`` ``[c_z, c]``;
    ``g_out_weight`` ``[c_z, c_z]``; LayerNorm vectors ``[c_z]`` / ``[c]``; biases optional) -> :class:`TriMulWeights`.  Copies (the
    interleaves), so call it once per parameter set; ``cast_to_bf16`` casts wider parameters (cuEquivariance's ``precision='bfloat16'``)."""
    for name, w in (("norm_in_weight", norm_in_weight), ("norm_in_bias", norm_in_bias), ("p_in_weight", p_in_weight), ("g_in_weight", g_in_weight)):
        if w is None:
            raise ValueError(f"{name} is required (this API never initialises parameters implicitly)")
    for name, w in (("norm_out_weight", norm_out_weight), ("norm_out_bias", norm_out_bias), ("p_out_weight", p_out_weight), ("g_out_weight", g_out_weight)):
        if w is None:
            raise ValueError(f"{name} is required (this API never initialises parameters implicitly)")
    if p_in_weight.ndim != 2 or p_in_weight.shape[0] % 2:
        raise ValueError(f"p_in_weight must be [2 c, c_z] (rows [a; b]), got {tuple(p_in_weight.shape)}")
    c = int(p_in_weight.shape[0]) // 2
    c_z = int(p_in_weight.shape[1])
    bf = lambda t, n, shp: _bf16_contiguous(t, n, shp, cast=cast_to_bf16)  # noqa: E731
    p_in = bf(p_in_weight, "p_in_weight", (2 * c, c_z))
    g_in = bf(g_in_weight, "g_in_weight", (2 * c, c_z))
    ln_in_w = bf(norm_in_weight, "norm_in_weight", (c_z,))
    ln_in_b = bf(norm_in_bias, "norm_in_bias", (c_z,))
    ln_out_w = bf(norm_out_weight, "norm_out_weight", (c,))
    ln_out_b = bf(norm_out_bias, "norm_out_bias", (c,))
    p_out = bf(p_out_weight, "p_out_weight", (c_z, c))
    g_out = bf(g_out_weight, "g_out_weight", (c_z, c_z))
    dev = p_in.device
    if p_in_bias is not None or g_in_bias is not None:
        pb = bf(p_in_bias, "p_in_bias", (2 * c,)) if p_in_bias is not None else torch.zeros(2 * c, dtype=torch.bfloat16, device=dev)
        gb = bf(g_in_bias, "g_in_bias", (2 * c,)) if g_in_bias is not None else torch.zeros(2 * c, dtype=torch.bfloat16, device=dev)
        b_in = k1.stack_projection_biases(pb, gb)
    else:
        b_in = None
    if p_out_bias is not None or g_out_bias is not None:
        pob = bf(p_out_bias, "p_out_bias", (c_z,)) if p_out_bias is not None else torch.zeros(c_z, dtype=torch.bfloat16, device=dev)
        gob = bf(g_out_bias, "g_out_bias", (c_z,)) if g_out_bias is not None else torch.zeros(c_z, dtype=torch.bfloat16, device=dev)
        b_out = k3.stack_tail_biases(pob, gob)
    else:
        b_out = None
    return TriMulWeights(
        c_z=c_z,
        c=c,
        w_in=k1.stack_projection_weights(p_in, g_in),
        b_in=b_in,
        norm_in_weight=ln_in_w,
        norm_in_bias=ln_in_b,
        norm_out_weight=ln_out_w,
        norm_out_bias=ln_out_b,
        p_out_weight=p_out,
        g_out_weight=g_out,
        b_out=b_out,
    )


# ---------------------------------------------------------------------------
# The prepared API
# ---------------------------------------------------------------------------
class TriangleMultiplicativeUpdateFwd(APIBase):
    """Triangle multiplicative update, forward: one declaration, one workspace, three launches on one stream.

    Declaration (the constructor): the pair representation's shape and dtype (``sample_x`` ``[B, N, N, c_z]`` bf16, a tensor or a
    :class:`cudnn.api_base.TensorDesc`), the ``direction``, the hidden width ``c_hidden`` (= ``p_in_weight.shape[0] // 2``) and the
    optional-tensor flags ``has_mask`` / ``has_in_bias`` / ``has_out_bias`` -- compile-time specialisations the execute arguments must match
    exactly (a mask handed to a plan compiled without one raises; so does a missing mask on a plan compiled with one).  ``k2_tile`` names
    a FROST GEMM tile config for the contraction instead of the measured per-geometry pick, ``knobs`` a :class:`heuristics.TriMulKnobs` whose
    SET axes override the measured pick for this geometry (an axis left ``None`` keeps the table's value: ``TriMulKnobs(clc_stages=4)`` changes
    the ring depth only).  Performance only; every member of every domain computes the same function; the vocabulary is validated by
    ``check_support``, the contraction tile's eligibility for this geometry probed there without a compile.

    ``check_support`` raises the typed declines listed in the module docstring, ``compile`` builds the three plans (cached per
    parameter set across instances), ``get_workspace_size`` is the planes + ``y`` carve, ``execute(x, mask, weights, out, workspace)``
    validates (dtype, shape, contiguity, alignment, flag agreement) and launches K1, K2, K3 on ``current_stream`` (None = torch's current
    stream).  No allocation, conversion or synchronisation on the execute path: a captured CUDA graph replays it.
    """

    def __init__(
        self,
        sample_x,
        direction: str = "outgoing",
        *,
        c_hidden: int,
        has_mask: bool = True,
        has_in_bias: bool = False,
        has_out_bias: bool = False,
        eps: float = 1e-5,
        k2_tile: Optional[str] = None,
        knobs: Optional[th.TriMulKnobs] = None,
    ):
        super().__init__()
        self._warn_experimental_api()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        shape, dtype, device = _shape_dtype_device(sample_x, "sample_x")
        if len(shape) == 3:
            shape = (1,) + shape
        if len(shape) != 4 or shape[1] != shape[2]:
            raise ValueError(f"sample_x must be [B, N, N, c_z] (or [N, N, c_z]), got {shape}")
        self.batch, self.n_res, _, self.c_z = shape
        self.c = int(c_hidden)
        self.direction = direction
        self.dtype = dtype
        self.device = torch.device(device)
        self.has_mask = bool(has_mask)
        self.has_in_bias = bool(has_in_bias)
        self.has_out_bias = bool(has_out_bias)
        self.eps = float(eps)
        # the performance knobs: the measured defaults for this geometry (heuristics.recommend: width, N bucket, batch bucket, direction), the
        # caller's knobs (its set axes) / k2_tile over them -- always a complete record; the vocabulary is checked in check_support (ValueError),
        # the contraction tile's eligibility probed there (NotImplementedError)
        self.knobs = th.resolve(self.n_res, self.c, self.batch, direction, k2_tile=k2_tile, knobs=knobs)
        self.k2_tile = self.knobs.k2_tile
        self.n_pad = k2.n_pad(self.n_res)
        self._plan_k1 = None
        self._plan_k3 = None
        self._k2 = None
        self._k2_binding = None

    # -- declaration checks -------------------------------------------------------------------------------------------------------
    def check_support(self) -> bool:
        """Typed declines, malformed declarations first (``ValueError``), then the serving envelope and the arch (``NotImplementedError``)."""
        b, n, c_z, c = self.batch, self.n_res, self.c_z, self.c
        if b < 1 or n < 1:
            raise ValueError(f"x must have B >= 1 and N >= 1, got B={b}, N={n}")
        if c_z % HIDDEN_DIM_MULTIPLE:
            raise ValueError(f"the pair width c_z must be a multiple of {HIDDEN_DIM_MULTIPLE}, got {c_z}")
        if c < 1 or c % HIDDEN_DIM_MULTIPLE:
            raise ValueError(f"the hidden width c (= p_in_weight.shape[0] // 2) must be a positive multiple of {HIDDEN_DIM_MULTIPLE}, got {c}")
        if not (self.eps > 0.0):
            raise ValueError(f"eps must be > 0, got {self.eps}")
        th.validate_knobs(self.knobs)  # an out-of-vocabulary tile-config name / tile order / ring depth is a malformed declaration
        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"triangle_multiplicative_update serves bf16 pair representations only, got {self.dtype}")
        if c_z not in SUPPORTED_CHANNELS:
            raise NotImplementedError(
                f"triangle_multiplicative_update serves c_z in {SUPPORTED_CHANNELS} today (the LayerNorm-prologue GEMM substrate's widths), got c_z={c_z}"
            )
        if c != c_z:
            raise NotImplementedError(
                f"triangle_multiplicative_update serves c_hidden == c_z today (one K for the output gate and projection GEMMs), got c_hidden={c}, c_z={c_z}"
            )
        if n < MIN_N_RES:
            raise NotImplementedError(f"triangle_multiplicative_update needs N >= {MIN_N_RES} (the output stage's 64-row MN-major TMA box), got N={n}")
        if self.direction == "incoming" and not k2.incoming_supported(n):
            raise NotImplementedError(
                f"the incoming direction needs N % {k2.CONTIGUOUS_EXTENT_MULTIPLE} == 0 today (the contraction reads MN-major operands whose contiguous "
                f"extent is N, and a TMA-fed contiguous extent must be a multiple of 8 bf16 elements), got N={n}"
            )
        if b * n * n * c_z >= _INT32_MAX or 2 * c * b * n * self.n_pad >= _INT32_MAX:
            raise NotImplementedError(
                f"triangle_multiplicative_update addresses its tensors with 32-bit element offsets: B*N*N*c_z = {b * n * n * c_z} and the planes' "
                f"2*c*B*N*n_pad = {2 * c * b * n * self.n_pad} must both stay below 2^31 = {_INT32_MAX}"
            )
        if self.device.type != "cuda":
            raise ValueError(f"x must be a CUDA tensor, got device {self.device}")
        index = self.device_index
        cc = tuple(torch.cuda.get_device_capability(index))
        if cc != _RUBIN:
            raise NotImplementedError(f"triangle_multiplicative_update is rendered for compute capability 10.7 (Rubin) only today; found {cc[0]}.{cc[1]}")
        msg = cutedsl_requirement_error("triangle_multiplicative_update")
        if msg:
            raise NotImplementedError(msg)
        msg = cutedsl_arch_requirement_error(cc)
        if msg:
            raise NotImplementedError(msg)
        self._probe_contraction_tile()
        self._is_supported = True
        return True

    def _probe_contraction_tile(self) -> None:
        """The contraction tile's eligibility for THIS geometry on THIS device without a ``cute.compile`` (the FROST GEMM engine's own
        ``check_support`` gates over the direction's operand majors, the folded ``c * B`` batch and the arch): a catalog name the engine
        cannot run here is a typed ``NotImplementedError`` from ``check_support``, never a bare error inside ``compile()``."""
        import cudnn
        import cudnn.gemm.frost  # noqa: F401  -- the pygraph recorder hook the GEMM analyzer reads
        from cudnn.gemm.frost.sm100.compiler import probe_supported
        from cudnn.gemm.frost.tile_config import by_name

        graph = k2.contraction_graph(cudnn, self.c, self.n_res, self.direction, batch=self.batch, padded=True)
        try:
            with torch.cuda.device(self.device_index):
                probe_supported(graph, by_name(self.k2_tile))
        except NotImplementedError:
            raise
        except Exception as error:  # the engine's untyped refusals (a synthesized name's geometry, a template gate) become the typed decline
            raise NotImplementedError(
                f"the contraction tile config {self.k2_tile!r} cannot serve the {self.direction} contraction at N={self.n_res}, c={self.c}, B={self.batch} "
                f"on this device: {type(error).__name__}: {error}"
            ) from error

    @property
    def device_index(self) -> int:
        """The declaration's CUDA device index (``torch.cuda.current_device()`` when the declaration named ``cuda`` without an index)."""
        return self.device.index if self.device.index is not None else torch.cuda.current_device()

    # -- workspace ----------------------------------------------------------------------------------------------------------------
    @property
    def planes_shape(self) -> Tuple[int, int, int, int]:
        return k1.planes_shape(self.c, self.batch, self.n_res)

    @property
    def y_shape(self) -> Tuple[int, int, int]:
        return k3.y_shape(self.c, self.batch, self.n_res)

    def _planes_bytes(self) -> int:
        p = self.planes_shape
        return _align_up(2 * p[0] * p[1] * p[2] * p[3])

    def _y_bytes(self) -> int:
        y = self.y_shape
        return _align_up(2 * y[0] * y[1] * y[2])

    def _slack_bytes(self) -> int:
        """The readable tail after the ``b`` plane the outgoing contraction over-reads at ``N % 8 != 0`` (:func:`k2.k2_slack_bytes`)."""
        return _align_up(k2.k2_slack_bytes(self.n_res)) if self.direction == "outgoing" else 0

    def get_workspace_size(self) -> int:
        """Bytes of the one workspace ``execute`` carves: the two channel-major planes ``a``, ``b`` (plus the contraction's over-read slack
        at ``N % 8 != 0``) and the contraction output ``y`` (bf16, the inner axis padded to a multiple of 8), each 256-byte aligned."""
        return self._planes_bytes() + self._slack_bytes() + self._y_bytes()

    def traffic_bytes(self) -> int:
        """The compulsory HBM traffic model of one call: ``18 N^2 C`` bytes per batch entry plus the mask (2 B per pair) -- read against the
        measured figure in the documentation."""
        return (
            k1.proj_bytes(self.c_z, self.c, self.batch, self.n_res, with_mask=self.has_mask, with_stats=False)
            + k2.contraction_bytes(self.c, self.n_res, batch=self.batch)
            + k3.tail_bytes(self.c_z, self.c, self.batch, self.n_res, with_stats=False)
        )

    def flop(self) -> int:
        return (
            k1.proj_flop(self.c_z, self.c, self.batch, self.n_res)
            + k2.contraction_flop(self.c, self.n_res, batch=self.batch)
            + k3.tail_flop(self.c_z, self.c, self.batch, self.n_res)
        )

    # -- compile ------------------------------------------------------------------------------------------------------------------
    def _k3_ln_pass_form(self) -> str:
        # the re-read LayerNorm forms re-read the MN-major operand in every pass, which the transpose prologue's K-major writes would overwrite
        # (a typed decline of the substrate): under `transpose` the output stage keeps the register form and the knob's value serves the
        # projection stage alone
        return self.knobs.ln_pass_form if self.knobs.k3_prologue == "mn_major" else "registers"

    def _k3_ln_once(self) -> str:
        """The output stage's LayerNorm-once work item: the knob's value where the item spans at least two GEMM tiles there (one pass, so
        ``"block"`` at two N tiles = c = 256), else today's per-tile item -- the substrate declines a one-tile item (K3 at c = 128; ``"passes"``
        under one pass), so the knob's value serves the projection stage alone on those plans.  GPU-free (pinned by the API tests)."""
        form = self.knobs.ln_once or ""
        if not form:
            return ""
        trial = k3.tail_params(self.c_z, self.c, has_bias=self.has_out_bias, want_stats=False, a0_prologue=self.knobs.k3_prologue, ln_once=form)
        return form if _ln_once_tiles_per_item(trial) >= 2 else ""

    def _register_split_hints(self) -> Tuple[int, int]:
        """The `.minnctapersm` launch hint of (K1, K3): the caller's `min_blocks_per_mp` override for both stages, else the per-RENDERING rule with
        each stage's own bias flag, LayerNorm form and (K1) tile order (`heuristics.register_split_hint`: whether a rendering spills at the flat
        allocation is a property of the rendering, not of the geometry); the width is the heuristics' (`c`, = `c_z` on every served plan).  GPU-free
        (pinned by the API tests)."""
        hint = self.knobs.min_blocks_per_mp
        if hint is not None:
            return int(hint), int(hint)
        return (
            th.register_split_hint("k1", self.c, self.has_in_bias, self.knobs.ln_pass_form, self.knobs.epi_math, self.knobs.k1_pass_order),
            th.register_split_hint("k3", self.c, self.has_out_bias, self._k3_ln_pass_form(), self.knobs.epi_math),
        )

    def compile(self) -> None:
        """Build the three plans: K1 (the columns arm of the substrate), K2 (the FROST GEMM engine's dense template for the per-N tile over the
        folded ``c * B`` batch), K3 (the accumulators arm).  Each is cached on its parameter set; a second instance of the same declaration
        reuses them."""
        self._ensure_support_checked()
        import cudnn
        import cudnn.gemm.frost  # noqa: F401  -- installs the pygraph recorder hook the GEMM compiler reads

        # Compile and load on the PLAN's device: the declaration may name a device other than torch's current one (a multi-GPU host with
        # ``x`` on ``cuda:1``).  A compiled launchable is bound to the context it was loaded in, so K1 / K3 and K2 are cached per
        # (parameter set, device) -- :func:`_substrate_plan_on`, :func:`_compile_contraction`.
        k3_ln_pass_form = self._k3_ln_pass_form()
        k1_hint, k3_hint = self._register_split_hints()
        k1_ln_once, k3_ln_once = self.knobs.ln_once or "", self._k3_ln_once()
        with torch.cuda.device(self.device_index):
            self._plan_k1 = _substrate_plan_on(
                k1.build_trimul_proj,
                k1.proj_params(
                    self.c_z,
                    self.c,
                    has_bias=self.has_in_bias,
                    has_mask=self.has_mask,
                    want_stats=False,
                    eps=self.eps,
                    pass_order=self.knobs.k1_pass_order,
                    clc_stages=self.knobs.clc_stages,
                    ln_pass_form=self.knobs.ln_pass_form,
                    epi_math=self.knobs.epi_math,
                    min_blocks_per_mp=k1_hint,
                    ln_once=k1_ln_once,
                ),
                self.device_index,
            )
            self._plan_k3 = _substrate_plan_on(
                k3.build_trimul_tail,
                k3.tail_params(
                    self.c_z,
                    self.c,
                    has_bias=self.has_out_bias,
                    want_stats=False,
                    eps=self.eps,
                    a0_prologue=self.knobs.k3_prologue,
                    clc_stages=self.knobs.clc_stages,
                    ln_pass_form=k3_ln_pass_form,
                    epi_math=self.knobs.epi_math,
                    min_blocks_per_mp=k3_hint,
                    ln_once=k3_ln_once,
                ),
                self.device_index,
            )
            self._k2 = _compile_contraction(cudnn, self.c, self.batch, self.n_res, self.direction, self.k2_tile, self.device_index)
        self._k2_binding = self._k2.binding
        self._compiled_kernel = (self._plan_k1, self._k2, self._plan_k3)

    # -- execute ------------------------------------------------------------------------------------------------------------------
    def _check_execute_args(self, x, mask, weights, out, workspace) -> None:
        b, n, c_z, c = self.batch, self.n_res, self.c_z, self.c
        bf16 = torch.bfloat16
        if isinstance(weights, TriMulWeights):
            _autograd_decline(
                [("x", x), ("mask", mask), ("out", out)]
                + [
                    (f"weights.{f}", getattr(weights, f))
                    for f in ("w_in", "b_in", "norm_in_weight", "norm_in_bias", "norm_out_weight", "norm_out_bias", "p_out_weight", "g_out_weight", "b_out")
                ]
            )
        else:
            _autograd_decline([("x", x), ("mask", mask), ("out", out)])
        # the PLAN's device first: its kernels are loaded on cuda:<device_index> and launched there -- tensors on another device would pass
        # every shape check below and hand foreign pointers to kernels running on the wrong GPU
        if self.device.type == "cuda" and (not isinstance(x, torch.Tensor) or x.device.type != "cuda" or x.device.index != self.device_index):
            raise ValueError(
                f"x lives on {getattr(x, 'device', None)}; this plan was declared on cuda:{self.device_index} and its kernels are loaded there "
                "(declare and compile one plan per device)"
            )
        if not isinstance(x, torch.Tensor) or x.dtype != bf16 or not x.is_contiguous() or tuple(x.shape) != (b, n, n, c_z):
            raise ValueError(
                f"x must be a contiguous bf16 [{b}, {n}, {n}, {c_z}] tensor (the declaration), got {getattr(x, 'shape', None)} {getattr(x, 'dtype', None)}"
            )
        if not isinstance(out, torch.Tensor) or out.dtype != bf16 or not out.is_contiguous() or tuple(out.shape) != (b, n, n, c_z):
            raise ValueError(f"out must be a contiguous bf16 [{b}, {n}, {n}, {c_z}] tensor, got {getattr(out, 'shape', None)} {getattr(out, 'dtype', None)}")
        if self.has_mask:
            if mask is None:
                raise ValueError("this plan was compiled with has_mask=True; a bf16 [B, N, N] mask must be passed (no silent all-valid fallback)")
            if mask.dtype != bf16 or not mask.is_contiguous() or tuple(mask.shape) != (b, n, n):
                raise ValueError(f"mask must be a contiguous bf16 [{b}, {n}, {n}] tensor, got {tuple(mask.shape)} {mask.dtype}")
        elif mask is not None:
            raise ValueError("this plan was compiled with has_mask=False; mask must be None (compile a plan with has_mask=True to apply one)")
        if not isinstance(weights, TriMulWeights):
            raise TypeError("weights must be a TriMulWeights (prepare_weights(...)); the kernels take the interleaved layouts, prepared once")
        if (weights.c_z, weights.c) != (c_z, c):
            raise ValueError(f"weights were prepared for (c_z, c) = {(weights.c_z, weights.c)}, the plan for {(c_z, c)}")
        if weights.has_in_bias != self.has_in_bias:
            raise ValueError(f"this plan was compiled with has_in_bias={self.has_in_bias}; the weights carry an input bias: {weights.has_in_bias}")
        if weights.has_out_bias != self.has_out_bias:
            raise ValueError(f"this plan was compiled with has_out_bias={self.has_out_bias}; the weights carry an output bias: {weights.has_out_bias}")
        if (
            not isinstance(workspace, torch.Tensor)
            or workspace.dtype != torch.uint8
            or workspace.numel() < self.get_workspace_size()
            or not workspace.is_contiguous()
        ):
            raise ValueError(f"workspace must be a contiguous uint8 tensor of at least get_workspace_size() = {self.get_workspace_size()} bytes")
        for name, t in (("x", x), ("out", out), ("workspace", workspace), ("mask", mask)):
            if t is not None and t.device != x.device:
                raise ValueError(f"{name} must live on x's device {x.device}, got {t.device}")
        if weights.device != x.device:
            raise ValueError(f"weights must live on x's device {x.device}, got {weights.device}")
        if workspace.data_ptr() % _WS_ALIGN:
            raise ValueError(f"workspace must be {_WS_ALIGN}-byte aligned")
        if out.data_ptr() == x.data_ptr():
            raise ValueError("out must not alias x (the output stage reads x while it writes out)")

    def _carve(self, workspace: torch.Tensor):
        """``(plane_a, plane_b, y)`` views over the workspace (no allocation)."""
        p = self.planes_shape
        planes = workspace[: 2 * p[0] * p[1] * p[2] * p[3]].view(torch.bfloat16).view(*p)
        yb = self._y_bytes()
        y_off = self._planes_bytes() + self._slack_bytes()
        y = workspace[y_off : y_off + yb][: 2 * self.y_shape[0] * self.y_shape[1] * self.y_shape[2]].view(torch.bfloat16).view(*self.y_shape)
        return planes, planes[0], planes[1], y

    def execute(self, x, mask, weights, out, workspace, current_stream: Optional[cuda.CUstream] = None) -> None:
        """Launch K1, K2, K3 on ``current_stream`` (None = torch's current stream on the plan's device).  Validates, never converts or
        allocates.  The launches run with the plan's device current (``torch.cuda.device`` -- a context switch, no synchronisation), so a
        plan declared on ``cuda:1`` is served from any current device."""
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        self._check_execute_args(x, mask, weights, out, workspace)
        stream = _stream_handle(current_stream, self.device)
        with torch.cuda.device(self.device_index):
            self._launch(x, mask, weights, out, workspace, stream)

    def _launch(self, x, mask, weights, out, workspace, stream: int) -> None:
        planes, plane_a, plane_b, y = self._carve(workspace)
        b, n, c_z = self.batch, self.n_res, self.c_z
        k1.run_trimul_proj(
            self._plan_k1,
            x,
            weights.w_in,
            planes,
            weights.norm_in_weight,
            weights.norm_in_bias,
            bias_stacked=weights.b_in,
            mask=mask,
            stats=None,
            stream=stream,
            k_block=self.knobs.k_block,
        )
        a_v, b_v, y_v = k2.k2_operand_views(plane_a, plane_b, y, channels=self.c, batch=b, n_res=n, direction=self.direction)
        bd = self._k2_binding
        self._k2({bd.a_operands[0]: a_v, bd.b_operands[0]: b_v, bd.outputs[0]: y_v}, stream=stream)
        k3.run_trimul_tail(
            self._plan_k3,
            y,
            x,
            weights.p_out_weight,
            weights.g_out_weight,
            out,
            weights.norm_out_weight,
            weights.norm_out_bias,
            weights.norm_in_weight,
            weights.norm_in_bias,
            biases=weights.b_out,
            stats=None,
            residual=None,
            stream=stream,
            k_block=self.knobs.k_block,
        )

    def make_cuda_graph(self, x, mask, weights, out, workspace, *, warmup: int = 2) -> "torch.cuda.CUDAGraph":
        """Capture the three launches for THESE buffers into one ``torch.cuda.CUDAGraph`` (replay = one graph launch per call; the
        buffers' contents may change between replays, their addresses may not).  ``warmup`` eager calls run first on a side stream."""
        side = torch.cuda.Stream(device=x.device)
        side.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(side):
            for _ in range(int(warmup)):
                self.execute(x, mask, weights, out, workspace, current_stream=cuda.CUstream(side.cuda_stream))
        torch.cuda.current_stream(x.device).wait_stream(side)
        torch.cuda.synchronize(x.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            self.execute(x, mask, weights, out, workspace, current_stream=cuda.CUstream(side.cuda_stream))
        return graph


_K2_LOCK = threading.Lock()
_K2_CACHE: Dict[Tuple[Any, ...], Any] = {}
_SUBSTRATE_LOCK = threading.Lock()
_SUBSTRATE_PLANS: Dict[Tuple[Any, int], Any] = {}


def _substrate_plan_on(build, params, device_index: int):
    """One cute-compiled launchable of the K1 / K3 substrate per (parameter set, device).  A CuTe-DSL launchable is bound to the CUDA
    context it was first loaded in -- a K1 launchable built on ``cuda:0`` fails on ``cuda:1`` with ``cudaErrorInvalidValue`` at launch, and
    vice versa (measured on a two-GPU host) -- while the substrate's ``build_*`` cache is per parameter set only.  Under the plan's device:
    when the substrate's cache fills NOW its launchable is bound here and is reused (the single-device case pays nothing extra); when it was
    filled earlier -- by a plan on another device, or by a caller outside this API -- the same template module gets a second ``cute.compile``
    for this device, cached per (params, device)."""
    from cudnn.gemm.frost import ln_prologue

    key = (params, int(device_index))
    with _SUBSTRATE_LOCK:
        hit = _SUBSTRATE_PLANS.get(key)
    if hit is not None:
        return hit
    with torch.cuda.device(device_index), _SUBSTRATE_LOCK:
        before = ln_prologue.build_ln_dual_gemm.cache_info().currsize
        base = build(params)
        built_here = ln_prologue.build_ln_dual_gemm.cache_info().currsize > before
        plan = base if built_here else ln_prologue.LnDualGemmPlan(params=base.params, module=base.module, launch=base.module.compile.__wrapped__())
        return _SUBSTRATE_PLANS.setdefault(key, plan)


def _compile_contraction(cudnn_module, channels: int, batch: int, n_res: int, direction: str, tile: str, device_index: int):
    """The compiled FROST GEMM instance of the contraction for ``(c * B, direction, tile, dtype, device)`` -- M / N / K are symbolic in the
    kernel, the batch count is baked, so the cache key is the folded batch, not N.  Keyed on the PLAN's device (not the current one): the
    caller enters ``torch.cuda.device(device_index)`` around the compile, so a plan for ``cuda:1`` never binds a module loaded on ``cuda:0``."""
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    index = int(device_index)
    key = (channels * batch, direction, tile, "bf16", index)
    with _K2_LOCK:
        hit = _K2_CACHE.get(key)
        if hit is not None:
            return hit
    graph = k2.contraction_graph(cudnn_module, channels, n_res, direction, batch=batch, padded=True)
    compiled = jit_from_cudnn_graph(graph, config=by_name(tile))
    with _K2_LOCK:
        return _K2_CACHE.setdefault(key, compiled)


# ---------------------------------------------------------------------------
# The drop-in function (cuEquivariance's signature)
# ---------------------------------------------------------------------------
_PLAN_LOCK = threading.Lock()
_PLANS: Dict[Tuple[Any, ...], TriangleMultiplicativeUpdateFwd] = {}
_WEIGHTS: "OrderedDict[Tuple[Any, ...], Tuple[TriMulWeights, tuple]]" = OrderedDict()
_WEIGHTS_MAX = 8

_PRECISION_DEFAULT = ("NONE", "DEFAULT")
_PRECISION_BF16 = ("BF16", "BFLOAT16")


def _parse_precision(precision) -> str:
    """cuEquivariance's ``precision`` spellings -> ``"default"`` (bf16 inputs required) or ``"bfloat16"`` (cast wider inputs); the
    rest (``"mxfp8"``, ``"ieee"``, ``"tf32"``, ...) are typed declines."""
    if precision is None:
        return "default"
    name = None
    if isinstance(precision, str):
        name = precision.upper()
    elif isinstance(precision, int) and not isinstance(precision, bool):
        name = {-1: "NONE", 0: "DEFAULT", 5: "BFLOAT16", 4: "MXFP8", 3: "IEEE", 1: "TF32", 2: "TF32X3"}.get(precision, str(precision))
    elif hasattr(precision, "name"):
        name = str(precision.name).upper()
    else:
        raise ValueError(f"precision must be None, a string or a cuEquivariance Precision, got {type(precision).__name__}")
    if name in _PRECISION_DEFAULT:
        return "default"
    if name in _PRECISION_BF16:
        return "bfloat16"
    if name == "MXFP8":
        raise NotImplementedError("precision='mxfp8' (block-scaled a / b operands) is not served by triangle_multiplicative_update yet")
    raise NotImplementedError(f"precision={precision!r} is not served by triangle_multiplicative_update (bf16 arithmetic only: None / 'default' / 'bfloat16')")


def _tensor_identity(t: Optional[torch.Tensor]):
    if t is None:
        return None
    return (t.data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype, str(t.device), t._version)


def _cached_weights(params: tuple, cast: bool) -> TriMulWeights:
    """One prepared weight set per PARAMETER IDENTITY (address, shape, strides, dtype, device and torch's in-place version counter
    of each of the twelve tensors); the entry keeps a reference to the source tensors so an address cannot be recycled under it.
    A bounded LRU: a model switches parameter sets rarely, a training loop bumps the version counter and misses (one re-interleave)."""
    key = (cast,) + tuple(_tensor_identity(t) for t in params)
    with _PLAN_LOCK:
        hit = _WEIGHTS.get(key)
        if hit is not None:
            _WEIGHTS.move_to_end(key)
            return hit[0]
    prepared = prepare_weights(*params, cast_to_bf16=cast)
    with _PLAN_LOCK:
        _WEIGHTS[key] = (prepared, tuple(t for t in params if t is not None))
        _WEIGHTS.move_to_end(key)
        while len(_WEIGHTS) > _WEIGHTS_MAX:
            _WEIGHTS.popitem(last=False)
    return prepared


def _cached_plan(x: torch.Tensor, direction: str, c: int, has_mask: bool, has_in_bias: bool, has_out_bias: bool, eps: float) -> TriangleMultiplicativeUpdateFwd:
    key = (tuple(x.shape), x.dtype, str(x.device), direction, int(c), bool(has_mask), bool(has_in_bias), bool(has_out_bias), float(eps))
    with _PLAN_LOCK:
        plan = _PLANS.get(key)
    if plan is not None:
        return plan
    plan = TriangleMultiplicativeUpdateFwd(x, direction, c_hidden=c, has_mask=has_mask, has_in_bias=has_in_bias, has_out_bias=has_out_bias, eps=eps)
    plan.check_support()
    plan.compile()
    with _PLAN_LOCK:
        return _PLANS.setdefault(key, plan)


def triangle_multiplicative_update(
    x: torch.Tensor,
    direction: str = "outgoing",
    mask: Optional[torch.Tensor] = None,
    norm_in_weight: Optional[torch.Tensor] = None,
    norm_in_bias: Optional[torch.Tensor] = None,
    p_in_weight: Optional[torch.Tensor] = None,
    p_in_bias: Optional[torch.Tensor] = None,
    g_in_weight: Optional[torch.Tensor] = None,
    g_in_bias: Optional[torch.Tensor] = None,
    norm_out_weight: Optional[torch.Tensor] = None,
    norm_out_bias: Optional[torch.Tensor] = None,
    p_out_weight: Optional[torch.Tensor] = None,
    p_out_bias: Optional[torch.Tensor] = None,
    g_out_weight: Optional[torch.Tensor] = None,
    g_out_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    precision=None,
    *,
    out: Optional[torch.Tensor] = None,
    stream=None,
) -> torch.Tensor:
    """The triangle multiplicative update of ``x`` (``[B, N, N, c_z]`` or ``[N, N, c_z]`` bf16) -- cuEquivariance's signature
    (``cuequivariance_ops_torch.triangle_multiplicative_update``, 0.12.0) with two keyword-only additions: ``out`` (write into this
    bf16 tensor instead of allocating) and ``stream`` (a raw CUDA stream handle the allocations and launches are ordered on; None = torch's
    current stream).  Returns the update (the caller adds the residual), shaped like ``x``.

    Differences from cuEquivariance, all typed: the eight weights are REQUIRED (no implicit random initialisation); ``precision`` is
    None / ``'default'`` (bf16 inputs required) or ``'bfloat16'`` (wider inputs cast once per call, as cuEquivariance does), ``'mxfp8'`` and
    the IEEE / TF32 modes decline; a non-contiguous ``x`` declines; autograd is not registered yet, so under grad mode an input or
    parameter with ``requires_grad`` is a typed decline (``NotImplementedError``), never a tensor without a ``grad_fn`` -- call under
    ``torch.no_grad()`` or detach for inference.  A mask of any real or bool dtype is cast once per call to the kernels' bf16 ``[B, N, N]`` (no copy when
    it already is).  One compiled plan per (shape, direction, flags) and one prepared weight set per parameter identity are cached in
    the process; warm a shape before capturing a CUDA graph around this call.
    """
    from cudnn._torch_stream import stream_context

    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    mode = _parse_precision(precision)
    cast = mode == "bfloat16"
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    squeeze = x.ndim == 3
    x4 = x.unsqueeze(0) if squeeze else x
    if x4.ndim != 4 or x4.shape[1] != x4.shape[2]:
        raise ValueError(f"x must be [B, N, N, c_z] or [N, N, c_z], got {tuple(x.shape)}")
    if p_in_weight is None or p_in_weight.ndim != 2 or p_in_weight.shape[0] % 2:
        raise ValueError("p_in_weight is required and must be [2 c, c_z] (rows [a; b])")
    for name, w in (
        ("norm_in_weight", norm_in_weight),
        ("norm_in_bias", norm_in_bias),
        ("g_in_weight", g_in_weight),
        ("norm_out_weight", norm_out_weight),
        ("norm_out_bias", norm_out_bias),
        ("p_out_weight", p_out_weight),
        ("g_out_weight", g_out_weight),
    ):
        if w is None:
            raise ValueError(f"{name} is required (this API never initialises parameters implicitly)")
    c = int(p_in_weight.shape[0]) // 2
    _autograd_decline(
        [
            ("x", x),
            ("mask", mask),
            ("norm_in_weight", norm_in_weight),
            ("norm_in_bias", norm_in_bias),
            ("p_in_weight", p_in_weight),
            ("p_in_bias", p_in_bias),
            ("g_in_weight", g_in_weight),
            ("g_in_bias", g_in_bias),
            ("norm_out_weight", norm_out_weight),
            ("norm_out_bias", norm_out_bias),
            ("p_out_weight", p_out_weight),
            ("p_out_bias", p_out_bias),
            ("g_out_weight", g_out_weight),
            ("g_out_bias", g_out_bias),
            ("out", out),
        ]
    )
    device = x4.device
    with stream_context(stream, device):
        if x4.dtype != torch.bfloat16:
            if not cast:
                raise NotImplementedError(
                    f"triangle_multiplicative_update serves bf16 pair representations only (got {x4.dtype}); pass precision='bfloat16' to cast like cuEquivariance"
                )
            x4 = x4.to(torch.bfloat16)
        if not x4.is_contiguous():
            raise ValueError("x must be contiguous ([B, N, N, c_z] row-major); the kernels read it natively and never repack")
        batch, n_res = int(x4.shape[0]), int(x4.shape[1])
        m = None
        if mask is not None:
            m = mask.unsqueeze(0) if mask.ndim == 2 else mask
            if m.ndim != 3 or tuple(m.shape) != (batch, n_res, n_res):
                raise ValueError(f"mask must be [B, N, N] = {(batch, n_res, n_res)} (or [N, N] for B = 1), got {tuple(mask.shape)}")
            if m.dtype != torch.bfloat16:
                m = m.to(torch.bfloat16)  # bool / fp32 masks: cast once per call (cuEquivariance's path too)
            if not m.is_contiguous():
                m = m.contiguous()
        has_in_bias = p_in_bias is not None or g_in_bias is not None
        has_out_bias = p_out_bias is not None or g_out_bias is not None
        # the plan first: its typed declines (width, N, direction, arch) must fire before any parameter is touched
        plan = _cached_plan(x4, direction, c, m is not None, has_in_bias, has_out_bias, float(eps))
        weights = _cached_weights(
            (
                norm_in_weight,
                norm_in_bias,
                p_in_weight,
                p_in_bias,
                g_in_weight,
                g_in_bias,
                norm_out_weight,
                norm_out_bias,
                p_out_weight,
                p_out_bias,
                g_out_weight,
                g_out_bias,
            ),
            cast,
        )
        if out is None:
            out4 = torch.empty_like(x4)
        else:
            out4 = out.unsqueeze(0) if out.ndim == 3 else out
        workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=device)
        plan.execute(x4, m, weights, out4, workspace, current_stream=cuda.CUstream(_stream_handle(stream, device)))
    return out4.squeeze(0) if squeeze else out4


__all__ = [
    "CUEQ_SIGNATURE_VERSION",
    "DIRECTIONS",
    "HIDDEN_DIM_MULTIPLE",
    "MIN_N_RES",
    "SUPPORTED_CHANNELS",
    "TriMulKnobs",
    "TriMulWeights",
    "TriangleMultiplicativeUpdateFwd",
    "prepare_weights",
    "triangle_multiplicative_update",
]
