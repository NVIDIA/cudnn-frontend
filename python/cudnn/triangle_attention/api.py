# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cudnn.triangle_attention`` -- AlphaFold-class triangle attention (a pair bias and a key mask on every row) on the FROST
SDPA engine, behind cuEquivariance's ``triangle_attention`` signature (pinned to cuequivariance-ops-torch 0.12.0).

    q, k, v  (B, N, H, S, D)        o = softmax_k( scale * q_i . k_j + bias[b, h, i, j] )  v_j      over the keys the mask keeps
    bias     (B, 1, H, S_q, S_kv)   the pair bias, one per outer batch entry, SHARED by the N rows of that entry
    mask     (B, N, 1, 1, S_kv)     a dense boolean key mask (False = the key is masked for every query of that row)
    kv_lengths (B, N, 1, 1, 1)      the right-padded alternative: keys ``j < kv_lengths[b, n]`` are live
    mask + mask_is_prefix=True      the SAME boolean mask ASSERTED prefix-shaped (every row True on ``[0, len)``, False after): its row
                                    lengths are derived on the DEVICE (ONE small kernel -- a row count into the caller's workspace, no
                                    cast pass, no host synchronisation, captured by a CUDA graph) and the call rides the ``kv_lengths`` route
    -> o (B, N, H, S_q, D), and with ``return_aux`` the fp32 row statistics (lse, max) (B, N, H, S_q)

This module writes no attention kernel (its one kernel is the prefix-length derivation, ``kernels/prefix_lengths.py``).  The attention
core is the shipped ``sdpa`` graph node served by the FROST row ``sdpa_fwd_prefill_sm107``
on its 32- / 64-wide head-dim tiles -- the additive bias held as a resident SMEM row-block or streamed per kv tile, the boolean
key mask folded into the kernel's bit-word mask, the right-padded lengths through the graph's padding path (``seq_len_kv`` = the
caller's lengths; the ``seq_len_q`` the graph contract requires beside it is the constant ``S_q`` -- every Q row live -- written into the
caller's workspace on the launch stream; zero kernel cost).  The API layer flattens ``(B, N)`` onto the graph's batch axis (``N`` rows
per outer entry, the bias broadcast over them -- the graph's existing ``[1, H, S_q, S_kv]`` bias contract, one launch per outer batch
entry), caches ONE compiled plan per declaration (shape, dtype, layout, mask form, outputs, scale) and launches it on the caller's stream.

Two surfaces:

* :class:`TriangleAttentionFwd` -- the prepared form (``check_support`` / ``get_workspace_size`` / ``compile`` / ``execute``): the
  five tensors' shapes, dtype AND strides are declared once (the kernel's TMA descriptors are built from them), the mask form and the
  outputs are compile-time choices, ``execute`` validates and launches (no allocation, no conversion, no synchronisation: a captured
  CUDA graph replays it).  Every tensor handed to ``execute`` must match the declaration -- a layout the kernel does not bind without
  a copy is a typed decline here, never a hidden repack.
* :func:`triangle_attention` -- the drop-in function with cuEquivariance's parameter names, order and defaults.  It caches one plan per
  declaration, allocates the output and -- where the caller's layout is not the kernel's -- repacks into it with a COUNTED copy
  (:func:`copy_counters`, one warning per layout naming the permutation), so a model's transposed tensors neither silently copy nor
  fail; the zero-copy layout is documented (the "layout_left" form ``(B, N, S, H, D)`` in memory, the bias head-major with a
  16-byte row stride).  Under grad mode with an input that requires grad the call runs through a ``torch.autograd.Function`` that
  saves ``(q, k, v, bias, o, lse)`` for the backward; the backward itself is a later item of this API and raises a typed
  ``NotImplementedError`` today.

Served today: compute capability 10.7 (Rubin), bf16 / fp16 (the bias in the same dtype; an fp32 bias is cast once per call by the
function, counted), ``D % 8 == 0`` with ``D <= 64`` (the row's small tiles; ``D < 32`` / ``32 < D < 64`` by TMA zero fill), any
``S_q`` / ``S_kv`` the bias form fits (resident up to the row's budget, streamed above it), ``B >= 1`` (one launch per outer batch
entry until the engine carries a bias batch group), the four key-mask forms (none / lengths / dense boolean / a prefix-asserted boolean
mask served as lengths), the forward only.
Everything else is a typed decline quoting the ENGINE's own reason (``NotImplementedError`` for a valid-but-unserved declaration,
``ValueError`` for a malformed one) -- including ``return_aux=True``: the row max is a side output the small-tile bias arm does not
store yet, so the request is declined rather than served by a second pass.

Masked-row conventions (documented, not chosen in code -- see ``docs/operations/attention/triangle_attention.md``): a row whose every key is
masked reads ``o = 0``, ``lse = max = -inf`` here (the FROST SDPA invariant for an empty key range, on BOTH mask forms);
cuEquivariance's ``kv_lengths`` route zeroes such a row (``o = lse = max = 0``) and its dense-mask route folds a finite ``-1e9``
(``o`` = the uniform average of ``v``, ``lse = max = -1e9``).  :func:`mask_to_kv_lengths` therefore changes what a fully masked row
reads only on cuEquivariance; here both forms agree -- and so does the ``mask_is_prefix`` form, which IS the lengths route.

The prefix-mask contract (``mask_is_prefix=True`` on both surfaces, :func:`prefix_mask_to_kv_lengths` for a caller with a plan of its own):
the CALLER asserts the mask is prefix-shaped; the API computes each row's True count on the device -- one launch of the row-popcount
kernel ``kernels/prefix_lengths.py`` (a warp per row, byte loads, ``popc(ballot)``, one int32 store into the caller's memory) -- and
binds it as that row's length, reading the mask on the host NEVER.  The count of a non-prefix row is still a length: such a row is served as the prefix of its True
count -- the WRONG keys, no error.  :data:`CHECK_PREFIX_MASK_ENV` ``= 1`` (or ``check_prefix_mask=True`` per call) turns on the debug
check, the device-wide prefix test with its ONE host synchronisation, which raises a typed ``ValueError`` naming the rows that violate
the contract; it is off by default and never runs inside a CUDA-graph capture.
"""

from __future__ import annotations

import math
import os
import threading
import warnings
from typing import Any, Dict, Optional, Tuple

import torch

from cudnn.api_base import APIBase, TensorDesc, check_buffers_disjoint
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver, ws_align

#: The cuEquivariance release whose ``triangle_attention`` signature (names, order, defaults) this module mirrors.
CUEQ_SIGNATURE_VERSION = "0.12.0"
#: The FROST engine row that serves the graph; its ``Capabilities`` row and ``mismatch()`` are the source of every typed decline.
ENGINE_ROW = "sdpa_fwd_prefill_sm107"
#: The key-mask forms a plan is declared with (``prefix_mask`` = ``mask=True, mask_is_prefix=True``: the boolean mask asserted prefix-shaped,
#: its lengths derived on the device and served through the ``kv_lengths`` graph form).
MASK_FORMS = ("none", "kv_lengths", "mask", "prefix_mask")
#: ``=1`` turns on the prefix-mask DEBUG check on every call that does not pass ``check_prefix_mask`` explicitly: the device-wide prefix test
#: (one host synchronisation per call) that raises a typed ``ValueError`` on a mask the caller asserted prefix-shaped but is not.
CHECK_PREFIX_MASK_ENV = "CUDNN_FRONTEND_TRIANGLE_ATTENTION_CHECK_PREFIX_MASK"
_RUBIN = (10, 7)
_ELEM_BYTES = 2  # bf16 / fp16
_TMA_ALIGN = 16  # the TMA-read operands (q, k, v, bias, o): the binder's `_ALIGN_TMA`
_ALIGN_WORD = 4  # the int32 lengths and the fp32 lse: the binder's `_ALIGN_F32` (`prepared.lens`, `_dense_lse`)
_ALIGN_BYTE = 1  # the boolean key mask: the kernel loads ONE BYTE per lane (`tile_dsl.mask.kv_mask_keep_words`) -- no alignment rule
_FROST_ENV = "CUDNN_FRONTEND_ENABLE_FROST_ENGINES"
_DTYPES = (torch.bfloat16, torch.float16)
_dense_bind_strides = None  # the engine's layout predicate, imported on first use


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------
def _declaration(t, name: str) -> Tuple[Tuple[int, ...], Tuple[int, ...], torch.dtype, torch.device]:
    """``(shape, stride, dtype, device)`` of a torch tensor or a :class:`TensorDesc`."""
    if isinstance(t, TensorDesc):
        dev = t.device if isinstance(t.device, torch.device) else torch.device(t.device)
        return tuple(int(s) for s in t.shape), tuple(int(s) for s in t.stride), t.dtype, dev
    if isinstance(t, torch.Tensor):
        return tuple(int(s) for s in t.shape), tuple(int(s) for s in t.stride()), t.dtype, t.device
    raise TypeError(f"{name} must be a torch.Tensor or a cudnn.api_base.TensorDesc, got {type(t).__name__}")


def _row_strides(shape5: Tuple[int, ...], stride5: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """The per-outer-batch-entry 4-D ``(N, H, S, D)`` geometry of a 5-D ``(B, N, H, S, D)`` declaration (what one launch binds)."""
    return tuple(shape5[1:]), tuple(stride5[1:])


def activation_binds_zero_copy(shape4: Tuple[int, ...], stride4: Tuple[int, ...]) -> bool:
    """Whether a logical ``(N, H, S, D)`` activation binds to the kernel's TMA descriptors without a copy -- the engine row's own
    predicate (``config_sm100.dense_bind_strides``): the head dim contiguous, the token / head / batch strides 16-byte multiples,
    the nest token-major (``stride(S) >= H * stride(H)``) -- the "layout_left" ``(B, N, S, H, D)`` memory form, compact or padded.
    A torch-contiguous ``(B, N, H, S, D)`` tensor is head-major and returns False."""
    global _dense_bind_strides
    if _dense_bind_strides is None:
        from cudnn.sdpa.fwd.config_sm100 import dense_bind_strides

        _dense_bind_strides = dense_bind_strides
    return _dense_bind_strides(tuple(shape4), tuple(stride4), _ELEM_BYTES) is not None


def bias_binds_zero_copy(shape4: Tuple[int, ...], stride4: Tuple[int, ...]) -> bool:
    """Whether a logical ``(1|B, 1|H, S_q, S_kv)`` bias binds without a copy: the key dim contiguous and every stride of an axis
    with extent > 1 a 16-byte multiple (the TMA global-stride rule: ``S_kv % 8 == 0`` for a compact bias, else a padded row
    stride) -- the engine row's own rule, spelled once more here so the function can decide a repack BEFORE the plan exists."""
    if len(shape4) != 4 or stride4[3] != 1:
        return False
    return all(not (n > 1 and (s * _ELEM_BYTES) % _TMA_ALIGN) for n, s in zip(shape4[:3], stride4[:3]))


def _stream_int(stream, device) -> int:
    """A raw CUDA stream handle (int) from None / int / a torch stream / a driver ``CUstream``.  None = torch's current stream on
    ``device`` (the plan's device, not the process's current one)."""
    if stream is None:
        from cudnn._torch_stream import _raw_current_stream

        raw = _raw_current_stream(torch, device)
        return int(raw) if raw is not None else torch.cuda.current_stream(device).cuda_stream
    if isinstance(stream, int):
        return stream
    if isinstance(stream, torch.cuda.Stream):
        return stream.cuda_stream
    try:
        return int(stream)
    except TypeError:
        return int(stream.getPtr())


# ---------------------------------------------------------------------------
# The caller's workspace -- the SDPA adapters' recipe (`ws_align` chunks dealt in order by `WorkspaceCarver`), never a local carve
# ---------------------------------------------------------------------------
def workspace_bytes(plan_bytes: int, n_q_lens: int, n_kv_lens: int = 0, n_mask_stage: int = 0) -> int:
    """Bytes ``execute`` needs: the plan's own ``plan_bytes`` and, on the ``kv_lengths`` form (``n_q_lens > 0``), ``n_q_lens`` int32
    words for the Q-side lengths -- two chunks of the SDPA adapters' shared carve (``ws_align``: 128-byte chunks dealt in order), so the
    lengths start at ``ws_align(plan_bytes)``.  The ``prefix_mask`` form adds one more chunk: ``n_kv_lens`` int32 words (the KV lengths
    the device derives, ``B * N``).  ``n_mask_stage`` (appended) sizes a fourth chunk after it -- the int32 mask staging of the two-kernel
    derivation this form shipped with first; the fused kernel stages nothing, the plan requests 0 words, and the parameter stays so a
    caller sizing by the four-chunk arithmetic still carves.  The other forms need exactly the plan's own bytes."""
    if n_q_lens <= 0 and n_kv_lens <= 0 and n_mask_stage <= 0:
        return int(plan_bytes)
    total = ws_align(plan_bytes)
    for n in (n_q_lens, n_kv_lens, n_mask_stage):
        if n > 0:
            total += ws_align(_ALIGN_WORD * int(n))
    return total


def carve_workspace_chunks(
    workspace, plan_bytes: int, n_q_lens: int, n_kv_lens: int, n_mask_stage: int, owner: str
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(the plan's own chunk, the Q-side lengths, the derived KV lengths, the int32 mask staging)`` -- each an int32 / uint8 view of the
    caller's ``workspace`` or None when its count is 0 -- carved in :func:`workspace_bytes`'s order by the shared ``WorkspaceCarver``: a
    missing, undersized, non-tensor or misaligned buffer raises ITS error (the SDPA adapters' text; its base rule is 16-byte alignment,
    which a torch allocation satisfies, while the lengths' reader needs 4)."""
    carver = WorkspaceCarver(workspace, workspace_bytes(plan_bytes, n_q_lens, n_kv_lens, n_mask_stage), owner)
    plan_ws = carver.take(int(plan_bytes), torch.uint8) if plan_bytes else None
    q_lens = carver.take(int(n_q_lens), torch.int32) if n_q_lens > 0 else None
    kv_lens = carver.take(int(n_kv_lens), torch.int32) if n_kv_lens > 0 else None
    stage = carver.take(int(n_mask_stage), torch.int32) if n_mask_stage > 0 else None
    return plan_ws, q_lens, kv_lens, stage


def carve_workspace(workspace, plan_bytes: int, n_q_lens: int, owner: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(the plan's own chunk or None, the Q-side lengths as an int32 view or None)`` -- :func:`carve_workspace_chunks` for the forms
    without a derived-lengths chunk (the ``kv_lengths`` form and the plain ones)."""
    plan_ws, q_lens, _kv_lens, _stage = carve_workspace_chunks(workspace, plan_bytes, n_q_lens, 0, 0, owner)
    return plan_ws, q_lens


def prefix_check_enabled() -> bool:
    """Whether :data:`CHECK_PREFIX_MASK_ENV` turns the prefix-mask debug check on for the calls that do not decide it themselves."""
    return os.environ.get(CHECK_PREFIX_MASK_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _check_mask_arg(mask) -> Tuple[int, int, int]:
    """``(B, N, S_kv)`` of a boolean ``(B, N, 1, 1, S_kv)`` mask, else the typed ``ValueError``."""
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError(f"mask must be a boolean torch.Tensor, got {type(mask).__name__} {getattr(mask, 'dtype', None)}")
    if mask.ndim != 5 or mask.shape[2] != 1 or mask.shape[3] != 1:
        raise ValueError(f"mask must be (B, N, 1, 1, S_kv), got {tuple(mask.shape)}")
    b, n, _, _, s_kv = mask.shape
    return int(b), int(n), int(s_kv)


def _row_counts_on_device(mask: torch.Tensor, lens: torch.Tensor, stream) -> None:
    """Each row's True count of the boolean ``(B, N, 1, 1, S_kv)`` ``mask`` into ``lens`` -- ``B * N`` contiguous int32 words of memory the
    caller owns (any view of them, row-major over ``(b, n)``) -- on ``stream`` (a raw handle, a torch stream or None = torch's current
    stream on the mask's device).  On a CUDA mask this is ONE launch of the row-popcount kernel (``kernels/prefix_lengths.py``: a warp per
    row, byte loads, ``popc(ballot)``, one int32 store): no cast pass, no staging, no allocation, nothing read on the host, and a CUDA
    graph captures it like any launch (a replay re-derives the lengths from the mask's contents at replay time).  The count is bitwise
    ``mask.sum(-1, dtype=torch.int32)`` on every mask and every layout (a batch slice, a padded row, a strided key view): a non-prefix
    row counts its True keys -- the asserted form's documented behaviour -- and a row with no True key reads 0.  A CPU mask takes torch's
    reduction (the GPU-free contract tests; no kernel, no stream).  A DSL below the floor, or one without this device's target, is a
    typed ``NotImplementedError`` naming the version (the kernel module's own gate)."""
    global _launch_prefix_lengths
    shape = mask.shape
    b, n, s_kv = shape[0], shape[1], shape[4]
    if not mask.is_cuda:
        torch.sum(mask.reshape(b, n, s_kv), dim=-1, dtype=torch.int32, out=lens.view(b, n))
        return
    if _launch_prefix_lengths is None:
        from .kernels.prefix_lengths import launch_prefix_lengths

        _launch_prefix_lengths = launch_prefix_lengths
    strides = mask.stride()
    # a unit axis carries an arbitrary stride in torch; it is never walked, so it is passed as 0
    _launch_prefix_lengths(
        mask.data_ptr(),
        lens.data_ptr(),
        b,
        n,
        s_kv,
        strides[0] if b > 1 else 0,
        strides[1] if n > 1 else 0,
        strides[4] if s_kv > 1 else 0,
        stream if isinstance(stream, int) else _stream_int(stream, mask.device),
        mask.get_device(),
    )


_launch_prefix_lengths = None  # the kernel module's launcher, imported on the first CUDA derivation (the package import stays torch- and DSL-free)


def assert_prefix_mask(mask: torch.Tensor, lengths: torch.Tensor) -> None:
    """The prefix-mask DEBUG check: every row of ``mask`` must be True exactly on ``[0, lengths[b, n])`` -- a device-wide test with ONE host
    synchronisation; a violation is a typed ``ValueError`` naming the rows.  Never call it inside a CUDA-graph capture."""
    b, n, s_kv = _check_mask_arg(mask)
    prefix = torch.arange(s_kv, device=mask.device, dtype=torch.int32).view(1, 1, 1, 1, s_kv) < lengths.view(b, n, 1, 1, 1)
    bad = (mask != prefix).view(b, n, s_kv).any(dim=-1)
    n_bad = int(bad.sum().item())  # the host synchronisation of the check
    if n_bad:
        rows = bad.nonzero()[:8].tolist()
        raise ValueError(
            f"triangle_attention: mask_is_prefix asserts a PREFIX-shaped mask (every row True on [0, len) and False after), but {n_bad} of the "
            f"{b * n} (b, n) rows violate it, e.g. (b, n) = {rows}: served as the prefix of its True count such a row attends the WRONG keys. "
            "Pass the mask without mask_is_prefix (the in-kernel dense mask) or repair it; mask_to_kv_lengths(mask) returns None for it."
        )


def prefix_mask_workspace_bytes(b: int, n: int, s_kv: int) -> int:
    """Bytes of the ``workspace`` :func:`prefix_mask_to_kv_lengths` needs: **0** -- the fused kernel reads the mask's bytes and writes the
    counts, staging nothing.  (The two-kernel form this function sized for staged the mask as int32, ``4 * B * N * S_kv`` bytes; the
    function and the ``workspace`` keyword stay so a caller written against it keeps working.)"""
    return 0


def prefix_mask_to_kv_lengths(
    mask: torch.Tensor, out: Optional[torch.Tensor] = None, *, workspace: Optional[torch.Tensor] = None, check: Optional[bool] = None, stream=None
) -> torch.Tensor:
    """``(B, N, 1, 1, 1)`` int32 right-padded lengths of a boolean mask ``(B, N, 1, 1, S_kv)`` the CALLER asserts prefix-shaped, computed on
    the device with no host synchronisation -- for a caller that holds a mask and a plan of its own (``TriangleAttentionFwd(...,
    kv_lengths=True)``, the graph's ``seq_len_kv``, cuEquivariance's ``kv_lengths``).  The sync-free twin of :func:`mask_to_kv_lengths`:
    that one TESTS the shape (and returns None when it fails) at the price of one synchronisation; this one TRUSTS the assertion.

    ONE launch of the row-popcount kernel (``kernels/prefix_lengths.py``: a warp per row, byte loads, ``popc(ballot)``, one int32 store):
    no cast pass, no staging, no allocation beyond ``out``, nothing read on the host.  ``out`` -- the int32 ``(B, N, 1, 1, 1)`` contiguous
    tensor to write (allocated when None); with it given the call allocates nothing and a CUDA graph captures it (a replay re-derives
    the lengths from the mask's contents).  ``workspace`` -- accepted for compatibility with the two-kernel form this function shipped
    with first and UNUSED: the fused kernel stages nothing (:func:`prefix_mask_workspace_bytes` is 0); a value must still be a contiguous
    uint8 tensor on the mask's device.  ``stream`` -- the raw handle or torch stream the launch is ordered on (None = torch's current
    stream).  ``check`` -- run :func:`assert_prefix_mask` after the count (ONE host synchronisation; a typed ``ValueError`` on a non-prefix
    mask); None = :data:`CHECK_PREFIX_MASK_ENV` decides.  Without the check a non-prefix row yields its True count, i.e. the prefix of
    that length (documented, not detected).  A row with no True key yields 0 -- the ``kv_lengths`` route's fully masked row (``o = 0``,
    ``lse = max = -inf`` on this provider).  Any mask layout is served (a batch slice, a padded row stride, a strided key view); a CPU
    mask takes torch's reduction.  A DSL below the floor is a typed ``NotImplementedError`` naming the version."""
    b, n, s_kv = _check_mask_arg(mask)
    dev = mask.device
    if out is None:
        out = torch.empty((b, n, 1, 1, 1), dtype=torch.int32, device=dev)
    else:
        if not isinstance(out, torch.Tensor) or out.dtype != torch.int32 or tuple(out.shape) != (b, n, 1, 1, 1):
            raise ValueError(
                f"out must be an int32 (B, N, 1, 1, 1) = {(b, n, 1, 1, 1)} tensor, got {type(out).__name__} {getattr(out, 'dtype', None)} {tuple(getattr(out, 'shape', ()))}"
            )
        if not out.is_contiguous() or out.device != dev:
            raise ValueError(f"out must be contiguous and on the mask's device {dev}, got strides {tuple(out.stride())} on {out.device}")
    if workspace is not None and (
        not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8 or not workspace.is_contiguous() or workspace.device != dev
    ):
        raise ValueError(
            f"workspace must be a contiguous uint8 tensor on the mask's device {dev} (unused by the fused derivation; accepted for compatibility), "
            f"got {type(workspace).__name__} {getattr(workspace, 'dtype', None)} on {getattr(workspace, 'device', None)}"
        )
    _row_counts_on_device(mask, out, stream)
    if check if check is not None else prefix_check_enabled():
        assert_prefix_mask(mask, out)
    return out


# ---------------------------------------------------------------------------
# The prepared API
# ---------------------------------------------------------------------------
class TriangleAttentionFwd(APIBase):
    """Triangle attention, forward: one declaration, one compiled plan, one launch per outer batch entry on one stream.

    Declaration (the constructor): ``sample_q`` / ``sample_k`` / ``sample_v`` ``(B, N, H, S, D)`` and ``sample_bias``
    ``(1|B, 1, 1|H, S_q, S_kv)`` (tensors or :class:`cudnn.api_base.TensorDesc`; their dtype AND strides are compile-time facts --
    the kernel's TMA descriptors are built from them), the key-mask form (``kv_lengths=True`` -> right-padded int32 lengths
    ``(B, N, 1, 1, 1)``; ``mask=True`` -> a dense boolean mask ``(B, N, 1, 1, S_kv)``; ``mask=True, mask_is_prefix=True`` -> the same
    boolean mask ASSERTED prefix-shaped, its row lengths derived on the device at ``execute`` and served through the ``kv_lengths`` graph
    form; neither -> every key live; both is a ``ValueError``, as in cuEquivariance), the outputs (``lse=True`` adds the fp32 log-sum-exp ``(B, N, H, S_q)``; ``score_max=True``
    requests the fp32 row max -- declined by the engine today), the softmax ``scale`` (None = ``1 / sqrt(D)``) and optionally
    ``sample_o`` (the output's layout; None = the kernels' compact BSHD form presented as ``(B, N, H, S, D)``).

    ``check_support`` raises the typed declines (malformed -> ``ValueError``; unserved -> ``NotImplementedError`` quoting the
    engine row's own reason), ``compile`` builds the plan (the FROST engine row's lowering, selected by name and pinned),
    ``get_workspace_size`` is the plan's requirement (0 on most declarations) plus, on the ``kv_lengths`` form, ``N`` int32 words for the
    Q-side lengths the graph's padding contract requires (``execute`` writes ``S_q`` there on its stream) and, on the ``prefix_mask`` form,
    ``B * N`` int32 words for the derived KV lengths plus ``B * N * S_kv`` int32 words the mask is staged in, ``execute(q, k, v, bias,
    kv_lengths, o, lse, score_max, workspace, current_stream, *, mask, check_prefix_mask)`` validates (device, dtype, shape, stride, the
    base alignment each operand's reader needs, form) and launches -- no allocation, no conversion, no synchronisation (the prefix-mask
    debug check, off by default, is the documented exception); a captured CUDA graph replays it.
    ``B > 1`` is served as ``B`` launches of the one plan on the per-entry slices (the bias of entry ``b`` is broadcast over its ``N``
    rows), ``launches_per_call`` says how many.
    """

    def __init__(
        self,
        sample_q,
        sample_k,
        sample_v,
        sample_bias,
        *,
        kv_lengths: bool = False,
        mask: bool = False,
        lse: bool = False,
        score_max: bool = False,
        scale: Optional[float] = None,
        sample_o=None,
        mask_is_prefix: bool = False,
    ):
        super().__init__()
        self._warn_experimental_api()
        if kv_lengths and mask:
            raise ValueError("pass either mask or kv_lengths, not both")
        if mask_is_prefix and not mask:
            raise ValueError(
                "mask_is_prefix=True requires mask=True: it is the prefix-asserted form of the boolean mask (pass kv_lengths=True when you hold lengths)"
            )
        q_shape, q_stride, q_dtype, q_dev = _declaration(sample_q, "sample_q")
        k_shape, k_stride, k_dtype, k_dev = _declaration(sample_k, "sample_k")
        v_shape, v_stride, v_dtype, v_dev = _declaration(sample_v, "sample_v")
        b_shape, b_stride, b_dtype, b_dev = _declaration(sample_bias, "sample_bias")
        for name, shape in (("sample_q", q_shape), ("sample_k", k_shape), ("sample_v", v_shape)):
            if len(shape) != 5:
                raise ValueError(f"{name} must be rank 5 (B, N, H, S, D), got shape {shape}")
        if len(b_shape) != 5:
            raise ValueError(f"sample_bias must be rank 5 (B, 1, H, S_q, S_kv), got shape {b_shape}")
        self.batch, self.n, self.h, self.s_q, self.d = q_shape
        _, _, self.h_kv, self.s_kv, d_k = k_shape
        if k_shape[:2] != q_shape[:2] or v_shape[:2] != q_shape[:2]:
            raise ValueError(f"q, k, v must agree on (B, N): q {q_shape}, k {k_shape}, v {v_shape}")
        if v_shape[2] != self.h_kv or v_shape[3] != self.s_kv:
            raise ValueError(f"k and v must agree on (H, S_kv): k {k_shape}, v {v_shape}")
        if d_k != self.d or v_shape[4] != self.d:
            raise ValueError(f"q, k, v must share the head dim D: q {q_shape}, k {k_shape}, v {v_shape} (D_qk == D_v on this row)")
        if self.h_kv < 1 or self.h % self.h_kv:
            raise ValueError(f"the KV head count must divide the Q head count (GQA), got H={self.h}, H_kv={self.h_kv}")
        if b_shape[0] not in (1, self.batch) or b_shape[1] != 1 or b_shape[2] not in (1, self.h) or b_shape[3:] != (self.s_q, self.s_kv):
            raise ValueError(f"bias must be (B|1, 1, H|1, S_q, S_kv) = ({self.batch}|1, 1, {self.h}|1, {self.s_q}, {self.s_kv}), got {b_shape}")
        if q_dtype != k_dtype or q_dtype != v_dtype:
            raise ValueError(f"q, k, v must share one dtype, got {q_dtype}, {k_dtype}, {v_dtype}")
        if not all(dev == q_dev for dev in (k_dev, v_dev, b_dev)):
            raise ValueError(f"q, k, v, bias must live on one device, got {q_dev}, {k_dev}, {v_dev}, {b_dev}")
        if scale is not None and not (float(scale) > 0.0 and math.isfinite(float(scale))):
            raise ValueError(f"scale must be a positive finite float, got {scale!r}")
        self.dtype = q_dtype
        self.bias_dtype = b_dtype
        self.device = torch.device(q_dev)
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(self.d)
        self.mask_form = "kv_lengths" if kv_lengths else (("prefix_mask" if mask_is_prefix else "mask") if mask else "none")
        self.want_lse = bool(lse)
        self.want_score_max = bool(score_max)
        # The 4-D geometry ONE launch binds (the per-outer-entry slice), plus the outer stride that steps the slices.
        self.q4, self.k4, self.v4 = _row_strides(q_shape, q_stride), _row_strides(k_shape, k_stride), _row_strides(v_shape, v_stride)
        self.outer_strides = (q_stride[0], k_stride[0], v_stride[0], b_stride[0] if b_shape[0] > 1 else 0)
        self.bias_broadcast_batch = b_shape[0] == 1
        self.bias4 = (tuple(b_shape[1:]), tuple(b_stride[1:]))  # (1, H|1, S_q, S_kv) per entry
        if sample_o is None:
            self.o4 = ((self.n, self.h, self.s_q, self.d), (self.s_q * self.h * self.d, self.d, self.h * self.d, 1))
            self.o_outer_stride = self.n * self.s_q * self.h * self.d
        else:
            o_shape, o_stride, o_dtype, o_dev = _declaration(sample_o, "sample_o")
            if o_shape != q_shape or o_dtype != q_dtype or torch.device(o_dev) != self.device:
                raise ValueError(f"sample_o must be a {q_shape} {q_dtype} tensor on {self.device}, got {o_shape} {o_dtype} on {o_dev}")
            self.o4 = _row_strides(o_shape, o_stride)
            self.o_outer_stride = o_stride[0]
        self.fallback_copies = 0  # incremented by the drop-in function when it repacked a tensor for this plan
        self._graph = None
        self._plan_index: Optional[int] = None
        self._handle = None
        self._uids: Tuple[int, ...] = ()
        self._workspace_bytes = 0  # what execute needs: the plan's own bytes (+ the Q-side lengths chunk on the kv_lengths form)
        self._plan_workspace_bytes = 0
        self.plan_name: Optional[str] = None
        self.knobs: Optional[dict] = None

    # -- declaration checks -------------------------------------------------------------------------------------------------------
    @property
    def device_index(self) -> int:
        """The declaration's CUDA device index (``torch.cuda.current_device()`` when the declaration named ``cuda`` without one)."""
        return self.device.index if self.device.index is not None else torch.cuda.current_device()

    @property
    def launches_per_call(self) -> int:
        """Kernel launches per ``execute``: one per outer batch entry (the bias of entry ``b`` is broadcast over its ``N`` rows)."""
        return self.batch

    @property
    def _n_q_lens(self) -> int:
        """The int32 words of Q-side lengths the caller's workspace carries: ``N`` on the ``kv_lengths`` and ``prefix_mask`` forms, else 0."""
        return self.n if self.mask_form in ("kv_lengths", "prefix_mask") else 0

    @property
    def _n_kv_lens(self) -> int:
        """The int32 words of DERIVED KV lengths the caller's workspace carries: ``B * N`` on the ``prefix_mask`` form, else 0."""
        return self.batch * self.n if self.mask_form == "prefix_mask" else 0

    @property
    def _n_mask_stage(self) -> int:
        """The int32 words of mask staging the caller's workspace carries: 0 on every form -- the fused derivation kernel reads the boolean
        mask directly (the two-kernel form staged it as ``B * N * S_kv`` words; the chunk slot stays in the carve, empty)."""
        return 0

    def check_support(self) -> bool:
        """Typed declines: malformed declarations first (``ValueError``), then the serving envelope -- dtype, arch, the DSL, the FROST
        opt-in -- and finally the engine row's own verdict on the real graph (``NotImplementedError`` quoting its reason)."""
        if self.batch < 1 or self.n < 1 or self.h < 1 or self.s_q < 1 or self.s_kv < 1 or self.d < 1:
            raise ValueError(f"every extent must be >= 1, got (B, N, H, S_q, S_kv, D) = {(self.batch, self.n, self.h, self.s_q, self.s_kv, self.d)}")
        if self.device.type != "cuda":
            raise ValueError(f"q, k, v, bias must be CUDA tensors, got device {self.device}")
        if self.dtype not in _DTYPES:
            raise NotImplementedError(f"triangle_attention serves bf16 and fp16 activations, got {self.dtype}")
        if self.bias_dtype != self.dtype:
            raise NotImplementedError(
                f"the bias must be the activations' dtype ({self.dtype}) on this engine (the resident block loads the io dtype); got {self.bias_dtype}. "
                "The drop-in function casts an fp32 bias once per call (counted)."
            )
        for name, (shape4, stride4) in (("q", self.q4), ("k", self.k4), ("v", self.v4), ("o", self.o4)):
            if not activation_binds_zero_copy(shape4, stride4):
                raise NotImplementedError(
                    f"{name}'s layout (per-entry shape {shape4}, strides {stride4}) does not bind to the kernel without a copy: the head dim must be "
                    "contiguous, the token / head / entry strides 16-byte multiples and the nest token-major (stride(S) >= H * stride(H); the "
                    "(B, N, S, H, D) memory form).  The drop-in function repacks such a tensor with a counted copy."
                )
        if not bias_binds_zero_copy(*self.bias4):
            raise NotImplementedError(
                f"the bias layout (per-entry shape {self.bias4[0]}, strides {self.bias4[1]}) does not bind: its key dim must be contiguous and every "
                "live stride a 16-byte multiple (S_kv % 8 == 0 for a compact bias, else a padded row stride).  The drop-in function pads it (counted)."
            )
        if self.batch > 1:
            for name, st in zip(("q", "k", "v", "bias", "o"), self.outer_strides + (self.o_outer_stride,)):
                if (st * _ELEM_BYTES) % _TMA_ALIGN:
                    raise NotImplementedError(
                        f"{name}'s outer batch stride {st} elements is not a 16-byte multiple: the per-entry slices would be misaligned for TMA"
                    )
        cc = tuple(torch.cuda.get_device_capability(self.device_index))
        if cc != _RUBIN:
            raise NotImplementedError(
                f"triangle_attention is served by the FROST row {ENGINE_ROW} (compute capability 10.7, Rubin) only today; found {cc[0]}.{cc[1]}"
            )
        msg = cutedsl_requirement_error("triangle_attention")
        if msg:
            raise NotImplementedError(msg)
        msg = cutedsl_arch_requirement_error(cc)
        if msg:
            raise NotImplementedError(msg)
        from cudnn.engines.manifest import opt_in_engines_enabled

        if not opt_in_engines_enabled():
            raise NotImplementedError(
                f"triangle_attention runs on the FROST engine row {ENGINE_ROW}, which is opt-in: set {_FROST_ENV}=1 in the environment before importing cudnn"
            )
        with torch.cuda.device(self.device_index):
            self._graph, self._plan_index = self._build_graph()
        self._is_supported = True
        return True

    def _io_dtype(self, cudnn):
        return cudnn.data_type.HALF if self.dtype == torch.float16 else cudnn.data_type.BFLOAT16

    def _stats_layout(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        n, h, s = self.n, self.h, self.s_q
        return (n, h, s, 1), (h * s, s, 1, 1)

    def _build_graph(self):
        """The ``sdpa`` graph of ONE outer batch entry on the declared layouts, planned; returns ``(graph, index of the row's plan)`` or
        raises the engine row's own reason."""
        import cudnn

        io = self._io_dtype(cudnn)
        f32 = cudnn.data_type.FLOAT
        g = cudnn.pygraph(io_data_type=io, intermediate_data_type=f32, compute_data_type=f32)
        tq = g.tensor(dim=list(self.q4[0]), stride=list(self.q4[1]), data_type=io, name="q")
        tk = g.tensor(dim=list(self.k4[0]), stride=list(self.k4[1]), data_type=io, name="k")
        tv = g.tensor(dim=list(self.v4[0]), stride=list(self.v4[1]), data_type=io, name="v")
        tb = g.tensor(dim=list(self.bias4[0]), stride=list(self.bias4[1]), data_type=io, name="bias")
        kw: Dict[str, Any] = dict(name="sdpa", q=tq, k=tk, v=tv, bias=tb, generate_stats=self.want_lse, attn_scale=self.scale)
        order = [tq, tk, tv, tb]
        if self.mask_form == "mask":
            tm = g.tensor(dim=[self.n, 1, 1, self.s_kv], stride=[self.s_kv, self.s_kv, self.s_kv, 1], data_type=cudnn.data_type.BOOLEAN, name="kv_mask")
            kw["kv_mask"] = tm
            order.append(tm)
        elif self.mask_form in ("kv_lengths", "prefix_mask"):
            # The graph's padding contract takes a length tensor on BOTH sides (the FE validator: "Padding mask requires
            # seq_len_q/seq_len_kv"), and the kernel bounds the live Q rows by seq_len_q.  Every Q row is live here, so seq_len_q is the
            # constant S_q on every entry: `execute` writes it into the caller's workspace on the launch stream and binds that region
            # (execute scratch carved from the workspace, declared by get_workspace_size -- never a plan-owned tensor, which would be
            # released at GC time, possibly inside someone else's CUDA-graph capture).  Nothing is allocated at build or at execute.
            slk = g.tensor(dim=[self.n, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="seq_len_kv")
            slq = g.tensor(dim=[self.n, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="seq_len_q")
            kw.update(seq_len_kv=slk, seq_len_q=slq, use_padding_mask=True)
            order.extend([slk, slq])
        sm = None
        if self.want_score_max:
            dim, stride = self._stats_layout()
            sm = g.tensor(dim=list(dim), stride=list(stride), data_type=f32, name="score_max")
            kw["score_max"] = sm
        try:
            o, stats = g.sdpa(**kw)
            o.set_output(True).set_dim(list(self.o4[0])).set_stride(list(self.o4[1]))
            order.append(o)
            if self.want_lse:
                dim, stride = self._stats_layout()
                stats.set_output(True).set_dim(list(dim)).set_stride(list(stride)).set_data_type(f32)
                order.append(stats)
            if sm is not None:
                sm.set_output(True)
                order.append(sm)
            g.validate()
            g.build_operation_graph()
        except cudnn.cudnnGraphNotSupportedError as error:
            raise ValueError(f"triangle_attention: the graph is not valid for this declaration: {error}") from error
        try:
            g.create_execution_plans([cudnn.heur_mode.A])
        except cudnn.cudnnGraphNotSupportedError as error:
            raise NotImplementedError(f"triangle_attention: no engine proposes a plan for this declaration -- {self._engine_reason(g, error)}") from error
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        ours = [i for i, nm in enumerate(names) if nm == ENGINE_ROW or nm.startswith(ENGINE_ROW + "[")]
        idx = next((i for i in ours if getattr(g.plans[i].knobs, "split_kv", None) in (None, 1)), None)
        if idx is None:
            raise NotImplementedError(f"triangle_attention: the FROST row {ENGINE_ROW} declines this declaration -- {self._engine_reason(g, None)}")
        self._order = order
        self.plan_name = names[idx]
        return g, idx

    @staticmethod
    def _engine_reason(g, error) -> str:
        """The engine row's own typed decline string for this graph (``engines.analyze_for`` -> ``mismatch``), the FE's error as the fallback."""
        try:
            from cudnn.sdpa.fwd import engines

            spec = next(sp for sp in engines.ENGINE_SPECS if sp.name == ENGINE_ROW)
            _facts, why = engines.analyze_for(spec, g, None)
            if why:
                return f"{ENGINE_ROW}: {why}"
        except Exception as probe_error:  # noqa: BLE001 -- the probe must never hide the decline itself
            return f"{ENGINE_ROW} could not be asked ({type(probe_error).__name__}: {probe_error})" + (f"; {error}" if error is not None else "")
        return f"{ENGINE_ROW} proposed no unsplit plan" + (f"; {error}" if error is not None else "")

    # -- compile ------------------------------------------------------------------------------------------------------------------
    def compile(self) -> None:
        """Pin the FROST row's plan on the graph and build it on the declaration's device (a compiled launchable is bound to the CUDA
        context it was loaded in); records the plan's name, its public knob record and its workspace requirement."""
        self._ensure_support_checked()
        import cudnn

        g = self._graph
        with torch.cuda.device(self.device_index):
            g.select_plan(self._plan_index)
            g.check_support()
            g.build_plans()
            self._plan_workspace_bytes = int(g.get_workspace_size())
            self._handle = cudnn.create_handle()
        self._workspace_bytes = workspace_bytes(self._plan_workspace_bytes, self._n_q_lens, self._n_kv_lens, self._n_mask_stage)
        if self.mask_form == "prefix_mask":
            # The derivation kernel's artifact for this device, built HERE so the first execute (a graph capture included) compiles nothing.
            from .kernels.prefix_lengths import prefix_lengths_plan

            prefix_lengths_plan(self.device_index)
        _engine_id, record = g.get_engine_and_knobs_at_index(self._plan_index)
        self.knobs = dict(record) if record else {}
        self._uids = tuple(t.uid for t in self._order)
        self._frame = None  # the bound argument frame (built by the engine's binder on the first execute, pointers refreshed per call)
        compiled = g._compiled_plans.get(self._plan_index)
        self._compiled_plan = compiled
        spec = getattr(getattr(compiled, "_prepared", None), "spec", None)
        # The pointer-refresh path serves the plain dense launch only: a prepared launch without a split / combine, a quantized
        # recipe, a native (C) binder or a ragged leg, and no workspace.  Anything else runs through the graph's execute.
        self._fast_spec = (
            spec
            if (
                spec is not None
                and self._plan_workspace_bytes == 0
                and getattr(spec, "combine", None) is None
                and getattr(spec, "quant", None) is None
                and getattr(spec, "native", None) is None
                and not getattr(spec, "ragged", False)
                and getattr(spec, "split", 1) == 1
            )
            else None
        )
        self._compiled_kernel = g

    def get_workspace_size(self) -> int:
        """Bytes of the workspace ``execute`` needs: the plan's own requirement (0 on most declarations; the dense-mask form at an
        ``S_kv`` that is not a multiple of the kv tile carries the engine's synthesized padding lengths) plus, on the ``kv_lengths`` and
        ``prefix_mask`` forms, ``N`` int32 words for the Q-side lengths the graph's padding contract requires -- ``execute`` writes ``S_q``
        there on its stream before the launches -- and, on the ``prefix_mask`` form, ``B * N`` int32 words for the KV lengths the device
        derives (the fused kernel stages nothing).  Sized and carved by the SDPA adapters' shared recipe (:func:`workspace_bytes`,
        :func:`carve_workspace_chunks`)."""
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before get_workspace_size()")
        return self._workspace_bytes

    # -- execute ------------------------------------------------------------------------------------------------------------------
    def _check_tensor(self, t, name: str, shape, stride, dtype, align: int = _TMA_ALIGN) -> None:
        """One operand against the declaration; ``align`` is the base alignment ITS reader needs -- the binder's own rule per role
        (16 bytes for the TMA-read q / k / v / bias / o, 4 for the int32 lengths and the fp32 lse, none for the byte mask), never a
        blanket 16: a check stricter than the binder's declines a valid per-entry slice with a wrong reason."""
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.device.type != "cuda" or t.device.index != self.device_index:
            raise ValueError(
                f"{name} lives on {t.device}; this plan was declared on cuda:{self.device_index} and its kernel is loaded there (declare and compile one plan per device)"
            )
        if t.dtype != dtype:
            raise ValueError(f"{name} must be {dtype} (the declaration), got {t.dtype}")
        if tuple(t.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)} (the declaration), got {tuple(t.shape)}")
        if stride is not None and tuple(t.stride()) != tuple(stride):
            raise ValueError(
                f"{name} must have strides {tuple(stride)} (the declaration; the kernel's descriptors are built from them), got {tuple(t.stride())}"
            )
        if align > 1 and t.data_ptr() % align:
            raise ValueError(f"{name} must be {align}-byte aligned (its base address is {t.data_ptr() % align} bytes past one)")

    def _check_execute_args(self, q, k, v, bias, kv_lengths, o, lse, score_max, workspace, mask) -> None:
        b, n, h, s_q, s_kv = self.batch, self.n, self.h, self.s_q, self.s_kv
        shape5 = (b, n, h, s_q, self.d)
        self._check_tensor(q, "q", shape5, (self.outer_strides[0],) + self.q4[1], self.dtype)
        self._check_tensor(k, "k", (b, n, self.h_kv, s_kv, self.d), (self.outer_strides[1],) + self.k4[1], self.dtype)
        self._check_tensor(v, "v", (b, n, self.h_kv, s_kv, self.d), (self.outer_strides[2],) + self.v4[1], self.dtype)
        bias_b = 1 if self.bias_broadcast_batch else b
        self._check_tensor(bias, "bias", (bias_b,) + self.bias4[0], ((self.outer_strides[3],) if bias_b > 1 else (bias.stride(0),)) + self.bias4[1], self.dtype)
        if o is None:
            raise ValueError("o is required: the prepared API never allocates (the drop-in function does)")
        self._check_tensor(o, "o", shape5, (self.o_outer_stride,) + self.o4[1], self.dtype)
        if self.mask_form == "kv_lengths":
            if kv_lengths is None:
                raise ValueError(
                    "this plan was declared with kv_lengths=True; an int32 (B, N, 1, 1, 1) kv_lengths tensor is required (no silent all-live fallback)"
                )
            self._check_tensor(kv_lengths, "kv_lengths", (b, n, 1, 1, 1), None, torch.int32, align=_ALIGN_WORD)
            if kv_lengths.stride(1) != 1:
                raise ValueError(f"kv_lengths must be contiguous along N (stride 1), got strides {tuple(kv_lengths.stride())}")
            if mask is not None:
                raise ValueError("this plan was declared with kv_lengths=True; mask must be None")
        elif self.mask_form in ("mask", "prefix_mask"):
            prefix = self.mask_form == "prefix_mask"
            if mask is None:
                raise ValueError(
                    "this plan was declared with mask=True, mask_is_prefix=True; a boolean (B, N, 1, 1, S_kv) mask tensor is required (its row lengths are derived on the device)"
                    if prefix
                    else "this plan was declared with mask=True; a boolean (B, N, 1, 1, S_kv) mask tensor is required (no silent all-live fallback)"
                )
            self._check_tensor(mask, "mask", (b, n, 1, 1, s_kv), None, torch.bool, align=_ALIGN_BYTE)
            if mask.stride(4) != 1 or mask.stride(1) != s_kv:
                raise ValueError(
                    f"mask must be contiguous ((B, N, 1, 1, S_kv) with stride 1 along the keys and S_kv along N), got strides {tuple(mask.stride())}"
                )
            if kv_lengths is not None:
                raise ValueError(
                    "this plan was declared with mask=True, mask_is_prefix=True; kv_lengths must be None (the lengths are derived from the mask on the device)"
                    if prefix
                    else "this plan was declared with mask=True; kv_lengths must be None"
                )
        elif kv_lengths is not None or mask is not None:
            raise ValueError("this plan was declared without a key mask; declare kv_lengths=True or mask=True to apply one")
        if self.want_lse:
            if lse is None:
                raise ValueError("this plan was declared with lse=True; an fp32 (B, N, H, S_q) lse tensor is required")
            self._check_tensor(lse, "lse", (b, n, h, s_q), (n * h * s_q, h * s_q, s_q, 1), torch.float32, align=_ALIGN_WORD)
        elif lse is not None:
            raise ValueError("this plan was declared with lse=False; lse must be None (declare lse=True to request it)")
        if score_max is not None:
            raise ValueError("score_max is not served by this plan (the engine row declines the row-max side output on the bias arm)")
        if self._workspace_bytes and isinstance(workspace, torch.Tensor):
            # A missing, undersized, non-tensor or misaligned workspace raises the shared carver's own error (carve_workspace, at
            # execute); the two facts the carver does not check are the buffer's device (the plan's) and its contiguity (the chunks
            # are views of the caller's bytes -- a strided buffer would be carved on a copy).
            if workspace.device.type != "cuda" or workspace.device.index != self.device_index:
                raise ValueError(f"workspace lives on {workspace.device}; this plan was declared on cuda:{self.device_index} and its kernel is loaded there")
            if not workspace.is_contiguous():
                raise ValueError(
                    f"workspace must be contiguous (its chunks are carved as views of the caller's bytes), got strides {tuple(workspace.stride())}"
                )
        # The buffers the kernel WRITES (o, the lse, the workspace's lengths / scratch) must not intersect any buffer it READS (q, k, v, the
        # bias, the lengths, the mask), nor each other -- as BYTE RANGES, never by pointer equality: an `o` view starting inside q (two views
        # over one allocation) has a distinct data_ptr and the kernel overwrites rows it has still to read.  Inputs may overlap each other
        # (q / k / v as slices of one projection buffer are read-only).  A typed ValueError naming both operands, before any launch.
        check_buffers_disjoint(
            (("o", o), ("lse", lse), ("workspace", workspace if self._workspace_bytes and isinstance(workspace, torch.Tensor) else None)),
            (("q", q), ("k", k), ("v", v), ("bias", bias), ("kv_lengths", kv_lengths), ("mask", mask)),
            what="the kernel",
        )

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        kv_lengths: Optional[torch.Tensor] = None,
        o: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        score_max: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream=None,
        *,
        mask: Optional[torch.Tensor] = None,
        check_prefix_mask: Optional[bool] = None,
    ) -> None:
        """Launch the plan on ``current_stream`` (None = torch's current stream on the plan's device): one launch per outer batch
        entry.  Validates every tensor against the declaration, never converts or allocates.

        Host path: the engine's prepared launch binds its argument frame ONCE per plan (the engine's own binder, on the first call --
        every dtype / layout / alignment / span rule of the row); every later call re-checks the tensors against the declaration here
        and refreshes the frame's POINTER slots (the geometry is the declaration's, fixed), then issues the kernel -- the host floor of
        an eager call is the API's checks plus the launch itself.  A plan the fast path does not serve (a plan workspace, a split, a
        native binder) runs through the graph's execute with the same operands.  The ``kv_lengths`` form first writes the Q-side lengths
        (``S_q`` on every entry) into the caller's workspace on the same stream -- one driver memset, stream-ordered before the launches,
        a memset node in a captured graph.  The ``prefix_mask`` form then derives the KV lengths from the boolean ``mask`` on the same
        stream (:func:`_row_counts_on_device`: ONE launch of the row-popcount kernel into the lengths chunk -- no cast pass, no staging,
        no allocation, nothing read on the host; captured by a graph) and binds them as the ``kv_lengths`` form does.
        ``check_prefix_mask`` (that form only): True runs :func:`assert_prefix_mask` after the derivation -- the device-wide prefix test
        with its ONE host synchronisation, a typed ``ValueError`` on a mask that is not prefix-shaped -- False skips it, None (default)
        lets :data:`CHECK_PREFIX_MASK_ENV` decide; never True inside a CUDA-graph capture."""
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        self._check_execute_args(q, k, v, bias, kv_lengths, o, lse, score_max, workspace, mask)
        stream_int = _stream_int(current_stream, self.device)
        plan_ws = q_lens = kv_lens = stage = None
        if self._workspace_bytes:
            plan_ws, q_lens, kv_lens, stage = carve_workspace_chunks(
                workspace, self._plan_workspace_bytes, self._n_q_lens, self._n_kv_lens, self._n_mask_stage, type(self).__name__
            )
        if q_lens is not None:
            self._write_q_lengths(q_lens, stream_int)
        if self.mask_form == "prefix_mask":
            kv_lengths = self._derive_kv_lengths(mask, kv_lens, stream_int, check_prefix_mask)
            mask = None  # from here on the call IS the kv_lengths form
        if self._fast_spec is not None:
            self._execute_frames(q, k, v, bias, kv_lengths, mask, o, lse, q_lens, stream_int)
        else:
            self._execute_graph(q, k, v, bias, kv_lengths, mask, o, lse, plan_ws, q_lens, stream_int)

    def _write_q_lengths(self, q_lens: torch.Tensor, stream_int: int) -> None:
        """The ``kv_lengths`` form: fill the carved ``N`` int32 Q-side lengths with ``S_q`` on the execute stream (one driver memset; no
        allocation, no host synchronisation) -- the graph's padding contract requires a Q-side length tensor and the kernel bounds the
        live Q rows by it; every Q row of a triangle-attention call is live."""
        from cudnn._device import ensure_current_context
        from cudnn.frost.buffers import fill_word_async

        ensure_current_context(stream_int, self.device_index)
        fill_word_async(q_lens.data_ptr(), self.n, self.s_q, stream_int)

    def _derive_kv_lengths(self, mask: torch.Tensor, kv_lens: torch.Tensor, stream_int: int, check: Optional[bool]) -> torch.Tensor:
        """The ``prefix_mask`` form: each row's True count, computed on the execute stream into the carved ``kv_lens`` chunk by ONE launch
        of the row-popcount kernel (the artifact :meth:`compile` built for this device), returned as the ``(B, N, 1, 1, 1)`` int32 view
        the ``kv_lengths`` form binds.  The mask is never read on the host; the optional debug check is."""
        _row_counts_on_device(mask, kv_lens, stream_int)
        lengths = kv_lens.view(self.batch, self.n, 1, 1, 1)
        if check if check is not None else prefix_check_enabled():
            assert_prefix_mask(mask, lengths)
        return lengths

    def _entry_views(self, q, k, v, bias, kv_lengths, mask, o, lse, b: int, q_lens=None) -> list:
        """The per-entry buffers in the graph's operand order (the declaration's 4-D geometry); with ``q_lens`` (the carved Q-side
        lengths) the ``kv_lengths`` form's ``(N, 1, 1, 1)`` view follows the KV lengths (the graph path), without it the role is bound
        from the carved view's facts (the frame path)."""
        bufs = [q[b], k[b], v[b], bias[0] if self.bias_broadcast_batch else bias[b]]
        if self.mask_form == "mask":
            bufs.append(mask[b])
        elif self.mask_form in ("kv_lengths", "prefix_mask"):
            bufs.append(kv_lengths[b])
            if q_lens is not None:
                bufs.append(q_lens.view(self.n, 1, 1, 1))
        bufs.append(o[b])
        if self.want_lse:
            bufs.append(lse[b])
        return bufs

    def _execute_graph(self, q, k, v, bias, kv_lengths, mask, o, lse, plan_ws, q_lens, stream_int: int) -> None:
        import cudnn

        cudnn.set_stream(self._handle, stream_int)
        g, handle, uids = self._compiled_kernel, self._handle, self._uids
        for b in range(self.batch):
            g.execute(tuple(self._entry_views(q, k, v, bias, kv_lengths, mask, o, lse, b, q_lens)), plan_ws, handle, tensor_uids=uids)

    # The engine's frame slots the per-call pointers land in, by role: (slot name, element bytes of the buffer).
    _PTR_SLOTS = (
        ("q", "q_ptr", 2),
        ("k", "k_ptr", 2),
        ("v", "v_ptr", 2),
        ("o", "o_ptr", 2),
        ("bias", "bias_ptr", 2),
        ("kv_mask", "kv_mask_ptr", 1),
        ("seq_kv_lens", "meta_ptr", 4),
        ("seq_q_lens", "seq_q_lens_addr", 4),
        ("lse", "lse_ptr", 4),
    )

    def _bind_frame(self, q, k, v, bias, kv_lengths, mask, o, lse, q_lens, cu_stream, stream_int: int) -> None:
        """The first call: the engine's binder validates entry 0's operands and builds the argument frame; the pointer slots and
        each role's per-entry byte step are recorded for every later call.  The ``kv_lengths`` form's Q-side lengths role is the
        carved workspace chunk (``q_lens``: ``N`` contiguous int32 words, its facts read like any operand's), shared by every entry."""
        from cudnn.sdpa.fwd import prepared as prep

        bufs = self._entry_views(q, k, v, bias, kv_lengths, mask, o, lse, 0)
        roles = ["q", "k", "v", "bias"]
        if self.mask_form == "mask":
            roles.append("kv_mask")
        elif self.mask_form in ("kv_lengths", "prefix_mask"):
            roles.append("seq_kv_lens")
        roles.append("o")
        if self.want_lse:
            roles.append("lse")
        facts = {role: prep.facts_of_tensor(t) for role, t in zip(roles, bufs)}
        if q_lens is not None:
            facts["seq_q_lens"] = prep.facts_of_tensor(q_lens)
        spec = self._fast_spec
        frame = prep.bind_dense(spec, facts, cu_stream, stream_int)
        ix = spec.index
        steps = {
            "q": self.outer_strides[0] * 2,
            "k": self.outer_strides[1] * 2,
            "v": self.outer_strides[2] * 2,
            "o": self.o_outer_stride * 2,
            "bias": 0 if self.bias_broadcast_batch else self.outer_strides[3] * 2,
            "seq_q_lens": 0,  # one region, every entry's Q rows are live
        }
        slots = []
        for role, slot, _elem in self._PTR_SLOTS:
            if role in facts and slot in ix:
                slots.append((role, ix[slot], steps.get(role)))
        self._frame = frame
        self._frame_slots = tuple(slots)
        self._frame_stream_slot = ix["stream"]

    def _execute_frames(self, q, k, v, bias, kv_lengths, mask, o, lse, q_lens, stream_int: int) -> None:
        from cudnn._device import ensure_current_context

        ensure_current_context(stream_int, self.device_index)
        cu_stream = self._compiled_plan._stream_handle(stream_int)
        if self._frame is None:
            self._bind_frame(q, k, v, bias, kv_lengths, mask, o, lse, q_lens, cu_stream, stream_int)
        base = {"q": q.data_ptr(), "k": k.data_ptr(), "v": v.data_ptr(), "o": o.data_ptr(), "bias": bias.data_ptr()}
        if mask is not None:
            base["kv_mask"] = mask.data_ptr()
        if kv_lengths is not None:
            base["seq_kv_lens"] = kv_lengths.data_ptr()
            base["seq_q_lens"] = q_lens.data_ptr()
        if lse is not None:
            base["lse"] = lse.data_ptr()
        steps = {
            "kv_mask": mask.stride(0) if mask is not None else 0,
            "seq_kv_lens": kv_lengths.stride(0) * 4 if kv_lengths is not None else 0,
            "lse": self.n * self.h * self.s_q * 4,
        }
        fn = self._fast_spec.fn
        template = self._frame
        template[self._frame_stream_slot] = cu_stream
        for b in range(self.batch):
            frame = list(template)
            for role, slot, step in self._frame_slots:
                frame[slot] = base[role] + b * (step if step is not None else steps[role])
            fn(*frame)

    def make_cuda_graph(self, q, k, v, bias, kv_lengths=None, o=None, lse=None, workspace=None, *, mask=None, warmup: int = 2) -> "torch.cuda.CUDAGraph":
        """Capture the launches for THESE buffers into one ``torch.cuda.CUDAGraph`` (replay = one graph launch per call; the buffers'
        contents may change between replays, their addresses may not).  ``warmup`` eager calls run first on a side stream.  The
        ``kv_lengths`` form's workspace fill is captured with the launches (a memset node), so the workspace is required here as in
        ``execute``; the ``prefix_mask`` form's derivation kernel is captured too (a replay follows the mask's contents).  The
        warm-up calls run the prefix-mask debug check as the environment decides; the captured call never does (a synchronisation
        inside a capture is an error)."""
        side = torch.cuda.Stream(device=q.device)
        side.wait_stream(torch.cuda.current_stream(q.device))
        with torch.cuda.stream(side):
            for _ in range(int(warmup)):
                self.execute(q, k, v, bias, kv_lengths, o, lse, None, workspace, side.cuda_stream, mask=mask)
        torch.cuda.current_stream(q.device).wait_stream(side)
        torch.cuda.synchronize(q.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            self.execute(q, k, v, bias, kv_lengths, o, lse, None, workspace, side.cuda_stream, mask=mask, check_prefix_mask=False)
        return graph


# ---------------------------------------------------------------------------
# The drop-in function (cuEquivariance's signature)
# ---------------------------------------------------------------------------
_PLAN_LOCK = threading.Lock()
_PLANS: Dict[Tuple[Any, ...], TriangleAttentionFwd] = {}
_COUNTS: Dict[str, int] = {}
_WARNED: set = set()


def copy_counters() -> Dict[str, int]:
    """Snapshot of the function's COUNTED layout fallbacks since the last reset: ``fallback_copies`` (every copy) and per kind --
    ``q_repack`` / ``k_repack`` / ``v_repack`` (an activation not in the kernel's token-major form), ``o_repack`` (the output copied
    back into the caller's layout), ``bias_repack`` (a bias whose key dim is not contiguous or whose row stride is not a 16-byte
    multiple: padded to ``S_kv`` rounded up to 8), ``bias_cast`` (an fp32 bias cast to the activations' dtype), ``mask_repack`` /
    ``kv_lengths_repack`` (a non-contiguous mask / lengths tensor), ``out_copy`` (an ``out=`` tensor the kernel could not write
    directly).  A model on its native layout should read 0 everywhere."""
    with _PLAN_LOCK:
        return dict(_COUNTS)


def reset_copy_counters() -> None:
    with _PLAN_LOCK:
        _COUNTS.clear()


def _count(kind: str, plan: Optional[TriangleAttentionFwd], what: str) -> None:
    with _PLAN_LOCK:
        _COUNTS[kind] = _COUNTS.get(kind, 0) + 1
        _COUNTS["fallback_copies"] = _COUNTS.get("fallback_copies", 0) + 1
        first = (kind, what) not in _WARNED
        _WARNED.add((kind, what))
    if plan is not None:
        plan.fallback_copies += 1
    if first:
        warnings.warn(
            f"cudnn.triangle_attention: {kind} -- {what}; the tensor is copied into the kernel's layout on every call "
            "(counted in cudnn.triangle_attention.copy_counters()). Produce it in the zero-copy layout to avoid the copy.",
            RuntimeWarning,
            stacklevel=4,
        )


def _bshd_empty(b: int, n: int, h: int, s: int, d: int, dtype, device) -> torch.Tensor:
    """A compact ``(B, N, S, H, D)`` allocation presented as ``(B, N, H, S, D)`` -- the kernels' zero-copy activation form."""
    return torch.empty((b, n, s, h, d), dtype=dtype, device=device).permute(0, 1, 3, 2, 4)


def _repack_activation(t: torch.Tensor, name: str, plan) -> torch.Tensor:
    """A COUNTED copy of an activation into the compact token-major form."""
    staged = _bshd_empty(*t.shape, t.dtype, t.device)
    staged.copy_(t)
    _count(
        f"{name}_repack",
        plan,
        f"{name} with shape {tuple(t.shape)} and strides {tuple(t.stride())} is not in the (B, N, S, H, D) memory form (head dim contiguous, token-major, 16-byte strides)",
    )
    return staged


def _repack_bias(bias: torch.Tensor, dtype: torch.dtype, plan) -> torch.Tensor:
    """ONE counted copy that casts (fp32 -> the io dtype) and / or pads the key row to a multiple of 8 elements (the TMA 16-byte
    row-stride rule) and makes the key dim contiguous."""
    b, one, h, s_q, s_kv = bias.shape
    pad = -(-s_kv // 8) * 8
    staged = torch.empty((b, one, h, s_q, pad), dtype=dtype, device=bias.device)[..., :s_kv]
    staged.copy_(bias)
    if bias.dtype != dtype:
        _count("bias_cast", plan, f"an {bias.dtype} bias is cast to the activations' dtype {dtype} (the engine loads the io dtype)")
    else:
        _count(
            "bias_repack",
            plan,
            f"a bias with shape {tuple(bias.shape)} and strides {tuple(bias.stride())} needs a contiguous key dim and 16-byte row strides (S_kv % 8 == 0 or a padded row)",
        )
    return staged


def _activation_zero_copy(t: torch.Tensor) -> bool:
    shape4, stride4 = _row_strides(tuple(t.shape), tuple(t.stride()))
    return activation_binds_zero_copy(shape4, stride4) and (t.shape[0] == 1 or (t.stride(0) * _ELEM_BYTES) % _TMA_ALIGN == 0)


def _bias_zero_copy(bias: torch.Tensor, dtype: torch.dtype) -> bool:
    shape4, stride4 = _row_strides(tuple(bias.shape), tuple(bias.stride()))
    return bias.dtype == dtype and bias_binds_zero_copy(shape4, stride4) and (bias.shape[0] == 1 or (bias.stride(0) * _ELEM_BYTES) % _TMA_ALIGN == 0)


def _cached_plan(q, k, v, bias, mask_form: str, want_lse: bool, score_max: bool, scale: float) -> TriangleAttentionFwd:
    """ONE compiled plan per declaration: everything the constructor specializes on -- the q / k / v shapes AND strides (the k / v shapes
    carry H_kv and S_kv: a KV head slice of the same buffer has the strides of the full tensor and a different shape, and a plan compiled for
    H_kv = 4 declined the H_kv = 2 call at execute when the key held the strides alone), the bias shape, strides and dtype, the activations'
    dtype, the device, the mask form, the outputs and the scale.  The mask / lengths tensors are not plan facts beyond the form (execute checks
    their contiguity per call)."""
    key = (
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k.shape),
        tuple(k.stride()),
        tuple(v.shape),
        tuple(v.stride()),
        tuple(bias.shape),
        tuple(bias.stride()),
        q.dtype,
        bias.dtype,
        str(q.device),
        mask_form,
        bool(want_lse),
        bool(score_max),
        float(scale),
    )
    with _PLAN_LOCK:
        plan = _PLANS.get(key)
    if plan is not None:
        return plan
    plan = TriangleAttentionFwd(
        q,
        k,
        v,
        bias,
        kv_lengths=(mask_form == "kv_lengths"),
        mask=(mask_form in ("mask", "prefix_mask")),
        lse=want_lse,
        score_max=score_max,
        scale=scale,
        mask_is_prefix=(mask_form == "prefix_mask"),
    )
    plan.check_support()
    plan.compile()
    with _PLAN_LOCK:
        return _PLANS.setdefault(key, plan)


class _Route:
    """The drop-in function's decisions for ONE layout signature (shapes, strides, dtypes, device, mask form, outputs, scale): which
    tensors the kernel binds as they are and which are repacked, and the plan the staged layouts compile to.  Decided once per
    signature from the engine's own layout predicates; a call on a known signature only re-checks what a layout cannot carry (the
    base alignment) and goes straight to the plan."""

    __slots__ = ("plan", "q_copy", "k_copy", "v_copy", "bias_copy", "mask_copy", "kvl_copy")

    def __init__(self, plan, q_copy, k_copy, v_copy, bias_copy, mask_copy, kvl_copy):
        self.plan, self.q_copy, self.k_copy, self.v_copy, self.bias_copy, self.mask_copy, self.kvl_copy = (
            plan,
            q_copy,
            k_copy,
            v_copy,
            bias_copy,
            mask_copy,
            kvl_copy,
        )


_ROUTES: Dict[Tuple[Any, ...], _Route] = {}


def _route_for(q, k, v, bias, mask, kv_lengths, mask_form: str, want_lse: bool, score_max: bool, scale: float) -> _Route:
    key = (
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k.shape),
        tuple(k.stride()),
        tuple(v.stride()),
        tuple(bias.shape),
        tuple(bias.stride()),
        q.dtype,
        bias.dtype,
        str(q.device),
        mask_form,
        None if mask is None else tuple(mask.stride()),
        None if kv_lengths is None else tuple(kv_lengths.stride()),
        bool(want_lse),
        bool(score_max),
        float(scale),
    )
    with _PLAN_LOCK:
        route = _ROUTES.get(key)
    if route is not None:
        return route
    q_copy, k_copy, v_copy = (not _activation_zero_copy(t) for t in (q, k, v))
    bias_copy = not _bias_zero_copy(bias, q.dtype)
    mask_copy = mask is not None and not mask.is_contiguous()
    kvl_copy = kv_lengths is not None and not kv_lengths.is_contiguous()
    # The plan is declared on the STAGED layouts: a repacked tensor's compact form, the caller's own otherwise.
    b, n, h, s_q, d = q.shape
    q_s = q if not q_copy else _bshd_empty(b, n, h, s_q, d, q.dtype, q.device)
    k_s = k if not k_copy else _bshd_empty(*k.shape, q.dtype, q.device)
    v_s = v if not v_copy else _bshd_empty(*v.shape, q.dtype, q.device)
    bias_s = (
        bias
        if not bias_copy
        else torch.empty((bias.shape[0], 1, bias.shape[2], s_q, -(-bias.shape[4] // 8) * 8), dtype=q.dtype, device=q.device)[..., : bias.shape[4]]
    )
    plan = _cached_plan(q_s, k_s, v_s, bias_s, mask_form, want_lse, score_max, scale)
    route = _Route(plan, q_copy, k_copy, v_copy, bias_copy, mask_copy, kvl_copy)
    with _PLAN_LOCK:
        return _ROUTES.setdefault(key, route)


def _run(plan: TriangleAttentionFwd, q, k, v, bias, kv_lengths, mask, o, lse, stream, check_prefix_mask: Optional[bool] = None) -> None:
    # The workspace per call (the torch allocator recycles it): the plan's own bytes and, on the kv_lengths form, the Q-side lengths the
    # graph's padding contract requires (the prefix_mask form adds its derived lengths, B * N words) -- never a cached or plan-owned
    # tensor.
    ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=q.device) if plan._workspace_bytes else None
    plan.execute(q, k, v, bias, kv_lengths, o, lse, None, ws, stream, mask=mask, check_prefix_mask=check_prefix_mask)


class _TriangleAttentionFunction(torch.autograd.Function):
    """The autograd seam: the forward launches the plan and saves what the backward will need (``q, k, v, bias, o, lse``, as
    cuEquivariance does); the backward itself is a later item of this API and raises typed until it lands, so a training caller is
    never handed a zero gradient."""

    @staticmethod
    def forward(ctx, plan, q, k, v, bias, kv_lengths, mask, stream, check_prefix_mask=None):
        o = _bshd_empty(*q.shape, q.dtype, q.device)
        lse = torch.empty((plan.batch, plan.n, plan.h, plan.s_q), dtype=torch.float32, device=q.device)
        _run(plan, q, k, v, bias, kv_lengths, mask, o, lse, stream, check_prefix_mask)
        ctx.save_for_backward(q, k, v, bias, o, lse)
        ctx.mark_non_differentiable(lse)
        return o, lse

    @staticmethod
    def backward(ctx, grad_o, grad_lse):
        raise NotImplementedError(
            "triangle_attention has no backward registered yet: dq, dk, dv and dbias are a later item of this API. Run the forward under "
            "torch.no_grad() (inference) or keep the gradient path on another provider until it lands."
        )  # the appended check_prefix_mask input takes no gradient either way


def mask_to_kv_lengths(mask: torch.Tensor) -> Optional[torch.Tensor]:
    """``(B, N, 1, 1, 1)`` int32 right-padded lengths for a PREFIX-shaped boolean mask ``(B, N, 1, 1, S_kv)`` (every row True on
    ``[0, len)`` and False after), else ``None`` -- so a caller passes ``kv_lengths=mask_to_kv_lengths(mask)`` for the zero-cost
    padding path and falls back to ``mask=mask`` (the in-kernel dense mask) when it returns None.  ONE host synchronisation (the
    prefix test is a device-wide ``all``): call it once per mask, not once per layer.

    Masked-row convention: on this provider the two forms agree (a row with ``len == 0`` reads ``o = 0``, ``lse = max = -inf`` either
    way); on cuEquivariance they differ (``kv_lengths``: the row is zeroed; dense mask: ``-1e9`` is folded, ``o`` = the uniform
    average), so converting a mask changes what cuEquivariance's fully masked rows read -- see the documentation.

    The sync-free twin for a caller who KNOWS the mask is prefix-shaped: :func:`prefix_mask_to_kv_lengths` (no test, no synchronisation,
    the lengths computed on the device), or ``mask_is_prefix=True`` on either API surface."""
    b, n, s_kv = _check_mask_arg(mask)
    m = mask.reshape(b, n, s_kv)
    lengths = m.sum(dim=-1, dtype=torch.int32)
    prefix = torch.arange(s_kv, device=mask.device, dtype=torch.int32).view(1, 1, s_kv) < lengths.unsqueeze(-1)
    if not bool(torch.equal(m, prefix)):
        return None
    return lengths.view(b, n, 1, 1, 1)


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_aux: bool = False,
    kv_lengths: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    stream=None,
    mask_is_prefix: bool = False,
    check_prefix_mask: Optional[bool] = None,
):
    """Triangle attention with cuEquivariance's signature (``cuequivariance_ops_torch.triangle_attention``, 0.12.0: the eight
    parameters in its order, ``kv_lengths`` positional-or-keyword as there) and four keyword-only additions appended: ``out``
    (write the output into this tensor instead of allocating; honoured on the inference
    path, copied into under grad mode), ``stream`` (a raw CUDA stream handle the allocations, copies and launches are ordered on;
    None = torch's current stream), ``mask_is_prefix`` (the caller ASSERTS ``mask`` is prefix-shaped -- every row True on ``[0, len)`` and
    False after -- so its row lengths are derived on the device, with no host synchronisation, and the call rides the ``kv_lengths``
    route instead of the in-kernel dense mask; a non-prefix row is then served as the prefix of its True count, the WRONG keys and no
    error) and ``check_prefix_mask`` (the debug check of that assertion: the device-wide prefix test with ONE host synchronisation, a
    typed ``ValueError`` naming the violating rows; None = :data:`CHECK_PREFIX_MASK_ENV` decides, off by default).  A fully masked row
    reads ``o = 0`` on every mask form of this provider.

    ``q, k, v`` ``(B, N, H, S, D)`` bf16 / fp16 (``k, v`` may carry fewer heads, dividing ``H``), ``bias`` ``(B|1, 1, H|1, S_q, S_kv)`` in
    the same dtype (an fp32 bias is cast once per call, counted), ``mask`` ``(B, N, 1, 1, S_kv)`` bool OR ``kv_lengths`` ``(B, N, 1, 1, 1)``
    int32 (never both), ``scale`` (None = ``1 / sqrt(D)``).  Returns ``o`` ``(B, N, H, S_q, D)`` in ``q``'s dtype; ``return_aux=True``
    asks for ``(o, lse, max)`` and is declined typed today (the row max is not stored by the engine's bias arm).

    Zero-copy layouts: activations in the ``(B, N, S, H, D)`` memory form (head dim contiguous, token-major, 16-byte strides -- a
    permuted view of a ``(B, N, S, H, D)`` buffer, compact or padded), the bias head-major with a contiguous key dim and 16-byte
    row strides (``S_kv % 8 == 0``), a contiguous mask / lengths tensor.  Anything else is repacked with a counted copy
    (:func:`copy_counters`; one ``RuntimeWarning`` per layout); the output follows ``q``: zero-copy ``q`` -> ``o`` in the same memory
    form, repacked ``q`` -> ``o`` in ``q``'s own layout (one more counted copy).  The layout decisions and the compiled plan are cached
    per (shape, dtype, strides, mask form, outputs, scale) in the process; warm a shape before capturing a CUDA graph around this call.
    """
    from cudnn._torch_stream import stream_context

    if mask is not None and kv_lengths is not None:
        raise ValueError("pass either mask or kv_lengths, not both")
    if mask_is_prefix and mask is None:
        raise ValueError("mask_is_prefix=True requires a boolean mask (it asserts the mask's shape; pass kv_lengths directly when you hold lengths)")
    for name, t in (("q", q), ("k", k), ("v", v), ("bias", bias)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.ndim != 5:
            raise ValueError(f"{name} must be rank 5 ((B, N, H, S, D) for q / k / v, (B, 1, H, S_q, S_kv) for the bias), got {tuple(t.shape)}")
    if q.dtype not in _DTYPES:
        raise NotImplementedError(f"triangle_attention serves bf16 and fp16 activations, got {q.dtype}")
    if bias.dtype not in _DTYPES + (torch.float32,) or (bias.dtype in _DTYPES and bias.dtype != q.dtype):
        raise NotImplementedError(f"the bias must be the activations' dtype ({q.dtype}) or fp32 (cast once per call), got {bias.dtype}")
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError(f"mask must be a boolean tensor (True = the key is live), got {mask.dtype}; pass an additive mask as the bias")
    if kv_lengths is not None and kv_lengths.dtype != torch.int32:
        raise ValueError(f"kv_lengths must be int32 (cuEquivariance's contract), got {kv_lengths.dtype}")
    b, n, h, s_q, d = q.shape
    mask_form = "kv_lengths" if kv_lengths is not None else (("prefix_mask" if mask_is_prefix else "mask") if mask is not None else "none")
    want_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad or bias.requires_grad)
    want_lse = bool(return_aux) or want_grad
    scale_f = float(scale) if scale is not None else 1.0 / math.sqrt(d)
    device = q.device
    with stream_context(stream, device):
        route = _route_for(q, k, v, bias, mask, kv_lengths, mask_form, want_lse, bool(return_aux), scale_f)
        plan = route.plan
        # The layout decisions are the route's; the base alignment is a per-call fact (a slice may move it) for the TMA-read operands
        # only -- the byte mask has no alignment rule and an int32 lengths tensor is word-aligned by construction, so a per-entry
        # slice of either binds as it is.
        q_s = _repack_activation(q, "q", plan) if (route.q_copy or q.data_ptr() % _TMA_ALIGN) else q
        k_s = _repack_activation(k, "k", plan) if (route.k_copy or k.data_ptr() % _TMA_ALIGN) else k
        v_s = _repack_activation(v, "v", plan) if (route.v_copy or v.data_ptr() % _TMA_ALIGN) else v
        bias_s = _repack_bias(bias, q.dtype, plan) if (route.bias_copy or bias.data_ptr() % _TMA_ALIGN) else bias
        mask_s = mask
        if route.mask_copy:
            mask_s = mask.contiguous()
            _count("mask_repack", plan, f"a mask with strides {tuple(mask.stride())} is made contiguous")
        kvl_s = kv_lengths
        if route.kvl_copy:
            kvl_s = kv_lengths.contiguous()
            _count("kv_lengths_repack", plan, f"a kv_lengths tensor with strides {tuple(kv_lengths.stride())} is made contiguous")
        q_copied = q_s is not q
        out_direct = out is not None and not q_copied and not want_grad and _out_binds(out, plan)
        if want_grad:
            o, lse = _TriangleAttentionFunction.apply(plan, q_s, k_s, v_s, bias_s, kvl_s, mask_s, stream, check_prefix_mask)
        else:
            o = out if out_direct else _bshd_empty(b, n, h, s_q, d, q.dtype, device)
            lse = torch.empty((b, n, h, s_q), dtype=torch.float32, device=device) if want_lse else None
            _run(plan, q_s, k_s, v_s, bias_s, kvl_s, mask_s, o, lse, stream, check_prefix_mask)
        if out is not None and not out_direct:
            if tuple(out.shape) != tuple(q.shape) or out.dtype != q.dtype or out.device != device:
                raise ValueError(f"out must be a {tuple(q.shape)} {q.dtype} tensor on {device}, got {tuple(out.shape)} {out.dtype} on {out.device}")
            out.copy_(o)
            _count("out_copy", plan, "out= is not in the kernels' (B, N, S, H, D) memory form (or the call runs under grad mode); the result is copied into it")
            o = out
        elif q_copied:
            # The output follows q: a caller whose q was head-major gets o in that same layout (one counted copy), never a
            # strided surprise.  A zero-copy q already produced o in the kernels' form, which IS the caller's form.
            o_caller = torch.empty_like(q)
            o_caller.copy_(o)
            _count("o_repack", plan, "q was repacked, so the output is copied back into q's own layout")
            o = o_caller
    if return_aux:  # unreachable today: a declaration with score_max=True is declined by check_support with the engine's reason
        return o, lse, None
    return o


def _out_binds(out: torch.Tensor, plan: TriangleAttentionFwd) -> bool:
    shape4, stride4 = _row_strides(tuple(out.shape), tuple(out.stride()))
    return (
        tuple(out.shape) == (plan.batch, plan.n, plan.h, plan.s_q, plan.d)
        and out.dtype == plan.dtype
        and (shape4, stride4) == plan.o4
        and (out.shape[0] == 1 or out.stride(0) == plan.o_outer_stride)
        and out.data_ptr() % _TMA_ALIGN == 0
    )


__all__ = [
    "CHECK_PREFIX_MASK_ENV",
    "CUEQ_SIGNATURE_VERSION",
    "ENGINE_ROW",
    "MASK_FORMS",
    "TriangleAttentionFwd",
    "activation_binds_zero_copy",
    "assert_prefix_mask",
    "bias_binds_zero_copy",
    "copy_counters",
    "mask_to_kv_lengths",
    "prefix_mask_to_kv_lengths",
    "prefix_mask_workspace_bytes",
    "reset_copy_counters",
    "triangle_attention",
]
