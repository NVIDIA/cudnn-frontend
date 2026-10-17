# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level CuTe/DLPack helpers shared by the sm_100 norm kernels.

Mirrors the role of ``cudnn.sdpa.utils``: DLPack -> cute.Tensor conversion and a
couple of small launch-side helpers. The heavy device-side primitives (cp.async
staging, block reductions) live in each pass's ``kernels/_common_sm100.py`` and
build on ``cudnn.frost.tile_dsl`` + ``cutlass.primitives``.
"""

from __future__ import annotations

from cutlass.cute.runtime import from_dlpack


def dyn(t):
    """Wrap a torch tensor as a dynamic-layout cute tensor (16-byte aligned).

    Detaches first when the tensor is on an autograd tape: ``__dlpack__`` refuses
    to export one that requires grad (``BufferError: Can't export tensors that
    require gradient``), and a caller handing a graph a parameter straight out of
    a module hits exactly that. ``detach()`` shares storage, so an output written
    through the detached alias still lands in the caller's buffer, and these
    kernels compute their gradients explicitly -- there is no tape to preserve.
    """
    if t.requires_grad:
        t = t.detach()
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic()


def sm_count():
    """Multiprocessor count of the current device (cached)."""
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


_SM_COUNT = None
_COOP_OCC = {}
COOP_OCC_MAX = 4


def run_coop(key, run):
    """Run ``run(occ)`` at the largest CTAs-per-SM the cooperative launch accepts.

    A cooperative grid must be fully co-resident, so its size is capped by the
    kernel's real occupancy -- which depends on its register count and therefore
    cannot be computed host-side. Sizing the grid to ONE CTA per SM (the obvious
    safe choice) leaves the machine at a fraction of its threads and costs ~20% on
    the NHWC norms; guessing higher trips ``CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE``.

    So try the biggest grid first and halve until it is accepted, remembering the
    answer per kernel shape. The oversize error comes from launch-time validation
    rather than from the device, so a rejected attempt leaves the context usable.
    """
    occ = _COOP_OCC.get(key)
    if occ is not None:
        return run(occ)
    occ = COOP_OCC_MAX
    while True:
        try:
            out = run(occ)
            _COOP_OCC[key] = occ
            return out
        except Exception as e:  # noqa: BLE001 - the DSL wraps the CUDA error type
            if occ <= 1 or "COOPERATIVE" not in str(e).upper():
                raise
            occ //= 2


_SMEM_CAPACITY = None


def smem_capacity():
    """Per-block shared memory budget for the CURRENT architecture, via the CuTe DSL.

    ``cutlass.memory.get_smem_capacity_in_bytes()`` resolves the running arch through
    ``CuTeDSL.get_arch_enum()`` and returns the per-BLOCK opt-in limit, which is what a
    launch is actually validated against. Two reasons not to hardcode it:

    * it is architecture-dependent -- 232448 on sm_90/sm_100, but 101376 on sm_120, so
      a Blackwell constant does not merely mis-tune elsewhere, it over-requests and the
      launch fails;
    * it is NOT ``shared_memory_per_multiprocessor`` (233472 on sm_100). Sizing against
      the per-SM figure leaves a 1 KB window where the driver rejects the launch
      instead of us rejecting it first.

    Queried once and cached; the call itself is ~4 us.
    """
    global _SMEM_CAPACITY
    if _SMEM_CAPACITY is None:
        from cutlass.memory import get_smem_capacity_in_bytes

        _SMEM_CAPACITY = int(get_smem_capacity_in_bytes())
    return _SMEM_CAPACITY


def smem_budget(preferred_bytes):
    """A measured-on-sm_100 smem budget, clamped to what this architecture has.

    Deliberately a clamp rather than a rescale: where the tuned value fits it is used
    unchanged, so sm_100 behaviour is bit-identical, and on a smaller-smem part the
    kernel degrades to something launchable instead of being silently retuned by a
    ratio nobody measured.
    """
    return min(int(preferred_bytes), smem_capacity())


_SMEM_PER_SM = None


def smem_per_sm():
    """Total shared memory per SM -- for PINNING occupancy, not for sizing one block.

    Distinct from :func:`smem_capacity`: the BatchNorm kernels deliberately request
    more than ``smem_per_sm() // (occ + 1)`` so that only ``occ`` CTAs can co-reside,
    which is what makes the per-CTA TMEM budget enforceable. The CuTe DSL exposes only
    the per-BLOCK limit (232448 on sm_100) and this is the per-SM total (233472), so
    it comes from the device properties -- still queried, not hardcoded.
    """
    global _SMEM_PER_SM
    if _SMEM_PER_SM is None:
        import torch

        prop = torch.cuda.get_device_properties(0)
        _SMEM_PER_SM = int(getattr(prop, "shared_memory_per_multiprocessor", 0)) or smem_capacity()
    return _SMEM_PER_SM


_EXT_STREAMS: dict = {}


def on_stream(raw):
    """Context manager putting torch's current stream on the graph's execution stream.

    The engines reach the GPU two ways -- CuTe kernel launches and torch ``copy_``
    into the graph's output buffers -- and BOTH follow torch's current stream.
    ``ExecutionContext.stream`` is the stream cuDNN was asked to run on
    (``cudnn.set_stream(handle, ...)``), which is not necessarily torch's current
    one, so without this the work lands on the wrong stream: a producer ordered
    ahead of the norm on the requested stream was not waited for, and the norm
    read the pre-producer input.

    ``None`` means the context carried no stream, and torch's current one is
    already the right answer. Handle 0 does NOT mean that: it is an explicit
    request for the legacy default stream, and treating it as "unspecified"
    leaves whatever side stream happens to be current in force -- a producer
    queued on the requested default stream then goes unwaited for and the norm
    reads the stale input. The two cases look identical and behave oppositely,
    so they are kept apart here.

    ``ExternalStream`` objects are cached because building one per execute is
    pure overhead on the hot path.
    """
    import torch

    if raw is None:
        return torch.cuda.stream(None)  # no stream requested: leave the current one alone
    handle = int(raw)
    if handle == 0:
        return torch.cuda.stream(torch.cuda.default_stream())  # explicitly the default stream
    s = _EXT_STREAMS.get(handle)
    if s is None:
        s = _EXT_STREAMS[handle] = torch.cuda.ExternalStream(handle)
    return torch.cuda.stream(s)


def fake_stream():
    """The compile-time placeholder for a host function's trailing stream argument.

    CuTe host functions do not pick up torch's current stream: a kernel launched
    without an explicit ``stream=`` goes to the default stream whatever torch is
    doing. So every norm host function takes a trailing ``stream`` and threads it
    into each ``.launch``; this is what ``cute.compile`` is handed in its place.
    """
    from cutlass.cute.runtime import make_fake_stream

    return make_fake_stream(use_tvm_ffi_env_stream=False)


def cur_stream():
    """Torch's current stream as a ``CUstream``, for the matching runtime argument.

    The engines make the graph's execution stream current around the whole call
    (see :func:`on_stream`), so reading it here is what carries
    ``ExecutionContext.stream`` down to the launches. Cached per handle: building
    a CUstream per launch is measurable on small norms.
    """
    import torch

    h = torch.cuda.current_stream().cuda_stream
    s = _CU_STREAMS.get(h)
    if s is None:
        from cuda.bindings import driver as _drv

        s = _CU_STREAMS[h] = _drv.CUstream(h)
    return s


_CU_STREAMS: dict = {}


_WS_ALIGN = 256  # keeps every carve-out 16B-vectorisable and dtype-reinterpretable


class Workspace:
    """Bump allocator over a flat uint8 buffer the caller already owns.

    The graph engines must not allocate inside execute (AGENTS.md Rule 1) and
    must not own device memory (Rule 8), but the cooperative kernels genuinely
    need scratch -- cross-CTA partials, retired-CTA counters, per-CTA dgamma
    partials. Those come from here instead, carved from the workspace the graph
    asked for via ``get_workspace_size()``.

    A ``Workspace`` wrapping ``None`` (or one that runs out of room) hands back
    ``None`` from :meth:`take`, and the ``ws_*`` helpers then fall back to a
    fresh tensor. That keeps the direct torch API -- which has no workspace and
    no Rule 1 to honour -- working unchanged, and makes an under-sized
    workspace a performance problem rather than a crash.
    """

    __slots__ = ("_buf", "_off", "peak")

    def __init__(self, buf=None):
        self._buf = buf if buf is not None and buf.numel() else None
        self._off = 0
        self.peak = 0  # high-water mark, so a test can check the reported size covers it

    def reset(self):
        """Rewind to the start; call once per execute, not per kernel."""
        self._off = 0
        return self

    def take(self, nelem, dtype):
        """``nelem`` elements of ``dtype``, or None when the buffer cannot serve it."""
        import torch

        if self._buf is None:
            return None
        esz = torch.empty(0, dtype=dtype).element_size()
        off = (self._off + _WS_ALIGN - 1) & ~(_WS_ALIGN - 1)
        end = off + nelem * esz
        if end > self._buf.numel():
            return None
        self._off = end
        if end > self.peak:
            self.peak = end
        return self._buf[off:end].view(dtype)


def ws_empty(ws, nelem, dtype, device):
    """Uninitialised scratch from ``ws``, or a fresh tensor when it cannot serve."""
    import torch

    t = ws.take(nelem, dtype) if ws is not None else None
    return torch.empty(nelem, dtype=dtype, device=device) if t is None else t


def ws_zeros(ws, nelem, dtype, device):
    """Zeroed scratch from ``ws``, or a fresh tensor when it cannot serve.

    Workspace memory is recycled across executes, so the zeroing is not
    optional here the way it is for a freshly allocated tensor.
    """
    import torch

    t = ws.take(nelem, dtype) if ws is not None else None
    if t is None:
        return torch.zeros(nelem, dtype=dtype, device=device)
    return t.zero_()


def ws_bytes(*parts):
    """Total workspace for ``(nelem, element_size)`` parts, with the alignment padding."""
    total = 0
    for nelem, esz in parts:
        total = ((total + _WS_ALIGN - 1) & ~(_WS_ALIGN - 1)) + nelem * esz
    return total


def ws_reset(ws):
    """Rewind a workspace before a launch attempt that may be retried.

    ``run_coop`` re-runs its closure at a lower occupancy when a cooperative
    grid is rejected, and the occupancy-stepping BatchNorm loops do the same.
    Without this the bump allocator would keep advancing across attempts and
    the retry would fall off the end of the buffer -- correct, because the
    helpers fall back to torch, but it would allocate exactly where the
    workspace exists to stop that.
    """
    if ws is not None:
        ws.reset()
