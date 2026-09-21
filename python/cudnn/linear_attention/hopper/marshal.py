# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operand resolution for the sm90 KDA engines: one pass, layout and dtype.

The kernels index compact row-major operands from a raw address and take an
int32 chunk table with an fp32 gate, beta and state. Callers do not always
oblige: an ordinary fused-projection slice is padded, and FlashInfer passes
int64 ``cu_seqlens``, a bf16 state pool and the gate at q's dtype.

Both mismatches are fixed the same way -- one ``copy_`` into a correctly shaped,
correctly typed buffer, which converts the dtype AND compacts the layout at
once. They are resolved TOGETHER, in a single pass per port, because doing them
in two passes is how the address of a repacked operand got overwritten by the
original padded one.

Three rules this file exists to keep:

* one decision per port. ``resolve_inputs`` returns the address to use, and
  nothing downstream re-derives it.
* every copy is issued ON THE EXECUTION STREAM. A torch op runs on torch's
  ambient stream, which has no dependency on the stream the kernel launches on.
  A default-stream handle (0, 1, 2) maps to torch's own default stream rather
  than an ``ExternalStream``: see :func:`stream_ctx`.
* scratch is PER EXECUTION, never cached on the plan. Two threads may execute
  one compiled graph concurrently with different operands; a buffer owned by
  the plan lets one call's conversion land in the other's launch. Nothing is
  allocated at all on the common path, where every operand is already packed
  and already the right dtype.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cudnn.frost import buffers

# Ports the kernels READ. Anything else is written in place, so a padded or
# mistyped one cannot be fixed without copying back.
_INPUT_PORTS = frozenset(
    {
        "q",
        "k",
        "v",
        "g",
        "beta",
        "cu_seqlens",
        "initial_state",
        "a_log",
        "dt_bias",
        "dO",
        "d_final_state",
    }
)


# torch dtype -> the name OperandBuffer.dtype reports. Comparing names keeps the
# no-conversion path free: reading the property is a C++ attribute access, while
# torch.from_dlpack costs ~1.6 us per operand.
# Ports are requested by dtype NAME, not a torch dtype: the engine module must
# not import torch at class-definition time (the package keeps `import cudnn`
# free of it), and OperandBuffer already reports its dtype as a name.
def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "int32": torch.int32}[name]


# cudaStream_t 0, cudaStreamLegacy and cudaStreamPerThread. torch's default
# stream is the legacy one and every blocking stream orders against it.
_DEFAULT_STREAM_HANDLES = frozenset({0, 1, 2})


def stream_ctx(stream: int, device=None):
    """``torch.cuda.stream`` context for the execution stream ``stream``.

    A default-stream handle is never wrapped in ``torch.cuda.ExternalStream``:
    on torch <= 2.12 and some 2.13 nightlies ``ExternalStream(0)`` is a fresh
    NON-BLOCKING pool stream, so a copy issued inside it does not order against
    the kernel launched on ``CUstream(0)``. Under GPU contention the kernel then
    read conversion buffers before the copy landed, and ``write_back`` copied a
    staged output before the kernel wrote it.
    """
    import torch

    handle = int(stream)
    default = torch.cuda.default_stream(device)
    if handle in _DEFAULT_STREAM_HANDLES or handle == default.cuda_stream:
        return torch.cuda.stream(default)
    current = torch.cuda.current_stream(device)
    if handle == current.cuda_stream:
        return torch.cuda.stream(current)
    return torch.cuda.stream(torch.cuda.ExternalStream(handle, device=device))


def packed(dim, stride) -> bool:
    """Compact row-major, tolerating size-1 axes with any stride."""
    if not dim or not stride:
        # Nothing declared to contradict: the IR packs an unstrided tensor.
        return True
    return buffers.is_contiguous(list(dim), list(stride))


def resolve_inputs(
    names: Sequence[str],
    views: Iterable,
    engine: str,
    stream: int,
    want: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, int], List]:
    """``({port: device address}, keepalive)``, every address packed and typed.

    ``want`` maps a port to the torch dtype the kernel needs; a port absent from
    it keeps whatever the caller passed. The keepalive list holds any buffer
    this made and MUST outlive the launch -- dropping one frees memory the
    kernel is about to read.
    """
    import torch

    want = want or {}
    addr: Dict[str, int] = {}
    fixups = []
    for name, view in zip(names, views):
        is_packed = packed(view.shape, view.stride())
        target = want.get(name)
        dtype_ok = target is None or buffers.dtype_name(view) == target
        if is_packed and dtype_ok:
            addr[name] = int(view.data_ptr())
            continue
        if name not in _INPUT_PORTS:
            raise NotImplementedError(
                f"{engine}: the sm90 kernels write packed row-major outputs, but '{name}' was "
                f"passed with shape={tuple(view.shape)} stride={tuple(view.stride())} "
                f"dtype={buffers.dtype_name(view)}. Pass a contiguous buffer of the expected "
                f"dtype, or select another KDA engine with plan_name."
            )
        fixups.append((name, view, target))

    if fixups:
        # One copy per port fixes layout and dtype together, on the execution
        # stream, into a buffer this call owns.
        keepalive: List = []
        sources = [(name, torch.from_dlpack(view), target) for name, view, target in fixups]
        with stream_ctx(stream, sources[0][1].device):
            for name, source, target in sources:
                dtype = _torch_dtype(target) if target else source.dtype
                buf = torch.empty(source.shape, dtype=dtype, device=source.device)
                buf.copy_(source)
                keepalive.append(buf)
                addr[name] = int(buf.data_ptr())
        return addr, keepalive
    return addr, []


def stage_output(view, target, stream: int) -> Tuple[int, Optional[Any], Optional[Any]]:
    """``(address the kernel writes, staging buffer, caller tensor)``.

    When the caller's buffer already has the wanted dtype the kernel writes it
    directly and both extra values are ``None``; otherwise it writes the staging
    buffer and the caller must :func:`write_back` after the launch. The staging
    buffer is per execution, for the same reason inputs are.
    """
    import torch

    if buffers.dtype_name(view) == target:
        return int(view.data_ptr()), None, None
    destination = torch.from_dlpack(view)
    with stream_ctx(stream, destination.device):
        buf = torch.empty(destination.shape, dtype=_torch_dtype(target), device=destination.device)
    return int(buf.data_ptr()), buf, destination


def write_back(staged: Optional[Any], destination: Optional[Any], stream: int) -> None:
    """Copy a staged output back into the caller's buffer, on the execution stream."""
    if staged is None or destination is None:
        return
    with stream_ctx(stream, destination.device):
        destination.copy_(staged)
