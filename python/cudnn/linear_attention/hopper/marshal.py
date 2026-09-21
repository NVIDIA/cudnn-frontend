# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operand resolution for the sm90 KDA engines: one pass, layout and dtype.

The kernels index compact row-major operands from a raw address and read the
chunk table as ``const int*`` and the gate, beta and state as ``const float*``
(no null checks); ``final_state`` is written as ``float*``. Production callers
do not oblige, and cannot be asked to: FlashInfer's public ``kda()`` contract
mandates bf16 ``g``, bf16 ``beta``, a bf16 ``initial_state`` and int32-or-int64
``cu_seqlens`` -- int64 REQUIRED under graph capture -- and vLLM's Kimi path
passes ``raw_g`` / ``raw_beta`` at the hidden dtype (bf16) with an fp32
recurrent state by default. Declining those dtypes (recipe R5) would remove
``kda_hopper_cuda`` from every production caller on sm90, and the fallback
``kda_hopper`` declines bf16 ``g`` and int64 ``cu`` as well. Reading them
natively is a kernel change (``kda_fused_sm90.cu`` / ``kda_bwd_sm90.cu``) and
is deferred.

So this module is an INTERIM Rule 2 exception, recorded as such: the conversion
copies stay, but their scratch is CARVED from the caller's workspace and sized
at BUILD from the graph's declared dtypes and strides (:func:`staging_ports`).
Nothing is allocated, and nothing is discovered from the runtime view: a port
the graph declared packed and of a native dtype has no staging, so a mismatched
runtime buffer raises ``NotImplementedError`` naming it instead of being
silently repacked (Rule 5: never adapt silently). A DECLARED padded input is
still repacked, through its staging.

Two rules this file keeps:

* one decision per port. :func:`resolve_inputs` returns the address to use,
  and nothing downstream re-derives it. Layout and dtype are resolved TOGETHER,
  in a single pass per port, because resolving them separately is how the
  address of a repacked operand got overwritten by the original padded one.
* every copy is issued ON THE EXECUTION STREAM (recipe R1, :func:`stream_ctx`).
  A default-stream handle (0, 1, 2) maps to torch's own default stream rather
  than an ``ExternalStream``.

Concurrency contract: two concurrent executes of ONE graph must pass distinct
workspaces. The staging lives in the workspace, so a shared one would let one
call's conversion land in the other's launch.
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


def stream_ctx(stream: int, device=None):
    """``torch.cuda.stream`` context for the execution stream ``stream`` (Rule 5:
    default-stream sentinels map to torch's default stream, see ``cudnn._torch_stream``)."""
    from cudnn._torch_stream import stream_context

    return stream_context(stream, device)


def packed(dim, stride) -> bool:
    """Compact row-major, tolerating size-1 axes with any stride."""
    if not dim or not stride:
        # Nothing declared to contradict: the IR packs an unstrided tensor.
        return True
    return buffers.is_contiguous(list(dim), list(stride))


def _declared_dtype(tensor) -> Optional[str]:
    from ..graph_analyzer import BUFFER_NAME_FROM_CUDNN

    return BUFFER_NAME_FROM_CUDNN.get(tensor.get_data_type())


def staging_ports(node, want_in: Mapping[str, str], want_out: Mapping[str, str] = ()) -> List[Tuple[str, str, Tuple[int, ...]]]:
    """Build-time staging: ``[(port, dtype, shape)]`` for every declared INPUT
    whose dtype or strides the kernel cannot read natively, and every declared
    OUTPUT whose dtype it cannot write natively. Decided from the declaration
    alone; the shape is the declared dim, the dtype what the kernel takes.

    An output the graph leaves untyped follows the declared ``initial_state``
    when there is one -- the op layer allocates it so, and ``kda_bwd`` declares
    ``d_initial_state`` dtype-like it -- else the kernel's own dtype.
    """
    state0 = node.inputs.get("initial_state")
    state_dtype = _declared_dtype(state0) if state0 is not None else None
    ports = []
    for port, tensor in node.inputs.items():
        if tensor is None:
            continue
        target = want_in.get(port)
        have = _declared_dtype(tensor)
        dtype = target or have
        if dtype is None:
            continue
        shape = tuple(int(d) for d in tensor.dim)
        if (target is not None and have is not None and have != target) or not packed(shape, tensor.stride):
            ports.append((port, dtype, shape))
    for port, target in dict(want_out).items():
        tensor = node.outputs.get(port)
        if tensor is None:
            continue
        if (_declared_dtype(tensor) or state_dtype or target) != target:
            ports.append((port, target, tuple(int(d) for d in tensor.dim)))
    return ports


def resolve_inputs(
    names: Sequence[str],
    views: Iterable,
    engine: str,
    stream: int,
    want: Optional[Mapping[str, Any]] = None,
    staging: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    """``{port: device address}``, every address packed and typed.

    ``want`` maps a port to the dtype NAME the kernel needs; a port absent from
    it keeps whatever the caller passed. ``staging`` maps a port to its
    workspace carve (any DLPack producer of the wanted dtype and declared
    shape); a port that needs converting and has none raises.
    """
    want = want or {}
    staging = staging or {}
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
        dst = staging.get(name)
        if dst is None:
            raise NotImplementedError(
                f"{engine}: '{name}' was passed with shape={tuple(view.shape)} stride={tuple(view.stride())} "
                f"dtype={buffers.dtype_name(view)}, but the graph declared it packed"
                f"{'' if target is None else f' and {target}'}, so the plan reserved no staging for it. "
                f"Declare the layout/dtype on the graph tensor so the plan can size its staging, "
                f"or pass a buffer matching the declaration."
            )
        fixups.append((name, view, dst))

    if fixups:
        # One copy per port fixes layout and dtype together, on the execution
        # stream, into the carve declared at build.
        import torch

        targets = [(name, torch.from_dlpack(dst), view) for name, view, dst in fixups]
        with stream_ctx(stream, targets[0][1].device):
            for name, dst, view in targets:
                dst.copy_(torch.from_dlpack(view))
                addr[name] = int(dst.data_ptr())
    return addr


def stage_output(view, target: str, stream: int, staging=None, port: str = "") -> Tuple[int, Optional[Any], Optional[Any]]:
    """``(address the kernel writes, staging tensor, caller tensor)``.

    When the caller's buffer already has the wanted dtype the kernel writes it
    directly and both extra values are ``None``; otherwise it writes the
    ``staging`` carve and the caller must :func:`write_back` on ``stream`` after
    the launch. Nothing is written here, so no stream work is issued.
    """
    if buffers.dtype_name(view) == target:
        return int(view.data_ptr()), None, None
    if staging is None:
        raise NotImplementedError(
            f"output '{port}' was passed as {buffers.dtype_name(view)} but the graph declared no dtype the "
            f"kernel cannot write natively, so the plan reserved no {target} staging for it. Declare the dtype "
            f"on the graph tensor so the plan can size its staging, or pass a {target} buffer."
        )
    import torch

    staged = torch.from_dlpack(staging)
    return int(staged.data_ptr()), staged, torch.from_dlpack(view)


def write_back(staged: Optional[Any], destination: Optional[Any], stream: int) -> None:
    """Copy a staged output back into the caller's buffer, on the execution stream."""
    if staged is None or destination is None:
        return
    with stream_ctx(stream, destination.device):
        destination.copy_(staged)
