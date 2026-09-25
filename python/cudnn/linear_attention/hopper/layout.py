# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed-layout handling for the sm90 KDA engines.

Both Hopper engines index their operands as compact row-major arrays -- the
fused CUDA kernels compute ``gq[(s0 + t) * H * 128 + head * 128 + d]`` from a
raw address and never see a stride, and the CuTe DSL kernel builds its state
slabs at a fixed ``stride=(128, 1)``. A caller whose q, initial_state or dO
carries a padded outer stride therefore gets a silently wrong answer rather
than an error, and an ordinary fused-projection slice is exactly that shape.

Other linear-attention engines need none of this because they hand the operand
to the kernel through DLPack, which carries the stride with it. That is also
why the fix cannot be "declare the strides on the graph": nothing in this repo
does, and the operand's ACTUAL geometry at execute is what every engine reads.

So there are two guards, for the two places a layout arrives:

* :func:`declared_layout_reason` reads what the GRAPH declares, in
  ``check_support``. That is the routing gate: a graph declaring a padded
  operand is declined, so another KDA engine serves it. The op layer declares
  no strides (the IR packs them), so this fires only for a graph built on the
  public API -- which is how FlashInfer drives cuDNN.
* :func:`packed_addresses` reads what the CALLER actually passed.
  ``_normalize`` takes each operand's geometry from the producer's own DLPack
  vtable and never checks it against the declaration, so declaring packed and
  passing padded is reachable and cannot be declined -- the plan is already
  chosen. An INPUT is repacked into a contiguous copy, which keeps the strided
  q/k/v slices the op layer has always accepted working on this engine; an
  OUTPUT cannot be fixed that way (the kernel writes it in place and a copy
  back would be a second silent behaviour), so it raises.

Every copy this module makes is issued ON THE EXECUTION STREAM. A torch op runs
on torch's ambient stream, which has no dependency on the stream the kernel is
launched on, so a repack done outside that context is a race of exactly the kind
this module exists to prevent.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from cudnn.frost import buffers

# Ports the kernels READ. Anything else is written in place, so a padded one
# cannot be repacked without copying back.
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


def _packed(dim, stride) -> bool:
    """Compact row-major, tolerating size-1 axes with any stride."""
    if not dim or not stride:
        # Nothing declared to contradict: the IR packs an unstrided tensor.
        return True
    return buffers.is_contiguous(list(dim), list(stride))


def declared_layout_reason(node, engine: str, inputs_too: bool = True) -> Optional[str]:
    """Why ``engine`` cannot serve ``node``'s declared layouts, or ``None``.

    ``inputs_too=False`` checks only the outputs, for an engine that repacks a
    padded input at execute. Declining on a padded INPUT would then be a
    capability loss rather than a safety gate -- FlashInfer declares real strides
    and passes fused-projection slices, and on sm90 there may be no other engine
    left to fall through to. A padded OUTPUT is always declined: it is written in
    place and cannot be repacked.
    """
    ports = list(node.outputs.items())
    if inputs_too:
        ports = list(node.inputs.items()) + ports
    for label, tensor in ports:
        if tensor is None:
            continue
        dim = list(getattr(tensor, "dim", None) or [])
        stride = list(getattr(tensor, "stride", None) or [])
        if not _packed(dim, stride):
            return f"{engine}: the sm90 kernels index packed row-major operands, but " f"'{label}' is declared dim={dim} stride={stride}"
    return None


def packed_addresses(names: Sequence[str], views: Iterable, engine: str, stream: int) -> Tuple[dict, List]:
    """``({port: device address}, keepalive)`` with every address packed.

    Any repack is issued on ``stream``, the stream the kernel will run on. The
    keepalive list holds the copies and MUST outlive the launch -- dropping one
    frees the buffer the kernel is about to read.
    """
    import torch

    addr = {}
    keepalive: List = []
    padded = []
    for name, view in zip(names, views):
        if _packed(view.shape, view.stride()):
            addr[name] = int(view.data_ptr())
            continue
        if name not in _INPUT_PORTS:
            raise NotImplementedError(
                f"{engine}: the sm90 kernels write packed row-major outputs, but '{name}' was "
                f"passed with shape={tuple(view.shape)} stride={tuple(view.stride())}. "
                f"Pass a contiguous output buffer, or select another KDA engine with plan_name."
            )
        padded.append((name, view))

    if padded:
        from cudnn.linear_attention.hopper.marshal import stream_ctx

        sources = [(name, torch.from_dlpack(view)) for name, view in padded]
        with stream_ctx(stream, sources[0][1].device):
            for name, source in sources:
                packed = source.contiguous()
                keepalive.append(packed)
                addr[name] = int(packed.data_ptr())
    return addr, keepalive
