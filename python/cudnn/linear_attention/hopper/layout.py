# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared-layout gate for the sm90 KDA engines.

Both Hopper engines index their operands as compact row-major arrays -- the
fused CUDA kernels compute ``gq[(s0 + t) * H * 128 + head * 128 + d]`` from a
raw address and never see a stride, and the CuTe DSL kernel builds its state
slabs at a fixed ``stride=(128, 1)``. A caller whose q, initial_state or dO
carries a padded outer stride would therefore get a silently wrong answer
rather than an error, and an ordinary fused-projection slice is exactly that
shape.

:func:`declared_layout_reason` reads what the GRAPH declares, in
``check_support``. That is the routing gate: a graph declaring a padded operand
this engine cannot serve is declined, so another KDA engine serves it. The op
layer declares no strides (the IR packs them), so this fires only for a graph
built on the public API -- which is how FlashInfer drives cuDNN.

What arrives at execute is checked against the declaration in ``marshal.py``:
the CUDA engine repacks a DECLARED padded input through staging it sized at
build from those declared strides, and raises ``NotImplementedError`` for a
padded input the graph declared packed (Rule 5: never adapt silently); the CuTe
DSL engine raises for any padded operand. A padded OUTPUT is always declined
here: it is written in place and cannot be repacked.
"""

from __future__ import annotations

from typing import Optional

from cudnn.frost import buffers


def _packed(dim, stride) -> bool:
    """Compact row-major, tolerating size-1 axes with any stride."""
    if not dim or not stride:
        # Nothing declared to contradict: the IR packs an unstrided tensor.
        return True
    return buffers.is_contiguous(list(dim), list(stride))


def declared_layout_reason(node, engine: str, inputs_too: bool = True) -> Optional[str]:
    """Why ``engine`` cannot serve ``node``'s declared layouts, or ``None``.

    ``inputs_too=False`` checks only the outputs, for an engine that stages a
    declared padded input through its workspace at execute. Declining on a
    padded INPUT would then be a capability loss rather than a safety gate --
    FlashInfer declares real strides and passes fused-projection slices, and on
    sm90 there may be no other engine left to fall through to. A padded OUTPUT
    is always declined: it is written in place and cannot be repacked.
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
