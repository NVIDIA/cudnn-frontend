# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operand dtype marshalling for the sm90 KDA engines.

The fused kernels take an ``int32`` chunk table and an ``fp32`` recurrent state,
and that is baked into device code this package vendors rather than authors. The
contract serving stacks actually use is different on both counts: FlashInfer
passes ``cu_seqlens`` as int64 or int32 and documents the state as fp32 **or
bf16**, and vLLM's Kimi path requires a bf16 state.

Declining those was the honest thing to do while nothing could be done about it,
but it sent the entire call to a slower engine over two small buffers --
``cu_seqlens`` is ``N + 1`` elements and the state is ``N * H * 128 * 128``,
both negligible beside q/k/v over thousands of tokens. So they are converted
here instead.

Two rules the conversions follow:

* every copy is issued **on the execution stream**. A torch op runs on torch's
  ambient stream, which has no dependency on the stream the kernel is launched
  on; a conversion done outside that context is a race.
* buffers are cached per plan and reused, so a steady-state call allocates
  nothing and stays capturable into a CUDA graph.

An output staged through a conversion buffer needs the copy back issued after
the launch, which is why :func:`stage_output` returns the pair rather than an
address alone.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cudnn.frost import buffers

# torch dtype -> the name OperandBuffer.dtype reports. Comparing names keeps the
# no-conversion path free: reading the property is a C++ attribute access, while
# torch.from_dlpack costs ~1.6 us per operand and would otherwise be paid on
# every call for every marshalled port whether or not anything needed doing.
_NAME = {"torch.float32": "float32", "torch.bfloat16": "bfloat16", "torch.int32": "int32"}


def _stream_ctx(stream: int):
    import torch

    return torch.cuda.stream(torch.cuda.ExternalStream(int(stream)))


def _scratch(cache: Dict[str, Any], key: str, shape, dtype, device):
    import torch

    buf = cache.get(key)
    if buf is None or tuple(buf.shape) != tuple(shape) or buf.dtype != dtype:
        buf = torch.empty(tuple(shape), dtype=dtype, device=device)
        cache[key] = buf
    return buf


def as_input(view, want, cache: Dict[str, Any], key: str, stream: int) -> int:
    """Device address of ``view`` as ``want``, converting on ``stream`` if needed.

    Returns the operand's own address when it already has the wanted dtype, so
    the common case costs one string comparison and no DLPack crossing.
    """
    import torch

    if buffers.dtype_name(view) == _NAME[str(want)]:
        return int(view.data_ptr())
    tensor = torch.from_dlpack(view)
    with _stream_ctx(stream):
        buf = _scratch(cache, key, tensor.shape, want, tensor.device)
        buf.copy_(tensor)  # dtype conversion, ordered on the execution stream
    return int(buf.data_ptr())


def stage_output(view, want, cache: Dict[str, Any], key: str, stream: int) -> Tuple[int, Optional[Any], Optional[Any]]:
    """``(address the kernel writes, staging buffer, caller tensor)``.

    When the caller's buffer already has the wanted dtype the kernel writes it
    directly and both extra values are ``None``. Otherwise the kernel writes the
    staging buffer and the caller must call :func:`write_back` after the launch.
    """
    import torch

    if buffers.dtype_name(view) == _NAME[str(want)]:
        return int(view.data_ptr()), None, None
    tensor = torch.from_dlpack(view)
    buf = _scratch(cache, key, tensor.shape, want, tensor.device)
    return int(buf.data_ptr()), buf, tensor


def write_back(staged: Optional[Any], destination: Optional[Any], stream: int) -> None:
    """Copy a staged output back into the caller's buffer, on the execution stream."""
    if staged is None or destination is None:
        return
    with _stream_ctx(stream):
        destination.copy_(staged)
