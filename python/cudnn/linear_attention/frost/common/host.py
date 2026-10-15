# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers shared by the FROST LA kernel modules (engine-invoked).  Importable without the CuTe DSL: the engines
and the semantic ops call :func:`family_slots` before the DSL version gate runs, so ``cutlass`` is imported inside
:func:`get_dtype` only."""

TENSOR_MAP_QWORDS = 128 // 8


def get_dtype(dtype):
    """dtype string -> cutlass DSL type (bf16/fp16 io, fp32/bf16/fp16 states)."""
    import cutlass

    name = str(dtype)
    if "bfloat16" in name:
        return cutlass.BFloat16
    if "float16" in name or "half" in name:
        return cutlass.Float16
    if "float32" in name:
        return cutlass.Float32
    raise ValueError(f"Unsupported dtype {dtype}, expected bfloat16, float16, or float32")


def tensormap_workspace_bytes(mod, B: int, expand_num: int = 1) -> int:
    """Runtime TMA-descriptor block for a kernel module, its per-batch arrays plus 128 alignment slack.  A module whose
    sub-token operands (k / v) load through the (token, slot) descriptor family declares ``TENSORMAP_FAMILY_OPERANDS``;
    each of those takes ``expand_num + 1`` arrays instead of one when ``expand_num`` does not divide the 64-row chunk."""
    arrays = mod.TENSORMAP_DESC_ARRAYS + getattr(mod, "TENSORMAP_FAMILY_OPERANDS", 0) * (0 if 64 % expand_num == 0 else expand_num)
    return TENSOR_MAP_QWORDS * 8 * arrays * B + 128


def family_slots(t, expand_num: int, expanded_rank: int) -> int:
    """Slots per token of a sub-token operand's load family: 1 when its expanded rows have one uniform stride (the
    expanded ``[rows, ...]`` form of rank ``expanded_rank``, or a token-major view whose token stride is ``expand_num``
    slot strides), so every chunk is one ``b_t``-row box; else ``expand_num`` (token-major with its own token stride,
    the box family)."""
    if expand_num == 1 or len(t.shape) == expanded_rank:
        return 1
    strides = tuple(int(stride) for stride in t.stride())
    if int(t.shape[0]) == 1 or strides[0] == expand_num * strides[1]:
        return 1
    return expand_num
