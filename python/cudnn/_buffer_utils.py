# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Buffer overlap checks without a tensor-framework or DSL import."""


def tensor_byte_span(t) -> tuple[int, int]:
    """Return the tensor byte span from shape, strides, and address metadata."""
    base = int(t.data_ptr())
    if t.numel() == 0:
        return base, base
    item = int(t.element_size())
    lo = hi = 0
    for size, stride in zip(t.shape, t.stride()):
        if size > 1:
            reach = (int(size) - 1) * int(stride)
            if reach < 0:
                lo += reach
            else:
                hi += reach
    return base + lo * item, base + (hi + 1) * item


def check_buffers_disjoint(writers, readers, *, what: str = "the kernels") -> None:
    """Reject overlapping writes before launching kernels; read-only inputs may overlap."""
    spans = [(name, tensor_byte_span(t)) for name, t in writers if t is not None]
    read = [(name, tensor_byte_span(t)) for name, t in readers if t is not None]
    for i, (w_name, (w_lo, w_hi)) in enumerate(spans):
        for r_name, (r_lo, r_hi) in read + spans[i + 1 :]:
            if w_lo < r_hi and r_lo < w_hi:
                raise ValueError(
                    f"{w_name} must not alias {r_name}: their byte ranges intersect ({w_name} [{w_lo:#x}, {w_hi:#x}), {r_name} [{r_lo:#x}, {r_hi:#x})) "
                    f"and {what} write {w_name} while {r_name} is still read or written; pass buffers that do not overlap"
                )
