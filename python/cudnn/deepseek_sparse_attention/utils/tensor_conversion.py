"""Common CuTe tensor conversion helpers."""

from cutlass.cute.runtime import from_dlpack


def to_cute_tensor(
    t,
    assumed_align: int = 16,
    leading_dim: int = -1,
    fully_dynamic: bool = False,
    enable_tvm_ffi: bool = True,
    divisibility=None,
):
    """Convert a torch tensor to a CuTe tensor for TVM FFI."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    if divisibility is not None:
        tensor = tensor.mark_compact_shape_dynamic(mode=leading_dim, stride_order=t.dim_order(), divisibility=divisibility)
    return tensor
