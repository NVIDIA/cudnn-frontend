"""
PyTorch custom operator wrapping cuDNN's Scaled Dot-Product Attention (SDPA).

Provides ``scaled_dot_product_attention`` as the public entry point, closely
matching the signature of ``torch.nn.functional.scaled_dot_product_attention``.

**Layout convention**: tensors are expected in **BHSD** layout
``(batch, num_heads, seq_len, head_dim)`` — matching both the cuDNN convention
and PyTorch's ``torch.nn.functional.scaled_dot_product_attention`` layout.

Graph caching ensures cuDNN graphs are built once per unique configuration
and reused across calls.
"""

import logging
import math
from typing import Optional, Tuple, Dict
from enum import IntEnum

import torch
import cudnn

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_cudnn_handles = {}
_fprop_cache: Dict[tuple, tuple] = {}
_bprop_cache: Dict[tuple, tuple] = {}
_sdpa_oss_layout_copy_warned = False

# Dtype mapping (module-level constant)
_TORCH_DTYPE_TO_CUDNN = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
    torch.int32: cudnn.data_type.INT32,
    torch.int64: cudnn.data_type.INT64,
}


# ---------------------------------------------------------------------------
# UID enum — explicit tensor UIDs for graph caching
# ---------------------------------------------------------------------------


class _UIDs(IntEnum):
    Q = 1
    K = 2
    V = 3
    O = 100
    STATS = 101
    DO = 200
    DQ = 201
    DK = 202
    DV = 203
    SEQ_LEN_Q = 300
    SEQ_LEN_KV = 301
    CUM_SEQ_LEN_Q = 302
    CUM_SEQ_LEN_KV = 303


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_handle(device: torch.device):
    """Return a lazily-initialised cuDNN handle with the current CUDA stream."""
    if device not in _cudnn_handles:
        _cudnn_handles[device] = cudnn.create_handle()
    stream = _get_current_stream(device)
    cudnn.set_stream(handle=_cudnn_handles[device], stream=stream)
    return _cudnn_handles[device]


def _get_current_stream(device: torch.device):
    """Return the caller's active CUDA stream for the given device."""
    return torch.cuda.current_stream(device).cuda_stream


def _device_supports_d256_oss(device: torch.device) -> bool:
    """Return True when the d=256 OSS kernels can run on this device."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor >= 100


def _torch_dtype_to_cudnn(dtype: torch.dtype):
    """Map a PyTorch dtype to a cuDNN data_type enum."""
    return _TORCH_DTYPE_TO_CUDNN[dtype]


def _diagonal_alignment_enum(val: int):
    """Convert int sentinel to cudnn.diagonal_alignment enum."""
    if val == 0:
        return cudnn.diagonal_alignment.TOP_LEFT
    return cudnn.diagonal_alignment.BOTTOM_RIGHT


def _stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
    """Return tensor dimensions ordered from fastest- to slowest-varying stride."""
    return tuple(sorted(range(tensor.ndim), key=lambda dim: tensor.stride()[dim]))


def _to_sdpa_oss_bhsd_layout(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Convert BHSD tensors to the stride pattern expected by the d=256 OSS kernels."""
    global _sdpa_oss_layout_copy_warned

    if tensor.ndim != 4:
        raise NotImplementedError("d=256 OSS path currently supports only rank-4 BHSD tensors")
    if _stride_order(tensor) == (3, 1, 2, 0):
        return tensor
    if not _sdpa_oss_layout_copy_warned:
        _logger.warning(
            "d=256 OSS path is copying tensor '%s' to normalize BHSD layout; this is a performance slow path",
            name,
        )
        _sdpa_oss_layout_copy_warned = True
    return tensor.transpose(1, 2).contiguous().transpose(1, 2)


def _packed_bhsd_stride(shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Return the packed BHSD stride used for ragged/token-packed tensors."""
    b, h, s, d = shape
    return (s * h * d, d, h * d, 1)


def _get_variant_pack_uid_order(graph, present_uids):
    if hasattr(graph, "_get_variant_pack_uids_sorted"):
        return graph._get_variant_pack_uids_sorted()
    return sorted(present_uids)


def _validate_d256_oss_args(
    is_causal: bool,
    diagonal_alignment: int,
    left_bound: int,
    right_bound: int,
    seq_len_q: Optional[torch.Tensor],
    seq_len_kv: Optional[torch.Tensor],
    cumulative_seq_len_q: Optional[torch.Tensor],
    cumulative_seq_len_kv: Optional[torch.Tensor],
) -> Tuple[int, int]:
    """Validate the subset of SDPA-op features supported by the d=256 OSS custom-op paths."""
    if diagonal_alignment != 0:
        raise NotImplementedError("d=256 OSS path supports only TOP_LEFT diagonal alignment")
    if seq_len_q is not None or seq_len_kv is not None:
        raise NotImplementedError("d=256 OSS path does not support seq_len_q/seq_len_kv")
    if cumulative_seq_len_q is not None or cumulative_seq_len_kv is not None:
        raise NotImplementedError("d=256 OSS path does not support cumulative_seq_len_q/cumulative_seq_len_kv")

    if not is_causal:
        if left_bound != -1 or right_bound != -1:
            raise NotImplementedError("d=256 OSS path supports only full-window non-causal attention")
        return (-1, -1)

    if right_bound not in (-1, 0):
        raise NotImplementedError("d=256 OSS path supports only causal right_bound=0")
    return (left_bound if left_bound >= 0 else -1, 0)


def _can_use_d256_oss_path(
    is_causal: bool,
    diagonal_alignment: int,
    left_bound: int,
    right_bound: int,
    seq_len_q: Optional[torch.Tensor],
    seq_len_kv: Optional[torch.Tensor],
    cumulative_seq_len_q: Optional[torch.Tensor],
    cumulative_seq_len_kv: Optional[torch.Tensor],
) -> bool:
    """Return True when the d=256 OSS kernels support this SDPA configuration."""
    try:
        _validate_d256_oss_args(
            is_causal,
            diagonal_alignment,
            left_bound,
            right_bound,
            seq_len_q,
            seq_len_kv,
            cumulative_seq_len_q,
            cumulative_seq_len_kv,
        )
    except NotImplementedError:
        return False
    return True


def _make_fprop_cache_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    is_causal: bool,
    diagonal_alignment: int,
    left_bound: int,
    right_bound: int,
    has_seq_len_q: bool,
    has_seq_len_kv: bool,
    has_cum_q: bool,
    has_cum_kv: bool,
):
    q_shape, q_stride = tuple(q.shape), tuple(q.stride())
    k_shape, k_stride = tuple(k.shape), tuple(k.stride())
    v_shape, v_stride = tuple(v.shape), tuple(v.stride())
    return (
        "fprop",
        q_shape,
        q_stride,
        q.dtype,
        k_shape,
        k_stride,
        k.dtype,
        v_shape,
        v_stride,
        v.dtype,
        attn_scale,
        is_causal,
        diagonal_alignment,
        left_bound,
        right_bound,
        has_seq_len_q,
        has_seq_len_kv,
        has_cum_q,
        has_cum_kv,
        q.device,
    )


def _make_bprop_cache_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    is_causal: bool,
    diagonal_alignment: int,
    left_bound: int,
    right_bound: int,
    has_seq_len_q: bool,
    has_seq_len_kv: bool,
    has_cum_q: bool,
    has_cum_kv: bool,
    is_deterministic: bool,
):
    q_shape, q_stride = tuple(q.shape), tuple(q.stride())
    k_shape, k_stride = tuple(k.shape), tuple(k.stride())
    v_shape, v_stride = tuple(v.shape), tuple(v.stride())
    return (
        "bprop",
        q_shape,
        q_stride,
        q.dtype,
        k_shape,
        k_stride,
        k.dtype,
        v_shape,
        v_stride,
        v.dtype,
        attn_scale,
        is_causal,
        diagonal_alignment,
        left_bound,
        right_bound,
        has_seq_len_q,
        has_seq_len_kv,
        has_cum_q,
        has_cum_kv,
        is_deterministic,
        q.device,
    )


# ---------------------------------------------------------------------------
# Forward graph builder
# ---------------------------------------------------------------------------


def _build_fprop_graph(
    handle,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    is_causal: bool,
    diagonal_alignment: int,
    left_bound: int,
    right_bound: int,
    seq_len_q: Optional[torch.Tensor],
    seq_len_kv: Optional[torch.Tensor],
    cumulative_seq_len_q: Optional[torch.Tensor],
    cumulative_seq_len_kv: Optional[torch.Tensor],
):
    """Build, validate, and compile a forward SDPA cuDNN graph."""

    io_dtype = _torch_dtype_to_cudnn(q.dtype)

    _logger.debug(f"Building forward graph for q: {q.shape}, k: {k.shape}, v: {v.shape}")

    q_shape, q_stride = tuple(q.shape), tuple(q.stride())
    k_shape, k_stride = tuple(k.shape), tuple(k.stride())
    v_shape, v_stride = tuple(v.shape), tuple(v.stride())

    B, H_q, S_q, D_qk = q_shape
    _, H_v, S_kv, D_v = v_shape

    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    # -- Input tensors --
    q_t = graph.tensor(name="q", dim=list(q_shape), stride=list(q_stride), data_type=io_dtype, uid=_UIDs.Q)
    k_t = graph.tensor(name="k", dim=list(k_shape), stride=list(k_stride), data_type=io_dtype, uid=_UIDs.K)
    v_t = graph.tensor(name="v", dim=list(v_shape), stride=list(v_stride), data_type=io_dtype, uid=_UIDs.V)

    # -- Optional tensors --
    seq_len_q_t = None
    seq_len_kv_t = None
    cum_q_t = None
    cum_kv_t = None

    if seq_len_q is not None:
        seq_len_q_t = graph.tensor(
            name="seq_len_q", dim=list(seq_len_q.shape), stride=list(seq_len_q.stride()), data_type=cudnn.data_type.INT32, uid=_UIDs.SEQ_LEN_Q
        )
    if seq_len_kv is not None:
        seq_len_kv_t = graph.tensor(
            name="seq_len_kv", dim=list(seq_len_kv.shape), stride=list(seq_len_kv.stride()), data_type=cudnn.data_type.INT32, uid=_UIDs.SEQ_LEN_KV
        )
    if cumulative_seq_len_q is not None:
        cum_q_t = graph.tensor(
            name="cum_seq_len_q",
            dim=list(cumulative_seq_len_q.shape),
            stride=list(cumulative_seq_len_q.stride()),
            data_type=cudnn.data_type.INT32,
            uid=_UIDs.CUM_SEQ_LEN_Q,
        )
    if cumulative_seq_len_kv is not None:
        cum_kv_t = graph.tensor(
            name="cum_seq_len_kv",
            dim=list(cumulative_seq_len_kv.shape),
            stride=list(cumulative_seq_len_kv.stride()),
            data_type=cudnn.data_type.INT32,
            uid=_UIDs.CUM_SEQ_LEN_KV,
        )

    # -- Ragged offsets --
    if cum_q_t is not None:
        q_t.set_ragged_offset(cum_q_t)
    if cum_kv_t is not None:
        k_t.set_ragged_offset(cum_kv_t)
        v_t.set_ragged_offset(cum_kv_t)

    # -- Mask configuration --
    use_padding = seq_len_q is not None or seq_len_kv is not None
    lb = left_bound if left_bound >= 0 else None
    rb = right_bound if right_bound >= 0 else None
    if is_causal and rb is None:
        rb = 0

    # -- SDPA forward --
    o_t, stats_t = graph.sdpa(
        name="sdpa",
        q=q_t,
        k=k_t,
        v=v_t,
        generate_stats=True,
        attn_scale=attn_scale,
        use_padding_mask=use_padding,
        seq_len_q=seq_len_q_t,
        seq_len_kv=seq_len_kv_t,
        diagonal_alignment=_diagonal_alignment_enum(diagonal_alignment),
        diagonal_band_left_bound=lb,
        diagonal_band_right_bound=rb,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    # -- Output shapes (BHSD contiguous) --
    o_shape = (B, H_q, S_q, D_v)
    o_stride = (H_q * S_q * D_v, S_q * D_v, D_v, 1)
    o_t.set_uid(_UIDs.O).set_output(True).set_dim(list(o_shape)).set_stride(list(o_stride))
    o_t.set_data_type(io_dtype)

    if cum_q_t is not None:
        o_t.set_ragged_offset(cum_q_t)

    stats_t.set_uid(_UIDs.STATS).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    if cum_q_t is not None:
        stats_t.set_ragged_offset(cum_q_t)

    # -- Build --
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace_size = graph.get_workspace_size()

    return graph, workspace_size


# ---------------------------------------------------------------------------
# Backward graph builder
# ---------------------------------------------------------------------------


def _build_bprop_graph(
    handle,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    dO: torch.Tensor,
    stats: torch.Tensor,
    attn_scale: float,
    is_causal: bool,
    diagonal_alignment: int,
    left_bound: int,
    right_bound: int,
    seq_len_q: Optional[torch.Tensor],
    seq_len_kv: Optional[torch.Tensor],
    cumulative_seq_len_q: Optional[torch.Tensor],
    cumulative_seq_len_kv: Optional[torch.Tensor],
    is_deterministic: bool,
):
    """Build, validate, and compile a backward SDPA cuDNN graph."""

    io_dtype = _torch_dtype_to_cudnn(q.dtype)

    _logger.debug(f"Building backward graph for q: {q.shape}, k: {k.shape}, v: {v.shape}, o: {o.shape}, dO: {dO.shape}")

    q_shape, q_stride = tuple(q.shape), tuple(q.stride())
    k_shape, k_stride = tuple(k.shape), tuple(k.stride())
    v_shape, v_stride = tuple(v.shape), tuple(v.stride())
    o_shape, o_stride = tuple(o.shape), tuple(o.stride())
    dO_shape, dO_stride = tuple(dO.shape), tuple(dO.stride())

    B, H_q, S_q, D_qk = q_shape
    _, H_k, S_kv, _ = k_shape
    _, H_v, _, D_v = v_shape

    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    # -- Input tensors --
    q_t = graph.tensor(name="q", dim=list(q_shape), stride=list(q_stride), data_type=io_dtype, uid=_UIDs.Q)
    k_t = graph.tensor(name="k", dim=list(k_shape), stride=list(k_stride), data_type=io_dtype, uid=_UIDs.K)
    v_t = graph.tensor(name="v", dim=list(v_shape), stride=list(v_stride), data_type=io_dtype, uid=_UIDs.V)
    o_t = graph.tensor(name="o", dim=list(o_shape), stride=list(o_stride), data_type=io_dtype, uid=_UIDs.O)
    dO_t = graph.tensor(name="dO", dim=list(dO_shape), stride=list(dO_stride), data_type=io_dtype, uid=_UIDs.DO)
    stats_t = graph.tensor(name="stats", dim=list(stats.shape), stride=list(stats.stride()), data_type=cudnn.data_type.FLOAT, uid=_UIDs.STATS)

    # -- Optional tensors --
    seq_len_q_t = None
    seq_len_kv_t = None
    cum_q_t = None
    cum_kv_t = None

    if seq_len_q is not None:
        seq_len_q_t = graph.tensor(
            name="seq_len_q", dim=list(seq_len_q.shape), stride=list(seq_len_q.stride()), data_type=cudnn.data_type.INT32, uid=_UIDs.SEQ_LEN_Q
        )
    if seq_len_kv is not None:
        seq_len_kv_t = graph.tensor(
            name="seq_len_kv", dim=list(seq_len_kv.shape), stride=list(seq_len_kv.stride()), data_type=cudnn.data_type.INT32, uid=_UIDs.SEQ_LEN_KV
        )
    if cumulative_seq_len_q is not None:
        cum_q_t = graph.tensor(
            name="cum_seq_len_q",
            dim=list(cumulative_seq_len_q.shape),
            stride=list(cumulative_seq_len_q.stride()),
            data_type=cudnn.data_type.INT32,
            uid=_UIDs.CUM_SEQ_LEN_Q,
        )
    if cumulative_seq_len_kv is not None:
        cum_kv_t = graph.tensor(
            name="cum_seq_len_kv",
            dim=list(cumulative_seq_len_kv.shape),
            stride=list(cumulative_seq_len_kv.stride()),
            data_type=cudnn.data_type.INT32,
            uid=_UIDs.CUM_SEQ_LEN_KV,
        )

    # -- Ragged offsets --
    if cum_q_t is not None:
        q_t.set_ragged_offset(cum_q_t)
        o_t.set_ragged_offset(cum_q_t)
        dO_t.set_ragged_offset(cum_q_t)
    if cum_kv_t is not None:
        k_t.set_ragged_offset(cum_kv_t)
        v_t.set_ragged_offset(cum_kv_t)

    # -- Mask configuration --
    use_padding = seq_len_q is not None or seq_len_kv is not None
    lb = left_bound if left_bound >= 0 else None
    rb = right_bound if right_bound >= 0 else None
    if is_causal and rb is None:
        rb = 0

    # Compute max_total_seq_len for ragged backward
    max_total_seq_len_q = None
    max_total_seq_len_kv = None
    if cumulative_seq_len_q is not None and seq_len_q is not None:
        total = torch.sum(seq_len_q).item()
        max_total_seq_len_q = ((total + 63) // 64) * 64
    if cumulative_seq_len_kv is not None and seq_len_kv is not None:
        total = torch.sum(seq_len_kv).item()
        max_total_seq_len_kv = ((total + 63) // 64) * 64

    # -- SDPA backward --
    dQ_t, dK_t, dV_t = graph.sdpa_backward(
        name="sdpa_backward",
        q=q_t,
        k=k_t,
        v=v_t,
        o=o_t,
        dO=dO_t,
        stats=stats_t,
        attn_scale=attn_scale,
        use_padding_mask=use_padding,
        seq_len_q=seq_len_q_t,
        seq_len_kv=seq_len_kv_t,
        max_total_seq_len_q=max_total_seq_len_q,
        max_total_seq_len_kv=max_total_seq_len_kv,
        diagonal_alignment=_diagonal_alignment_enum(diagonal_alignment),
        diagonal_band_left_bound=lb,
        diagonal_band_right_bound=rb,
        use_deterministic_algorithm=is_deterministic,
    )

    # -- Output shapes (BHSD order) --
    dq_shape, dq_stride = q_shape, q_stride
    dk_shape, dk_stride = k_shape, k_stride
    dv_shape, dv_stride = v_shape, v_stride

    dQ_t.set_uid(_UIDs.DQ).set_output(True).set_dim(list(dq_shape)).set_stride(list(dq_stride))
    dK_t.set_uid(_UIDs.DK).set_output(True).set_dim(list(dk_shape)).set_stride(list(dk_stride))
    dV_t.set_uid(_UIDs.DV).set_output(True).set_dim(list(dv_shape)).set_stride(list(dv_stride))

    if cum_q_t is not None:
        dQ_t.set_ragged_offset(cum_q_t)
    if cum_kv_t is not None:
        dK_t.set_ragged_offset(cum_kv_t)
        dV_t.set_ragged_offset(cum_kv_t)

    # -- Build --
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace_size = graph.get_workspace_size()

    return graph, workspace_size


# ---------------------------------------------------------------------------
# Forward custom op
# ---------------------------------------------------------------------------


_lib = torch.library.Library("cudnn", "DEF")

_lib.define(
    "sdpa(Tensor q, Tensor k, Tensor v, float attn_scale, "
    "bool is_causal=False, int diagonal_alignment=0, "
    "int left_bound=-1, int right_bound=-1, "
    "Tensor? seq_len_q=None, Tensor? seq_len_kv=None, "
    "Tensor? cumulative_seq_len_q=None, Tensor? cumulative_seq_len_kv=None"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "sdpa_bwd(Tensor dO, Tensor q, Tensor k, Tensor v, Tensor o, Tensor stats, "
    "float attn_scale, bool is_causal=False, int diagonal_alignment=0, "
    "int left_bound=-1, int right_bound=-1, "
    "Tensor? seq_len_q=None, Tensor? seq_len_kv=None, "
    "Tensor? cumulative_seq_len_q=None, Tensor? cumulative_seq_len_kv=None, "
    "bool is_deterministic=False"
    ") -> (Tensor, Tensor, Tensor)"
)


def _sdpa_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    cuDNN SDPA forward (internal). BHSD layout.

    Args:
        q: Query tensor (B, H_q, S_q, D_qk)
        k: Key tensor   (B, H_k, S_kv, D_qk)
        v: Value tensor (B, H_v, S_kv, D_v)
        attn_scale: Attention scale factor (typically 1/sqrt(D_qk))
        is_causal: If True, apply causal mask (right_bound defaults to 0)
        diagonal_alignment: 0 = TOP_LEFT, 1 = BOTTOM_RIGHT
        left_bound: Left sliding window bound (>= 0 to enable, -1 = disabled)
        right_bound: Right sliding window bound (>= 0 to enable, -1 = disabled)
        seq_len_q: Actual query sequence lengths (B, 1, 1, 1) INT32
        seq_len_kv: Actual key/value sequence lengths (B, 1, 1, 1) INT32
        cumulative_seq_len_q: Ragged offset for Q (B+1, 1, 1, 1) INT32
        cumulative_seq_len_kv: Ragged offset for KV (B+1, 1, 1, 1) INT32

    Returns:
        (O, Stats): Output tensor (B, H_q, S_q, D_v) and softmax stats (B, H_q, S_q, 1)
    """
    is_d256 = q.shape[-1] == 256 and k.shape[-1] == 256 and v.shape[-1] == 256
    can_use_d256_oss_fwd = is_d256 and _can_use_d256_oss_path(
        is_causal,
        diagonal_alignment,
        left_bound,
        right_bound,
        seq_len_q,
        seq_len_kv,
        cumulative_seq_len_q,
        cumulative_seq_len_kv,
    )
    use_d256_oss_fwd = can_use_d256_oss_fwd and _device_supports_d256_oss(q.device)
    if can_use_d256_oss_fwd and not use_d256_oss_fwd:
        _logger.debug("Falling back to cuDNN graph d=256 forward path because OSS kernel requires SM100+, got device %s", q.device)
    if use_d256_oss_fwd:
        try:
            return sdpa_fwd_d256(
                q,
                k,
                v,
                attn_scale,
                is_causal,
                diagonal_alignment,
                left_bound,
                right_bound,
                seq_len_q,
                seq_len_kv,
                cumulative_seq_len_q,
                cumulative_seq_len_kv,
            )
        except ImportError as e:
            _logger.warning("Falling back to cuDNN graph d=256 forward path because OSS dependencies are unavailable: %s", e)

    handle = _get_handle(q.device)

    cache_key = _make_fprop_cache_key(
        q,
        k,
        v,
        attn_scale,
        is_causal,
        diagonal_alignment,
        left_bound,
        right_bound,
        seq_len_q is not None,
        seq_len_kv is not None,
        cumulative_seq_len_q is not None,
        cumulative_seq_len_kv is not None,
    )

    if cache_key not in _fprop_cache:
        graph, workspace_size = _build_fprop_graph(
            handle,
            q,
            k,
            v,
            attn_scale,
            is_causal,
            diagonal_alignment,
            left_bound,
            right_bound,
            seq_len_q,
            seq_len_kv,
            cumulative_seq_len_q,
            cumulative_seq_len_kv,
        )
        uid_order = _get_variant_pack_uid_order(
            graph,
            [
                int(_UIDs.Q),
                int(_UIDs.K),
                int(_UIDs.V),
                int(_UIDs.O),
                int(_UIDs.STATS),
                *([int(_UIDs.SEQ_LEN_Q)] if seq_len_q is not None else []),
                *([int(_UIDs.SEQ_LEN_KV)] if seq_len_kv is not None else []),
                *([int(_UIDs.CUM_SEQ_LEN_Q)] if cumulative_seq_len_q is not None else []),
                *([int(_UIDs.CUM_SEQ_LEN_KV)] if cumulative_seq_len_kv is not None else []),
            ],
        )
        _fprop_cache[cache_key] = (graph, workspace_size, uid_order)

    graph, workspace_size, uid_order = _fprop_cache[cache_key]

    # Allocate outputs and workspace (BHSD layout)
    # Workspace is per-call — PyTorch's caching allocator recycles the allocation.
    B, H_q, S_q, D_qk = q.shape
    _, H_v, S_kv, D_v = v.shape
    o_gpu = torch.empty(B, H_q, S_q, D_v, dtype=q.dtype, device=q.device)
    stats_gpu = torch.empty(B, H_q, S_q, 1, dtype=torch.float32, device=q.device)
    workspace = torch.empty(max(workspace_size, 1), device=q.device, dtype=torch.uint8)

    # UID → tensor map for sorted pointer extraction
    uid_to_tensor = {
        int(_UIDs.Q): q,
        int(_UIDs.K): k,
        int(_UIDs.V): v,
        int(_UIDs.O): o_gpu,
        int(_UIDs.STATS): stats_gpu,
    }
    if seq_len_q is not None:
        uid_to_tensor[int(_UIDs.SEQ_LEN_Q)] = seq_len_q
    if seq_len_kv is not None:
        uid_to_tensor[int(_UIDs.SEQ_LEN_KV)] = seq_len_kv
    if cumulative_seq_len_q is not None:
        uid_to_tensor[int(_UIDs.CUM_SEQ_LEN_Q)] = cumulative_seq_len_q
    if cumulative_seq_len_kv is not None:
        uid_to_tensor[int(_UIDs.CUM_SEQ_LEN_KV)] = cumulative_seq_len_kv

    if hasattr(graph, "_execute_with_ptrs"):
        ptrs = [uid_to_tensor[uid].data_ptr() for uid in uid_order]
        graph._execute_with_ptrs(ptrs, workspace.data_ptr(), int(handle))
    else:
        graph.execute(uid_to_tensor, workspace, handle=handle)

    return o_gpu, stats_gpu


_lib.impl("sdpa", _sdpa_impl, "CUDA")


@torch.library.register_fake("cudnn::sdpa")
def _sdpa_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H_q, S_q, D_qk = q.shape
    _, H_v, S_kv, D_v = v.shape
    O = torch.empty(B, H_q, S_q, D_v, dtype=q.dtype, device=q.device)
    Stats = torch.empty(B, H_q, S_q, 1, dtype=torch.float32, device=q.device)
    return O, Stats


# ---------------------------------------------------------------------------
# Specialized d=256 custom ops
# ---------------------------------------------------------------------------


@torch.library.custom_op("cudnn::sdpa_fwd_d256", mutates_args=())
def sdpa_fwd_d256(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SDPA forward for d=256 via the SM100 FE OSS API."""

    window_size = _validate_d256_oss_args(
        is_causal,
        diagonal_alignment,
        left_bound,
        right_bound,
        seq_len_q,
        seq_len_kv,
        cumulative_seq_len_q,
        cumulative_seq_len_kv,
    )

    from cudnn.sdpa import sdpa_fwd_wrapper_sm100_d256

    q_fwd = _to_sdpa_oss_bhsd_layout(q, "q")
    k_fwd = _to_sdpa_oss_bhsd_layout(k, "k")
    v_fwd = _to_sdpa_oss_bhsd_layout(v, "v")

    o, lse = sdpa_fwd_wrapper_sm100_d256(
        q_tensor=q_fwd,
        k_tensor=k_fwd,
        v_tensor=v_fwd,
        is_causal=is_causal,
        window_size=window_size,
        scale_softmax=attn_scale,
        current_stream=_get_current_stream(q.device),
    )

    return o.contiguous(), lse.unsqueeze(-1).contiguous()


@sdpa_fwd_d256.register_fake
def _sdpa_fwd_d256_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty((*q.shape[:3], 1), dtype=torch.float32, device=q.device)


@torch.library.custom_op("cudnn::sdpa_bwd_d256", mutates_args=())
def sdpa_bwd_d256(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    stats: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
    is_deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDPA backward for d=256 via the SM100 FE OSS API."""
    del is_deterministic

    window_size = _validate_d256_oss_args(
        is_causal,
        diagonal_alignment,
        left_bound,
        right_bound,
        seq_len_q,
        seq_len_kv,
        cumulative_seq_len_q,
        cumulative_seq_len_kv,
    )

    from cudnn.sdpa import sdpa_bwd_wrapper_sm100_d256

    q_bwd = _to_sdpa_oss_bhsd_layout(q, "q")
    k_bwd = _to_sdpa_oss_bhsd_layout(k, "k")
    v_bwd = _to_sdpa_oss_bhsd_layout(v, "v")
    o_bwd = _to_sdpa_oss_bhsd_layout(o, "o")
    dO_bwd = _to_sdpa_oss_bhsd_layout(dO, "dO")
    lse_bwd = stats.squeeze(-1).contiguous()
    cum_q_bwd = None
    cum_kv_bwd = None

    dQ, dK, dV = sdpa_bwd_wrapper_sm100_d256(
        q_tensor=q_bwd,
        k_tensor=k_bwd,
        v_tensor=v_bwd,
        o_tensor=o_bwd,
        do_tensor=dO_bwd,
        lse_tensor=lse_bwd,
        cum_seqlen_q_tensor=cum_q_bwd,
        cum_seqlen_k_tensor=cum_kv_bwd,
        is_causal=is_causal,
        window_size=window_size,
        scale_softmax=attn_scale,
        current_stream=_get_current_stream(q.device),
    )

    return dQ.contiguous(), dK.contiguous(), dV.contiguous()


@sdpa_bwd_d256.register_fake
def _sdpa_bwd_d256_fake(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    stats: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
    is_deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _sdpa_bwd_impl(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    stats: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
    is_deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    cuDNN SDPA backward (graph-based).

    Returns:
        (dQ, dK, dV) in BHSD layout matching Q, K, V shapes.
    """
    handle = _get_handle(dO.device)

    cache_key = _make_bprop_cache_key(
        q,
        k,
        v,
        attn_scale,
        is_causal,
        diagonal_alignment,
        left_bound,
        right_bound,
        seq_len_q is not None,
        seq_len_kv is not None,
        cumulative_seq_len_q is not None,
        cumulative_seq_len_kv is not None,
        is_deterministic,
    )

    if cache_key not in _bprop_cache:
        graph, workspace_size = _build_bprop_graph(
            handle,
            q,
            k,
            v,
            o,
            dO,
            stats,
            attn_scale,
            is_causal,
            diagonal_alignment,
            left_bound,
            right_bound,
            seq_len_q,
            seq_len_kv,
            cumulative_seq_len_q,
            cumulative_seq_len_kv,
            is_deterministic,
        )
        uid_order = _get_variant_pack_uid_order(
            graph,
            [
                int(_UIDs.Q),
                int(_UIDs.K),
                int(_UIDs.V),
                int(_UIDs.O),
                int(_UIDs.DO),
                int(_UIDs.STATS),
                int(_UIDs.DQ),
                int(_UIDs.DK),
                int(_UIDs.DV),
                *([int(_UIDs.SEQ_LEN_Q)] if seq_len_q is not None else []),
                *([int(_UIDs.SEQ_LEN_KV)] if seq_len_kv is not None else []),
                *([int(_UIDs.CUM_SEQ_LEN_Q)] if cumulative_seq_len_q is not None else []),
                *([int(_UIDs.CUM_SEQ_LEN_KV)] if cumulative_seq_len_kv is not None else []),
            ],
        )
        _bprop_cache[cache_key] = (graph, workspace_size, uid_order)

    graph, workspace_size, uid_order = _bprop_cache[cache_key]

    # Allocate gradient outputs and workspace (same shapes as Q, K, V)
    dQ_gpu = torch.empty_like(q)
    dK_gpu = torch.empty_like(k)
    dV_gpu = torch.empty_like(v)
    workspace = torch.empty(max(workspace_size, 1), device=dO.device, dtype=torch.uint8)

    # UID → tensor map for sorted pointer extraction
    uid_to_tensor = {
        int(_UIDs.Q): q,
        int(_UIDs.K): k,
        int(_UIDs.V): v,
        int(_UIDs.O): o,
        int(_UIDs.DO): dO,
        int(_UIDs.STATS): stats,
        int(_UIDs.DQ): dQ_gpu,
        int(_UIDs.DK): dK_gpu,
        int(_UIDs.DV): dV_gpu,
    }
    if seq_len_q is not None:
        uid_to_tensor[int(_UIDs.SEQ_LEN_Q)] = seq_len_q
    if seq_len_kv is not None:
        uid_to_tensor[int(_UIDs.SEQ_LEN_KV)] = seq_len_kv
    if cumulative_seq_len_q is not None:
        uid_to_tensor[int(_UIDs.CUM_SEQ_LEN_Q)] = cumulative_seq_len_q
    if cumulative_seq_len_kv is not None:
        uid_to_tensor[int(_UIDs.CUM_SEQ_LEN_KV)] = cumulative_seq_len_kv

    if hasattr(graph, "_execute_with_ptrs"):
        ptrs = [uid_to_tensor[uid].data_ptr() for uid in uid_order]
        graph._execute_with_ptrs(ptrs, workspace.data_ptr(), int(handle))
    else:
        graph.execute(uid_to_tensor, workspace, handle=handle)

    return dQ_gpu, dK_gpu, dV_gpu


_lib.impl("sdpa_bwd", _sdpa_bwd_impl, "CUDA")


@torch.library.register_fake("cudnn::sdpa_bwd")
def _sdpa_bwd_fake(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    stats: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
    is_deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# ---------------------------------------------------------------------------
# Autograd registration
# ---------------------------------------------------------------------------


def _sdpa_setup_context(ctx, inputs, output):
    q, k, v, attn_scale, is_causal, diagonal_alignment, left_bound, right_bound, seq_len_q, seq_len_kv, cumulative_seq_len_q, cumulative_seq_len_kv = inputs
    o, stats = output

    tensors_to_save = [q, k, v, o, stats]
    ctx.has_seq_len_q = seq_len_q is not None
    ctx.has_seq_len_kv = seq_len_kv is not None
    ctx.has_cum_q = cumulative_seq_len_q is not None
    ctx.has_cum_kv = cumulative_seq_len_kv is not None

    if ctx.has_seq_len_q:
        tensors_to_save.append(seq_len_q)
    if ctx.has_seq_len_kv:
        tensors_to_save.append(seq_len_kv)
    if ctx.has_cum_q:
        tensors_to_save.append(cumulative_seq_len_q)
    if ctx.has_cum_kv:
        tensors_to_save.append(cumulative_seq_len_kv)

    ctx.save_for_backward(*tensors_to_save)

    ctx.attn_scale = attn_scale
    ctx.is_causal = is_causal
    ctx.diagonal_alignment = diagonal_alignment
    ctx.left_bound = left_bound
    ctx.right_bound = right_bound


def _sdpa_backward(ctx, dO, dStats):
    # dO from autograd may have zero strides (e.g. from o.sum().backward()),
    # which cuDNN cannot handle. Make it contiguous.
    dO = dO.contiguous()

    saved = list(ctx.saved_tensors)
    q, k, v, o, stats = saved[:5]
    idx = 5
    seq_len_q = saved[idx] if ctx.has_seq_len_q else None
    if ctx.has_seq_len_q:
        idx += 1
    seq_len_kv = saved[idx] if ctx.has_seq_len_kv else None
    if ctx.has_seq_len_kv:
        idx += 1
    cum_q = saved[idx] if ctx.has_cum_q else None
    if ctx.has_cum_q:
        idx += 1
    cum_kv = saved[idx] if ctx.has_cum_kv else None

    is_d256 = q.shape[-1] == 256
    can_use_d256_oss_bwd = is_d256 and _can_use_d256_oss_path(
        ctx.is_causal,
        ctx.diagonal_alignment,
        ctx.left_bound,
        ctx.right_bound,
        seq_len_q,
        seq_len_kv,
        cum_q,
        cum_kv,
    )
    use_d256_oss_bwd = can_use_d256_oss_bwd and _device_supports_d256_oss(q.device)

    if is_d256:
        if use_d256_oss_bwd:
            try:
                dQ, dK, dV = sdpa_bwd_d256(
                    dO,
                    q,
                    k,
                    v,
                    o,
                    stats,
                    ctx.attn_scale,
                    ctx.is_causal,
                    ctx.diagonal_alignment,
                    ctx.left_bound,
                    ctx.right_bound,
                    seq_len_q,
                    seq_len_kv,
                    cum_q,
                    cum_kv,
                )
            except ImportError as e:
                _logger.warning("Falling back to cuDNN graph d=256 backward path because OSS dependencies are unavailable: %s", e)
                dQ, dK, dV = torch.ops.cudnn.sdpa_bwd(
                    dO,
                    q,
                    k,
                    v,
                    o,
                    stats,
                    ctx.attn_scale,
                    ctx.is_causal,
                    ctx.diagonal_alignment,
                    ctx.left_bound,
                    ctx.right_bound,
                    seq_len_q,
                    seq_len_kv,
                    cum_q,
                    cum_kv,
                )
        elif can_use_d256_oss_bwd:
            dQ, dK, dV = torch.ops.cudnn.sdpa_bwd(
                dO,
                q,
                k,
                v,
                o,
                stats,
                ctx.attn_scale,
                ctx.is_causal,
                ctx.diagonal_alignment,
                ctx.left_bound,
                ctx.right_bound,
                seq_len_q,
                seq_len_kv,
                cum_q,
                cum_kv,
            )
        else:
            raise NotImplementedError("d=256 backward path does not support seq_len_q/seq_len_kv/cum_q/cum_kv")
    else:
        dQ, dK, dV = torch.ops.cudnn.sdpa_bwd(
            dO,
            q,
            k,
            v,
            o,
            stats,
            ctx.attn_scale,
            ctx.is_causal,
            ctx.diagonal_alignment,
            ctx.left_bound,
            ctx.right_bound,
            seq_len_q,
            seq_len_kv,
            cum_q,
            cum_kv,
        )

    # Return gradients for: q, k, v, attn_scale, is_causal, diagonal_alignment,
    # left_bound, right_bound, seq_len_q, seq_len_kv, cum_q, cum_kv
    return dQ, dK, dV, None, None, None, None, None, None, None, None, None


torch.library.register_autograd("cudnn::sdpa", _sdpa_backward, setup_context=_sdpa_setup_context)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    *,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cumulative_seq_len_q: Optional[torch.Tensor] = None,
    cumulative_seq_len_kv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """cuDNN-accelerated Scaled Dot-Product Attention.

    API closely mirrors ``torch.nn.functional.scaled_dot_product_attention``.

    **Layout**: tensors use **BHSD** layout ``(batch, num_heads, seq_len, head_dim)``,
    matching PyTorch's convention.

    Args:
        query: Query tensor ``(B, H_q, S_q, D)``.
        key: Key tensor ``(B, H_k, S_kv, D)``.
        value: Value tensor ``(B, H_v, S_kv, D_v)``.
        attn_mask: Not yet supported. Must be ``None``.
        dropout_p: Not yet supported. Must be ``0.0``.
        is_causal: If ``True``, applies a causal (upper-triangular) mask.
        scale: Attention scale factor. Defaults to ``1 / sqrt(D)`` when ``None``.
        enable_gqa: When ``False``, raises ``ValueError`` if ``H_q != H_k``.
            When ``True``, grouped-query attention is enabled (cuDNN handles
            this automatically via head dimension broadcast).

        diagonal_alignment: cuDNN extension. ``0`` = TOP_LEFT, ``1`` = BOTTOM_RIGHT.
        left_bound: cuDNN extension. Left sliding-window bound (``-1`` = disabled).
        right_bound: cuDNN extension. Right sliding-window bound (``-1`` = disabled).
        seq_len_q: cuDNN extension. Actual query sequence lengths ``(B, 1, 1, 1)`` INT32.
        seq_len_kv: cuDNN extension. Actual key/value sequence lengths ``(B, 1, 1, 1)`` INT32.
        cumulative_seq_len_q: cuDNN extension. Ragged offset for Q ``(B+1, 1, 1, 1)`` INT32.
        cumulative_seq_len_kv: cuDNN extension. Ragged offset for KV ``(B+1, 1, 1, 1)`` INT32.

    Note:
        For head dimension ``256``, the specialized backward path currently supports
        only plain BHSD tensors and does not support ``seq_len_q``, ``seq_len_kv``,
        ``cumulative_seq_len_q``, or ``cumulative_seq_len_kv``.

    Returns:
        Output tensor ``(B, H_q, S_q, D_v)``.
    """
    if attn_mask is not None:
        raise NotImplementedError("attn_mask is not yet supported by cuDNN SDPA")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not yet supported by cuDNN SDPA")
    if not enable_gqa and query.shape[1] != key.shape[1]:
        raise ValueError(f"query has {query.shape[1]} heads but key has {key.shape[1]} heads. " f"Set enable_gqa=True for grouped-query attention.")

    d = query.shape[-1]
    attn_scale = scale if scale is not None else (1.0 / math.sqrt(d))

    o, _stats = torch.ops.cudnn.sdpa(
        query,
        key,
        value,
        attn_scale,
        is_causal=is_causal,
        diagonal_alignment=diagonal_alignment,
        left_bound=left_bound,
        right_bound=right_bound,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        cumulative_seq_len_q=cumulative_seq_len_q,
        cumulative_seq_len_kv=cumulative_seq_len_kv,
    )
    return o
