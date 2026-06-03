"""
PyTorch custom operator wrapping cuDNN's MoE Grouped Matmul.

Provides ``moe_grouped_matmul`` as the public entry point for performing
expert-parallel grouped matrix multiplication used in Mixture-of-Experts layers.

**Layout convention**:
- Token tensor: ``(1, total_tokens, hidden_size)`` row-major
- Weight tensor: ``(num_experts, hidden_size, output_size)`` with column-major inner dims
- first_token_offset: ``(batch_size * num_experts, 1, 1)`` INT32

Graph caching ensures cuDNN graphs are built once per unique configuration
and reused across calls.
"""

import logging
from typing import Optional, Dict
from enum import IntEnum

import torch
import cudnn

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_cudnn_handles = {}
_graph_cache: Dict[tuple, tuple] = {}

# Dtype mapping (module-level constant)
_TORCH_DTYPE_TO_CUDNN = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
    torch.int32: cudnn.data_type.INT32,
}

# Mode string to cuDNN enum mapping (lazily initialised to avoid import-time
# failures on cuDNN versions that pre-date MoE support).
_MODE_STR_TO_CUDNN: Optional[Dict[str, object]] = None


def _get_mode_mapping() -> Dict[str, object]:
    """Return the mode-string → cuDNN-enum dict, building it on first call."""
    global _MODE_STR_TO_CUDNN
    if _MODE_STR_TO_CUDNN is None:
        try:
            _moe_mode = cudnn.moe_grouped_matmul_mode
        except AttributeError as exc:
            raise RuntimeError("cuDNN MoE Grouped Matmul is not available in the installed cuDNN build. " "Upgrade to cuDNN >= 9.18.0.") from exc
        _MODE_STR_TO_CUDNN = {
            "none": _moe_mode.NONE,
            "gather": _moe_mode.GATHER,
            "scatter": _moe_mode.SCATTER,
        }
    return _MODE_STR_TO_CUDNN


# ---------------------------------------------------------------------------
# UID enum -- explicit tensor UIDs for graph caching
# ---------------------------------------------------------------------------


class _UIDs(IntEnum):
    TOKEN = 1
    WEIGHT = 2
    FIRST_TOKEN_OFFSET = 3
    TOKEN_INDEX = 4
    TOKEN_KS = 5
    OUTPUT = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_handle(device: torch.device):
    """Return a lazily-initialised cuDNN handle with the current CUDA stream."""
    if device not in _cudnn_handles:
        _cudnn_handles[device] = cudnn.create_handle()
    stream = torch.cuda.current_stream(device).cuda_stream
    cudnn.set_stream(handle=_cudnn_handles[device], stream=stream)
    return _cudnn_handles[device]


def _torch_dtype_to_cudnn(dtype: torch.dtype):
    """Map a PyTorch dtype to a cuDNN data_type enum."""
    return _TORCH_DTYPE_TO_CUDNN[dtype]


def _make_cache_key(
    token: torch.Tensor,
    weight: torch.Tensor,
    first_token_offset: torch.Tensor,
    has_token_index: bool,
    token_index_shape: Optional[tuple],
    has_token_ks: bool,
    token_ks_shape: Optional[tuple],
    mode: str,
    top_k: int,
):
    return (
        "moe_grouped_matmul",
        tuple(token.shape),
        tuple(token.stride()),
        token.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
        weight.dtype,
        tuple(first_token_offset.shape),
        tuple(first_token_offset.stride()),
        has_token_index,
        token_index_shape,
        has_token_ks,
        token_ks_shape,
        mode,
        top_k,
        token.device,
    )


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_graph(
    handle,
    token: torch.Tensor,
    weight: torch.Tensor,
    first_token_offset: torch.Tensor,
    token_index: Optional[torch.Tensor],
    token_ks: Optional[torch.Tensor],
    mode: str,
    top_k: int,
):
    """Build, validate, and compile a MoE Grouped Matmul cuDNN graph."""

    io_dtype = _torch_dtype_to_cudnn(token.dtype)
    # MoE kernel supports HALF and BFLOAT16 io with FLOAT compute,
    # and HALF io with HALF compute.
    compute_dtype = cudnn.data_type.FLOAT

    _logger.debug(f"Building MoE grouped matmul graph for token: {token.shape}, weight: {weight.shape}, mode: {mode}")

    # Token: (1, M, K) — use actual strides
    token_shape = list(token.shape)
    token_stride = list(token.stride())

    # Weight: (num_experts, K, N)
    num_experts, K, N = weight.shape
    weight_shape = list(weight.shape)
    weight_stride = list(weight.stride())

    # first_token_offset: (batch_size * num_experts, 1, 1) — use actual strides
    fto_shape = list(first_token_offset.shape)
    fto_stride = list(first_token_offset.stride())

    graph = cudnn.pygraph(
        intermediate_data_type=io_dtype,
        compute_data_type=compute_dtype,
        handle=handle,
    )

    # -- Input tensors --
    token_t = graph.tensor(
        name="token",
        dim=token_shape,
        stride=token_stride,
        data_type=io_dtype,
        uid=_UIDs.TOKEN,
    )
    weight_t = graph.tensor(
        name="weight",
        dim=weight_shape,
        stride=weight_stride,
        data_type=io_dtype,
        uid=_UIDs.WEIGHT,
    )
    fto_t = graph.tensor(
        name="first_token_offset",
        dim=fto_shape,
        stride=fto_stride,
        data_type=cudnn.data_type.INT32,
        uid=_UIDs.FIRST_TOKEN_OFFSET,
    )

    # -- Optional tensors --
    token_index_t = None
    token_ks_t = None

    if token_index is not None:
        token_index_t = graph.tensor(
            name="token_index",
            dim=list(token_index.shape),
            stride=list(token_index.stride()),
            data_type=cudnn.data_type.INT32,
            uid=_UIDs.TOKEN_INDEX,
        )

    if token_ks is not None:
        token_ks_t = graph.tensor(
            name="token_ks",
            dim=list(token_ks.shape),
            stride=list(token_ks.stride()),
            data_type=cudnn.data_type.INT32,
            uid=_UIDs.TOKEN_KS,
        )

    # -- MoE Grouped Matmul --
    cudnn_mode = _get_mode_mapping()[mode]

    output_t = graph.moe_grouped_matmul(
        token=token_t,
        weight=weight_t,
        first_token_offset=fto_t,
        token_index=token_index_t,
        token_ks=token_ks_t,
        mode=cudnn_mode,
        compute_data_type=compute_dtype,
        top_k=top_k,
        name="moe",
    )

    # -- Output shape --
    # For GATHER mode: output M dim = token_index dim[1]
    # For NONE/SCATTER mode: output M dim = token dim[1]
    if mode == "gather" and token_index is not None:
        M_out = token_index.shape[1]
    else:
        M_out = token.shape[1]

    o_shape = [1, M_out, N]
    o_stride = [M_out * N, N, 1]

    output_t.set_output(True).set_data_type(io_dtype)
    output_t.set_uid(_UIDs.OUTPUT).set_dim(o_shape).set_stride(o_stride)

    # -- Build --
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace_size = graph.get_workspace_size()

    return graph, workspace_size


# ---------------------------------------------------------------------------
# Custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("cudnn::moe_grouped_matmul", mutates_args=())
def _moe_grouped_matmul_op(
    token: torch.Tensor,
    weight: torch.Tensor,
    first_token_offset: torch.Tensor,
    token_index: Optional[torch.Tensor] = None,
    token_ks: Optional[torch.Tensor] = None,
    mode: str = "none",
    top_k: int = 1,
) -> torch.Tensor:
    """
    cuDNN MoE Grouped Matmul (internal).

    Args:
        token: Token tensor (1, M, K)
        weight: Weight tensor (num_experts, K, N) -- row-major, transposed internally
        first_token_offset: First token offset (batch_size * num_experts, 1, 1) INT32
        token_index: Optional token index (1, num_tokens, 1) INT32, for GATHER/SCATTER modes
        token_ks: Optional token ks (1, num_tokens, 1) INT32, for SCATTER mode
        mode: "none", "gather", or "scatter"
        top_k: Top-k value for SCATTER mode

    Returns:
        Output tensor (1, M_out, N)
    """
    if mode not in _get_mode_mapping():
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: none, gather, scatter")
    if mode == "scatter" and (token_index is None or token_ks is None):
        raise ValueError("SCATTER mode requires both token_index and token_ks")
    if mode == "gather" and token_index is None:
        raise ValueError("GATHER mode requires token_index")

    handle = _get_handle(token.device)

    cache_key = _make_cache_key(
        token,
        weight,
        first_token_offset,
        token_index is not None,
        tuple(token_index.shape) if token_index is not None else None,
        token_ks is not None,
        tuple(token_ks.shape) if token_ks is not None else None,
        mode,
        top_k,
    )

    if cache_key not in _graph_cache:
        graph, workspace_size = _build_graph(
            handle,
            token,
            weight,
            first_token_offset,
            token_index,
            token_ks,
            mode,
            top_k,
        )
        _graph_cache[cache_key] = (graph, workspace_size)

    graph, workspace_size = _graph_cache[cache_key]

    # Allocate output
    _, _, N = weight.shape
    if mode == "gather" and token_index is not None:
        M_out = token_index.shape[1]
    else:
        M_out = token.shape[1]

    output_gpu = torch.empty(1, M_out, N, dtype=token.dtype, device=token.device)

    # Variant pack (UID -> tensor). Use int() since cudnn._execute checks `type(x) is int`.
    variant_pack = {
        int(_UIDs.TOKEN): token,
        int(_UIDs.WEIGHT): weight,
        int(_UIDs.FIRST_TOKEN_OFFSET): first_token_offset,
        int(_UIDs.OUTPUT): output_gpu,
    }
    if token_index is not None:
        variant_pack[int(_UIDs.TOKEN_INDEX)] = token_index
    if token_ks is not None:
        variant_pack[int(_UIDs.TOKEN_KS)] = token_ks

    workspace = torch.empty(workspace_size, device=token.device, dtype=torch.uint8)
    graph.execute(variant_pack, workspace, handle=handle)

    return output_gpu


@_moe_grouped_matmul_op.register_fake
def _moe_grouped_matmul_fake(
    token: torch.Tensor,
    weight: torch.Tensor,
    first_token_offset: torch.Tensor,
    token_index: Optional[torch.Tensor] = None,
    token_ks: Optional[torch.Tensor] = None,
    mode: str = "none",
    top_k: int = 1,
) -> torch.Tensor:
    _, _, N = weight.shape
    if mode == "gather" and token_index is not None:
        M_out = token_index.shape[1]
    else:
        M_out = token.shape[1]
    return torch.empty(1, M_out, N, dtype=token.dtype, device=token.device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def moe_grouped_matmul(
    token: torch.Tensor,
    weight: torch.Tensor,
    first_token_offset: torch.Tensor,
    token_index: Optional[torch.Tensor] = None,
    token_ks: Optional[torch.Tensor] = None,
    mode: str = "none",
    top_k: int = 1,
) -> torch.Tensor:
    """cuDNN-accelerated MoE Grouped Matmul.

    Performs a grouped matrix multiplication across experts, as used in
    Mixture-of-Experts (MoE) layers. Each expert has its own weight matrix,
    and tokens are routed to experts via ``first_token_offset``.

    Args:
        token: Token tensor ``(1, M, K)`` where ``M = batch_size * token_num * top_k``
            and ``K = hidden_size``.
        weight: Weight tensor ``(num_experts, K, N)``
        first_token_offset: Expert routing offsets ``(batch_size * num_experts, 1, 1)``
            INT32 tensor indicating the starting token index for each expert.
        token_index: Optional token index ``(1, num_tokens, 1)`` INT32 tensor.
            Required for ``"gather"`` and ``"scatter"`` modes.
        token_ks: Optional token ks ``(1, num_tokens, 1)`` INT32 tensor.
            Required for ``"scatter"`` mode.
        mode: Routing mode. One of:
            - ``"none"``: Direct grouped matmul (tokens already routed).
            - ``"gather"``: Gather tokens before matmul using ``token_index``.
            - ``"scatter"``: Scatter tokens after matmul using ``token_index``
              and ``token_ks``.
        top_k: Top-k routing value. Must be set when ``mode="scatter"``.
            Defaults to ``1``.

    Returns:
        Output tensor ``(1, M_out, N)`` where ``M_out`` depends on the mode:
        for ``"gather"`` mode it equals ``token_index.shape[1]``, otherwise
        it equals ``token.shape[1]``.
    """
    return _moe_grouped_matmul_op(
        token,
        weight,
        first_token_offset,
        token_index=token_index,
        token_ks=token_ks,
        mode=mode,
        top_k=top_k,
    )
