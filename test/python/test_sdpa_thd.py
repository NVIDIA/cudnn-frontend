"""
Test for SDPA with dynamic shapes and THD (Token-Head-Dimension) layout.

THD layout is a ragged/packed format where:
- Q: [total_q_tokens, num_heads, head_dim] - packed Q tensor
- K/V: can be BHSD or THD format
- O: [total_q_tokens, num_heads, head_dim] - packed output tensor

This is similar to FlashInfer's cuDNN prefill implementation.

The recommended way to run tests:
> pytest -vv -s -rA test_sdpa_dynamic_shapes.py
"""

import cudnn
import pytest
import torch
import math
from looseversion import LooseVersion
from dataclasses import dataclass
from typing import List, Optional, Tuple
from test_utils import torch_fork_set_rng

# =========================================
# Helper Functions and Data Classes
# =========================================

from enum import Enum, auto


class UIDs(Enum):
    Q_UID = auto()
    K_UID = auto()
    V_UID = auto()
    O_UID = auto()
    RAGGED_Q_UID = auto()
    RAGGED_O_UID = auto()
    ACTUAL_SEQ_LENS_Q_UID = auto()
    ACTUAL_SEQ_LENS_KV_UID = auto()


@dataclass
class SDPAConfig:
    """Configuration for SDPA test."""

    batch_size: int
    num_heads_q: int
    num_heads_k: int
    num_heads_v: int
    head_dim_qk: int
    head_dim_v: int
    max_seq_len_q: int
    max_seq_len_kv: int
    dtype: torch.dtype = torch.bfloat16
    is_causal: bool = False
    attn_scale: Optional[float] = None

    def __post_init__(self):
        if self.attn_scale is None:
            self.attn_scale = 1.0 / math.sqrt(self.head_dim_qk)


def convert_to_cudnn_type(torch_type: torch.dtype) -> cudnn.data_type:
    """Convert PyTorch dtype to cuDNN data type."""
    type_map = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
    }
    if torch_type not in type_map:
        raise ValueError(f"Unsupported tensor data type: {torch_type}")
    return type_map[torch_type]


def generate_variable_seq_lens(
    batch_size: int,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate variable sequence lengths for Q and KV."""
    # Generate random sequence lengths, ensuring seq_len_q <= seq_len_kv
    seq_len_q = torch.randint(1, max_seq_len_q + 1, (batch_size,), generator=rng, dtype=torch.int32, device="cuda")
    seq_len_kv = torch.randint(1, max_seq_len_kv + 1, (batch_size,), generator=rng, dtype=torch.int32, device="cuda")

    # Ensure seq_len_q <= seq_len_kv for each batch
    seq_len_q = torch.minimum(seq_len_q, seq_len_kv)

    return seq_len_q, seq_len_kv


def compute_ragged_offsets(
    seq_lens: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Compute exclusive prefix sum (ragged offsets) for THD layout.

    For THD layout, the ragged offset for batch i is the cumulative sum of
    (seq_len[0:i] * num_heads * head_dim).

    Args:
        seq_lens: [batch_size] - sequence lengths
        num_heads: number of attention heads
        head_dim: dimension per head

    Returns:
        ragged_offset: [batch_size + 1, 1, 1, 1] - exclusive prefix sum
    """
    batch_size = seq_lens.shape[0]

    # Compute element counts per batch
    elements_per_batch = seq_lens * num_heads * head_dim

    # Exclusive prefix sum: [0, elem0, elem0+elem1, ...]
    ragged_offset = torch.zeros(batch_size + 1, dtype=torch.int64, device=seq_lens.device)
    ragged_offset[1:] = torch.cumsum(elements_per_batch, dim=0)

    # Reshape to [batch_size+1, 1, 1, 1] as expected by cuDNN
    ragged_offset = ragged_offset.view(-1, 1, 1, 1)

    return ragged_offset


def create_thd_tensor(
    seq_lens: torch.Tensor,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rng: torch.Generator,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a THD (Token-Head-Dimension) layout tensor.

    THD layout: [total_tokens, num_heads, head_dim]
    The tensor is packed - all sequences are concatenated.

    Args:
        seq_lens: [batch_size] - sequence length per batch
        num_heads: number of attention heads
        head_dim: dimension per head
        dtype: tensor data type
        rng: random number generator
        mean: mean for random initialization
        std: std for random initialization

    Returns:
        tensor: [total_tokens, num_heads, head_dim]
        ragged_offset: [batch_size+1, 1, 1, 1]
    """
    total_tokens = int(seq_lens.sum().item())

    # Create the packed tensor
    tensor = torch.empty(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    tensor.normal_(mean=mean, std=std, generator=rng)

    # Compute ragged offsets
    ragged_offset = compute_ragged_offsets(seq_lens, num_heads, head_dim)

    return tensor, ragged_offset


def create_bhsd_tensor(
    batch_size: int,
    num_heads: int,
    max_seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    rng: torch.Generator,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """
    Create a BHSD (Batch-Head-Seq-Dim) shape tensor with BSHD strides.

    Shape: [batch_size, num_heads, max_seq_len, head_dim]
    Strides: [seq*heads*dim, dim, heads*dim, 1] (BSHD stride order)

    This is an interleaved format where memory is laid out as BSHD.

    Args:
        batch_size: batch size
        num_heads: number of attention heads
        max_seq_len: maximum sequence length
        head_dim: dimension per head
        dtype: tensor data type
        rng: random number generator
        mean: mean for random initialization
        std: std for random initialization

    Returns:
        tensor: [batch_size, num_heads, max_seq_len, head_dim] with BSHD strides
    """
    # Allocate contiguous storage in BSHD order
    total_elements = batch_size * max_seq_len * num_heads * head_dim
    storage = torch.empty(total_elements, dtype=dtype, device="cuda")
    storage.normal_(mean=mean, std=std, generator=rng)

    # Create view with BHSD shape but BSHD strides
    # Strides: [S*H*D, D, H*D, 1]
    strides = (
        max_seq_len * num_heads * head_dim,  # batch stride
        head_dim,  # head stride
        num_heads * head_dim,  # seq stride
        1,  # dim stride
    )
    tensor = torch.as_strided(storage, (batch_size, num_heads, max_seq_len, head_dim), strides)
    return tensor


def thd_to_bhsd(
    thd_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """
    Convert THD layout tensor to BHSD shape with BSHD strides (for reference computation).

    THD layout: [total_tokens, num_heads, head_dim] with strides [H*D, D, 1]
    BHSD with BSHD strides: [B, H, S, D] with strides [S*H*D, D, H*D, 1]

    Both layouts have the same underlying memory pattern per batch:
    - THD[t, h, d] -> memory[t*H*D + h*D + d]
    - BHSD[b, h, s, d] with BSHD strides -> memory[b*S*H*D + s*H*D + h*D + d]

    So within a batch, the relative offsets are identical - no transpose needed!

    Args:
        thd_tensor: [total_tokens, num_heads, head_dim]
        seq_lens: [batch_size] - sequence length per batch
        max_seq_len: maximum sequence length for padding

    Returns:
        bhsd_tensor: [batch_size, num_heads, max_seq_len, head_dim] with BSHD strides
    """
    batch_size = seq_lens.shape[0]
    _, num_heads, head_dim = thd_tensor.shape

    # Allocate storage in BSHD physical order: [B, S, H, D] contiguous
    storage = torch.zeros(batch_size, max_seq_len, num_heads, head_dim, dtype=thd_tensor.dtype, device=thd_tensor.device)

    # Copy data batch by batch using direct copy - no transpose needed!
    # THD [seq, H, D] has same memory layout as BSHD [B, S, H, D] within each batch
    offset = 0
    for i in range(batch_size):
        seq_len = int(seq_lens[i].item())
        # Direct copy: THD [seq_len, H, D] -> BSHD storage[i, :seq_len, H, D]
        storage[i, :seq_len, :, :] = thd_tensor[offset : offset + seq_len]
        offset += seq_len

    # Create BHSD view with BSHD strides from the BSHD storage
    # storage is [B, S, H, D] contiguous, we want [B, H, S, D] view
    bhsd_tensor = storage.permute(0, 2, 1, 3)  # [B, S, H, D] -> [B, H, S, D]

    return bhsd_tensor


def bhsd_to_thd(
    bhsd_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Convert BHSD shape tensor (with BSHD strides) to THD layout.

    BHSD with BSHD strides: [B, H, S, D] with strides [S*H*D, D, H*D, 1]
    THD layout: [total_tokens, num_heads, head_dim] with strides [H*D, D, 1]

    Both layouts have the same underlying memory pattern per batch - no transpose needed!

    Args:
        bhsd_tensor: [batch_size, num_heads, max_seq_len, head_dim] with BSHD strides
        seq_lens: [batch_size] - sequence length per batch

    Returns:
        thd_tensor: [total_tokens, num_heads, head_dim]
    """
    batch_size = seq_lens.shape[0]
    _, num_heads, _, head_dim = bhsd_tensor.shape
    total_tokens = int(seq_lens.sum().item())

    # Create output tensor
    thd_tensor = torch.empty(total_tokens, num_heads, head_dim, dtype=bhsd_tensor.dtype, device=bhsd_tensor.device)

    # Convert BHSD [B, H, S, D] back to BSHD view [B, S, H, D] for direct copy
    bshd_tensor = bhsd_tensor.permute(0, 2, 1, 3)  # [B, H, S, D] -> [B, S, H, D]

    # Copy data batch by batch - no transpose needed!
    # BSHD[b, s, h, d] has same memory layout as THD[t, h, d]
    offset = 0
    for i in range(batch_size):
        seq_len = int(seq_lens[i].item())
        # Direct copy: BSHD [1, seq_len, H, D] -> THD [seq_len, H, D]
        thd_tensor[offset : offset + seq_len] = bshd_tensor[i, :seq_len, :, :]
        offset += seq_len

    return thd_tensor


# =========================================
# Reference Implementation
# =========================================


def compute_sdpa_reference(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    seq_len_q: torch.Tensor,
    seq_len_kv: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Compute SDPA reference output in float32.

    Args:
        q_bhsd: [batch, heads_q, max_seq_q, head_dim_qk]
        k_bhsd: [batch, heads_k, max_seq_kv, head_dim_qk]
        v_bhsd: [batch, heads_v, max_seq_kv, head_dim_v]
        seq_len_q: [batch] - actual sequence lengths for Q
        seq_len_kv: [batch] - actual sequence lengths for K/V
        attn_scale: attention scaling factor
        is_causal: whether to apply causal masking

    Returns:
        o_bhsd: [batch, heads_q, max_seq_q, head_dim_v]
    """
    batch_size, num_heads_q, max_seq_q, head_dim_qk = q_bhsd.shape
    _, num_heads_k, max_seq_kv, _ = k_bhsd.shape
    _, num_heads_v, _, head_dim_v = v_bhsd.shape

    # Convert to float32 for reference computation
    q = q_bhsd.to(dtype=torch.float32)
    k = k_bhsd.to(dtype=torch.float32)
    v = v_bhsd.to(dtype=torch.float32)

    # Handle GQA/MQA by expanding K and V
    if num_heads_q != num_heads_k:
        assert num_heads_q % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k"
        k = k.unsqueeze(2).expand(-1, -1, num_heads_q // num_heads_k, -1, -1)
        k = k.reshape(batch_size, num_heads_q, max_seq_kv, head_dim_qk)
    if num_heads_q != num_heads_v:
        assert num_heads_q % num_heads_v == 0, "num_heads_q must be divisible by num_heads_v"
        v = v.unsqueeze(2).expand(-1, -1, num_heads_q // num_heads_v, -1, -1)
        v = v.reshape(batch_size, num_heads_q, max_seq_kv, head_dim_v)

    # Compute attention scores: [batch, heads, seq_q, seq_kv]
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * attn_scale

    # Create padding mask
    device = q.device
    q_mask = torch.zeros(batch_size, 1, max_seq_q, 1, dtype=torch.bool, device=device)
    kv_mask = torch.zeros(batch_size, 1, 1, max_seq_kv, dtype=torch.bool, device=device)
    for i in range(batch_size):
        q_mask[i, :, seq_len_q[i] :, :] = True
        kv_mask[i, :, :, seq_len_kv[i] :] = True

    # Apply padding mask
    scores = scores.masked_fill(kv_mask, float("-inf"))

    # Apply causal mask if requested
    if is_causal:
        causal_mask = torch.ones(max_seq_q, max_seq_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Mask out padded Q positions (set to 0)
    attn_weights = attn_weights.masked_fill(q_mask, 0.0)

    # Compute output
    output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)

    # Zero out padded positions in output
    output_mask = torch.zeros(batch_size, 1, max_seq_q, 1, dtype=torch.bool, device=device)
    for i in range(batch_size):
        output_mask[i, :, seq_len_q[i] :, :] = True
    output = output.masked_fill(output_mask, 0.0)

    return output


# =========================================
# cuDNN Graph Builder
# =========================================

graph_cache = {}


def lookup_graph_from_cache(batch_size: int, h_q: int, h_k: int, h_v: int, d_qk: int, d_v: int, max_s_kv: int, causal: bool) -> cudnn.pygraph:
    """
    Lookup a graph from the cuDNN graph cache.
    """
    key = (batch_size, h_q, h_k, h_v, d_qk, d_v, max_s_kv, causal)

    if key in graph_cache:
        return graph_cache[key]

    return None


def add_to_cudnn_graph_cache(batch_size: int, h_q: int, h_k: int, h_v: int, d_qk: int, d_v: int, max_s_kv: int, causal: bool, graph: cudnn.pygraph) -> None:
    """
    Add a graph to the cuDNN graph cache.
    """

    key = (batch_size, h_q, h_k, h_v, d_qk, d_v, max_s_kv, causal)

    graph_cache[key] = graph
    return None


def build_cudnn_sdpa_thd_graph(
    cudnn_handle,
    config: SDPAConfig,
    seq_len_q: torch.Tensor,
    seq_len_kv: torch.Tensor,
    q_ragged_offset: torch.Tensor,
    o_ragged_offset: torch.Tensor,
    q_gpu: torch.Tensor,
    k_gpu: torch.Tensor,
    v_gpu: torch.Tensor,
    o_gpu: torch.Tensor,
):
    """
    Build cuDNN graph for SDPA with THD layout for Q and O.

    Q and O are in THD (Token-Head-Dimension) ragged layout.
    K and V are in BHSD (Batch-Head-Seq-Dim) layout.
    """
    batch_size = config.batch_size
    h_q = config.num_heads_q
    h_k = config.num_heads_k
    h_v = config.num_heads_v
    d_qk = config.head_dim_qk
    d_v = config.head_dim_v
    max_s_q = config.max_seq_len_q
    max_s_kv = config.max_seq_len_kv

    cudnn_dtype = convert_to_cudnn_type(config.dtype)

    # Look up pre-built graph from cache

    graph = lookup_graph_from_cache(batch_size, h_q, h_k, h_v, d_qk, d_v, max_s_kv, config.is_causal)

    if graph is not None:
        print("Returning existing graph since it already exists")
        return graph

    # Create the graph
    graph = cudnn.pygraph(
        io_data_type=cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
        is_dynamic_shape_enabled=True,
    )

    # Q tensor in THD layout with BHSD logical shape
    # Physical shape: [total_q_tokens, h_q, d_qk]
    # Logical shape for cuDNN: [batch, heads, max_seq_q, head_dim]
    # Stride: THD -> [h_q * d_qk, d_qk, h_q * d_qk, 1] (bshd stride order)
    q = graph.tensor(
        dim=(batch_size, h_q, max_s_q, d_qk),
        stride=(h_q * d_qk, d_qk, h_q * d_qk, 1),  # bshd stride order for THD
        data_type=cudnn_dtype,
        name="Q",
        uid=UIDs.Q_UID.value,
    )

    # Q ragged offset tensor
    q_ragged = graph.tensor(
        dim=(batch_size + 1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT64,
        name="Q_ragged_offset",
        uid=UIDs.RAGGED_Q_UID.value,
    )
    q.set_ragged_offset(q_ragged)

    # K tensor in BHSD layout
    k = graph.tensor(
        dim=(batch_size, h_k, max_s_kv, d_qk),
        stride=(h_k * max_s_kv * d_qk, d_qk, h_k * d_qk, 1),  # bshd stride order
        data_type=cudnn_dtype,
        name="K",
        uid=UIDs.K_UID.value,
    )

    # V tensor in BHSD layout
    v = graph.tensor(
        dim=(batch_size, h_v, max_s_kv, d_v),
        stride=(h_v * max_s_kv * d_v, d_v, h_v * d_v, 1),  # bshd stride order
        data_type=cudnn_dtype,
        name="V",
        uid=UIDs.V_UID.value,
    )

    # Sequence length tensors
    seq_len_q_tensor = graph.tensor(
        dim=(batch_size, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT32,
        name="seq_len_q",
        uid=UIDs.ACTUAL_SEQ_LENS_Q_UID.value,
    )

    seq_len_kv_tensor = graph.tensor(
        dim=(batch_size, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT32,
        name="seq_len_kv",
        uid=UIDs.ACTUAL_SEQ_LENS_KV_UID.value,
    )

    # Call SDPA
    o, stats = graph.sdpa(
        name="sdpa_thd",
        q=q,
        k=k,
        v=v,
        attn_scale=config.attn_scale,
        use_padding_mask=True,
        seq_len_q=seq_len_q_tensor,
        seq_len_kv=seq_len_kv_tensor,
        use_causal_mask=config.is_causal,
        generate_stats=False,
    )

    # Output tensor in THD layout
    o.set_output(True).set_dim((batch_size, h_q, max_s_q, d_v)).set_stride((h_q * d_v, d_v, h_q * d_v, 1)).set_data_type(  # bshd stride order for THD
        cudnn_dtype
    )
    o.set_uid(UIDs.O_UID.value)

    # O ragged offset tensor (reuse Q's ragged offset structure for d_qk == d_v)
    o_ragged = graph.tensor(
        dim=(batch_size + 1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT64,
        name="O_ragged_offset",
        uid=UIDs.RAGGED_O_UID.value,
    )
    o.set_ragged_offset(o_ragged)

    # Validate and build the graph
    try:
        graph.validate()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"Graph not supported: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during graph validation: {e}")

    try:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"Graph not supported after validation: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error after graph validation: {e}")

    add_to_cudnn_graph_cache(batch_size, h_q, h_k, h_v, d_qk, d_v, max_s_kv, config.is_causal, graph)

    return graph


def execute_cudnn_sdpa_thd(
    cudnn_handle,
    config: SDPAConfig,
    q_gpu: torch.Tensor,
    k_gpu: torch.Tensor,
    v_gpu: torch.Tensor,
    seq_len_q: torch.Tensor,
    seq_len_kv: torch.Tensor,
    q_ragged_offset: torch.Tensor,
    o_ragged_offset: torch.Tensor,
) -> torch.Tensor:
    """
    Execute cuDNN SDPA with THD layout.

    Args:
        cudnn_handle: cuDNN handle
        config: SDPA configuration
        q_gpu: [total_q_tokens, num_heads_q, head_dim_qk] - Q in THD layout
        k_gpu: [batch, num_heads_k, max_seq_kv, head_dim_qk] - K in BHSD layout
        v_gpu: [batch, num_heads_v, max_seq_kv, head_dim_v] - V in BHSD layout
        seq_len_q: [batch] - actual Q sequence lengths
        seq_len_kv: [batch] - actual KV sequence lengths
        q_ragged_offset: [batch+1, 1, 1, 1] - Q ragged offsets
        o_ragged_offset: [batch+1, 1, 1, 1] - O ragged offsets

    Returns:
        o_gpu: [total_q_tokens, num_heads_q, head_dim_v] - output in THD layout
    """
    total_q_tokens = q_gpu.shape[0]

    # Allocate output tensor
    o_gpu = torch.empty(total_q_tokens, config.num_heads_q, config.head_dim_v, dtype=config.dtype, device="cuda")

    # Reshape seq_len tensors to [batch, 1, 1, 1]
    seq_len_q_4d = seq_len_q.view(-1, 1, 1, 1)
    seq_len_kv_4d = seq_len_kv.view(-1, 1, 1, 1)

    # Build the graph
    graph = build_cudnn_sdpa_thd_graph(cudnn_handle, config, seq_len_q, seq_len_kv, q_ragged_offset, o_ragged_offset, q_gpu, k_gpu, v_gpu, o_gpu)

    q_shape = [config.batch_size, config.num_heads_q, config.max_seq_len_q, config.head_dim_qk]
    o_shape = [config.batch_size, config.num_heads_q, config.max_seq_len_q, config.head_dim_v]
    # Create variant pack
    variant_pack = {
        UIDs.Q_UID.value: q_gpu,
        UIDs.RAGGED_Q_UID.value: q_ragged_offset,
        UIDs.K_UID.value: k_gpu,
        UIDs.V_UID.value: v_gpu,
        UIDs.ACTUAL_SEQ_LENS_Q_UID.value: seq_len_q_4d,
        UIDs.ACTUAL_SEQ_LENS_KV_UID.value: seq_len_kv_4d,
        UIDs.O_UID.value: o_gpu,
        UIDs.RAGGED_O_UID.value: o_ragged_offset,
    }

    # Allocate workspace
    workspace_size = graph.get_workspace_size()
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")

    # Execute
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)
    override_uids = [
        UIDs.Q_UID.value,
        UIDs.RAGGED_Q_UID.value,
        UIDs.K_UID.value,
        UIDs.V_UID.value,
        UIDs.ACTUAL_SEQ_LENS_Q_UID.value,
        UIDs.ACTUAL_SEQ_LENS_KV_UID.value,
        UIDs.O_UID.value,
        UIDs.RAGGED_O_UID.value,
    ]
    override_shapes = [
        q_shape,
        q_ragged_offset.shape,
        k_gpu.shape,
        v_gpu.shape,
        seq_len_q_4d.shape,
        seq_len_kv_4d.shape,
        o_shape,
        o_ragged_offset.shape,
    ]
    override_strides = [
        q_gpu.stride(),
        q_ragged_offset.stride(),
        k_gpu.stride(),
        v_gpu.stride(),
        seq_len_q_4d.stride(),
        seq_len_kv_4d.stride(),
        o_gpu.stride(),
        o_ragged_offset.stride(),
    ]
    graph.execute(variant_pack, workspace, handle=cudnn_handle, override_uids=override_uids, override_shapes=override_shapes, override_strides=override_strides)
    torch.cuda.synchronize()

    return o_gpu


# =========================================
# Test Functions
# =========================================


def compare_outputs(
    output_gpu: torch.Tensor,
    output_ref: torch.Tensor,
    seq_lens: torch.Tensor,
    atol: float = 0.02,
    rtol: float = 0.02,
    tag: str = "output",
) -> int:
    """
    Compare GPU output with reference, accounting for padding.

    Returns number of mismatches.
    """
    # Convert THD output to BHSD for comparison if needed
    if output_gpu.dim() == 3:  # THD layout
        # Both should be in THD already for this comparison
        actual = output_gpu.float()
        expected = output_ref.float()
    else:
        actual = output_gpu.float()
        expected = output_ref.float()

    mismatches = torch.where(torch.isclose(actual, expected, rtol=rtol, atol=atol) == False)
    mismatch_cnt = mismatches[0].numel()

    if mismatch_cnt > 0:
        percentage = 100 * mismatch_cnt / actual.numel()
        print(f"\n{tag}: {mismatch_cnt:,} mismatches ({percentage:.2f}%)")

        # Show first few mismatches
        for idx in range(min(10, mismatch_cnt)):
            pos = tuple(m[idx].item() for m in mismatches)
            diff = actual[pos] - expected[pos]
            print(f"  idx{pos}: gpu={actual[pos]:+.6e}, ref={expected[pos]:+.6e}, diff={diff:+.2e}")
    else:
        print(f"{tag}: All values match within tolerance (atol={atol}, rtol={rtol})")

    return mismatch_cnt


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
def test_sdpa_thd_dynamic_shapes(cudnn_handle):
    """Basic test for SDPA with THD layout."""
    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0 or higher")

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SDPA with THD layout requires SM90 or higher")

    print("\n" + "=" * 80)
    print("Test: SDPA with THD layout (basic)")
    print("=" * 80)

    # Configurations
    configs = [
        SDPAConfig(
            batch_size=4,
            num_heads_q=8,
            num_heads_k=8,
            num_heads_v=8,
            head_dim_qk=128,
            head_dim_v=128,
            max_seq_len_q=256,
            max_seq_len_kv=512,
            dtype=torch.bfloat16,
            is_causal=False,
        ),
        SDPAConfig(
            batch_size=4,
            num_heads_q=8,
            num_heads_k=8,
            num_heads_v=8,
            head_dim_qk=128,
            head_dim_v=128,
            max_seq_len_q=512,
            max_seq_len_kv=512,
            dtype=torch.bfloat16,
            is_causal=False,
        ),
        SDPAConfig(
            batch_size=4,
            num_heads_q=8,
            num_heads_k=8,
            num_heads_v=8,
            head_dim_qk=128,
            head_dim_v=128,
            max_seq_len_q=384,
            max_seq_len_kv=512,
            dtype=torch.bfloat16,
            is_causal=False,
        ),
    ]

    rng = torch.Generator(device="cuda").manual_seed(42)
    for config in configs:
        print(
            f"Config: batch={config.batch_size}, h_q={config.num_heads_q}, "
            f"h_k={config.num_heads_k}, h_v={config.num_heads_v}, "
            f"d_qk={config.head_dim_qk}, d_v={config.head_dim_v}, "
            f"max_s_q={config.max_seq_len_q}, max_s_kv={config.max_seq_len_kv}"
        )

        # Generate variable sequence lengths
        seq_len_q, seq_len_kv = generate_variable_seq_lens(config.batch_size, config.max_seq_len_q, config.max_seq_len_kv, rng)
        print(f"seq_len_q: {seq_len_q.tolist()}")
        print(f"seq_len_kv: {seq_len_kv.tolist()}")

        # Create Q in THD layout
        q_thd, q_ragged_offset = create_thd_tensor(seq_len_q, config.num_heads_q, config.head_dim_qk, config.dtype, rng)
        print(f"Q shape (THD): {q_thd.shape}")

        # Create K, V in BHSD layout
        k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.max_seq_len_kv, config.head_dim_qk, config.dtype, rng)
        v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.max_seq_len_kv, config.head_dim_v, config.dtype, rng)
        print(f"K shape (BHSD): {k_bhsd.shape}")
        print(f"V shape (BHSD): {v_bhsd.shape}")

        # Compute O ragged offsets (same structure as Q for d_qk == d_v)
        o_ragged_offset = compute_ragged_offsets(seq_len_q, config.num_heads_q, config.head_dim_v)

        # Execute cuDNN SDPA
        print("\nExecuting cuDNN SDPA with THD layout...")
        o_thd_gpu = execute_cudnn_sdpa_thd(cudnn_handle, config, q_thd, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, q_ragged_offset, o_ragged_offset)
        print(f"Output shape (THD): {o_thd_gpu.shape}")

        # Compute reference
        print("\nComputing reference output...")
        # Convert Q from THD to BHSD for reference
        q_bhsd_ref = thd_to_bhsd(q_thd, seq_len_q, config.max_seq_len_q)

        # print(f"Q BHSD shape: {q_bhsd_ref.shape} {q_bhsd_ref[0, :, :, 0:10]}")
        # print(f"Q  THD shape: {q_thd.shape} {q_thd[0, :, 0:10]}")

        o_bhsd_ref = compute_sdpa_reference(q_bhsd_ref, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, config.attn_scale, config.is_causal)

        # Convert reference output from BHSD to THD
        o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_len_q)

        # Compare outputs
        print("\nComparing outputs...")
        err_count = compare_outputs(o_thd_gpu, o_thd_ref, seq_len_q, atol=0.02, rtol=0.02)

        if err_count > 0:
            pytest.fail(f"SDPA THD test failed with {err_count} mismatches")
        else:
            print("\n" + "=" * 80)
            print("TEST PASSED: SDPA with THD layout")
            print("=" * 80)


@pytest.mark.L0
@torch_fork_set_rng(seed=123)
def test_sdpa_thd_gqa(cudnn_handle):
    """Test SDPA with THD layout and GQA (Grouped Query Attention)."""
    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0 or higher")

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SDPA with THD layout requires SM80 or higher")

    print("\n" + "=" * 80)
    print("Test: SDPA with THD layout + GQA")
    print("=" * 80)

    # Configuration with GQA (h_q > h_k = h_v)
    config = SDPAConfig(
        batch_size=4,
        num_heads_q=8,
        num_heads_k=2,  # GQA: 4 Q heads per K head
        num_heads_v=2,
        head_dim_qk=64,
        head_dim_v=64,
        max_seq_len_q=128,
        max_seq_len_kv=256,
        dtype=torch.bfloat16,
        is_causal=False,
    )

    print(
        f"Config: batch={config.batch_size}, h_q={config.num_heads_q}, "
        f"h_k={config.num_heads_k}, h_v={config.num_heads_v}, "
        f"d_qk={config.head_dim_qk}, d_v={config.head_dim_v}"
    )

    rng = torch.Generator(device="cuda").manual_seed(123)

    # Generate variable sequence lengths
    seq_len_q, seq_len_kv = generate_variable_seq_lens(config.batch_size, config.max_seq_len_q, config.max_seq_len_kv, rng)
    print(f"seq_len_q: {seq_len_q.tolist()}")
    print(f"seq_len_kv: {seq_len_kv.tolist()}")

    # Create Q in THD layout
    q_thd, q_ragged_offset = create_thd_tensor(seq_len_q, config.num_heads_q, config.head_dim_qk, config.dtype, rng)

    # Create K, V in BHSD layout
    k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.max_seq_len_kv, config.head_dim_qk, config.dtype, rng)
    v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.max_seq_len_kv, config.head_dim_v, config.dtype, rng)

    # Compute O ragged offsets
    o_ragged_offset = compute_ragged_offsets(seq_len_q, config.num_heads_q, config.head_dim_v)

    # Execute cuDNN SDPA
    print("\nExecuting cuDNN SDPA with THD + GQA...")
    o_thd_gpu = execute_cudnn_sdpa_thd(cudnn_handle, config, q_thd, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, q_ragged_offset, o_ragged_offset)

    # Compute reference
    print("\nComputing reference output...")
    q_bhsd_ref = thd_to_bhsd(q_thd, seq_len_q, config.max_seq_len_q)
    o_bhsd_ref = compute_sdpa_reference(q_bhsd_ref, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, config.attn_scale, config.is_causal)
    o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_len_q)

    # Compare outputs
    print("\nComparing outputs...")
    err_count = compare_outputs(o_thd_gpu, o_thd_ref, seq_len_q, atol=0.02, rtol=0.02)

    if err_count > 0:
        pytest.fail(f"SDPA THD + GQA test failed with {err_count} mismatches")
    else:
        print("\n" + "=" * 80)
        print("TEST PASSED: SDPA with THD layout + GQA")
        print("=" * 80)


@pytest.mark.L0
@torch_fork_set_rng(seed=456)
def test_sdpa_thd_causal(cudnn_handle):
    """Test SDPA with THD layout and causal masking."""
    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0 or higher")

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SDPA with THD layout requires SM80 or higher")

    print("\n" + "=" * 80)
    print("Test: SDPA with THD layout + Causal Masking")
    print("=" * 80)

    # Configuration
    config = SDPAConfig(
        batch_size=4,
        num_heads_q=8,
        num_heads_k=8,
        num_heads_v=8,
        head_dim_qk=64,
        head_dim_v=64,
        max_seq_len_q=256,
        max_seq_len_kv=256,
        dtype=torch.bfloat16,
        is_causal=True,  # Enable causal masking
    )

    print(f"Config: batch={config.batch_size}, h_q={config.num_heads_q}, " f"is_causal={config.is_causal}")

    rng = torch.Generator(device="cuda").manual_seed(456)

    # Generate variable sequence lengths (ensure seq_len_q == seq_len_kv for causal)
    seq_len_q, seq_len_kv = generate_variable_seq_lens(config.batch_size, config.max_seq_len_q, config.max_seq_len_kv, rng)
    # For causal attention, typically s_q == s_kv
    seq_len_kv = seq_len_q.clone()

    print(f"seq_len_q: {seq_len_q.tolist()}")
    print(f"seq_len_kv: {seq_len_kv.tolist()}")

    # Create tensors
    q_thd, q_ragged_offset = create_thd_tensor(seq_len_q, config.num_heads_q, config.head_dim_qk, config.dtype, rng)
    k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.max_seq_len_kv, config.head_dim_qk, config.dtype, rng)
    v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.max_seq_len_kv, config.head_dim_v, config.dtype, rng)
    o_ragged_offset = compute_ragged_offsets(seq_len_q, config.num_heads_q, config.head_dim_v)

    # Execute cuDNN SDPA
    print("\nExecuting cuDNN SDPA with THD + Causal...")
    o_thd_gpu = execute_cudnn_sdpa_thd(cudnn_handle, config, q_thd, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, q_ragged_offset, o_ragged_offset)

    # Compute reference
    print("\nComputing reference output...")
    q_bhsd_ref = thd_to_bhsd(q_thd, seq_len_q, config.max_seq_len_q)
    o_bhsd_ref = compute_sdpa_reference(q_bhsd_ref, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, config.attn_scale, config.is_causal)
    o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_len_q)

    # Compare outputs
    print("\nComparing outputs...")
    err_count = compare_outputs(o_thd_gpu, o_thd_ref, seq_len_q, atol=0.02, rtol=0.02)

    if err_count > 0:
        pytest.fail(f"SDPA THD + Causal test failed with {err_count} mismatches")
    else:
        print("\n" + "=" * 80)
        print("TEST PASSED: SDPA with THD layout + Causal Masking")
        print("=" * 80)


@pytest.mark.L0
@torch_fork_set_rng(seed=789)
def test_sdpa_thd_seq1(cudnn_handle):
    """Test SDPA with THD layout for seq_len_q=1 (decode-like)."""
    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0 or higher")

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SDPA with THD layout requires SM80 or higher")

    print("\n" + "=" * 80)
    print("Test: SDPA with THD layout (seq_q=1, decode-like)")
    print("=" * 80)

    # Configuration for decode-like scenario (s_q=1)
    config = SDPAConfig(
        batch_size=8,
        num_heads_q=8,
        num_heads_k=8,
        num_heads_v=8,
        head_dim_qk=128,
        head_dim_v=128,
        max_seq_len_q=1,  # Single token query
        max_seq_len_kv=512,  # Long context
        dtype=torch.bfloat16,
        is_causal=False,
    )

    print(
        f"Config: batch={config.batch_size}, h={config.num_heads_q}, "
        f"d={config.head_dim_qk}, max_s_q={config.max_seq_len_q}, max_s_kv={config.max_seq_len_kv}"
    )

    rng = torch.Generator(device="cuda").manual_seed(789)

    # All batches have seq_len_q = 1
    seq_len_q = torch.ones(config.batch_size, dtype=torch.int32, device="cuda")
    # Variable KV lengths
    seq_len_kv = torch.randint(1, config.max_seq_len_kv + 1, (config.batch_size,), generator=rng, dtype=torch.int32, device="cuda")
    print(f"seq_len_q: {seq_len_q.tolist()}")
    print(f"seq_len_kv: {seq_len_kv.tolist()}")

    # Create tensors
    q_thd, q_ragged_offset = create_thd_tensor(seq_len_q, config.num_heads_q, config.head_dim_qk, config.dtype, rng)
    k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.max_seq_len_kv, config.head_dim_qk, config.dtype, rng)
    v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.max_seq_len_kv, config.head_dim_v, config.dtype, rng)
    o_ragged_offset = compute_ragged_offsets(seq_len_q, config.num_heads_q, config.head_dim_v)

    print(f"Q shape (THD): {q_thd.shape}")

    # Execute cuDNN SDPA
    print("\nExecuting cuDNN SDPA with THD (seq_q=1)...")
    o_thd_gpu = execute_cudnn_sdpa_thd(cudnn_handle, config, q_thd, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, q_ragged_offset, o_ragged_offset)

    # Compute reference
    print("\nComputing reference output...")
    q_bhsd_ref = thd_to_bhsd(q_thd, seq_len_q, config.max_seq_len_q)
    o_bhsd_ref = compute_sdpa_reference(q_bhsd_ref, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, config.attn_scale, config.is_causal)
    o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_len_q)

    # Compare outputs
    print("\nComparing outputs...")
    err_count = compare_outputs(o_thd_gpu, o_thd_ref, seq_len_q, atol=0.02, rtol=0.02)

    if err_count > 0:
        pytest.fail(f"SDPA THD seq_q=1 test failed with {err_count} mismatches")
    else:
        print("\n" + "=" * 80)
        print("TEST PASSED: SDPA with THD layout (seq_q=1)")
        print("=" * 80)


@pytest.mark.L0
@torch_fork_set_rng(seed=999)
def test_sdpa_thd_large_batch(cudnn_handle):
    """Test SDPA with THD layout with larger batch sizes."""
    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0 or higher")

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SDPA with THD layout requires SM80 or higher")

    print("\n" + "=" * 80)
    print("Test: SDPA with THD layout (large batch)")
    print("=" * 80)

    # Configuration with larger batch
    config = SDPAConfig(
        batch_size=32,
        num_heads_q=8,
        num_heads_k=2,  # GQA
        num_heads_v=2,
        head_dim_qk=128,
        head_dim_v=128,
        max_seq_len_q=512,
        max_seq_len_kv=512,
        dtype=torch.bfloat16,
        is_causal=False,
    )

    print(f"Config: batch={config.batch_size}, h_q={config.num_heads_q}, " f"h_k={config.num_heads_k}, d={config.head_dim_qk}")

    rng = torch.Generator(device="cuda").manual_seed(999)

    # Generate variable sequence lengths
    seq_len_q, seq_len_kv = generate_variable_seq_lens(config.batch_size, config.max_seq_len_q, config.max_seq_len_kv, rng)
    print(f"seq_len_q range: [{seq_len_q.min().item()}, {seq_len_q.max().item()}]")
    print(f"seq_len_kv range: [{seq_len_kv.min().item()}, {seq_len_kv.max().item()}]")
    print(f"Total Q tokens: {seq_len_q.sum().item()}")

    # Create tensors
    q_thd, q_ragged_offset = create_thd_tensor(seq_len_q, config.num_heads_q, config.head_dim_qk, config.dtype, rng)
    k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.max_seq_len_kv, config.head_dim_qk, config.dtype, rng)
    v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.max_seq_len_kv, config.head_dim_v, config.dtype, rng)
    o_ragged_offset = compute_ragged_offsets(seq_len_q, config.num_heads_q, config.head_dim_v)

    # Execute cuDNN SDPA
    print("\nExecuting cuDNN SDPA with THD (large batch)...")
    o_thd_gpu = execute_cudnn_sdpa_thd(cudnn_handle, config, q_thd, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, q_ragged_offset, o_ragged_offset)

    # Compute reference
    print("\nComputing reference output...")
    q_bhsd_ref = thd_to_bhsd(q_thd, seq_len_q, config.max_seq_len_q)
    o_bhsd_ref = compute_sdpa_reference(q_bhsd_ref, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, config.attn_scale, config.is_causal)
    o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_len_q)

    # Compare outputs
    print("\nComparing outputs...")
    err_count = compare_outputs(o_thd_gpu, o_thd_ref, seq_len_q, atol=0.02, rtol=0.02)

    if err_count > 0:
        pytest.fail(f"SDPA THD large batch test failed with {err_count} mismatches")
    else:
        print("\n" + "=" * 80)
        print("TEST PASSED: SDPA with THD layout (large batch)")
        print("=" * 80)


# =========================================
# Main Entry Point
# =========================================

if __name__ == "__main__":
    print("This is a pytest script.")
    print("Run with: pytest -vv -s -rA test_sdpa_dynamic_shapes.py")
