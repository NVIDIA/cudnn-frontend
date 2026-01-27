"""
Test for SDPA with chunked prefill using THD (Token-Head-Dimension) layout.

Chunked prefill processes long sequences by splitting them into smaller chunks
to reduce memory usage. For a sequence of 4096 tokens with chunk_size=1024:
- Chunk 0: Q[0:1024] attends to K[0:1024], V[0:1024]
- Chunk 1: Q[1024:2048] attends to K[0:2048], V[0:2048]
- Chunk 2: Q[2048:3072] attends to K[0:3072], V[0:3072]
- Chunk 3: Q[3072:4096] attends to K[0:4096], V[0:4096]

THD layout is a ragged/packed format where:
- Q: [chunk_tokens, num_heads, head_dim] - packed Q tensor for current chunk
- K/V: BHSD format [batch, heads, accumulated_seq_len, head_dim]
- O: [chunk_tokens, num_heads, head_dim] - packed output tensor

The recommended way to run tests:
> pytest -vv -s -rA test_sdpa_chunked_prefill.py
"""

import cudnn
import pytest
import torch
import math
from looseversion import LooseVersion
from dataclasses import dataclass
from typing import List, Optional, Tuple
from test_utils import torch_fork_set_rng
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
class ChunkedPrefillConfig:
    batch_size: int
    num_heads_q: int
    num_heads_k: int
    num_heads_v: int
    head_dim_qk: int
    head_dim_v: int
    total_seq_len: int
    chunk_size: int
    dtype: torch.dtype = torch.bfloat16
    is_causal: bool = False
    attn_scale: Optional[float] = None

    def __post_init__(self):
        if self.attn_scale is None:
            self.attn_scale = 1.0 / math.sqrt(self.head_dim_qk)
        assert self.total_seq_len % self.chunk_size == 0

    @property
    def num_chunks(self) -> int:
        return self.total_seq_len // self.chunk_size


def convert_to_cudnn_type(torch_type):
    type_map = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
    }
    return type_map[torch_type]


def compute_ragged_offsets(seq_lens, num_heads, head_dim):
    batch_size = seq_lens.shape[0]
    elements_per_batch = seq_lens * num_heads * head_dim
    ragged_offset = torch.zeros(batch_size + 1, dtype=torch.int64, device=seq_lens.device)
    ragged_offset[1:] = torch.cumsum(elements_per_batch, dim=0)
    return ragged_offset.view(-1, 1, 1, 1)


def create_thd_tensor(seq_lens, num_heads, head_dim, dtype, rng, mean=0.0, std=1.0):
    total_tokens = int(seq_lens.sum().item())
    tensor = torch.empty(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    tensor.normal_(mean=mean, std=std, generator=rng)
    ragged_offset = compute_ragged_offsets(seq_lens, num_heads, head_dim)
    return tensor, ragged_offset


def create_bhsd_tensor(batch_size, num_heads, max_seq_len, head_dim, dtype, rng, mean=0.0, std=1.0):
    total_elements = batch_size * max_seq_len * num_heads * head_dim
    storage = torch.empty(total_elements, dtype=dtype, device="cuda")
    storage.normal_(mean=mean, std=std, generator=rng)
    strides = (max_seq_len * num_heads * head_dim, head_dim, num_heads * head_dim, 1)
    return torch.as_strided(storage, (batch_size, num_heads, max_seq_len, head_dim), strides)


def thd_to_bhsd(thd_tensor, seq_lens, max_seq_len):
    batch_size = seq_lens.shape[0]
    _, num_heads, head_dim = thd_tensor.shape
    storage = torch.zeros(batch_size, max_seq_len, num_heads, head_dim, dtype=thd_tensor.dtype, device=thd_tensor.device)
    offset = 0
    for i in range(batch_size):
        seq_len = int(seq_lens[i].item())
        storage[i, :seq_len, :, :] = thd_tensor[offset : offset + seq_len]
        offset += seq_len
    return storage.permute(0, 2, 1, 3)


def bhsd_to_thd(bhsd_tensor, seq_lens):
    batch_size = seq_lens.shape[0]
    _, num_heads, _, head_dim = bhsd_tensor.shape
    total_tokens = int(seq_lens.sum().item())
    thd_tensor = torch.empty(total_tokens, num_heads, head_dim, dtype=bhsd_tensor.dtype, device=bhsd_tensor.device)
    bshd_tensor = bhsd_tensor.permute(0, 2, 1, 3)
    offset = 0
    for i in range(batch_size):
        seq_len = int(seq_lens[i].item())
        thd_tensor[offset : offset + seq_len] = bshd_tensor[i, :seq_len, :, :]
        offset += seq_len
    return thd_tensor


def compute_sdpa_reference_with_offset(q_bhsd, k_bhsd, v_bhsd, seq_len_q, seq_len_kv, attn_scale, is_causal=False, causal_offset=0):
    batch_size, num_heads_q, max_seq_q, head_dim_qk = q_bhsd.shape
    _, num_heads_k, max_seq_kv, _ = k_bhsd.shape
    _, num_heads_v, _, head_dim_v = v_bhsd.shape

    q = q_bhsd.to(dtype=torch.float32)
    k = k_bhsd.to(dtype=torch.float32)
    v = v_bhsd.to(dtype=torch.float32)

    if num_heads_q != num_heads_k:
        k = k.unsqueeze(2).expand(-1, -1, num_heads_q // num_heads_k, -1, -1).reshape(batch_size, num_heads_q, max_seq_kv, head_dim_qk)
    if num_heads_q != num_heads_v:
        v = v.unsqueeze(2).expand(-1, -1, num_heads_q // num_heads_v, -1, -1).reshape(batch_size, num_heads_q, max_seq_kv, head_dim_v)

    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * attn_scale

    device = q.device
    q_mask = torch.zeros(batch_size, 1, max_seq_q, 1, dtype=torch.bool, device=device)
    kv_mask = torch.zeros(batch_size, 1, 1, max_seq_kv, dtype=torch.bool, device=device)
    for i in range(batch_size):
        q_mask[i, :, seq_len_q[i] :, :] = True
        kv_mask[i, :, :, seq_len_kv[i] :] = True

    scores = scores.masked_fill(kv_mask, float("-inf"))

    if is_causal:
        # For chunked prefill, Q position q (in chunk) corresponds to global position (causal_offset + q)
        # Q can attend to K[0:causal_offset+q+1], so mask K[k] when k > causal_offset + q
        # This means k - q > causal_offset, or k - q >= causal_offset + 1
        # triu_(diagonal=d) sets True where (col - row) >= d
        causal_mask = torch.ones(max_seq_q, max_seq_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1 + causal_offset)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = attn_weights.masked_fill(q_mask, 0.0)
    output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)

    output_mask = torch.zeros(batch_size, 1, max_seq_q, 1, dtype=torch.bool, device=device)
    for i in range(batch_size):
        output_mask[i, :, seq_len_q[i] :, :] = True
    return output.masked_fill(output_mask, 0.0)


def compute_chunked_prefill_reference(q_bhsd, k_bhsd, v_bhsd, config, attn_scale):
    batch_size, num_heads_q, total_seq, _ = q_bhsd.shape
    head_dim_v = v_bhsd.shape[3]
    chunk_size = config.chunk_size
    num_chunks = config.num_chunks
    device = q_bhsd.device

    output = torch.zeros(batch_size, num_heads_q, total_seq, head_dim_v, dtype=torch.float32, device=device)

    for chunk_idx in range(num_chunks):
        q_start = chunk_idx * chunk_size
        q_end = q_start + chunk_size
        kv_end = q_end

        q_chunk = q_bhsd[:, :, q_start:q_end, :]
        k_chunk = k_bhsd[:, :, :kv_end, :]
        v_chunk = v_bhsd[:, :, :kv_end, :]

        seq_len_q = torch.full((batch_size,), chunk_size, dtype=torch.int32, device=device)
        seq_len_kv = torch.full((batch_size,), kv_end, dtype=torch.int32, device=device)

        o_chunk = compute_sdpa_reference_with_offset(
            q_chunk, k_chunk, v_chunk, seq_len_q, seq_len_kv, attn_scale, is_causal=config.is_causal, causal_offset=q_start
        )
        output[:, :, q_start:q_end, :] = o_chunk

    return output


graph_cache = {}


def build_cudnn_sdpa_chunk_graph(cudnn_handle, batch_size, h_q, h_k, h_v, d_qk, d_v, chunk_size, kv_seq_len, dtype, attn_scale, is_causal, causal_offset=0):
    cudnn_dtype = convert_to_cudnn_type(dtype)
    cache_key = (batch_size, h_q, h_k, h_v, d_qk, d_v, chunk_size, kv_seq_len, is_causal, causal_offset)

    if cache_key in graph_cache:
        return graph_cache[cache_key]

    graph = cudnn.pygraph(
        io_data_type=cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
        is_dynamic_shape_enabled=True,
    )

    q = graph.tensor(dim=(batch_size, h_q, chunk_size, d_qk), stride=(h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_dtype, name="Q", uid=UIDs.Q_UID.value)
    q_ragged = graph.tensor(
        dim=(batch_size + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="Q_ragged_offset", uid=UIDs.RAGGED_Q_UID.value
    )
    q.set_ragged_offset(q_ragged)

    k = graph.tensor(
        dim=(batch_size, h_k, kv_seq_len, d_qk), stride=(h_k * kv_seq_len * d_qk, d_qk, h_k * d_qk, 1), data_type=cudnn_dtype, name="K", uid=UIDs.K_UID.value
    )
    v = graph.tensor(
        dim=(batch_size, h_v, kv_seq_len, d_v), stride=(h_v * kv_seq_len * d_v, d_v, h_v * d_v, 1), data_type=cudnn_dtype, name="V", uid=UIDs.V_UID.value
    )

    seq_len_q_tensor = graph.tensor(
        dim=(batch_size, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_len_q", uid=UIDs.ACTUAL_SEQ_LENS_Q_UID.value
    )
    seq_len_kv_tensor = graph.tensor(
        dim=(batch_size, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_len_kv", uid=UIDs.ACTUAL_SEQ_LENS_KV_UID.value
    )

    # For chunked prefill with causal masking, use diagonal_band_right_bound to shift the causal diagonal
    # right_bound = causal_offset means Q[i] can attend to K[0:causal_offset+i+1]
    # This correctly handles chunks where Q positions represent later positions in the full sequence
    o, stats = graph.sdpa(
        name="sdpa_chunk",
        q=q,
        k=k,
        v=v,
        attn_scale=attn_scale,
        use_padding_mask=True,
        seq_len_q=seq_len_q_tensor,
        seq_len_kv=seq_len_kv_tensor,
        diagonal_band_right_bound=causal_offset if is_causal else None,
        generate_stats=False,
    )

    o.set_output(True).set_dim((batch_size, h_q, chunk_size, d_v)).set_stride((h_q * d_v, d_v, h_q * d_v, 1)).set_data_type(cudnn_dtype)
    o.set_uid(UIDs.O_UID.value)

    o_ragged = graph.tensor(
        dim=(batch_size + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="O_ragged_offset", uid=UIDs.RAGGED_O_UID.value
    )
    o.set_ragged_offset(o_ragged)

    try:
        graph.validate()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"Graph not supported: {e}")

    try:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"Graph not supported after validation: {e}")

    graph_cache[cache_key] = graph
    return graph


def execute_cudnn_sdpa_chunk(
    cudnn_handle, graph, q_chunk, k_chunk, v_chunk, seq_len_q, seq_len_kv, q_ragged_offset, o_ragged_offset, dtype, h_q, d_qk, d_v, batch_size, chunk_size
):
    total_q_tokens = q_chunk.shape[0]
    o_chunk = torch.empty(total_q_tokens, h_q, d_v, dtype=dtype, device="cuda")

    seq_len_q_4d = seq_len_q.view(-1, 1, 1, 1)
    seq_len_kv_4d = seq_len_kv.view(-1, 1, 1, 1)

    variant_pack = {
        UIDs.Q_UID.value: q_chunk,
        UIDs.RAGGED_Q_UID.value: q_ragged_offset,
        UIDs.K_UID.value: k_chunk,
        UIDs.V_UID.value: v_chunk,
        UIDs.ACTUAL_SEQ_LENS_Q_UID.value: seq_len_q_4d,
        UIDs.ACTUAL_SEQ_LENS_KV_UID.value: seq_len_kv_4d,
        UIDs.O_UID.value: o_chunk,
        UIDs.RAGGED_O_UID.value: o_ragged_offset,
    }

    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)

    q_chunk_shape = (batch_size, h_q, chunk_size, d_qk)
    o_chunk_shape = (batch_size, h_q, chunk_size, d_v)

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
        q_chunk_shape,
        q_ragged_offset.shape,
        k_chunk.shape,
        v_chunk.shape,
        seq_len_q_4d.shape,
        seq_len_kv_4d.shape,
        o_chunk_shape,
        o_ragged_offset.shape,
    ]
    override_strides = [
        q_chunk.stride(),
        q_ragged_offset.stride(),
        k_chunk.stride(),
        v_chunk.stride(),
        seq_len_q_4d.stride(),
        seq_len_kv_4d.stride(),
        o_chunk.stride(),
        o_ragged_offset.stride(),
    ]

    graph.execute(variant_pack, workspace, handle=cudnn_handle, override_uids=override_uids, override_shapes=override_shapes, override_strides=override_strides)
    torch.cuda.synchronize()
    return o_chunk


def create_bhsd_view(tensor, batch_size, num_heads, seq_len, head_dim):
    if tensor.shape == (batch_size, num_heads, seq_len, head_dim):
        bshd = tensor.permute(0, 2, 1, 3).contiguous()
        strides = (seq_len * num_heads * head_dim, head_dim, num_heads * head_dim, 1)
        return torch.as_strided(bshd.view(-1), (batch_size, num_heads, seq_len, head_dim), strides)
    storage = tensor.contiguous().view(-1)
    strides = (seq_len * num_heads * head_dim, head_dim, num_heads * head_dim, 1)
    return torch.as_strided(storage, (batch_size, num_heads, seq_len, head_dim), strides)


def extract_thd_chunk(thd_tensor, batch_size, total_seq_len, chunk_idx, chunk_size):
    """
    Extract a chunk from THD tensor for all batches.

    THD layout packs all tokens of batch 0 first, then batch 1, etc.
    For chunk_idx, we need positions [chunk_idx*chunk_size : (chunk_idx+1)*chunk_size]
    from each batch.

    Args:
        thd_tensor: [total_tokens, num_heads, head_dim] - packed tensor
        batch_size: number of batches
        total_seq_len: sequence length per batch
        chunk_idx: which chunk to extract (0-indexed)
        chunk_size: size of each chunk

    Returns:
        chunk: [batch_size * chunk_size, num_heads, head_dim]
    """
    chunk_start = chunk_idx * chunk_size
    chunk_end = chunk_start + chunk_size

    chunks = []
    for b in range(batch_size):
        batch_offset = b * total_seq_len
        chunks.append(thd_tensor[batch_offset + chunk_start : batch_offset + chunk_end, :, :])

    return torch.cat(chunks, dim=0)


def store_thd_chunk(o_full_thd, o_chunk, batch_size, total_seq_len, chunk_idx, chunk_size):
    """
    Store a chunk back into the full THD output tensor.

    Args:
        o_full_thd: [total_tokens, num_heads, head_dim] - output tensor to fill
        o_chunk: [batch_size * chunk_size, num_heads, head_dim] - chunk output
        batch_size: number of batches
        total_seq_len: sequence length per batch
        chunk_idx: which chunk (0-indexed)
        chunk_size: size of each chunk
    """
    chunk_start = chunk_idx * chunk_size
    chunk_end = chunk_start + chunk_size

    for b in range(batch_size):
        batch_offset = b * total_seq_len
        chunk_offset = b * chunk_size
        o_full_thd[batch_offset + chunk_start : batch_offset + chunk_end, :, :] = o_chunk[chunk_offset : chunk_offset + chunk_size, :, :]


def execute_chunked_prefill_cudnn(cudnn_handle, config, q_full_thd, k_full_bhsd, v_full_bhsd):
    batch_size, chunk_size, num_chunks = config.batch_size, config.chunk_size, config.num_chunks
    total_seq_len = config.total_seq_len
    h_q, h_k, h_v, d_qk, d_v, dtype = config.num_heads_q, config.num_heads_k, config.num_heads_v, config.head_dim_qk, config.head_dim_v, config.dtype

    o_full_thd = torch.empty(q_full_thd.shape[0], h_q, d_v, dtype=dtype, device="cuda")

    for chunk_idx in range(num_chunks):
        print(f"  Processing chunk {chunk_idx + 1}/{num_chunks}...")
        kv_end = (chunk_idx + 1) * chunk_size

        # Extract Q chunk from THD tensor (properly handling batch layout)
        q_chunk = extract_thd_chunk(q_full_thd, batch_size, total_seq_len, chunk_idx, chunk_size)

        # K/V up to current chunk end
        k_chunk_bhsd = create_bhsd_view(k_full_bhsd[:, :, :kv_end, :].contiguous(), batch_size, h_k, kv_end, d_qk)
        v_chunk_bhsd = create_bhsd_view(v_full_bhsd[:, :, :kv_end, :].contiguous(), batch_size, h_v, kv_end, d_v)

        seq_len_q = torch.full((batch_size,), chunk_size, dtype=torch.int32, device="cuda")
        seq_len_kv = torch.full((batch_size,), kv_end, dtype=torch.int32, device="cuda")
        q_ragged_offset = compute_ragged_offsets(seq_len_q, h_q, d_qk)
        o_ragged_offset = compute_ragged_offsets(seq_len_q, h_q, d_v)

        causal_offset = chunk_idx * chunk_size if config.is_causal else 0
        graph = build_cudnn_sdpa_chunk_graph(
            cudnn_handle, batch_size, h_q, h_k, h_v, d_qk, d_v, chunk_size, kv_end, dtype, config.attn_scale, config.is_causal, causal_offset
        )
        o_chunk = execute_cudnn_sdpa_chunk(
            cudnn_handle,
            graph,
            q_chunk,
            k_chunk_bhsd,
            v_chunk_bhsd,
            seq_len_q,
            seq_len_kv,
            q_ragged_offset,
            o_ragged_offset,
            dtype,
            h_q,
            d_qk,
            d_v,
            batch_size,
            chunk_size,
        )

        # Store output chunk back into full THD tensor
        store_thd_chunk(o_full_thd, o_chunk, batch_size, total_seq_len, chunk_idx, chunk_size)

    return o_full_thd


def compare_outputs(output_gpu, output_ref, atol=0.02, rtol=0.02, tag="output"):
    actual, expected = output_gpu.float(), output_ref.float()
    mismatches = torch.where(~torch.isclose(actual, expected, rtol=rtol, atol=atol))
    mismatch_cnt = mismatches[0].numel()
    if mismatch_cnt > 0:
        print(f"\n{tag}: {mismatch_cnt:,} mismatches ({100 * mismatch_cnt / actual.numel():.2f}%)")
        for idx in range(min(10, mismatch_cnt)):
            pos = tuple(m[idx].item() for m in mismatches)
            print(f"  idx{pos}: gpu={actual[pos]:+.6e}, ref={expected[pos]:+.6e}, diff={actual[pos] - expected[pos]:+.2e}")
    else:
        print(f"{tag}: All values match within tolerance")
    return mismatch_cnt


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
def test_chunked_prefill_basic(cudnn_handle):
    if LooseVersion(cudnn.backend_version_string()) < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0+")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Requires SM80+")

    print("\n" + "=" * 80 + "\nTest: Chunked Prefill (non-causal)\n" + "=" * 80)
    config = ChunkedPrefillConfig(
        batch_size=2,
        num_heads_q=8,
        num_heads_k=8,
        num_heads_v=8,
        head_dim_qk=128,
        head_dim_v=128,
        total_seq_len=4096,
        chunk_size=1024,
        dtype=torch.bfloat16,
        is_causal=False,
    )
    rng = torch.Generator(device="cuda").manual_seed(42)
    seq_lens = torch.full((config.batch_size,), config.total_seq_len, dtype=torch.int32, device="cuda")

    q_thd, _ = create_thd_tensor(seq_lens, config.num_heads_q, config.head_dim_qk, config.dtype, rng)
    k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.total_seq_len, config.head_dim_qk, config.dtype, rng)
    v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.total_seq_len, config.head_dim_v, config.dtype, rng)

    o_thd_gpu = execute_chunked_prefill_cudnn(cudnn_handle, config, q_thd, k_bhsd, v_bhsd)
    q_bhsd_ref = thd_to_bhsd(q_thd, seq_lens, config.total_seq_len)
    o_bhsd_ref = compute_chunked_prefill_reference(q_bhsd_ref, k_bhsd, v_bhsd, config, config.attn_scale)
    o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_lens)

    if compare_outputs(o_thd_gpu, o_thd_ref) > 0:
        pytest.fail("Test failed")
    print("\nTEST PASSED")


@pytest.mark.L0
@torch_fork_set_rng(seed=123)
def test_chunked_prefill_causal(cudnn_handle):
    if LooseVersion(cudnn.backend_version_string()) < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0+")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Requires SM80+")

    print("\n" + "=" * 80 + "\nTest: Chunked Prefill (causal)\n" + "=" * 80)
    config = ChunkedPrefillConfig(
        batch_size=2,
        num_heads_q=8,
        num_heads_k=8,
        num_heads_v=8,
        head_dim_qk=128,
        head_dim_v=128,
        total_seq_len=4096,
        chunk_size=1024,
        dtype=torch.bfloat16,
        is_causal=True,
    )
    rng = torch.Generator(device="cuda").manual_seed(123)
    seq_lens = torch.full((config.batch_size,), config.total_seq_len, dtype=torch.int32, device="cuda")

    q_thd, _ = create_thd_tensor(seq_lens, config.num_heads_q, config.head_dim_qk, config.dtype, rng)
    k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.total_seq_len, config.head_dim_qk, config.dtype, rng)
    v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.total_seq_len, config.head_dim_v, config.dtype, rng)

    o_thd_gpu = execute_chunked_prefill_cudnn(cudnn_handle, config, q_thd, k_bhsd, v_bhsd)
    q_bhsd_ref = thd_to_bhsd(q_thd, seq_lens, config.total_seq_len)
    o_bhsd_ref = compute_chunked_prefill_reference(q_bhsd_ref, k_bhsd, v_bhsd, config, config.attn_scale)
    o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_lens)

    if compare_outputs(o_thd_gpu, o_thd_ref) > 0:
        pytest.fail("Test failed")
    print("\nTEST PASSED")


@pytest.mark.L0
@torch_fork_set_rng(seed=456)
def test_chunked_prefill_gqa(cudnn_handle):
    if LooseVersion(cudnn.backend_version_string()) < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0+")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Requires SM80+")

    print("\n" + "=" * 80 + "\nTest: Chunked Prefill (GQA)\n" + "=" * 80)
    config = ChunkedPrefillConfig(
        batch_size=2,
        num_heads_q=8,
        num_heads_k=2,
        num_heads_v=2,
        head_dim_qk=128,
        head_dim_v=128,
        total_seq_len=4096,
        chunk_size=1024,
        dtype=torch.bfloat16,
        is_causal=False,
    )
    rng = torch.Generator(device="cuda").manual_seed(456)
    seq_lens = torch.full((config.batch_size,), config.total_seq_len, dtype=torch.int32, device="cuda")

    q_thd, _ = create_thd_tensor(seq_lens, config.num_heads_q, config.head_dim_qk, config.dtype, rng)
    k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.total_seq_len, config.head_dim_qk, config.dtype, rng)
    v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.total_seq_len, config.head_dim_v, config.dtype, rng)

    o_thd_gpu = execute_chunked_prefill_cudnn(cudnn_handle, config, q_thd, k_bhsd, v_bhsd)
    q_bhsd_ref = thd_to_bhsd(q_thd, seq_lens, config.total_seq_len)
    o_bhsd_ref = compute_chunked_prefill_reference(q_bhsd_ref, k_bhsd, v_bhsd, config, config.attn_scale)
    o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_lens)

    if compare_outputs(o_thd_gpu, o_thd_ref) > 0:
        pytest.fail("Test failed")
    print("\nTEST PASSED")


@pytest.mark.L0
@torch_fork_set_rng(seed=789)
def test_chunked_prefill_gqa_causal(cudnn_handle):
    if LooseVersion(cudnn.backend_version_string()) < "9.10.0":
        pytest.skip("THD layout requires cuDNN 9.10.0+")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Requires SM80+")

    print("\n" + "=" * 80 + "\nTest: Chunked Prefill (GQA + causal)\n" + "=" * 80)
    config = ChunkedPrefillConfig(
        batch_size=2,
        num_heads_q=8,
        num_heads_k=2,
        num_heads_v=2,
        head_dim_qk=128,
        head_dim_v=128,
        total_seq_len=4096,
        chunk_size=1024,
        dtype=torch.bfloat16,
        is_causal=True,
    )
    rng = torch.Generator(device="cuda").manual_seed(789)
    seq_lens = torch.full((config.batch_size,), config.total_seq_len, dtype=torch.int32, device="cuda")

    q_thd, _ = create_thd_tensor(seq_lens, config.num_heads_q, config.head_dim_qk, config.dtype, rng)
    k_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_k, config.total_seq_len, config.head_dim_qk, config.dtype, rng)
    v_bhsd = create_bhsd_tensor(config.batch_size, config.num_heads_v, config.total_seq_len, config.head_dim_v, config.dtype, rng)

    o_thd_gpu = execute_chunked_prefill_cudnn(cudnn_handle, config, q_thd, k_bhsd, v_bhsd)
    q_bhsd_ref = thd_to_bhsd(q_thd, seq_lens, config.total_seq_len)
    o_bhsd_ref = compute_chunked_prefill_reference(q_bhsd_ref, k_bhsd, v_bhsd, config, config.attn_scale)
    o_thd_ref = bhsd_to_thd(o_bhsd_ref, seq_lens)

    if compare_outputs(o_thd_gpu, o_thd_ref) > 0:
        pytest.fail("Test failed")
    print("\nTEST PASSED")


if __name__ == "__main__":
    print("Run with: pytest -vv -s -rA test_sdpa_chunked_prefill.py")
