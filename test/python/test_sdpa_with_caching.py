import pytest
from test_utils import torch_fork_set_rng

import torch
import cudnn
import math
import random

from enum import Enum

import logging

logger = logging.getLogger()

batch_buckets = {
    1: None,
    2: None,
    4: None,
    8: None,
    12: None,
    16: None,
    20: None,
    24: None,
    28: None,
    32: None,
}

cuda_graphs = {}

H_Q = H_K = H_V = 6
D_QK = D_VO = (
    128  # If you are changing D_VO != D_QK, you need to change the code in create_qkv_tensors for ragged offsets of O
)

MAX_SEQ_LEN_Q = 1024
MAX_SEQ_LEN_KV = 1024

ATTN_SCALE = float(1.0 / math.sqrt(D_QK))

device = "cuda:0"


class UIDs(Enum):
    RESERVED_INVALID_UID = 0

    Q_UID = 1  # Query tensor
    K_UID = 2  # Key cache tensor
    V_UID = 3  # Value cache tensor

    ACTUAL_SEQ_LENS_Q_UID = 100  # Actual sequence lengths for query tensor
    ACTUAL_SEQ_LENS_KV_UID = 101  # Actual sequence lengths for key/value tensor

    BLOCK_TABLES_UID = 200  # Block tables tensor
    BLOCK_TABLES_K_UID = 201  # Block tables tensor for key
    BLOCK_TABLES_V_UID = 202  # Block tables tensor for value

    RAGGED_Q_UID = 50  # Ragged query tensor
    RAGGED_O_UID = 51  # Ragged output tensor
    RAGGED_STATS_UID = 52  # Ragged stats tensor
    RAGGED_K_UID = 53  # Ragged key tensor
    RAGGED_V_UID = 54  # Ragged value tensor

    O_UID = 1000  # Output tensor
    STATS_UID = 1001  # Stats tensor


def create_qkv_tensors(batch_size, actual_seq_lens_q, actual_seq_lens_kv):

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q_gpu = torch.randn(cumsum_s_qo, H_Q, D_QK, device=device, dtype=torch.bfloat16)

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0) * D_QK * H_Q,
        ]
    ).int()

    q_indptr = torch.as_strided(q_indptr, (batch_size + 1, 1, 1, 1), (1, 1, 1, 1))

    k_gpu = (
        torch.randn(batch_size * H_K * MAX_SEQ_LEN_KV * D_QK)
        .half()
        .cuda()
        .as_strided(
            (batch_size, H_K, MAX_SEQ_LEN_KV, D_QK),
            (MAX_SEQ_LEN_KV * H_K * D_QK, D_QK, H_K * D_QK, 1),
        )
    )
    v_gpu = (
        torch.randn(batch_size * H_V * MAX_SEQ_LEN_KV * D_VO)
        .half()
        .cuda()
        .as_strided(
            (batch_size, H_V, MAX_SEQ_LEN_KV, D_VO),
            (MAX_SEQ_LEN_KV * H_V * D_VO, D_VO, H_V * D_VO, 1),
        )
    )

    out_gpu = torch.empty_like(q_gpu)

    return q_gpu, k_gpu, v_gpu, q_indptr, out_gpu


def _sdpa_key_fn(handle, batch_size):
    return batch_size


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
@cudnn.graph_cache(key_fn=_sdpa_key_fn)
def lookup_or_create_sdpa_graph(handle, batch_size):
    with cudnn.graph(handle) as (g, _):
        logger.info(f"Creating graph with batch_size: {batch_size}")
        cudnn_q = g.tensor(
            name="q",
            dim=(batch_size, H_Q, MAX_SEQ_LEN_Q, D_QK),
            stride=(H_Q * D_QK, D_QK, D_QK * H_Q, 1),
            data_type=cudnn.data_type.BFLOAT16,
            uid=UIDs.Q_UID.value,
        )

        ragged_q = g.tensor(
            name="ragged_q",
            dim=(batch_size + 1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT32,
            uid=UIDs.RAGGED_Q_UID.value,
        )
        cudnn_q.set_ragged_offset(ragged_q)

        cudnn_k = g.tensor(
            name="k",
            dim=(batch_size, H_K, MAX_SEQ_LEN_KV, D_QK),
            stride=(H_K * D_QK, D_QK, D_QK * H_K, 1),
            data_type=cudnn.data_type.BFLOAT16,
            uid=UIDs.K_UID.value,
        )

        cudnn_v = g.tensor(
            name="v",
            dim=(batch_size, H_V, MAX_SEQ_LEN_KV, D_VO),
            stride=(H_V * D_VO, D_VO, D_VO * H_V, 1),
            data_type=cudnn.data_type.BFLOAT16,
            uid=UIDs.V_UID.value,
        )

        cudnn_actual_seq_lens_q = g.tensor(
            name="actual_seq_lens_q",
            dim=(batch_size, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT32,
            uid=UIDs.ACTUAL_SEQ_LENS_Q_UID.value,
        )

        cudnn_actual_seq_lens_kv = g.tensor(
            name="actual_seq_lens_kv",
            dim=(batch_size, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT32,
            uid=UIDs.ACTUAL_SEQ_LENS_KV_UID.value,
        )

        O, Stats = g.sdpa(
            name="sdpa",
            q=cudnn_q,
            k=cudnn_k,
            v=cudnn_v,
            seq_len_q=cudnn_actual_seq_lens_q,
            seq_len_kv=cudnn_actual_seq_lens_kv,
            use_padding_mask=True,
            attn_scale=ATTN_SCALE,
            generate_stats=False,
            use_causal_mask_bottom_right=False,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        O.set_uid(UIDs.O_UID.value).set_output(True).set_dim(
            [batch_size, H_Q, MAX_SEQ_LEN_Q, D_VO]
        ).set_stride([MAX_SEQ_LEN_Q * D_VO * H_Q, D_VO, D_VO * H_Q, 1]).set_data_type(
            cudnn.data_type.BFLOAT16
        )

        O.set_ragged_offset(ragged_q)

        tensors_to_return = [cudnn_q, cudnn_k, cudnn_v, O]

        assert Stats is None

        return g, tensors_to_return

    return g


def pad_batch_size(batch_size, actual_seq_lens_q, actual_seq_lens_kv, ragged_offset_q):
    batch_buckets_keys = list(batch_buckets.keys())
    batch_size_padded = next(
        (b for b in batch_buckets_keys if b >= batch_size), batch_buckets_keys[-1]
    )
    zeros = torch.zeros(
        (batch_size_padded - batch_size, 1, 1, 1),
        dtype=actual_seq_lens_q.dtype,
        device=actual_seq_lens_q.device,
    )
    actual_seq_lens_q_padded = torch.cat([actual_seq_lens_q, zeros], dim=0)
    actual_seq_lens_kv_padded = torch.cat([actual_seq_lens_kv, zeros], dim=0)
    ragged_offset_q_padded = torch.cat([ragged_offset_q, zeros], dim=0)
    return (
        batch_size_padded,
        actual_seq_lens_q_padded,
        actual_seq_lens_kv_padded,
        ragged_offset_q_padded,
    )


def execute_sample(g, var_map, workspace, cudnn_handle):
    torch.cuda.nvtx.range_push("graph.execute sample")
    g.execute(var_map, workspace, handle=cudnn_handle)
    torch.cuda.nvtx.range_pop()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_ragged_sdpa_with_caching(cudnn_handle):
    if cudnn.backend_version_string() < "9.13.0":
        pytest.skip("This sample is only supported on cuDNN 9.13.0 or higher")

    if torch.cuda.get_device_properties(0).major < 9:
        pytest.skip("Ragged SDPA is only supported on Hopper or higher")

    # test set up basics
    seed = 1

    random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Initilizing the buckets")

    # The goal of the batch_buckets is to create a graph for each batch size in the batch_buckets
    # This is done to avoid creating a new graph for each sample, at run time.

    # In this sample, we bucket the batch sizes. But in theory this can be done for any arbitrary parameter.
    # For example, you can bucket by sequence length, or masking pattern, etc.

    for _batch_size in batch_buckets.keys():
        batch_buckets[_batch_size] = lookup_or_create_sdpa_graph(
            cudnn_handle, _batch_size
        )

    logger.info(f"Buckets initialized")

    sample_size = 4
    iter_count = 10

    # Samples is a list of tuples of (padded_batch_size, padded_actual_seq_lens_q, padded_actual_seq_lens_kv, ragged_offset_q, q_gpu, k_gpu, v_gpu, out_gpu)
    samples = []
    for _ in range(sample_size):
        batch_size = random.randint(1, 32)

        actual_seq_lens_q = torch.randint(
            1,
            MAX_SEQ_LEN_Q + 1,
            (batch_size, 1, 1, 1),
            dtype=torch.int32,
            device=device,
        )
        actual_seq_lens_kv = torch.randint(
            MAX_SEQ_LEN_Q,
            MAX_SEQ_LEN_KV + 1,
            (batch_size, 1, 1, 1),
            dtype=torch.int32,
            device=device,
        )

        q_gpu, k_gpu, v_gpu, ragged_offset_q, out_gpu = create_qkv_tensors(
            batch_size, actual_seq_lens_q, actual_seq_lens_kv
        )

        samples.append(
            (
                batch_size,
                actual_seq_lens_q,
                actual_seq_lens_kv,
                ragged_offset_q,
                q_gpu,
                k_gpu,
                v_gpu,
                out_gpu,
            )
        )

    # We pre-allocate the workspace to avoid creating a new workspace for each sample
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)

    for iter in range(iter_count):

        for sample in samples:

            torch.cuda.nvtx.range_push(f"Execute sample {iter}")
            (
                batch_size,
                actual_seq_lens_q,
                actual_seq_lens_kv,
                ragged_offset_q,
                q_gpu,
                k_gpu,
                v_gpu,
                out_gpu,
            ) = sample

            torch.cuda.nvtx.range_push("Padding the tensors of interest")
            (
                padded_batch_size,
                padded_actual_seq_lens_q,
                padded_actual_seq_lens_kv,
                padded_ragged_offset_q,
            ) = pad_batch_size(
                batch_size, actual_seq_lens_q, actual_seq_lens_kv, ragged_offset_q
            )
            torch.cuda.nvtx.range_pop()

            logger.info(
                f"Executing the sample with actual batch_size: {batch_size} and padded_batch_size: {padded_batch_size}"
            )

            # This will not create a new graph, it will return the graph from the bucket by the key function
            torch.cuda.nvtx.range_push("Look up the graph")
            g, tensors = lookup_or_create_sdpa_graph(
                cudnn_handle,
                padded_batch_size,
            )
            torch.cuda.nvtx.range_pop()

            var_map = {
                UIDs.Q_UID.value: q_gpu,
                UIDs.K_UID.value: k_gpu,
                UIDs.V_UID.value: v_gpu,
                UIDs.O_UID.value: out_gpu,
                UIDs.ACTUAL_SEQ_LENS_Q_UID.value: padded_actual_seq_lens_q,
                UIDs.ACTUAL_SEQ_LENS_KV_UID.value: padded_actual_seq_lens_kv,
                UIDs.RAGGED_Q_UID.value: padded_ragged_offset_q,
            }

            if cuda_graphs.get(padded_batch_size) is None:
                # Wrap the sample execution into a CUDA graph and execute the captured graph
                cuda_graph = torch.cuda.CUDAGraph()

                # Capture the execution into the CUDA graph
                with torch.cuda.graph(cuda_graph):
                    execute_sample(g, var_map, workspace, cudnn_handle)

                cuda_graphs[padded_batch_size] = cuda_graph

            # Now, launch the captured CUDA graph
            cuda_graphs[padded_batch_size].replay()
            torch.cuda.synchronize()

            torch.cuda.nvtx.range_pop()
