"""
This test harness allows for testing the various options of the attention operator. See example usage under "main" below.

The full documentation on the attention operator can be found in: https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html

Notebooks that demonstrate the attention operator can be found here:
- Introductory example: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/50_scaled_dot_product_attention.ipynb
- Example with paged caches: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/samples/python/52_scaled_dot_product_attention_with_paged_caches.ipynb
"""

import cudnn
import pytest
import torch
import math
from looseversion import LooseVersion

import random
import os

from test_utils import torch_fork_set_rng

input_type_options = [torch.float16, torch.bfloat16]
layout_options = ["bshd_bshd_bshd", "bs3hd", "sbh3d"]
head_group_options = ["multi_head", "group_query", "multi_query"]
bias_options = [False, True]
alibi_mask_options = [False, True]
padding_mask_options = [False, True]
causal_mask_options = [False, True]
causal_mask_bottom_right_options = [False, True]
diagonal_alignment_options = [
    cudnn.diagonal_alignment.TOP_LEFT,
    cudnn.diagonal_alignment.BOTTOM_RIGHT,
]
left_bound_options = [False, True]
right_bound_options = [False, True]
dropout_options = [False, True]
ragged_options = [False, True]
is_infer_options = [False, True]
page_table_options = [0, 1, "_packed"]


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


def compute_ref(
    q,
    k,
    v,
    attn_scale=1.0,
    bias=None,
    is_alibi=False,
    padding=None,
    sliding_window=(None, None),
    diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
    dropout_prob=0.0,
    dropout_mask=None,
    generate_stats=False,
    device="cuda",
):
    b, h_q, s_q, d_qk = q.shape
    _, h_k, s_kv, _ = k.shape
    _, h_v, _, d_v = v.shape

    assert k.shape == (b, h_k, s_kv, d_qk)
    assert v.shape == (b, h_v, s_kv, d_v)

    # use float32 datatype and math for reference computation
    q = q.to(dtype=torch.float32, device=device)
    k = k.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)

    assert isinstance(sliding_window, tuple) and len(sliding_window) == 2

    # Set/override left and right bounds explicitly
    left_bound = sliding_window[0]
    right_bound = sliding_window[1]

    # expand tensors for GQA and MQA
    if h_q != h_k:
        assert h_q % h_k == 0
        k = k.unsqueeze(2)
        k = k.expand(-1, -1, h_q // h_k, -1, -1)
        k = k.reshape(k.size(0), -1, k.size(3), k.size(4))
    if h_q != h_v:
        assert h_q % h_v == 0
        v = v.unsqueeze(2)
        v = v.expand(-1, -1, h_q // h_v, -1, -1)
        v = v.reshape(v.size(0), -1, v.size(3), v.size(4))

    if left_bound is not None:
        swa_mask_zero = torch.ones(1, 1, s_q, 1, dtype=torch.bool, device=device)
        swa_mask_zero[:, :, s_kv + left_bound - 1 :, :] = False
        q = q * swa_mask_zero
    # generate masks to compute reference values for padding mask
    # (also called variable sequence length)
    if padding is not None:
        q_mask = torch.zeros(b, 1, s_q, 1, dtype=torch.bool, device=device)
        k_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        v_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        s_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        p_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        seq_len_q, seq_len_kv = padding
        for i, (m, n) in enumerate(zip(seq_len_q, seq_len_kv)):
            q_mask[i, :, m:, :] = True
            k_mask[i, :, n:, :] = True
            v_mask[i, :, n:, :] = True
            s_mask[i, :, :, n:] = True
            p_mask[i, :, m:, :] = True

        q = q.masked_fill(q_mask, 0.0)
        k = k.masked_fill(k_mask, 0.0)
        v = v.masked_fill(v_mask, 0.0)

    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * attn_scale

    # Attention masks are applied in the following order:
    # - Bias mask
    # - Alibi mask
    # - Padding mask
    # - Causal mask
    if bias is not None:
        s = s + bias
    if is_alibi:
        index_row = torch.arange(s_q, dtype=torch.float32, device=device).view(-1, 1)
        index_col = torch.arange(s_kv, dtype=torch.float32, device=device)
        distance = index_col - index_row

        # Get the closest power of 2 to `n_heads`.
        # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
        # and then add the remaining slopes.
        n = 2 ** math.floor(math.log2(h_q))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n))

        # If `n_heads` is not a power of 2, then we add the remaining slopes.
        # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
        # And pick the slopes upto `n_heads`.
        if n < h_q:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (h_q - n), 2))
            # Concatenate the slopes with the remaining slopes.
            m = torch.cat([m, m_hat])

        # Reshape the tensor to [1, num_heads, 1, 1]
        m = m.view(1, -1, 1, 1).to(device=device)

        alibi_mask = distance.to(dtype=torch.float32) * m
        s = s + alibi_mask
    if padding is not None:
        s = s.masked_fill(s_mask, float("-inf"))
    if diagonal_alignment == diagonal_alignment.TOP_LEFT and right_bound is not None:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1 + right_bound)
        s = s.masked_fill(causal_mask, float("-inf"))
    elif (
        diagonal_alignment == diagonal_alignment.BOTTOM_RIGHT
        and right_bound is not None
    ):
        causal_mask_bottom_right = None
        if padding:
            causal_mask_bottom_right = torch.ones(
                b, 1, s_q, s_kv, dtype=torch.bool, device=device
            )
            seq_len_q, seq_len_kv = padding
            for i in range(b):
                causal_mask_bottom_right[i, :, :, :].triu_(
                    diagonal=seq_len_kv[i] - seq_len_q[i] + 1 + right_bound
                )
        else:
            causal_mask_bottom_right = torch.ones(
                s_q, s_kv, dtype=torch.bool, device=device
            )
            causal_mask_bottom_right.triu_(diagonal=s_kv - s_q + 1 + right_bound)
        s = s.masked_fill(causal_mask_bottom_right, float("-inf"))
    if left_bound is not None:
        assert diagonal_alignment is not None
        if diagonal_alignment == diagonal_alignment.TOP_LEFT:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
            swa_mask.tril_(diagonal=-1 * left_bound)

        elif diagonal_alignment == diagonal_alignment.BOTTOM_RIGHT:
            # BRCM + SWA for variable sequence lengths
            if padding:
                swa_mask = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
                seq_len_q, seq_len_kv = padding
                for i in range(b):
                    swa_mask[i, :, :, :].tril_(
                        diagonal=seq_len_kv[i] - seq_len_q[i] - left_bound
                    )
            # BRCM + SWA for fixed sequence lengths
            else:
                swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
                swa_mask.tril_(diagonal=-1 * left_bound + (s_kv - s_q))

        swa_mask &= swa_mask_zero.view(s_q, 1)
        s = s.masked_fill(swa_mask, float("-inf"))

    p = torch.softmax(s, dim=-1)

    if left_bound is not None:
        p = p * swa_mask_zero
    if padding is not None:
        p = p.masked_fill(p_mask, 0.0)

    # apply dropout mask over softmax outputs
    if dropout_prob != 0.0:
        assert (
            dropout_mask != None
        ), "PyTorch reference must have dropout_mask for dropout"
        p = (p * dropout_mask) / (1 - dropout_prob)

    o = torch.einsum("bhqk,bhkd->bhqd", p, v)

    # softmax stats is used for backwards computation
    if generate_stats:
        # amax (NOT absolute max) is used here to evenly distribute gradient
        row_max = torch.amax(s, -1, True)
        row_exp = torch.exp(s - row_max)
        row_sum = torch.sum(row_exp, -1, True)
        stats = row_max + torch.log(row_sum)
        return o, stats

    return o


# Generator for layout combinations
# | layout          | GQA             | Packed      | GQA and Packed |
# |-----------------|-----------------|-------------|----------------|
# | bshd_bshd_bshd  | bshd_bshd_bshd  | thd_thd_thd | thd_thd_thd    |
# | bs3hd           | bshd_bs2hd      | t3hd        | thd_t2hd       |
# | sbh3d           | sbhd_sbh2d      |             |                |
def generate_layout(
    layout,
    head_group,
    shape_q,
    shape_k,
    shape_v,
    shape_o,
    is_packed=False,
    seq_len_q=None,
    seq_len_kv=None,
):
    b, h_q, s_q, d_qk = shape_q
    b, h_k, s_kv, d_qk = shape_k
    b, h_v, s_kv, d_v = shape_v
    b, h_q, s_q, d_v = shape_o

    assert shape_q == (b, h_q, s_q, d_qk)
    assert shape_k == (b, h_k, s_kv, d_qk)
    assert shape_v == (b, h_v, s_kv, d_v)
    assert shape_o == (b, h_q, s_q, d_v)

    if layout == "bshd_bshd_bshd":
        if not is_packed:
            # bshd_bshd_bshd
            stride_q = (s_q * h_q * d_qk, d_qk, h_q * d_qk, 1)
            stride_k = (s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)
            stride_v = (s_kv * h_v * d_v, d_v, h_v * d_v, 1)
            stride_o = (s_q * h_q * d_v, d_v, h_q * d_v, 1)
            offset_q = 0
            offset_k = offset_q + b * s_q * h_q * d_qk
            offset_v = offset_k + b * s_kv * h_k * d_qk
        else:
            # thd_thd_thd
            assert seq_len_q is not None
            assert seq_len_kv is not None
            t_q = torch.sum(seq_len_q)
            t_kv = torch.sum(seq_len_kv)
            stride_q = (s_q * h_q * d_qk, d_qk, h_q * d_qk, 1)
            stride_k = (s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)
            stride_v = (s_kv * h_v * d_v, d_v, h_v * d_v, 1)
            stride_o = (s_q * h_q * d_v, d_v, h_q * d_v, 1)
            offset_q = 0
            offset_k = offset_q + t_q * h_q * d_qk
            offset_v = offset_k + t_kv * h_k * d_qk
    elif layout == "bs3hd":
        if not is_packed:
            if head_group == "multi_head":
                # bs3hd
                assert (h_q == h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
                h, s, d = h_q, s_q, d_qk
                stride_q = (s * 3 * h * d, d, 3 * h * d, 1)
                stride_k = (s * 3 * h * d, d, 3 * h * d, 1)
                stride_v = (s * 3 * h * d, d, 3 * h * d, 1)
                stride_o = (s * h * d, d, h * d, 1)
                offset_q = 0
                offset_k = offset_q + h * d
                offset_v = offset_k + h * d
            else:
                # bshd_bs2hd
                assert (h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
                h_kv, s, d = h_k, s_q, d_qk
                stride_q = (s * h_q * d, d, h_q * d, 1)
                stride_k = (s * 2 * h_kv * d, d, 2 * h_kv * d, 1)
                stride_v = (s * 2 * h_kv * d, d, 2 * h_kv * d, 1)
                stride_o = (s * h_q * d, d, h_q * d, 1)
                offset_q = 0
                offset_k = offset_q + s * b * h_q * d
                offset_v = offset_k + h_kv * d
        else:  # is_packed
            assert seq_len_q is not None
            assert seq_len_kv is not None
            t_q = torch.sum(seq_len_q)
            t_kv = torch.sum(seq_len_kv)
            if head_group == "multi_head":
                # t3hd
                assert (h_q == h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
                h, s, d = h_q, s_q, d_qk
                stride_q = (s * 3 * h * d, d, 3 * h * d, 1)
                stride_k = (s * 3 * h * d, d, 3 * h * d, 1)
                stride_v = (s * 3 * h * d, d, 3 * h * d, 1)
                stride_o = (s * h * d, d, h * d, 1)
                offset_q = 0
                offset_k = offset_q + h * d
                offset_v = offset_k + h * d
            else:
                # thd_t2hd
                assert (h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
                h_kv, s, d = h_k, s_q, d_qk
                stride_q = (s * h_q * d, d, h_q * d, 1)
                stride_k = (s * 2 * h_kv * d, d, 2 * h_kv * d, 1)
                stride_v = (s * 2 * h_kv * d, d, 2 * h_kv * d, 1)
                stride_o = (s * h_q * d, d, h_q * d, 1)
                offset_q = 0
                offset_k = offset_q + t_q * h_q * d
                offset_v = offset_k + h_kv * d
    elif layout == "sbh3d":
        if head_group == "multi_head":
            # sbh3d
            assert (h_q == h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
            h, s, d = h_q, s_q, d_qk
            stride_q = (h * 3 * d, 3 * d, b * h * 3 * d, 1)
            stride_k = (h * 3 * d, 3 * d, b * h * 3 * d, 1)
            stride_v = (h * 3 * d, 3 * d, b * h * 3 * d, 1)
            stride_o = (h * d, d, b * h * d, 1)
            offset_q = 0
            offset_k = offset_q + d
            offset_v = offset_k + d
        else:
            # sbhd_sbh2d
            assert (h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
            h_kv, s, d = h_k, s_q, d_qk
            stride_q = (h_q * d, d, b * h_q * d, 1)
            stride_k = (h_kv * 2 * d, 2 * d, b * h_kv * 2 * d, 1)
            stride_v = (h_kv * 2 * d, 2 * d, b * h_kv * 2 * d, 1)
            stride_o = (h_q * d, d, b * h_q * d, 1)
            offset_q = 0
            offset_k = offset_q + s * b * h_q * d
            offset_v = offset_k + d
    else:
        raise ValueError("layout must be 'bshd_bshd_bshd', 'bs3hd', or 'sbh3d'")

    return stride_q, stride_k, stride_v, stride_o, offset_q, offset_k, offset_v


def generate_ragged_offset(
    layout,
    head_group,
    shape_q,
    shape_k,
    shape_v,
    shape_o,
    seq_len_q,
    seq_len_kv,
    cudnn_version,
):
    b, h_q, s_q, d_qk = shape_q
    b, h_k, s_kv, d_qk = shape_k
    b, h_v, s_kv, d_v = shape_v
    b, h_q, s_q, d_v = shape_o

    assert shape_q == (b, h_q, s_q, d_qk)
    assert shape_k == (b, h_k, s_kv, d_qk)
    assert shape_v == (b, h_v, s_kv, d_v)
    assert shape_o == (b, h_q, s_q, d_v)

    # Compute the exclusive prefix sum for ragged sequence dimension
    # tensor has shape (B, 1, 1, 1)
    # output has shape (B+1, 1, 1, 1)
    # ex) tensor = [[[[2, 4, 1, 6]]]]
    #     output = [[[[0, 2, 6, 7, 13]]]]
    def compute_exclusive_prefix_sum(tensor):
        assert tensor.size(1) == tensor.size(2) == tensor.size(3) == 1
        return torch.cat(
            (
                torch.zeros(1, 1, 1, 1, dtype=tensor.dtype, device=tensor.device),
                torch.cumsum(tensor, dim=0),
            )
        )

    if layout == "bshd_bshd_bshd":
        # thd_thd_thd
        q_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_qk
        k_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_k * d_qk
        v_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_v * d_v
        o_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_v
    elif layout == "bs3hd":
        if head_group == "multi_head":
            # t3hd
            assert torch.equal(seq_len_q, seq_len_kv)
            assert (h_q == h_k == h_v) and (d_qk == d_v)
            seq_len, h, d = seq_len_q, h_q, d_qk
            q_ragged_offset = compute_exclusive_prefix_sum(seq_len) * 3 * h * d
            k_ragged_offset = compute_exclusive_prefix_sum(seq_len) * 3 * h * d
            v_ragged_offset = compute_exclusive_prefix_sum(seq_len) * 3 * h * d
            o_ragged_offset = compute_exclusive_prefix_sum(seq_len) * h * d
        else:
            # thd_t2hd
            assert (h_k == h_v) and (d_qk == d_v)
            seq_len, h_kv, d = seq_len_q, h_k, d_qk
            q_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d
            k_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * 2 * h_kv * d
            v_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * 2 * h_kv * d
            o_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d
    else:  # sbh3d
        raise ValueError()

    q_ragged_offset = q_ragged_offset.to(
        dtype=torch.int64 if cudnn_version >= "9.6.0" else torch.int32
    )
    k_ragged_offset = k_ragged_offset.to(
        dtype=torch.int64 if cudnn_version >= "9.6.0" else torch.int32
    )
    v_ragged_offset = v_ragged_offset.to(
        dtype=torch.int64 if cudnn_version >= "9.6.0" else torch.int32
    )
    o_ragged_offset = o_ragged_offset.to(
        dtype=torch.int64 if cudnn_version >= "9.6.0" else torch.int32
    )

    return q_ragged_offset, k_ragged_offset, v_ragged_offset, o_ragged_offset


# @brief Convert a padded page table into a packed page table
# @return packed_page_table: packed page table
# @return ragged_offset: offset into the packed page table
def convert_uniform_to_ragged_page_tables(
    uniform_tensor, seq_len, block_size, cudnn_version
):
    [B, H, S, D] = uniform_tensor.size()
    ragged_offset = torch.zeros(
        B + 1, 1, 1, 1, dtype=torch.int32, device=uniform_tensor.device
    )  # Initialize with first offset as 0
    for i in range(1, B + 1):
        prev_seq_len = seq_len[i - 1]
        num_pages_prev_batch = (prev_seq_len + block_size - 1) // block_size
        next_batch_offset = ragged_offset[i - 1] + num_pages_prev_batch
        ragged_offset[i, 0, 0, 0] = next_batch_offset

    ragged_offset.to(dtype=torch.int64 if cudnn_version >= "9.6.0" else torch.int32)
    # ragged_offset.to(dtype=torch.int32)

    packed_page_table = torch.zeros(B * S, H, D).to(
        dtype=uniform_tensor.dtype, device=uniform_tensor.device
    )

    uniform_tensor_thd = torch.einsum("bhsd->bshd", uniform_tensor).reshape(B * S, H, D)

    t_0 = 0
    for b, t_1 in enumerate(ragged_offset.flatten()[1:]):
        packed_page_table[t_0:t_1, :, :] = uniform_tensor_thd[
            b * S : b * S + (t_1 - t_0), :, :
        ]
        t_0 = t_1

    packed_page_table = packed_page_table.reshape(B, S, H, D)
    packed_page_table = torch.einsum("bshd->bhsd", packed_page_table)

    return packed_page_table, ragged_offset


def convert_ragged_to_uniform(ragged_tensor, seq_len):
    # limitations:
    # 1. tensor is bhsd dim order and bshd stride order (may be interleaved)
    # 2. ragged tensor is packed and in-order, therefore
    #    ragged offset is monatomically increasing
    assert ragged_tensor.dim() == 4
    b, h, s, d = ragged_tensor.size()
    b_stride, h_stride, s_stride, d_stride = ragged_tensor.stride()
    assert b_stride >= s_stride >= h_stride >= d_stride
    assert seq_len.dim() == 4 and (b, 1, 1, 1) == seq_len.size()

    # ragged offset is given in 4D, convert to 1D locally
    seq_len = seq_len.flatten()

    # convert bhsd to bshd and flatten
    uniform_tensor = torch.zeros(b, s, h, d).to(
        dtype=ragged_tensor.dtype, device=ragged_tensor.device
    )
    ragged_tensor_thd = torch.einsum("bhsd->bshd", ragged_tensor).reshape(b * s, h, d)

    # copy
    t = 0
    for b, s in enumerate(seq_len):
        uniform_tensor[b, 0:s, :, :] = ragged_tensor_thd[t : t + s, :, :]
        t += s

    # convert back to bshd to bhsd
    uniform_tensor = torch.einsum("bshd->bhsd", uniform_tensor)
    return uniform_tensor


def generate_actual_seq_lens(
    b, s_q, s_kv, layout, head_group, is_padding, force_sq_less_or_equal_than_skv
):
    seq_len_q_gpu = None
    seq_len_kv_gpu = None

    if is_padding:
        seq_len_q_gpu = torch.randint(
            1, s_q + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda"
        )

        if not (layout == "bs3hd" and head_group == "multi_head"):
            seq_len_kv_gpu = torch.randint(
                1, s_kv + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda"
            )
            # Avoid seq_len_q > seq_len_kv (known limitation):
            if force_sq_less_or_equal_than_skv:
                seq_len_q_gpu = torch.max(
                    torch.tensor(1), seq_len_q_gpu % seq_len_kv_gpu
                )
        else:
            seq_len_kv_gpu = seq_len_q_gpu

    return (seq_len_q_gpu, seq_len_kv_gpu)


# fmt: off
@pytest.mark.parametrize("is_infer", is_infer_options, ids=lambda p: f"infer{int(p)}")
@pytest.mark.parametrize("is_ragged", ragged_options, ids=lambda p: f"ragged{int(p)}")
@pytest.mark.parametrize("is_dropout", dropout_options, ids=lambda p: f"dropout{int(p)}")
@pytest.mark.parametrize("is_right_bound", right_bound_options, ids=lambda p: f"right_bound{int(p)}")
@pytest.mark.parametrize("is_left_bound", left_bound_options, ids=lambda p: f"left_bound{int(p)}")
@pytest.mark.parametrize("diagonal_alignment", diagonal_alignment_options, ids=lambda p: f"diagonal_alignment_{p}".replace("diagonal_alignment.",""))
@pytest.mark.parametrize("is_padding", padding_mask_options, ids=lambda p: f"padding{int(p)}")
@pytest.mark.parametrize("is_alibi", alibi_mask_options, ids=lambda p: f"alibi{int(p)}")
@pytest.mark.parametrize("is_bias", bias_options, ids=lambda p: f"bias{int(p)}")
@pytest.mark.parametrize("paged_attention", page_table_options, ids=lambda p: f"paged{p}")
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("layout", layout_options)
@pytest.mark.parametrize("input_type", input_type_options, ids=lambda p: str(p))
@pytest.mark.L0
# fmt: on
@torch_fork_set_rng(seed=0)
def test_sdpa(
    input_type,
    layout,
    head_group,
    paged_attention,
    is_bias,
    is_alibi,
    is_padding,
    diagonal_alignment,
    is_left_bound,
    is_right_bound,
    is_dropout,
    is_ragged,
    is_infer,
    request,
    cudnn_handle
):
    is_paged_attention = (paged_attention == "_packed" or paged_attention == 1)
    is_packed_paged_attention = (paged_attention == "_packed")

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    if cudnn_version < "8.9.3":
        pytest.skip("SDPA fprop requires cudnn 8.9.3 or higher")

    if head_group != "multi_head" and cudnn_version < "8.9.7":
        pytest.skip("GQA and MQA is only supported 8.9.7 onwards.")

    if is_alibi and cudnn_version < "8.9.4":
        pytest.skip("ALiBi mask is only supported 8.9.4 onwards.")

    if is_padding and cudnn_version < "8.9.3":
        pytest.skip("Padding mask is only supported 8.9.3 onwards.")

    if is_dropout and cudnn_version < "8.9.6":
        pytest.skip("Dropout reference is only supported on 8.9.6 onwards.")

    if is_ragged and cudnn_version < "9":
        pytest.skip("Ragged tensor is only supported 9.0.0 onwards")

    if is_ragged and layout == "bs3hd" and cudnn_version < "9.1.0":
        pytest.skip("t3hd is only supported on 9.1.0 onwards")

    if is_ragged and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Ragged tensor is only supported hopper")

    if is_ragged and not (layout == "bshd_bshd_bshd" or layout == "bs3hd"):
        pytest.skip("Ragged tensor is only tested with thd_thd_thd and t3hd")

    if is_ragged and not is_padding:
        pytest.skip("Ragged tensor is only tested with packed variable length tensors")

    if is_paged_attention and (not is_padding or cudnn_version < "9.5" or not layout == "bshd_bshd_bshd" or is_ragged):
        pytest.skip("Paged attention is only tested with packed variable length tensors, bshd_bshd_bshd, no ragged offsets, and only on cuDNNv9.5 or greater")

    if is_packed_paged_attention and cudnn_version < "9.10.2":
        pytest.skip("Packed paged attention is only supported on cuDNNv9.10.2 or greater")

    # -------------------------- default randomized parameter testing ------------------------
    # batch size
    b = 2
    # query sequence length

    s_q_choices = [ 24, 256, 512, 1024, 2048]

    if cudnn_version >= "9.7.0" and layout == "bshd_bshd_bshd":
        s_q_choices.extend([1])

    s_q = random.choice(s_q_choices)

    if s_q == 1:
        is_left_bound = False
        request.config.option.mha_left_bound = None
        
    # key+value sequence length
    s_kv = (
        random.choice([24, 32, 256, 512, 1024, 2048])
        if layout == "bshd_bshd_bshd"
        else s_q
    )

    d_choices = [32, 56, 64, 128]

    # for hopper always extend d_choices to 256
    if cudnn_version >= "9.5":
        d_choices.extend([192, 256])

    if cudnn_version >= "9.9.0":
        d_choices.extend([200, 264, 520])

    # query+key embedding dimension per head
    d_qk = random.choice(d_choices)
    # value embedding dimension per head
    d_v = (
        random.choice(d_choices)
        if (layout == "bshd_bshd_bshd")
        else d_qk
    )
    # number of heads
    h_q = 6
    if head_group == "multi_head":
        h_k = 6
        h_v = 6
    elif head_group == "group_query":
        h_k = random.choice([6, 3, 2, 1])
        h_v = random.choice([6, 3, 2, 1]) if layout == "bshd_bshd_bshd" else h_k
    elif head_group == "multi_query":
        h_k = 1
        h_v = 1
    else:
        assert False, "Head group must be either MHA, GQA, or MQA"

    # block size for paged attention
    block_size = random.choice([32, 64, 128])

    # Left/right bound should only be specified if we requested to set a left_bound/right_bound
    assert (request.config.option.mha_left_bound is None) or is_left_bound
    assert (request.config.option.mha_right_bound is None) or is_right_bound

    # If bounds are requested: randomly pick between 1/0 and s_kv/4
    left_bound = max(1, random.choice([1, s_kv//4])) if is_left_bound else None
    right_bound = random.choice([0, s_kv//4]) if is_right_bound else None

    # -------------------------- override test parameters if args are provided ----------------
    b = int(request.config.option.mha_b) if request.config.option.mha_b != None else b
    s_q = int(request.config.option.mha_s_q) if request.config.option.mha_s_q != None else s_q
    s_kv = int(request.config.option.mha_s_kv) if request.config.option.mha_s_kv != None else s_kv
    d_qk = int(request.config.option.mha_d_qk) if request.config.option.mha_d_qk != None else d_qk
    d_v = int(request.config.option.mha_d_v) if request.config.option.mha_d_v != None else d_v
    h_q = int(request.config.option.mha_h_q) if request.config.option.mha_h_q != None else h_q
    h_k = int(request.config.option.mha_h_k) if request.config.option.mha_h_k != None else h_k
    h_v = int(request.config.option.mha_h_v) if request.config.option.mha_h_v != None else h_v
    block_size = int(request.config.option.mha_block_size) if request.config.option.mha_block_size != None else block_size
    left_bound = int(request.config.option.mha_left_bound) if request.config.option.mha_left_bound != None else left_bound
    right_bound = int(request.config.option.mha_right_bound) if request.config.option.mha_right_bound != None else right_bound

    if s_q == 1:
        is_dropout = False
        request.config.option.mha_dropout = None

    if d_qk != d_v and cudnn_version < "8.9.6":
        pytest.skip("d_qk != d_v is only supported on 8.9.6 onwards.")

    if d_qk != d_v and is_ragged and cudnn_version < "9.1":
        pytest.skip("d_qk != d_v is not supported with ragged offset")

    print("\n=============== TEST CMD TO REPRODUCE ===============")
    cmd = f"pytest {request.node.nodeid} --mha_b={b} --mha_s_q={s_q} --mha_s_kv={s_kv} --mha_d_qk={d_qk} --mha_d_v={d_v} --mha_h_q={h_q} --mha_h_k={h_k} --mha_h_v={h_v} --mha_block_size={block_size}"
    if left_bound is not None:
        cmd += f" --mha_left_bound={left_bound}"
    if right_bound is not None:
        cmd += f" --mha_right_bound={right_bound}"

    print(cmd)
    print("=====================================================")

    attn_scale = 0.125
    dropout_prob = 0.1 if is_dropout else 0.0

    shape_q = (b, h_q, s_q, d_qk)
    shape_k = (b, h_k, s_kv, d_qk)
    shape_v = (b, h_v, s_kv, d_v)
    shape_o = (b, h_q, s_q, d_v)

    qkv_num_elems = math.prod(shape_q) + math.prod(shape_k) + math.prod(shape_v)

    (stride_q, stride_k, stride_v, stride_o, offset_q, offset_k, offset_v) = (
        generate_layout(
            layout,
            head_group,
            shape_q,
            shape_k,
            shape_v,
            shape_o,
        )
    )

    qkv_gpu = torch.randn(qkv_num_elems, dtype=input_type, device="cuda") - 0.5
    q_gpu = torch.as_strided(qkv_gpu, shape_q, stride_q, storage_offset=offset_q)
    k_gpu = torch.as_strided(qkv_gpu, shape_k, stride_k, storage_offset=offset_k)
    v_gpu = torch.as_strided(qkv_gpu, shape_v, stride_v, storage_offset=offset_v)

    bias_gpu = (
        torch.randn(1, h_q, s_q, s_kv, device="cuda", dtype=input_type)
        if is_bias
        else None
    )

    seq_len_q_gpu, seq_len_kv_gpu = generate_actual_seq_lens(b, s_q, s_kv, layout, head_group, is_padding, is_left_bound or (is_right_bound and diagonal_alignment == diagonal_alignment.BOTTOM_RIGHT))

    if is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    rng_dump_gpu = (
        torch.zeros((b, h_q, s_q, s_kv), dtype=torch.float32, device="cuda")
        if is_dropout
        else None
    )

    if is_ragged:
        (
            q_ragged_offset_gpu,
            k_ragged_offset_gpu,
            v_ragged_offset_gpu,
            o_ragged_offset_gpu,
        ) = generate_ragged_offset(
            layout,
            head_group,
            shape_q,
            shape_k,
            shape_v,
            shape_o,
            seq_len_q_gpu,
            seq_len_kv_gpu,
            cudnn_version
        )

    o_gpu = torch.empty(
        b * h_q * s_q * d_v, dtype=input_type, device="cuda"
    ).as_strided(shape_o, stride_o)
    stats_gpu = (
        torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device="cuda")
        if not is_infer
        else None
    )

    def create_container_and_page_table(tensor, block_size):
        B, H, S, D = tensor.shape
        # num_blocks = math.ceil(S/block_size) * B
        blocks_per_batch = math.ceil(S/block_size)

        padding_seq = (blocks_per_batch * block_size) - S
        if padding_seq > 0:
            zeros = torch.zeros(B,H,padding_seq,D, device='cuda', dtype=tensor.dtype)
            cat_tensor = torch.cat((tensor, zeros), axis = 2)
        else:
            cat_tensor = tensor

        reshaped = torch.cat((cat_tensor.clone()).chunk(blocks_per_batch, dim=2), dim=0)

        table_size = math.ceil(S/block_size)
        page_table_temp = torch.linspace(0, B*table_size-1, B*table_size, device='cuda', dtype=torch.int32).reshape(table_size,1,B,1)
        page_table_temp = torch.transpose(page_table_temp,0,2)

        alt_stride = (blocks_per_batch, blocks_per_batch, 1, 1)

        # Make batch size outer dimension (required for SM100 kernel)
        page_table_dims = (B, 1, blocks_per_batch, 1)
        page_table = torch.randn(blocks_per_batch * B).int().cuda().as_strided(page_table_dims, alt_stride)
        page_table.copy_(page_table_temp)

        return(reshaped, page_table)


    container_k_gpu = None
    container_v_gpu = None
    page_table_k_gpu = None
    page_table_v_gpu = None

    if is_paged_attention:
        container_k_gpu, page_table_k_gpu = create_container_and_page_table(k_gpu, block_size)
        container_v_gpu, page_table_v_gpu = create_container_and_page_table(v_gpu, block_size)

    if is_packed_paged_attention:
        page_table_k_gpu, page_table_k_ragged_offset_gpu = convert_uniform_to_ragged_page_tables(page_table_k_gpu, seq_len_kv_gpu, block_size, cudnn_version)
        page_table_v_gpu, page_table_v_ragged_offset_gpu = convert_uniform_to_ragged_page_tables(page_table_v_gpu, seq_len_kv_gpu, block_size, cudnn_version)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
    # cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
        sm_version=sm_version
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu) if not is_paged_attention else graph.tensor_like(container_k_gpu)
    v = graph.tensor_like(v_gpu) if not is_paged_attention else graph.tensor_like(container_v_gpu)

    page_table_k = graph.tensor_like(page_table_k_gpu) if is_paged_attention else None
    page_table_v = graph.tensor_like(page_table_v_gpu) if is_paged_attention else None

    bias = graph.tensor_like(bias_gpu) if is_bias else None

    seq_len_q = graph.tensor_like(seq_len_q_gpu) if is_padding else None
    seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if is_padding else None

    if is_dropout:
        seed = graph.tensor_like(seed_gpu)
        offset = graph.tensor_like(offset_gpu)
        dropout_tuple = (dropout_prob, seed, offset)

    rng_dump = graph.tensor_like(rng_dump_gpu) if is_dropout else None

    q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if is_ragged else None
    k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if is_ragged else None
    v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if is_ragged else None
    o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if is_ragged else None

    page_table_k_ragged_offset = graph.tensor_like(page_table_k_ragged_offset_gpu) if is_packed_paged_attention else None
    page_table_v_ragged_offset = graph.tensor_like(page_table_v_ragged_offset_gpu) if is_packed_paged_attention else None

    if is_ragged:
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)

    if is_packed_paged_attention:
        page_table_k.set_ragged_offset(page_table_k_ragged_offset)
        page_table_v.set_ragged_offset(page_table_v_ragged_offset)
   
    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=not is_infer,
        attn_scale=attn_scale,
        bias=bias,
        use_alibi_mask=is_alibi,
        use_padding_mask=is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_band_left_bound=left_bound, 
        diagonal_band_right_bound=right_bound,
        diagonal_alignment=diagonal_alignment,
        dropout=dropout_tuple if is_dropout else None,
        rng_dump=rng_dump,
        paged_attention_k_table=page_table_k,
        paged_attention_v_table=page_table_v,
        paged_attention_max_seq_len_kv=s_kv if is_paged_attention else None
    )

    o.set_output(True).set_dim(shape_o).set_stride(stride_o)
    if is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    if is_infer == False:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    try:
        graph.validate()
    except cudnn.cudnnGraphNotSupportedError as e:
        print("Graph not supported")
        pytest.xfail(repr(e))
    except Exception as e:
        pytest.fail(repr(e))

    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    #graph.create_execution_plans([cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu if not is_paged_attention else container_k_gpu,
        v: v_gpu if not is_paged_attention else container_v_gpu,
        bias: bias_gpu,
        seq_len_q: seq_len_q_gpu,
        seq_len_kv: seq_len_kv_gpu,
        q_ragged_offset: q_ragged_offset_gpu if is_ragged else None,
        k_ragged_offset: k_ragged_offset_gpu if is_ragged else None,
        v_ragged_offset: v_ragged_offset_gpu if is_ragged else None,
        o_ragged_offset: o_ragged_offset_gpu if is_ragged else None,
        o: o_gpu,
        stats: stats_gpu,
        rng_dump: rng_dump_gpu,
        page_table_k: page_table_k_gpu,
        page_table_v: page_table_v_gpu,
        page_table_k_ragged_offset: page_table_k_ragged_offset_gpu if is_packed_paged_attention else None,
        page_table_v_ragged_offset: page_table_v_ragged_offset_gpu if is_packed_paged_attention else None
    }

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # compare with torch autograd reference
    q_ref = q_gpu.float()
    k_ref = k_gpu.float()
    v_ref = v_gpu.float()

    if is_ragged:
        q_ref = convert_ragged_to_uniform(q_ref, seq_len_q_gpu.detach())
        k_ref = convert_ragged_to_uniform(k_ref, seq_len_kv_gpu.detach())
        v_ref = convert_ragged_to_uniform(v_ref, seq_len_kv_gpu.detach())

    if is_bias:
        bias_ref = bias_gpu.float()

    if is_padding:
        seq_len_q_ref = seq_len_q_gpu.flatten()
        seq_len_kv_ref = seq_len_kv_gpu.flatten()

    if is_dropout:
        rng_dump_ref = rng_dump_gpu.float()

    ret = compute_ref(
        q_ref,
        k_ref,
        v_ref,
        attn_scale=attn_scale,
        bias=bias_ref if is_bias else None,
        is_alibi=is_alibi,
        padding=(seq_len_q_ref, seq_len_kv_ref) if is_padding else None,
        sliding_window=(left_bound, right_bound),
        diagonal_alignment=diagonal_alignment,
        generate_stats=(is_infer == False),
        dropout_prob=dropout_prob,
        dropout_mask=rng_dump_ref if is_dropout else None,
    )
    if is_infer == False:
        o_ref, stats_ref = ret
    else:
        o_ref = ret

    if is_ragged:
        o_gpu = convert_ragged_to_uniform(o_gpu, seq_len_q_gpu.detach())

    if is_padding:
        # zero out padded region of the output for comparison
        for i, m in enumerate(seq_len_q_ref):
            o_ref[i, :, m:, :] = 0
            o_gpu[i, :, m:, :] = 0
            if is_infer == False:
                stats_ref[i, :, m:, :] = 0
                stats_gpu[i, :, m:, :] = 0

    torch.testing.assert_close(o_ref, o_gpu, check_dtype=False, atol=2e-2, rtol=2e-2)
    if is_infer == False:
        torch.testing.assert_close(stats_ref, stats_gpu, atol=2e-2, rtol=2e-2)

    
    



# fmt: off
@pytest.mark.parametrize("is_ragged", ragged_options, ids=lambda p: f"ragged{int(p)}")
@pytest.mark.parametrize("is_dropout", dropout_options, ids=lambda p: f"dropout{int(p)}")
@pytest.mark.parametrize("is_right_bound", right_bound_options, ids=lambda p: f"right_bound{int(p)}")
@pytest.mark.parametrize("is_left_bound", left_bound_options, ids=lambda p: f"left_bound{int(p)}")
@pytest.mark.parametrize("diagonal_alignment", diagonal_alignment_options, ids=lambda p: f"diagonal_alignment_{p}".replace("diagonal_alignment.",""))
@pytest.mark.parametrize("is_padding", padding_mask_options, ids=lambda p: f"padding{int(p)}")
@pytest.mark.parametrize("is_alibi", alibi_mask_options, ids=lambda p: f"alibi{int(p)}")
@pytest.mark.parametrize("is_bias", bias_options, ids=lambda p: f"bias{int(p)}")
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("layout", layout_options)
@pytest.mark.parametrize("input_type", input_type_options, ids=lambda p: str(p))
@pytest.mark.L0
# fmt: on
@torch_fork_set_rng(seed=0)
def test_sdpa_backward(
    input_type,
    layout,
    head_group,
    is_bias,
    is_alibi,
    is_padding,
    diagonal_alignment,
    is_right_bound,
    is_left_bound,
    is_dropout,
    is_ragged,
    request,
    cudnn_handle
):

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    if cudnn_version < "8.9.3":
        pytest.skip("SDPA bprop requires cudnn 8.9.3 or higher")

    if head_group != "multi_head" and cudnn_version < "8.9.7":
        pytest.skip("GQA and MQA is only supported 8.9.7 onwards.")

    if is_bias and cudnn_version < "8.9.6":
        pytest.skip("dBias is only supported 8.9.6 onwards.")

    if is_bias and cudnn_version < "9" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("dBias is only supported on hopper before v9.")

    if is_alibi and cudnn_version < "8.9.4":
        pytest.skip("ALiBi mask is only supported 8.9.4 onwards.")

    if is_padding and cudnn_version < "8.9.3":
        pytest.skip("Padding mask is only supported 8.9.3 onwards.")

    if is_dropout and cudnn_version < "8.9.6":
        pytest.skip("RNG dump is only supported on 8.9.6 onwards.")

    if is_ragged and cudnn_version < "9":
        pytest.skip("Ragged tensor is only supported 9.0.0 onwards")

    if is_ragged and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Ragged tensor is only supported hopper")

    if is_ragged and not (layout == "bshd_bshd_bshd" or layout == "bs3hd"):
        pytest.skip("Ragged tensor is only tested with thd_thd_thd and t3hd")

    if is_ragged and layout == "bs3hd" and cudnn_version < "9.1.0":
        pytest.skip("t3hd is only supported on 9.1.0 onwards")

    if is_ragged and not is_padding:
        pytest.skip("Ragged tensor is only tested with packed variable length tensors")

    # -------------------------- default randomized parameter testing ------------------------
    # batch size
    b = 2
    # query sequence length
    s_q = random.choice([8, 16, 24, 32, 256, 512, 1024])
    # key+value sequence length
    s_kv = (
        random.choice([8, 16, 24, 32, 256, 512, 1024])
        if layout == "bshd_bshd_bshd"
        else s_q
    )
    # query+key embedding dimension per head
    d_qk = random.choice([32, 56, 64, 128])
    # value embedding dimension per head
    d_v = (
        random.choice([64, 96, 128])
        if (layout == "bshd_bshd_bshd")
        else d_qk
    )
    # number of heads
    h_q = 6
    if head_group == "multi_head":
        h_k = 6
        h_v = 6
    elif head_group == "group_query":
        h_k = random.choice([6, 3, 2, 1])
        h_v = random.choice([6, 3, 2, 1]) if layout == "bshd_bshd_bshd" else h_k
    elif head_group == "multi_query":
        h_k = 1
        h_v = 1
    else:
        assert False, "Head group must be either MHA, GQA, or MQA"

    # test both deterministic and nondeterministic implementation
    if cudnn_version < "9":
        os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "0"
    is_deterministic = random.choice([True, False])

    
    # Left/right bound should only be specified if we requested to set a left_bound/right_bound
    assert (request.config.option.mha_left_bound is None) or is_left_bound
    assert (request.config.option.mha_right_bound is None) or is_right_bound

    # If bounds are requested: randomly pick between 1/0 and s_kv/4
    left_bound = max(1, random.choice([1, s_kv//4])) if is_left_bound else None
    right_bound = random.choice([0, s_kv//4]) if is_right_bound else None

    # -------------------------- override test parameters if args are provided ----------------
    b = int(request.config.option.mha_b) if request.config.option.mha_b != None else b
    s_q = int(request.config.option.mha_s_q) if request.config.option.mha_s_q != None else s_q
    s_kv = int(request.config.option.mha_s_kv) if request.config.option.mha_s_kv != None else s_kv
    d_qk = int(request.config.option.mha_d_qk) if request.config.option.mha_d_qk != None else d_qk
    d_v = int(request.config.option.mha_d_v) if request.config.option.mha_d_v != None else d_v
    h_q = int(request.config.option.mha_h_q) if request.config.option.mha_h_q != None else h_q
    h_k = int(request.config.option.mha_h_k) if request.config.option.mha_h_k != None else h_k
    h_v = int(request.config.option.mha_h_v) if request.config.option.mha_h_v != None else h_v
    is_deterministic = (
        bool(int(request.config.option.mha_deterministic))
        if request.config.option.mha_deterministic != None
        else is_deterministic
    )
    left_bound = int(request.config.option.mha_left_bound) if request.config.option.mha_left_bound != None else left_bound
    right_bound = int(request.config.option.mha_right_bound) if request.config.option.mha_right_bound != None else right_bound

    if d_qk != d_v and cudnn_version < "8.9.6":
        pytest.skip("d_qk != d_v is only supported on 8.9.6 onwards.")

    if d_qk != d_v and is_ragged and cudnn_version < "9.1":
        pytest.skip("d_qk != d_v is not supported with ragged offset")

    if is_alibi and not (diagonal_alignment == diagonal_alignment.TOP_LEFT and (right_bound is not None and right_bound == 0)):
        pytest.skip("ALiBi mask is only supported with causal mask")

    if (
        is_deterministic
        and cudnn_version < "9"
        and torch.cuda.get_device_capability()[0] < 9
    ):
        pytest.skip("Ampere deterministic implementation is not supported below 9.0.0")

    print("\n=============== TEST CMD TO REPRODUCE ===============")
    cmd = f"pytest {request.node.nodeid} --mha_b={b} --mha_s_q={s_q} --mha_s_kv={s_kv} --mha_d_qk={d_qk} --mha_d_v={d_v} --mha_h_q={h_q} --mha_h_k={h_k} --mha_h_v={h_v} --mha_deterministic={int(is_deterministic)}"
    if left_bound is not None:
        cmd += f" --mha_left_bound={left_bound}"
    if right_bound is not None:
        cmd += f" --mha_right_bound={right_bound}"
    print(cmd)
    print("=====================================================")

    attn_scale = 0.125
    dropout_prob = 0.1 if is_dropout else 0.0

    shape_q = (b, h_q, s_q, d_qk)
    shape_k = (b, h_k, s_kv, d_qk)
    shape_v = (b, h_v, s_kv, d_v)
    shape_o = (b, h_q, s_q, d_v)

    qkv_num_elems = math.prod(shape_q) + math.prod(shape_k) + math.prod(shape_v)

    (stride_q, stride_k, stride_v, stride_o, offset_q, offset_k, offset_v) = (
        generate_layout(
            layout,
            head_group,
            shape_q,
            shape_k,
            shape_v,
            shape_o,
        )
    )

    qkv_gpu = torch.randn(qkv_num_elems, dtype=input_type, device="cuda") - 0.5
    q_gpu = torch.as_strided(qkv_gpu, shape_q, stride_q, storage_offset=offset_q)
    k_gpu = torch.as_strided(qkv_gpu, shape_k, stride_k, storage_offset=offset_k)
    v_gpu = torch.as_strided(qkv_gpu, shape_v, stride_v, storage_offset=offset_v)

    dQKV_gpu = torch.empty(qkv_num_elems, dtype=input_type, device="cuda")
    dQ_gpu = torch.as_strided(dQKV_gpu, shape_q, stride_q, storage_offset=offset_q)
    dK_gpu = torch.as_strided(dQKV_gpu, shape_k, stride_k, storage_offset=offset_k)
    dV_gpu = torch.as_strided(dQKV_gpu, shape_v, stride_v, storage_offset=offset_v)

    dO_gpu = 0.1 * torch.randn(
        b * h_q * s_q * d_v, dtype=input_type, device="cuda"
    ).as_strided(shape_o, stride_o)

    bias_gpu = (
        torch.randn(1, h_q, s_q, s_kv, device="cuda", dtype=input_type)
        if is_bias
        else None
    )
    dBias_gpu = (
        torch.randn(1, h_q, s_q, s_kv, device="cuda", dtype=input_type)
        if is_bias
        else None
    )

    seq_len_q_gpu, seq_len_kv_gpu = generate_actual_seq_lens(b, s_q, s_kv, layout, head_group, is_padding, is_left_bound or (is_right_bound and diagonal_alignment == diagonal_alignment.BOTTOM_RIGHT))

    # maxT = next_multiple_of_64(sum(seq_len))
    max_t_q = ((torch.sum(seq_len_q_gpu).item() + 63) // 64) * 64 if is_ragged else None
    max_t_kv = ((torch.sum(seq_len_kv_gpu).item() + 63) // 64) * 64 if is_ragged else None

    if is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    rng_dump_gpu = (
        torch.zeros((b, h_q, s_q, s_kv), dtype=torch.float32, device="cuda")
        if is_dropout
        else None
    )

    if is_ragged:
        (
            q_ragged_offset_gpu,
            k_ragged_offset_gpu,
            v_ragged_offset_gpu,
            o_ragged_offset_gpu,
        ) = generate_ragged_offset(
            layout,
            head_group,
            shape_q,
            shape_k,
            shape_v,
            shape_o,
            seq_len_q_gpu,
            seq_len_kv_gpu,
            cudnn_version
        )

    o_gpu = torch.empty(
        b * h_q * s_q * d_v, dtype=input_type, device="cuda"
    ).as_strided(shape_o, stride_o)
    stats_gpu = torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device="cuda")

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
    # forward cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
        sm_version=sm_version
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)

    bias = graph.tensor_like(bias_gpu) if is_bias else None

    seq_len_q = graph.tensor_like(seq_len_q_gpu) if is_padding else None
    seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if is_padding else None

    if is_dropout:
        seed = graph.tensor_like(seed_gpu)
        offset = graph.tensor_like(offset_gpu)
        dropout_tuple = (dropout_prob, seed, offset)

    rng_dump = graph.tensor_like(rng_dump_gpu) if is_dropout else None

    q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if is_ragged else None
    k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if is_ragged else None
    v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if is_ragged else None
    o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if is_ragged else None

    if is_ragged:
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)

    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=attn_scale,
        bias=bias,
        use_alibi_mask=is_alibi,
        use_padding_mask=is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_band_left_bound=left_bound, 
        diagonal_band_right_bound=right_bound,
        diagonal_alignment=diagonal_alignment,
        dropout=dropout_tuple if is_dropout else None,
        rng_dump=rng_dump,
    )

    o.set_output(True).set_dim(shape_o).set_stride(stride_o)
    if is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    try:
        graph.validate()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.xfail(repr(e))
    except Exception as e:
        pytest.fail(repr(e))

    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        bias: bias_gpu,
        seq_len_q: seq_len_q_gpu,
        seq_len_kv: seq_len_kv_gpu,
        q_ragged_offset: q_ragged_offset_gpu if is_ragged else None,
        k_ragged_offset: k_ragged_offset_gpu if is_ragged else None,
        v_ragged_offset: v_ragged_offset_gpu if is_ragged else None,
        o_ragged_offset: o_ragged_offset_gpu if is_ragged else None,
        o: o_gpu,
        stats: stats_gpu,
        rng_dump: rng_dump_gpu,
    }

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    if cudnn_version < "8.9.6" and is_padding:
        # zero out padded region of the output and stats
        for i, m in enumerate(seq_len_q_gpu):
            o_gpu[i, :, m:, :] = 0
            stats_gpu[i, :, m:, :] = 0

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)
    sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]

    # backward cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
        sm_version = sm_version
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)
    o = graph.tensor_like(o_gpu)
    dO = graph.tensor_like(dO_gpu)
    stats = graph.tensor_like(stats_gpu)

    bias = graph.tensor_like(bias_gpu) if is_bias else None
    dBias = (
        graph.tensor_like(dBias_gpu).set_stride((h_q * s_q * s_kv, s_q * s_kv, s_kv, 1))
        if is_bias
        else None
    )

    seq_len_q = graph.tensor_like(seq_len_q_gpu) if is_padding else None
    seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if is_padding else None

    if is_dropout:
        seed = graph.tensor_like(seed_gpu)
        offset = graph.tensor_like(offset_gpu)
        dropout_tuple = (dropout_prob, seed, offset)

    q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if is_ragged else None
    k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if is_ragged else None
    v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if is_ragged else None
    o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if is_ragged else None

    if is_ragged:
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)
        o.set_ragged_offset(o_ragged_offset)
        dO.set_ragged_offset(o_ragged_offset)

    dQ, dK, dV = graph.sdpa_backward(
        name="sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=o,
        dO=dO,
        stats=stats,
        attn_scale=attn_scale,
        bias=bias,
        dBias=dBias,
        use_alibi_mask=is_alibi,
        use_padding_mask=is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        max_total_seq_len_q=max_t_q,
        max_total_seq_len_kv=max_t_kv,
        diagonal_band_left_bound=left_bound, 
        diagonal_band_right_bound=right_bound,
        diagonal_alignment=diagonal_alignment,
        dropout=dropout_tuple if is_dropout else None,
        use_deterministic_algorithm=is_deterministic,
    )

    dQ.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
    dK.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
    dV.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())
    if is_ragged:
        dQ.set_ragged_offset(q_ragged_offset)
        dK.set_ragged_offset(k_ragged_offset)
        dV.set_ragged_offset(v_ragged_offset)

    try:
        graph.validate()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.xfail(repr(e))
    except Exception as e:
        pytest.fail(repr(e))

    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        dO: dO_gpu,
        stats: stats_gpu,
        dQ: dQ_gpu,
        dK: dK_gpu,
        dV: dV_gpu,
        bias: bias_gpu,
        dBias: dBias_gpu,
        seq_len_q: seq_len_q_gpu,
        seq_len_kv: seq_len_kv_gpu,
        q_ragged_offset: q_ragged_offset_gpu if is_ragged else None,
        k_ragged_offset: k_ragged_offset_gpu if is_ragged else None,
        v_ragged_offset: v_ragged_offset_gpu if is_ragged else None,
        o_ragged_offset: o_ragged_offset_gpu if is_ragged else None,
    }

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # compare with torch autograd reference
    q_ref = q_gpu.detach().float().requires_grad_()
    k_ref = k_gpu.detach().float().requires_grad_()
    v_ref = v_gpu.detach().float().requires_grad_()
    dO_ref = dO_gpu.detach().float()

    if is_ragged:
        q_ref = convert_ragged_to_uniform(q_ref, seq_len_q_gpu.detach())
        k_ref = convert_ragged_to_uniform(k_ref, seq_len_kv_gpu.detach())
        v_ref = convert_ragged_to_uniform(v_ref, seq_len_kv_gpu.detach())
        dO_ref = convert_ragged_to_uniform(dO_ref, seq_len_q_gpu.detach())

    if is_bias:
        bias_ref = bias_gpu.detach().float().requires_grad_()

    if is_padding:
        seq_len_q_ref = seq_len_q_gpu.detach().flatten()
        seq_len_kv_ref = seq_len_kv_gpu.detach().flatten()

    if is_dropout:
        rng_dump_ref = rng_dump_gpu.detach().float()

    o_ref = compute_ref(
        q_ref,
        k_ref,
        v_ref,
        attn_scale=attn_scale,
        bias=bias_ref if is_bias else None,
        is_alibi=is_alibi,
        padding=(seq_len_q_ref, seq_len_kv_ref) if is_padding else None,
        sliding_window=(left_bound, right_bound),
        diagonal_alignment=diagonal_alignment,
        dropout_prob=dropout_prob,
        dropout_mask=rng_dump_ref if is_dropout else None,
        generate_stats=False,
    )

    outputs_ref = [o_ref]
    inputs_ref = [q_ref, k_ref, v_ref]

    if is_bias:
        inputs_ref.append(bias_ref)

    [dQ_ref, dK_ref, dV_ref, *opt_refs] = list(
        torch.autograd.grad(outputs=outputs_ref, inputs=inputs_ref, grad_outputs=dO_ref)
    )

    if is_bias:
        dBias_ref = opt_refs.pop(0)

    if is_ragged:
        dQ_gpu = convert_ragged_to_uniform(dQ_gpu, seq_len_q_gpu.detach())
        dK_gpu = convert_ragged_to_uniform(dK_gpu, seq_len_kv_gpu.detach())
        dV_gpu = convert_ragged_to_uniform(dV_gpu, seq_len_kv_gpu.detach())

    if is_padding:
        # zero out padded region of the output for comparison
        for i, (m, n) in enumerate(zip(seq_len_q_ref, seq_len_kv_ref)):
            dQ_ref[i, :, m:, :] = 0
            dQ_gpu[i, :, m:, :] = 0
            dK_ref[i, :, n:, :] = 0
            dK_gpu[i, :, n:, :] = 0
            dV_ref[i, :, n:, :] = 0
            dV_gpu[i, :, n:, :] = 0

    torch.cuda.synchronize()

    torch.testing.assert_close(dQ_ref, dQ_gpu, check_dtype=False, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        dK_ref,
        dK_gpu,
        check_dtype=False,
        atol=2e-2 if input_type != torch.bfloat16 else 7e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        dV_ref,
        dV_gpu,
        check_dtype=False,
        atol=2e-2 if input_type != torch.bfloat16 else 7e-2,
        rtol=2e-2,
    )
    if is_bias:
        torch.testing.assert_close(
            dBias_ref, dBias_gpu, check_dtype=False, atol=2e-2, rtol=2e-2
        )
