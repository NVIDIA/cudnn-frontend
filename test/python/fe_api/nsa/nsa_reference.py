"""
Reference implementations for NSA (Native Sparse Attention) tests.
Contains CPU/GPU reference implementations for verification.
"""

import torch
import cudnn
import math


def convert_thd_to_bshd(thd_tensor, seq_len: torch.Tensor, s: int):
    assert thd_tensor.dim() == 3
    t, h, d = thd_tensor.size()

    if seq_len.dim() == 1:
        seq_len = seq_len.view(-1, 1, 1, 1)
    assert seq_len.dim() == 4
    assert seq_len.size(1) == seq_len.size(2) == seq_len.size(3) == 1
    b = seq_len.size(0)
    seq_len = seq_len.flatten()

    bshd_tensor = torch.zeros(
        (b, s, h, d), dtype=thd_tensor.dtype, device=thd_tensor.device
    )

    cumulative_seq_len = torch.cumsum(seq_len, dim=0) - seq_len
    for bi in range(b):
        t_beg = cumulative_seq_len[bi]
        t_end = t_beg + seq_len[bi]
        bshd_tensor[bi, : seq_len[bi], :, :] = thd_tensor[t_beg:t_end, :, :]

    # Return a view with layout (b, h, s, d) while keeping strides as if (b, s, h, d)
    return bshd_tensor.permute(0, 2, 1, 3)


def convert_bshd_to_thd(bshd_tensor, seq_len: torch.Tensor, maxT: int):
    assert bshd_tensor.dim() == 4
    b, h, s, d = bshd_tensor.size()

    if seq_len.dim() == 1:
        seq_len = seq_len.view(-1, 1, 1, 1)
    assert seq_len.dim() == 4
    assert seq_len.size(1) == seq_len.size(2) == seq_len.size(3) == 1
    seq_len = seq_len.flatten()

    thd_tensor = torch.zeros(
        (maxT, h, d), dtype=bshd_tensor.dtype, device=bshd_tensor.device
    )

    # Interpret input as (b, s, h, d) in memory while keeping the (b, h, s, d) layout
    bshd_base = bshd_tensor.permute(0, 2, 1, 3)

    cumulative_seq_len = torch.cumsum(seq_len, dim=0) - seq_len
    for bi in range(b):
        t_beg = cumulative_seq_len[bi]
        t_end = t_beg + seq_len[bi]
        thd_tensor[t_beg:t_end, :, :] = bshd_base[bi, : seq_len[bi], :, :]

    return thd_tensor


def run_ref_nsa_selection_attention(
    Q_in,
    K_in,
    V_in,
    O_out,
    L_out,
    M_out,
    seq_lens,
    block_indices,
    block_counts,
    block_size,
    softmax_scale,
    dtype=torch.float32,
):
    """
    Reference implementation of NSA selection attention.

    This is a CPU-based reference implementation for verifying the correctness
    of the CUDA NSA implementation.

    Args:
        Q_in: Query tensor of shape (T, H_q, D)
        K_in: Key tensor of shape (T, H_kv, D)
        V_in: Value tensor of shape (T, H_kv, D_v)
        O_out: Output tensor of shape (T, H_q, D_v)
        L_out: Log-sum-exp tensor of shape (T, H_q, 1)
        M_out: Max values tensor of shape (T, H_q, 1)
        seq_lens: List of sequence lengths for each batch
        block_indices: Block indices tensor
        block_counts: Block counts tensor
        block_size: Size of each block
        softmax_scale: Softmax scaling factor
        dtype: Data type for computation

    Returns:
        Tuple of (O_out, L_out, M_out) with updated values
    """
    # Q.shape: (T, H_q, D) -> (T, h_kv, g, D)
    # K.shape: (T, H_kv, D) -> (T, h_kv, 1, D)
    # V.shape: (T, H_kv, D_v) -> (T, h_kv, 1, D_v)
    # O.shape: (T, H_q, D_v) -> (T, h_kv, g, D_v)
    # L.shape: (T, H_q, 1) -> (T, h_kv, g)
    # M.shape: (T, H_q, 1) -> (T, h_kv, g)
    # seq_lens.shape: (batch_size)
    # block_indices.shape: (T, h_kv, topk_size)
    # block_counts.shape: (T, h_kv)

    t, h_q, d = Q_in.shape
    _, h_kv, d_v = V_in.shape

    head_num_kv = h_kv
    total_seq_len = t
    GQA_group_size = h_q // head_num_kv

    Q = Q_in.view(t, h_kv, GQA_group_size, d).to(dtype=dtype)
    K = K_in.view(t, h_kv, 1, d).to(dtype=dtype)
    V = V_in.view(t, h_kv, 1, d_v).to(dtype=dtype)
    O = O_out.view(t, h_kv, GQA_group_size, d_v).to(dtype=dtype)
    L = L_out.view(t, h_kv, GQA_group_size).to(dtype=torch.float32)
    M = M_out.view(t, h_kv, GQA_group_size).to(dtype=torch.float32)

    seq_offset = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        seq_end = seq_offset + seq_len

        for h in range(h_kv):
            # Extract Q, K, V for current sequence and head
            q_seq = Q[seq_offset:seq_end, h, :, :]  # [seq_len, GQA_group_size, d]
            k_seq = K[seq_offset:seq_end, h, 0, :]  # [seq_len, d]
            v_seq = V[seq_offset:seq_end, h, 0, :]  # [seq_len, d_v]

            # Step 1: Compute full Q @ K^T attention matrix
            # q_seq: [seq_len, GQA_group_size, d] @ k_seq.T: [d, seq_len] -> [seq_len, GQA_group_size, seq_len]
            qk_scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * softmax_scale

            # Step 2: Create block selection mask
            mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=qk_scores.device,
                dtype=torch.float32,
            )
            seq_block_counts = block_counts[seq_offset:seq_end, h]  # [seq_len]
            seq_block_indices = block_indices[
                seq_offset:seq_end, h, :
            ]  # [seq_len, topk_size]
            topk_size = seq_block_indices.size(-1)
            block_range = torch.arange(topk_size, device=mask.device).unsqueeze(
                0
            )  # [1, topk_size]
            valid_mask = block_range < seq_block_counts.unsqueeze(
                1
            )  # [seq_len, topk_size]

            query_indices, block_indices_flat = torch.where(valid_mask)
            if len(query_indices) > 0:
                block_ids = seq_block_indices[query_indices, block_indices_flat]
                token_starts = block_ids * block_size
                token_ends = torch.clamp((block_ids + 1) * block_size, max=seq_len)

                block_sizes = token_ends - token_starts
                max_block_size = block_sizes.max().item() if len(block_sizes) > 0 else 0

                if max_block_size > 0:
                    offsets = torch.arange(
                        max_block_size, device=mask.device
                    )  # [max_block_size]

                    num_blocks = len(block_ids)
                    offsets_expanded = offsets.unsqueeze(0).expand(
                        num_blocks, -1
                    )  # [num_blocks, max_block_size]
                    block_sizes_expanded = block_sizes.unsqueeze(1)  # [num_blocks, 1]
                    token_starts_expanded = token_starts.unsqueeze(1)  # [num_blocks, 1]
                    query_indices_expanded = query_indices.unsqueeze(
                        1
                    )  # [num_blocks, 1]

                    position_valid = (
                        offsets_expanded < block_sizes_expanded
                    )  # [num_blocks, max_block_size]

                    token_positions = (
                        token_starts_expanded + offsets_expanded
                    )  # [num_blocks, max_block_size]

                    valid_positions = torch.where(position_valid)
                    if len(valid_positions[0]) > 0:
                        block_idx_flat = valid_positions[0]
                        offset_idx = valid_positions[1]

                        final_query_indices = query_indices_expanded[block_idx_flat, 0]
                        final_key_indices = token_positions[block_idx_flat, offset_idx]

                        mask[final_query_indices, final_key_indices] = 0.0

            # Step 3: Apply mask to attention scores
            qk_scores_fp32 = qk_scores.float() + mask.unsqueeze(
                1
            )  # [seq_len, 1, seq_len] -> [seq_len, GQA_group_size, seq_len]

            # Step 4: Compute softmax
            qk_max = torch.max(qk_scores_fp32, dim=-1, keepdim=True)[
                0
            ]  # [seq_len, GQA_group_size, 1]
            qk_exp = torch.exp(
                qk_scores_fp32 - qk_max
            )  # [seq_len, GQA_group_size, seq_len]
            qk_sum = torch.sum(
                qk_exp, dim=-1, keepdim=True
            )  # [seq_len, GQA_group_size, 1]
            attn_weights = qk_exp / qk_sum  # [seq_len, GQA_group_size, seq_len]

            # Step 5: Compute output O = attention_weights @ V
            # attn_weights: [seq_len, GQA_group_size, seq_len] @ v_seq: [seq_len, d_v] -> [seq_len, GQA_group_size, d_v]
            output = torch.matmul(
                attn_weights, v_seq.float()
            )  # [seq_len, GQA_group_size, d_v]

            # Store results
            O[seq_offset:seq_end, h, :, :] = output.to(dtype)

            # Store L (sum of exp) and M (max) statistics - reusing computed values
            # L should store the sum of exponentials (row_sum), not logsumexp, to match reference
            L[seq_offset:seq_end, h, :] = qk_sum.squeeze(
                -1
            )  # [seq_len, GQA_group_size]
            M[seq_offset:seq_end, h, :] = qk_max.squeeze(
                -1
            )  # [seq_len, GQA_group_size]

        seq_offset = seq_end

    return O.view(t, h_q, d_v), L.view(t, h_q, 1), M.view(t, h_q, 1)


def run_ref_nsa_compression_attention(
    Q,
    K,
    V,
    scale_softmax,
    scale_output,
    lse_calculation=False,
    bottom_right_align=False,
):
    """
    Reference implementation for CompressionAttention.

    Args:
        Q: (B, H_q, S_q, D)
        K: (B, H_k, S_k, D)
        V: (B, H_k, S_k, D_v)
        scale_softmax (float): softmax scale
        scale_output (float): output scale applied after matmul
        lse_calculation (bool): whether to compute LSE
        bottom_right_align (bool): align end of q to end of k (not used here)

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: (O_ref, LSE_ref or None)
            O_ref: (B, H_q, S_q, D_v)
            LSE_ref: (B, H_q, S_q) if lse_calculation else None
    """
    assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4
    b, h_q, s_q, d = Q.shape
    _, h_k, s_k, d_v = V.shape

    # Handle GQA/MQA head broadcasting
    if h_q != h_k:
        repeat_factor = h_q // h_k
        K = K.repeat_interleave(repeat_factor, dim=1)
        V = V.repeat_interleave(repeat_factor, dim=1)

    batch_size = Q.size(0)
    ref_list = []
    lse_list = [] if lse_calculation else None
    for batch_idx in range(batch_size):
        q_i = Q[batch_idx]
        k_i = K[batch_idx]
        v_i = V[batch_idx]

        s_i = torch.einsum("hqd,hkd->hqk", q_i, k_i) * scale_softmax
        s_q_i = q_i.shape[1]
        s_k_i = k_i.shape[1]

        # Causal compressed mask
        q_coords = torch.arange(0, s_q_i, device=s_i.device).view(-1, 1)
        num_compress_blocks = s_k_i
        stride = max(1, s_q_i // max(1, s_k_i))
        k_coords = (
            ((torch.arange(0, num_compress_blocks, device=s_i.device) + 1) * stride) - 1
        ).view(1, -1)
        _mask = k_coords > q_coords
        s_i = s_i.masked_fill(_mask, -torch.inf)

        if lse_calculation:
            lse_i = torch.logsumexp(s_i, dim=-1)

        p_i = torch.softmax(s_i, dim=-1)
        p_i = p_i.masked_fill(_mask, 0)

        ref_i = torch.einsum("hqk,hkd->hqd", p_i, v_i)
        ref_i = ref_i * scale_output
        ref_list.append(ref_i)
        if lse_calculation:
            lse_list.append(lse_i)

    O_ref = torch.stack(ref_list)
    if lse_calculation:
        LSE_ref = torch.stack(lse_list).float()
    else:
        LSE_ref = None

    return O_ref, LSE_ref


def run_ref_nsa_swa(
    q,
    k,
    v,
    attn_scale=None,
    padding=None,
    left_bound=None,
    right_bound=None,
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

    if left_bound != None:
        swa_mask_zero = torch.ones(1, 1, s_q, 1, dtype=torch.bool, device=device)
        swa_mask_zero[:, :, s_kv + left_bound - 1 :, :] = False
        q = q * swa_mask_zero

    # generate masks to compute reference values for padding mask (also called variable sequence length)
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

    s = torch.einsum("bhqd,bhkd->bhqk", q, k)
    if attn_scale is not None:
        s = s * attn_scale

    if padding is not None:
        s = s.masked_fill(s_mask, float("-inf"))

    if right_bound != None:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1 + right_bound)
        s = s.masked_fill(causal_mask, float("-inf"))

    if left_bound != None:
        swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        swa_mask.tril_(diagonal=-1 * left_bound)
        swa_mask &= swa_mask_zero.view(s_q, 1)
        s = s.masked_fill(swa_mask, float("-inf"))

    p = torch.softmax(s, dim=-1)

    if left_bound != None:
        p = p * swa_mask_zero
    if padding is not None:
        p = p.masked_fill(p_mask, 0.0)

    o = torch.einsum("bhqk,bhkd->bhqd", p, v)

    # softmax stats is used for backwards computation
    if generate_stats:
        # amax (NOT absolute max) is used here to evenly distribute gradient
        row_max = torch.amax(s, -1, True)
        row_exp = torch.exp(s - row_max)
        row_sum = torch.sum(row_exp, -1, True)
        stats = row_max + torch.log(row_sum)
        # stats = stats.squeeze(dim=-1)
        return o, stats

    return o


def check_ref_nsa_selection_attention(
    Q,
    K,
    V,
    O,
    L,
    M,
    block_indices,
    block_counts,
    test_config,
):
    if test_config["skip_ref"]:
        print(
            f'Skipped reference computation for selection attention with config: b={test_config["b"]}, seq_len={test_config["s_q"]}, h_q={test_config["h_q"]}, h_kv={test_config["h_kv"]}, d={test_config["d"]}'
        )
    O_ref = torch.zeros_like(O, dtype=torch.float32)
    L_ref = torch.zeros_like(L, dtype=torch.float32)
    M_ref = torch.zeros_like(M, dtype=torch.float32)
    O_ref, L_ref, M_ref = run_ref_nsa_selection_attention(
        Q,
        K,
        V,
        O_ref,
        L_ref,
        M_ref,
        test_config["actual_s_q"].cuda(),
        block_indices,
        block_counts,
        test_config["block_size"],
        test_config["scale_softmax"],
        dtype=test_config["dtype"],
    )

    torch.testing.assert_close(O, O_ref, atol=0.01, rtol=1e-05)
    # torch.testing.assert_close(L, L_ref, atol=0.01, rtol=1e-05)
    # torch.testing.assert_close(M, M_ref, atol=0.01, rtol=1e-05)


def check_ref_nsa_compression_attention(
    Q,
    K,
    V,
    O,
    LSE=None,
    scale_output=1.0,
    scale_softmax=None,
    atol=0.01,
    rtol=1e-05,
    test_config=None,
):
    if test_config["skip_ref"]:
        print(
            f'Skipped reference computation for compression attention with config: b={test_config["b"]}, seq_len={test_config["s_q"]}, h_q={test_config["h_q"]}, h_k={test_config["h_k"]}, d={test_config["d"]}'
        )
        return
    scale_softmax = (
        scale_softmax
        if scale_softmax is not None
        else (
            test_config["scale_softmax"]
            if test_config is not None
            else 1.0 / math.sqrt(test_config["d_qk"])
        )
    )

    if test_config["layout"] == "thd":
        assert (
            "actual_s_q" in test_config
        ), "actual_s_q is required when using T,H,D layout"
        seq_len_q = test_config["actual_s_q"].to(device=Q.device)
        max_seq_len_q = int(seq_len_q.max().item())

        # Convert THD -> (B, H, S, D)
        q_bshd = convert_thd_to_bshd(Q, seq_len_q, max_seq_len_q)
        k_bshd = convert_thd_to_bshd(K, seq_len_q, max_seq_len_q)
        v_bshd = convert_thd_to_bshd(V, seq_len_q, max_seq_len_q)

        O_ref_bshd, LSE_ref_bsh = run_ref_nsa_compression_attention(
            q_bshd,
            k_bshd,
            v_bshd,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
            lse_calculation=LSE is not None,
        )

        # Convert O_ref back to THD for comparison
        total_T = int(seq_len_q.sum().item())
        O_ref_thd = convert_bshd_to_thd(O_ref_bshd, seq_len_q, total_T).to(
            dtype=O.dtype
        )
        torch.testing.assert_close(O, O_ref_thd, atol=atol, rtol=rtol)

        if LSE is not None:
            LSE_bhs1 = LSE_ref_bsh.unsqueeze(-1)
            LSE_thd = convert_bshd_to_thd(LSE_bhs1, seq_len_q, total_T)
            torch.testing.assert_close(LSE, LSE_thd, atol=atol, rtol=rtol)
    elif test_config["layout"] == "bshd":
        O_ref, LSE_ref = run_ref_nsa_compression_attention(
            Q,
            K,
            V,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
            lse_calculation=LSE is not None,
        )

        torch.testing.assert_close(O, O_ref.to(dtype=O.dtype), atol=atol, rtol=rtol)
        if LSE is not None:
            if (LSE.ndim == LSE_ref.ndim + 1) and (LSE.shape[-1] == 1):
                LSE_ref = LSE_ref.unsqueeze(-1)
            torch.testing.assert_close(LSE, LSE_ref, atol=atol, rtol=rtol)
    else:
        raise ValueError(f"Invalid layout: {test_config['layout']}")


def check_ref_nsa_swa(
    Q,
    K,
    V,
    O,
    Stats=None,
    seq_len_q=None,
    seq_len_kv=None,
    max_seq_len_q=None,
    max_seq_len_kv=None,
    test_config=None,
):
    if test_config is not None and test_config["skip_ref"]:
        print(
            f'Skipped reference computation for SWA with config: b={test_config["b"]}, seq_len={test_config["s_q"]}, h_q={test_config["h_q"]}, h_kv={test_config["h_kv"]}, d={test_config["d"]}'
        )
        return
    q_ref, k_ref, v_ref, o_ref, stats_ref = None, None, None, None, None

    if test_config["layout"] == "thd":
        q_ref = convert_thd_to_bshd(Q, seq_len_q, max_seq_len_q).float()
        k_ref = convert_thd_to_bshd(K, seq_len_kv, max_seq_len_kv).float()
        v_ref = convert_thd_to_bshd(V, seq_len_kv, max_seq_len_kv).float()
    else:
        q_ref = Q.float()
        k_ref = K.float()
        v_ref = V.float()
    o_ref, stats_ref = run_ref_nsa_swa(
        q_ref,
        k_ref,
        v_ref,
        attn_scale=test_config["scale_softmax"],
        padding=(seq_len_q, seq_len_kv) if test_config["layout"] == "thd" else None,
        left_bound=test_config["window_size"],
        right_bound=0,
        generate_stats=True,
    )

    if test_config["layout"] == "thd":
        total_seq_len_q = torch.sum(seq_len_q).item()
        o_ref = convert_bshd_to_thd(o_ref, seq_len_q, total_seq_len_q)
        stats_ref = convert_bshd_to_thd(stats_ref, seq_len_q, total_seq_len_q)

    o_ref = o_ref.to(dtype=test_config["dtype"])

    torch.testing.assert_close(O, o_ref, atol=0.01, rtol=1e-05)
    torch.testing.assert_close(Stats, stats_ref, atol=0.01, rtol=1e-05)
