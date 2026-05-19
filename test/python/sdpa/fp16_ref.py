import torch
import math
import cudnn

# fmt: off

def compute_ref(
    q,
    k,
    v,
    attn_scale=None,
    bias=None,
    block_mask=None,
    is_alibi=False,
    padding=None,
    diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    left_bound=None,
    right_bound=None,
    dropout_prob=0.0,
    dropout_mask=None,
    sink_token=None,
    torch_type=torch.float16,
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

    # Handle sink token
    has_sink_token = sink_token is not None
    if has_sink_token:
        sink_token = sink_token.to(dtype=torch.float32, device=device)

    # Generate padding masks and zero out padded positions in Q, K, V
    q_row_mask = None
    if padding is not None:
        q_mask = torch.zeros(b, 1, s_q, 1, dtype=torch.bool, device=device)
        k_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        v_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        seq_len_q, seq_len_kv = padding
        for i, (m, n) in enumerate(zip(seq_len_q, seq_len_kv)):
            q_mask[i, :, m:, :] = True
            k_mask[i, :, n:, :] = True
            v_mask[i, :, n:, :] = True

        q_row_mask = q_mask
        q = q.masked_fill(q_mask, 0.0)
        k = k.masked_fill(k_mask, 0.0)
        v = v.masked_fill(v_mask, 0.0)

    # Build combined_bias tensor encoding all masks as -inf
    combined_bias = torch.zeros((b, h_q, s_q, s_kv), dtype=torch.float32, device=device)

    if bias is not None:
        combined_bias = combined_bias + bias

    if is_alibi:
        index_row = torch.arange(s_q, dtype=torch.float32, device=device).view(-1, 1)
        index_col = torch.arange(s_kv, dtype=torch.float32, device=device)
        distance = index_col - index_row

        n = 2 ** math.floor(math.log2(h_q))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n))
        if n < h_q:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (h_q - n), 2))
            m = torch.cat([m, m_hat])
        m = m.view(1, -1, 1, 1).to(device=device)
        alibi_mask = distance.to(dtype=torch.float32) * m
        combined_bias = combined_bias + alibi_mask

    if padding is not None:
        kv_col_mask = torch.zeros(b, 1, 1, s_kv, dtype=torch.bool, device=device)
        seq_len_q, seq_len_kv = padding
        for i, (_, n) in enumerate(zip(seq_len_q, seq_len_kv)):
            kv_col_mask[i, :, :, n:] = True
        combined_bias = combined_bias.masked_fill(kv_col_mask, float("-inf"))

    if diag_align == diag_align.TOP_LEFT and right_bound is not None:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1 + right_bound)
        combined_bias = combined_bias.masked_fill(causal_mask, float("-inf"))
    elif diag_align == diag_align.BOTTOM_RIGHT and right_bound is not None:
        if padding:
            causal_mask_bottom_right = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
            seq_len_q, seq_len_kv = padding
            for i in range(b):
                causal_mask_bottom_right[i, :, :, :].triu_(diagonal=seq_len_kv[i] - seq_len_q[i] + 1 + right_bound)
        else:
            causal_mask_bottom_right = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
            causal_mask_bottom_right.triu_(diagonal=s_kv - s_q + 1 + right_bound)
        combined_bias = combined_bias.masked_fill(causal_mask_bottom_right, float("-inf"))

    if left_bound is not None:
        assert diag_align is not None
        if diag_align == diag_align.TOP_LEFT:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
            swa_mask.tril_(diagonal=-1 * left_bound)
        elif diag_align == diag_align.BOTTOM_RIGHT:
            if padding:
                swa_mask = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
                seq_len_q, seq_len_kv = padding
                for i in range(b):
                    swa_mask[i, :, :, :].tril_(diagonal=seq_len_kv[i] - seq_len_q[i] - left_bound)
            else:
                swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
                swa_mask.tril_(diagonal=-1 * left_bound + (s_kv - s_q))
        combined_bias = combined_bias.masked_fill(swa_mask, float("-inf"))

    if block_mask is not None:
        TILE_M = 128
        TILE_N = 128
        block_mask = block_mask.to(dtype=torch.uint8, device=device)
        block_mask = ((block_mask[..., None] & (1 << torch.arange(8, device=block_mask.device))) != 0).reshape(block_mask.shape[0], block_mask.shape[1], block_mask.shape[2], block_mask.shape[3] * 8)
        block_mask = block_mask.unsqueeze(3).unsqueeze(5)
        block_mask = block_mask.repeat(1, 1, 1, TILE_M, 1, TILE_N)
        block_mask = block_mask.reshape(block_mask.shape[0], block_mask.shape[1], block_mask.shape[2] * TILE_M, block_mask.shape[4] * TILE_N)
        block_mask = block_mask[:, :, :s_q, :s_kv]
        combined_bias += torch.where(block_mask, torch.tensor(0.0), torch.tensor(float('-inf')))

    block_size = 128
    num_blocks = (s_kv + block_size - 1) // block_size

    if has_sink_token:
        m_old = sink_token.expand(b, h_q, s_q, 1).clone()
        l_old = torch.ones((b, h_q, s_q, 1), dtype=torch.float32, device=device)
    else:
        m_old = torch.full((b, h_q, s_q, 1), float('-inf'), dtype=torch.float32, device=device)
        l_old = torch.zeros((b, h_q, s_q, 1), dtype=torch.float32, device=device)
    o = torch.zeros((b, h_q, s_q, d_v), dtype=torch.float32, device=device)

    for j in range(num_blocks):
        start_idx = j * block_size
        end_idx = min((j + 1) * block_size, s_kv)
        k_block = k[:, :, start_idx:end_idx, :]
        v_block = v[:, :, start_idx:end_idx, :]

        s_block = torch.einsum("bhqd,bhkd->bhqk", q, k_block)
        if attn_scale is not None:
            s_block = s_block * attn_scale
        s_block = s_block + combined_bias[:, :, :, start_idx:end_idx]

        if q_row_mask is not None:
            s_block = s_block.masked_fill(q_row_mask, float('-inf'))

        m_block = s_block.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_old, m_block)

        correction = torch.exp(m_old - m_new).nan_to_num()
        o = o * correction
        l_old = l_old * correction

        p_block = torch.exp(s_block - m_new).nan_to_num().to(torch_type).float()

        if q_row_mask is not None:
            p_block = p_block.masked_fill(q_row_mask, 0.0)

        l_new = l_old + p_block.sum(dim=-1, keepdim=True)

        # apply dropout mask over softmax outputs
        if dropout_prob != 0.0:
            assert dropout_mask is not None, "PyTorch reference must have dropout_mask for dropout"
            dropout_mask_block = dropout_mask[:, :, :, start_idx:end_idx]
            p_block = (p_block * dropout_mask_block) / (1 - dropout_prob)

        o = o + torch.einsum("bhqk,bhkd->bhqd", p_block, v_block)
        m_old = m_new
        l_old = l_new

    o_ref = o / l_old.clamp(min=1.0)
    o_ref = o_ref.to(torch_type).float()

    score_max_ref = m_old
    score_sum_exp_ref = l_old
    stats_ref = torch.log(score_sum_exp_ref) + score_max_ref

    return o_ref, stats_ref, score_max_ref, score_sum_exp_ref


def compute_ref_backward(
    q,
    k,
    v,
    o,
    dO,
    attn_scale=None,
    bias=None,
    is_alibi=False,
    padding=None,
    diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    left_bound=None,
    right_bound=None,
    dropout_prob=0.0,
    dropout_mask=None,
    sink_token=None,
    torch_type=torch.float16,
    device="cuda",
):
    b, h_q, s_q, d_qk = q.shape
    _, h_k, s_kv, _ = k.shape
    _, h_v, _, d_v = v.shape

    # use float32 datatype and math for reference computation
    q = q.to(dtype=torch.float32, device=device)
    k = k.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)
    o = o.to(dtype=torch.float32, device=device)
    dO = dO.to(dtype=torch.float32, device=device)

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

    has_sink_token = sink_token is not None
    if has_sink_token:
        sink_token = sink_token.to(dtype=torch.float32, device=device)

    # Generate padding masks and zero out padded positions
    q_row_mask = None
    p_mask = None
    if padding is not None:
        q_mask = torch.zeros(b, 1, s_q, 1, dtype=torch.bool, device=device)
        k_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        v_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        p_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        s_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        seq_len_q, seq_len_kv = padding
        for i, (m, n) in enumerate(zip(seq_len_q, seq_len_kv)):
            q_mask[i, :, m:, :] = True
            k_mask[i, :, n:, :] = True
            v_mask[i, :, n:, :] = True
            s_mask[i, :, :, n:] = True
            p_mask[i, :, m:, :] = True

        q_row_mask = q_mask
        q = q.masked_fill(q_mask, 0.0)
        k = k.masked_fill(k_mask, 0.0)
        v = v.masked_fill(v_mask, 0.0)

    # Compute attention scores
    s = torch.einsum("bhqd,bhkd->bhqk", q, k)
    if attn_scale is not None:
        s = s * attn_scale

    # Apply masks in same order as forward
    if bias is not None:
        s = s + bias

    if is_alibi:
        index_row = torch.arange(s_q, dtype=torch.float32, device=device).view(-1, 1)
        index_col = torch.arange(s_kv, dtype=torch.float32, device=device)
        distance = index_col - index_row

        n = 2 ** math.floor(math.log2(h_q))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n))
        if n < h_q:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (h_q - n), 2))
            m = torch.cat([m, m_hat])
        m = m.view(1, -1, 1, 1).to(device=device)
        alibi_mask = distance.to(dtype=torch.float32) * m
        s = s + alibi_mask

    if padding is not None:
        s = s.masked_fill(s_mask, float("-inf"))

    if diag_align == diag_align.TOP_LEFT and right_bound is not None:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1 + right_bound)
        s = s.masked_fill(causal_mask, float("-inf"))
    elif diag_align == diag_align.BOTTOM_RIGHT and right_bound is not None:
        if padding:
            causal_mask_bottom_right = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
            seq_len_q, seq_len_kv = padding
            for i in range(b):
                causal_mask_bottom_right[i, :, :, :].triu_(diagonal=seq_len_kv[i] - seq_len_q[i] + 1 + right_bound)
        else:
            causal_mask_bottom_right = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
            causal_mask_bottom_right.triu_(diagonal=s_kv - s_q + 1 + right_bound)
        s = s.masked_fill(causal_mask_bottom_right, float("-inf"))

    if left_bound is not None:
        assert diag_align is not None
        if diag_align == diag_align.TOP_LEFT:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
            swa_mask.tril_(diagonal=-1 * left_bound)
        elif diag_align == diag_align.BOTTOM_RIGHT:
            if padding:
                swa_mask = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
                seq_len_q, seq_len_kv = padding
                for i in range(b):
                    swa_mask[i, :, :, :].tril_(diagonal=seq_len_kv[i] - seq_len_q[i] - left_bound)
            else:
                swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
                swa_mask.tril_(diagonal=-1 * left_bound + (s_kv - s_q))
        s = s.masked_fill(swa_mask, float("-inf"))

    # Compute softmax with optional sink token
    p_sink = None
    if has_sink_token:
        sink_expanded = sink_token.expand(b, h_q, s_q, 1)
        s_extended = torch.cat([sink_expanded, s], dim=-1)
        p_extended = s_extended.softmax(dim=-1).nan_to_num()
        p_sink = p_extended[:, :, :, 0:1]
        p = p_extended[:, :, :, 1:]
    else:
        p = s.softmax(dim=-1).nan_to_num()

    all_inf = torch.isneginf(s).all(dim=-1, keepdim=True)
    if torch.any(all_inf):
        p = torch.where(all_inf, torch.zeros_like(p), p)

    if padding is not None:
        p = p.masked_fill(p_mask, 0.0)
        if p_sink is not None:
            p_sink = p_sink.masked_fill(p_mask[:, :, :, 0:1], 0.0)

    # Apply dropout to P
    if dropout_prob != 0.0:
        assert dropout_mask is not None, "PyTorch reference must have dropout_mask for dropout"
        p_dropped = (p * dropout_mask) / (1 - dropout_prob)
    else:
        p_dropped = p
    
    p_dropped = p_dropped.to(torch_type).float()

    # D = sum(o * dO, dim=-1)
    D = (o * dO).sum(dim=-1, keepdim=True)

    # dP = dO @ V^T, then apply dropout mask
    dP = torch.einsum("bhqd,bhkd->bhqk", dO, v)
    if dropout_prob != 0.0:
        dP = (dP * dropout_mask) / (1 - dropout_prob)

    # dS = P * (dP - D) * attn_scale
    dS = p * (dP - D)
    if attn_scale is not None:
        dS = dS * attn_scale
    dS = dS.to(torch_type).float()

    # dBias = dS / attn_scale (undo the scale baked into dS), summed over batch
    dBias = None
    if bias is not None:
        dBias_raw = p * (dP - D)  # dS without attn_scale
        dBias = dBias_raw.sum(dim=0, keepdim=True)

    # dSink_token: gradient for sink token
    dSink_token = None
    if has_sink_token:
        dS_sink = -p_sink * D
        dSink_token = dS_sink.sum(dim=(0, 2), keepdim=True)

    # dQ = dS @ K
    dQ = torch.einsum("bhqk,bhkd->bhqd", dS, k)

    # dK = dS^T @ Q
    dK = torch.einsum("bhqk,bhqd->bhkd", dS, q)

    # dV = P_dropped^T @ dO
    dV = torch.einsum("bhqk,bhqd->bhkd", p_dropped, dO)

    # GQA reduction for dK and dV
    if h_q != h_k:
        dK = dK.reshape(b, h_k, h_q // h_k, s_kv, d_qk).sum(dim=2)
    if h_q != h_v:
        dV = dV.reshape(b, h_v, h_q // h_v, s_kv, d_v).sum(dim=2)

    dQ = dQ.to(torch_type).float()
    dK = dK.to(torch_type).float()
    dV = dV.to(torch_type).float()
    if dBias is not None:
        dBias = dBias.to(torch_type).float()

    return dQ, dK, dV, dBias, dSink_token
