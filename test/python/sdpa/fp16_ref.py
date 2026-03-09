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

    # Handle sink token - a scalar attention score per head that competes with Q*K scores
    # sink_token shape: (1, h_q, 1, 1) - a scalar per head
    # The sink absorbs attention but contributes zero to the output (no V for sink)
    # This affects softmax: M = max(max_k(s_k), sink), p_i = exp(s_i - M) / (sum_k(exp(s_k - M)) + exp(sink - M))
    has_sink_token = sink_token is not None
    if has_sink_token:
        sink_token = sink_token.to(dtype=torch.float32, device=device)

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

    if diag_align == diag_align.TOP_LEFT and right_bound is not None:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1 + right_bound)
        s = s.masked_fill(causal_mask, float("-inf"))
    elif diag_align == diag_align.BOTTOM_RIGHT and right_bound is not None:
        causal_mask_bottom_right = None
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
            # BRCM + SWA for variable sequence lengths
            if padding:
                swa_mask = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
                seq_len_q, seq_len_kv = padding
                for i in range(b):
                    swa_mask[i, :, :, :].tril_(diagonal=seq_len_kv[i] - seq_len_q[i] - left_bound)
            # BRCM + SWA for fixed sequence lengths
            else:
                swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
                swa_mask.tril_(diagonal=-1 * left_bound + (s_kv - s_q))
        s = s.masked_fill(swa_mask, float("-inf"))

    if block_mask is not None:
        TILE_M = 128
        TILE_N = 128

        block_mask = block_mask.to(dtype=torch.uint8, device=device)
        block_mask = ((block_mask[..., None] & (1 << torch.arange(8, device=block_mask.device))) != 0).reshape(block_mask.shape[0], block_mask.shape[1], block_mask.shape[2], block_mask.shape[3] * 8)
        block_mask = block_mask.unsqueeze(3).unsqueeze(5)
        block_mask = block_mask.repeat(1, 1, 1, TILE_M, 1, TILE_N)
        block_mask = block_mask.reshape(block_mask.shape[0], block_mask.shape[1], block_mask.shape[2] * TILE_M, block_mask.shape[4] * TILE_N)
        block_mask = block_mask[:, :, :s_q, :s_kv]
        s += torch.where(block_mask, torch.tensor(0.0), torch.tensor(float('-inf')))

    # Compute softmax with optional sink_token, score_max and score_sum_exp (last two are for muon clip support)
    # * sink_token is a scalar per head that competes with Q*K scores in softmax
    #   but contributes zero to output (no V for sink), shape (1, h_q, 1, 1)
    # * score_max: global max of attention scores per batch-head pair, shape (b, h_q, s_q, 1)
    # * score_sum_exp: sum of exp(s - score_max) across all positions, shape (b, h_q, s_q, 1)
    score_max_ref = s.amax(dim=-1, keepdim=True)  # (b, h_q, s_q, 1)
    if has_sink_token:
        # sink_token shape: (1, h_q, 1, 1) -> broadcast to (b, h_q, s_q, 1)
        score_max_ref = torch.maximum(score_max_ref, sink_token)  # (b, h_q, s_q, 1)

    # Compute normalized exponentials
    s_normalized = torch.where(torch.isneginf(s), float('-inf'), s - score_max_ref)
    exp_s = torch.exp(s_normalized)  # (b, h_q, s_q, s_kv)

    # Sum across kv dimension for each query, then sum across query dimension
    score_sum_exp_ref = exp_s.sum(dim=(-1), keepdim=True)  # (b, h_q, s_q, 1)

    # Sum of exponentials including sink
    if has_sink_token:
        exp_sink = torch.exp(sink_token - score_max_ref)  # (b, h_q, s_q, 1)
        score_sum_exp_ref += exp_sink  # (b, h_q, s_q, 1)

    # Stats output
    stats_ref = torch.log(score_sum_exp_ref) + score_max_ref

    # Softmax probabilities (sink doesn't contribute to output, so no sink column in p)
    p = exp_s / score_sum_exp_ref  # (b, h_q, s_q, s_kv)

    all_inf = torch.isneginf(s).all(dim=-1, keepdim=True)
    if torch.any(all_inf):
        p = torch.where(all_inf, torch.zeros_like(p), p)

    if padding is not None:
        p = p.masked_fill(p_mask, 0.0)

    # apply dropout mask over softmax outputs
    if dropout_prob != 0.0:
        assert dropout_mask != None, "PyTorch reference must have dropout_mask for dropout"
        p = (p * dropout_mask) / (1 - dropout_prob)

    o_ref = torch.einsum("bhqk,bhkd->bhqd", p, v)

    return o_ref, stats_ref, score_max_ref, score_sum_exp_ref
