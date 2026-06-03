import torch
import cudnn

from .helpers import get_fp8_scale_factor, get_fp8_descale_factor

# fmt: off


def compute_ref(q, k, v, attn_scale,
                q_descale, k_descale, v_descale,
                s_scale, s_descale, torch_itype,
                torch_otype,
                padding=None, bias=None,
                left_bound=None, right_bound=None, diag_align=None, sink_token=None,
                rescale_threshold=0.0):
    """Compute forward pass reference with online softmax tiling.
    Returns (o_quant, stats, o_amax)."""
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape

    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    q = q.float()
    k = k.float()
    v = v.float()

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q_row_mask = None
    kv_col_mask = None
    if padding is not None:
        seq_len_q_pad, seq_len_kv_pad = padding
        q_mask_bhsd = torch.zeros((len(seq_len_q_pad), 1, s_q, 1), dtype=torch.bool, device=q.device)
        kv_mask_bhsd = torch.zeros((len(seq_len_kv_pad), 1, s_kv, 1), dtype=torch.bool, device=q.device)
        kv_col_mask = torch.zeros((len(seq_len_kv_pad), 1, 1, s_kv), dtype=torch.bool, device=q.device)

        for i, (m, n) in enumerate(zip(seq_len_q_pad, seq_len_kv_pad)):
            q_mask_bhsd[i, :, m:, :] = True
            kv_mask_bhsd[i, :, n:, :] = True
            kv_col_mask[i, :, :, n:] = True

        q_row_mask = q_mask_bhsd

        q = q.masked_fill(q_mask_bhsd, 0.0)
        k = k.masked_fill(kv_mask_bhsd, 0.0)
        v = v.masked_fill(kv_mask_bhsd, 0.0)

    # Build combined_bias (shape: b, h_q, s_q, s_kv) before the tiled loop.
    # Masking is encoded as -inf so it survives the per-block slicing.
    combined_bias = torch.zeros((b, h_q, s_q, s_kv), dtype=torch.float32, device=q.device)
    if bias is not None:
        combined_bias = combined_bias + bias.float()
    if right_bound is not None and diag_align is not None:
        causal = torch.full((s_q, s_kv), float('-inf'), dtype=torch.float32, device=q.device)
        if diag_align == cudnn.diagonal_alignment.TOP_LEFT:
            causal = torch.triu(causal, diagonal=1 + right_bound)
        else:
            causal = torch.triu(causal, diagonal=s_kv - s_q + 1 + right_bound)
        combined_bias = combined_bias + causal
    if left_bound is not None and diag_align is not None:
        swa = torch.full((s_q, s_kv), float('-inf'), dtype=torch.float32, device=q.device)
        if diag_align == cudnn.diagonal_alignment.TOP_LEFT:
            swa = torch.tril(swa, diagonal=-1 * left_bound)
        else:
            swa = torch.tril(swa, diagonal=-1 * left_bound + (s_kv - s_q))
        combined_bias = combined_bias + swa

    block_size = 128
    num_blocks = (s_kv + block_size - 1) // block_size

    # Initialize online softmax state. If sink_token is present, use it as the
    # initial m_old with l_old=1, effectively adding a virtual attention score.
    if sink_token is not None:
        m_old = sink_token.float().expand(b, h_q, s_q, 1).clone()
        l_old = torch.ones((b, h_q, s_q, 1), dtype=torch.float32, device=q.device)
    else:
        m_old = torch.full((b, h_q, s_q, 1), float('-inf'), dtype=torch.float32, device=q.device)
        l_old = torch.zeros((b, h_q, s_q, 1), dtype=torch.float32, device=q.device)
    o = torch.zeros((b, h_q, s_q, d_v), dtype=torch.float32, device=q.device)

    for j in range(num_blocks):
        start_idx = j * block_size
        end_idx = min((j + 1) * block_size, s_kv)
        k_block = k[:, :, start_idx:end_idx, :]
        v_block = v[:, :, start_idx:end_idx, :]

        # Q (FP8) @ K^T (FP8) -> S (FP32)
        s_block = torch.einsum("bhqd,bhkd->bhqk", q.float(), k_block.float()) * q_descale * k_descale * attn_scale
        s_block = s_block + combined_bias[:, :, :, start_idx:end_idx]

        if padding is not None:
            s_block = s_block.masked_fill(q_row_mask, float('-inf'))
            s_block = s_block.masked_fill(kv_col_mask[:, :, :, start_idx:end_idx], float('-inf'))

        m_block = s_block.max(dim=-1, keepdim=True).values

        NEG_INF = float('-inf')
        is_first = (m_old == NEG_INF)
        exceeds_threshold = (m_block - m_old > rescale_threshold)
        should_update = is_first | exceeds_threshold
        m_new = torch.where(should_update, m_block, m_old)

        exp_input = m_old - m_new
        needs_correction = (exp_input < -rescale_threshold)
        correction = torch.where(needs_correction, torch.exp(exp_input), torch.ones_like(exp_input))
        correction = correction.nan_to_num()

        o = o * correction
        l_old = l_old * correction

        p_block = torch.exp(s_block - m_new).nan_to_num()
        if q_row_mask is not None:
            p_block = p_block.masked_fill(q_row_mask, 0.0)
        l_new = l_old + p_block.sum(dim=-1, keepdim=True)

        # P (FP32) -> P (FP8)
        s_scale_effective = s_scale * (2.0 ** (-rescale_threshold))
        s_descale_effective = s_descale * (2.0 ** rescale_threshold)
        p_block_quant = ((p_block * s_scale_effective).to(torch_itype)).float()

        o = o + torch.einsum("bhqk,bhkd->bhqd", p_block_quant, v_block.float()) * v_descale * s_descale_effective
        m_old = m_new
        l_old = l_new

    o = o / l_old.clamp(min=1.0)
    stats = (m_old + torch.log(l_old)).float()
    o = o.transpose(1, 2)

    o_amax = o.abs().max().item()
    o_scale = get_fp8_scale_factor(o_amax, torch_otype)
    o_quant = (o * o_scale).to(torch_otype)

    return o_quant, stats, o_amax

def compute_ref_backward(q, k, v, o, dO, attn_scale,
                         q_descale, k_descale, v_descale,
                         s_scale, s_descale, torch_itype,
                         o_descale, dO_descale,
                         torch_otype,
                         padding=None, bias=None,
                         left_bound=None, right_bound=None, diag_align=None, sink_token=None):
    """Compute backward pass reference.
    Returns (dQ, dK, dV, dSink_token, dP_amax, dQ_amax, dK_amax, dV_amax)."""
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, _ = v.shape

    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    q = q.float()
    k = k.float()
    v = v.float()

    q_row_mask = None
    p_mask = None
    if padding is not None:
        seq_len_q_pad, seq_len_kv_pad = padding
        q_mask_bhsd = torch.zeros((len(seq_len_q_pad), 1, s_q, 1), dtype=torch.bool, device=q.device)
        kv_mask_bhsd = torch.zeros((len(seq_len_kv_pad), 1, s_kv, 1), dtype=torch.bool, device=q.device)
        p_mask = torch.zeros((len(seq_len_kv_pad), 1, 1, s_kv), dtype=torch.bool, device=q.device)

        for i, (m, n) in enumerate(zip(seq_len_q_pad, seq_len_kv_pad)):
            q_mask_bhsd[i, :, m:, :] = True
            kv_mask_bhsd[i, :, n:, :] = True
            p_mask[i, :, :, n:] = True

        q_row_mask = q_mask_bhsd

        q = q.masked_fill(q_mask_bhsd.transpose(1, 2), 0.0)
        k = k.masked_fill(kv_mask_bhsd.transpose(1, 2), 0.0)
        v = v.masked_fill(kv_mask_bhsd.transpose(1, 2), 0.0)

    # Compute P from Q and K
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * q_descale * k_descale * attn_scale

    if padding is not None:
        s = s.masked_fill(q_row_mask, float('-inf'))
        s = s.masked_fill(p_mask, float('-inf'))

    if bias is not None:
        s = s + bias.float()
    if right_bound is not None and diag_align is not None:
        if diag_align == cudnn.diagonal_alignment.TOP_LEFT:
            causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device)
            causal_mask.triu_(diagonal=1 + right_bound)
        else:
            causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device)
            causal_mask.triu_(diagonal=s_kv - s_q + 1 + right_bound)
        s = s.masked_fill(causal_mask, float('-inf'))
    if left_bound is not None and diag_align is not None:
        if diag_align == cudnn.diagonal_alignment.TOP_LEFT:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device)
            swa_mask.tril_(diagonal=-1 * left_bound)
        else:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device)
            swa_mask.tril_(diagonal=-1 * left_bound + (s_kv - s_q))
        s = s.masked_fill(swa_mask, float('-inf'))

    # If sink_token is present, prepend it as a virtual attention score
    p_sink = None
    if sink_token is not None:
        sink_expanded = sink_token.float().expand(b, h_q, s_q, 1)
        s_extended = torch.cat([sink_expanded, s], dim=-1)
        p_extended = s_extended.softmax(dim=-1).nan_to_num()
        p_sink = p_extended[:, :, :, 0:1]
        p = p_extended[:, :, :, 1:]
    else:
        p = s.softmax(dim=-1).nan_to_num()

    if q_row_mask is not None:
        p = p.masked_fill(q_row_mask, 0.0)
        if p_sink is not None:
            p_sink = p_sink.masked_fill(q_row_mask, 0.0)

    # P (FP32) -> P (FP8)
    p_quant = (p * s_scale).to(torch_itype)

    # P (FP8) @ dO (FP8) -> dV (FP32)
    dV = torch.einsum("bhqk,bqhd->bkhd", p_quant.float(), dO.float()) * s_descale * dO_descale

    # dO (FP8) @ V (FP8) -> dP (FP32)
    dP = torch.einsum("bqhd,bkhd->bhqk", dO.float(), v.float()) * dO_descale * v_descale

    # Compute dS
    o_float = o.float()
    dO_float = dO.float()
    D = (o_float * dO_float).sum(dim=-1, keepdim=True).transpose(1, 2) * o_descale * dO_descale
    dS = p * (dP - D) * attn_scale

    # Compute dSink_token if sink_token was provided
    # Formula: dSink = -exp(sink - logsumexp) * D summed over batch and sequence
    # Note: attn_scale is NOT applied here because sink_token is added directly to scores,
    # not multiplied by attn_scale like Q @ K.T
    dSink_token = None
    if sink_token is not None:
        dS_sink = -p_sink * D
        dSink_token = dS_sink.sum(dim=(0, 2), keepdim=True)

    dP_amax = dP.abs().max().item()
    dP_scale = get_fp8_scale_factor(dP_amax, torch_otype)
    dP_descale = get_fp8_descale_factor(dP_amax, torch_itype)

    # dS (FP32) -> dS (FP8)
    dS_quant = ((dS * dP_scale).to(torch_itype)).float()

    # dS (FP8) @ K (FP8) -> dQ (FP32)
    dQ = torch.einsum("bhqk,bkhd->bqhd", dS_quant, k.float()) * k_descale * dP_descale

    # dS^T (FP8) @ Q (FP8) -> dK (FP32)
    dK = torch.einsum("bhqk,bqhd->bkhd", dS_quant, q.float()) * q_descale * dP_descale

    # Handle GQA reduction
    if h_q != h_k:
        dK = dK.reshape(dK.shape[0], dK.shape[1], h_k, h_q // h_k, dK.shape[3]).sum(dim=3)
    if h_q != h_v:
        dV = dV.reshape(dV.shape[0], dV.shape[1], h_v, h_q // h_v, dV.shape[3]).sum(dim=3)

    dQ_amax = dQ.abs().max().item()
    dK_amax = dK.abs().max().item()
    dV_amax = dV.abs().max().item()

    # dQ (FP32) -> dQ (FP8)
    dQ = (dQ * get_fp8_scale_factor(dQ_amax, torch_otype)).to(torch_otype)
    # dK (FP32) -> dK (FP8)
    dK = (dK * get_fp8_scale_factor(dK_amax, torch_otype)).to(torch_otype)
    # dV (FP32) -> dV (FP8)
    dV = (dV * get_fp8_scale_factor(dV_amax, torch_otype)).to(torch_otype)

    return dQ, dK, dV, dSink_token, dP_amax, dQ_amax, dK_amax, dV_amax
