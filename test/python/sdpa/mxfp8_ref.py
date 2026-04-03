import torch
import cudnn

# fmt: off

def compute_ref(q_fp8, k_fp8, v_fp8, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale, torch_itype=torch.float8_e4m3fn, output_type=torch.bfloat16,
                left_bound=None, right_bound=None, diag_align=None, sink_token=None):
    """
    Compute reference SDPA with MXFP8 dequantization.
    Takes FP8 inputs and converts to FP32 to match cuDNN behavior.
    Supports GQA/MQA where K and V have fewer heads than Q.

    If sink_token is provided (shape: 1, h_q, 1, 1), it's used as a virtual attention
    score that competes with Q*K scores in softmax but contributes zero to the output
    (no V for sink). This is implemented via online softmax initialization:
    m_old = sink_token, l_old = 1.
    """
    # Convert FP8 to FP32 (matches cuDNN's input handling)
    q_f32 = q_fp8.float()
    k_f32 = k_fp8.float()
    v_f32 = v_fp8.float()

    b, h_q, s_q, d_qk = q_f32.shape
    _, h_k, s_kv, _ = k_f32.shape
    _, h_v, _, d_vo = v_f32.shape

    # GQA: expand K, V to match Q's head count
    if h_k != h_q:
        assert h_q % h_k == 0, "h_q must be divisible by h_k for GQA"
        repeats = h_q // h_k
        k_f32 = k_f32.repeat_interleave(repeats, dim=1)
        sf_k_ref = sf_k_ref.view(b, h_k, *sf_k_ref.shape[1:]).repeat_interleave(repeats, dim=1).reshape(b * h_q, *sf_k_ref.shape[1:])

    if h_v != h_q:
        assert h_q % h_v == 0, "h_q must be divisible by h_v for GQA"
        repeats = h_q // h_v
        v_f32 = v_f32.repeat_interleave(repeats, dim=1)
        sf_v_ref = sf_v_ref.view(b, h_v, *sf_v_ref.shape[1:]).repeat_interleave(repeats, dim=1).reshape(b * h_q, *sf_v_ref.shape[1:])

    # Reshape for batch processing: [B, H, S, D] -> [B*H, S, D]
    q = q_f32.reshape(b * h_q, s_q, d_qk)
    k = k_f32.reshape(b * h_q, s_kv, d_qk)
    v = v_f32.reshape(b * h_q, s_kv, d_vo)

    # Dequantize Q and K (scale factors apply to d_qk dimension)
    q_dq = q * sf_q_ref
    k_dq = k * sf_k_ref
    # Dequantize V (scale factors apply to s_kv dimension)
    v_dq = v * sf_v_ref

    bias = torch.zeros((b * h_q, s_q, s_kv), dtype=torch.float32, device=q.device)
    if right_bound is not None and diag_align is not None:
        causal = torch.full((s_q, s_kv), float('-inf'), dtype=torch.float32, device=q.device)
        if diag_align == cudnn.diagonal_alignment.TOP_LEFT:
            causal = torch.triu(causal, diagonal=1 + right_bound)
        else:
            causal = torch.triu(causal, diagonal=s_kv - s_q + 1 + right_bound)
        bias = bias + causal.unsqueeze(0)
    if left_bound is not None and diag_align is not None:
        swa = torch.full((s_q, s_kv), float('-inf'), dtype=torch.float32, device=q.device)
        if diag_align == cudnn.diagonal_alignment.TOP_LEFT:
            swa = torch.tril(swa, diagonal=-1 * left_bound)
        else:
            swa = torch.tril(swa, diagonal=-1 * left_bound + (s_kv - s_q))
        bias = bias + swa.unsqueeze(0)

    block_size = 128
    num_blocks = (s_kv + block_size - 1) // block_size

    # Initialize online softmax state. If sink_token is present, use it as the
    # initial m_old with l_old=1, effectively adding a virtual attention score.
    if sink_token is not None:
        # sink_token shape: (1, h_q, 1, 1) -> expand to (b * h_q, s_q, 1)
        sink_expanded = sink_token.float().reshape(1, h_q, 1, 1).expand(b, h_q, s_q, 1)
        m_old = sink_expanded.reshape(b * h_q, s_q, 1).clone()
        l_old = torch.ones((b * h_q, s_q, 1), dtype=torch.float32, device=q.device)
    else:
        m_old = torch.full((b * h_q, s_q, 1), float('-inf'), dtype=torch.float32, device=q.device)
        l_old = torch.zeros((b * h_q, s_q, 1), dtype=torch.float32, device=q.device)
    o = torch.zeros((b * h_q, s_q, d_vo), dtype=torch.float32, device=q.device)

    for j in range(num_blocks):
        start_idx = j * block_size
        end_idx = min((j + 1) * block_size, s_kv)
        k_block = k_dq[:, start_idx:end_idx, :]
        v_block = v_dq[:, start_idx:end_idx, :]

        # Q (FP32) @ K^T (FP32) -> S (FP32)
        s_block = torch.einsum("bqd,bkd->bqk", q_dq, k_block) * attn_scale
        s_block = s_block + bias[:, :, start_idx:end_idx]

        m_block = s_block.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_old, m_block)

        correction = torch.exp(m_old - m_new).nan_to_num()
        o = o * correction
        l_old = l_old * correction

        p_block = torch.exp(s_block - m_new).nan_to_num()
        l_new = l_old + p_block.sum(dim=-1, keepdim=True)

        # P (FP32) -> P (FP8)
        p_block_quant = p_block.to(torch_itype).float()

        o = o + torch.einsum("bqk,bkd->bqd", p_block_quant, v_block)
        m_old = m_new
        l_old = l_new

    o = o / l_old.clamp(min=1.0)

    # O (FP32) -> O (output)
    o_ref = o.reshape(b, h_q, s_q, d_vo).to(output_type).float()

    stats = m_old + torch.log(l_old)
    stats_ref = stats.reshape(b, h_q, s_q, 1).float()

    return o_ref, stats_ref

def compute_ref_backward(q_fp8, q_t_fp8, k_fp8, k_t_fp8, v_fp8, o_f16, dO_f16, dO_fp8, dO_t_fp8, attn_scale,
                         sf_q_ref, sf_q_t_ref, sf_k_ref, sf_k_t_ref, sf_v_ref, sf_dO_ref, sf_dO_t_ref,
                         torch_itype=torch.float8_e4m3fn, torch_otype=torch.bfloat16,
                         left_bound=None, right_bound=None, diag_align=None, sink_token=None):
    """
    Compute backward pass reference for MXFP8 SDPA.

    If sink_token is provided, the virtual sink is included in softmax normalization
    and dSink_token is computed: dS_sink = -p_sink * D (no attn_scale), then summed
    over batch and query dimensions.
    """
    # Convert FP8 to FP32
    q_f32 = q_fp8.float()
    q_t_f32 = q_t_fp8.float()
    k_f32 = k_fp8.float()
    k_t_f32 = k_t_fp8.float()
    v_f32 = v_fp8.float()
    dO_f32 = dO_fp8.float()
    dO_t_f32 = dO_t_fp8.float()

    b, h_q, s_q, d_qk = q_f32.shape
    _, h_k, s_kv, _ = k_f32.shape
    _, h_v, _, d_vo = v_f32.shape

    # GQA: expand K, K_T, V and their scale factors to match Q's head count
    # Scale factors are in [B*H, S, D] format: reshape first dim as (B, H) to interleave heads
    if h_k != h_q:
        assert h_q % h_k == 0, "h_q must be divisible by h_k for GQA"
        repeats = h_q // h_k
        k_f32 = k_f32.repeat_interleave(repeats, dim=1)
        k_t_f32 = k_t_f32.repeat_interleave(repeats, dim=1)
        sf_k_ref = sf_k_ref.view(b, h_k, *sf_k_ref.shape[1:]).repeat_interleave(repeats, dim=1).reshape(b * h_q, *sf_k_ref.shape[1:])
        sf_k_t_ref = sf_k_t_ref.view(b, h_k, *sf_k_t_ref.shape[1:]).repeat_interleave(repeats, dim=1).reshape(b * h_q, *sf_k_t_ref.shape[1:])

    if h_v != h_q:
        assert h_q % h_v == 0, "h_q must be divisible by h_v for GQA"
        repeats = h_q // h_v
        v_f32 = v_f32.repeat_interleave(repeats, dim=1)
        sf_v_ref = sf_v_ref.view(b, h_v, *sf_v_ref.shape[1:]).repeat_interleave(repeats, dim=1).reshape(b * h_q, *sf_v_ref.shape[1:])

    # Reshape for batch processing: [B, H, S, D] -> [B*H_q, S, D]
    q = q_f32.reshape(b * h_q, s_q, d_qk)
    q_t = q_t_f32.reshape(b * h_q, s_q, d_qk)
    k = k_f32.reshape(b * h_q, s_kv, d_qk)
    k_t = k_t_f32.reshape(b * h_q, s_kv, d_qk)
    v = v_f32.reshape(b * h_q, s_kv, d_vo)
    dO = dO_f32.reshape(b * h_q, s_q, d_vo)
    dO_t = dO_t_f32.reshape(b * h_q, s_q, d_vo)

    # Dequantize for BMM1 (Q @ K^T): D-dimension scale factors
    q_dq = q * sf_q_ref
    k_dq = k * sf_k_ref

    # Dequantize for dO @ V^T: D-scale for dO, S-scale for V
    dO_dq = dO * sf_dO_ref
    v_dq = v * sf_v_ref

    # Dequantize for P^T @ dO_T -> dV: S-scale for dO_T
    dO_t_dq = dO_t * sf_dO_t_ref

    # Dequantize for dS @ K_T -> dQ: S-scale for K_T
    k_t_dq = k_t * sf_k_t_ref

    # Dequantize for dS^T @ Q_T -> dK: S-scale for Q_T
    q_t_dq = q_t * sf_q_t_ref

    s = torch.einsum("bqd,bkd->bqk", q_dq, k_dq) * attn_scale

    if right_bound is not None and diag_align is not None:
        if diag_align == cudnn.diagonal_alignment.TOP_LEFT:
            mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=s.device).triu(diagonal=1 + right_bound)
        else:
            mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=s.device).triu(diagonal=s_kv - s_q + 1 + right_bound)
        s = s.masked_fill(mask.unsqueeze(0), float('-inf'))
    if left_bound is not None and diag_align is not None:
        if diag_align == cudnn.diagonal_alignment.TOP_LEFT:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=s.device).tril(diagonal=-1 * left_bound)
        else:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=s.device).tril(diagonal=-1 * left_bound + (s_kv - s_q))
        s = s.masked_fill(swa_mask.unsqueeze(0), float('-inf'))

    # If sink_token is present, prepend it as a virtual attention score
    p_sink = None
    if sink_token is not None:
        # sink_token shape: (1, h_q, 1, 1) -> expand to (b * h_q, s_q, 1)
        sink_expanded = sink_token.float().reshape(1, h_q, 1, 1).expand(b, h_q, s_q, 1)
        sink_3d = sink_expanded.reshape(b * h_q, s_q, 1)
        s_extended = torch.cat([sink_3d, s], dim=-1)  # (b * h_q, s_q, 1 + s_kv)
        p_extended = s_extended.softmax(dim=-1).nan_to_num().float()
        p_sink = p_extended[:, :, 0:1]  # probability for sink (b * h_q, s_q, 1)
        p = p_extended[:, :, 1:]  # probabilities for actual K positions (b * h_q, s_q, s_kv)
    else:
        p = s.softmax(dim=-1).nan_to_num().float()

    p_fp8 = p * 256.0
    p_fp8 = p_fp8.to(torch_itype).float()
    p_fp8 *= 1.0 / 256.0

    # Use BF16 inputs for D
    o_f16 = o_f16.float().reshape(b * h_q, s_q, d_vo)
    dO_f16 = dO_f16.float().reshape(b * h_q, s_q, d_vo)
    D = (o_f16 * dO_f16).sum(dim=-1, keepdim=True)

    # dO @ V -> dP
    dP = torch.einsum("bqd,bkd->bqk", dO_dq, v_dq)

    # dS = P * (dP - D) * attn_scale
    dS = p * (dP - D) * attn_scale

    # Compute dSink_token if sink_token was provided
    # Formula: dSink = -exp(sink - logsumexp) * D summed over batch and sequence
    # Note: attn_scale is NOT applied because sink_token is added directly to scores,
    # not multiplied by attn_scale like Q @ K.T
    dSink_token = None
    if sink_token is not None:
        dS_sink = -p_sink * D  # (b * h_q, s_q, 1)
        # Reshape to (b, h_q, s_q, 1) and sum over batch and query dims
        dS_sink_4d = dS_sink.reshape(b, h_q, s_q, 1)
        dSink_token = dS_sink_4d.sum(dim=(0, 2), keepdim=True)  # (1, h_q, 1, 1)

    from .mxfp8 import quantize_to_mxfp8
    s_q_padded = ((s_q + 127) // 128) * 128
    dS_4d = dS.reshape(b, h_q, s_q, s_kv)
    dS_fp8, sf_dS_ref, _, dS_fp8_t, sf_dS_t_ref, _ = quantize_to_mxfp8(
        dS_4d, b, h_q, s_q, s_kv, block_size=32, fp8_dtype=torch_itype
    )

    # D-quantized dS (along s_kv)
    dS_fp32 = dS_fp8.float().reshape(b * h_q, s_q, s_kv) * sf_dS_ref

    # S-quantized dS (along s_q)
    dS_fp32_t = dS_fp8_t.float().reshape(b * h_q, s_q, s_kv) * sf_dS_t_ref

    # P @ dO -> dV
    dV = torch.einsum("bqk,bqd->bkd", p_fp8, dO_t_dq)

    # dS @ K -> dQ
    dQ = torch.einsum("bqk,bkd->bqd", dS_fp32, k_t_dq)

    # dS^T @ Q -> dK
    dK = torch.einsum("bqk,bqd->bkd", dS_fp32_t, q_t_dq)

    # Handle GQA reduction for dK and dV (in FP32 before type conversion)
    if h_k != h_q:
        dK = dK.reshape(b, h_q, s_kv, d_qk)
        dK = dK.reshape(b, h_k, h_q // h_k, s_kv, d_qk).sum(dim=2)

    if h_v != h_q:
        dV = dV.reshape(b, h_q, s_kv, d_vo)
        dV = dV.reshape(b, h_v, h_q // h_v, s_kv, d_vo).sum(dim=2)

    # Reshape to [B, H, S, D] and convert to output type
    dQ = dQ.reshape(b, h_q, s_q, d_qk).to(torch_otype).float()
    dK = dK.reshape(b, h_k, s_kv, d_qk).to(torch_otype).float()
    dV = dV.reshape(b, h_v, s_kv, d_vo).to(torch_otype).float()

    return dQ, dK, dV, dSink_token
