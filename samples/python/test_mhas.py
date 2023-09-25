import cudnn
import pytest
import torch
import math

import itertools
import random


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


def make_tensor_attr(graph, torch_tensor, name="", dim=None, stride=None, is_pass_by_value=None):
    return graph.tensor(
        name=name,
        dim=dim if dim else torch_tensor.size(),
        stride=stride if stride else torch_tensor.stride(),
        data_type=convert_to_cudnn_type(torch_tensor.dtype),
        is_pass_by_value=is_pass_by_value,
    )


def compare_tensors(expected, actual, tensor_name, rtol=2e-2, atol=2e-2, fudge=1e-9, print_compare=False):
    assert expected.shape == actual.shape

    expected = expected.to(dtype=torch.float64, device="cuda").flatten()
    actual = actual.to(dtype=torch.float64, device="cuda").flatten()

    n_elem = torch.numel(expected)

    mae = (expected - actual).abs().mean().item()
    perr = ((expected - actual).abs().sum() / expected.abs().sum()).item()
    snr = (expected**2).mean().sqrt() / ((expected - actual) ** 2).mean().sqrt()
    snr_db = (10 * torch.log10(snr)).item()

    absolute_error = (expected - actual).abs()
    relative_error = absolute_error / torch.where(expected.abs() < fudge, fudge, expected.abs())

    abs_error_indices = absolute_error > atol
    rel_error_indices = relative_error > rtol
    n_abs_errors = torch.sum(abs_error_indices)
    n_rel_errors = torch.sum(rel_error_indices)
    error_indices = torch.logical_and(abs_error_indices, rel_error_indices)
    n_errors = torch.sum(error_indices)

    n_nans = torch.isnan(actual).sum()
    n_zeros = n_elem - torch.count_nonzero(actual)

    if print_compare or n_errors != 0:
        print(f"========== {tensor_name} ==========")
        print(f"Absolute Tolerance = {atol}")
        print(f"Relative Tolerance = {rtol}")
        print(f"Number of elements = {n_elem}")
        print(f"Number of absolute errors = {n_abs_errors} ({n_abs_errors * 100 / n_elem:.2f}%)")
        print(f"Number of relative errors = {n_rel_errors} ({n_rel_errors * 100 / n_elem:.2f}%)")
        print(f"Number of errors (absolute and relative) = {n_errors} ({(n_errors * 100)/n_elem:.2f}%)")
        print(f"Maximum absolute error = {absolute_error.max():.4f}")
        print(f"Maximum relative error = {relative_error.max():.4f}")
        print(f"Mean average error = {mae:.4f}")
        print(f"Perr error = {perr:.4f} = 1/{1/perr:.2f}")
        print(f"Signal to noise ratio = {snr.item():.2f} = {snr_db:.2f}dB")
        print(f"Number of Nans = {n_nans} ({n_nans * 100 / n_elem:.2f}%)")
        print(f"Number of Zeros = {n_zeros} ({n_zeros * 100 / n_elem:.2f}%)")
        print("===================================\n")

    return n_errors


def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$\frac{1}{2^{\frac{8}{n}}} = 2^{-\frac{8}{n}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$$
    """

    # Get the closest power of 2 to `n_heads`.
    # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
    # and then add the remaining slopes.
    n = 2 ** math.floor(math.log2(n_heads))
    # $2^{-\frac{8}{n}}$
    m_0 = 2.0 ** (-8.0 / n)
    # $2^{-1\frac{8}{n}}, 2^{-2 \frac{8}{n}}, 2^{-3 \frac{8}{n}}, \dots$
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    # If `n_heads` is not a power of 2, then we add the remaining slopes.
    # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
    # And pick the slopes upto `n_heads`.
    if n < n_heads:
        # $2^{-\frac{8}{2n}}$
        m_hat_0 = 2.0 ** (-4.0 / n)
        # $2^{-1\frac{8}{2n}}, 2^{-3 \frac{8}{2n}}, 2^{-5 \frac{8}{2n}}, \dots$
        # Note that we take steps by $2$ to avoid slopes added previously.
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        # Concatenate the slopes with the remaining slopes.
        m = torch.cat([m, m_hat])

    # Reshape the tensor to [1, num_heads, 1, 1]
    m = m.view(1, -1, 1, 1).to(device="cuda")

    return m


def compute_o_stats(q, k, v, attn_scale=1.0, bias=None, is_alibi=False, padding=None, is_causal=False, device="cuda"):
    b, h, s_q, d = q.shape
    _, _, s_kv, _ = k.shape

    assert k.shape == (b, h, s_kv, d)
    assert v.shape == (b, h, s_kv, d)

    if padding is not None:
        seq_len_q, seq_len_kv = padding
        q_mask = torch.zeros(b, 1, s_q, 1, dtype=torch.bool, device=device)
        k_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        v_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        s_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        for i, (m, n) in enumerate(zip(seq_len_q, seq_len_kv)):
            q_mask[i, :, m:, :] = True
            k_mask[i, :, n:, :] = True
            v_mask[i, :, n:, :] = True
            s_mask[i, :, m:, :] = True
            s_mask[i, :, :, n:] = True

    q = q.to(dtype=torch.float32, device=device)
    k = k.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)
    if padding is not None:
        q.masked_fill_(q_mask, 0)
        k.masked_fill_(k_mask, 0)
        v.masked_fill_(v_mask, 0)
    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * attn_scale
    if bias is not None:
        s.add_(bias)
    if is_alibi:
        lin_bias = ((torch.arange(s_kv, dtype=q.dtype)) - torch.arange(s_q, dtype=q.dtype).view(-1, 1))
        s.add_(lin_bias.to(device=device) * get_slopes(h))
    if padding is not None:
        s.masked_fill_(s_mask, float("-inf"))
    if is_causal:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device).triu_(diagonal=1)
        s.masked_fill_(causal_mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    if padding is not None:
        p.masked_fill_(s_mask, 0)
    o = torch.einsum("bhqk,bhkd->bhqd", p, v)
    # amax (NOT absolute max) is used here to evenly distribute gradient
    row_max = torch.amax(s, -1, True)
    row_exp = torch.exp(s - row_max)
    row_sum = torch.sum(row_exp, -1, True)
    stats = row_max + torch.log(row_sum)

    return o, stats


class ScaledDotProductAttentionPyT(torch.nn.Module):
    def __init__(self, is_causal=False, is_bias=False, is_alibi=False, attn_scale=1.0):
        super(ScaledDotProductAttentionPyT, self).__init__()
        self.is_bias = is_bias
        self.is_causal = is_causal
        self.is_alibi = is_alibi
        self.attn_scale = attn_scale

    def forward(self, q, k, v, bias=None):
        b, h, s_q, d = q.shape
        _, _, s_kv, _ = k.shape

        assert k.shape == (b, h, s_kv, d)
        assert v.shape == (b, h, s_kv, d)

        assert self.is_bias == (bias != None)

        s = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.attn_scale
        if self.is_bias:
            s.add_(bias)
        if self.is_alibi:
            s.add_(((torch.arange(s_kv, dtype=q.dtype)) - torch.arange(s_q, dtype=q.dtype).view(-1, 1)) * get_slopes(h))
        if self.is_causal:
            causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool).triu_(diagonal=1).cuda()
            s.masked_fill_(causal_mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("bhqk,bhkd->bhqd", p, v)
        return o

alibi_mask_options = [False, True]
padding_mask_options = [False, True]
causal_mask_options = [False, True]
layout_options = ["non_interleaved", "bs3hd", "sbh3d"]
dropout_options = [False]
is_infer_options = [False, True]
bias_options = [False, True]
input_type_options = [torch.float16, torch.bfloat16]

all_options_forward = [
    elem
    for elem in itertools.product(
        *[
            alibi_mask_options,
            padding_mask_options,
            causal_mask_options,
            layout_options,
            dropout_options,
            is_infer_options,
            bias_options,
            input_type_options
        ]
    )
]

all_options_backward = [
    elem
    for elem in itertools.product(
        *[
            causal_mask_options,
            dropout_options,
            bias_options,
            input_type_options
        ]
    )
]

@pytest.fixture(params=all_options_forward)
def param_extract_forward(request):
    return request.param


@pytest.mark.skipif(cudnn.backend_version() < 8903, reason="requires cudnn 8.9.3 or higher")
def test_scale_dot_product_flash_attention(param_extract_forward, print_compare=False):
    (
        is_alibi,
        is_padding,
        is_causal,
        layout,
        is_dropout,
        is_infer,
        is_bias,
        input_type
    ) = param_extract_forward

    if is_alibi and cudnn.backend_version() < 8904:
        pytest.skip("ALiBi mask is only supported 8.9.4 onwards.")

    if is_padding and cudnn.backend_version() < 8903:
        pytest.skip("Padding mask is only supported 8.9.3 onwards.")

    s_q_choices = [256, 512, 1024, 2048]
    d_choices = [64, 128]

    b = 3
    h = 4
    s_q = random.choice(s_q_choices)
    s_kv = s_q
    d = random.choice(d_choices)

    print(f"{str(param_extract_forward)} s={s_q} d={d}")

    attn_scale_val = 0.125
    dropout_prob = 0.1 if is_dropout else 0.0

    shape_q = (b, h, s_q, d)
    shape_k = (b, h, d, s_kv)
    shape_v = (b, h, s_kv, d)
    shape_o = (b, h, s_q, d)

    if layout == "sbh3d":
        stride_q = (3 * h * d, 3 * d, b * 3 * h * d, 1)
        stride_k = (3 * h * d, 3 * d, 1, b * 3 * h * d)
        stride_v = (3 * h * d, 3 * d, b * 3 * h * d, 1)
        stride_o = (h * d, d, b * h * d, 1)
        stride_order_o = (2, 1, 3, 0)

        offset_q = d * 0
        offset_k = d * 1
        offset_v = d * 2
    elif layout == "bs3hd":
        stride_q = (s_q * 3 * h * d, d, 3 * h * d, 1)
        stride_k = (s_q * 3 * h * d, d, 1, 3 * h * d)
        stride_v = (s_q * 3 * h * d, d, 3 * h * d, 1)
        stride_o = (s_q * h * d, d, h * d, 1)
        stride_order_o = (3, 1, 2, 0)

        offset_q = h * d * 0
        offset_k = h * d * 1
        offset_v = h * d * 2
    elif layout == "non_interleaved":
        stride_q = (d * s_q * h, d * s_q, d, 1)
        stride_k = (d * s_kv * h, d * s_kv, 1, d)
        stride_v = (d * s_kv * h, d * s_kv, d, 1)
        stride_o = (d * s_q * h, d * s_q, d, 1)
        stride_order_o = (3, 2, 1, 0)

        offset_q = 0
        offset_k = offset_q + b * d * s_q * h
        offset_v = offset_k + b * d * s_kv * h
    else:
        assert False, "Layout should be either sbh3d or bs3hd or non_interleaved"

    qkv_gpu = 1 * (torch.randn(b * s_q * 3 * h * d, dtype=input_type, device="cuda") - 0.5)
    q_gpu = torch.as_strided(qkv_gpu, shape_q, stride_q, storage_offset=offset_q)
    k_gpu = torch.as_strided(qkv_gpu, shape_k, stride_k, storage_offset=offset_k)
    v_gpu = torch.as_strided(qkv_gpu, shape_v, stride_v, storage_offset=offset_v)

    if attn_scale_val != 1.0:
        attn_scale_cpu = torch.full((1, 1, 1, 1), attn_scale_val, dtype=torch.float32, device="cpu")

    if is_bias:
        bias_gpu = torch.randn(b, 1, s_q, s_kv, requires_grad=False, device="cuda", dtype=input_type)

    if is_padding:
        seq_len_q_gpu = torch.randint(0, s_q + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda")
        seq_len_kv_gpu = torch.randint(0, s_kv + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda")

    if is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    o_gpu = torch.empty(*shape_o, dtype=input_type, device="cuda").as_strided(shape_o, stride_o)
    if is_infer == False:
        stats_gpu = torch.empty(b, h, s_q, 1, dtype=torch.float32, device="cuda")

    # cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = make_tensor_attr(graph, q_gpu, "q")
    k = make_tensor_attr(graph, k_gpu, "k")
    v = make_tensor_attr(graph, v_gpu, "v")

    if attn_scale_val != 1.0:
        attn_scale = make_tensor_attr(graph, attn_scale_cpu, "attn_scale", is_pass_by_value=True)

    if is_bias:
        bias = make_tensor_attr(graph, bias_gpu, "bias")

    if is_padding:
        seq_len_q = make_tensor_attr(graph, seq_len_q_gpu, "seq_len_q")
        seq_len_kv = make_tensor_attr(graph, seq_len_kv_gpu, "seq_len_kv")

    if is_dropout:
        seed = make_tensor_attr(graph, seed_gpu, "seed")
        offset = make_tensor_attr(graph, offset_gpu, "attn_scale")
        dropout_tuple = (dropout_prob, seed, offset)

    o, stats = graph.scaled_dot_product_flash_attention(
        name="scaled_dot_product_flash_attention",
        q=q,
        k=k,
        v=v,
        is_inference=is_infer,
        attn_scale=attn_scale if attn_scale_val != 1.0 else None,
        bias=bias if is_bias else None,
        use_alibi_mask=is_alibi,
        use_padding_mask=is_padding,
        seq_len_q=seq_len_q if is_padding else None,
        seq_len_kv=seq_len_kv if is_padding else None,
        use_causal_mask=is_causal,
        dropout=dropout_tuple if is_dropout else None,
    )

    o.set_output(True).set_dim(shape_o).set_stride(stride_o)
    if is_infer == False:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.check_support()
    graph.build()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu
    }

    if attn_scale_val != 1.0:
        variant_pack[attn_scale] = attn_scale_cpu

    if is_bias:
        variant_pack[bias] = bias_gpu

    if is_padding:
        variant_pack[seq_len_q] = seq_len_q_gpu
        variant_pack[seq_len_kv] = seq_len_kv_gpu

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    if is_infer == False:
        variant_pack[stats] = stats_gpu

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)

    # compare with torch reference
    q_ref = q_gpu.detach().float()
    k_ref = k_gpu.permute(0, 1, 3, 2).detach().float()
    v_ref = v_gpu.detach().float()

    if is_bias:
        bias_ref = bias_gpu.detach().float()

    if is_padding:
        seq_len_q_ref = seq_len_q_gpu.detach().flatten()
        seq_len_kv_ref = seq_len_kv_gpu.detach().flatten()

    o_ref, stats_ref = compute_o_stats(
        q_ref,
        k_ref,
        v_ref,
        attn_scale=attn_scale_val,
        bias=bias_ref if is_bias else None,
        is_alibi=is_alibi,
        is_causal=is_causal,
        padding=(seq_len_q_ref, seq_len_kv_ref) if is_padding else None
    )

    if is_padding:
        # zero out padded region of the output for comparison
        for i, (m, n) in enumerate(zip(seq_len_q_ref, seq_len_kv_ref)):
            o_ref[i, :, m:, :] = 0
            o_gpu[i, :, m:, :] = 0
            if is_infer == False:
                stats_ref[i, :, m:, :] = 0
                stats_gpu[i, :, m:, :] = 0

    assert compare_tensors(o_ref, o_gpu, "O", print_compare=print_compare) == 0
    if is_infer == False:
        assert compare_tensors(stats_ref, stats_gpu, "stats", print_compare=print_compare) == 0


@pytest.fixture(params=all_options_backward)
def param_extract_backward(request):
    return request.param


@pytest.mark.skipif(cudnn.backend_version() < 8903, reason="requires cudnn 8.9.3 or higher")
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="requires ampere or higher")
def test_scale_dot_product_flash_attention_backward(param_extract_backward, print_compare=False):
    (
        is_causal,
        is_dropout,
        is_bias,
        input_type
    ) = param_extract_backward

    layout = "naive"

    s_q_choices = [256, 512, 1024]
    d_choices = [64, 128]

    b = 3
    h = 4
    s_q = random.choice(s_q_choices)
    s_kv = s_q
    d = random.choice(d_choices)

    print(f"{str(param_extract_backward)} s={s_q} d={d}")

    attn_scale_val = 0.125
    dropout_prob = 0.1 if is_dropout else 0.0

    q_gpu = 1 * (torch.randn((b, h, s_q, d), dtype=input_type, device="cuda") - 0.5)
    k_gpu = 1 * (torch.randn((b, h, s_kv, d), dtype=input_type, device="cuda") - 0.5)
    v_gpu = 1 * (torch.randn((b, h, s_kv, d), dtype=input_type, device="cuda") - 0.5)
    dO_gpu = 0.1 * torch.randn((b, h, s_q, d), dtype=input_type, device="cuda")

    if attn_scale_val != 1.0:
        attn_scale_cpu = torch.full((1, 1, 1, 1), attn_scale_val, dtype=torch.float32, device="cpu")

    if is_bias:
        bias_gpu = torch.randn(b, 1, s_q, s_kv, device="cuda", dtype=input_type)

    if is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    o_gpu, stats_gpu = compute_o_stats(
        q_gpu,
        k_gpu,
        v_gpu,
        is_causal=is_causal,
        bias=bias_gpu if is_bias else None,
        attn_scale=attn_scale_val
    )
    o_gpu = o_gpu.to(dtype=input_type).detach().clone()
    stats_gpu = stats_gpu.to(dtype=torch.float32).detach().clone()

    dQ_gpu = torch.empty((b, h, s_q, d), dtype=input_type, device="cuda")
    dK_gpu = torch.empty((b, h, s_kv, d), dtype=input_type, device="cuda")
    dV_gpu = torch.empty((b, h, s_kv, d), dtype=input_type, device="cuda")

    # cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = make_tensor_attr(graph, q_gpu, name="q")
    k = make_tensor_attr(graph, k_gpu, dim=(b, h, d, s_kv), stride=(h * s_kv * d, s_kv * d, 1, d), name="k")
    v = make_tensor_attr(graph, v_gpu, dim=(b, h, d, s_kv), stride=(h * s_kv * d, s_kv * d, 1, d), name="v")
    o = make_tensor_attr(graph, o_gpu, name="o")
    dO = make_tensor_attr(graph, dO_gpu, name="dO")
    stats = make_tensor_attr(graph, stats_gpu, name="stats")

    if attn_scale_val != 1.0:
        attn_scale = make_tensor_attr(graph, attn_scale_cpu, is_pass_by_value=True, name="attn_scale")

    if is_bias:
        bias = make_tensor_attr(graph, bias_gpu, "bias")

    if is_dropout:
        seed = make_tensor_attr(graph, seed_gpu, "seed")
        offset = make_tensor_attr(graph, offset_gpu, "attn_scale")
        dropout_tuple = (dropout_prob, seed, offset)

    dQ, dK, dV = graph.scaled_dot_product_flash_attention_backward(
        name="scaled_dot_product_flash_attention",
        q=q,
        k=k,
        v=v,
        o=o,
        dO=dO,
        stats=stats,
        attn_scale=attn_scale if attn_scale_val != 1.0 else None,
        bias=bias if is_bias else None,
        use_causal_mask=is_causal,
        dropout=dropout_tuple if is_dropout else None,
    )

    dQ.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
    dK.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
    dV.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())

    graph.check_support()
    graph.build()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        dO: dO_gpu,
        stats: stats_gpu,
        dQ: dQ_gpu,
        dK: dK_gpu,
        dV: dV_gpu
    }

    if attn_scale_val != 1.0:
        variant_pack[attn_scale] = attn_scale_cpu

    if is_bias:
        variant_pack[bias] = bias_gpu

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)

    # compare with torch autograd reference
    nn_ref = ScaledDotProductAttentionPyT(
        is_causal=is_causal,
        is_bias=is_bias,
        attn_scale=attn_scale_val
    ).cuda().float()

    q_ref = q_gpu.detach().float()
    q_ref.requires_grad = True
    k_ref = k_gpu.detach().float()
    k_ref.requires_grad = True
    v_ref = v_gpu.detach().float()
    v_ref.requires_grad = True
    dO_ref = dO_gpu.detach().float()

    if is_bias:
        bias_ref = bias_gpu.detach().float()
        bias_ref.requires_grad = True

    o_ref = nn_ref(q_ref, k_ref, v_ref, bias=bias_ref if is_bias else None)

    outputs_ref = [o_ref]
    inputs_ref = [q_ref, k_ref, v_ref]

    if is_bias:
        inputs_ref.append(bias_ref)

    [dq_ref, dk_ref, dv_ref, *opt_refs] = list(torch.autograd.grad(
        outputs=outputs_ref,
        inputs=inputs_ref,
        grad_outputs=dO_ref
    ))

    assert compare_tensors(dq_ref, dQ_gpu, "dQ", print_compare=print_compare) == 0
    assert compare_tensors(dk_ref, dK_gpu, "dK", print_compare=print_compare) == 0
    assert compare_tensors(dv_ref, dV_gpu, "dV", print_compare=print_compare) == 0

    if is_bias:
        db_ref = opt_refs.pop(0)

if __name__ == "__main__":
    """
    option_forward = (alibi_mask, padding_mask, causal_mask, layout, dropout_enable, is_infer, bias_enable, input_type)
    option_backward = (is_causal, is_dropout, is_bias, input_type)
    test_scale_dot_product_flash_attention((False, False, False, "bs3hd", False, False, False, torch.float16), print_compare=True)
    test_scale_dot_product_flash_attention_backward((False, False, False, torch.float16), print_compare=True)
    """

    print("==========running forward tests==========")
    for option in all_options_forward:
        test_scale_dot_product_flash_attention(option)

    print("==========running backward tests==========")
    for option in all_options_backward:
        test_scale_dot_product_flash_attention_backward(option)
