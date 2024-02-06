import cudnn
import pytest
import torch
import math

import itertools
import random
import os


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


def compare_tensors(expected, actual, name, rtol=2e-2, atol=2e-2, fudge=1e-9):
    assert expected.shape == actual.shape

    expected = expected.float().cuda().flatten()
    actual = actual.float().cuda().flatten()

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

    if n_errors + n_nans != 0:
        print(f"========== Comparison for {name} ==========")
        print(f"Absolute Tolerance = {atol}")
        print(f"Relative Tolerance = {rtol}")
        print(f"Number of elements = {n_elem}")
        print(f"Number of absolute errors = {n_abs_errors} ({n_abs_errors * 100 / n_elem:.2f}%)")
        print(f"Number of relative errors = {n_rel_errors} ({n_rel_errors * 100 / n_elem:.2f}%)")
        print(f"Number of errors (absolute and relative) = {n_errors} ({(n_errors * 100)/n_elem:.2f}%)")
        print(f"Maximum absolute error = {absolute_error.max():.4f}")
        print(f"Maximum relative error = {relative_error.max():.4f}")
        print(f"Mean average error = {mae:.4f}")
        print(f"Perr error = {perr:.4f} = 1/{(1/perr) if perr != 0 else float('inf'):.2f}")
        print(f"Signal to noise ratio = {snr.item():.2f} = {snr_db:.2f}dB")
        print(f"Number of Nans = {n_nans} ({n_nans * 100 / n_elem:.2f}%)")
        print(f"Number of Zeros = {n_zeros} ({n_zeros * 100 / n_elem:.2f}%)")
        print("===================================\n")

    return n_errors + n_nans


def compute_ref(
    q,
    k,
    v,
    attn_scale=1.0,
    bias=None,
    is_alibi=False,
    padding=None,
    is_causal=False,
    dropout_prob=0.0,
    dropout_mask=None,
    compute_stats=False,
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

    # generate masks to compute reference values for padding mask
    # (also called variable sequence length)
    if padding is not None:
        q_mask = torch.ones(b, 1, s_q, 1, dtype=torch.bool, device=device)
        k_mask = torch.ones(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        v_mask = torch.ones(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        s_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        p_mask = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        seq_len_q, seq_len_kv = padding
        for i, (m, n) in enumerate(zip(seq_len_q, seq_len_kv)):
            q_mask[i, :, m:, :] = False
            k_mask[i, :, n:, :] = False
            v_mask[i, :, n:, :] = False
            s_mask[i, :, :, n:] = True
            p_mask[i, :, m:, :] = False

    if padding is not None:
        q = q * q_mask
        k = k * k_mask
        v = v * v_mask

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
    if is_causal:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device).triu_(diagonal=1)
        s = s.masked_fill(causal_mask, float("-inf"))

    p = torch.softmax(s, dim=-1)
    if padding is not None:
        p = p * p_mask

    # apply dropout mask over softmax outputs
    if dropout_prob != 0.0:
        assert dropout_mask != None, "PyTorch reference must have dropout_mask for dropout"
        p = (p * dropout_mask) / (1 - dropout_prob)

    o = torch.einsum("bhqk,bhkd->bhqd", p, v)

    # softmax stats is used for backwards computation
    if compute_stats:
        # amax (NOT absolute max) is used here to evenly distribute gradient
        row_max = torch.amax(s, -1, True)
        row_exp = torch.exp(s - row_max)
        row_sum = torch.sum(row_exp, -1, True)
        stats = row_max + torch.log(row_sum)
        return o, stats

    return o


input_type_options = [torch.float16, torch.bfloat16]
layout_options = ["non_interleaved", "bs3hd", "sbh3d"]
head_group_options = ["multi_head", "group_query", "multi_query"]
bias_options = [False, True]
alibi_mask_options = [False, True]
padding_mask_options = [False, True]
causal_mask_options = [False, True]
dropout_options = [False, True]
ragged_options = [False, True]
is_infer_options = [False, True]

all_options_forward = [
    elem
    for elem in itertools.product(
        *[
            input_type_options,
            layout_options,
            head_group_options,
            bias_options,
            alibi_mask_options,
            padding_mask_options,
            causal_mask_options,
            dropout_options,
            ragged_options,
            is_infer_options,
        ]
    )
]

all_options_backward = [
    elem
    for elem in itertools.product(
        *[
            input_type_options,
            layout_options,
            head_group_options,
            bias_options,
            alibi_mask_options,
            padding_mask_options,
            causal_mask_options,
            dropout_options,
            ragged_options,
        ]
    )
]


def generate_layout(layout, head_group, shape_q, shape_k, shape_v, shape_o):
    b, h_q, s_q, d_qk = shape_q
    b, h_k, s_kv, d_qk = shape_k
    b, h_v, s_kv, d_v = shape_v
    b, h_q, s_q, d_v = shape_o

    assert shape_q == (b, h_q, s_q, d_qk)
    assert shape_k == (b, h_k, s_kv, d_qk)
    assert shape_v == (b, h_v, s_kv, d_v)
    assert shape_o == (b, h_q, s_q, d_v)

    if layout == "sbh3d":
        if head_group == "multi_head":
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
            # group_query and multi_query
            # sbhd + sbh2d
            assert (h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
            h_kv, s, d = h_k, s_q, d_qk
            stride_q = (h_q * d, d, b * h_q * d, 1)
            stride_k = (h_kv * 2 * d, 2 * d, b * h_kv * 2 * d, 1)
            stride_v = (h_kv * 2 * d, 2 * d, b * h_kv * 2 * d, 1)
            stride_o = (h_q * d, d, b * h_q * d, 1)
            offset_q = 0
            offset_k = offset_q + s * b * h_q * d
            offset_v = offset_k + d
    elif layout == "bs3hd":
        if head_group == "multi_head":
            assert (h_q == h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
            h, s, d = h_q, s_q, d_qk
            stride_q = (s * 3 * h * d, d, 3 * h * d, 1)
            stride_k = (s * 3 * h * d, d, 3 * h * d, 1)
            stride_v = (s * 3 * h * d, d, 3 * h * d, 1)
            stride_o = (s * h * d, d, h * d, 1)
            offset_q = 0
            offset_k = offset_q + h_q * d
            offset_v = offset_k + h_k * d
        else:
            # group_query and multi_query
            # bshd + bs2hd
            assert (h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
            h_kv, s, d = h_k, s_q, d_qk
            stride_q = (s * h_q * d, d, h_q * d, 1)
            stride_k = (s * 2 * h_kv * d, d, 2 * h_kv * d, 1)
            stride_v = (s * 2 * h_kv * d, d, 2 * h_kv * d, 1)
            stride_o = (s * h_q * d, d, h_q * d, 1)
            offset_q = 0
            offset_k = offset_q + s * b * h_q * d
            offset_v = offset_k + h_kv * d
    else:
        # bshd non_interleaved layout
        stride_q = (s_q * h_q * d_qk, d_qk, h_q * d_qk, 1)
        stride_k = (s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)
        stride_v = (s_kv * h_v * d_v, d_v, h_v * d_v, 1)
        stride_o = (s_q * h_q * d_v, d_v, h_q * d_v, 1)
        offset_q = 0
        offset_k = offset_q + b * s_q * h_q * d_qk
        offset_v = offset_k + b * s_kv * h_k * d_qk

    return stride_q, stride_k, stride_v, stride_o, offset_q, offset_k, offset_v


def compute_exclusive_prefix_sum(tensor):
    # tensor has shape (B, 1, 1, 1)
    # output has shape (B+1, 1, 1, 1)
    # ex) tensor = [[[[2, 4, 1, 6]]]]
    #     output = [[[[0, 2, 6, 7, 13]]]]
    assert tensor.size(1) == tensor.size(2) == tensor.size(3) == 1
    return torch.cat((torch.zeros(1, 1, 1, 1, dtype=tensor.dtype, device=tensor.device), torch.cumsum(tensor, dim=0)))


def convert_ragged_to_uniform(ragged_tensor, ragged_offset):
    # limitations:
    # 1. tensor is non-interleaved with bhsd dim order and bshd stride order
    # 2. ragged tensor is packed and in-order, therefore
    #    ragged offset is monatomically increasing
    assert ragged_tensor.dim() == 4
    b, h, s, d = ragged_tensor.size()
    b_stride, h_stride, s_stride, d_stride = ragged_tensor.stride()
    assert b_stride >= s_stride >= h_stride >= d_stride
    assert ragged_offset.dim() == 4 and (b + 1, 1, 1, 1) == ragged_offset.size()

    # ragged offset is given in 4D, convert to 1D locally
    ragged_offset = ragged_offset.flatten()

    # convert bhsd to bshd and flatten
    ragged_tensor_flat = torch.einsum("bhsd->bshd", ragged_tensor).flatten()
    uniform_tensor_flat = torch.zeros_like(ragged_tensor_flat)

    # copy
    for i, num_elements in enumerate(ragged_offset[1:] - ragged_offset[:-1]):
        unif_a = i * s * h * d
        unif_b = unif_a + num_elements
        ragg_a = ragged_offset[i]
        ragg_b = ragg_a + num_elements
        uniform_tensor_flat[unif_a:unif_b] = ragged_tensor_flat[ragg_a:ragg_b]

    # unflatten and convert bshd to bhsd
    uniform_tensor = uniform_tensor_flat.view(b, s, h, d)
    uniform_tensor = torch.einsum("bshd->bhsd", uniform_tensor)
    return uniform_tensor


@pytest.fixture(params=all_options_forward)
def param_extract_forward(request):
    return request.param


@pytest.mark.parametrize("input_type", input_type_options)
@pytest.mark.parametrize("layout", layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_bias", bias_options)
@pytest.mark.parametrize("is_alibi", alibi_mask_options)
@pytest.mark.parametrize("is_padding", padding_mask_options)
@pytest.mark.parametrize("is_causal", causal_mask_options)
@pytest.mark.parametrize("is_dropout", dropout_options)
@pytest.mark.parametrize("is_ragged", ragged_options)
@pytest.mark.parametrize("is_infer", is_infer_options)
def test_sdpa(input_type,
        layout,
        head_group,
        is_bias,
        is_alibi,
        is_padding,
        is_causal,
        is_dropout,
        is_ragged,
        is_infer):
    if cudnn.backend_version() < 8903:
        pytest.skip("SDPA fprop requires cudnn 8.9.3 or higher")

    if head_group != "multi_head" and cudnn.backend_version() < 8907:
        pytest.skip("GQA and MQA is only supported 8.9.7 onwards.")

    if is_alibi and cudnn.backend_version() < 8904:
        pytest.skip("ALiBi mask is only supported 8.9.4 onwards.")

    if is_padding and cudnn.backend_version() < 8903:
        pytest.skip("Padding mask is only supported 8.9.3 onwards.")

    if is_dropout and cudnn.backend_version() < 8906:
        pytest.skip("Dropout reference is only supported on 8.9.6 onwards.")

    if is_ragged and cudnn.backend_version() < 90000:
        pytest.skip("Ragged tensor is only supported 9.0.0 onwards")

    if is_ragged and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Ragged tensor is only supported hopper")

    if is_ragged and layout != "non_interleaved":
        pytest.skip("Ragged tensor is only tested with non-interleaved bshd layout")

    if is_ragged and not is_padding:
        pytest.skip("Ragged tensor is only tested with packed variable length tensors")

    # batch size
    b = 2
    # query sequence length
    s_q = random.choice([8, 16, 24, 32, 256, 512, 1024, 2048])
    # key+value sequence length
    s_kv = random.choice([8, 16, 24, 32, 256, 512, 1024, 2048]) if layout == "non_interleaved" else s_q
    # query+key embedding dimension per head
    d_qk = random.choice([32, 56, 64, 128])
    # value embedding dimension per head
    d_v = random.choice([64, 96, 128]) if (layout == "non_interleaved" and not is_ragged) else d_qk
    # number of heads
    h_q = 6
    if head_group == "multi_head":
        h_k = 6
        h_v = 6
    elif head_group == "group_query":
        h_k = random.choice([6, 3, 2, 1])
        h_v = random.choice([6, 3, 2, 1]) if layout == "non_interleaved" else h_k
    elif head_group == "multi_query":
        h_k = 1
        h_v = 1
    else:
        assert False, "Head group must be either MHA, GQA, or MQA"

    if d_qk != d_v and cudnn.backend_version() < 8906:
        pytest.skip("d_qk != d_v is only supported on 8.9.6 onwards.")

    if cudnn.backend_version() < 90000:
        if ((s_q % 64 != 0) or (s_kv % 64 != 0)) and (is_padding or is_dropout):
            pytest.skip("s_q not a multiple of 64 with padding/dropout is not supported with cudnn version 9.0.0")

    if cudnn.backend_version() < 8906:
        pytest.skip("d not a multiple of 64, not-multiple-of-64 seq_kv is not supported below 8.9.6")

    if (d_qk % 64 != 0) and cudnn.backend_version() < 8906:
        pytest.skip("d not a multiple of 64 is not supported below 8.9.6")

    if (d_qk % 64 != 0) and cudnn.backend_version() < 8906:
        pytest.skip("d not a multiple of 64 is not supported below 8.9.6")

    # TODO file bug
    if d_qk != d_v and is_ragged:
        pytest.skip("d_qk != d_v is not supported with ragged offset")

    print(f"{s_q=} {s_kv=} {d_qk=} {d_v=} {h_q=} {h_k=} {h_v=}")

    attn_scale = 0.125
    dropout_prob = 0.1 if is_dropout else 0.0

    shape_q = (b, h_q, s_q, d_qk)
    shape_k = (b, h_k, s_kv, d_qk)
    shape_v = (b, h_v, s_kv, d_v)
    shape_o = (b, h_q, s_q, d_v)

    qkv_num_elems = math.prod(shape_q) + math.prod(shape_k) + math.prod(shape_v)

    (stride_q, stride_k, stride_v, stride_o, offset_q, offset_k, offset_v) = generate_layout(
        layout,
        head_group,
        shape_q,
        shape_k,
        shape_v,
        shape_o,
    )

    qkv_gpu = torch.randn(qkv_num_elems, dtype=input_type, device="cuda") - 0.5
    q_gpu = torch.as_strided(qkv_gpu, shape_q, stride_q, storage_offset=offset_q)
    k_gpu = torch.as_strided(qkv_gpu, shape_k, stride_k, storage_offset=offset_k)
    v_gpu = torch.as_strided(qkv_gpu, shape_v, stride_v, storage_offset=offset_v)

    bias_gpu = torch.randn(1, h_q, s_q, s_kv, device="cuda", dtype=input_type) if is_bias else None

    seq_len_q_gpu = torch.randint(1, s_q + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda") if is_padding else None
    seq_len_kv_gpu = torch.randint(1, s_kv + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda") if is_padding else None

    if is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    rng_dump_gpu = torch.empty((b, h_q, s_q, s_kv), dtype=torch.float32, device="cuda") if is_dropout else None

    q_ragged_offset_gpu = (compute_exclusive_prefix_sum(seq_len_q_gpu) * h_q * d_qk).int() if is_ragged else None
    k_ragged_offset_gpu = (compute_exclusive_prefix_sum(seq_len_kv_gpu) * h_k * d_qk).int() if is_ragged else None
    v_ragged_offset_gpu = (compute_exclusive_prefix_sum(seq_len_kv_gpu) * h_v * d_v).int() if is_ragged else None
    o_ragged_offset_gpu = (compute_exclusive_prefix_sum(seq_len_q_gpu) * h_q * d_v).int() if is_ragged else None

    o_gpu = torch.empty(b * h_q * s_q * d_v, dtype=input_type, device="cuda").as_strided(shape_o, stride_o)
    stats_gpu = torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device="cuda") if not is_infer else None

    # cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
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
        is_inference=is_infer,
        attn_scale=attn_scale,
        bias=bias,
        use_alibi_mask=is_alibi,
        use_padding_mask=is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        use_causal_mask=is_causal,
        dropout=dropout_tuple if is_dropout else None,
        rng_dump=rng_dump,
    )

    o.set_output(True).set_dim(shape_o).set_stride(stride_o)
    if is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    if is_infer == False:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
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
        q_ragged_offset: q_ragged_offset_gpu,
        k_ragged_offset: k_ragged_offset_gpu,
        v_ragged_offset: v_ragged_offset_gpu,
        o_ragged_offset: o_ragged_offset_gpu,
        o: o_gpu,
        stats: stats_gpu,
        rng_dump: rng_dump_gpu,
    }

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    # compare with torch autograd reference
    q_ref = q_gpu.detach().float()
    k_ref = k_gpu.detach().float()
    v_ref = v_gpu.detach().float()

    if is_ragged:
        q_ref = convert_ragged_to_uniform(q_ref, q_ragged_offset_gpu.detach())
        k_ref = convert_ragged_to_uniform(k_ref, k_ragged_offset_gpu.detach())
        v_ref = convert_ragged_to_uniform(v_ref, v_ragged_offset_gpu.detach())

    if is_bias:
        bias_ref = bias_gpu.detach().float()

    if is_padding:
        seq_len_q_ref = seq_len_q_gpu.detach().flatten()
        seq_len_kv_ref = seq_len_kv_gpu.detach().flatten()

    if is_dropout:
        rng_dump_ref = rng_dump_gpu.detach().float()

    ret = compute_ref(
        q_ref,
        k_ref,
        v_ref,
        attn_scale=attn_scale,
        bias=bias_ref if is_bias else None,
        is_alibi=is_alibi,
        padding=(seq_len_q_ref, seq_len_kv_ref) if is_padding else None,
        is_causal=is_causal,
        compute_stats=(is_infer == False),
        dropout_prob=dropout_prob,
        dropout_mask=rng_dump_ref if is_dropout else None,
    )
    if is_infer == False:
        o_ref, stats_ref = ret
    else:
        o_ref = ret

    if is_ragged:
        o_gpu = convert_ragged_to_uniform(o_gpu, o_ragged_offset_gpu.detach())

    if is_padding:
        # zero out padded region of the output for comparison
        for i, m in enumerate(seq_len_q_ref):
            o_ref[i, :, m:, :] = 0
            o_gpu[i, :, m:, :] = 0
            if is_infer == False:
                stats_ref[i, :, m:, :] = 0
                stats_gpu[i, :, m:, :] = 0

    assert compare_tensors(o_ref, o_gpu, "O") == 0
    if is_infer == False:
        assert compare_tensors(stats_ref, stats_gpu, "stats") == 0


@pytest.mark.parametrize("input_type", input_type_options)
@pytest.mark.parametrize("layout", layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_bias", bias_options)
@pytest.mark.parametrize("is_alibi", alibi_mask_options)
@pytest.mark.parametrize("is_padding", padding_mask_options)
@pytest.mark.parametrize("is_causal", causal_mask_options)
@pytest.mark.parametrize("is_dropout", dropout_options)
@pytest.mark.parametrize("is_ragged", ragged_options)
def test_sdpa_backward(input_type,
        layout,
        head_group,
        is_bias,
        is_alibi,
        is_padding,
        is_causal,
        is_dropout,
        is_ragged):
    if cudnn.backend_version() < 8903:
        pytest.skip("SDPA bprop requires cudnn 8.9.3 or higher")

    if head_group != "multi_head" and cudnn.backend_version() < 8907:
        pytest.skip("GQA and MQA is only supported 8.9.7 onwards.")

    if is_bias and cudnn.backend_version() < 8906:
        pytest.skip("dBias is only supported 8.9.6 onwards.")

    if is_bias and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("dBias is only supported on hopper onwards.")

    if is_bias and is_padding:
        pytest.skip("dBias is not supported with padding mask")

    if is_alibi and not is_causal:
        pytest.skip("ALiBi mask is only supported with causal mask")

    if is_alibi and cudnn.backend_version() < 8904:
        pytest.skip("ALiBi mask is only supported 8.9.4 onwards.")

    if is_padding and cudnn.backend_version() < 8903:
        pytest.skip("Padding mask is only supported 8.9.3 onwards.")

    if is_dropout and cudnn.backend_version() < 8906:
        pytest.skip("RNG dump is only supported on 8.9.6 onwards.")

    if is_ragged and cudnn.backend_version() < 90000:
        pytest.skip("Ragged tensor is only supported 9.0.0 onwards")

    if is_ragged and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Ragged tensor is only supported hopper")

    if is_ragged and layout != "non_interleaved":
        pytest.skip("Ragged tensor is only tested with non-interleaved bshd layout")

    if is_ragged and head_group != "multi_head":
        pytest.skip("Ragged offset is only supported with multi_head")

    if is_ragged and not is_padding:
        pytest.skip("Ragged tensor is only tested with packed variable length tensors")

    # test both dP workspace optimization by lowering dP workspace limit to 8MB
    os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = str(8 * 1024 * 1024)

    # batch size
    b = 2
    # query sequence length
    s_q = random.choice([8, 16, 24, 32, 256, 512, 1024])
    # key+value sequence length
    s_kv = random.choice([8, 16, 24, 32, 256, 512, 1024]) if layout == "non_interleaved" else s_q
    # query+key embedding dimension per head
    d_qk = random.choice([32, 56, 64, 128])
    # value embedding dimension per head
    d_v = random.choice([64, 96, 128]) if (layout == "non_interleaved" and not is_ragged) else d_qk
    # number of heads
    h_q = 6
    if head_group == "multi_head":
        h_k = 6
        h_v = 6
    elif head_group == "group_query":
        h_k = random.choice([6, 3, 2, 1])
        h_v = random.choice([6, 3, 2, 1]) if layout == "non_interleaved" else h_k
    elif head_group == "multi_query":
        h_k = 1
        h_v = 1
    else:
        assert False, "Head group must be either MHA, GQA, or MQA"

    if d_qk != d_v and cudnn.backend_version() < 8906:
        pytest.skip("d_qk != d_v is only supported on 8.9.6 onwards.")

    if (cudnn.backend_version() < 90000):
        if (s_q < 64):
            pytest.skip("s_q less than 64 is not supported before cudnn 9.0.0")

        if ((s_q % 64 != 0) or (s_kv % 64 != 0)) and (is_padding or is_dropout):
            pytest.skip("s_q not a multiple of 64 with padding/dropout is not supported with cudnn version 9.0.0")

    if ((s_q % 64 != 0) or (s_kv % 64 != 0)) and is_bias:
        pytest.skip("cudnn backend does not support bias with non-64-aligned seq_q or seq_kv.")

    if (s_kv % 64 != 0) and cudnn.backend_version() < 8906:
        pytest.skip("not-multiple-of-64 seq_kv is not supported below 8.9.6")

    if (d_qk % 64 != 0) and cudnn.backend_version() < 8906:
        pytest.skip("d not a multiple of 64 is not supported below 8.9.6")

    # TODO file bug
    if d_qk != d_v and is_ragged:
        pytest.skip("d_qk != d_v is not supported with ragged offset")

    print(f"{s_q=} {s_kv=} {d_qk=} {d_v=} {h_q=} {h_k=} {h_v=}")

    attn_scale = 0.125
    dropout_prob = 0.1 if is_dropout else 0.0

    shape_q = (b, h_q, s_q, d_qk)
    shape_k = (b, h_k, s_kv, d_qk)
    shape_v = (b, h_v, s_kv, d_v)
    shape_o = (b, h_q, s_q, d_v)

    qkv_num_elems = math.prod(shape_q) + math.prod(shape_k) + math.prod(shape_v)

    (stride_q, stride_k, stride_v, stride_o, offset_q, offset_k, offset_v) = generate_layout(
        layout,
        head_group,
        shape_q,
        shape_k,
        shape_v,
        shape_o,
    )

    qkv_gpu = torch.randn(qkv_num_elems, dtype=input_type, device="cuda") - 0.5
    q_gpu = torch.as_strided(qkv_gpu, shape_q, stride_q, storage_offset=offset_q)
    k_gpu = torch.as_strided(qkv_gpu, shape_k, stride_k, storage_offset=offset_k)
    v_gpu = torch.as_strided(qkv_gpu, shape_v, stride_v, storage_offset=offset_v)

    dQKV_gpu = torch.empty(qkv_num_elems, dtype=input_type, device="cuda")
    dQ_gpu = torch.as_strided(dQKV_gpu, shape_q, stride_q, storage_offset=offset_q)
    dK_gpu = torch.as_strided(dQKV_gpu, shape_k, stride_k, storage_offset=offset_k)
    dV_gpu = torch.as_strided(dQKV_gpu, shape_v, stride_v, storage_offset=offset_v)

    dO_gpu = 0.1 * torch.randn(b * h_q * s_q * d_v, dtype=input_type, device="cuda").as_strided(shape_o, stride_o)

    bias_gpu = torch.randn(1, h_q, s_q, s_kv, device="cuda", dtype=input_type) if is_bias else None
    dBias_gpu = torch.randn(1, h_q, s_q, s_kv, device="cuda", dtype=input_type) if is_bias else None

    seq_len_q_gpu = torch.randint(1, s_q + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda") if is_padding else None
    seq_len_kv_gpu = torch.randint(1, s_kv + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda") if is_padding else None

    if is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    rng_dump_gpu = torch.empty((b, h_q, s_q, s_kv), dtype=torch.float32, device="cuda") if is_dropout else None

    q_ragged_offset_gpu = (compute_exclusive_prefix_sum(seq_len_q_gpu) * h_q * d_qk).int() if is_ragged else None
    k_ragged_offset_gpu = (compute_exclusive_prefix_sum(seq_len_kv_gpu) * h_k * d_qk).int() if is_ragged else None
    v_ragged_offset_gpu = (compute_exclusive_prefix_sum(seq_len_kv_gpu) * h_v * d_v).int() if is_ragged else None
    o_ragged_offset_gpu = (compute_exclusive_prefix_sum(seq_len_q_gpu) * h_q * d_v).int() if is_ragged else None

    o_gpu = torch.empty(b * h_q * s_q * d_v, dtype=input_type, device="cuda").as_strided(shape_o, stride_o)
    stats_gpu = torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device="cuda")

    # forward cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
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
        is_inference=False,
        attn_scale=attn_scale,
        bias=bias,
        use_alibi_mask=is_alibi,
        use_padding_mask=is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        use_causal_mask=is_causal,
        dropout=dropout_tuple if is_dropout else None,
        rng_dump=rng_dump,
    )

    o.set_output(True).set_dim(shape_o).set_stride(stride_o)
    if is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
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
        q_ragged_offset: q_ragged_offset_gpu,
        k_ragged_offset: k_ragged_offset_gpu,
        v_ragged_offset: v_ragged_offset_gpu,
        o_ragged_offset: o_ragged_offset_gpu,
        o: o_gpu,
        stats: stats_gpu,
        rng_dump: rng_dump_gpu,
    }

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    if cudnn.backend_version() < 8906 and is_padding:
        # zero out padded region of the output and stats
        for i, m in enumerate(seq_len_q_gpu):
            o_gpu[i, :, m:, :] = 0
            stats_gpu[i, :, m:, :] = 0

    # backward cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)
    o = graph.tensor_like(o_gpu)
    dO = graph.tensor_like(dO_gpu)
    stats = graph.tensor_like(stats_gpu)

    bias = graph.tensor_like(bias_gpu) if is_bias else None
    dBias = graph.tensor_like(dBias_gpu).set_stride((h_q * s_q * s_kv, s_q * s_kv, s_kv, 1)) if is_bias else None

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
        use_causal_mask=is_causal,
        dropout=dropout_tuple if is_dropout else None,
    )

    dQ.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
    dK.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
    dV.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())
    if is_ragged:
        dQ.set_ragged_offset(q_ragged_offset)
        dK.set_ragged_offset(k_ragged_offset)
        dV.set_ragged_offset(v_ragged_offset)

    graph.validate()
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
        q_ragged_offset: q_ragged_offset_gpu,
        k_ragged_offset: k_ragged_offset_gpu,
        v_ragged_offset: v_ragged_offset_gpu,
        o_ragged_offset: o_ragged_offset_gpu,
    }

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    # compare with torch autograd reference
    q_ref = q_gpu.detach().float()
    q_ref.requires_grad = True
    k_ref = k_gpu.detach().float()
    k_ref.requires_grad = True
    v_ref = v_gpu.detach().float()
    v_ref.requires_grad = True
    dO_ref = dO_gpu.detach().float()

    if is_ragged:
        q_ref = convert_ragged_to_uniform(q_ref, q_ragged_offset_gpu.detach())
        k_ref = convert_ragged_to_uniform(k_ref, k_ragged_offset_gpu.detach())
        v_ref = convert_ragged_to_uniform(v_ref, v_ragged_offset_gpu.detach())
        dO_ref = convert_ragged_to_uniform(dO_ref, o_ragged_offset_gpu.detach())

    if is_bias:
        bias_ref = bias_gpu.detach().float()
        bias_ref.requires_grad = True

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
        is_causal=is_causal,
        dropout_prob=dropout_prob,
        dropout_mask=rng_dump_ref if is_dropout else None,
        compute_stats=False,
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
        dQ_gpu = convert_ragged_to_uniform(dQ_gpu, q_ragged_offset_gpu.detach())
        dK_gpu = convert_ragged_to_uniform(dK_gpu, k_ragged_offset_gpu.detach())
        dV_gpu = convert_ragged_to_uniform(dV_gpu, v_ragged_offset_gpu.detach())

    if is_padding:
        # zero out padded region of the output for comparison
        for i, (m, n) in enumerate(zip(seq_len_q_ref, seq_len_kv_ref)):
            dQ_ref[i, :, m:, :] = 0
            dQ_gpu[i, :, m:, :] = 0
            dK_ref[i, :, n:, :] = 0
            dK_gpu[i, :, n:, :] = 0
            dV_ref[i, :, n:, :] = 0
            dV_gpu[i, :, n:, :] = 0
            if is_bias:
                dBias_ref[i, :, m:, :] = 0
                dBias_ref[i, :, :, n:] = 0

    assert compare_tensors(dQ_ref, dQ_gpu, "dQ") == 0
    assert compare_tensors(dK_ref, dK_gpu, "dK", atol=2e-2 if input_type != torch.bfloat16 else 4e-2) == 0
    assert compare_tensors(dV_ref, dV_gpu, "dV") == 0
    if is_bias:
        assert compare_tensors(dBias_ref, dBias_gpu, "dBias") == 0


if __name__ == "__main__":
    """
    option_forward = (input_type, layout, head_group, is_bias, is_alibi, is_padding, is_causal, is_dropout, is_ragged, is_infer)
    option_backward = (input_type, layout, head_group, is_bias, is_alibi, is_padding, is_causal, is_dropout, is_ragged)
    test_sdpa(torch.float16, "bs3hd", "multi_head", False, False, False, False, False, False, False)
    test_sdpa_backward(torch.float16, "bs3hd", "multi_head", False, False, False, False, False, False)
    """

    print("==========running forward tests==========")
    for option in all_options_forward:
        try:
            print(f"Running {option}")
            test_sdpa(*option)
        except pytest.skip.Exception as e:
            print(f"Skipped {option}\n{e}")

    print("==========running backward tests==========")
    for option in all_options_backward:
        try:
            print(f"Running {option}")
            test_sdpa_backward(*option)
        except pytest.skip.Exception as e:
            print(f"Skipped {option}\n{e}")
