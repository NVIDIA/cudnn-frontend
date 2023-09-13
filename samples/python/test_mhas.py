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

def perr(a, b):
    a, b = a.float(), b.float()
    diff = (a-b)
    return (diff.abs().sum() / a.abs().sum()).item()

def mean_avg_error(a, b):
    a, b = a.float(), b.float()
    diff = (a-b)
    return diff.abs().mean().item()

def compare_tensors(a_, b_, tensor_name):

    print("================================================")
    print(tensor_name)
    print("================================================")

    n_elem = torch.numel(a_)

    abs_err_tol = 0.1
    rel_err_tol = 0.1

    if a_.cuda:
        a = a_.to(device='cpu')

    if b_.cuda:
        b = b_.to(device='cpu')

    mae = mean_avg_error(a, b)
    some_perr = perr(a, b)

    absolute_error = torch.abs(a - b)
    relative_error = torch.div(absolute_error, a + 0.0000000001)

    max_abs_error = torch.max(absolute_error)
    max_rel_error = torch.max(relative_error)

    abs_error_indices = absolute_error > abs_err_tol
    rel_error_indices = relative_error > rel_err_tol

    n_abs_errors = torch.sum(abs_error_indices)
    n_rel_errors = torch.sum(rel_error_indices)

    error_indices = torch.logical_and(abs_error_indices, rel_error_indices)
    n_errors = torch.sum(error_indices)

    print("Absolute Tolerance = {}".format(abs_err_tol))
    print("Relative Tolerance = {}".format(rel_err_tol))
    print("Number of elements = {}".format(n_elem))

    print("Number of absolute errors = {} ({:.2f}%)". format(n_abs_errors, (n_abs_errors * 100)/n_elem))
    print("Number of relative errors = {} ({:.2f}%)". format(n_rel_errors, (n_rel_errors * 100)/n_elem))    

    print("Number of errors (absolute and relative) = {} ({:.2f}%)". format(n_errors, (n_errors * 100)/n_elem))

    print("Maximum absolute error = {:.4f}".format(max_abs_error))
    print("Maximum relative error = {:.4f}".format(max_rel_error))
    print("Mean average error = {:.4f}".format(mae))
    print("Perr error = {:.4f}".format(some_perr))
    print("Number of Nans = {} ({:.2f}%)".format(torch.sum(torch.isnan(b)), torch.sum(torch.isnan(b) * 100/n_elem)))
    print("Number of Zeros = {} ({:.2f}%)".format(n_elem - torch.count_nonzero(b), (n_elem - torch.count_nonzero(b)) * 100/n_elem))

    print("================================================\n")

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
    m = m.view(1, -1, 1, 1).to(device='cuda')

    return m

class scaled_dot_product_attention(torch.nn.Module):
    def forward(self, query, key, value, is_causal, is_infer, bias, is_alibi, attn_scale):
        _, h, s_q, d = query.shape
        _, _, s_kv, _ = key.shape

        S = query @ key.transpose(-2, -1) * attn_scale
        S = S.to(dtype=torch.float32)
        if bias is not None:
            S.add_(bias)
        if is_alibi:
            S.add_(((torch.arange(s_kv, dtype=torch.float32, device = 'cuda')) - torch.arange(s_q, dtype=torch.float32, device = 'cuda').view(-1, 1)) * get_slopes(h))
        if is_causal:
            causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device = 'cuda').triu_(diagonal=1)
            S.masked_fill_(causal_mask, float('-inf')) 

        Stats = None
        if not is_infer:
            row_max, _ = torch.max(S, -1, True)
            row_exp = torch.exp(S - row_max)
            row_sum = torch.sum(row_exp, -1, True)
            Stats = row_max + torch.log(row_sum)

        return torch.softmax(S, dim=-1).to(dtype=value.dtype) @ value, Stats
    
alibi_mask_options = [True, False]
padding_mask_options = [True, False]
causal_mask_options = [True, False]
layout_options      = ["non_interleaved", "bs3hd", "sbh3d"]
dropout             = [False]
is_infer_options    = [True, False]
bias                = [True, False]
input_type_options  = [torch.float16, torch.bfloat16]

all_options = [elem for elem in itertools.product(*[alibi_mask_options, padding_mask_options, causal_mask_options, layout_options, dropout, is_infer_options, bias, input_type_options])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

@pytest.mark.skipif(cudnn.backend_version() < 8903, reason="requires cudnn 8.9 or higher")
def test_scale_dot_product_flash_attention(param_extract):

    alibi_mask, padding_mask, causal_mask, layout, dropout_enable, is_infer, bias_enable, input_type = param_extract
    
    if alibi_mask and cudnn.backend_version() < 8904:
        pytest.skip("ALiBi mask is only supported 8.9.4 onwards.")
        
    if padding_mask and cudnn.backend_version() < 8903:
        pytest.skip("Padding mask is only supported 8.9.3 onwards.")

    s_q_choices = [256, 512, 1024, 2048] 
    d_choices   = [64,128]
    
    b = 32
    h = 12
    s_q  = random.choice(s_q_choices)
    s_kv  = s_q
    d = random.choice(d_choices)
    
    print(param_extract)
    print ("d = {} s_kv = {} s_q = {} ".format(d, s_kv, s_q))

    attn_scale = 0.125
    
    if dropout_enable == False:
        dropout_prob = 1.0
    else:
        dropout_prob = 0.1

    shape_Q = (b, h, s_q, d)

    shape_K = (b, h, d, s_kv)

    shape_V = (b, h, s_kv, d)

    stride_sbh3d = (3 * h * d, 3 * d, b * 3 * h * d, 1)
    stride_sbh3d_t = (3 * h * d, 3 * d, 1, b * 3 * h * d)
    stride_sbhd = (h * d, d, b * h * d, 1)

    stride_bs3hd = (s_q * 3 * h * d, d, 3 * h * d, 1)
    stride_bs3hd_t = (s_q * 3 * h * d, d, 1, 3 * h * d)
    stride_bshd = (s_q * h * d, d, h * d, 1)

    offset_multiple_sbh3d = d
    offset_multiple_bs3hd = h * d
    
    bias_gpu = torch.randn(b, 1, s_q, s_kv, requires_grad=False, device="cuda", dtype = input_type) if bias_enable else None

    if layout == 'sbh3d':
        stride_Q = stride_sbh3d
        stride_K = stride_sbh3d_t
        stride_V = stride_sbh3d

        stride_O = stride_sbhd

        offset_Q = offset_multiple_sbh3d * 0
        offset_K = offset_multiple_sbh3d * 1
        offset_V = offset_multiple_sbh3d * 2
    elif layout == 'bs3hd':
        stride_Q = stride_bs3hd
        stride_K = stride_bs3hd_t
        stride_V = stride_bs3hd

        stride_O = stride_bshd

        offset_Q = offset_multiple_bs3hd * 0
        offset_K = offset_multiple_bs3hd * 1
        offset_V = offset_multiple_bs3hd * 2
    elif layout == 'non_interleaved':
        stride_Q = (1 * d * s_q *  h, 1 * d *  s_q, 1 * d, 1)
        stride_K = (1 * d * s_kv * h, 1 * d * s_kv, 1, 1 * d)
        stride_V = (1 * d * s_kv * h, 1 * d * s_kv, 1 * d, 1)
        
        stride_O = (d * s_q * h, d * s_q, d, 1)

        offset_Q = 0
        offset_K = offset_Q + b * d * s_q *  h
        offset_V = offset_K + b * d * s_kv * h

    else:
        assert False, "Layout should be either sbh3d or bs3hd or non_interleaved"

    qkv_gpu = 1 *  (torch.randn(b * s_q * 3 * h * d, dtype=input_type, device="cuda") - 0.5)

    Q_gpu = torch.as_strided(qkv_gpu, shape_Q, stride_Q, storage_offset=offset_Q)
    K_gpu = torch.as_strided(qkv_gpu, shape_K, stride_K, storage_offset=offset_K)
    V_gpu = torch.as_strided(qkv_gpu, shape_V, stride_V, storage_offset=offset_V)
    
    if padding_mask:
        seq_len_Q_gpu = torch.full((b,1,1,1), s_q, dtype=torch.int32, device="cuda")
        seq_len_KV_gpu = torch.full((b,1,1,1), s_kv, dtype=torch.int32, device="cuda")

    Attn_scale_cpu = torch.full((1,1,1,1), attn_scale, dtype=torch.float32, device="cpu")

    Seed_gpu = torch.full((1,1,1,1), 123456, dtype=torch.int64, device="cuda")
    Offset_gpu = torch.full((1,1,1,1), 1, dtype=torch.int64, device="cuda")
    
    # Cudnn graph
    graph = cudnn.pygraph(io_data_type = convert_to_cudnn_type(input_type), intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)
    Q = graph.tensor(name = "Q", dim = Q_gpu.size(), stride = Q_gpu.stride(), data_type = convert_to_cudnn_type(Q_gpu.dtype))
    K = graph.tensor(name = "K", dim = K_gpu.size(), stride = K_gpu.stride(), data_type = convert_to_cudnn_type(K_gpu.dtype))
    V = graph.tensor(name = "V", dim = V_gpu.size(), stride = V_gpu.stride(), data_type = convert_to_cudnn_type(V_gpu.dtype))
    Attn_scale = graph.tensor(name = "Attn_scale", dim = Attn_scale_cpu.size(), stride = Attn_scale_cpu.stride(), data_type = convert_to_cudnn_type(Attn_scale_cpu.dtype), is_pass_by_value = True)
    Seed = graph.tensor(name = "Seed", dim = Seed_gpu.size(), stride = Seed_gpu.stride(), data_type = convert_to_cudnn_type(Seed_gpu.dtype))
    Offset = graph.tensor(name = "Offset", dim = Offset_gpu.size(), stride = Offset_gpu.stride(), data_type = convert_to_cudnn_type(Offset_gpu.dtype))
    
    Bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride(),data_type = convert_to_cudnn_type(Q_gpu.dtype)) if bias_enable else None
    
    dropout_tuple = None
    if dropout_enable == True:
        dropout_tuple = (dropout_prob, Seed, Offset)

    seq_len_Q = None
    seq_len_KV = None
    if padding_mask:
        seq_len_Q = graph.tensor(name = "seq_len_Q", dim = seq_len_Q_gpu.size(), stride = seq_len_Q_gpu.stride(), data_type = convert_to_cudnn_type(seq_len_Q_gpu.dtype))
        seq_len_KV = graph.tensor(name = "seq_len_KV", dim = seq_len_KV_gpu.size(), stride = seq_len_KV_gpu.stride(), data_type = convert_to_cudnn_type(seq_len_KV_gpu.dtype))
    

    O, Stats = graph.scaled_dot_product_flash_attention(name = "scaled_dot_product_flash_attention"
                                            , q = Q, k = K, v = V
                                            , seq_len_q = seq_len_Q, seq_len_kv = seq_len_KV
                                            , is_inference = is_infer
                                            , bias = Bias
                                            , dropout = dropout_tuple       
                                            , attn_scale = Attn_scale
                                            , use_alibi_mask = alibi_mask
                                            , use_padding_mask = padding_mask
                                            , use_causal_mask = causal_mask
                                            )

    O.set_output(True).set_stride(stride_O)
    
    if is_infer == False:
        Stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    
    graph.check_support()
    
    graph.build()

    O_actual = torch.zeros(b * s_q * h * d, dtype=input_type, device="cuda")
    Stats_actual = torch.zeros(b * h * s_q * 1, dtype=torch.float32, device="cuda")

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    cudnn_to_torch_tensor = {Q: Q_gpu, K: K_gpu, V: V_gpu
                   , Seed: Seed_gpu, Offset: Offset_gpu
                   , Attn_scale: Attn_scale_cpu
                   , O: O_actual, Stats: Stats_actual}
    
    if bias_enable:
        cudnn_to_torch_tensor[Bias] = bias_gpu

    if padding_mask:
        cudnn_to_torch_tensor[seq_len_Q] = seq_len_Q_gpu
        cudnn_to_torch_tensor[seq_len_KV] = seq_len_KV_gpu

    graph.execute(cudnn_to_torch_tensor, workspace)

    torch.set_printoptions(precision = 2, linewidth = 2560, threshold = 1000000, sci_mode = False)

    Stats_reorg = Stats_actual.view(b, h, s_q, 1)

    if layout == 'sbh3d':
        O_reorg = O_actual.view([s_q, b, h, d]).permute(1, 2, 0, 3)
    elif layout == 'bs3hd':
        O_reorg = O_actual.view([b, s_q, h, d]).permute(0, 2, 1, 3)
    elif layout == 'non_interleaved':
        O_reorg = O_actual.view([b, h, s_q, d])
    else:
        assert False, "Layout should be either sbh3d or bs3hd or non_interleaved"

    # Cpu reference
    sdpa = scaled_dot_product_attention()
    O_expected, Stats_expected = sdpa(Q_gpu, K_gpu.permute(0, 1, 3, 2), V_gpu, is_causal = causal_mask, is_infer = is_infer, bias = bias_gpu, is_alibi = alibi_mask, attn_scale = attn_scale)

    if is_infer == False:
        assert compare_tensors(Stats_expected, Stats_reorg, "Stats") == 0
    
    assert compare_tensors(O_expected, O_reorg, "O") == 0
    
if __name__ == "__main__":
    test_scale_dot_product_flash_attention((True, "bs3hd", False, False, True))