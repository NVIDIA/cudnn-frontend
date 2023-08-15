import cudnn
import pytest
import torch

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

causal_mask_options = [True, False]
layout_options      = ["bs3hd", "sbh3d"]
dropout             = [True, False]
is_infer            = [True, False]
bias                = [True, False]

all_options = [elem for elem in itertools.product(*[causal_mask_options, layout_options, dropout, is_infer, bias])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

@pytest.mark.skipif(cudnn.get_cudnn_version() < 8903, reason="requires cudnn 8.9 or higher")
def test_scale_dot_product_flash_attention(param_extract):

    causal_mask, layout, dropout_enable, infer_mode, bias_enable = param_extract
    
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
     
    stride_explicit_q   = (h * s_q * d, s_q * d, d, 1)
    stride_explicit_k_t = (h * d * s_kv, d * s_kv, 1, d)
    stride_explicit_v   = (h * d * s_kv, d * s_kv, d, 1)

    offset_multiple_sbh3d = d
    offset_multiple_bs3hd = h * d
    
    bias_gpu = torch.randn(b, 1, s_q, s_kv, requires_grad=False, device="cuda", dtype = torch.bfloat16)

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
    else:
        assert False, "Layout should be either sbh3d or bs3hd"

    qkv_gpu = 1 *  (torch.randn(b * s_q * 3 * h * d, dtype=torch.bfloat16, device="cuda") - 0.5)

    Q_gpu = torch.as_strided(qkv_gpu, shape_Q, stride_Q, storage_offset=offset_Q)
    K_gpu = torch.as_strided(qkv_gpu, shape_K, stride_K, storage_offset=offset_K)
    V_gpu = torch.as_strided(qkv_gpu, shape_V, stride_V, storage_offset=offset_V)

    Attn_scale_cpu = torch.full((1,1,1,1), attn_scale, dtype=torch.float32, device="cpu")

    Seed_gpu = torch.full((1,1,1,1), 123456, dtype=torch.int64, device="cuda")
    Offset_gpu = torch.full((1,1,1,1), 1, dtype=torch.int64, device="cuda")

    if layout == 'sbh3d':
        qkv_cpu = qkv_gpu.clone().detach().to(device='cpu').view([s_q, b, h, 3, d])
        Q_cpu = qkv_cpu[:, :, :, 0, :].contiguous().permute(1, 2, 0, 3)
        K_cpu = qkv_cpu[:, :, :, 1, :].contiguous().permute(1, 2, 0, 3)
        V_cpu = qkv_cpu[:, :, :, 2, :].contiguous().permute(1, 2, 0, 3)
    elif layout == 'bs3hd':
        qkv_cpu = qkv_gpu.clone().detach().to(device='cpu').view([b, s_q, 3, h, d])
        Q_cpu = qkv_cpu[:, :, 0, :, :].contiguous().permute(0, 2, 1, 3)
        K_cpu = qkv_cpu[:, :, 1, :, :].contiguous().permute(0, 2, 1, 3)
        V_cpu = qkv_cpu[:, :, 2, :, :].contiguous().permute(0, 2, 1, 3)
    else:
        assert False, "Layout should be either sbh3d or bs3hd"


    torch.eq(Q_gpu.to(device='cpu'), Q_cpu)
    torch.eq(K_gpu.to(device='cpu'), K_cpu.permute(0, 1, 3, 2))
    torch.eq(V_gpu.to(device='cpu'), V_cpu)

    bias_cpu = bias_gpu.clone().detach().to(device='cpu')
    
    # Cudnn graph
    graph = cudnn.pygraph(io_data_type = cudnn.data_type.BFLOAT16, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)
    Q = graph.tensor(name = "Q", dim = Q_gpu.size(), stride = Q_gpu.stride(), data_type = convert_to_cudnn_type(Q_gpu.dtype))
    K = graph.tensor(name = "K", dim = K_gpu.size(), stride = K_gpu.stride(), data_type = convert_to_cudnn_type(K_gpu.dtype))
    V = graph.tensor(name = "V", dim = V_gpu.size(), stride = V_gpu.stride(), data_type = convert_to_cudnn_type(V_gpu.dtype))
    Bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride(),data_type = convert_to_cudnn_type(Q_gpu.dtype))
    Attn_scale = graph.tensor(name = "Attn_scale", dim = Attn_scale_cpu.size(), stride = Attn_scale_cpu.stride(), data_type = convert_to_cudnn_type(Attn_scale_cpu.dtype), is_pass_by_value = True)
    Seed = graph.tensor(name = "Seed", dim = Seed_gpu.size(), stride = Seed_gpu.stride(), data_type = convert_to_cudnn_type(Seed_gpu.dtype))
    Offset = graph.tensor(name = "Offset", dim = Offset_gpu.size(), stride = Offset_gpu.stride(), data_type = convert_to_cudnn_type(Offset_gpu.dtype))
    
    dropout_tuple = None
    if dropout_enable == True:
        dropout_tuple = (dropout_prob, Seed, Offset)

    if bias_enable:
        O, Stats = graph.scaled_dot_product_flash_attention(name = "scaled_dot_product_flash_attention"
                                                , q = Q, k = K, v = V
                                                , is_inference = infer_mode
                                                , bias = Bias
                                                , dropout = dropout_tuple       
                                                , attn_scale = Attn_scale
                                                , use_causal_mask = causal_mask
                                                )
    else:        
        O, Stats = graph.scaled_dot_product_flash_attention(name = "scaled_dot_product_flash_attention"
                                    , q = Q, k = K, v = V
                                    , is_inference = infer_mode
                                    , dropout = dropout_tuple
                                    , attn_scale = Attn_scale
                                    , use_causal_mask = causal_mask
                                    )

    O.set_output(True).set_stride(stride_O)
    
    if is_infer == False:
        Stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    
    graph.check_support()
    
    graph.build()

    O_actual = torch.zeros(b * s_q * h * d, dtype=torch.bfloat16, device="cuda")
    Stats_actual = torch.zeros(b * h * s_q * 1, dtype=torch.float32, device="cuda")

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    print("Executing the cudnn graph execute")
    graph.execute({Q: Q_gpu, K: K_gpu, V: V_gpu, Seed: Seed_gpu, Offset: Offset_gpu
                   , Attn_scale: Attn_scale_cpu, Bias: bias_gpu
                   , O: O_actual, Stats: Stats_actual}
                   , workspace)

    torch.set_printoptions(precision = 2, linewidth = 2560, threshold = 1000000, sci_mode = False)

    stats_reorg = Stats_actual.view(b, h, s_q, 1)

    if layout == 'sbh3d':
        O_reorg = O_actual.view([s_q, b, h, d]).permute(1, 2, 0, 3)
    elif layout == 'bs3hd':
        O_reorg = O_actual.view([b, s_q, h, d]).permute(0, 2, 1, 3)
    else:
        assert False, "Layout should be either sbh3d or bs3hd"

    # Cpu reference
    S_cpu = torch.matmul(Q_cpu.float(), K_cpu.permute(0, 1, 3, 2).float())

    after_scale_S_cpu = (S_cpu * attn_scale)
    
    if bias_enable:
        after_scale_S_cpu = after_scale_S_cpu + bias_cpu
        
    if causal_mask is True:
        causal_mask_cpu = torch.triu(torch.ones(s_q, s_q, dtype=torch.bool, device='cpu'), 1)
        
        S_masked_cpu = after_scale_S_cpu.masked_fill_(causal_mask_cpu, float('-inf'))
    else:
        S_masked_cpu = after_scale_S_cpu

    row_max_cpu = torch.max(S_masked_cpu, -1, True)[0]

    row_exp_cpu = torch.exp(S_masked_cpu - row_max_cpu)

    row_sum_cpu = torch.sum(row_exp_cpu, -1, True).to(torch.float)

    softmax_stats_cpu = row_max_cpu + torch.log(row_sum_cpu)

    softmax_cpu = torch.divide(row_exp_cpu, row_sum_cpu)

    softmax_w_func_cpu = torch.softmax(S_masked_cpu, -1)

    torch.eq(softmax_cpu, softmax_w_func_cpu)

    O_cpu = torch.matmul(softmax_cpu, V_cpu.float())


    if dropout_enable == False:
        if is_infer == False:
            assert compare_tensors(softmax_stats_cpu, stats_reorg, "stats") == 0
        assert compare_tensors(O_cpu, O_reorg, "O") == 0
    else:
        compare_tensors(O_cpu, O_reorg, "O")


causal_mask_options = [True, False]
dropout             = [True, False]
is_infer            = [True, False]
bias                = [True, False]

all_options_non_interleaved = [elem for elem in itertools.product(*[causal_mask_options, dropout, is_infer, bias])]

@pytest.fixture(params=all_options_non_interleaved)
def param_extract_non_interleaved(request):
  return request.param

@pytest.mark.skipif(cudnn.get_cudnn_version() < 8903, reason="requires cudnn 8.9 or higher")
def test_scale_dot_product_flash_attention_non_interleaved(param_extract_non_interleaved):

    causal_mask, dropout_enable, infer_mode, bias_enable = param_extract_non_interleaved
        
    s_q_choices = [256, 512, 1024, 2048] 
    d_choices   = [64,128]
    
    b = 32
    h = 12
    s_q  = random.choice(s_q_choices)
    s_kv  = int(s_q / 2)
    d = random.choice(d_choices)
    
    attn_scale = 0.125
    
    Attn_scale_cpu = torch.full((1,1,1,1), attn_scale, dtype=torch.float32, device="cpu")
    
    if dropout_enable == False:
        dropout_prob = 1.0
    else:
        dropout_prob = 0.1
    Seed_gpu = torch.full((1,1,1,1), 123456, dtype=torch.int64, device="cuda")
    Offset_gpu = torch.full((1,1,1,1), 1, dtype=torch.int64, device="cuda")


    shape_Q = (b, h, s_q, d)
    shape_K = (b, h, d, s_kv)
    shape_V = (b, h, s_kv, d)

    stride_Q = (1 * d * s_q *  h, 1 * d *  s_q, 1 * d, 1)
    stride_K = (1 * d * s_kv * h, 1 * d * s_kv, 1, 1 * d)
    stride_V = (1 * d * s_kv * h, 1 * d * s_kv, 1 * d, 1)
    
    q_buffer = 1 *  (torch.randn(b * h * s_q * d,  dtype=torch.bfloat16, device="cuda") - 0.5)
    k_buffer = 1 *  (torch.randn(b * h * d * s_kv, dtype=torch.bfloat16, device="cuda") - 0.5)
    v_buffer = 1 *  (torch.randn(b * h * s_kv * d, dtype=torch.bfloat16, device="cuda") - 0.5)

    Q_gpu = torch.as_strided(q_buffer, shape_Q, stride_Q)
    K_gpu = torch.as_strided(k_buffer, shape_K, stride_K)
    V_gpu = torch.as_strided(v_buffer, shape_V, stride_V)
    bias_gpu = torch.randn(b, 1, s_q, s_kv, requires_grad=False, device="cuda", dtype = torch.bfloat16)

    Q_cpu = Q_gpu.clone().to(device="cpu")
    K_cpu = K_gpu.clone().to(device="cpu")
    V_cpu = V_gpu.clone().to(device="cpu")
    bias_cpu = bias_gpu.clone().detach().to(device='cpu')
    

    graph = cudnn.pygraph(io_data_type = cudnn.data_type.BFLOAT16, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)
    Q = graph.tensor(name = "Q", dim = Q_gpu.size(), stride = Q_gpu.stride(), data_type = convert_to_cudnn_type(Q_gpu.dtype))
    K = graph.tensor(name = "K", dim = K_gpu.size(), stride = K_gpu.stride(), data_type = convert_to_cudnn_type(K_gpu.dtype))
    V = graph.tensor(name = "V", dim = V_gpu.size(), stride = V_gpu.stride(), data_type = convert_to_cudnn_type(V_gpu.dtype))
    Bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride(),data_type = convert_to_cudnn_type(Q_gpu.dtype))
    Attn_scale = graph.tensor(name = "Attn_scale", dim = Attn_scale_cpu.size(), stride = Attn_scale_cpu.stride(), data_type = convert_to_cudnn_type(Attn_scale_cpu.dtype), is_pass_by_value = True)
    Seed = graph.tensor(name = "Seed", dim = Seed_gpu.size(), stride = Seed_gpu.stride(), data_type = convert_to_cudnn_type(Seed_gpu.dtype))
    Offset = graph.tensor(name = "Offset", dim = Offset_gpu.size(), stride = Offset_gpu.stride(), data_type = convert_to_cudnn_type(Offset_gpu.dtype))

    dropout_tuple = None
    if dropout_enable == True:
        dropout_tuple = (dropout_prob, Seed, Offset)

    if bias_enable:
        O, Stats = graph.scaled_dot_product_flash_attention(name = "scaled_dot_product_flash_attention"
                                                , q = Q, k = K, v = V
                                                , is_inference = infer_mode
                                                , bias = Bias
                                                , dropout = dropout_tuple       
                                                , attn_scale = Attn_scale
                                                , use_causal_mask = causal_mask
                                                )
    else:        
        O, Stats = graph.scaled_dot_product_flash_attention(name = "scaled_dot_product_flash_attention"
                                    , q = Q, k = K, v = V
                                    , is_inference = infer_mode
                                    , dropout = dropout_tuple
                                    , attn_scale = Attn_scale
                                    , use_causal_mask = causal_mask
                                    )

    stride_O = (d * s_q * h, d * s_q, d, 1)
    shape_O  = (b, h, s_q, d)
    O.set_output(True).set_stride(stride_O)

    if is_infer == False:
        Stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        
    graph.check_support()
    
    graph.build()
    
    O_buffer = torch.zeros(b * s_q * h * d, dtype=torch.bfloat16, device="cuda")
    O_actual = torch.as_strided(O_buffer, shape_O, stride_O)
    
    Stats_buffer = torch.zeros(b * s_q * h * 1, dtype=torch.bfloat16, device="cuda")
    Stats_actual = torch.as_strided(Stats_buffer, (b, h, s_q , 1), (h * s_q, s_q, 1, 1))

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    print("Executing the cudnn graph execute")
    graph.execute({Q: Q_gpu, K: K_gpu, V: V_gpu, Seed: Seed_gpu, Offset: Offset_gpu,
                   Attn_scale: Attn_scale_cpu, Bias: bias_gpu,
                   O: O_actual, Stats: Stats_actual},
                   workspace)
    
    torch.set_printoptions(precision = 2, linewidth = 2560, threshold = 1000000, sci_mode = False)
    
    S_cpu = torch.matmul(Q_cpu.float(), K_cpu.float())
    
    after_scale_S_cpu = (S_cpu * attn_scale)
    
    if bias_enable:
        after_scale_S_cpu = after_scale_S_cpu + bias_cpu

    if causal_mask is True:
        causal_mask_cpu = torch.triu(torch.ones(s_q, s_kv, dtype=torch.bool, device='cpu'), 1)
        
        S_masked_cpu = after_scale_S_cpu.masked_fill_(causal_mask_cpu, float('-inf'))
    
    else:
        S_masked_cpu = after_scale_S_cpu

    
    row_max_cpu = torch.max(S_masked_cpu, -1, True)[0]

    row_exp_cpu = torch.exp(S_masked_cpu - row_max_cpu)

    row_sum_cpu = torch.sum(row_exp_cpu, -1, True).to(torch.float)
    
    softmax_stats_cpu = row_max_cpu + torch.log(row_sum_cpu)

    softmax_cpu = torch.divide(row_exp_cpu, row_sum_cpu)

    softmax_w_func_cpu = torch.softmax(S_masked_cpu, -1)
    
    torch.eq(softmax_cpu, softmax_w_func_cpu)

    O_cpu = torch.matmul(softmax_cpu, V_cpu.float())
     
    if dropout_enable == False:
        if is_infer == False:
            assert compare_tensors(softmax_stats_cpu, Stats_actual, "stats") == 0
        assert compare_tensors(O_cpu, O_actual, "O") == 0
    
if __name__ == "__main__":
    test_scale_dot_product_flash_attention((True, "bs3hd", False, False, True))
    test_scale_dot_product_flash_attention_non_interleaved((True, False, False, True))