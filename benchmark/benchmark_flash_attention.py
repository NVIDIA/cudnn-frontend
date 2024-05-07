import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import os
import csv
import itertools

from einops import rearrange, repeat


# benchmarking functions from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/benchmark.py
def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(*inputs, y, grad):
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*inputs, y=y, grad=grad)",
        globals={"f": f, "inputs": inputs, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_fwd_bwd(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


try:
    import cudnn
except ImportError:
    cudnn = None
assert cudnn is not None


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


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    # batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, "b t h d -> b h t d")
    k = rearrange(k, "b s h d -> b h s d")
    v = rearrange(v, "b s h d -> b h s d")
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean


print("Is flash sdp enabled in Pytorch : " + str(torch._C._get_flash_sdp_enabled()))
print("cudnn backend version : " + str(cudnn.backend_version()))

filename = "benchmark_results.csv"
csvfile = open(filename, "w")
csvwriter = csv.writer(csvfile)

repeats = 30
device = "cuda"
dtype = torch.bfloat16

bs_seqlen_vals = [
    (32, 512),
    (16, 1024),
    (8, 2048),
    (4, 4096),
    (2, 8192),
    (1, 16384),
    (1, 32768),
    (1, 65536),
]
causal_vals = [False, True]
headdim_vals = [128]
dim = 2048
dropout_p = 0.0

fields = [
    "Batch",
    "Number of heads",
    "Sequence length",
    "Head dim",
    "causal",
    "dropout_p",
    "pytorch (TFlops/s fwd)",
    "pytorch (TFlops/s bwd)",
    "pytorch (TFlops/s fwd + bwd)",
    "cudnn BF16 (TFlops/s fwd)",
    "cudnn BF16 (TFlops/s bwd)",
    "cudnn BF16 (TFlops/s fwd + bwd)",
]

csvwriter.writerow(fields)

methods = ["Pytorch"]
if cudnn is not None:
    methods += ["cudnn_bf16"]

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

for causal, headdim, bs_seqlen in itertools.product(
    causal_vals, headdim_vals, bs_seqlen_vals
):
    batch_size, seqlen = bs_seqlen
    config = (causal, headdim, batch_size, seqlen)
    nheads = dim // headdim

    if "Pytorch" in methods:
        qkv = torch.randn(
            batch_size,
            seqlen,
            3,
            nheads,
            headdim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        try:
            qkv = qkv.detach().requires_grad_(True)
            f, b = time_fwd_bwd(
                attention_pytorch,
                qkv,
                dropout_p,
                causal=causal,
                repeats=repeats,
                verbose=False,
            )
        except:  # Skip if OOM
            f, b = float("nan"), float("nan")
        time_f[config, "Pytorch"] = f
        time_b[config, "Pytorch"] = b

    if (
        ("cudnn_fp16" in methods or "cudnn_bf16" in methods)
        and device == "cuda"
        and cudnn is not None
    ):
        os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "0"

        is_causal = causal
        is_dropout = False if (abs(dropout_p - 0.0) < 1e-6) else True
        is_infer = False
        input_type = dtype
        attn_scale = headdim ** (-0.5)
        dropout_prob = dropout_p if is_dropout else 0.0

        shape_qkvo = (batch_size, nheads, seqlen, headdim)
        stride_qkv = (seqlen * 3 * nheads * headdim, headdim, 3 * nheads * headdim, 1)
        stride_o = (seqlen * nheads * headdim, headdim, nheads * headdim, 1)
        offset_q, offset_k, offset_v = [nheads * headdim * i for i in range(3)]

        qkv_gpu = (
            torch.randn(
                batch_size * seqlen * 3 * nheads * headdim,
                dtype=input_type,
                device="cuda",
            )
            - 0.5
        )
        q_gpu, k_gpu, v_gpu = [
            torch.as_strided(qkv_gpu, shape_qkvo, stride_qkv, storage_offset=offset)
            for offset in [offset_q, offset_k, offset_v]
        ]
        o_gpu = torch.empty(*shape_qkvo, dtype=input_type, device="cuda").as_strided(
            shape_qkvo, stride_o
        )
        dQ_gpu, dK_gpu, dV_gpu = [
            torch.empty_like(tensor) for tensor in [q_gpu, k_gpu, v_gpu]
        ]
        dO_gpu = torch.randn_like(o_gpu) - 0.5

        stats_gpu = (
            torch.empty(
                batch_size, nheads, seqlen, 1, dtype=torch.float32, device="cuda"
            )
            if not is_infer
            else None
        )

        if is_dropout:
            seed_gpu = torch.full(
                (1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda"
            )
            offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

        # cuDNN graph forward
        graph_fwd = cudnn.pygraph(
            io_data_type=convert_to_cudnn_type(input_type),
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        q_fwd = graph_fwd.tensor_like(q_gpu)
        k_fwd = graph_fwd.tensor_like(k_gpu)
        v_fwd = graph_fwd.tensor_like(v_gpu)

        if is_dropout:
            seed_fwd = graph_fwd.tensor_like(seed_gpu)
            offset_fwd = graph_fwd.tensor_like(offset_gpu)
            dropout_tuple = (dropout_prob, seed_fwd, offset_fwd)

        o_fwd, stats_fwd = graph_fwd.sdpa(
            q=q_fwd,
            k=k_fwd,
            v=v_fwd,
            is_inference=is_infer,
            attn_scale=attn_scale,
            use_causal_mask=is_causal,
            dropout=dropout_tuple if is_dropout else None,
        )

        o_fwd.set_output(True).set_dim(o_gpu.size()).set_stride(o_gpu.stride())
        (
            stats_fwd.set_output(True)
            .set_dim(stats_gpu.size())
            .set_stride(stats_gpu.stride())
            .set_data_type(cudnn.data_type.FLOAT)
            if not is_infer
            else None
        )

        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A])
        graph_fwd.check_support()
        graph_fwd.build_plans()

        # cuDNN graph backward
        graph_bwd = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        q_bwd = graph_bwd.tensor_like(q_gpu)
        k_bwd = graph_bwd.tensor_like(k_gpu)
        v_bwd = graph_bwd.tensor_like(v_gpu)
        o_bwd = graph_bwd.tensor_like(o_gpu)
        dO_bwd = graph_bwd.tensor_like(dO_gpu)
        stats_bwd = graph_bwd.tensor_like(stats_gpu)

        if is_dropout:
            seed_bwd = graph_fwd.tensor_like(seed_gpu)
            offset_bwd = graph_fwd.tensor_like(offset_gpu)
            dropout_tuple = (dropout_prob, seed_bwd, offset_bwd)

        dQ_bwd, dK_bwd, dV_bwd = graph_bwd.sdpa_backward(
            q=q_bwd,
            k=k_bwd,
            v=v_bwd,
            o=o_bwd,
            dO=dO_bwd,
            stats=stats_bwd,
            attn_scale=attn_scale,
            use_causal_mask=is_causal,
            dropout=dropout_tuple if is_dropout else None,
        )

        dQ_bwd.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
        dK_bwd.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
        dV_bwd.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())

        graph_bwd.validate()
        graph_bwd.build_operation_graph()
        graph_bwd.create_execution_plans([cudnn.heur_mode.A])
        graph_bwd.check_support()
        graph_bwd.build_plans()

        variant_pack_fwd = {
            q_fwd: q_gpu,
            k_fwd: k_gpu,
            v_fwd: v_gpu,
            o_fwd: o_gpu,
            stats_fwd: stats_gpu,
        }
        variant_pack_bwd = {
            q_bwd: q_gpu,
            k_bwd: k_gpu,
            v_bwd: v_gpu,
            o_bwd: o_gpu,
            dO_bwd: dO_gpu,
            stats_bwd: stats_gpu,
            dQ_bwd: dQ_gpu,
            dK_bwd: dK_gpu,
            dV_bwd: dV_gpu,
        }
        if is_dropout:
            variant_pack_fwd[seed_fwd] = seed_gpu
            variant_pack_fwd[offset_fwd] = offset_gpu
            variant_pack_bwd[seed_bwd] = seed_gpu
            variant_pack_bwd[offset_bwd] = offset_gpu

        workspace = torch.empty(
            max(graph_fwd.get_workspace_size(), graph_bwd.get_workspace_size()),
            device="cuda",
            dtype=torch.uint8,
        )

        f = time_fwd(
            graph_fwd.execute,
            variant_pack_fwd,
            workspace,
            repeats=repeats,
            verbose=False,
        )
        b = time_fwd(
            graph_bwd.execute,
            variant_pack_bwd,
            workspace,
            repeats=repeats,
            verbose=False,
        )

        time_f[config, "cudnn_bf16"] = f
        time_b[config, "cudnn_bf16"] = b

    row = []
    row.append(str(batch_size))
    row.append(str(2048 // headdim))
    row.append(str(seqlen))
    row.append(str(headdim))
    row.append(str(causal))
    row.append(str(dropout_p))

    print(
        f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###"
    )
    for method in methods:
        time_f_b[config, method] = time_f[config, method] + time_b[config, method]
        speed_f[config, method] = efficiency(
            flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
            time_f[config, method],
        )
        speed_b[config, method] = efficiency(
            flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
            time_b[config, method],
        )
        speed_f_b[config, method] = efficiency(
            flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
            time_f_b[config, method],
        )
        print(
            f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
            f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
            f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
        )
        row.append(str(speed_f[config, method]))
        row.append(str(speed_b[config, method]))
        row.append(str(speed_f_b[config, method]))
    csvwriter.writerow(row)

csvfile.close()
