# Sparse Attention Training Forward Benchmark

Microbenchmark for `cudnn.sparse_attention_forward_wrapper` under the
geometry of three production sparse-attention architectures:

| variant | model | heads (Q/KV) | d_k/d_v | granularity | top-k | index scope | sink |
|---|---|---|---|---|---|---|---|
| `dsv4` | DeepSeek-V4 DSA/CSA | 64/1 (MQA latent, K≡V) | 512/512 (RoPE in-place, dims 448-511) | 1 (token) | 2048 | shared | yes |
| `qwen3.8` | Qwen3.8-Flash-Next QSA | 24/2 | 256/256 | 4 (micro-block) | 512 entries (2048 tok) | shared | no |
| `minimax` | MiniMax-M3 MSA | 64/4 | 128/128 | 128 (block) | 16 blocks (2048 tok) | per KV-head group | no |
| `glm5.2` | GLM-5/5.1/5.2 DSA | 64/1 (MQA latent, K≡V) | 576/512 (512 latent + 64 RoPE) | 1 (token) | 2048 | shared | no |
| `glm5.3-flash` | GLM-5.3-Flash DSA layers | 64/1 (MQA latent, K≡V) | 512/512 (NoPE, rope-free) | 1 (token) | 2048 | shared | no |

Indices are causal-realistic: query row `i` selects unique random entries
from its causal prefix (up to the variant's top-k), with `topk_length`
carrying per-row valid counts, so short-prefix rows are ragged exactly as in
real prefill. Index generation is row-chunked (no `S x S` buffer).

Reported TFLOPS use the exact 2-matmul count (QK^T + PV) computed from the
generated valid lengths:

```
FLOPs = 2 * sum(topk_length) * granularity * heads_per_group * (d_k + d_v)
```

## How to run

No device kernel is registered yet ( `backend="default"` raises), so today
the harness runs against the PyTorch reference — functional only,
reference-speed:

```bash
python benchmark_sparse_attention_forward.py --variant dsv4,qwen3.8,minimax,glm5.2,glm5.3-flash \
    --seqlens 4096 --backend reference --q-chunk 1024
```

Once a kernel registers under `backend="default"`, the same command without
`--backend`/`--q-chunk` produces real numbers, and `profile` mode wraps one
warmed call in `cudaProfilerStart/Stop` + NVTX for nsys/ncu:

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -o safwd \
  python benchmark_sparse_attention_forward.py profile --variant dsv4 --seqlens 8192
```

Options: `--seqlens` (comma-separated, `seqlen_kv = seqlen_q`), `--variant`
(comma-separated subset), `--dtype bfloat16|float16`, `--q-chunk N` (split
each call over query-row chunks — exact under this API because indices are
storage-native global ids), `--warmup/--repeat`, `--csv out.csv`.
