# Sparse Attention Training Forward Benchmark

Microbenchmark for `cudnn.sparse_attention_forward_wrapper` (device kernels)
and the normative PyTorch oracle (`reference_sparse_attention_forward`)
under the geometry of named, production sparse-attention architectures.
Named configs come from real model shapes filed in the upstream tracking
issues (`#825`-`#827`), not synthetic random shapes.

| variant | model | issue | heads (Q/KV) | d_k/d_v | granularity | top-k | index scope | sink | backend |
|---|---|---|---|---|---|---|---|---|---|
| `dsv4` | DeepSeek-V4 DSA core attention | — | 64/1 (MQA latent, K≡V) | 512/512 (RoPE in-place, dims 448-511) | 1 (token) | 2048 | shared | yes | device |
| `csa-dsv4-1024` | DeepSeek-V4 CSA, compressed entries | #825 | 64/1 (MQA latent, K≡V) | 512/512 | 4 (m=4 compression) | 1024 entries | shared | yes | reference |
| `csa-dsv4-512` | DeepSeek-V4 CSA, compressed entries | #825 | 64/1 (MQA latent, K≡V) | 512/512 | 4 (m=4 compression) | 512 entries | shared | yes | reference |
| `qwen3.8` | Qwen3.8-Flash-Next QSA, issue-literal | #826 | 24/2 | 256/256 | 4 (micro-block) | 512 entries (2048 tok), forced-tail | shared | no | reference |
| `qwen3.8-gqa` | Qwen3.8-Flash-Next QSA, GQA-substrate shape | #826 | 24/2 | 256/256 | 4 (micro-block) | 512 entries (2048 tok), forced-tail | per KV-head group (G=H_kv) | no | device |
| `minimax` | MiniMax-M3 MSA | — | 64/4 | 128/128 | 128 (block) | 16 blocks (2048 tok) | per KV-head group | no | device |
| `glm5.2` | GLM-5/5.1/5.2 DSA | #827 | 64/1 (MQA latent, K≡V) | 576/512 (512 latent + 64 RoPE) | 1 (token) | 2048 | shared | no | device |
| `glm5.3-flash` | GLM-5.3-Flash DSA layers (NoPE MLA) | #827 | 64/1 (MQA latent, K≡V) | 512/512 (rope-free) | 1 (token) | 2048 | shared | no | device |

"backend" is `--backend default`'s automatic choice (`VariantConfig.expect_device_kernel`,
kept in lockstep with `python/cudnn/sparse_attention/fwd/api.py`'s
`check_support` envelopes): `device` means the shape lands in a registered
kernel envelope (SM100 DSA-prefill for the `D_k in {512, 576}`/`H_kv=1`
latent shapes, SM100 GQA-substrate for `G=H_kv`/granularity in `(4, 64,
128)`), `reference` means no kernel serves that exact shape yet, so the
benchmark runs the PyTorch oracle directly.

Indices are causal-realistic: query row `i` selects unique random entries
from its causal prefix (up to the variant's top-k), with `topk_length`
carrying per-row valid counts, so short-prefix rows are ragged exactly as in
real prefill. Index generation is row-chunked (no `S x S` buffer). The
`qwen3.8*` variants additionally force each row's own (possibly incomplete)
trailing block into the selection, matching QSA's "always attend the
current block" semantics.

Reported TFLOPS use the exact 2-matmul count (QK^T + PV) computed from the
generated valid lengths:

```
FLOPs = 2 * sum(topk_length) * granularity * heads_per_group * (d_k + d_v)
```

## Known gaps (documented, not benchmarked)

* **`csa-dsv4-*` window branch (#825).** Full DeepSeek-V4 CSA unions the
  selected compressed entries with a last-128-raw-token sliding window and
  the per-head sink logits in *one* softmax denominator (the deferred
  `ExtraKV` argument). `sparse_attention_forward_wrapper`'s frozen signature
  has no such argument yet, so these variants benchmark the compressed-entry
  slice only -- not a full CSA union softmax.
* **Indexer / top-k cost** (issues #829-#831) is out of scope for this
  harness entirely; it times core attention only, given already-selected
  indices.
* **qwen3.8 shared-index shape (#826)** has no registered device kernel
  (`group_scope=1` != `G=H_kv`), so `qwen3.8` reports reference-oracle
  numbers; `qwen3.8-gqa` reports device numbers for the per-KV-group-index
  shape the registered kernel actually serves.

## How to run

```bash
# Everything (auto backend: device kernel where registered, else the PyTorch oracle):
python benchmark_sparse_attention_forward.py \
    --variant dsv4,csa-dsv4-1024,csa-dsv4-512,qwen3.8,qwen3.8-gqa,minimax,glm5.2,glm5.3-flash \
    --seqlens 4096,8192

# Reference-only variants need --q-chunk to bound oracle memory at longer seqlens
# (the oracle materializes a dense (rows, topk*granularity) gather, unlike the
# device kernels' streaming mainloop):
python benchmark_sparse_attention_forward.py --variant csa-dsv4-1024,qwen3.8 \
    --seqlens 8192,16384 --q-chunk 512

# Force the reference oracle even for device-kernel-served variants:
python benchmark_sparse_attention_forward.py --variant dsv4 --seqlens 4096 --backend reference --q-chunk 1024
```

`profile` mode wraps one warmed call in `cudaProfilerStart/Stop` + NVTX for
nsys/ncu:

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -o safwd \
  python benchmark_sparse_attention_forward.py profile --variant dsv4 --seqlens 8192
```

Options: `--seqlens` (comma-separated, `seqlen_kv = seqlen_q`), `--variant`
(comma-separated subset), `--dtype bfloat16|float16`, `--backend
default|reference`, `--q-chunk N` (split each call over query-row chunks —
exact under this API because indices are storage-native global ids),
`--warmup/--repeat`, `--csv out.csv`.

## Results

`results/device_kernel_sweep.csv` — `dsv4`, `glm5.2`, `glm5.3-flash`,
`qwen3.8-gqa`, `minimax` (all device-kernel-served) at
`seqlens = 4096,8192,16384,32768,65536`, bf16, `--warmup 2 --repeat 5`.

`results/reference_only_sweep.csv` — `csa-dsv4-1024`, `csa-dsv4-512`,
`qwen3.8` (reference-oracle-only shapes) at `seqlens = 4096,8192,16384`,
bf16, `--q-chunk 512 --warmup 1 --repeat 3` (small `--repeat`/`--q-chunk`
because the oracle's dense gather is far slower than a streaming device
kernel at these entry counts; not a source of device-kernel numbers).

Both sweeps stop at 65536/16384 rather than the issues' full 4K-1M range:
at `H_q=64, D=512` bf16, `Q` alone is ~4.3 GiB at 65536 and grows linearly,
and the reference oracle's dense per-row gather is the binding constraint
for the non-kernel shapes well before 1M tokens on a single GPU. Extending
past this range is a follow-up, not a fabricated number.

**`minimax`'s row above is the scalar-kernel number** (`gqa_prefill_bf16_sm100`,
~1-1.4% MFU) -- see `flops_fwd()`'s docstring in
`benchmark_sparse_attention_forward.py` for the exact reconciliation, and
note there is no `1.32` TFLOPS figure anywhere in this file or
`results/device_kernel_sweep.csv`; any report claiming that number for this
kernel did not come from this benchmark and should be treated as
unverified. A round-5 tcgen05 (tensor-core) mainloop for exactly this cell
(`gqa_prefill_bf16_tcgen05_sm100`, `python/cudnn/sparse_attention/fwd/sm100_gqa/`)
now exists and is wired as the default route for a genuinely tile-uniform
selection, but **its `cute.compile()` did not finish within round 5's
session budget** (see that module's docstring for the bisection that
narrowed the pathological compile time to its `mma_ss` call) -- so no
tcgen05 row is added to `results/device_kernel_sweep.csv` this round. Per
this file's own standing rule (see the note directly above), a number is
only ever added here once it has actually been observed from a completed
run of this benchmark or an equivalent direct call to the wrapper -- not
before.
