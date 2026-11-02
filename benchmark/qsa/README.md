# QSA comparison

Run from the repository root with a CUDA PyTorch environment, Triton, and the
CuTeDSL version required by the QSA API. Use an installed cuDNN Frontend package
or pass `--frontend-source` to test this checkout over an installed package.
NumPy is used for input hashing.

See [results.md](results.md) for measured B200 and B300 results.

```bash
python benchmark/qsa/benchmark_qsa.py \
  --miles-source /path/to/miles/kernels/attention/qsa/block_sparse_attention.py \
  --frontend-source "$PWD" --output qsa-results.json
```

The external Miles file is loaded unchanged. Its autograd context supplies the
original FP32 output and LSE. No Miles source is shipped here. For the traced
Miles baseline, use commit `0cc65167f991ab1f0d228f6c9b648d89bc60a203`; the file's
SHA256 is `18542596944a5c3f4a231ce53f9d660f65a23246072ff060223a23e2d1cfbae1`.

The suite uses BF16 packed THD, D256, block size 4, 12/1 and 24/2 query/KV heads,
and lengths from 129 through 131072. Packed cases have unaligned sequence ends.
Small cases include empty rows. The synthetic selector draws uniform scores,
keeps the best 512 complete causal blocks, and adds the partial causal tail.
Equal random scores use the block index as a tie breaker. Reference gathers use
ascending block order so separate processes reproduce saved state exactly.
It is not a captured model trace. `--seed` defaults to 17. Each implementation
gets a fresh process; input and saved-state hashes must match before a ratio is
reported. GPU random streams can differ across GPU models; compare hashes when
comparing machines.

The independent chunked FP32 reference checks out, out_fp32, LSE, dQ, dK and dV
before timing. Both backward calls use the same reference FP32 output and
natural-log LSE. Checks require matching empty-row markers, elementwise
`atol=rtol=0.04`, and relative L2 error at most 0.025. `--directions forward`
limits validation and timing to the three forward outputs while backward is
under development. `--max-tokens 4096` runs a small suite.

Primary timings are medians of synchronized wall time for full public calls.
CUDA event medians and all raw samples are also saved. Calls include metadata,
allocations, delta computation, casts and helper kernels. Input generation,
the reference, hashing, warmup and compilation are outside the timed region.
This measures repeated warm calls; it does not flush GPU caches or use graph
replay. The JSON records first-call compile time, source hashes, GPU, driver,
CUDA, PyTorch, Triton and DSL versions.

Use `--components --min-tokens 4096 --max-tokens 32768` to add native compaction,
preallocated gather-only forward, backward prep/main/finish, and combined backward
kernel timings. Forward components do not describe the public dense route.
These use eager launches with prebuilt IDs and buffers. Backward
accumulators are reset before each main-kernel sample, outside its timed region.
The isolated backward main excludes the public nonfinite scan and fallback
dispatch. Component sums need not equal full-call timings; the full call remains primary.

Memory fields give live bytes before each pass, peak allocated/reserved bytes,
and peak extra allocation. At 128K, the bitmap needs 4 GiB and forward int32 IDs
need 16 GiB. Paired Blackwell 12:1 backward uses 8 GiB for IDs. Reference saved
state remains live during timing. Output and log files are kept per case and implementation.
An illegal device access cannot invalidate another process's result. A failed
implementation has no speedup ratio, and the command exits nonzero after the
remaining cases finish. `--timeout` sets the limit per process.

The original Miles forward failed at 128K on B300 and B200.
Its bitmap address uses an int32 row index multiplied by the row stride. At
128K, the largest bitmap offset is 2^32−1, outside signed int32. A separate B300
control with identical input hashes changed only the three bitmap row-offset
multiplications to int64: the original failed and the control passed all three
forward checks. The default command keeps the original baseline and its failure.
The labeled 128K table in [results.md](results.md) uses the Int64-only control;
those ratios are not unchanged-Miles results.

## Unchanged Relax reference

Use the public `qsa_mixed_ops` directory at Relax commit
[`d16f2e3`](https://github.com/xiaoliang0601/Relax/tree/d16f2e3ec45be059db6377295090b1d49257892c/relax/backends/megatron/megatron_patch/qsa_mixed_ops).
Its unchanged `grouped.py` has SHA256
`2168d82a28514243fe015d0e6e45ad6487c06c9350936814f2937dee9d583f36`.
The same grouped source is present at `77f717d`.

```bash
python benchmark/qsa/benchmark_relax.py \
  --relax-source /path/to/Relax/relax/backends/megatron/megatron_patch/qsa_mixed_ops \
  --frontend-source "$PWD" --indexed --lengths 131072 --output relax-128k.json
```

Omit `--indexed` to compare the bitmap API. Use `--lengths 4099,8193` for unaligned
packed boundaries. `--frontend-source` verifies the actual imported checkout;
it raises if an editable install still points elsewhere. The harness loads both
external files unchanged, records input/source hashes, creates BF16 4Q/D128
indexer scores, selects 512 complete causal blocks, and appends the incomplete
tail. It does not substitute the Miles implementation for the Relax baseline.

For the public bounded-workspace backward, add `--causal-topk`. This declares
the stricter [canonical selection contract](../../docs/operations/sparse_attention/qsa_indexed.md#complete-blocks-and-causal-tails)
and passes packed `cu_seqlens` and the maximum document length to the public
entry point. Sparse 12Q/1KV requires the CUDA 13 CuTeDSL compiler. Add
`--compare-general` to measure the default indexed backward on the same saved
state. For example, use `--indexed --causal-topk --compare-general --lengths
43691,43691,43690` for an unaligned packed 128K comparison.

All six outputs must pass cross-comparison and an independent chunked FP32
oracle before any timings are emitted. Relax backward consumes the saved state from its own `grouped._Grouped.forward`
call; native backward consumes `out_fp32` and `lse` from its own QSA forward call
(`qsa_indexed_forward` with `--indexed`). Both timing orders are retained, and both implementations are checked
again after timing. Forward, backward, and combined forward-plus-backward are
timed directly; combined times are not sums of separately measured stages.
Reported times cover warm full attention calls, including
allocations, metadata, backward preparation and final casts. They exclude
compilation, selector/top-k, token expansion and framework/TP overhead. Peak
extra allocated bytes are measured over the same live fixture state. Indexed
runs use a direct token-gather FP32 oracle and do not allocate a quadratic
bitmap. Metadata sizes are reported separately where applicable; peak extra
allocation includes outputs and workspace, not just sparse records. The JSON
also records hostname, scheduler allocation, GPU memory/SM count, and a full
`nvidia-smi -q` snapshot so production GPU provenance and operating limits can
be checked. If that optional CLI is unavailable or fails, the report retains
`nvidia_smi=null` and `nvidia_smi_error`; CUDA validation and timing can still
proceed. Such a report needs separate hardware verification before claiming
production GPU qualification. These figures are synthetic attention measurements.

For selector and route-expansion attribution, run Relax's own unchanged
`examples/models/qwen3.8-flash-next/benchmark_qsa.py` with both `--selector eager`
and `--selector compile`. That script models a selector TP row shard and forces
grouped attention over the full pack. It excludes full projection/pooling/RoPE,
collectives, the mixed Flash/Flex dispatcher, and model training. The BF16-input
DSA wide-top-k draft is not a direct replacement for these FP32 selector scores.
