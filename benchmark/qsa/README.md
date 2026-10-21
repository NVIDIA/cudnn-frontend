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
preallocated forward, backward prep/main/finish, and combined backward kernel
timings. These use eager launches with prebuilt IDs and buffers. Backward
accumulators are reset before each main-kernel sample, outside its timed region.
Component sums need not equal full-call timings; the full call remains primary.

Memory fields give live bytes before each pass, peak allocated/reserved bytes,
and peak extra allocation. The dense bitmap and the current native int32 ID
buffer alone need 4 GiB and 16 GiB at 128K. Reference saved state remains live
during timing. Output and log files are kept per case and implementation.
An illegal device access cannot invalidate another process's result. A failed
implementation has no speedup ratio, and the command exits nonzero after the
remaining cases finish. `--timeout` sets the limit per process.

The original Miles forward failed at 128K on B300 and B200.
Its bitmap address uses an int32 row index multiplied by the row stride. At
128K, the largest bitmap offset is 2^32−1, outside signed int32. A separate B300
control with identical input hashes changed only the three bitmap row-offset
multiplications to int64: the original failed and the control passed all three
forward checks. The harness keeps the original baseline and its failure; the
control is not used for speedup ratios.
