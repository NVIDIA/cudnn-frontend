# CSA Fused Compressor

**This is an experimental API and subject to change.**

## Overview

The CSA module hosts CuTe-DSL kernels for the CSA/HCA experimental attention variants
(the components that are not shared with the [DSA module](dsa.md)). Its first operation
is the **fused Compressor**: one forward and one backward kernel for the `Compressor`
gated-softmax pooling region (THD packed layout) used by CSA/HCA in Megatron-LM.

The kernels were ported from Megatron-LM at the maintainers' request
([Megatron-LM PR #5984](https://github.com/NVIDIA/Megatron-LM/pull/5984); measurements
and numerics in
[Megatron-LM issue #5968](https://github.com/NVIDIA/Megatron-LM/issues/5968)). The eager
region they replace decomposes into ~39 forward and ~51 backward kernel launches per
call (at `compress_ratio = 4`) and materializes `(total_comp, 2*ratio, 1, head_dim)`
window intermediates; the fused path is 1 + 1 kernels (plus three grad-buffer zero-fills
in backward).

### Semantics

For each THD segment `s` (`cu_seqlens[s]..cu_seqlens[s+1]`) and each output block `b` of
`ratio` tokens, with the overlapping window form (`coff == 2`, window size `2 * ratio`):

- `k in [0, ratio)`: previous block's token, first-half projection column, APE row `k`
  — invalid for the segment's first block (score `-inf`, kv `0`);
- `k in [ratio, 2*ratio)`: own block's token, second-half projection column, APE row
  `k - ratio`.

```text
out[b, j] = sum_k kv[w(b,k), c(k,j)] * softmax_k(score[w(b,k), c(k,j)] + ape[k % ratio, c(k,j)])
```

computed in fp32 with a single final bf16 rounding. Per-segment tail tokens
(`seqlen % ratio`) are dropped, as in the eager code. Output rows beyond
`cu_seqlens_comp[-1]` (a static CUDA-graph capacity) are computed with first-in-segment
semantics from token 0, exactly like the eager gather; the backward ignores incoming
gradients on such padding rows.

### Numerics

All arithmetic is fp32 with one final bf16 rounding; `mul.rn.f32` / `fma.rn.f32` are
pinned in PTX so results do not depend on compiler FMA contraction. Against an
fp32-intermediate eager reference (same op order, fp32 throughout), `dKV`/`dScore` are
**bit-identical** and the forward matches within one bf16 rounding step on a tiny
fraction of elements. Forward, `dKV` and `dScore` are bitwise run-to-run deterministic.
`dAPE` is reduced with one fp32 atomic per `(k, dim)` per CTA and is **not** bitwise
run-to-run deterministic; the backward APIs raise under
`torch.use_deterministic_algorithms(True)`.

### Support surface (`check_support`)

- Compute capability **10.0** (the only validated architecture so far; the kernels use
  no arch-specific features, wider enablement is possible after validation)
- `ratio == 4`, `coff == 2` (the production CSA/HCA configuration; the kernels are
  generic over `(ratio, head_dim, coff in {1, 2})` and the gate can be lifted once
  validated)
- BF16 `kv` / `score` / `out`, FP32 `ape`, int32 `cu_seqlens` / `cu_seqlens_comp`
- int32 flat offsets: `total_tokens * coff * head_dim < 2**31`
- contiguous tensors on one CUDA device, with 16-byte-aligned base pointers (4-byte
  for the int32 cu_seqlens) — contiguity does not imply base alignment for
  storage-offset views, so this is checked per call

## Installation

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

## API Usage

### High-level wrappers

```python
from cudnn import CSA

# forward: (total_tokens, coff*head_dim) BF16 kv/score, (ratio, coff*head_dim) FP32 ape,
# (B+1,) int32 cu_seqlens / cu_seqlens_comp
result = CSA.csa_compressor_forward_wrapper(
    kv, score, ape, cu_seqlens, cu_seqlens_comp,
    ratio=4, head_dim=128, coff=2,
    total_comp=None,  # defaults to cu_seqlens_comp[-1] (synchronizes); pass a static
                      # capacity explicitly to stay CUDA-graph capture-safe
    stream=None,
)
out = result["out"]  # (total_comp, head_dim) BF16

grads = CSA.csa_compressor_backward_wrapper(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out,
    ratio=4, head_dim=128, coff=2, stream=None,
)
grad_kv, grad_score, grad_ape = grads  # BF16, BF16, FP32
```

The wrappers cache compiled API instances; the underlying JIT is shared per
`(ratio, head_dim, coff, device)`, so runtime shape changes never recompile.

### Class API

```python
from cudnn import CSACompressorForward, CSACompressorBackward

op = CSACompressorForward(
    sample_kv, sample_score, sample_ape, sample_cu_seqlens, sample_cu_seqlens_comp,
    sample_out, ratio=4, coff=2,
)
op.check_support()
op.compile()
op.execute(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, current_stream=None)
```

`CSACompressorBackward.execute` additionally takes `grad_out` and **zero-initialized**
`grad_kv` / `grad_score` / `grad_ape` buffers (the kernel's stores are disjoint and
atomic-free for `grad_kv`/`grad_score`; unconsumed positions keep their exact zeros,
matching autograd).

### CUDA graphs

The launch path is capture-compatible once the kernels for a `(ratio, head_dim, coff)`
configuration are compiled: run one warmup call (or `compile()`) per configuration
before capturing, and pass `total_comp` explicitly. A call that would JIT under capture
raises a `RuntimeError` instead of corrupting the capture.

### Environment variables

- `CUDNNFE_CSA_COMPRESSOR_FAST_LAUNCH=0` — disable the cached-launch host optimization
  (a per-config snapshot of the CuTe-DSL launch state, replayed with in-place argument
  mutation; it removes tens of microseconds of per-call host overhead for these
  microsecond-scale kernels). The snapshot construction introspects
  private-but-stable DSL internals; on any structural mismatch (e.g. a future
  `nvidia-cutlass-dsl` upgrade) it falls back to the regular launch path automatically.

## Performance

Measured on 1x B200 (CC 10.0, driver 590.48.01); BF16 `kv`/`score`, FP32 `ape`; `ratio = 4`, `coff = 2`;
THD packs of 8192-token sequences; eager baseline = the exact replaced region of
Megatron-LM `Compressor._forward_thd` on identical inputs.

*Isolated GPU kernel time* (nsys, sum of kernel durations per iteration, 50 iterations
after 20 warmup; no launch/host overhead; backward includes its grad-buffer zero-fills):

| THD pack | head_dim | eager fwd | fused fwd | fwd | eager bwd | fused bwd | bwd |
|---|---|---|---|---|---|---|---|
| 1 x 8192 | 128 | 114.0 us | 6.0 us | **19.0x** | 184.8 us | 15.3 us | **12.1x** |
| 3 x 8192 | 128 | 226.4 us | 12.5 us | **18.1x** | 351.2 us | 29.5 us | **11.9x** |
| 1 x 8192 | 512 | 259.5 us | 15.9 us | **16.3x** | 421.7 us | 33.6 us | **12.6x** |
| 3 x 8192 | 512 | 663.1 us | 42.0 us | **15.8x** | 1154.7 us | 90.8 us | **12.7x** |

*End-to-end wall clock of the same region* (CUDA events, median of 100; includes launch
overhead; eager backward goes through torch autograd, fused backward is the explicit
backward wrapper call — see the measurement-basis note above; not comparable to the
kernel-time numbers):

| THD pack | head_dim | eager fwd | fused fwd | fwd | eager bwd | fused bwd | bwd |
|---|---|---|---|---|---|---|---|
| 1 x 8192 | 128 | 339.7 us | 38.9 us | **8.7x** | 533.2 us | 55.3 us | **9.6x** |
| 3 x 8192 | 128 | 372.4 us | 39.9 us | **9.3x** | 661.9 us | 62.9 us | **10.5x** |
| 1 x 8192 | 512 | 405.7 us | 41.6 us | **9.8x** | 637.2 us | 63.9 us | **10.0x** |
| 3 x 8192 | 512 | 821.6 us | 68.9 us | **11.9x** | 1475.3 us | 110.9 us | **13.3x** |

Environment: driver 590.48.01, PyTorch 2.13.0 (CUDA 13.3), `nvidia-cutlass-dsl` 4.6.1.
Measurement basis: identical inputs over exactly the replaced region for both
implementations; eager backward = torch autograd of the recorded eager graph; fused
backward = the backward wrapper (kernel + three grad zero-fills + host validation, no
autograd engine).

## Testing

```bash
pytest test/python/fe_api/csa/test_CSA_compressor.py
```

The tests validate numerics against an fp32-intermediate eager reference (bitwise
`dKV`/`dScore`), the upstream eager numerics, and an fp64 oracle, plus ragged packs,
static-capacity padding, run-to-run determinism, CUDA-graph capture/replay, and
`check_support` boundaries.
