# FP8 Projection for Decode Rows (SM100-class)

**Experimental.** `cudnn.Fp8ProjectionDecode` (also `cudnn.gemm.Fp8ProjectionDecode`) is a dense projection of FP8 E4M3
activations by an FP8 E4M3 weight at decode-time row counts, as ONE CuTe DSL launch:

```text
out[t, n] = bf16(scale * sum_k activation[t, k] * weight[n, k])

weight      [OUT_FEATURES, IN_FEATURES]  torch.float8_e4m3fn, row-major
activation  [rows, IN_FEATURES]          torch.float8_e4m3fn, rows 1 .. 32
scale       one element                  torch.float32 (the caller's product of the activation scale and the weight scale)
out         [rows, OUT_FEATURES]         torch.bfloat16, caller-owned
```

The sum is accumulated in fp32 on the tensor core (`tcgen05` FP8 MMA), scaled once and cast once to bf16 (round to
nearest even; no saturation -- a value past the bf16 range is an infinity of its sign). The K axis is cut into `split_k`
slices, one CTA of a cluster each; a slice is one accumulation chain over its 128-element K tiles in ascending K, and the
slice sums are added in slice order through DSMEM (`st.async`) on the cluster's first CTA. Under dependent launch the
weight stream is issued BEFORE the kernel waits for its predecessor, so the weight bytes are already moving while the
kernel that produces `activation` still runs.

It is a direct-call step entry, the same kind as the fused GDN decode step and the shared-expert decode pair: an inference
step with no backward, called by an engine that owns the tensors. It is not a graph node and it does not touch the GEMM
engine.

## Usage

```python
import torch
import cudnn
from cudnn.gemm import ProjectionGeometry

dev = torch.device("cuda", 0)
entry = cudnn.Fp8ProjectionDecode(dev, ProjectionGeometry(4608, 8192))    # split_k = 4 by the geometry rule
entry.compile(rows)                                                       # plan time; entry.compile() compiles rows 1 .. 32
out = entry(weight, activation, scale, out)                               # launches on torch's current stream of dev

# another split is another byte class and another entry:
entry_split2 = cudnn.Fp8ProjectionDecode(dev, ProjectionGeometry(4608, 8192), split_k=2)
```

Nothing is captured at construction: the four tensors are launch arguments, so one entry serves every layer of its
geometry. Every launch re-runs the binding checks (type, dtype, device, shape, contiguity, alignment) before the compiled
kernel is looked up; the host path makes views only -- no conversion, no copy, no device allocation, no device read.
`out` must not overlap `weight`, `activation` or `scale` (not checked).

**The row count is a compile key.** The kernel is specialised on `rows` (its output bounds and its token tile are
trace-time constants), so an entry holds one rendering per row count: at most 32, each a fraction of a second to compile.
This is a stated, bounded exception to the rule that a runtime extent is not a compile key; `GdnDecodeFused` keys on its
row count the same way. `compile(rows)` is the plan-time site -- after it a launch at that row count is a cache hit that
compiles nothing. A launch at a row count that was never compiled compiles it on first use; a serving path should not
rely on that. `entry.cfg_for(rows)` returns the rendering of a row count and `entry.key(rows)` its compile key.

## JAX

The JAX entry requires CuTeDSL >= 4.9.0 for experimental-launcher support in
`cutlass.jax`. Earlier versions are declined with a version-specific error.

```python
import jax
from cudnn.ops import fp8_projection_decode_forward_jax

with jax.default_device(device):
    (out,) = jax.jit(fp8_projection_decode_forward_jax)(weight, activation, alpha)
```

The JAX entry accepts `float8_e4m3fn` weight `[N,K]` and activation `[T,K]`,
plus FP32 `alpha` with shape `()` or `[1]`. It returns a fresh BF16 `[T,N]`
array owned by XLA. Geometry, row-count and split restrictions match the native
entry. Pass `split_k` as a static keyword, for example by closing over it in
the jitted function. The default split depends only on geometry.

Original operands reach the native kernel without a framework conversion or
input reshape. Execution uses XLA's CUDA stream with dependent launch disabled.
Eager calls select the operands' GPU; tracing uses `jax.default_device` for
architecture-specific planning. Unrelated visible GPUs and JAX process count
do not restrict local calls. Automatic differentiation and `vmap` are
unsupported. JAX coverage is in
`test/python/gemm/jax/test_fp8_projection_decode_jax.py`.
It uses the standard `cudnn.jax.call` / `cutlass.jax` bridge. GPU qualification
of this CuTeDSL 4.9 bridge path remains pending.

## Numerics contract

The output bytes are a function of `(geometry, split_k)` on one runtime, and of nothing else:

- `split_k` changes the fp32 association (the number of slice sums and adds), so it is a NUMERICS attribute. It is fixed
  at construction -- given, or `split_k_for(geometry)`, a function of the geometry alone (4 at
  `ProjectionGeometry(4608, 8192)`, 2 at `ProjectionGeometry(8192, 2048)`) -- and never depends on the row count or on
  the part. The entry never switches the split by itself.
- Everything else is a performance form (`cudnn.gemm.ProjectionForm`): the weight rows per CTA (`MMA_M`), the depth of
  the operand ring (`STAGES`), the tensor-memory columns a CTA allocates (`TMEM_COLS`), the token tile (`TOKEN_TILE`),
  whether the weight stream runs ahead of the dependent-launch wait (`PREWAIT_WEIGHTS`) and when the epilogue releases
  the tensor memory (`HOLD_TMEM`). No field of a form enters a sum. Without `form=` the entry takes a measured pick per
  row count on the part's own readings; `form=ProjectionForm(...)` overrides the pick as a whole for every row count and
  cannot name a split.
- A row's bytes are the same alone and among up to 32 rows, at any position, with any neighbours (batch invariance), and
  identical across repeated launches. See [determinism](/operations/determinism).

Served byte classes -- every other pair is declined by name:

| geometry | `split_k` | note |
| --- | --- | --- |
| `ProjectionGeometry(4608, 8192)` | 4 | the default of the geometry |
| `ProjectionGeometry(4608, 8192)` | 2 | explicit `split_k=2`; another byte class than split 4 |
| `ProjectionGeometry(8192, 2048)` | 2 | the default of the geometry |

On the validated cc 10.3 part, split 2 measured faster than split 4 at 17 .. 32 rows of
`ProjectionGeometry(4608, 8192)`. The entry still never changes the split with the row count: a caller that wants
split 2 in that band builds a second entry and owns the statement that the two bands are different byte classes; one
entry is batch-invariant over all of 1 .. 32 rows.

Against an fp64 reference of the E4M3 operands every finite element satisfies
`|out - ref| <= half a bf16 ulp + |scale| * sum_k |activation| |weight| * (IN_FEATURES + split_k + 1) * 2^-23`; operands
whose partial sums are exact in fp32 (small integers, subnormal codes, the largest code with equal or alternating signs)
give the host's bytes. A NaN code in one activation row makes that output row NaN and leaves the other rows' bytes
untouched; a NaN code in a weight row makes that output column NaN.

## The measured pick

The pick reads the row count and the part (SM count, shared memory per SM, per-CTA opt-in) and returns a form. The token
tile is the smallest of 8 / 16 / 32 that holds the rows, a CTA allocates the 32 tensor-memory columns that tile needs,
and the ring depth comes from a table measured on a cc 10.3 part (152 SMs); it is lowered until the grid fits one wave
on the part. On the cc 10.3 readings:

| class | rows | weight tile, stages, token tile | shared memory per CTA | CTAs | CTAs per SM |
| --- | --- | --- | ---: | ---: | ---: |
| `(4608, 8192)`, split 4 | 1 .. 4 | 64, 11, 8 | 107,904 B | 288 | 2 |
| `(4608, 8192)`, split 4 | 5 .. 8 | 64, 7, 8 | 71,040 B | 288 | 3 |
| `(4608, 8192)`, split 4 | 9 .. 16 | 64, 7, 16 | 84,352 B | 288 | 2 |
| `(4608, 8192)`, split 4 | 17 .. 32 | 64, 7, 32 | 110,976 B | 288 | 2 |
| `(4608, 8192)`, split 2 | 1 .. 8 / 9 .. 16 / 17 .. 32 | 64, 8, 8 / 16 / 32 | 76,160 / 86,400 / 106,880 B | 144 | 3 / 2 / 2 |
| `(8192, 2048)`, split 2 | 1 .. 8 / 9 .. 16 / 17 .. 32 | 64, 8, 8 / 16 / 32 | 76,160 / 86,400 / 106,880 B | 256 | 3 / 2 / 2 |

"CTAs per SM" is what shared memory lets the launcher place; with 32 columns each, every placed CTA also gets its
tensor memory (an SM has 512 columns). The entry validates that ledger: a form whose placed CTAs would ask for more than
512 columns is declined by name, because a CTA without columns would park inside its allocation. That the same forms
stay one wave on 148 and 212 SMs is a slot-count model (SMs times CTAs per SM), pinned by a host test; it is not a
placement measurement. Other ring depths, the 128-row weight tile, 64 / 128 columns, a wider token tile and the knobs
below are reachable through `form=` where the ledger admits them, and every one of them writes the default's bytes.

`HOLD_TMEM` (default off): off, the epilogue releases the tensor memory right after its accumulator read; on, it holds
the columns until every register read from them is consumed, which moves the deallocation behind the output store.
Measured on a cc 10.3 part the held form is the slower one, so it stays an opt-in.

## Dependent launch

`pdl=True` (default off) puts the programmatic-stream-serialization attribute on every launch. The kernel then waits for
its predecessor right before it reads `activation`; the barrier inits, the tensor-memory allocation and the first weight
K tiles run ahead of that wait. **Caller obligation:** no kernel that can still be in flight when the launch starts
writes `weight` or `scale` (static weights and a static scale satisfy it).

`pdl_release` (default: follows `pdl`, as `GdnDecodeFused` does) releases the successor after the kernel's last weight
issue. That creates an obligation for the consumer: a kernel launched behind this one WITH the attribute may start while
this one runs, must issue its own wait before it reads `out`, and -- if it allocates tensor memory above that wait --
must fit beside this kernel's columns (at most 96 per SM at the default forms). A consumer launched without the
attribute is unaffected.

## Launch facts

The entry states what a row count launches without launching it: three read-only answers, with `cfg_for`'s argument.

| fact | answer |
| --- | --- |
| `kernel_symbols_for(rows)` | the symbol of the kernel compiled for `rows`, as a one-element tuple; empty until that row count is compiled |
| `form_for(rows)` | the form as plain data: every field of the rendering `cfg_for` returns under its own name -- `GEOM` (a nested `dict`), `SPLIT_K`, `ROWS`, the resolved performance form (`MMA_M`, `STAGES`, `TOKEN_TILE`, `TMEM_COLS`, `PREWAIT_WEIGHTS`, `HOLD_TMEM`), the launch knobs `PDL` / `PDL_RELEASE` and the part's readings. A fresh `dict` on every call |
| `launch_shape_for(rows)` | `(grid, block, shared_memory)`: grid `(OUT_FEATURES / MMA_M, split_k, 1)`, block `(256, 1, 1)` and the shared memory the launch reserves per thread block (the "shared memory per CTA" column of the pick table above) -- `((72, 4, 1), (256, 1, 1), 107904)` for one row of `ProjectionGeometry(4608, 8192)` at split 4 under the pick above. A profiler's kernel record of the launch reads the same three values |

The symbol names the weight's shape and nothing of the split, the form or the row count: every rendering of one
geometry shares it. Compare the form, not the name. A row count outside 1 .. 32 is a typed decline (`ValueError` for
one that is not a positive `int`, `NotImplementedError` past 32), and none of the three compiles, launches or touches
device memory.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x (validated on cc 10.3 and cc 10.7; cc 10.0 is admitted and unverified); every other capability is declined by name |
| geometry and split | the three byte classes above; any other geometry or split is declined by name |
| rows | 1 .. 32 per call (a compile key; `compile()` covers all 32); 33 and up declined by name, never served by another kernel |
| dtypes | FP8 E4M3 weight and activation; one fp32 scale (per tensor); bf16 out; fp32 accumulation |
| layouts | contiguous operands, 32-byte aligned storage (4-byte for `scale`), every operand on the entry's device; `out` exactly `[rows, OUT_FEATURES]` |
| determinism | bitwise replay and batch invariance at every row count 1 .. 32 (the [status page](/operations/determinism)) |
| backward | none: an inference step |
| DSL | nvidia-cutlass-dsl >= 4.8.0 that provides the `cutlass.cute.experimental` layer the kernel is written on; the layer is probed by name before the kernel module is imported, and a DSL without it is declined by name. On cc 10.7 the DSL must compile with CUDA 13.4 or newer (its cu13 flavor); its CUDA 12 flavor has no target for that part and is declined by name before any compile |

Typed declines, all before any DSL import: `TypeError` for a non-tensor or a dtype, `ValueError` for a shape, an
alignment, a device or a knob combination, `NotImplementedError` for a row count, a geometry, a split, a memory layout,
a part or a DSL the kernel does not serve. No copy is ever made to serve a layout.

The kernel is derived from FlashInfer's cluster split-K dense GEMM (Apache-2.0); the kernel file's header and
`THIRD_PARTY_LICENSES.txt` carry the notice and the list of changes.

Tests: `test/python/gemm/frost/projection_decode/test_fp8_projection_decode.py` (host declines, config and pick pins,
the launch facts against the configuration arithmetic and against the profiler's record of a launch, the device cells)
and `test_fp8_projection_decode_determinism.py` (replay, batch invariance, the pick's byte neutrality, goldens).
Source: `python/cudnn/gemm/frost/projection_decode/`.
