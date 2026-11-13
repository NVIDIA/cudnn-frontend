# SwiGLU FP4 Gate / Up Product for Decode Rows (SM100-class)

**Experimental.** `cudnn.SwigluFp4Decode` (also `cudnn.gemm.SwigluFp4Decode`) is the gate / up product of a block-scaled FP4
(NVFP4) MLP at decode-time row counts, with `SiLU(gate) * up` and the FP4 re-quantization of the result fused into the
epilogue, as ONE CuTe DSL launch:

```text
gate[t, c] = bf16(alpha * sum_k activation[t, k] * weight[row_gate(c), k])        up[t, c] likewise with row_up(c)
v[t, c]    = bf16(silu(gate[t, c]) * up[t, c])
per block of 16 channels of a row:  byte = E4M3(gs * (max|v| * fp32(1/6)));  encode = 0 for a zero byte, else rcp(value(byte) * rcp(gs))
out[t, c]  = E2M1(v[t, c] * encode)          out_scale[t, block] = byte
```

| operand | dtype | shape / size | notes |
| --- | --- | --- | --- |
| `weight` | `torch.uint8` | `[2 * INTERMEDIATE, HIDDEN / 2]` | two FP4 (E2M1) codes per byte, the even column in the low nibble; rows in the interleaved order below; contiguous, 32-byte aligned |
| `weight_scale` | `torch.float8_e4m3fn` or `torch.uint8` | exactly `2 * INTERMEDIATE * HIDDEN / 16` bytes, any shape | one E4M3 byte per (row, 16 columns) in the 128 x 4 scale layout of the interleaved rows; 16-byte aligned |
| `activation` | `torch.uint8` | `[rows, HIDDEN / 2]`, rows 1 .. 64 | the row count is read from this shape; 32-byte aligned |
| `activation_scale` | as `weight_scale` | at least `128 * HIDDEN / 16` bytes | the whole 128-row scale band is loaded, so the storage must exist; the bytes of rows past `rows` never reach an output |
| `alpha` | `torch.float32` | one element | the caller's scalar of the product (the reciprocal of the two operands' global scales); any finite value |
| `out_global_scale` (`gs`) | `torch.float32` | one element | the global scale of the FP4 output: a positive normal value in `2**-100 .. 2**100` |
| `out` | `torch.uint8` | exactly `[rows, INTERMEDIATE / 2]` | caller-owned; every byte of the rows is written; 16-byte aligned |
| `out_scale` | as `weight_scale` | at least `128 * INTERMEDIATE / 16` bytes | caller-owned; the bytes of the live rows are written at their positions in the 128 x 4 scale layout, every other byte is left untouched |

The product is accumulated in fp32 on the tensor core (`tcgen05` block-scaled MMA) with the token rows on the narrow tile
of the instruction -- 8, 16, 32 or 64 tokens -- so 1 .. 64 rows do not pay a 128- or 256-row tile. The product itself is
never stored. Under dependent launch the first weight tiles are issued BEFORE the kernel waits for its predecessor, so the
weight bytes are already moving while the kernel that produces `activation` still runs.

It is a direct-call step entry, the same kind as the fused GDN decode step, the shared-expert decode pair and the FP8
projection for decode rows: an inference step with no backward, called by an engine that owns the tensors. It is not a
graph node and it does not touch the GEMM engine.

## Usage

```python
import torch
import cudnn
from cudnn.gemm import SwigluGeometry
from cudnn.gemm.frost.swiglu_decode import interleave_gate_up

dev = torch.device("cuda", 0)
w, ws = interleave_gate_up(weight, weight_scale)                 # load time, once per weight: gate / up rows into tiles of 64 + 64
entry = cudnn.SwigluFp4Decode(dev, SwigluGeometry(5120, 17408))  # pdl=True for the dependent-launch attribute
entry.compile()                                                  # plan time: the four token-tile forms; typed declines before any DSL import
out, out_scale = entry(w, ws, activation, activation_scale, alpha, out_global_scale, out, out_scale)   # rows 1 .. 64, read from activation
```

Nothing is captured at construction: the eight tensors are launch arguments, so one entry serves every layer of its
geometry, and the entry owns no device memory. Every launch re-runs the binding checks (type, dtype, device, shape, byte
count, contiguity, alignment) before the compiled kernel is looked up; the host path makes views only -- no conversion, no
copy, no device allocation, no device read. `alpha` and `out_global_scale` are read on the device. The launch goes to
torch's current stream of the entry's device. `out` or `out_scale` overlapping an input is undefined (not checked).

**The row count is a runtime value.** The kernel is compiled once per token tile, not per row count: `compile()` builds the
four forms, after which a launch at any served row count is a cache hit that compiles nothing; `compile(rows)` builds the
form of one row count. A launch of a form that was never compiled compiles it on first use; a serving path should not rely
on that. `entry.cfg_for(rows)` returns the form of a row count and `entry.key(rows)` its compile key (the form; the row
count is not part of it).

### The weight layout

The kernel reads the weight in tiles of 128 rows: rows `128 j .. 128 j + 63` are gate channels `64 j .. 64 j + 63`, rows
`128 j + 64 .. 128 j + 127` the up channels of the same 64. `interleave_gate_up(weight, weight_scale)` builds that layout
from a weight whose gate rows `0 .. INTERMEDIATE - 1` are followed by its up rows, and permutes the scale bytes the same
way. It is a byte permutation -- nothing is dequantized or requantized -- that returns NEW tensors: one interleaved copy
per weight is the caller's memory (100.3 MB at `SwigluGeometry(5120, 17408)`). It runs at load time, never on the launch
path, and its own preconditions are typed (`TypeError` for a non-tensor or a dtype; `ValueError` for a row count that is
not a multiple of 128, a column count that is not a multiple of 64, another scale byte count, two devices).

## Numerics contract

There is no numerics knob: no split, no reduction across rows, no association choice. The output bytes are a function of
the operand values on one runtime, and of nothing else:

1. one fp32 accumulation chain per element in ascending K;
2. `alpha *`, then round-to-nearest-even to bf16 (gate and up);
3. SiLU as `x / (1 + 2**(x * fp32(-log2 e)))` with an approximate exponential and an approximate division;
4. `* up`, the second rounding to bf16;
5. the block maximum of `|v|` over 16 channels;
6. the block scale `E4M3(gs * (max|v| * fp32(1/6)))`, round-to-nearest-even, saturating at 448; a block whose scale byte is
   zero writes zero codes; otherwise the codes are scaled by `rcp(value(byte) * rcp(gs))`, two approximate reciprocals;
7. E2M1 round-to-nearest-even, saturating at 6.

- A row's bytes -- its codes and each of its block scales -- depend on that row's operands only: they are the same alone
  and among up to 64 rows, at any position, with any neighbours (batch invariance), and identical across repeated
  launches. See [determinism](/operations/determinism).
- Every field of the config is a performance form (below): the token tile, the weight stages issued ahead of the wait, the
  L2 policy of the weight stream, the cap of the persistent grid and the launch attribute. None of them changes a byte.
- Against an fp64 evaluation of the same chain on the exactly dequantized operands, every block scale lies in the interval
  of bytes its propagated error admits, and every code is within half a code step of its own block scale plus that error
  (the test utilities derive both bounds). Operands whose every intermediate is exact -- integer sums, bf16 ties of both
  roundings in both directions, a gate of zero, power-of-two scales -- give a host-built expectation byte for byte.
- This rounding chain is the entry's own. `cudnn.ops.nvfp4_block_scale_quantize` applies its encode scale to the elements
  before the block maximum, and it and the grouped GLU + Hadamard + quant fusion clamp the reciprocal of a zero block
  scale to the largest fp32 where this entry writes zero codes: the bytes of those APIs are not this entry's.
- Not a contract: equal bytes across GPU architectures or CuTe DSL compilers (the exponential, the division and the
  reciprocals are approximate instructions); the codes of a row whose own activation scales hold a NaN code (such a row
  leaves every other row's bytes unchanged); and the codes under an `out_global_scale` outside `2**-100 .. 2**100`,
  where a reciprocal can leave the normal range.

## Forms and knobs

`cudnn.gemm.frost.swiglu_decode.Cfg` holds the performance fields; `cfg=Cfg(...)` passes a whole config.

| field | default | meaning |
| --- | --- | --- |
| `TOKEN_TILE` | 8 | a FLOOR: a row count runs the narrowest of 8 / 16 / 32 / 64 tokens that holds it, not below the floor. One token tile per call |
| `PREWAIT_STAGES` | 2 | weight stages the loader issues ahead of the dependent-launch wait: 2 or 0 |
| `L2_EVICT_FIRST` | `True` | the weight stream is loaded with the evict-first L2 policy |
| `MAX_CLUSTERS` | 148 | cap of the persistent grid (one thread block per 128 weight rows up to the cap); above the part's SM count the launch takes a second wave |
| `PDL` | `False` | the dependent-launch attribute; the constructor's `pdl=` sets it |

| rows | token tile | operand stages | staging stages | shared memory per thread block | tensor-memory columns used |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 .. 8 | 8 | 5 | 7 | 231,424 B | 80 |
| 9 .. 16 | 16 | 4 | 12 | 231,424 B | 96 |
| 17 .. 32 | 32 | 4 | 4 | 231,424 B | 128 |
| 33 .. 64 | 64 | 3 | 7 | 230,400 B | 192 |

The stage counts come from one constant capacity on every part, so a token tile is the same kernel everywhere; the part's
own readings (shared memory per SM, per-block maximum, SM count) only decide whether the part can hold a form, at
construction. A thread block is 192 threads -- four epilogue warps, one MMA warp, one loader warp -- and allocates the
whole 512 tensor-memory columns of its SM; shared memory places one thread block per SM.

## Dependent launch

`pdl=True` (default off) puts the programmatic-stream-serialization attribute on every launch. Five of the six warps then
wait for the predecessor at once; the loader warp waits after it has issued its pre-wait weight stages.

**As a dependent.** The kernel reads `activation`, `activation_scale`, `alpha` and `out_global_scale` below its wait, so
they may be the predecessor's outputs. **Caller obligation:** no kernel that can still be running when the launch starts
-- the immediate stream predecessor and every earlier launch of an uninterrupted chain of early releases -- writes `weight` or
`weight_scale` (static weights satisfy it). The tensor-memory allocation is below the wait: a waiting thread block holds
shared memory and no tensor memory.

**As a predecessor.** A thread block signals its dependents when its loader warp has finished, BEFORE the block's last
output tile is written. A kernel launched behind this entry WITH the attribute may therefore start while `out` and
`out_scale` are still being written, and must issue its own wait before it reads them. A kernel launched without the
attribute starts after this one has completed. There is no release knob: the release point is the kernel's own.

## Launch facts

The entry states what a row count launches without launching it: three read-only answers, with `cfg_for`'s argument.

| fact | answer |
| --- | --- |
| `kernel_symbols_for(rows)` | the symbol of the kernel compiled for the form of `rows`, as a one-element tuple; empty until that form is compiled |
| `form_for(rows)` | the form as plain data: every field of the config `cfg_for` returns under its own name -- `TOKEN_TILE`, `PREWAIT_STAGES`, `L2_EVICT_FIRST`, `MAX_CLUSTERS`, `PDL`, the part's readings and `GEOM` (a nested `dict`). A fresh `dict` on every call |
| `launch_shape_for(rows)` | `(grid, block, shared_memory)`: grid `(1, 1, min(INTERMEDIATE / 64, MAX_CLUSTERS))`, block `(192, 1, 1)` and the shared memory of the form (the table above) -- `((1, 1, 148), (192, 1, 1), 231424)` for one row of `SwigluGeometry(5120, 17408)`. A profiler's kernel record of the launch reads the same three values |

One call is one kernel launch. The symbol carries the form right behind the kernel's name --
`cudnn_kernel_frost_swiglu_fp4_decode_h5120_i17408_t8_pw2_evf_c148_nopdl_...` -- so two forms of one entry never share a
symbol. A row count outside 1 .. 64 is a typed decline (`ValueError` for one that is not a positive `int`,
`NotImplementedError` past 64), and none of the three compiles, launches or touches device memory.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x (validated on cc 10.0; cc 10.3 and cc 10.7 are admitted and unverified -- the kernel's rendering is pinned for their targets); every other capability is declined by name |
| geometry | `SwigluGeometry(5120, 17408)` and the small validation geometry `SwigluGeometry(5120, 128)`; any other geometry is declined by name |
| rows | 1 .. 64 per call (a runtime value; `compile()` covers all of them with four forms); 65 and up declined by name, never served by another kernel |
| dtypes | FP4 E2M1 weight and activation with E4M3 block scales per 16 elements; two fp32 scalars; FP4 E2M1 out with E4M3 block scales; fp32 accumulation, bf16 intermediate |
| layouts | contiguous operands on the entry's device; 32-byte aligned `weight` and `activation`, 16-byte aligned scale buffers and outputs, 4-byte aligned scalars; `out` exactly `[rows, INTERMEDIATE / 2]` |
| determinism | bitwise replay and batch invariance at every row count 1 .. 64 (the [status page](/operations/determinism)) |
| backward | none: an inference step |
| DSL | nvidia-cutlass-dsl >= 4.8.0 compiling with CUDA 13.4 or later (its cu13 flavor under a CUDA 13 driver), with the part's target. The output bytes follow the compiler's lowering of three approximate instructions and were qualified with that compiler; an older one, such as the DSL's CUDA 12 flavor, is declined by name before any compile. The names the kernel is written on are probed before the kernel module is imported, and a DSL without one of them is declined by name. The work tiles are dealt by the DSL's own static persistent tile scheduler, which 4.8.0 marks as leaving its wheel: a compile there raises a `DeprecationWarning`, and a DSL that no longer ships it is one of those declines |

Typed declines, all before any DSL import: `TypeError` for a non-tensor, a dtype, or a geometry or config of another
type; `ValueError` for a shape, a byte count, an alignment, a device mismatch, a knob outside its domain, two sources for
the geometry, or a row count that is not a positive `int`; `NotImplementedError` for a row count past 64, a geometry, a
memory layout, a part or a DSL the kernel does not serve. No copy is ever made to serve a layout.

The kernel is derived from FlashInfer's persistent block-scaled dense GEMM (BSD-3-Clause notice retained, in an
Apache-2.0 project); the kernel file's header and `THIRD_PARTY_LICENSES.txt` carry the notice and the list of changes.

Tests: `test/python/gemm/frost/swiglu_decode/test_swiglu_fp4_decode.py` (host declines, config and ledger pins, the
rendering pin per target, the launch facts against the configuration arithmetic and against the profiler's record of a
launch, the device cells: the fp64 reference, exact cells, edge rows, every knob arm bitwise with the default, a waiting
dependent behind the entry, CUDA-graph replay, a warm stress with byte references) and
`test_swiglu_fp4_decode_determinism.py` (replay, batch invariance at every row count, the token-tile pick's byte
neutrality, goldens). Source: `python/cudnn/gemm/frost/swiglu_decode/`.
