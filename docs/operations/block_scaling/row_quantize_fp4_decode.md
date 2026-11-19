# FP4 Row Quantization Decode Step (SM100-class)

**Experimental.** `cudnn.RowQuantizeFp4Decode` (also `cudnn.row_quantize.frost.RowQuantizeFp4Decode`) quantizes the rows of a
bf16 slab to block-scaled FP4 (NVFP4) under a global scale, at decode-time row counts, as ONE CuTe DSL launch: the codes and the
block scales that a block-scaled FP4 multiply reads next, in that multiply's layout.

```text
per block of sixteen consecutive elements of a row (fp32 arithmetic on the bf16 values):
amax        = max |x_j|
scale byte  = e4m3( scale * (amax * fp32(1/6)) )      nearest-even, saturating at 448; `value` is the byte's fp32 value
encode      = 0  when value == 0,  else  rcp( value * rcp(scale) )
code_j      = e2m1( x_j * encode )                    nearest-even on {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating at +-6, sign kept
```

`x` is `[rows, width]` bf16 with `rows` in 1 .. 1024 (a runtime argument) and `width` 5120 (fixed at construction). `scale` is
the GLOBAL scale of the operand: one fp32 element (0-d or `[1]`), the reciprocal of the consuming multiply's static input
scale. The call returns `(codes, scales)`: `codes` uint8 `[rows, width / 2]`, two codes per byte with the even element in the
low nibble; `scales` `float8_e4m3fn` `[ceil(rows / 128) * 128, width / 16]`, one block scale per sixteen elements in the
128 x 4 scale layout over (rows, blocks). A consumer reads `code * block_scale / scale`. Compute capability 10.x; a CuTe
DSL >= 4.8.0.

It is a decode-step entry. Two neighbours of this library do related work and are not the same operation:

- [`cudnn.ops.nvfp4_block_scale_quantize`](/operations/BlockScaling/nvfp4_block_scale_conversion) converts whole 128-row bands
  (`M` a multiple of 128) with a caller workspace and another rounding rule: it multiplies every element by the encode scale
  first and takes the block maximum of the products. Its bytes are not this entry's. Use it to convert a stored tensor; use
  this entry for the rows of a decode step, 1 .. 1024 of them, inside a chain of dependent launches.
- The NVFP4 form of [`cudnn.AddNormQuantDecode`](/operations/Normalizations/add_norm_quant_decode) writes the SAME codes and
  the same block scales for the bf16 rows it normalizes (the suite holds that equality by raw bytes): a caller whose rows come
  out of that step can take its codes directly; a caller whose rows come from elsewhere uses this entry.

## Numerics contract

The rounding chain is the contract, not a tolerance. It is the block-scale convention with a global scale that this library
implements as `cudnn.frost.tile_dsl.pointwise.nvfp4_block_scale_global`. Per block of sixteen elements:

1. `amax = max |x_j|` over the block: exact in any order for a block without a NaN.
2. `s = scale * (amax * fp32(1/6))`: TWO fp32 multiplications in that association -- `amax * fp32(1/6)` first, the constant
   as the bit pattern `0x3E2AAAAB`.
3. The block scale is `s` rounded to E4M3 (nearest-even, saturating at 448). The stored byte is that code; `value` is its
   fp32 value.
4. The encode scale is `0` when `value` is zero, else `rcp( value * rcp(scale) )`, where `rcp` is the part's approximate
   reciprocal (`rcp.approx.ftz.f32`).
5. `code_j = e2m1( x_j * encode )`: one fp32 multiplication per element, then nearest on the E2M1 grid with ties to the
   even code, saturation at +-6 and the sign kept (a negative value that rounds to zero is the code 8).

What changes bytes: the sixteen values of a block and the value of `scale`. What never changes a byte: the row count, a row's
position, the other rows, the other blocks of the row, `pdl`, `pdl_release`, and who owns an output. A row alone writes the
bytes it writes inside a call of up to 1024 rows. A block scale's POSITION in the scale buffer follows the row index (the
layout); its value does not.

Stated behaviours: a zero block has scale byte 0 and the codes of its signed zeros (0 for `+0`, 8 for `-0`); a block whose
scale rounds to the zero code although the block is not zero encodes as zeros with the elements' signs; a block whose scale
saturates at 448 encodes its largest elements at +-6.

Against a reference written from the definition above, every block scale is BITWISE the reference's, and every code is one
of at most two adjacent codes around the exact quotient `x * scale / block_scale` (the bound of the two approximate
reciprocals, `5 x 2^-23` relative) -- one code wherever the quotient is not within that bound of an E2M1 rounding midpoint.
Where `scale` and a block's scale value are powers of two, both reciprocals are exact and the codes are bitwise
`e2m1(x * scale / block_scale)`.

`scale` is by contract a finite positive fp32 in `2**-100 .. 2**100`. Its value is never read on the host: outside that range
the reciprocal chain overflows or flushes. A block that holds a NaN or an infinity leaves THAT BLOCK's codes and scale
unspecified; every other block of the row and every other row is unaffected. (The block maximum is taken with `max.f32`,
which returns the other operand when one is a NaN; the NVFP4 form of `cudnn.AddNormQuantDecode` keeps the NaN. The two write
the same bytes for every block without a NaN and may differ on a block that holds one.)

The codes follow the part's `rcp.approx.ftz` unit. Equal bytes across GPU architectures are not part of the contract; a code
at an exact rounding tie under a block scale that is not a power of two follows the approximate reciprocal (see
[determinism](/operations/determinism)).

## Usage

```python
import torch
import cudnn

dev = torch.device("cuda", 0)
quantize = cudnn.RowQuantizeFp4Decode(dev, 5120)             # one entry per width and launch arm
quantize.compile(rows)                                       # plan time: typed declines, DSL gates, the compile, outputs(rows)
codes, scales = quantize(x, scale)                           # codes uint8 [rows, 2560]; scales E4M3, 128 x 4 layout
codes, scales = quantize(x, scale, codes=my_codes, scales=my_scales)   # or into caller-owned buffers

chained = cudnn.RowQuantizeFp4Decode(dev, 5120, pdl=True)    # inside a chain of dependent launches: wait + release
```

`pdl` and `pdl_release` are keyword-only. One compiled kernel serves every row count: all tensors are launch arguments, the
slabs are bound with a runtime extent and `rows` is a runtime argument, so no shape enters the compiled kernel, the compile
key is the config alone, and a captured CUDA graph holds pointers, never values. `compile()` needs no operand. The launch
goes to torch's current stream of the entry's device. That device need not be the caller's current one: the compile and the
launch run under the entry's device, and the caller's current device is restored afterwards. There is no workspace; a warm
call allocates nothing and reads no device memory to the host.

| operand | dtype | shape | notes |
| --- | --- | --- | --- |
| `x` | bf16 | `[rows, width]`, rows 1 .. 1024 | read only |
| `scale` | fp32 | one element (0-d or `[1]`) | the global scale; read before the wait under `pdl` |
| `codes` (optional) | uint8 | `[>= rows, width / 2]` | two codes per byte, the even element in the low nibble |
| `scales` (optional) | `float8_e4m3fn` | `[>= ceil(rows / 128) * 128, width / 16]` | the 128 x 4 scale layout over (rows, blocks) |

Every operand is contiguous, starts at a 16-byte aligned address (`scale`: 4-byte) and lives on the entry's device; no
written operand may overlap another operand.

**The scale layout.** Bands of 128 rows follow one another; a band is one 512-byte atom per four scale columns; inside an
atom the byte of scale `(row, column)` is `(row % 32) * 16 + ((row // 32) % 4) * 4 + column % 4`. At `width` 5120 a row has
320 block scales, 80 whole atoms per band: the buffer has no padded column, and its rows past the call's row count up to the
next multiple of 128 are padding.

**Output ownership is per output.** An output the caller passes is the caller's and is returned as given; rows past the
call's are never written. An output the caller does not pass is the entry's: one pair per row count, `scales` zero-filled,
allocated by `compile(rows)` / `outputs(rows)` or at the first call of that row count. `compile()` without a row count
allocates the pair of ONE row (2,560 code bytes and one 128-row scale band): the compile binds it, also for a caller that
passes its own outputs on every call. Call `compile(rows)` for every row
count before a CUDA-graph capture. Entry-owned outputs are valid until the entry's next call at that row count from ANY host
thread: stream order does not serialize host calls, so two host threads that share one entry pass their own outputs or build
one entry each. The per-row-count pairs are never freed, so a caller that sweeps many row counts passes its own outputs. An
entry-owned output is an operand of the launch like a passed one, so the overlap rule covers it: a call whose `x`, `scale` or
passed output overlaps an entry-owned output that the call does not pass is declined with `ValueError` before the launch.

**The kernel writes the live rows' scale bytes only.** An entry-owned scale buffer is zero in its padded rows. A caller-owned
`scales` buffer serves one row count, or its owner clears it when the row count shrinks: after a call with more rows, a call
with fewer leaves the earlier call's bytes in the rows between the new `rows` and `ceil(rows / 128) * 128`.

Everything the entry does not serve is declined by name: `TypeError` for a non-tensor, a dtype (`x` bf16, `scale` fp32,
`codes` uint8, `scales` `float8_e4m3fn`), a `width` that is not an `int`, a launch knob that is not a `bool` or `None`, or a
`rows` of `compile` / `outputs` that is not an `int`; `ValueError` for a width outside the kernel's structure (a multiple of
512 in 512 .. 16384), `pdl_release=True` without `pdl=True`, a shape, an alignment, a device mismatch, an overlap, or an
empty call; `NotImplementedError` for a structurally valid width other than 5120, a row count above 1024, a non-contiguous
operand, a compute capability outside 10.x, and the CuTe DSL gates (not installed, below 4.8.0, without the part's target, or
with a compiler that cannot assemble it). The operand checks import no DSL module and run before every compile and every
launch; the architecture gate and the DSL gates run in `compile()`, before the kernel module is imported.

## Launch knobs and the caller's obligations

Both knobs default to off and neither changes a byte (the suite compares their bytes).

- `pdl=True`: the launch carries the programmatic-stream-serialization attribute and the kernel issues
  `griddepcontrol.wait` AFTER it has loaded `scale` and BEFORE it reads `x` or writes anything. Obligation: `scale` is not
  written by any launch that may still be running when this one starts (it is a constant of the layer).
- `pdl_release` (`None` follows `pdl`; `True` needs `pdl=True`): `griddepcontrol.launch_dependents` directly behind this
  kernel's OWN wait; on this entry the release never precedes the wait. Obligation: every launch that can start before this
  one completes issues its own wait before it reads this launch's outputs and before it writes any operand of this call.

`pdl=True` alone is wait + release; `pdl=True, pdl_release=False` is wait only. The suite pins the wait's position in the
kernel's text for every target -- the load of `scale` is the only global access in front of it, the loads of `x` and both
stores follow it, and the release is the next instruction -- and on a device it runs the step behind a predecessor that
releases its dependents before it writes `x`, and the release with dependent launches on the stream that wait and copy the
outputs; each eager and captured.

## Launch facts

The entry states what a row count launches without launching it: three read-only answers, with `cfg_for`'s argument.

| fact | answer |
| --- | --- |
| `kernel_symbols_for(rows)` | the symbol of the entry's compiled kernel as a one-element tuple; empty until the entry is compiled. One compiled kernel serves every row count. The symbol carries the form directly behind the kernel's name -- `cudnn_kernel_frost_row_quantize_fp4_w5120_wait_release_...`: the width and the launch arm -- so two forms never share a symbol |
| `form_for(rows)` | the form as plain data: the three fields of the entry's config (`WIDTH`, `PDL`, `PDL_RELEASE`). The compile key is the config alone, so nothing else is part of it. A fresh `dict` on every call |
| `launch_shape_for(rows)` | `(grid, block, shared_memory)`: one thread block per row, `width / 16` threads, no shared memory -- `((3, 1, 1), (320, 1, 1), 0)` for three rows. A profiler's kernel record of the launch reads the same three values |

A row count outside 1 .. 1024 is a typed decline (`ValueError` for one that is not a positive `int`, `NotImplementedError`
past 1024), and none of the three compiles, launches or touches device memory.

## The kernel

One thread block per row; one thread per block of sixteen elements (320 threads at `width` 5120). A thread loads its sixteen
values with two 128-bit loads, forms the block maximum in registers, the block scale and the encode scale, stores the block's
eight code bytes with one 64-bit store and its scale byte with one byte store. No value crosses a thread: the kernel has no
barrier, no warp shuffle and no shared memory. The global scale is read in front of the dependent-launch wait.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x; validated on cc 10.0. The kernel's rendering is pinned for the `sm_100a`, `sm_103a` and `sm_107a` targets, and from the kernel's entry on the `sm_103a` and `sm_107a` renderings are the `sm_100a` text: cc 10.3 and cc 10.7 are admitted on that basis and have not run on a device, and any other 10.x minor is admitted by the range alone. 9.0 and 12.x are declined by type |
| rows | 1 .. 1024 per call (runtime argument); 1025 and up declined by name. Six thread blocks of 320 threads share an SM, so one scheduling wave holds 888 rows on a 148-SM part and 1,272 on a 212-SM part: the 1024-row call crosses a wave on a part of at most 170 SMs, and no served call does on a 212-SM part. The kernel holds no state between thread blocks |
| width | 5120, fixed at construction; another multiple of 512 up to 16384 is declined by name |
| forms | one kernel body; three launch arms (plain, wait, wait + release) |
| dtypes | bf16 `x`; one fp32 `scale`; packed E2M1 `codes` (uint8); E4M3 `scales`; fp32 arithmetic |
| layouts | contiguous slabs; 16-byte aligned operands (`scale`: 4-byte); no written operand overlapping another |
| determinism | bitwise replay and batch invariance (the [status page](/operations/determinism)) |
| DSL | nvidia-cutlass-dsl >= 4.8.0. On cc 10.0 / 10.3 both compiler flavours of the DSL (CUDA 12.9 and CUDA 13.4) assemble the kernel and are served; on cc 10.0 both ran under CuTe DSL 4.8.0 and 4.8.1 and wrote the same bytes. On cc 10.7 the compiler must be CUDA 13.4 or later (the CUDA 12.9 flavour rejects the `sm_107a` target and is declined by name before JIT). Other DSL versions are served unverified |

Tests: `test/python/row_quantize/frost/test_row_quantize_fp4_decode.py` (host declines, the config's limits and tables, the
rendering pinned per target with the byte every global load and store addresses and the compiled kernel's operand wiring, and
tied to the entry's own launch arguments, the scale positions against the layout a block-scaled FP4 multiply reads, the
launch facts against the configuration arithmetic and against the profiler's record of a launch, the device cells against
the references above under eleven global scales, the scale byte of every finite bf16 block maximum, the exact cells, the
equality with the NVFP4 form of the add + RMSNorm + quantize step, blocks with a NaN or an infinity, output ownership, the
bit-neutral arms, the wait behind a predecessor that released early, the release beside a dependent launch, CUDA-graph
replay) and `test_row_quantize_fp4_decode_determinism.py` (replay, batch invariance, locality, goldens). Source:
`python/cudnn/row_quantize/frost/`.
