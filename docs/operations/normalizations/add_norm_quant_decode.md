# Add + RMSNorm + Quantize Decode Step (SM100-class)

**Experimental.** `cudnn.AddNormQuantDecode` (also `cudnn.norm.frost.AddNormQuantDecode`) is the residual add, the
RMSNorm and the quantization of the next projection's input, at decode-time row counts, as ONE CuTe DSL launch:

```text
v        = x + residual                  (add=True; fp32)          residual <- bf16(v), in place
v        = x                             (add=False)               residual <- x when store_residual=True
y        = bf16( v * rsqrt(mean(v^2) + eps) * (1 + weight) )       fp32 arithmetic, one rounding to bf16 (nearest-even)
out      = y                                                       (bf16=True)
codes    = e4m3( y * (1 / scale) )                                 (quant="e4m3_static": the consumer's static per-tensor scale)
codes, scales = nvfp4( y ; global scale )                          (quant="nvfp4": 16-element blocks, E4M3 block scales)
```

`x` and `residual` are `[rows, hidden]` bf16 with `rows` in 1 .. 1024 (a runtime argument) and `hidden` 5120 (fixed at
construction). `weight` is the zero-centred norm weight, `[hidden]` or `[1, hidden]` bf16: the kernel applies
`1 + weight`. `scale` is one fp32 element (0-d or `[1]`). Compute capability 10.x; a CuTe DSL >= 4.8.0.

It is a decode-step entry: the residual add, the normalization and the quantization the next projection consumes, in one
launch. For a standalone normalization use the [Normalizations](/operations/Normalizations) graph API: its RMSNorm
kernel sums a row in another order, so its bytes are not this entry's, and it has no residual add and no quantized
output.

## Forms

Three residual modes times five output sets: fifteen forms, each one compiled kernel that serves every row count and
every layer. `add`, `bf16` and `quant` are required keywords; the forms are peers and none is a default.

| residual mode | arguments | effect |
| --- | --- | --- |
| add | `add=True` | `v = x + residual`; `residual` is updated in place with `bf16(v)` |
| store | `add=False, store_residual=True` | `v = x`; `residual` is written with `x` (the first layer's residual) |
| none | `add=False` | `v = x`; the call takes no `residual` (`None`) |

| output set | arguments | the call returns `(out, codes, scales)` |
| --- | --- | --- |
| bf16 | `bf16=True, quant=None` | `out` bf16 `[rows, hidden]`; `None`; `None` |
| bf16 + E4M3 | `bf16=True, quant="e4m3_static"` | `out`; `codes` `float8_e4m3fn` `[rows, hidden]`; `None` |
| E4M3 | `bf16=False, quant="e4m3_static"` | `None`; `codes`; `None` |
| bf16 + NVFP4 | `bf16=True, quant="nvfp4"` | `out`; `codes` uint8 `[rows, hidden / 2]`; `scales` `float8_e4m3fn` `[ceil(rows / 128) * 128, hidden / 16]` |
| NVFP4 | `bf16=False, quant="nvfp4"` | `None`; `codes`; `scales` |

The returned tuple has the same three slots for every form, with `None` in a slot the form does not produce. `quant` is
one vocabulary (`None` and `cudnn.norm.frost.kernel.add_norm_quant_decode_config.QUANT_FORMS`). A form that writes no
output (`bf16=False, quant=None`) and `add` together with `store_residual` are declined at construction.

NVFP4 codes are two per byte with the even element in the low nibble. NVFP4 block scales are stored in the 128 x 4
scale layout over (rows, blocks): tiles of 128 rows, one after the other; the scale buffer therefore has
`ceil(rows / 128) * 128` rows.

## Numerics contract

The sum tree and the rounding chain are the contract, not a tolerance. Per row, in fp32 on the bf16 operands:

1. `v = x + residual`: one fp32 addition per element (under `add`). The updated residual is `bf16(v)`, nearest-even.
2. The sum of squares is ONE fixed tree whose shape depends on `hidden` alone: a thread owns eight contiguous elements
   and folds them with one fused multiply-add each, in element order, from `+0`; the 32 lanes of a warp are summed by a
   butterfly with offsets 1, 2, 4, 8, 16; the warps (20 at `hidden` 5120) by the same butterfly.
3. `inv = rsqrt( total / hidden + eps )`: an IEEE division by the width, one addition, the part's approximate
   reciprocal square root (`rsqrt.approx.f32`).
4. `y = bf16( (v * inv) * (1 + weight) )`: `1 + weight` is one fp32 addition, then two multiplications and ONE rounding
   to bf16 (nearest-even). This bf16 value is the normalized value of record: `out` stores it and the codes are formed
   from it, so the codes are those of quantizing `out` itself.
5. `quant="e4m3_static"`: `code = e4m3( y * (1 / scale) )` with the correctly rounded fp32 reciprocal
   (`div.rn.f32(1, scale)`), nearest-even, saturating at +-448 (codes `0x7E` / `0xFE`). A consumer reads
   `code * scale`.
6. `quant="nvfp4"`: per block of 16 elements, the block scale is `e4m3( scale * (max|y| * fp32(1/6)) )` (the stored
   byte; two fp32 multiplications and one nearest-even rounding). The codes are `e2m1( y * q )` (nearest-even,
   saturating at +-6) with `q = rcp( block_scale * rcp(scale) )`, where `rcp` is the part's approximate reciprocal
   (`rcp.approx.ftz.f32`); a block whose scale byte is zero writes the codes of the signed zero. A consumer reads
   `code * block_scale / scale`.

What changes bytes: `hidden` (the tree and the divisor), `eps`, `add` (the function that is normalized), `quant` (which
codes exist) and the operand values; `scale` moves codes and block scales only. What never changes a byte: the row
count, a row's position, the other rows, `bf16`, `store_residual`, `pdl`, `pdl_release`, and who owns an output.
A row alone writes the bytes it writes inside a call of up to 1024 rows. A block scale's POSITION in the scale buffer
follows the row index (the layout); its value does not.

Against an fp64 evaluation of the bf16 operands, every `out` element is one of the two bf16 values that bracket the
exact normalized value (the kernel's fp32 value is within `20 x 2^-24` relative of it), and on random operands the
count of elements that are not the nearest of the two is bounded from the same arithmetic (a share of at most
`2 x 20 x 2^-24 x 2^8`, asserted with a binomial slack). The updated residual is
bitwise `bf16(fp32(x) + fp32(residual))`. The E4M3 codes and the NVFP4 block scales are bitwise the host models above
applied to the kernel's own `out`. Every NVFP4 code is one of at most two adjacent codes around the exact quotient
`y * scale / block_scale` (the bound of the two approximate reciprocals), and one code wherever the quotient is not
within that bound of an E2M1 rounding midpoint.

`scale` is by contract a finite positive fp32 in `2**-100 .. 2**100`. Its value is never read on the host: outside that
range the reciprocal chains overflow or flush. A NaN or infinite element makes its own row's `out` non-finite and
leaves that row's codes unspecified; every other row is unaffected. A row whose squares underflow normalizes against
`eps`; a row whose sum of squares overflows fp32 normalizes to zeros.

The bytes follow the part's `rsqrt.approx` and `rcp.approx.ftz` units. Equal bytes across GPU architectures are not
part of the contract; in particular an NVFP4 code at an exact rounding tie under a block scale that is not a power of
two follows the approximate reciprocal (see [determinism](/operations/determinism)).

## Usage

```python
import torch
import cudnn

dev = torch.device("cuda", 0)
norm = cudnn.AddNormQuantDecode(dev, 5120, 1e-6, add=True, bf16=True, quant="e4m3_static")   # one entry per form
norm.compile(rows)                                  # plan time: typed declines, DSL gates, the compile, outputs(rows)
out, codes, _ = norm(x, residual, weight, scale)    # residual <- bf16(x + residual); out bf16; codes E4M3
out, codes, _ = norm(x, residual, weight, scale, out=my_out, codes=my_codes)   # or into caller-owned buffers

first = cudnn.AddNormQuantDecode(dev, 5120, 1e-6, add=False, store_residual=True, bf16=False, quant="e4m3_static")
_, codes, _ = first(x, residual, weight, scale)     # residual <- x

mlp = cudnn.AddNormQuantDecode(dev, 5120, 1e-6, add=True, bf16=False, quant="nvfp4")
_, codes4, scales = mlp(x, residual, weight, global_scale)   # codes uint8 [rows, 2560]; scales E4M3, 128 x 4 layout
```

Everything behind `eps` is keyword-only. One compiled kernel serves every row count and every layer of a model: all
tensors are launch arguments, the slabs are bound with a runtime extent and `rows` is a runtime argument, so no shape
enters the compiled kernel, the compile key is the config alone, and a captured CUDA graph holds pointers, never
values. `compile()` needs no operand. The launch goes to torch's current stream of the entry's device. That device need
not be the caller's current one: the compile and the launch run under the entry's device, and the caller's current
device is restored afterwards. There is no workspace; a warm call allocates nothing and reads no device memory to the
host.

| operand | dtype | shape | notes |
| --- | --- | --- | --- |
| `x` | bf16 | `[rows, hidden]`, rows 1 .. 1024 | read only |
| `residual` | bf16 | `[>= rows, hidden]` | required in the add and store modes, `None` otherwise; in / out (add), out (store) |
| `weight` | bf16 | `[hidden]` or `[1, hidden]` | the zero-centred norm weight; read before the wait under `pdl` |
| `scale` | fp32 | one element (0-d or `[1]`) | required with a `quant`, `None` otherwise; read before the wait under `pdl` |
| `out` (optional) | bf16 | `[>= rows, hidden]` | only with `bf16=True` |
| `codes` (optional) | `float8_e4m3fn` / uint8 | `[>= rows, hidden]` / `[>= rows, hidden / 2]` | only with a `quant` |
| `scales` (optional) | `float8_e4m3fn` | `[>= ceil(rows / 128) * 128, hidden / 16]` | only with `quant="nvfp4"` |

Every operand is contiguous, starts at a 16-byte aligned address (`scale`: 4-byte) and lives on the entry's device; no
written operand may overlap another operand.

**Output ownership is per output.** An output the caller passes is the caller's and is returned as given; rows past the
call's are never written. An output the caller does not pass is the entry's: one set per row count holding only the
form's outputs, `scales` zero-filled, allocated by `compile(rows)` / `outputs(rows)` or at the first call of that row
count. Call `compile(rows)` for every row count before a CUDA-graph capture. Entry-owned outputs are valid until the
entry's next call at that row count from ANY host thread: stream order does not serialize host calls, so two host
threads that share one entry pass their own outputs or build one entry each. The per-row-count sets are never freed, so
a caller that sweeps many row counts passes its own outputs. An entry-owned output is an operand of the launch like a
passed one, so the overlap rule covers it: a call whose `x`, `residual`, `weight`, `scale` or passed output overlaps an
entry-owned output that the call does not pass (a tensor an earlier call of that row count returned, fed back in) is
declined with `ValueError` before the launch. To feed a returned output back in, pass a caller-owned output.

**The kernel writes the live rows' scale bytes only.** A caller-owned `scales` buffer serves one row count, or its
owner clears it when the row count shrinks: after a call with more rows, a call with fewer leaves the earlier call's
bytes in the rows between the new `rows` and `ceil(rows / 128) * 128`.

Everything the entry does not serve is declined by name: `TypeError` for a non-tensor, a dtype, a `hidden` that is not
an `int`, a flag that is not a `bool`, or a `rows` of `compile` / `outputs` that is not an `int`; `ValueError` for a
width outside the kernel's structure (a multiple of 256 in 256 .. 8192), an `eps` that is not a finite positive Python
`float` or `int` (the message names the type it got), `add` with `store_residual`, a `quant` outside the vocabulary, a
form without an output, `pdl_release=True` without `pdl=True`, an operand the form requires and did not get or got and
does not use, a shape, an alignment, a device mismatch, an overlap, or an empty call; `NotImplementedError` for a
structurally valid width other than 5120, a row count above 1024, a non-contiguous operand, a compute capability
outside 10.x, and the CuTe DSL gates (not installed, below 4.8.0, without the part's target, or with a compiler that
cannot assemble it). The operand checks import no DSL module and run before every compile and every launch; the
architecture gate and the DSL gates run in `compile()`, before the kernel module is imported.

## Launch knobs and the caller's obligations

Both knobs default to off and neither changes a byte (the suite compares their bytes).

- `pdl=True`: the launch carries the programmatic-stream-serialization attribute and the kernel issues
  `griddepcontrol.wait` AFTER it has loaded `weight` and `scale` and BEFORE it reads `x` / `residual` or writes
  anything. Obligation: `weight` and `scale` are not written by any launch that may still be running when this one
  starts (they are constants of the layer).
- `pdl_release` (`None` follows `pdl`; `True` needs `pdl=True`): `griddepcontrol.launch_dependents` directly behind
  this kernel's OWN wait; on this entry the release never precedes the wait. Obligation: every launch that can start
  before this one completes issues its own wait before it reads this launch's outputs and before it writes any operand
  of this call. Whether the release helps depends on the row count (a released consumer occupies SMs); a caller that
  wants it for some row counts only builds two entries.

`pdl=True` alone is wait + release; `pdl=True, pdl_release=False` is wait only. The suite pins the wait's position in
the kernel's text for every form and target: the loads of `weight` and `scale` are the only global accesses in front
of it, the loads of `x` and `residual` and every store follow it, and the release is the next instruction. On a
device it runs the step behind a predecessor that releases its dependents before it writes `x` and `residual`, the
release with a dependent launch on the stream that waits and copies the outputs, and two steps chained through one
buffer; each eager and captured.

## Launch facts

The entry states what a row count launches without launching it: three read-only answers, with `cfg_for`'s argument.

| fact | answer |
| --- | --- |
| `kernel_symbols_for(rows)` | the symbol of the entry's compiled kernel as a one-element tuple; empty until the entry is compiled. One compiled kernel serves every row count. The symbol carries the form directly behind the kernel's name -- `cudnn_kernel_frost_add_rms_norm_quant_h5120_add_bf16_e4m3_static_wait_release_eps358637bd_...`: the width, the residual mode, the outputs, the launch arm and the fp32 bits of `eps` -- so two forms whose kernels differ never share a symbol. The `eps` text is the fp32 value the kernel adds: two `eps` values that round to one fp32 value give two entries that compile the same kernel under one symbol |
| `form_for(rows)` | the form as plain data: the eight fields of the entry's config (`HIDDEN`, `EPS`, `ADD`, `STORE_RESIDUAL`, `BF16_OUT`, `QUANT`, `PDL`, `PDL_RELEASE`). The compile key is the config alone, so nothing else is part of it. A fresh `dict` on every call |
| `launch_shape_for(rows)` | `(grid, block, shared_memory)`: one thread block per row, `hidden / 8` threads, one fp32 slot of shared memory per warp -- `((3, 1, 1), (640, 1, 1), 80)` for three rows. A profiler's kernel record of the launch reads the same three values |

A row count outside 1 .. 1024 is a typed decline (`ValueError` for one that is not a positive `int`,
`NotImplementedError` past 1024), and none of the three compiles, launches or touches device memory.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x; validated on cc 10.0. The kernel's rendering is pinned for the `sm_100a`, `sm_103a` and `sm_107a` targets (one PTX body for the three): cc 10.3 and cc 10.7 are admitted on that basis and have not run on a device, and any other 10.x minor is admitted by the range alone. 9.0 and 12.x are declined by type |
| rows | 1 .. 1024 per call (runtime argument); 1025 and up declined by name. Three thread blocks of 640 threads share an SM, so one scheduling wave holds 444 rows on a 148-SM part and 636 on a 212-SM part: 1024 rows cross it |
| hidden | 5120, fixed at construction; another multiple of 256 up to 8192 is declined by name (each width is its own sum tree) |
| forms | the fifteen forms above; three launch arms (plain, wait, wait + release) |
| dtypes | bf16 `x` / `residual` / `weight` / `out`; one fp32 `scale`; E4M3 or packed E2M1 `codes`; E4M3 `scales`; fp32 arithmetic, rounded once to bf16 |
| layouts | contiguous slabs; 16-byte aligned operands (`scale`: 4-byte); no written operand overlapping another |
| determinism | bitwise replay and batch invariance (the [status page](/operations/determinism)) |
| DSL | nvidia-cutlass-dsl >= 4.8.0. On cc 10.0 / 10.3 the CUDA 12.9 and CUDA 13.4 compiler flavours of 4.8.0 render the same PTX and are both served (on cc 10.0 both ran and wrote the same bytes); on cc 10.7 the compiler must be CUDA 13.4 or later (the CUDA 12.9 flavour rejects the `sm_107a` target and is declined by name before JIT) |

Tests: `test/python/norm/frost/test_add_norm_quant_decode.py` (host declines, the config's limits and tables, the
rendering pinned per target and tied to the entry's own launch arguments, the launch facts against the configuration
arithmetic and against the profiler's record of a launch, the device cells against the references above, exact and
edge rows, rows whose sum of squares overflows, output ownership, the bit-neutral arms, the wait behind a predecessor
that released early, the release beside a dependent launch, a chain of two steps, CUDA-graph replay, an entry on a
device that is not the caller's current one) and `test_add_norm_quant_decode_determinism.py` (replay, batch invariance,
goldens). Source: `python/cudnn/norm/frost/`.
