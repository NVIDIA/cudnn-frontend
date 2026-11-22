# FP8 Row Quantization Decode Step (SM100-class)

**Experimental.** `cudnn.RowQuantizeFp8Decode` (also `cudnn.row_quantize.frost.RowQuantizeFp8Decode`) quantizes the rows of a
bf16 slab to FP8 (E4M3), each row under its own scale, at decode-time row counts, as ONE CuTe DSL launch: the codes and the
per-row scales that an FP8 multiply with a dynamic per-row input scale reads next.

```text
per row (fp32 arithmetic on the bf16 values):
amax    = max |x_j|
scale   = max( amax / 448, fp32(1 / (448 * 512)) )     one correctly rounded division; the floor is the bits 0x36924925
inv     = 1 / scale                                    one correctly rounded division
code_j  = e4m3( clamp(x_j * inv, -448, 448) )          one multiplication, then nearest-even to E4M3, saturating
```

`x` is `[rows, width]` bf16 with `rows` in 1 .. 1024 (a runtime argument) and `width` 17408 (fixed at construction). The
call returns `(codes, scales)`: `codes` `float8_e4m3fn` `[rows, width]`, one code per element; `scales` fp32 `[rows, 1]`,
one scale per row. A consumer reads `code * scale`. Compute capability 10.x; a CuTe DSL >= 4.8.0.

It is a decode-step entry. Its neighbours quantize differently and are not the same operation:

- [`cudnn.AttentionGateQuantDecode`](/operations/Attention/gate_quant_decode) and the FP8 form of
  [`cudnn.AddNormQuantDecode`](/operations/Normalizations/add_norm_quant_decode) quantize under ONE static scale the caller
  passes; this entry computes a scale per row from the row itself and returns it.
- [`cudnn.RowQuantizeFp4Decode`](/operations/BlockScaling/row_quantize_fp4_decode) writes 4-bit codes with one block scale per
  sixteen elements under a global scale.

## Numerics contract

The rounding chain is the contract, not a tolerance, and every step of it is exactly specified:

1. `amax = max |x_j|` over the row: exact in any order for a row without a NaN.
2. `scale = max(amax / 448, floor)`: ONE correctly rounded fp32 division and one exact maximum. `floor` is the fp32 bit
   pattern `0x36924925`, the fp32 value of `1 / (448 * 512)`: a row whose maximum is at most `2**-9` has the floor scale.
3. `inv = 1 / scale`: ONE correctly rounded fp32 division (never an approximate reciprocal).
4. `code_j = e4m3(clamp(x_j * inv, -448, 448))`: one fp32 multiplication, rounded to fp32; then the nearest E4M3 value with
   an exact tie going to the even code, saturation at +-448, and the sign kept (a negative value that rounds to zero is the
   code `0x80`).

Against a reference written from these four lines, every code byte and every scale bit is BITWISE the reference's. Because
no step is approximate, a row's bytes are a function of its values alone: they do not depend on the GPU architecture, on the
DSL's compiler flavour, on `width` beyond the values it holds, on the row count, on the row's position, on the other rows,
on `pdl` / `pdl_release`, or on who owns an output. A row alone writes the bytes it writes inside a call of up to 1024 rows.

Stated behaviours: a zero row has the floor scale and the codes of its signed zeros (`0x00` for `+0`, `0x80` for `-0`); once
the row maximum is above `2**-9`, the elements that hold it have the codes of +-448 (`0x7E` / `0xFE`); under the floor scale
a row's codes are `e4m3(x * fp32(1 / floor))`.

A row that holds a NaN or an infinity leaves THAT ROW's codes and scale unspecified; every other row is unaffected. (The row
maximum is taken with `max.f32`, which returns the other operand when one is a NaN, and an infinite maximum gives an infinite
scale.) Nothing is checked on the host.

## Usage

```python
import torch
import cudnn

dev = torch.device("cuda", 0)
quantize = cudnn.RowQuantizeFp8Decode(dev, 17408)            # one entry per width and launch arm
quantize.compile(rows)                                       # plan time: typed declines, DSL gates, the compile, outputs(rows)
codes, scales = quantize(x)                                  # codes E4M3 [rows, 17408]; scales fp32 [rows, 1]
codes, scales = quantize(x, codes=my_codes, scales=my_scales)   # or into caller-owned buffers

chained = cudnn.RowQuantizeFp8Decode(dev, 17408, pdl=True)   # inside a chain of dependent launches: release + wait
```

`pdl` and `pdl_release` are keyword-only. One compiled kernel serves every row count: the operands cross the launch as
addresses and `rows` is a runtime argument, so no shape, stride or alignment enters the compiled kernel, the compile key is
the config alone, and a captured CUDA graph holds addresses, never values. `compile()` needs no operand and allocates
nothing. The launch goes to torch's current stream of the entry's device. That device need not be the caller's current one:
the compile and the launch run under the entry's device, and the caller's current device is restored afterwards. There is
no workspace; a warm call allocates nothing and reads no device memory to the host.

| operand | dtype | shape | notes |
| --- | --- | --- | --- |
| `x` | bf16 | `[rows, width]`, rows 1 .. 1024 | read only; contiguous along its last axis; its row stride a multiple of 8 elements (the rows of a wider slab bind as they are); 16-byte aligned |
| `codes` (optional) | `float8_e4m3fn` | `[>= rows, width]` | contiguous; 8-byte aligned; rows past the call's are not written |
| `scales` (optional) | fp32 | `[>= rows, 1]` | contiguous; 4-byte aligned; rows past the call's are not written |

Every operand lives on the entry's device; no written operand may overlap another operand.

**Output ownership is per output.** An output the caller passes is the caller's and is returned as given. An output the
caller does not pass is the entry's: one pair per row count, allocated by `compile(rows)` / `outputs(rows)` or at the first
call of that row count. The pairs are never freed (a pair of `r` rows holds `r * (width + 4)` bytes), so a caller that sweeps
many row counts passes its own outputs. Call `compile(rows)` for every row count before a CUDA-graph capture; it may run on
any stream. The allocation of a pair is stream work of torch's current stream of the entry's device at that moment (the
entry does not initialize the pair, but under deterministic algorithms torch fills it, and its blocks can be recycled ones
that stream still owes a write), and every launch that writes bytes of the pair is ordered behind that work on whatever
stream it runs, however the tensor reached the call: the entry's own output of the row count, or a tensor taken with
`outputs(rows)` and passed back, whole or as a view, by this entry or by another entry that orders its launches the same way
(`cudnn.frost.init_order`; the test is one of addresses). The launch stream waits for an event recorded behind the
allocation; the host does not wait, and no stream and no device is synchronized. A capturing stream cannot wait for an event
recorded outside the capture, so a captured call that finds the work still pending waits for that one event on the host
before its captured launch; a pair allocated before the capture needs no wait once its stream's earlier work is done. A pair
allocated INSIDE a capture is not ordered: allocate before the capture. `outputs(rows)` returns the pair without waiting: a
caller that reads or writes the returned tensors itself, outside a call of an entry, orders that itself. Entry-owned outputs
are valid until the entry's next call at that row count from ANY host thread: stream order does not serialize host calls, so
two host threads that share one entry pass their own outputs or build one entry each. An entry-owned output is an operand of
the launch like a passed one, so the overlap rule covers it: a call whose `x` or passed output overlaps an entry-owned
output that the call does not pass is declined with `ValueError` before the launch.

Everything the entry does not serve is declined by name: `TypeError` for a non-tensor, a dtype (`x` bf16, `codes`
`float8_e4m3fn`, `scales` fp32), a `width` that is not an `int`, a launch knob that is not a `bool` or `None`, or a `rows`
of `compile` / `outputs` that is not an `int`; `ValueError` for a width outside the kernel's structure (a multiple of 8 in
8 .. 24576), `pdl_release=True` without `pdl=True`, a shape, a row stride of `x` that is not whole 16-byte vectors, an
alignment, a device mismatch, an overlap, or an empty call; `NotImplementedError` for a structurally valid width other than
17408, a row count above 1024, an `x` that is not contiguous along its last axis, an output that is not contiguous, a
compute capability outside 10.x, and the CuTe DSL gates (not installed, below 4.8.0, without the part's target, or with a
compiler that cannot assemble it). The operand checks import no DSL module and run before every launch; the architecture
gate and the DSL gates run in `compile()`, before the kernel module is imported.

## Launch knobs and the caller's obligations

Both knobs default to off and neither changes a byte (the suite compares their bytes).

- `pdl=True`: the launch carries the programmatic-stream-serialization attribute and the kernel issues
  `griddepcontrol.wait` BEFORE it reads `x` or writes anything; it reads nothing in front of the wait.
- `pdl_release` (`None` follows `pdl`; `True` needs `pdl=True`): `griddepcontrol.launch_dependents` as the kernel's FIRST
  statement, in front of its own wait. Obligation: every launch that can start before this one completes issues its own
  wait before it reads this launch's outputs and before it writes any operand of this call.

`pdl=True` alone is release + wait; `pdl=True, pdl_release=False` is wait only. The suite holds the positions in the
kernel's rendering for every claimed target -- the wait in front of every global load and store, the release in front of
the wait -- and on a device it runs the step behind a predecessor that releases its dependents before it writes `x`, and
the release with dependent launches on the stream that wait and copy the outputs; each eager and captured.

## Launch facts

The entry states what a row count launches without launching it: three read-only answers, with `cfg_for`'s argument.

| fact | answer |
| --- | --- |
| `kernel_symbols_for(rows)` | the symbol of the entry's compiled kernel as a one-element tuple; empty until the entry is compiled. One compiled kernel serves every row count. The symbol carries the form directly behind the kernel's name -- `cudnn_kernel_frost_row_quantize_fp8_w17408_release_wait_...`: the width and the launch arm -- so two forms never share a symbol |
| `form_for(rows)` | the form as plain data: the three fields of the entry's config (`WIDTH`, `PDL`, `PDL_RELEASE`). The compile key is the config alone, so nothing else is part of it. A fresh `dict` on every call |
| `launch_shape_for(rows)` | `(grid, block, shared_memory)`: one thread block per row, 1024 threads, 128 bytes of shared memory -- `((3, 1, 1), (1024, 1, 1), 128)` for three rows. A profiler's kernel record of the launch reads the same three values |

A row count outside 1 .. 1024 is a typed decline (`ValueError` for one that is not a positive `int`, `NotImplementedError`
past 1024), and none of the three compiles, launches or touches device memory.

## The kernel

One thread block per row. A thread owns whole vectors of eight consecutive elements: at `width` 17408 a row is 2,176
vectors and a block is 1024 threads, thread `t` owning the vectors `t`, `t + 1024` and -- for `t < 128` -- `t + 2048`. A
thread loads each of its vectors with one 128-bit load and forms the maximum magnitude of its values in registers; the row
maximum is ONE block reduction (`cudnn.frost.tile_dsl.rowwise.block_reduce`, `kind="max"`: a warp maximum, one shared-memory
word per warp, one thread-block barrier, a second warp maximum); every thread then forms the scale and its reciprocal and
stores each of its vectors' eight code bytes with one 64-bit store; thread 0 stores the scale. A thread without a vector in
the last pass neither loads nor stores for it. The kernel has no mbarrier and no loop; its one shared-memory buffer is
written once and read once per launch.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x; validated on cc 10.0. The kernel compiles for the `sm_100a`, `sm_103a` and `sm_107a` targets with the same synchronization and dependent-launch positions (the suite's rendering cell): cc 10.3 and cc 10.7 are admitted on that basis and have not run on a device, and any other 10.x minor is admitted by the range alone. 9.0 and 12.x are declined by type |
| rows | 1 .. 1024 per call (runtime argument); 1025 and up declined by name. Two thread blocks of 1024 threads share an SM, so one scheduling wave holds 296 rows on a 148-SM part and 424 on a 212-SM part: the 1024-row call crosses waves on every served part. The kernel holds no state between thread blocks |
| width | 17408, fixed at construction; another multiple of 8 up to 24576 is declined by name |
| forms | one kernel body; three launch arms (plain, wait, release + wait) |
| dtypes | bf16 `x`; E4M3 `codes`; fp32 `scales`; fp32 arithmetic |
| layouts | `x`: rows of a slab whose row stride is whole 16-byte vectors, 16-byte aligned; contiguous outputs (`codes` 8-byte, `scales` 4-byte aligned); no written operand overlapping another |
| determinism | bitwise replay and batch invariance (the [status page](/operations/determinism)) |
| DSL | nvidia-cutlass-dsl >= 4.8.0. On cc 10.0 / 10.3 both compiler flavours of the DSL (CUDA 12.9 and CUDA 13.4) are served; on cc 10.0 both ran under CuTe DSL 4.8.0 and wrote the same bytes. On cc 10.7 the compiler must be CUDA 13.4 or later (the CUDA 12.9 flavour rejects the `sm_107a` target and is declined by name before JIT). Other DSL versions are served unverified |

Tests: `test/python/row_quantize/frost/test_row_quantize_fp8_decode.py` (the width's structure and thread plan, the host
declines, the order behind an entry-owned allocation, the gates, the launch facts against the configuration arithmetic and
against the profiler's record of a launch, the kernel's rendering per claimed target, and the device cells: the reference
above at 1 .. 1024 rows, every finite bf16 magnitude as a row maximum, rows whose bytes are stated as literals, the rows of
a wider slab and a repeated row, rows with a NaN or an infinity, output ownership, the bit-neutral arms, the wait behind a
predecessor that released early, the release beside a dependent launch, CUDA-graph replay, the warm path, a warm launch
behind a pending allocation on another stream) and `test_row_quantize_fp8_decode_determinism.py` (replay, batch invariance,
locality, goldens). Source: `python/cudnn/row_quantize/frost/`.
