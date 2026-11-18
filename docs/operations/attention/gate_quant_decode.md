# Attention Gate + Quantize Decode Step (SM100-class)

**Experimental.** `cudnn.AttentionGateQuantDecode` (also `cudnn.attention_gate.frost.AttentionGateQuantDecode`) is the
output gate of an attention block fused with the static FP8 quantization of the following projection's input, at
decode-time row counts, as ONE CuTe DSL launch:

```text
out[r, c] = e4m3( bf16( attn[r, c] / (1 + exp(-gate[r, c])) ) * (1 / scale) )
```

`attn` (the attention output) and `gate` (the PRE-sigmoid gate) are `[rows, width]` bf16 with `rows` in 1 .. 160 (a
runtime argument) and `width` a positive multiple of 8 up to 32768 (`heads x head_dim` of the attention output, fixed at
construction). `scale` is one fp32 element (0-d or `[1]`): the static per-tensor dequantization scale of the consumer, a
projection with an E4M3 input. `out` is `[rows, width]` E4M3 (`torch.float8_e4m3fn`), written into a caller-owned
`[>= rows, width]` buffer or the entry's own. Compute capability 10.x; a CuTe DSL >= 4.8.0.

The kernel is an element-wise map: thread `i` of a flat grid owns eight consecutive elements of the row-major slab (one
128-bit load of each operand, one convert and one byte store per element). It has no reduction, no shared memory and
no barrier, so every output byte is a function of its own `(attn, gate)` pair and of `scale`.

## Numerics contract

The rounding chain is the contract, not a tolerance. Per element, in fp32 on the bf16 operands:

1. `1 + exp(-gate)` with the CuTe DSL's precise exponential, then the IEEE division `attn / (1 + exp(-gate))`
   (`div.rn.f32`).
2. The quotient is rounded ONCE to bf16 (round-to-nearest-even).
3. That bf16 value is multiplied by the correctly rounded fp32 reciprocal of `scale` (`div.rn.f32(1, scale)`) and
   rounded ONCE to E4M3 (round-to-nearest-even, saturating at +-448: codes `0x7E` / `0xFE`).

This is the convention of a chain that stores the gated attention output as bf16 and quantizes it for a projection whose
static input scale is `scale`. The gated value is the DIVISION `attn / (1 + exp(-gate))`: a chain that rounds the
product `attn * sigmoid(gate)` instead writes a code one apart on rare elements, so a reference must spell the
division. The entry has no numerics argument: no knob, pick or row count changes a byte.

Against an fp64 evaluation of the bf16 operands, every byte is the E4M3 code of one of the two bf16 values that bracket
the exact gated value, and the code of the nearest one wherever the exact value is more than 2.5 fp32 units away from
the midpoint of that bracket. The test suite asserts both statements on drawn and adversarial operands and on every
finite gate value; its census module measures them over every `(attn, gate)` bf16 pair at five scales.

`scale` is a positive normal fp32 value by contract. Its value is never read on the host: a zero, subnormal, negative
or non-finite scale yields saturated, sign-flipped or NaN bytes. A NaN or infinite operand element yields the E4M3 NaN
byte or a saturated code for that element only.

The bytes follow the installed CuTe DSL's lowering of the exponential and the part's `ex2.approx` unit. Equal bytes
across DSL versions or across GPU architectures are not part of the contract (see
[determinism](/operations/determinism)).

## Usage

```python
import torch
import cudnn

dev = torch.device("cuda", 0)
entry = cudnn.AttentionGateQuantDecode(dev, width=2048)   # heads x head_dim of the attention output
entry.compile(attn, gate, scale)                          # plan time: typed declines, DSL gates, the compile
codes = entry(attn, gate, scale)                          # [rows, 2048] float8_e4m3fn: a view of the entry-owned buffer
codes = entry(attn, gate, scale, out)                     # or into a caller-owned [>= rows, 2048] float8_e4m3fn buffer
```

One compiled kernel serves every row count and every layer of a model: all four tensors are launch arguments, the
slabs are bound with a runtime extent and the element count `rows x width` is a runtime argument, so no shape enters
the compiled kernel, the compile key is the config alone, and a captured CUDA graph holds pointers, never values. The
launch goes to torch's current stream of the entry's device. That device need not be the caller's current one: the
compile and the launch run under the entry's device, and the caller's current device is restored afterwards. There is
no workspace; a warm call allocates nothing and reads no device memory to the host.

Everything the entry does not serve is declined by name: `TypeError` for a non-tensor, a dtype, a `width` that is not
an `int`, or a `cfg` of another type; `ValueError` for a shape, a non-contiguous operand, an alignment, a device
mismatch, an `out` that overlaps an input, an empty call, or a knob outside its vocabulary; `NotImplementedError` for a
row count above 160, an unserved width, a compute capability outside 10.x, and the CuTe DSL gates (not installed, below
4.8.0, without the part's target, or with a compiler that cannot assemble it). The operand checks import no DSL module
and run before every compile and every launch; the architecture gate and the DSL gates run in `compile()`, before the
kernel module is imported.

**Output ownership.** With `out` the caller owns the storage and gets the same tensor back; rows past the call's are
never written. Without it the entry returns a view of ONE entry-owned `[160, width]` buffer whose bytes are valid until
the entry's NEXT call from any host thread. Stream order does not serialize host calls, so two host threads that share
one entry, on one stream or two, pass their own `out` (or build one entry each). The entry-owned buffer is allocated
at the first `compile()` or call that passes no `out`; compile without `out` before a CUDA-graph capture if a captured
call will use it.

**Layout.** `attn`, `gate` and `out` are contiguous and start at a 16-byte aligned address (`scale`: 4-byte); every
operand lives on the entry's device.

## Knobs and the caller's obligation

`cudnn.attention_gate.frost.Cfg(THREADS=256, PDL_RELEASE=False)`, passed as `cfg=`. Both fields are performance-only
and bit-neutral (the suite compares their bytes):

- `THREADS` (128 or 256): threads per CTA, hence the grid. Without `cfg` the entry takes a pick from the element count
  and the part (`default_cfg`); the pick table holds one row today, 256 threads everywhere.
- `PDL_RELEASE`, also the keyword-only launch knob `pdl_release=` of the entry (`None` keeps the config's field):
  `griddepcontrol.launch_dependents` as the kernel's first statement. Default off. The launch itself is ordinary: it
  never carries the programmatic-stream-serialization attribute and the kernel holds no `griddepcontrol.wait`.

With the release on, every grid that can start before this kernel completes -- a grid launched with the
programmatic-stream-serialization attribute, and transitively its own dependents -- must issue its own
`griddepcontrol.wait` before it reads `out` and before it writes `attn`, `gate` or `scale`. A consumer launched without
the attribute is unaffected. The release is verified standalone only: its bytes equal the default's with no dependent
grid on the stream; a chain with a dependent launched under the attribute is not covered by this suite.

## Launch facts

The entry states what a row count launches without launching it: three read-only answers, with `cfg_for`'s argument.

| fact | answer |
| --- | --- |
| `kernel_symbols_for(rows)` | the symbol of the kernel compiled for the config of `rows`, as a one-element tuple; empty until that config is compiled. One compiled kernel serves every row count of the config |
| `form_for(rows)` | the form as plain data: `{"THREADS": ..., "PDL_RELEASE": ...}`, the fields of the config `cfg_for` returns with the launch knob applied. The compile key is the config alone, so nothing else is part of it. A fresh `dict` on every call |
| `launch_shape_for(rows)` | `(grid, block, shared_memory)`: a flat grid of `ceil(rows x width / 8 / THREADS)` thread blocks, `THREADS` threads, no shared memory -- `((3, 1, 1), (256, 1, 1), 0)` for one row of width 6144. A profiler's kernel record of the launch reads the same three values |

A row count outside 1 .. 160 is a typed decline (`ValueError` for one that is not a positive `int`,
`NotImplementedError` past 160), and none of the three compiles, launches or touches device memory.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x; validated on cc 10.3 and cc 10.7. The kernel's rendering is pinned for the `sm_100a`, `sm_103a` and `sm_107a` targets: cc 10.0 is admitted on that basis and has not run on a device, and any other 10.x minor is admitted by the range alone. 9.0 and 12.x are declined by type |
| rows | 1 .. 160 per call (runtime argument); 161 and up declined by name |
| width | a positive multiple of 8 up to 32768, fixed at construction |
| dtypes | bf16 `attn` / `gate`; one fp32 `scale`; E4M3 `out`; fp32 arithmetic, rounded once to bf16 and once to E4M3 |
| layouts | contiguous slabs; 16-byte aligned operands (`scale`: 4-byte); `out` must not overlap an input |
| determinism | bitwise replay and batch invariance (the [status page](/operations/determinism)) |
| DSL | nvidia-cutlass-dsl >= 4.8.0. On cc 10.0 / 10.3 the CUDA 12.9 and CUDA 13.4 compiler flavours of 4.8.0 render the same PTX and are both served; on cc 10.7 the compiler must be CUDA 13.4 or later (the CUDA 12.9 flavour rejects the `sm_107a` target and is declined by name before JIT) |

Tests: `test/python/attention_gate/frost/test_gate_quant_decode.py` (host declines, config and pick pins, the rendering
pinned per target and tied to the entry's own launch arguments, the launch facts against the configuration arithmetic
and against the profiler's record of a launch, the device cells against the fp64 reference, an entry
on a device that is not the caller's current one), `test_gate_quant_decode_determinism.py` (replay, batch
invariance, goldens) and `test_gate_quant_decode_census.py` (`pytest -m L2`: every bf16 pair). Source:
`python/cudnn/attention_gate/frost/`.
