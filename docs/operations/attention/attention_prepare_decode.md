# Attention Preparation Decode Step (SM100-class)

**Experimental.** `cudnn.AttentionPrepareDecode` (also `cudnn.attention_prepare.frost.AttentionPrepareDecode`, with
`cudnn.attention_prepare.AttentionPrepareGeometry`) prepares the operands of a paged attention for the tokens of a
decode step, as ONE CuTe DSL launch: per token the RMSNorm of every query and key head row, the rotation of its
rotary part by the token's position, the gate rows copied beside the queries, and the keys and values written into
paged pools at the token's slot.

```text
n(x, w)              = bf16( x * rsqrt(mean(x^2) + eps) * (1 + w) )               one head row, fp32 arithmetic, one rounding to bf16
rot(n, p)[r]         = bf16( n[r] * cos[p_r, r] - n[r + HALF] * sin[p_r, r] )     r < HALF = ROTARY_DIM / 2; p_r = the token's position for index r
rot(n, p)[r + HALF]  = bf16( n[r + HALF] * cos[p_r, r] + n[r] * sin[p_r, r] )
rot(n, p)[e]         = n[e]                                                       e >= ROTARY_DIM

q_out[t, h]          = rot(n(q[t, h], q_weight), positions of t)                  h < HEADS
gate_out[t, h]       = gate[t, h]                                                 a bit copy
k_pool[page, h, row] = rot(n(k[t, h], k_weight), positions of t)                  h < KV_HEADS; page, row = divmod(slot_mapping[t], page_size)
v_pool[page, h, row] = v[t, h]                                                    a bit copy
```

A token whose slot is negative (the padding convention), or at or past `pages * page_size`, or that lies past the
slot vector's length, writes no pool row; its `q_out` and `gate_out` rows are written like every other token's.

It is a decode-step entry: an inference step with no backward, called by an engine that owns the tensors. Compute
capability 10.x; a CuTe DSL >= 4.8.0.

## Construction facts

```python
AttentionPrepareDecode(device, geometry, eps, *, positions, page_size, position_dtype, slot_dtype,
                       mrope_section=None, pdl=None, pdl_release=None)
```

| argument | meaning | served |
| --- | --- | --- |
| `geometry` | `AttentionPrepareGeometry(HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM)` | `(24, 4, 256, 64)` |
| `eps` | the norm's epsilon, a Python `float` | any finite positive value |
| `positions` | the position form, REQUIRED: `"1d"` (one position per token) or `"3d"` (three position rows; a rotary index reads one of them) | both |
| `mrope_section` | `"3d"` only, required there: `(t, h, w)` with `t + h + w == ROTARY_DIM / 2`. Rotary index `r` reads row 1 when `r % 3 == 1 and r < 3 h`, row 2 when `r % 3 == 2 and r < 3 w`, and row 0 otherwise | any such triple of positive ints |
| `page_size` | REQUIRED: the rows of one page of BOTH pools | 64, 16 |
| `position_dtype`, `slot_dtype` | REQUIRED: the integer dtype of `positions` / `slot_mapping`, by name: `"int32"` or `"int64"` | both |
| `pdl`, `pdl_release` | the launch knobs (below) | three arms |

**Every compile-time fact is a construction fact.** The geometry, `eps`, the position form with its section, the page
size and the two integer dtypes are constants of the compiled kernel, so `compile()` needs no operand and nothing
compiles inside a step. The page size is a constant because the kernel divides the slot by it (a shift for a power
of two; a run-time page size would be an integer division per key thread). An integer dtype is a load width. The
consequences: a pool whose page size is not the entry's is declined by name; a `positions` or `slot_mapping` tensor
of the other integer dtype is a `TypeError` that names the entry's dtype; and a caller that needs two page sizes or
two integer dtypes builds two entries. The slot vector's LENGTH is a run-time argument: one compiled kernel serves
every length, the empty vector included.

## Usage

```python
import torch
import cudnn
from cudnn.attention_prepare import AttentionPrepareGeometry

dev = torch.device("cuda", 0)
prepare = cudnn.AttentionPrepareDecode(dev, AttentionPrepareGeometry(24, 4, 256, 64), 1e-6, positions="3d",
                                       mrope_section=(11, 11, 10), page_size=64,
                                       position_dtype="int64", slot_dtype="int64")
prepare.compile(tokens)                        # plan time: typed declines, DSL gates, the compile, outputs(tokens)

# the operands as views: nothing is copied
pairs = slab[:, : 24 * 512].view(tokens, 24, 512)          # one projection slab [tokens, 14336]
q, gate = pairs[..., :256], pairs[..., 256:]               # strides (14336, 512, 1)
k = slab[:, 12288:13312].view(tokens, 4, 256)              # strides (14336, 256, 1)
v = slab[:, 13312:].view(tokens, 4, 256)
k_pool, v_pool = cache[..., :256], cache[..., 256:]        # one packed cache [pages, 4, 64, 512]

q_out, gate_out = prepare(q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping)
q_out, gate_out = prepare(q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping,
                          q_out=my_q, gate_out=my_gate)    # or into caller-owned buffers
```

One compiled kernel serves every token count and every layer of a model: the kernel takes every operand as its
device address and every free stride as a byte count, and the token count is its grid. No shape, stride or alignment
of an operand enters the compiled kernel, the compile key is the config alone, and a captured CUDA graph holds
pointers, never values. The launch goes to torch's current stream of the entry's device. That device need not be the
caller's current one: the compile and the launch run under the entry's device, and the caller's current device is
restored afterwards. There is no workspace; a warm call allocates nothing and reads no device memory to the host.

| operand | dtype | shape | notes |
| --- | --- | --- | --- |
| `q`, `gate` | bf16 | `[tokens, HEADS, HEAD_DIM]`, tokens 1 .. 1024 | views: the last axis contiguous, the other two strides whole 16-byte vectors; read only |
| `k`, `v` | bf16 | `[tokens, KV_HEADS, HEAD_DIM]` | as `q` |
| `q_weight`, `k_weight` | bf16 | `[HEAD_DIM]` | the zero-centred norm weights (the kernel applies `1 + weight`); read before the wait under `pdl` |
| `cos_sin` | bf16 | `[P, ROTARY_DIM]`, contiguous | per position `ROTARY_DIM / 2` cosines, then `ROTARY_DIM / 2` sines |
| `positions` | the entry's `position_dtype` | `[tokens]` (`"1d"`) or `[3, tokens]` (`"3d"`) | the token axis contiguous; the row stride of a `[3, tokens]` tensor is free |
| `k_pool`, `v_pool` | bf16 | `[pages, KV_HEADS, page_size, HEAD_DIM]` | the last axis contiguous, the other strides whole 16-byte vectors; written at live slots only |
| `slot_mapping` | the entry's `slot_dtype` | `[n]`, `0 <= n <= tokens`, contiguous | one flat slot per token; a token at or past `n` writes no pool row |
| `q_out`, `gate_out` (optional) | bf16 | `[>= tokens, HEADS, HEAD_DIM]`, contiguous | the first `tokens` rows are written |

Every bf16 operand starts at a 16-byte aligned address, the integer operands at a multiple of their element size;
every operand lives on the entry's device. No written operand (`q_out`, `gate_out`, the pools) may share a byte with
another operand. Two views that interleave in one allocation -- the two halves of a packed cache -- are served: the
check proves that they share no byte. A layout the check cannot separate is declined. A pool is written by one
thread block per head row, so its own rows may not share a byte either: a pool view with an expanded axis or with
rows that overlap by a part is declined.

**Caller contracts the host cannot check** (they are device data, and the entry never reads device memory): two
tokens of one call do not share a non-negative slot (two thread blocks would write one pool row); every position is a
row of `cos_sin`. A NaN or infinite element makes its own head row non-finite; every other row is unaffected.

**Output ownership is per output.** `q_out` / `gate_out` the caller passes are the caller's and are returned as
given; rows past the call's tokens are never written. An output the caller does not pass is the entry's: one pair per
token count, allocated by `compile(tokens)` / `outputs(tokens)` or at the first call of that token count. Call
`compile(tokens)` for every token count before a CUDA-graph capture. Entry-owned outputs are valid until the entry's
next call at that token count from ANY host thread. An entry-owned output is an operand of the launch like a passed
one: a call whose other operand overlaps an entry-owned output that the call does not pass is declined with
`ValueError` before the launch. The pools are always the caller's.

**A launch is ordered behind the stream work of an entry-owned pair's allocation.** The pair is allocated on torch's
current stream of the entry's device at that moment. The entry does not initialize it, but that stream can still write
it: under deterministic algorithms torch fills it, and a recycled block can carry a write the stream still owes it.
Every launch that writes bytes of the pair waits for that work on whatever stream it runs and however the tensor
reached the call: the entry's own output, a tensor taken with `outputs(tokens)` and passed back as `q_out` /
`gate_out` or as a pool, whole or as a view, at this token count or another, by this entry or by another entry that
orders its launches the same way (today `AddNormQuantDecode`, `AttentionPrepareDecode` and `AttentionGateQuantDecode`);
the test is one of addresses. On the allocating stream that is stream order; on another stream the launch
stream waits for an event recorded behind the allocation, without a host wait and without synchronizing a stream or
the device, so `compile(tokens)` may run on any stream. A captured call that finds the work still pending waits for
that one event on the host before its launch (a capturing stream cannot wait for an event recorded outside its
capture): set-up, never part of a replay, and not needed for a pair allocated before the capture on the capturing
thread's stream. Once a call has seen the work complete nothing is kept of it; while it is pending on the device,
every call of such an entry on that device reads the device's records. What stays the caller's: the operands a
launch only reads, and the caller's own reads and writes of a returned tensor outside a call. `outputs(tokens)`
returns a copy of the entry's table (the same tensors) without waiting.

Everything the entry does not serve is declined by name. At construction, without torch and without the DSL:
`TypeError` for a `geometry` that is not an `AttentionPrepareGeometry` of `int` fields or a launch knob that is not a
`bool`; `ValueError` for a non-positive geometry field, an `eps` that is not a finite positive Python `float` or
`int`, a `positions` outside `("1d", "3d")`, a missing, misplaced or inconsistent `mrope_section`, a `page_size` that
is not a positive power-of-two `int`, an integer dtype name outside `("int32", "int64")`, `pdl_release=True` without
`pdl=True`; `NotImplementedError` for a `HEAD_DIM` or `ROTARY_DIM / 2` that is not a power of two, a geometry other
than `(24, 4, 256, 64)` and a page size other than 64 or 16. Before every launch: `TypeError` for a non-tensor, a
bf16 operand of another dtype, or a `positions` / `slot_mapping` tensor whose integer dtype is not the entry's;
`ValueError` for a shape, a stride that is not whole 16-byte vectors, an alignment, a device mismatch, an overlap (a
written operand with another operand, or a pool's own rows), a
slot vector longer than the call, an empty call, and everything the paged-pool contract declines about the pools and
the slot vector (`cudnn.gated_attention_block.cache_write_contract`, with the entry's page size: a pool of another
page size is named; that contract raises `ValueError` for a non-tensor and for a dtype too); `NotImplementedError` for
a token count above 1024, a `cos_sin` that is not bf16 (another table dtype is another byte class), an operand that
is not contiguous where the kernel addresses flat, a compute capability outside 10.x, and the CuTe DSL gates (not
installed, below 4.8.0, without the part's target, or with a compiler that cannot assemble it). The paged-pool
contract's package imports the CuTe DSL: on a host without the DSL a call that reaches the pool checks gets the typed
decline that names the missing DSL, after every check in front of them has run.

## Numerics contract

The sum tree and the rounding chain are the contract, not a tolerance. Per head row, in fp32 on the bf16 operands:

1. One thread block per (token, head) of `HEAD_DIM / 2` threads in full warps; a thread owns two contiguous
   elements. Its partial sum of squares is `fma(x0, x0, x1 * x1)`: one multiply and one fused multiply-add.
2. The row total is ONE fixed tree whose shape depends on the head width alone: inside each warp a butterfly with
   the offsets 16, 8, 4, 2, 1; then over the warp sums (4 at `HEAD_DIM` 256) the offsets 2, 1.
3. `inv = rsqrt(total / HEAD_DIM + eps)` with the part's approximate full-range division (`div.full.f32`) and its
   approximate reciprocal square root with flush-to-zero (`rsqrt.approx.ftz.f32`), one addition between them.
4. `n = bf16( (x * inv) * (1 + weight) )`: `1 + weight` is one fp32 addition, then two multiplications and ONE
   rounding to bf16 (nearest-even). This bf16 value is the normalized value of record: the pass-through elements
   store it and the rotation reads it.
5. The rotation: each rotated value is one fp32 multiply (the rounded product) and one fused multiply-add of two
   bf16 values and that product, rounded once to bf16: `fma(n[r], cos, -(n[r + HALF] * sin))` for the first half;
   for the second half `fma(n[r + HALF], cos, n[r] * sin)` under `"3d"` and `fma(n[r], sin, n[r + HALF] * cos)`
   under `"1d"`. A product of two bf16 values is exact in fp32 unless it underflows, so the two orders of the second
   half write the same bytes on every operand whose products are normal fp32 values; where a product underflows
   (bf16 subnormals, magnitudes near the bottom of the fp32 range) they differ in the sign of a zero or in a
   subnormal's last bit. Each position form keeps one order.
6. The gate rows and the value rows are bit copies.

What changes bytes: the geometry (the tree and the divisor), `eps`, the position form with its section (which
position row an index reads; under underflow also item 5), and the operand values. What never changes a byte: the
token count, a token's place in the call, the other tokens, the page size and the integer dtypes (they select
addresses and load widths), `pdl`, `pdl_release`, and who owns an output. A token alone writes the bytes it writes
inside a call of up to 1024 tokens.

Against a PyTorch chain with the same rounding points whose mean and root are evaluated in fp64 (the suite's twin),
every pass-through element is within one bf16 unit, the unit taken at the larger of the two magnitudes, and every
rotated element inside `unit(out) + |cos| unit(n1) + |sin| unit(n2)` (its two normalized operands may each be one
unit off): the root is approximate, so the kernel's normalized value differs from the twin's by a few parts in ten
million before its rounding. The gate rows, the value rows and every pool row no live slot names are bitwise.

The bytes follow the part's `div.full` and `rsqrt.approx.ftz` units. Equal bytes across GPU architectures are not
part of the contract (see [determinism](/operations/determinism)).

## Launch knobs and the caller's obligations

Both knobs default to off and neither changes a byte (the suite compares their bytes).

- `pdl=True`: the launch carries the programmatic-stream-serialization attribute and the kernel issues
  `griddepcontrol.wait` AFTER it has loaded its two weights and BEFORE it reads the slot, the head rows, the copied
  rows, the positions or the table, and before it writes anything. Obligation: `q_weight` and `k_weight` are not
  written by any launch that may still be running when this one starts (they are constants of the layer).
- `pdl_release` (`None` follows `pdl`; `True` needs `pdl=True`): `griddepcontrol.launch_dependents` as the kernel's
  FIRST statement, in front of its own wait. Obligation: every launch that can start before this one completes
  issues its own wait before it reads `q_out`, `gate_out` or the pools and before it writes any operand of this
  call.

`pdl=True` alone is release + wait; `pdl=True, pdl_release=False` is wait only. The suite pins both positions in the
kernel's text for every form and target: the release is in front of every global access, the load of the weights is
the only global access in front of the wait, and every other load and both stores follow it. On a device it runs
the step behind a predecessor that releases its dependents before it writes the slab, the positions and the slot
vector, and the release with dependent launches on the stream that wait and copy the outputs; each eager and
captured.

## Launch facts

The entry states what a token count launches without launching it: three read-only answers, with `cfg_for`'s
argument.

| fact | answer |
| --- | --- |
| `kernel_symbols_for(rows)` | the symbol of the entry's compiled kernel as a one-element tuple; empty until the entry is compiled. One compiled kernel serves every token count. The symbol carries the form directly behind the kernel's name -- `cudnn_kernel_frost_attention_prepare_h24_4_d256_r64_3d_11_11_10_p64_i64_i64_plain_eps358637bd_...`: the geometry, the position form with its section, the page size, the position and slot dtypes, the launch arm and the fp32 bits of `eps` -- so two forms never share a symbol |
| `form_for(rows)` | the form as plain data: the nine fields of the entry's config (`GEOMETRY` as a nested `dict`, `EPS`, `POSITIONS`, `SECTION`, `PAGE_SIZE`, `POSITION_DTYPE`, `SLOT_DTYPE`, `PDL`, `PDL_RELEASE`). The compile key is the config alone. A fresh `dict` on every call |
| `launch_shape_for(rows)` | `(grid, block, shared_memory)`: one thread block per (token, head) over the query and the key / value heads, `HEAD_DIM / 2` threads, one fp32 slot of shared memory per warp -- `((3, 28, 1), (128, 1, 1), 16)` for three tokens. A profiler's kernel record of the launch reads the same three values |

A token count outside 1 .. 1024 is a typed decline (`ValueError` for one that is not a positive `int`,
`NotImplementedError` past 1024), and none of the three compiles, launches or touches device memory.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x; validated on cc 10.0. The kernel's rendering is pinned for the `sm_100a`, `sm_103a` and `sm_107a` targets: cc 10.3 and cc 10.7 are admitted on that basis and have not run on a device, and any other 10.x minor is admitted by the range alone. 9.0 and 12.x are declined by type |
| tokens | 1 .. 1024 per call (runtime argument); 1025 and up declined by name. A token is 28 thread blocks of 128 threads and sixteen such blocks share an SM, so one scheduling wave holds 84 tokens on a 148-SM part: 1024 tokens are more than twelve waves |
| geometry | `(24, 4, 256, 64)`, fixed at construction; another geometry is declined by name (each head width is its own sum tree) |
| forms | two position forms, two page sizes, two integer dtypes each for positions and slots; three launch arms (plain, wait, release + wait) |
| dtypes | bf16 operands and outputs; a bf16 rotary table; int32 or int64 positions and slots; fp32 arithmetic, rounded once to bf16 per stored value |
| layouts | strided `[tokens, heads, HEAD_DIM]` views and `[pages, heads, page_size, HEAD_DIM]` pools with a contiguous last axis and 16-byte whole outer strides; contiguous weights, table and outputs; no written operand sharing a byte with another, no two rows of a pool sharing a byte |
| determinism | bitwise replay and batch invariance (the [status page](/operations/determinism)) |
| DSL | nvidia-cutlass-dsl >= 4.8.0. On cc 10.0 / 10.3 the CUDA 12.9 and CUDA 13.4 compiler flavours of 4.8.0 render the same PTX body and are both served (on cc 10.0 both ran and wrote the same bytes); on cc 10.7 the compiler must be CUDA 13.4 or later (the CUDA 12.9 flavour rejects the `sm_107a` target and is declined by name before JIT) |

Tests: `test/python/attention_prepare/frost/test_attention_prepare_decode.py` (the construction declines with the
three construction facts, the binding declines, the overlap rule on interleaved views and on a pool's own rows, the config's limits and
tables, the rendering pinned per target, the launch facts against the configuration arithmetic and against the
profiler's record of a launch, the device cells against the twin above, the fused product of each rotary half
bitwise on rows whose products underflow, edge rows, the slot predicate, output
ownership, the bit-neutral arms, the wait behind a predecessor that released early, the release beside dependent
launches, CUDA-graph replay, an entry on a device that is not the caller's current one) and
`test_attention_prepare_decode_determinism.py` (replay, batch invariance, goldens). Source:
`python/cudnn/attention_prepare/frost/`. The kernel follows the thread shape and the arithmetic of a public
Apache-2.0 kernel; its module header and `THIRD_PARTY_LICENSES.txt` name the source.
