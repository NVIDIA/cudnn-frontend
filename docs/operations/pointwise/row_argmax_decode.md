# Row Argmax Decode Step (SM100-class)

**Experimental.** `cudnn.RowArgmaxDecode` (also `cudnn.row_argmax.frost.RowArgmaxDecode`) picks, per row of a bf16 slab, the
column of the row maximum -- or the entry of an id table at that column -- at decode-time row counts, as ONE CuTe DSL launch: the
greedy choice over the logits of a restricted vocabulary whose column order is not the id order.

```text
per row of x:
column  = the LOWEST column whose value is greater than -inf and not below any other value of the row;
          0 when the row holds no such value (every element is -inf or a NaN)
out[r]  = column                 without an id table   (int32)
        = ids[column]            with one              (a bit copy, in the table's dtype)
```

`x` is `[rows, width]` bf16 with `rows` in 1 .. 1024 (a runtime argument) and `width` 65536 (fixed at construction). `ids` is
absent, or an int32 or int64 table of `width` elements. The call returns `out`: one result per row, int32 without a table, else
the table's dtype. Compute capability 10.x; a CuTe DSL >= 4.8.0. Inference only: there is no backward.

It is a decode-step entry. One neighbour of this library does related work and is not the same operation: an argmax written as
a [custom reduction](/operations/CustomReduction) of the graph API (`cudnn.reduction_modifiers.argmax`) is a graph node with its
own tie and NaN policy and an index stream as its result. Use it inside a graph; use this entry as a direct call for the rows of
a decode step, 1 .. 1024 of them, with the id-table lookup in the same launch.

## Numerics contract

The rule is the contract, and it is exact: compares, selects and bit copies, no arithmetic on a value and no rounding. The result
is one value on every part, in any evaluation order.

1. A value is eligible when it is greater than `-inf`. A NaN is not eligible and is never chosen; it does not hide another
   element either.
2. Among the eligible values the largest one decides; `+0` and `-0` are equal.
3. Of the columns that hold the largest eligible value the LOWEST one is the row's column.
4. A row without an eligible value -- every element is `-inf` or a NaN -- gives column 0.
5. Without a table the result is the column as int32; with a table it is the table's element at the column, copied bit for bit.

On rows without a NaN the column is `torch.argmax`'s. What changes a result: the values of the row and, with a table, the table.
What never does: the row count, a row's position, the other rows, the row stride of the slab, and who owns the output. A row
alone gives the result it gives inside a call of up to 1024 rows.

## Usage

```python
import torch
import cudnn

dev = torch.device("cuda", 0)
argmax = cudnn.RowArgmaxDecode(dev, 65536)       # one entry per width
argmax.compile(rows)                             # plan time: typed declines, DSL gates, the compiles, outputs(rows)
tokens = argmax(logits, ids)                     # ids int64 [65536]: tokens int64 [rows]
columns = argmax(logits)                         # no table: columns int32 [rows]
tokens = argmax(logits, ids, out=my_tokens)      # or into a caller-owned vector
```

The table kind -- no table, an int32 table or an int64 table -- is a fact of the call. It is a load and a store width of the
kernel, so each kind is one compiled kernel. `compile()` needs no operand and compiles all three, so that no later call compiles
whatever table it binds; a call on an entry that was not compiled compiles its own kind on first use. Every compiled kernel serves
every row count: the operands are passed as addresses, the row stride of `x` and the row count are runtime arguments, so no shape
or stride enters a compiled kernel and a captured CUDA graph holds addresses, never values. The launch is ordinary -- there is no
launch knob, no dependent-launch wait and no release -- and goes to torch's current stream of the entry's device. That device
need not be the caller's current one: the compile and the launch run under the entry's device, and the caller's current device is
restored afterwards. There is no workspace; a warm call allocates nothing and reads no device memory to the host.

| operand | dtype | shape | notes |
| --- | --- | --- | --- |
| `x` | bf16 | `[rows, width]`, rows 1 .. 1024 | read only; a row contiguous; the row stride free in whole 16-byte vectors (a view of a wider slab binds as it is); 16-byte aligned |
| `ids` (optional) | int32 or int64 | `[width]` | read only; contiguous; aligned to its element |
| `out` (optional) | int32 without `ids`, else the dtype of `ids` | `[>= rows]` | contiguous; aligned to its element; elements past `rows` are never written |

Every operand lives on the entry's device, and `out` may not share a byte with `x` or `ids`.

**Output ownership.** An `out` the caller passes is the caller's and is returned as given. Without it the entry returns its own
vector of the row count and the result's dtype: per row count one int32 and one int64 vector of `rows` elements, allocated
together by `compile(rows)` / `outputs(rows)` or at the first call of that row count without `out`. Call `compile(rows)` for
every row count before a CUDA-graph capture; it may run on any stream. The allocation of a pair is stream work of torch's current
stream of the entry's device at that moment (the entry does not initialize the pair, but under deterministic algorithms torch
fills it, and its blocks can be recycled ones that stream still owes a write), and every launch that writes bytes of the pair is
ordered behind that work on whatever stream it runs, however the tensor reached the call: by default, or taken with
`outputs(rows)` and passed back as `out`, whole or as a view, at this row count or another. The launch stream waits for an event
recorded behind the allocation; the host does not wait, and no stream and no device is synchronized. A capturing stream cannot
wait for an event recorded outside the capture, so a captured call that finds that work still pending waits for that one event on
the host before its captured launch. A pair allocated inside a stream capture is not ordered: allocate before the capture.
`outputs(rows)` returns the pair without waiting, as a copy of the entry's table: a caller that reads or writes the returned
tensors itself, outside a call of the entry, orders that itself. Entry-owned outputs are valid until the entry's next call at
that row count from ANY host thread: stream order does not serialize host calls, so two host threads that share one entry pass
their own `out` or build one entry each. The per-row-count pairs are never freed, so a caller that sweeps many row counts passes
its own `out`.

Everything the entry does not serve is declined by name: `TypeError` for a non-tensor, a dtype (`x` bf16; `ids` int32 or int64;
`out` the result's dtype) or a `width` or a `rows` of `compile` / `outputs` that is not an `int`; `ValueError` for a width outside
the kernel's structure (a multiple of 8 in 8 .. 65536), a shape, a row stride of `x` that is not whole 16-byte vectors, an
alignment, a device mismatch, an `out` that overlaps `x` or `ids`, or an empty call; `NotImplementedError` for a structurally
valid width other than 65536, a row count above 1024, an operand that is not contiguous where the kernel addresses flat, a
compute capability outside 10.x, and the CuTe DSL gates (not installed, below 4.8.0, without the part's target, or with a compiler
that cannot assemble it). The operand checks import no DSL module and run before every launch; the architecture gate and the DSL
gates run before the kernel module is imported.

## Launch facts

The entry states what a row count launches without launching it: three read-only answers, with `cfg_for`'s argument.

| fact | answer |
| --- | --- |
| `kernel_symbols_for(rows)` | the symbols of the kernels the entry has compiled, in compile order: one per compiled table kind, none until a kind is compiled. Every compiled kernel serves every row count. A symbol carries the form directly behind the kernel's name -- `cudnn_kernel_frost_row_argmax_w65536_int64_...`: the width and the table kind -- so two forms never share a symbol |
| `form_for(rows)` | the form as plain data: the one field of the entry's config (`WIDTH`). The table kind is a fact of a call's operands, not of a row count. A fresh `dict` on every call |
| `launch_shape_for(rows)` | `(grid, block, shared_memory)`: one thread block per row, 1024 threads, 256 bytes of shared memory -- `((3, 1, 1), (1024, 1, 1), 256)` for three rows, whatever the table kind. A profiler's kernel record of the launch reads the same three values |

A row count outside 1 .. 1024 is a typed decline (`ValueError` for one that is not a positive `int`, `NotImplementedError` past
1024), and none of the three compiles, launches or touches device memory.

## The kernel

One thread block per row; 1024 threads in 32 full warps; a thread owns eight 16-byte vectors of its row -- 64 columns, the
vectors `thread + 1024 * pass` -- and scans them in ascending column order with a strict ordered compare and two selects per
element, from the start value `(-inf, column 0)`. The one value per thread then goes through the row primitive of the tile
library (`cudnn.frost.tile_dsl.rowwise.block_reduce`, `kind="max_index"`): a warp maximum, the lowest index that holds it, one
shared-memory word per warp for each, ONE thread-block barrier and a second warp level. Thread 0 reads the table's element at the
winning column and writes the row's result. The shared-memory words are written once per launch, so nothing is refilled and
nothing is released; the kernel has no other synchronization object.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x; validated on cc 10.0. The kernel's rendering is checked for the `sm_100a`, `sm_103a` and `sm_107a` targets, and from the kernel's entry on the `sm_103a` and `sm_107a` renderings are the `sm_100a` text: cc 10.3 and cc 10.7 are admitted on that basis and have not run on a device, and any other 10.x minor is admitted by the range alone. 9.0 and 12.x are declined by type |
| rows | 1 .. 1024 per call (runtime argument); 1025 and up declined by name. Two thread blocks of 1024 threads share an SM, so one scheduling wave holds 296 rows on a 148-SM part and 424 on a 212-SM part: a call above that crosses waves. The kernel holds no state between thread blocks |
| width | 65536, fixed at construction; another multiple of 8 up to 65536 is declined by name |
| forms | one kernel body; three table kinds (none, int32, int64), one compiled kernel each |
| dtypes | bf16 `x`; int32 or int64 `ids`; `out` int32 without a table, else the table's dtype |
| layouts | `x` with a free row stride of whole 16-byte vectors; contiguous `ids` and `out`; `out` overlapping neither `x` nor `ids` |
| determinism | bitwise replay and batch invariance (the [status page](/operations/determinism)); the result is exact, so it is the same on every part |
| DSL | nvidia-cutlass-dsl >= 4.8.0. On cc 10.0 / 10.3 both compiler flavours of the DSL (CUDA 12.9 and CUDA 13.4) assemble the kernel and are served; on cc 10.0 both ran under CuTe DSL 4.8.0 and gave the same results, and CuTe DSL 4.8.1 ran with CUDA 13.4. On cc 10.7 the compiler must be CUDA 13.4 or later (the CUDA 12.9 flavour rejects the `sm_107a` target and is declined by name before JIT). Other DSL versions are served unverified |

Tests: `test/python/row_argmax/frost/test_row_argmax_decode.py` (host declines, the config's limits and tables, the launch
facts, the reference against hand-written rule rows, the order of a launch behind an entry-owned allocation on stand-ins, the
rendering per target; on a device every byte of the result against the reference for the three table kinds, the rule rows,
`torch.argmax` on rows without a NaN, strided slab views, output ownership, one compile for every row count, CUDA-graph replay,
a warm launch behind an allocation that is pending on another stream, the profiler's record of a launch) and
`test_row_argmax_decode_determinism.py` (replay, batch invariance, locality, goldens). Source: `python/cudnn/row_argmax/frost/`.
