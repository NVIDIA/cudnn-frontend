# Launch plans: repeated calls launched from C++

A frontend Python API spends host time on every call before its kernels launch:
argument checks, cache keys, graph and engine dispatch, workspace carving, and the
DSL's argument conversion. For small and medium problems that host time is longer
than the kernels, so the GPU waits on Python. A *launch plan* records what one call
does and replays it from C++ for the later calls with the same arguments, so the
API's Python runs once per argument signature instead of once per call.

Launch plans live in `cudnn.launch_plan` (Python) and `_compiled_module._LaunchPlan`
(C++, `python/launch_plan.cpp`). They need no tvm-ffi or framework headers; kernels
are called through the tvm-ffi C ABI of the `libtvm_ffi.so` the process loaded, which
has to report the version (0.1) that ABI was written for. Python hands each kernel to
C++ through a tvm-ffi call, so a plan holds its own reference to it. Where that
cannot work (another tvm-ffi ABI, another platform than Linux, a CuTeDSL whose call
path is not the one recordings watch), nothing is planned and every call runs the
function.

## Using it in an API

```python
from cudnn.launch_plan import launch, planned


@planned
def my_op(x, *, alpha):
    out = torch.empty_like(x)
    scratch = torch.zeros(n, device=x.device)  # dropped at return: per-call scratch
    launch(compiled, x, scratch, alpha, stream)  # every kernel launch goes through launch()
    launch(compiled2, scratch, out, stream)
    return out
```

- `@planned` wraps the function. Calls are keyed by each tensor argument's dtype,
  shape, strides and device and by the other arguments' values and types; stream arguments are
  bound per call, not keyed. A `CUstream` or an object with `__cuda_stream__` is
  recognized anywhere; a parameter that may carry a plain integer handle must be
  named: `@planned(stream="current_stream")`. The kernel itself must receive the
  stream as a `CUstream` (or `__cuda_stream__` object): an `int` kernel argument is
  recorded as a constant, since an integer cannot be told apart from a stream by value.
- `launch(fn, *args)` is `fn(*args)` for a compiled kernel (a `tvm_ffi.Function`, or a
  CuTeDSL function compiled with `--enable-tvm-ffi`). Outside a recording it adds one
  thread-local check.

## What happens per call

| Call for a signature | What runs |
|---|---|
| The first 15 | The function, as before (the first one compiles and fills its caches). The first 8 are counted by the first argument's shape alone; from the 9th the call is keyed. |
| The 16th | The function, recorded: each `launch()` (the tvm-ffi function and the positional arguments it receives), each tensor allocated and filled, and the result. |
| Later | Python allocates the returned tensors and one scratch buffer and passes their addresses, the argument tensors' addresses and the stream to `_LaunchPlan.execute`, which runs the fills (`cuMemsetD*Async`) and kernel calls in the recorded order. The function's Python does not run. |

A recording costs about as much host time as 16 planned calls save
(`launch_plan.HOT_CALLS`), so a signature that is not called that often, as with
sequence lengths that keep changing, never pays for one. Nor does it pay for a
key: until its first argument's shape has been seen 8 times, a call only counts
that shape (under a microsecond).

Every kernel argument of the recording is bound to where it comes from in the call:

- an argument tensor, or a view into one (by byte offset, dims, strides and dtype);
- a tensor the call allocates: one it returns, or scratch it drops (placed in a
  per-call arena);
- the stream (the call's stream argument, or the current stream);
- a constant value: an int, float or bool.

No argument is bound by an address of the recorded call.

## When a call is not planned

The recording declines, and the signature keeps running the function, when the call
does anything the plan could not reproduce:

- it takes a host tensor (its values could decide the launches unseen);
- a torch operation on a CUDA tensor other than an allocation, a fill, or a view
  (for example arithmetic, a copy, or `.item()`);
- a CuTeDSL kernel launched other than through `launch()`;
- a kernel argument that is none of the above: a tuple, or a tensor or address the
  call was neither passed nor allocated (a buffer the API keeps across calls);
- a tensor the call allocates and keeps;
- an allocation on a stream other than the current one or a stream argument.

A kernel reloaded by the persistent compile cache (`cudnn.frost.compiled_cache`) is
watched (`launch_plan.watch`), so a direct call of it declines too; any other raw
`tvm_ffi.Function` called directly cannot be seen. A planned function must launch
every kernel through `launch()`, and what it does must follow from its arguments
(and the environment variables it names): a replay does not rerun its Python.

A later call runs the function instead of the plan whenever the function could
have behaved differently on it, since a replay skips the API's checks and choices:

- its argument tensors are less aligned than the recorded call's (up to 256 bytes);
- its argument tensors overlap each other differently from the recorded call's, or
  overlapping ones sit at another byte offset from each other (an API may reject an
  output that aliases an input, pick a packed layout, or address one argument
  through another).

The key also holds the current CUDA device and the values of the environment
variables the function reads per call (`planned(env=("NAME", ...))`). A call keys the
same however its arguments are passed (by position, by keyword, or left at their
defaults); JAX arrays, tensors inside lists, and functions taking `*args` or
`**kwargs` are not planned.

Each planned function keeps its 256 most recently used signatures
(`CUDNN_FRONTEND_LAUNCH_PLANS_MAX`, read at import), so a workload whose shapes keep
changing cannot grow the plans without bound.

## Checking a planned function

`CUDNN_FRONTEND_LAUNCH_PLANS=verify` (read at import) checks plans instead of using
them. Every call of a planned function runs the function, then records it and
replays the recording on copies of the arguments at other addresses. The replay has
to return, and leave in its arguments, bit for bit what the call did, or the call
raises. New tensors are zero-filled in this mode so bytes no kernel writes compare
equal. A call inside a CUDA graph capture is not checked. At exit the process prints
how many calls matched and why others were not planned. Run a planned API's tests in this mode: it is how a launch the recording
misses, or a result that does not follow from the arguments, shows up.

`CUDNN_FRONTEND_LAUNCH_PLANS=0` (read at import) turns plans off.
