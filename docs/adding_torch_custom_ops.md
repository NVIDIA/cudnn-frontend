# Adding Torch Custom Ops in cuDNN Frontend

Best practices for wrapping cuDNN graph ops as PyTorch custom ops with minimal CPU overhead.

## File location

Custom ops live in `python/cudnn/experimental/ops/`. Each op gets its own file
(e.g., `sdpa.py`, `rmsnorm.py`, `layernorm.py`, `moe.py`). Export from
`python/cudnn/experimental/ops/__init__.py`.

## Registration: use torch.Library, NOT @torch.library.custom_op

`@torch.library.custom_op` adds ~36 us per-call overhead vs ~10 us for direct
`torch.Library` registration (PyTorch issue #139500). Always use the direct API:

```python
import torch

_lib = torch.library.Library("cudnn", "DEF")

# Define schema — must list all tensor and scalar args with types
_lib.define(
    "my_op(Tensor x, Tensor w, float eps, bool training=False, "
    "Tensor? bias=None) -> (Tensor, Tensor)"
)

# Register CUDA implementation
def _my_op_impl(x, w, eps, training=False, bias=None):
    # ... build/cache graph, execute ...
    return output, aux

_lib.impl("my_op", _my_op_impl, "CUDA")

# Register fake (for torch.compile shape inference)
@torch.library.register_fake("cudnn::my_op")
def _my_op_fake(x, w, eps, training=False, bias=None):
    return torch.empty_like(x), torch.empty(...)

# Register autograd (if backward is needed)
torch.library.register_autograd("cudnn::my_op", _my_op_backward, setup_context=_my_op_setup_ctx)
```

User calls via: `torch.ops.cudnn.my_op(x, w, eps)` or wrap in a public function.

## Graph caching

Build graphs once per unique (shape, stride, dtype, config) tuple. Cache as module-level dict:

```python
_cache: Dict[tuple, tuple] = {}

def _make_cache_key(x, w, eps, has_bias):
    return (
        "my_op",
        tuple(x.shape), tuple(x.stride()), x.dtype,
        tuple(w.shape), tuple(w.stride()), w.dtype,
        eps, has_bias, x.device,
    )
```

**Include device in the key** — different GPUs may get different engine plans.

## Graph building pattern

Use `tensor_like()` for automatic shape/stride/dtype inference from DLPack tensors:

```python
graph = cudnn.pygraph(handle=handle, ...)
X = graph.tensor_like(x)
W = graph.tensor_like(w)
# ... chain ops ...
Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
graph.check_support()
graph.build_plans()  # also prepares variant pack template
```

## Execute: use sorted_ptrs path

Cache `uid_order` alongside the graph. Use `_execute_with_ptrs` instead of dict-based execute:

```python
if cache_key not in _cache:
    graph, ws_size = _build_graph(...)
    uid_order = graph._get_variant_pack_uids_sorted()
    _cache[cache_key] = (graph, ws_size, uid_order)

graph, ws_size, uid_order = _cache[cache_key]

# Allocate workspace per-call (PyTorch's caching allocator recycles efficiently)
workspace = torch.empty(max(ws_size, 1), device=x.device, dtype=torch.uint8)

# Build uid→tensor map, extract sorted ptrs
uid_to_tensor = {X.get_uid(): x, W.get_uid(): w, Y.get_uid(): y_out}
ptrs = [uid_to_tensor[uid].data_ptr() for uid in uid_order]
graph._execute_with_ptrs(ptrs, workspace.data_ptr(), int(handle))
```

**Do NOT cache workspace tensors** — they can race on different CUDA streams.
Allocate per-call; PyTorch's caching allocator recycles the allocation without
hitting cudaMalloc (~2 us overhead). This matches PyTorch's own conv/SDPA pattern.

## UID management

Use an IntEnum for explicit UIDs — makes the code self-documenting and cache keys stable:

```python
from enum import IntEnum

class _UIDs(IntEnum):
    X = 1
    W = 2
    BIAS = 3
    Y = 100
    # ...
```

Set UIDs explicitly when building the graph:
```python
X = graph.tensor(dim=..., stride=..., data_type=..., uid=int(_UIDs.X))
```

Or use `tensor_like` which auto-assigns UIDs in insertion order (1, 2, 3, ...).

## Handle management

Cache one handle per device. Call `set_stream` once per call (costs ~5 us):

```python
_handles = {}

def _get_handle(device):
    if device not in _handles:
        _handles[device] = cudnn.create_handle()
    cudnn.set_stream(handle=_handles[device], stream=torch.cuda.current_stream(device).cuda_stream)
    return _handles[device]
```

**Future optimization**: skip `set_stream` if stream hasn't changed since last call.

## Autograd pattern

```python
def _setup_context(ctx, inputs, output):
    x, w, eps, training, bias = inputs
    y, aux = output
    ctx.save_for_backward(x, w, y, aux)
    ctx.eps = eps
    # save non-tensor args as ctx attributes

def _backward(ctx, dY, d_aux):
    x, w, y, aux = ctx.saved_tensors
    dX, dW = torch.ops.cudnn.my_op_bwd(dY, x, w, y, aux, ctx.eps)
    return dX, dW, None, None, None  # None for non-differentiable args

torch.library.register_autograd("cudnn::my_op", _backward, setup_context=_setup_context)
```

## Public API wrapper

Provide a user-friendly function that matches PyTorch conventions:

```python
def my_op(x, w, eps=1e-5, bias=None):
    """cuDNN-accelerated my_op. Matches torch.nn.functional.my_op API."""
    # input validation
    # ...
    y, _aux = torch.ops.cudnn.my_op(x, w, eps, bias=bias)
    return y
```

## Performance checklist

- [ ] Use `torch.Library.define/impl`, not `@torch.library.custom_op`
- [ ] Cache graph + uid_order + workspace in module-level dict
- [ ] Use `graph._execute_with_ptrs(sorted_ptrs)` not `graph.execute(dict)`
- [ ] Use explicit UIDs (IntEnum) for stable cache keys
- [ ] Cache cuDNN handle per device
- [ ] Allocate workspace per-call (do NOT cache — stream safety)
- [ ] Consider `out=` param for output tensors in performance-critical paths
- [ ] Include device in cache key
- [ ] Test with tiny tensors to measure CPU overhead in isolation

## CPU overhead budget (per call, Blackwell release build)

| Component | Cost (us) | Notes |
|---|---|---|
| torch.ops dispatch + autograd | ~25 | PyTorch-side, unavoidable with torch.Library |
| `set_stream` | ~5 | pybind11 cross-language call |
| `torch.empty` per output tensor | ~2 each | consider `out=` or pre-alloc |
| cache key build + lookup | ~1.5 | tuple construction + dict hash |
| uid→tensor dict + list comp | ~1 | Python overhead |
| `graph._execute_with_ptrs` | ~19 | 1.7 us FE + 0.8 us varpack + 5.6 us backend |
| **Total (well-optimized)** | **~52** | vs ~18 us for native ATen ops |

The ~34 us gap vs native ATen is torch.ops dispatcher + autograd overhead.
For inference without torch.compile, calling `_my_op_impl` directly saves ~25 us.
