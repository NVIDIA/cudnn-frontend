# Causal Conv1d

`cudnn.ops.causal_conv1d` is the model-facing full-sequence operation. Its
tensor and state semantics match the commonly consumed `causal-conv1d`
interface:

- `x[B, D, T]`
- `weight[D, W]` and optional `bias[D]`
- reserved `seq_idx[B, T]` compatibility keyword (currently only `None` is
  accepted)
- optional packed `cu_seqlens[N + 1]` (mutually exclusive with `seq_idx`)
- dense state: optional `initial_states[B, D, W - 1]` and returned
  `final_states[B, D, W - 1]`
- packed state: optional `initial_states[N, D, W - 1]` and returned
  `final_states[N, D, W - 1]`, where `N = len(cu_seqlens) - 1`

Model implementations that already own contiguous `[B, T, D]` storage pass
its `transpose(1, 2)` view. The public shape stays `[B, D, T]`, while the
current native implementation receives the original contiguous storage
without a copy.

```python
from cudnn.ops import causal_conv1d

output = causal_conv1d(x, weight, bias, activation="silu")
output, final_states = causal_conv1d(
    x,
    weight,
    bias,
    initial_states=initial_states,
    return_final_states=True,
    activation="silu",
)
```

The returned width-four ``final_states[B, D, 3]`` can be passed directly to
``cudnn.ops.causal_conv1d_update`` for the next token. The update operation
mutates that state in place; no layout conversion is required.

The result is an ordinary Tensor or tuple. Kernel compilation, architecture
class names, schedule choice, workspace allocation, intermediate result
wrappers, and CUDA stream handles are backend details.

The current optimized route covers dense and `cu_seqlens`-packed BF16-activation
width-four SiLU forward and backward, including mathematical `W - 1` initial
and final state. Weights may be BF16, or FP32 for the bias-free contract used by
GLM linear-attention blocks; an FP32 weight with bias declines explicitly. The
implementation adapts state to a private full-width buffer without exposing its
storage or layout. `seq_idx` remains reserved and declines explicitly until a
matching backend exists.

Packed `cu_seqlens` values are validated on the device to avoid a host read.
Malformed metadata—including a first offset other than zero, a final offset
other than the runtime token total, non-increasing offsets, or an empty
sequence—executes a device trap. The resulting sticky CUDA failure is not a
recoverable Python exception; the process must discard that CUDA context before
continuing GPU work.

The current backward implementation accumulates dweight with FP32 atomics, so
its dweight result is not bitwise reproducible across launches, even when
PyTorch deterministic algorithms are enabled. A deterministic dweight route
is not currently exposed by this semantic operation.
