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

The default `backend="auto"` route covers dense and `cu_seqlens`-packed BF16-activation
width-four SiLU forward and backward, including mathematical `W - 1` initial
and final state. Weights may be BF16, or FP32 without bias (BF16 activations
with an FP32 depthwise filter); the FP32-weight epilogue is only tested without
bias, so an FP32 weight with bias declines explicitly. The implementation adapts
state to a private full-width buffer without exposing its storage or layout.
`seq_idx` remains reserved and declines explicitly until a matching backend
exists.

`final_states_out` must not share memory with `x`, `weight`, `bias`,
`cu_seqlens`, or `initial_states`. The final state is written after the forward
and the inputs are saved for backward, so an aliased output would corrupt them;
overlapping storage is rejected with `ValueError`.

For `backend="auto"`, packed `cu_seqlens` values are validated on the device to avoid a host read.
Malformed metadata—including a first offset other than zero, a final offset
other than the runtime token total, non-increasing offsets, or an empty
sequence—executes a device trap. The resulting sticky CUDA failure is not a
recoverable Python exception; the process must discard that CUDA context before
continuing GPU work.

The default backend honors `torch.use_deterministic_algorithms`. Its dweight
accumulates with FP32 atomics and is not bitwise reproducible across launches.
When deterministic algorithms are enabled, bias-free calls select a deterministic
dweight route (unique FP32 partials followed by a fixed-order reduction). Calls
with `bias` have no deterministic dbias route: they raise `RuntimeError`, or warn
and run with atomics under `warn_only=True`, mirroring PyTorch's own semantics.

## Cake backend

`causal_conv1d(..., backend="cake")` opts into the vendored deterministic Cake
kernels. `backend="auto"` retains the default route. The Cake path computes
width-four depthwise causal convolution followed by SiLU; tap 3 multiplies
the current token. Dense batches reset history at each batch boundary. Packed
calls use `cu_seqlens` and reset history at every sequence boundary.

```python
from cudnn.ops import causal_conv1d

# x_btd is contiguous BF16 [batch, tokens, channels].
y_bdt = causal_conv1d(x_btd.transpose(1, 2), weight, bias, "silu", backend="cake")
```

Requires CUDA toolkit 13.4+, Torch, TVM-FFI, and an available Cake route on
SM100a/SM103a/SM107a. Weight is contiguous FP32 or BF16 `[channels, 4]`; bias
is optional contiguous FP32 or BF16 `[channels]`. Channel count is divisible
by eight. Input storage must be contiguous BTD, passed as its BDT transpose;
addresses are 16-byte aligned. There are no layout copies or dtype casts.
Tokens, channels and sequence count must fit positive signed int32 values.
Empty inputs/sequences, `seq_idx`, initial state, and returned state are
unsupported. A route absent from an architecture's manifest raises at preparation.

Packed prefix sums are contiguous CUDA int32, start at zero, end at the total
token count, and increase strictly. A device validator checks these values
before every forward/backward launch. Invalid values cause an asynchronous CUDA
error; they are never read on the host.

Autograd is supported for native BTD gradients (BDT transpose at this API).
A broadcast or other non-native gradient raises with a layout error. Parameter
gradients have their parameter's dtype and use a fixed reduction order. The SiLU
weight-gradient derivative uses the recomputed FP32 preactivation. Double
backward is unsupported. The default backend's support and dispatch are unchanged.

### Prepared Cake execution

`cudnn.causal_conv1d_cake.api.CausalConv1dPlan(tokens, channels, sequences,
weight_dtype=torch.float32, bias_dtype=None, device=None)` prepares the exact
forward/backward route. Its inputs are contiguous `[tokens, channels]`, with
an explicit packed prefix tensor even for dense batches. It exposes:

- `get_workspace_size()`: bytes in a contiguous uint8 workspace.
- `forward(x, weight, cu_seqlens, out, workspace, *, bias=None, stream=None)`.
- `backward(dy, x, weight, cu_seqlens, dx, dweight, workspace, *, bias=None,
  grad_bias=None, stream=None)`.

Prepare once outside CUDA Graph capture. Execution writes caller outputs and
scratch, has no device allocation or host synchronization, and supports capture
and the selected Torch stream. Forward clears its work counters on that stream.
Every buffer is validated before launch; output/scratch storage must not overlap
any input or each other. Concurrent calls require separate scratch. The semantic
wrapper allocates outputs and scratch on the current stream.

The deterministic package is copied unchanged from
[miles-ext-kernels at f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce/miles_ext_kernels/cake/_vendor/cake_causal_conv1d_det_train),
with build manifests and file hashes in `_cake/_vendor/VENDORED_CAUSAL_CONV1D.json`.
No Miles or FlashInfer installation is required.

## SE backend

`causal_conv1d(..., backend="se")` selects the CuTe DSL SE implementation.
It supports autograd and `torch.compile`; `backend="auto"` keeps its existing
routing. CuTe DSL 4.8.0 or newer is required.

```python
# Dense: BF16 channel-last physical storage, SM103, width four, SiLU.
y = causal_conv1d(x_btd.transpose(1, 2), weight, bias, "silu", backend="se")

# Packed: x_tc concatenates the sequences without padding.
y, final = causal_conv1d(
    x_tc.T[None], weight, bias, "silu", backend="se",
    cu_seqlens=offsets, initial_states=initial,
    return_final_states=True,
)
```

Packed calls support matching FP16, BF16, or FP32 activations and parameters,
filter widths 2 through 8, and identity or SiLU activation. Channels must be
positive and divisible by eight. The existing packed schedules admit
SM80/86/89/90/100/103/120. Initial and final history contain exactly `W - 1`
samples. Input, weight, bias, and initial-history gradients are supported;
double backward is unsupported. Strided packed inputs use the existing scalar
or tiled schedules. The wrapper preserves dense input memory formats when
returning its output.

Offsets are contiguous CUDA int32 values on the input device. They start at
zero, end at the token count, and are nondecreasing. Repeated offsets represent
empty sequences. An all-empty pack is supported. Offset values are checked
on the device without a host read; invalid values cause a device trap and
invalidate the CUDA context. Every replay reads the current offsets.

Dense SE currently admits SM103 BF16 width-four SiLU with matching parameters,
positive batch and token counts, and channel-last physical storage. It
prepares a contiguous `[4, channels]` filter for forward and saves that tensor
for backward. Broadcast output gradients are materialized at the allocating
autograd boundary. Dense initial/final history is not supported by this route;
use an equivalent packed call when history is needed.

For contiguous BF16, width-four SiLU packed backward on SM100/SM103, the persistent
schedule is selected at 8192 or more tokens and 768 or 2048 channels, with no
state gradients. Other cases retain the tiled schedules. Dense backward uses
the persistent schedule at 2048 channels and at least 100 million activation
elements, with more than one token per sequence. The kernel retains its input
and pending-gradient windows in registers across eight-token TMA stages.

The BF16 K4 SiLU parameter reduction on SM100/SM103 assigns each filter tap
and bias gradient to separate thread blocks. Each output retains its fixed
split-sum order. Small tiled calls cap the split count at the number of
64-token tiles; packs with 256 channels and 4096–8191 tokens use one split
per tile. FP32 parameter outputs retain FP64 final accumulation.

Packed parameter gradients and the persistent dense path use fixed-order
reductions. They are repeatable for a fixed configuration. Parameter gradients
can change bits when sequence packing or reduction grouping changes. The
small dense tiled path uses atomic parameter reductions and rejects strict
`torch.use_deterministic_algorithms(True)`; `warn_only=True` permits it with a
warning. These modes do not promise agreement with another backend's bits.

The BF16 persistent parameter path accumulates in FP32 without compensation.
Cake's corresponding path uses compensated intermediate sums and an FP64
final reduction. Performance comparisons therefore need separate numerical
checks; matching the output dtype does not establish equal worst-case
accumulation error. See the [determinism support page](../determinism.md) for
validated guarantees and the test coverage report for hardware-specific results.

Dense calls also require legal CUDA launch geometry: batch and the selected
forward token-tile count must each fit 65535, and channels must leave room for
signed int32 tile arithmetic. Unsupported geometry raises before compilation;
packed offsets avoid the dense batch-grid limit.
