# MoE + Expert Parallel API proposal

Status: public API stub, implementation proposal, and executable PyTorch
reference for both forward and backward. The API currently allocates
uninitialized output storage without launching a device kernel. Its numerical
comparison is therefore marked as a strict expected failure under
`test/python/fe_api/moe_ep`; remove that marker when backend execution is
connected.

The design removes workspace pointers, peer pointer mappers, streams, and
scheduler tuning from the semantic user interface.

## Decision summary

- The constructor contains static model, EP, capacity, and numerical choices.
- `__call__` contains only runtime tensors.
- Each rank supplies local tokens and local expert weights. Expert IDs in the
  routing table are global.
- Expert ownership is contiguous and uniform. EP rank `r` owns
  `[r * E_local, (r + 1) * E_local)`.
- `-1` is the only dropped/unused route value. Other out-of-range IDs are
  errors.
- The first half of FC1 is `gate`; the second half is `up`. SwiGLU is
  `silu(gate) * up`.
- `output_format` means the public, post-top-k-reduction `(T, H)` result. This is
  an extension of the MegaMoE kernel interface, whose public result is BF16.
- `combine_format` independently describes each per-route FC2 contribution on
  the EP return path. This corresponds to the `combine_quant`
  and `combine_sf` planes.
- BF16, MXFP8, and NVFP4 are supported for both choices. Quantized output is a
  data-plus-scale object; it is never represented as a scale-free PyTorch
  tensor.

The distinction between public output and combine traffic is intentional. If
"MXFP8/NVFP4 output" is meant only as a transport optimization, set
`combine_format` to that format and keep `output_format="bf16"`.

## Proposed API

The production class should be named `MoeEp`. The checked-in semantic
implementation is named `MoeEpReference` so it cannot be confused with a fused
kernel.

```python
class MoeEp:
    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        max_tokens_per_rank: Optional[int] = None,
        output_format: Literal["bf16", "mxfp8", "nvfp4"] = "bf16",
        combine_format: Literal["bf16", "mxfp8", "nvfp4"] = "bf16",
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        generate_c: bool = False,
    ) -> None: ...

    def __call__(
        self,
        activation: Tensor | BlockScaledTensor,
        fc1_weight: Tensor | BlockScaledTensor,
        fc2_weight: Tensor | BlockScaledTensor,
        topk_idx: Tensor,
        topk_weights: Tensor,
    ) -> (
        Tensor
        | BlockScaledTensor
        | tuple[Tensor | BlockScaledTensor, Tensor, Tensor]
    ): ...

    def backward(
        self,
        grad_output: Tensor,
        activation: Tensor | BlockScaledTensor,
        fc1_weight: Tensor | BlockScaledTensor,
        fc2_weight: Tensor | BlockScaledTensor,
        topk_idx: Tensor,
        topk_weights: Tensor,
        fc1_c: Tensor,
        route_metadata: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
```

An initialized `ep_group` enables EP. `None` deliberately means a one-rank
execution, even if a default distributed process group exists. This prevents an
operator from silently communicating on the wrong group.

### Constructor contract

| Argument | Meaning |
|---|---|
| `num_experts` | Global expert count `E`; must be divisible by EP size. |
| `hidden_size` | Model hidden dimension `H`. |
| `intermediate_size` | Post-SwiGLU dimension `I`; FC1 has `2 * I` columns. |
| `top_k` | Fixed routing width `K`, with `1 <= K <= E`. |
| `ep_group` | Process group whose group-relative rank determines expert ownership. |
| `max_tokens_per_rank` | Maximum local input tokens `T`; optional in the reference and required by a static-workspace implementation. |
| `output_format` | Encoding returned after top-k reduction. |
| `combine_format` | Encoding/rounding of each route contribution before top-k reduction. |
| `apply_topk_in_fc1` | Multiply the post-SwiGLU intermediate by the router weight before FC2; otherwise multiply FC2 output. |
| `gate_up_clamp` | If set, use `gate = min(gate, abs(limit))` and `up = clamp(up, -abs(limit), abs(limit))`. |
| `generate_c` | Training integration: additionally return `fc1_c` (raw pre-SwiGLU FC1 accumulator of every route this rank's experts processed) and its row-aligned `route_metadata`. |

The constructor validates static format alignment. MXFP8 output requires
`H % 32 == 0`; NVFP4 output requires `H % 16 == 0`. A production implementation
may also validate SM architecture, NVSHMEM availability, symmetric heap size,
and CUDA graph constraints here.

Scheduler settings such as `token_back_mode`, `group_hint`, `flag_batch`,
`epi_flag_batch`, and scheduler stages belong in an optional backend/tuning
object. They do not change the result and should not be positional API
arguments.

`fc2_in_kernel_topk_reduce` is different: BF16 atomic reduction can change the
rounding order. The first contract therefore specifies Form A semantics
(round each route to `combine_format`, reduce in FP32, encode once). Form B may
be used behind the API only if it satisfies the agreed tolerance; otherwise it
must become an explicit numerical mode rather than an invisible tuning flag.

### Forward tensor contract

Let `T` be this rank's token count and `E_local = E / ep_size`.

| Tensor | Logical shape | Required properties |
|---|---:|---|
| `activation` | `(T, H)` | BF16/FP tensor, or block-scaled along axis 1. |
| `fc1_weight` | `(E_local, H, 2I)` | Local experts only; block-scaled weights use axis 1. |
| `fc2_weight` | `(E_local, I, H)` | Local experts only; block-scaled weights use axis 1. |
| `topk_idx` | `(T, K)` | INT32 or INT64 global expert IDs; `-1` means unused. |
| `topk_weights` | `(T, K)` | Floating router/combine weights. They are not implicitly normalized. |

All tensors must be on one device. Quantized data and its scale tensor must
also share a device. Biases, shared experts, nonuniform expert placement, and
capacity-based route dropping are outside this first API.

### Return value

- BF16: a `torch.bfloat16` tensor with logical shape `(T, H)`.
- MXFP8/NVFP4: a `BlockScaledTensor` containing `data`, `scale`, `format`,
  `logical_shape`, and the scaled axis. `dequantize()` reconstructs a regular
  tensor.

With `generate_c=True`, the call instead returns
`(output, fc1_c, route_metadata)`; see the training-integration sections
below.

The return type is fixed by the constructor, so an individual module instance
does not change its output structure across calls.

### Backward call contract

`backward` requires the operator to be constructed with `generate_c=True`; it
consumes the forward stash and is executable today as
`MoeEpReference.backward` (the device implementation is pending, like
forward). It is a collective: every rank in `ep_group` must call it, because
gradients re-dispatch along the identical forward routes.

| Argument | Shape | Provided by |
|---|---:|---|
| `grad_output` | `(T, H)` | incoming gradient of the *dequantized* public output (all encodes are straight-through). |
| `activation`, `fc1_weight`, `fc2_weight`, `topk_idx`, `topk_weights` | as in forward | the framework re-supplies the same forward inputs; `topk_idx` deterministically regenerates the dispatch plan. |
| `fc1_c`, `route_metadata` | `(local_routes, 2I)`, `(local_routes, 4)` | the `generate_c=True` forward stash, passed back unchanged. |

Returns four FP32 tensors:

| Return | Shape | Meaning |
|---|---:|---|
| `grad_activation` | `(T, H)` | summed over this token's valid routes; gradients w.r.t. the dequantized activation values. |
| `grad_fc1_weight` | `(E_local, H, 2I)` | accumulated over every route this rank's experts processed, from all source ranks. |
| `grad_fc2_weight` | `(E_local, I, H)` | same accumulation domain as `grad_fc1_weight`. |
| `grad_topk_weights` | `(T, K)` | per-route router-weight gradient; exact zero at `-1` slots. |

The save-set, recompute rules, and numerical conventions behind this
signature are specified in "Saved tensors for backward" below.

### Example

```python
moe = MoeEpReference(
    num_experts=8,
    hidden_size=4096,
    intermediate_size=14336,
    top_k=2,
    ep_group=ep_group,
    max_tokens_per_rank=2048,
    combine_format="mxfp8",
    output_format="nvfp4",
)

# Each rank passes its own tokens and its contiguous E_local weight shard.
output = moe(activation, local_fc1, local_fc2, topk_idx, topk_weights)
assert output.logical_shape == (activation.shape[0], 4096)
output_bf16 = output.dequantize(torch.bfloat16)
```

## Mathematical semantics

For valid route `(t, k)` with global expert `e` and router weight `p[t, k]`:

```text
z[t,k]       = fp32(x[t]) @ fp32(W1[e])
gate, up     = split(z[t,k], I)
hidden[t,k]  = silu(gate) * up

if apply_topk_in_fc1:
    hidden[t,k] *= p[t,k]

expert[t,k]  = hidden[t,k] @ fp32(W2[e])

if not apply_topk_in_fc1:
    expert[t,k] *= p[t,k]

combine[t,k] = dequantize(quantize(expert[t,k], combine_format))
result[t]    = sum_k(combine[t,k])
output       = encode(result, output_format)
```

For BF16 combine, `quantize/dequantize` above means a BF16 round trip. Invalid
slots contribute exact zero. Accumulation across top-k slots is FP32 and the
public encoding is applied after reduction.

Moving the router weight across FC2 is algebraically equivalent in exact
arithmetic, but not necessarily after low-precision rounding. The option is
therefore semantic and is fixed in the constructor, matching the MegaMoE kernel.

## Block-scaled representation

The API uses logical, unswizzled scales. Backend-specific F8_128x4 swizzling is
an implementation detail performed while constructing tensor maps or staging
weights.

| Format | Payload | Scale | Block | Quantized axis |
|---|---|---|---:|---|
| BF16 | BF16 | none | n/a | n/a |
| MXFP8 | FP8 E4M3 | FP8 E8M0 | 32 | reduction/output axis |
| NVFP4 | packed FP4 E2M1, low nibble first | FP8 E4M3 | 16 | reduction/output axis |

MXFP8 scale calculation is:

```text
raw_scale = amax(block) / 448
scale = 2 ** ceil(log2(raw_scale))       # E8M0 round toward +infinity
data = e4m3(clamp(block / scale, -448, 448))
```

NVFP4 scale calculation is:

```text
raw_scale = amax(block) / 6
scale = e4m3(raw_scale)                  # round to nearest, saturate finite
data = e2m1(clamp(block / scale, -6, 6)) # two values per byte
```

E2M1 conversion in the reference uses round-to-nearest, ties-to-even. Logical
shapes may be padded to a complete block internally, but padding is not visible
through `logical_shape`.

Examples of scale shapes are:

| Logical tensor | MXFP8 scale | NVFP4 scale |
|---|---:|---:|
| activation `(T, H)` | `(T, ceil(H/32))` | `(T, ceil(H/16))` |
| FC1 `(E_local, H, 2I)` | `(E_local, ceil(H/32), 2I)` | `(E_local, ceil(H/16), 2I)` |
| FC2 `(E_local, I, H)` | `(E_local, ceil(I/32), H)` | `(E_local, ceil(I/16), H)` |
| output `(T, H)` | `(T, ceil(H/32))` | `(T, ceil(H/16))` |

NVFP4 payload shape replaces the quantized axis by `ceil(axis_extent / 2)`.

## Expert-parallel execution

The semantic data flow is:

```text
local x, topk_idx, topk_weights
              |
              v
flatten valid routes -> stable sort by destination EP rank
              |
              v
variable all-to-all dispatch (token, local expert id, route weight)
              |
              v
group by local expert -> FC1 -> SwiGLU -> FC2 -> combine-format round trip
              |
              v
reverse variable all-to-all in the exact dispatch order
              |
              v
scatter to local [token, top-k, hidden] plane -> FP32 top-k sum
              |
              v
encode public output
```

The reference uses two variable-split `all_to_all_single` phases. The target
MegaMoE kernel can use NVSHMEM pull for dispatch and direct remote stores or
token-back warps for return. Those are different transport mechanisms with the
same observable mapping.

Stable ordering is required only so the reverse exchange can return results
without sending source token metadata to the expert rank. The source rank keeps
its local `(token, top-k slot)` permutation and scatters returned rows back into
the combine plane.

Ranks may have different `T`. Zero-token ranks and zero-count peer splits must
participate in all collectives. A production workspace must reserve enough
inbound assignments for its documented capacity policy. The conservative bound
is `ep_size * max_tokens_per_rank * top_k`; a smaller bound requires an explicit
router capacity/drop contract.

## Mapping to the MegaMoE interface

| concept | Proposed API |
|---|---|
| `static_expert_shape=(E, 2I, H)` | `num_experts`, `intermediate_size`, `hidden_size` |
| `world_size` | inferred from `ep_group` |
| `num_topk` | `top_k` |
| `max_tokens_per_rank` | same name |
| `activation` + `activation_sf` | one `BlockScaledTensor` |
| `fc1_weight` + `fc1_weight_sf` | one `BlockScaledTensor` |
| `fc2_weight` + `fc2_weight_sf` | one `BlockScaledTensor` |
| `topk_idx`, `topk_weights` | same runtime tensors |
| internal `combine_quant`, `combine_sf` | selected by `combine_format`, not passed by the caller |
| BF16 `output_activation` | BF16 case of the returned value |
| new quantized public output | `BlockScaledTensor` selected by `output_format` |
| `local_workspace`, `shared_workspace` | owned/cached by the implementation |
| `peer_rank_ptr_mapper_host` | derived from the EP communication backend |
| `max_active_clusters`, `stream` | backend launch state; current PyTorch stream is used |
| `token_comm_args` | private lowering/kernel argument bundle |
| `generate_c`, `fc1_c` | same names; `fc1_c` is the second returned value |
| `src_token_topk_idx`, `token_src_metadata` | `route_metadata`, the third returned value |

### `fc1_c` and `route_metadata` (training integration)

With `generate_c=True`, `__call__` returns `(output, fc1_c, route_metadata)`.
`fc1_c` is BF16 with shape `(local_routes, 2 * intermediate_size)`, where
`local_routes` is the data-dependent number of valid routes assigned to this
rank's experts. It stays **expert-rank-local** — matching the kernel, which
writes `fc1_c` where FC1 ran and never ships it back — because the backward
pass re-dispatches gradients to the expert ranks, which is where the stashed
activations are consumed.

Row semantics: grouped by local expert (ascending); within an expert, ordered
by source rank, then the source rank's token-major route order. Values are the
raw FC1 accumulator captured **before** SwiGLU, before the gate/up clamp, and
without the router weight (which applies after SwiGLU when
`apply_topk_in_fc1=True`). The kernel's 128-row per-expert padding is a layout
detail and is absent from the logical contract.

`route_metadata` is Int32 `(local_routes, 4)` with columns
`(local_expert, src_rank, src_token, src_slot)`; row `i` identifies the route
behind `fc1_c` row `i`. This is the information the backward pass needs to
re-dispatch output gradients to the right expert-rank rows and to scatter
input gradients back to `(src_token, src_slot)` on the source ranks. It is
the public form of the kernel's `src_token_topk_idx`/`token_src_metadata`
routing words, which the dispatch phase already materializes on the expert
rank.

### Saved tensors for backward

`MoeEpReference.backward(grad_output, activation, fc1_weight, fc2_weight,
topk_idx, topk_weights, fc1_c, route_metadata)` returns
`(grad_activation, grad_fc1_weight, grad_fc2_weight, grad_topk_weights)` and
is the executable statement of the save-set. What must survive from forward
to backward:

| Tensor | Where it lives | Why backward needs it |
|---|---|---|
| `fc1_c` | expert rank (stash) | Sole recompute source: gate/up split, clamp masks, SwiGLU, and the FC2 input `h` are rebuilt from it. Also yields `dW1 = xᵀ · d_c` and `d_x = d_c · W1ᵀ`. |
| `route_metadata` | expert rank (stash) | Reconstructs the receive-order ↔ `fc1_c`-row permutation (sort by `(src_rank, src_token, src_slot)`), groups rows by local expert, and drives the gradient return scatter. |
| `fc1_weight`, `fc2_weight` | expert rank (resident params) | `d_x = d_c · W1ᵀ`, `d_h = d_y · W2ᵀ`. |
| `activation` | source rank (framework input) | FC1 input `x` for `dW1`. Re-dispatched in backward along the identical routes (`topk_idx` is deterministic), or alternatively the forward's dispatched copy `(pool, H)` is stashed on the expert rank to trade memory for the second dispatch. |
| `topk_idx`, `topk_weights` | source rank (framework inputs) | `topk_idx` regenerates the exact dispatch plan; `topk_weights` reconstructs `h' = w·h` for `dW2` and produces `d_w` (returned per `(src_token, src_slot)`). |
| `grad_output` | source rank (incoming) | One row per route is dispatched to the expert rank: `d_y[route] = grad_output[src_token]` (top-k reduce is a sum). |

Deliberately **not** saved: the post-SwiGLU `fc1_output`/`fc1_output_sf`
(recomputed from `fc1_c`), the `combine_quant`/`combine_sf` planes, the
public output, and all counters/flags.

Numerical conventions of the reference backward, to be matched by a fused
kernel: all quantization round-trips (input decode, `combine_format`,
`output_format`) are straight-through — `grad_output` is the gradient of the
dequantized `(T, H)` output; the bf16 `fc1_c` stash is the recompute source,
so backward SwiGLU math runs on bf16-rounded accumulator values (standard
stash precision); clamp gradients are inclusive at the bounds, matching
`torch.clamp`; when `apply_topk_in_fc1=True` the router-weight gradient is
`d_w = ⟨d_h', h⟩` with `h` pre-weight, otherwise `d_w = ⟨d_y, y_pre⟩`.

## Production implementation plan

1. **Constructor and plan cache**
   - Resolve group-relative EP rank/size and local expert range.
   - Validate static dimensions, format combinations, architecture, and capacity.
   - Build a cache key from static configuration plus runtime data/scale dtypes
     and strides.
   - Size local and symmetric workspaces. Allocate through a backend context,
     not on every call.

2. **Forward validation and views**
   - Flatten `BlockScaledTensor` objects into payload/scale kernel arguments.
   - Validate `(T, K)` routing and local weight descriptors.
   - Slice internal dispatch, counter, FC1, combine, and scale planes from owned
     workspaces. The caller never passes these pointers.

3. **Dispatch**
   - Count valid routes per destination and expert.
   - Prefix-sum counts, place routing words, and transfer activation payload,
     activation scale, and router weight.
   - Preserve `(source rank, source token, source slot)` in compact metadata or
     in a reversible placement order.

4. **Local expert kernel**
   - Use one specialization per input/weight family and combine format.
   - Accumulate GEMMs in FP32.
   - Apply the documented gate/up clamp and router-weight location.
   - Quantize route contributions with block boundaries aligned to `H`.

5. **Return and reduction**
   - Direct epilogue remote stores are the default fast path.
   - Standalone/reused token-back warps remain tuning choices.
   - Reduce the internal top-k plane in FP32.
   - BF16-cast or block-quantize the reduced result according to
     `output_format`.

6. **Runtime behavior**
   - Use the current PyTorch CUDA stream.
   - Avoid host synchronization after plan creation.
   - Reset counters in-kernel so repeated calls and CUDA graph replay are safe.
   - Keep peer-visible allocation addresses stable for the lifetime of the
     plan/workspace object.

For quantized public output, the fused final top-k reducer should emit payload
and logical scales together. It should not first materialize a BF16 `(T, H)`
buffer unless that fallback is selected.

## Reference implementation

The executable reference is
`test/python/fe_api/moe_ep/moe_ep_reference.py`. It accepts ordinary floating
tensors or `BlockScaledTensor` inputs and weights. With an explicit process
group it executes actual variable-size PyTorch collectives, so it checks both
MoE math and EP ordering.

The reference is a correctness oracle, not a performance model:

- GEMMs and top-k accumulation use FP32.
- It models combine and public-output rounding, but not CTA tile-dependent
  accumulation order.
- Scale tensors are logical, not atom-swizzled.
- It uses collective push communication rather than NVSHMEM pull/remote store.
- Backward is an explicit `backward` method that re-dispatches gradients with
  the same collectives; the reference is not wrapped in a
  `torch.autograd.Function`, so framework integration supplies that layer.
- It has no expert-capacity drop policy beyond explicit `-1` routes.

## Validation and test matrix

The accompanying tests cover:

- MXFP8 and NVFP4 quantize/dequantize shape and dtype contracts;
- quantization along the weight reduction axis;
- BF16 output for both router-weight locations and gate/up clamp;
- MXFP8 and NVFP4 quantized combine round-trips;
- mixed block-scaled activation, FC1, and FC2 inputs;
- `fc1_c`/`route_metadata` capture: values, expert grouping, and row order;
- backward against autograd replicas — fp32 and block-scaled inputs, both
  router-weight locations, gate/up clamp, and zero gradient for `-1` routes;
- four EP ranks (one GPU each, NCCL) with unequal local token counts and
  cross-rank routes: forward, `fc1_c`/`route_metadata`, and backward gradients;
- API acceptance of block-scaled inputs and the `generate_c` allocation
  contract;
- strict expected-failure gates comparing the API against the reference for
  every output format, quantized inputs, and `generate_c`, armed until the
  device kernel lands.

Production validation should add CUDA/NVSHMEM runs for every pair of
`combine_format` and `output_format`, skewed/all-to-one routing, empty experts,
zero-token ranks, duplicate expert selections, maximum capacity, CUDA graph
replay, and comparisons against this reference after dequantization.

## Proposed first-version boundaries

These choices should remain explicit until there is a concrete model requiring
more surface area:

- contiguous, equal expert partition only;
- no expert bias;
- no shared/dense expert inside this operator;
- no implicit top-k normalization;
- no capacity factor or implicit route drop;
- backward semantics are fixed by `MoeEpReference.backward` (consuming the
  `generate_c=True` stash); the bundled device backward (bprop) implementation
  (`python/cudnn/moe_ep/_megamoe_backend/megamoe/bwd_kernel`) comes from the
  Flashinfer team, not the FastKernel team; the `torch.autograd` wrapper is
  not part of this first version;
- logical scales at the Python boundary, backend swizzle internally.
