# MoE + Expert Parallel API

Status: public API, validated lazy backend seam, internal runtime/workspace
owners, and executable PyTorch reference. The current device target is the
Rubin training `fwd_glu` kernel plus the restricted `bwd_dglu` path on exactly
SM107 (compute capability 10.7). It accepts MXFP8 E4M3/E8M0 operands or plain
BF16/FP16/FP32 operands staged to MXFP8, supports BF16 or MXFP8 combine with
BF16 output, and any positive EP size for forward and backward. Unsupported
combinations fail explicitly rather than returning uncomputed storage.

The design removes workspace pointers, peer pointer mappers, streams, and
individual scheduler knobs from the semantic runtime interface. Performance
tuning is encapsulated in the optional `MoeEpTuningConfig`.

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
- In the public contract, `output_format` describes the
  post-top-k-reduction `(T, H)` result and `combine_format` independently
  describes each per-route FC2 contribution on the EP return path.
- The public API reserves BF16, MXFP8, and NVFP4 spellings for both choices.
  Quantized output is a data-plus-scale object, never a scale-free PyTorch
  tensor. This is a future contract, not the current device capability.
- The current Rubin backend accepts `combine_format="bf16"` or `"mxfp8"` and
  requires `output_format="bf16"`. NVFP4 combine and quantized public output
  are rejected before runtime initialization.

The distinction between public output and combine traffic is intentional. A
future transport-only MXFP8/NVFP4 mode would set `combine_format` to that
format and keep `output_format="bf16"`. The current SM107 MXFP8 combine path is
the training kernel's direct `MXFP8(FP32 accumulator)` conversion.

## Initial device-backend implementation scope

The public contract below remains the target. Device support will be enabled
incrementally, and unsupported combinations must fail explicitly rather than
return uninitialized storage.

The current implementation connects the complete EP-subgroup path through the
executable CuTeDSL backend. Its deliberately narrow capability is:

- the device must report exactly compute capability `(10, 7)` (SM107). If
  `CUTE_DSL_ARCH` is unset, the compile runner sets it to `sm_107a`; an
  existing value must target `sm_107` or `sm_107a`;
- `activation`, `fc1_weight`, and `fc2_weight` may be MXFP8
  `BlockScaledTensor` objects using logical, unswizzled scales and `axis=1`,
  or plain BF16/FP16/FP32 tensors that staging quantizes to MXFP8;
- MXFP8 payloads use FP8 E4M3 and scales use FP8 E8M0, with the shapes specified
  in "Block-scaled representation";
- `combine_format` may be `"bf16"` or `"mxfp8"` and `output_format` must be
  `"bf16"`; both
  `generate_c=False` and eager-only `generate_c=True` forward are supported;
- `generate_c=True` performs one lockstep route-count collective and returns
  fresh compact `fc1_c`/`route_metadata` tensors. The default path uses
  128-row internal expert alignment. Opt-in
  `backward_wgrad_mode="operands"` requires 256-row alignment and additionally
  returns caller-owned MXFP8 grouped-wgrad operands. Training execution does
  not support CUDA Graph capture; the restricted device backward returns
  activation and router-weight gradients; backward hardware acceptance
  currently covers EP1/EP2/EP4;
- forward and backward support optional `gate_up_clamp`; Rubin training
  execution requires `apply_topk_in_fc1=True`;
- `max_tokens_per_rank` must be explicitly positive; `top_k <= 32`,
  `H % 128 == 0`, and `I % 256 == 0`;
- `ep_group=None` remains explicit single-rank execution. Distributed execution
  accepts any initialized `torch.distributed.ProcessGroup`, including
  non-contiguous global-rank membership, with any positive EP size. The public
  contract requires `top_k <= num_experts` and the device path additionally
  requires `top_k <= 32`; `top_k` may exceed `experts_per_rank`. Expert
  ownership, peer tables, and route metadata use dense group-relative EP ranks;
  experts remain contiguous and equally partitioned;
- every subgroup rank must call forward, backward, warmup, and graph replay
  (where supported) in lockstep, including zero-token and zero-valid-route
  ranks. Teardown must also be coordinated, although `close()` does not insert
  a process-group barrier. Validation is rank-local, and collective
  participation is a caller contract rather than an extra host-synchronized
  control collective;
- the first lazy subgroup forward launch performs a one-time readiness
  rendezvous after staging and JIT: each rank synchronizes its current stream,
  all-gathers the effective tuning signature and rejects a mismatch, then
  enters a process-group barrier before peer metadata writes can begin. The
  first distributed backward launch likewise synchronizes and barriers without
  a second tuning gather; EP1 skips both rendezvous. Subsequent eager launches
  and graph replays do not add this control collective;
- while the process-global NVSHMEM runtime is active, all backend instances in
  that process share one CUDA device binding, one ordered EP-subgroup
  membership, and one reference count; distributed deployment uses one process
  per GPU. A different subgroup cannot become active in the same process until
  the first is fully released. A non-WORLD subgroup does not attach to an
  externally initialized NVSHMEM runtime whose membership cannot be verified;
  a matching full-WORLD external runtime may be attached but is never finalized
  by this backend;
- each operator owns its local workspace and its NVSHMEM-symmetric workspace,
  plus transformed-weight and reduction scratch. Allocations are plan-scoped
  and stable across eager calls; `generate_c=True` additionally owns the BF16
  high-watermark C buffer described above;
- the forward and backward compile paths instantiate the vendored Rubin
  training `Sm107MegaMoEMxfp8GluKernel` and
  `Sm107MegaMoEMxfp8DgluKernel`, respectively. They do not select the
  Blackwell inference MegaMoE implementation.

Plain BF16/FP16/FP32 operands do not select a separate floating-point kernel
specialization: staging quantizes them to MXFP8 E4M3/E8M0 before launch.
Pre-quantized MXFP8 payloads and logical scales are preserved rather than
dequantized and requantized.

The private CuTeDSL source snapshot lives under
`python/cudnn/moe_ep/_megamoe_backend/cutedsl_src`. Its
`VENDOR_INFO.md` records the upstream source revisions, vendoring dates,
copied-file manifest, local import adaptations, and update procedure. The
Apache-2.0 headers and redistribution terms are packaged beside the source.
The current kernel entry points are vendored under
`kernel_src/rubin/training/mega/fwd_glu` and
`kernel_src/rubin/training/mega/bwd_dglu`; those packages must use
package-relative imports, must not depend on a sibling `cutedsl_megamoe`
checkout, and must not import `kernel_src.blackwell`.

After installing the `moe_ep` optional dependencies, run the L0 device
validation with:

```bash
python -m pytest \
  test/python/fe_api/moe_ep/test_moe_ep_forward.py \
  test/python/fe_api/moe_ep/test_moe_ep_backward.py \
  -m L0
```

These forward and backward core suites include public-contract and capability
checks, staging, runtime, reference, and supported single-rank numerical
coverage. On SM107, the L0 forward suite exercises the production compile and
launch path. Full numerical parity, stress, CUDA Graph replay, multi-rank
forward, and device backward acceptance are covered by L1 tests and the
hardware/container runner rather than by this L0 command alone.

The ordinary L1 hardware/container runner is single-node: its distributed
forward matrix uses `mp.spawn` for EP2, EP3, and EP4. EP7, EP12, EP15, and
EP16 use the torchrun-native multi-node suite instead. From the first node of
an existing Slurm allocation, start one torchrun agent per selected node with:

```bash
torchrun \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  -m pytest \
  test/python/fe_api/moe_ep/test_moe_ep_forward_multinode.py \
  -m "L1 and moe_ep_multinode" \
  -k "${PYTEST_FILTER}"
```

NVSHMEM requires the same number of EP PEs on every participating node. Use
`NNODES=7`, `NPROC_PER_NODE=2`, and `PYTEST_FILTER=ep7-world14` for EP7; its
subgroup selects local rank zero on every node. Use `NNODES=5`,
`NPROC_PER_NODE=4`, and `PYTEST_FILTER=ep15-world20` for EP15; its subgroup
selects local ranks zero through two on every node. EP12 uses `NNODES=3`,
`NPROC_PER_NODE=4`, and `PYTEST_FILTER=ep12-world12`; EP16 uses `NNODES=4`,
`NPROC_PER_NODE=4`, and `PYTEST_FILTER=ep16-world16`. The remaining WORLD
ranks synchronize without constructing `MoeEp`. The Slurm harness must provide
a distinct `NODE_RANK` to each node and shared `MASTER_ADDR`/`MASTER_PORT`
values. Each case checks BF16 and MXFP8 combine against the executable
reference plus all-`-1` route behavior. Capability support alone is not a
hardware PASS: preserve the torchrun logs for acceptance evidence.

PyTorch is a prerequisite of the `MoeEp` API and remains in the `moe_ep`
optional extra rather than becoming a base dependency of the entire
`nvidia-cudnn-frontend` distribution. The same extra selects the CUDA-13
CuTeDSL stack and `nvshmem4py-cu13>=0.3.1`. Given an installed PyTorch,
ordinary `import cudnn` and `import cudnn.moe_ep` do not import CuTeDSL,
NVSHMEM4Py, or CUDA Python and do not initialize CUDA. Missing CuTeDSL or
NVSHMEM runtime components are reported as `BackendUnavailableError` only
when a supported device forward first needs the backend.

The hardware, API, stress, and packaging runner defaults to
`MOE_EP_DEPENDENCY_MODE=rubin-internal`. In this mode it first removes every
installed `nvidia-cutlass-dsl*` distribution, then installs the latest
pre-release `nvidia-cutlass-dsl` from NVIDIA's Rubin-capable CUTLASS DSL master
index with PyPI as a dependency fallback. The runner verifies that
`cutlass.utils.rubin_helpers` is importable before compiling the kernel.
`MOE_EP_DEPENDENCY_MODE=latest` and `minimum` retain public-release acceptance;
the latter pins CUTLASS DSL, NVSHMEM4Py, and Apache TVM FFI to 4.8.0, 0.3.1,
and 0.1.11 respectively. The public 4.7.0 wheel does not contain
`rubin_helpers`. The packaging runner also validates an isolated wheel import
and the vendored Rubin import closure.

## Public API

The production class is `cudnn.moe_ep.MoeEp`. The test-only
`MoeEpReference` is the executable semantic and numerical oracle; it is not the
device backend.

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
        backward_wgrad_mode: Literal["none", "operands"] = "none",
        token_padding_size: int = 128,
        sf_padding_size: int = 128,
        tuning: Optional[MoeEpTuningConfig] = None,
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
        | tuple[
            Tensor | BlockScaledTensor,
            Tensor,
            Tensor,
            MoeEpWgradForwardStash,
        ]
    ): ...

    def warmup(
        self,
        activation: Tensor | BlockScaledTensor,
        fc1_weight: Tensor | BlockScaledTensor,
        fc2_weight: Tensor | BlockScaledTensor,
        topk_idx: Tensor,
        topk_weights: Tensor,
    ) -> None: ...

    def backward(
        self,
        grad_output: Tensor,
        fc1_weight: Tensor | BlockScaledTensor,
        fc2_weight: Tensor | BlockScaledTensor,
        topk_idx: Tensor,
        topk_weights: Tensor,
        fc1_c: Tensor,
        route_metadata: Tensor,
        *,
        wgrad_forward_stash: Optional[MoeEpWgradForwardStash] = None,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[Tensor, Tensor, MoeEpWgradOperands]
    ): ...

    def close(self) -> None: ...

    def __enter__(self) -> "MoeEp": ...

    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...
```

An initialized `ep_group` enables EP. `None` deliberately means a one-rank
execution, even if a default distributed process group exists. This prevents an
operator from silently communicating on the wrong group.

`close()` is terminal and idempotent. Once device workspaces have been created,
the backend synchronizes outstanding CUDA work before releasing transformed
weights, workspace, and runtime ownership. It does not issue a process-group
barrier; distributed callers must coordinate teardown so one rank cannot
release symmetric storage while a peer is still launching or replaying.
The context-manager form is preferred when deterministic release matters.

### Constructor contract

| Argument | Meaning |
|---|---|
| `num_experts` | Global expert count `E`; must be divisible by EP size. |
| `hidden_size` | Model hidden dimension `H`. |
| `intermediate_size` | Post-SwiGLU dimension `I`; FC1 has `2 * I` columns. |
| `top_k` | Fixed routing width `K`, with `1 <= K <= E`. |
| `ep_group` | Process group whose group-relative rank determines expert ownership. |
| `max_tokens_per_rank` | Maximum local input tokens `T`; optional in the reference and constructor, but the current device capability gate requires an explicit positive value on first execution. |
| `output_format` | Encoding returned after top-k reduction. |
| `combine_format` | Encoding/rounding of each route contribution before top-k reduction. |
| `apply_topk_in_fc1` | Multiply the post-SwiGLU intermediate by the router weight before FC2; otherwise multiply the combine-rounded FC2 route contribution in the standalone top-k reducer. |
| `gate_up_clamp` | If set, use `gate = clamp(gate, max=abs(limit))` and `up = clamp(up, min=-abs(limit), max=abs(limit))`. |
| `generate_c` | Training integration: additionally return `fc1_c` (the BF16-rounded pre-SwiGLU FC1 accumulator of every route this rank's experts processed) and its row-aligned `route_metadata`. |
| `backward_wgrad_mode` | `"none"` preserves the default API. `"operands"` opts into caller-owned MXFP8 operands for external grouped FC1/FC2 wgrad GEMMs; it requires `generate_c=True`, `token_padding_size=256`, and `sf_padding_size=128`. |
| `token_padding_size` | Token-dimension padding used by the Rubin execution plan; positive integer, default 128. |
| `sf_padding_size` | Scale-factor padding; positive multiple of 128, default 128. Operand mode currently requires exactly 128. |
| `tuning` | Optional `MoeEpTuningConfig`; every rank in an EP group must use the same effective configuration. |

The constructor validates public static dimensions, format alignment, padding,
and tuning. MXFP8 format spellings require `H % 32 == 0`; NVFP4 format
spellings require `H % 16 == 0`. Device-specific requirements such as SM107,
`H % 128 == 0`, `I % 256 == 0`, `top_k <= 32`, and explicit positive capacity
are checked lazily before backend initialization.

The wgrad mode is strictly opt-in. With the default
`backward_wgrad_mode="none"`, the constructor default, forward return, backward
signature, backward return, padding default, allocation behavior, and
documented numerical semantics are unchanged.

Scheduler settings are exposed through the optional `MoeEpTuningConfig`
object. The current public knobs are `token_back_mode`, `epi_flag_batch`,
`token_in_flag_batch`, `group_hint`, and `reduce_topk_in_kernel`; they are
keyword configuration rather than positional runtime arguments.

`reduce_topk_in_kernel=True` enables the in-kernel top-k reduction path and is
restricted to BF16 combine/output, `apply_topk_in_fc1=True`, and
`token_back_mode="epi_warps"`. BF16 reduction order can change rounding, so
this path is accepted against the documented numerical tolerance rather than
bitwise Form A equality.

### Forward tensor contract

Let `T` be this rank's token count and `E_local = E / ep_size`.

| Tensor | Logical shape | Required properties |
|---|---:|---|
| `activation` | `(T, H)` | BF16/FP16/FP32 tensor, or block-scaled along axis 1. |
| `fc1_weight` | `(E_local, H, 2I)` | Local experts only; block-scaled weights use axis 1. |
| `fc2_weight` | `(E_local, I, H)` | Local experts only; block-scaled weights use axis 1. |
| `topk_idx` | `(T, K)` | INT32 or INT64 global expert IDs; `-1` means unused. |
| `topk_weights` | `(T, K)` | Floating router/combine weights. They are not implicitly normalized. |

All tensors must be on one device. Quantized data and its scale tensor must
also share a device. Biases, shared experts, nonuniform expert placement, and
capacity-based route dropping are outside this first API.

The first successful backend creation binds a `MoeEp` instance and its stable
workspace allocations to that device. A later call on another device raises
`ValueError`; callers must create one `MoeEp` instance per device.

### Return value

- BF16: a `torch.bfloat16` tensor with logical shape `(T, H)`.
- MXFP8/NVFP4: a `BlockScaledTensor` containing `data`, `scale`, `format`,
  `logical_shape`, and the scaled axis. `dequantize()` reconstructs a regular
  tensor.

With `generate_c=True`, the call instead returns
`(output, fc1_c, route_metadata)`; see the training-integration sections
below.

With both `generate_c=True` and `backward_wgrad_mode="operands"`, it returns
`(output, fc1_c, route_metadata, wgrad_forward_stash)`. The fourth item belongs
to that exact routed forward call and must be passed to its corresponding
backward call.

The return type is fixed by the constructor, so an individual module instance
does not change its output structure across calls.

### Backward call contract

`backward` requires the operator to be constructed with `generate_c=True`; the
production entry point is `MoeEp.backward`, while `MoeEpReference.backward`
defines its executable semantic oracle. The Rubin MXFP8 device path supports
BF16/MXFP8 combine, BF16 output, any positive EP size,
`apply_topk_in_fc1=True`, and optional `gate_up_clamp`. It is a collective:
every rank in `ep_group` must call it because gradients re-dispatch along the
identical forward routes.

| Argument | Shape | Provided by |
|---|---:|---|
| `grad_output` | `(T, H)` | Any floating dtype on the request device; incoming gradient of the *dequantized* public output (all encodes are straight-through). |
| `fc1_weight`, `fc2_weight`, `topk_idx`, `topk_weights` | as in forward | the framework re-supplies the forward weights and routing inputs; `topk_idx` deterministically regenerates the dispatch plan. |
| `fc1_c`, `route_metadata` | `(local_routes, 2I)`, `(local_routes, 4)` | the `generate_c=True` forward stash, passed back unchanged. |
| `wgrad_forward_stash` | `MoeEpWgradForwardStash` | Keyword-only and required only in operand mode. It contains the forward `x.T` operand plus padded expert offsets/counts and exact route identity. Passing it in default mode is an error. |

Default mode returns two FP32 tensors:

| Return | Shape | Meaning |
|---|---:|---|
| `grad_activation` | `(T, H)` | summed over this token's valid routes; gradients w.r.t. the dequantized activation values. |
| `grad_topk_weights` | `(T, K)` | per-route router-weight gradient; exact zero at `-1` slots. |

Operand mode returns
`(grad_activation, grad_topk_weights, wgrad_operands)`. The first two values
retain the default meanings and dtypes; `wgrad_operands` is described below.

On the Rubin device path, dGLU materializes `grad_activation` in BF16 and the
public wrapper widens it to FP32, so its numerical granularity remains BF16.
The semantic router-weight gradient is recomputed from the unquantized
floating `grad_output` and returned in FP32.

Internally, the compact public `fc1_c` rows are lowered into an external
pool-layout BF16 `fc1_preact` tensor. The kernel also writes a pre-zeroed,
symmetric source-domain FP32 dprob plane with shape
`(max_tokens_per_rank, top_k)` for its peer-atomic ABI; that internal plane is
not the public return because the public straight-through semantics use the
FP32 recomputation described above.

The save-set, recompute rules, and numerical conventions behind this
signature are specified in "Saved tensors for backward" below.

### Grouped-wgrad operand contract

`backward_wgrad_mode="operands"` exposes the operands needed by the existing
`GroupedGemmWgradSm100` / `grouped_gemm_wgrad_wrapper_sm100` Tensor2D ABI. It
does not launch those GEMMs and does not return dense weight gradients.

For local expert `e`, let `R_e` be its valid route count,
`P_e = ceil(R_e / 256) * 256`, and `Kp = sum_e P_e`. Every expert occupies one
contiguous range in the shared K dimension. Valid rows precede zero padding;
an empty expert contributes zero extent, so cumulative offsets may repeat.
`expert_offsets[e] = sum_{j <= e} P_j`, and `valid_route_counts[e] = R_e`.

Forward returns `MoeEpWgradForwardStash` with:

| Field | Logical shape | Meaning |
|---|---:|---|
| `fc1_a`, `fc1_sfa` | `(H, Kp)`, `(round_up(H,128), round_up(Kp/32,4))` | MXFP8 `x.T` data and assembled E8M0 scales. |
| `expert_offsets` | `(E_local,)` | Int32 cumulative 256-padded expert ends. |
| `valid_route_counts` | `(E_local,)` | Int32 unpadded route counts. |
| `route_metadata` | `(local_routes, 4)` | The same compact route identity returned beside `fc1_c`. |

Backward returns `MoeEpWgradOperands`. It carries those five fields plus:

| Field | Logical shape | Meaning |
|---|---:|---|
| `fc1_b`, `fc1_sfb` | `(Kp, 2I)`, `(round_up(2I,128), round_up(Kp/32,4))` | MXFP8 `dC`, where `C = x @ W1`. |
| `fc2_a`, `fc2_sfa` | `(I, Kp)`, `(round_up(I,128), round_up(Kp/32,4))` | MXFP8 route-weighted recomputed `(p * h).T`, `h = silu(gate) * up`. |
| `fc2_b`, `fc2_sfb` | `(Kp, H)`, `(round_up(H,128), round_up(Kp/32,4))` | MXFP8 unweighted routed `dY`. |

For each expert range, including its zero padding, the represented dense
operations are:

```text
dW1[e] = fc1_a[e] @ fc1_b[e] = x[e].T @ dC[e]
dW2[e] = fc2_a[e] @ fc2_b[e] = (p[e] * h[e]).T @ dY[e]
```

The data operands are E4M3. Scale factors are E8M0 with logical 1x32 scaling
and grouped-wgrad's assembled physical 128x4 scale tiles. A operands have
unit K stride; B operands use the grouped-wgrad K-major view with unit K
stride.

The Rubin staging/quantization order is part of the operand model:

1. `x` is first staged to MXFP8 along H (plain inputs only), routed and padded,
   then column-requantized along expert K to form `fc1_a`.
2. BF16 `fc1_c` is recomputed through clamp/SwiGLU without a router weight;
   that `h` is column-requantized along K to form `fc2_a`.
3. `grad_output` is staged to MXFP8 along H before re-dispatch. The route
   weight is then applied exactly once, and the result is
   column-requantized along K to form `fc2_b`.
4. Staged `dY` and backward-staged `W2.T` produce `dH`; the route weight is
   applied before the SwiGLU derivative, and the resulting `dC` is directly
   column-requantized along K to form `fc1_b`.

All returned stash and operand tensors are caller-owned allocations. They do
not alias reusable execution-plan workspace and are not overwritten by later
forward/backward calls. The caller must retain the forward stash through its
matching backward call and retain the returned operands until every external
grouped-wgrad consumer has completed. Route metadata and counts are validated
against the matching call; stashes are not interchangeable between different
routing inputs.

### Example

The following illustrates the reserved future quantized-output contract. It is
valid reference-level API semantics, but the current SM107 device backend
rejects `output_format="nvfp4"`. The shown `combine_format="mxfp8"` is supported
by the device backend when paired with `output_format="bf16"`.

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
combine[t,k] = dequantize(quantize(expert[t,k], combine_format))

if not apply_topk_in_fc1:
    combine[t,k] *= p[t,k]

result[t]    = sum_k(combine[t,k])
output       = encode(result, output_format)
```

For BF16 combine, `quantize/dequantize` above means a BF16 round trip. Invalid
slots contribute exact zero. Accumulation across top-k slots is FP32 and the
public encoding is applied after reduction.

For `combine_format="mxfp8"`, the current Rubin forward and backward paths
directly convert each FP32 route accumulator to MXFP8 before top-k reduction.

Moving the router weight from the post-SwiGLU intermediate to the
combine-rounded FC2 contribution is algebraically equivalent only without the
intervening low-precision conversions. The option is therefore semantic and is
fixed in the constructor, matching the MegaMoE kernel.

## Block-scaled representation

The API uses logical, unswizzled scales. Backend-specific F8_128x4 swizzling is
an implementation detail performed while constructing tensor maps or staging
weights.

| Format | Payload | Scale | Block | Quantized axis |
|---|---|---|---:|---|
| BF16 | BF16 | none | n/a | n/a |
| MXFP8 | FP8 E4M3 | FP8 E8M0 | 32 | public `axis=1`: contraction axis for weights, feature/output axis otherwise |
| NVFP4 | packed FP4 E2M1, low nibble first | FP8 E4M3 | 16 | public `axis=1`: contraction axis for weights, feature/output axis otherwise |

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

| concept | Public API |
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
| forward column-requant `x.T` | `MoeEpWgradForwardStash.fc1_a/fc1_sfa` in operand mode |
| dGLU column-requant `dC`, `(p*h).T`, `dY` | `MoeEpWgradOperands` returned by backward |

### `fc1_c` and `route_metadata` (training integration)

With `generate_c=True`, `__call__` returns `(output, fc1_c, route_metadata)`.
Operand mode appends `wgrad_forward_stash` as a fourth item.
`fc1_c` is BF16 with shape `(local_routes, 2 * intermediate_size)`, where
`local_routes` is the data-dependent number of valid routes assigned to this
rank's experts. It stays **expert-rank-local** — matching the kernel, which
writes `fc1_c` where FC1 ran and never ships it back — because the backward
pass re-dispatches output gradients to the expert ranks, which is where the
stashed preactivations are consumed.

Row semantics: grouped by local expert (ascending); within an expert, ordered
by source rank, then the source rank's token-major route order. Values are the
FC1 FP32 accumulator rounded to BF16 and captured **before** SwiGLU, before the
gate/up clamp, and without the router weight (which applies after SwiGLU when
`apply_topk_in_fc1=True`). Columns `[0:I)` are gate and `[I:2I)` are up. The
kernel's mode-dependent per-expert padding (128 rows by default, 256 in operand
mode) and internal gate/up-interleaved layout are absent from the logical
`fc1_c` contract.

`route_metadata` is Int32 `(local_routes, 4)` with columns
`(local_expert, src_rank, src_token, src_slot)`; row `i` identifies the route
behind `fc1_c` row `i`. This is the information the backward pass needs to
re-dispatch output gradients to the right expert-rank rows and to scatter
input gradients back to `(src_token, src_slot)` on the source ranks. It is
the public form of the kernel's `src_token_topk_idx`/`token_src_metadata`
routing words, which the dispatch phase already materializes on the expert
rank.

### Saved tensors for backward

By default, `MoeEpReference.backward(grad_output, fc1_weight, fc2_weight,
topk_idx, topk_weights, fc1_c, route_metadata)` returns
`(grad_activation, grad_topk_weights)` and is the executable statement of the
save-set. Operand mode also takes `wgrad_forward_stash=` and appends a
`WgradOperandsReference` result. What must survive from forward to backward:

| Tensor | Where it lives | Why backward needs it |
|---|---|---|
| `fc1_c` | expert rank (stash) | Sole recompute source: gate/up split, clamp masks, SwiGLU, and the FC2 input `h` are rebuilt from it; it yields `d_x = d_c · W1ᵀ`. |
| `route_metadata` | expert rank (stash) | Reconstructs the receive-order ↔ `fc1_c`-row permutation (sort by `(src_rank, src_token, src_slot)`), groups rows by local expert, and drives the gradient return scatter. |
| `fc1_weight`, `fc2_weight` | expert rank (resident params) | `d_x = d_c · W1ᵀ`, `d_h = d_y · W2ᵀ`. |
| `topk_idx`, `topk_weights` | source rank (framework inputs) | `topk_idx` regenerates the exact dispatch plan; `topk_weights` scales the activation-gradient path and produces `d_w` (returned per `(src_token, src_slot)`). |
| `grad_output` | source rank (incoming) | One row per route is dispatched to the expert rank: `d_y[route] = grad_output[src_token]` (top-k reduce is a sum). |
| `wgrad_forward_stash` | expert rank (operand mode only) | Caller-owned MXFP8 `x.T` plus 256-padded offsets/counts and route identity; supplies FC1 A and fixes the grouped K segmentation. |

Deliberately **not** saved: the post-SwiGLU `fc1_output`/`fc1_output_sf`
(recomputed from `fc1_c`), the `combine_quant`/`combine_sf` planes, the
public output, and all counters/flags.

In the reference backward, input decode, `combine_format`, and `output_format`
round trips are straight-through: `grad_output` is the gradient of the
dequantized `(T, H)` output. The BF16 `fc1_c` stash is the recompute source, so
backward SwiGLU math runs on BF16-rounded accumulator values; clamp gradients
are inclusive at the bounds, matching `torch.clamp`. When
`apply_topk_in_fc1=True`, the router-weight gradient is
`d_w = ⟨d_h', h⟩` with `h` pre-weight; otherwise it is
`d_w = ⟨d_y, y_pre⟩`. The Rubin device dGLU path additionally MXFP8-stages
`grad_output` and transposed weights, modeled in parity tests by
`backward_operand_format="mxfp8"`, while recomputing `grad_topk_weights` from
the original floating `grad_output`.

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
   - Bootstrap and allocate once outside CUDA graph capture; capture requires a
     prior warmup whose stream has completed before capture begins.
   - `MoeEp.warmup(...)` runs one complete eager forward and synchronizes its
     CUDA device. For an EP subgroup, all member ranks must call it concurrently
     and the caller must align ranks before capture; the method deliberately
     does not issue a process-group barrier.
   - Distributed graph capture is per-rank and replay is lockstep. Capture
     records the cross-rank kernel without executing it; every replay must be
     issued by every EP rank in the same iteration.
   - Cache successful eager route-value validation by tensor identity and
     version so unchanged routing avoids repeated host synchronization.
   - Reset counters in-kernel so repeated calls and CUDA graph replay are safe.
   - Keep peer-visible allocation addresses stable for the lifetime of the
     plan/workspace object.
   - A captured graph has a static-storage contract: the `MoeEp` instance,
     workspace, captured input/output storages, and transformed weights must
     outlive every replay. Weights must not be modified after capture; replace
     or modify them only after retiring the graph, then warm up and recapture.
   - The same `MoeEp` instance must not be used concurrently by graph replay
     and eager execution. Eager calls on different streams are serialized by a
     completion event before shared workspace staging.
   - Warmup validates route values. Capture/replay requires every route to
     remain `-1` or a valid global expert ID; data-dependent host validation is
     not capturable.
   - Inference tensors without a PyTorch version counter are repacked on every
     eager call and are not accepted as weights during graph capture.
   - `generate_c=True` is eager-only because its exact `(local_routes, 2I)`
     result requires data-dependent route counting and compaction. Capture is
     rejected before the count collective, allocation, or kernel launch.
   - Distributed callers must coordinate `close()` because symmetric allocation
     release and an owned NVSHMEM finalization must not race a peer that is
     still launching or replaying the operator. `close()` itself does not issue
     a process-group barrier.

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
- `WgradForwardStashReference` and `WgradOperandsReference` model logical
  MXFP8 values, 256-route expert padding, and dense
  `x.T @ dC` / `(p*h).T @ dY` results. Their scales are logical rather than
  the production 128x4 physical assembly.
- It has no expert-capacity drop policy beyond explicit `-1` routes.

Device parity tests configure the same `MoeEpReference` with
`intermediate_format="mxfp8"` to model the internal post-SwiGLU MXFP8 round
trip before FC2, and with `backward_operand_format="mxfp8"` to model dGLU
operand staging. These options are diagnostic backend approximations and are
deliberately not part of the public mathematical contract above.

## Validation and test matrix

Public-contract and reference tests remain broader than the current device
backend. They cover MXFP8/NVFP4 representation and quantization semantics,
both router-weight locations, quantized combine/output round trips, and the
explicit reference backward. Those checks do not imply kernel capability.

The status below applies specifically to the production Rubin device backend.
“Hardware-validated” means that the case passed on SM107 (compute capability
10.7); CPU/reference tests, collection, and skips do not establish that status.

### Hardware-validated on SM107

- EP1 forward has passed with MXFP8 E4M3/E8M0 operands and with mixed plain
  BF16/FP16/FP32 activation/weights, BF16 or MXFP8 combine, BF16 output,
  `apply_topk_in_fc1=True`, gate/up clamp, all-`-1` routes, and fresh output
  ownership. INT32 and INT64 routing indices and supported shape/top-k cases
  through `top_k=32` have also passed.
- Eager EP1 `generate_c=True` has passed with the Rubin-required
  `apply_topk_in_fc1=True`. The checks cover compact BF16 `fc1_c`, INT32
  `route_metadata`, pre-clamp/unweighted values, repeated calls, and fresh
  tensor ownership.
- EP1 warmup, non-default tuning, 100-call eager stress, and CUDA Graph replay
  have passed. CUDA Graph replay with post-FC2 router weighting has also passed.
- Single-node WORLD-group eager forward parity has passed for EP2, EP3, and
  EP4 with BF16 and MXFP8 combine, including all-`-1` route behavior.
- A WORLD4 eager test has passed for disjoint non-contiguous EP2 subgroups
  `[0,2]` and `[1,3]`, including the case where subgroup rank zero is not global
  rank zero.
- Multi-node eager forward parity has passed for EP12/WORLD12 (three nodes,
  four PEs per node) and EP16/WORLD16 (four nodes, four PEs per node), with
  BF16 and MXFP8 combine and all-`-1` route behavior.
- MXFP8 backward has passed for EP1, EP2, and EP4 with BF16 and
  MXFP8 combine. It checks activation and router-weight gradients against the
  executable reference; EP1 additionally covers repeated calls and explicitly
  reordered forward stashes.
- The returned operand field/stride/scale ABI has passed direct execution
  through both FC1 and FC2 grouped-wgrad GEMMs on SM100 using
  reference-generated operands. This establishes consumer ABI integration, not
  end-to-end Rubin operand production.

### Supported but awaiting hardware validation

- Forward capability accepts every positive EP size. EP5, EP6, EP8 through
  EP11, EP13, EP14, and sizes above EP16 have no current hardware acceptance
  case. Sizes above EP16 use the generated vector peer-offset path instead of
  the fixed 128-byte by-value table used through EP16.
- EP7/WORLD14 is defined as a seven-node subgroup with one EP PE per node, and
  EP15/WORLD20 is defined as a five-node subgroup with three EP PEs per node.
  Both acceptance cases remain pending on allocations with the required node
  counts.
- Forward CUDA Graph capture/replay and lifecycle contracts apply to
  distributed EP groups, but current EP2+ acceptance runs cover eager parity
  only; distributed stress and graph replay remain pending.
- Plain/mixed operands, gate/up clamp, post-FC2 router weighting, and explicit
  `generate_c=True` output semantics are hardware-validated at EP1, but are not
  separately validated across every supported distributed EP size.
- End-to-end Rubin forward/backward production of wgrad operands remains
  awaiting SM107 hardware validation. The available SM100 test can execute the
  grouped-wgrad consumer but cannot execute the SM107-only MegaMoE producer.

### Currently unsupported by the device backend

- Devices other than SM107, non-positive or unspecified `max_tokens_per_rank`,
  `hidden_size` not divisible by 128,
  `intermediate_size` not divisible by 256, and `top_k > 32`.
- Native NVFP4 operands or combine, any non-BF16 public output, and plain
  operand dtypes other than BF16/FP16/FP32. MXFP8 operands and BF16/MXFP8
  combine with BF16 output are supported.
- Backward execution with `apply_topk_in_fc1=False`, non-BF16 output, or
  backward CUDA Graph capture.
- Wgrad operand mode with padding other than 256, without `generate_c=True`,
  or outside the restricted Rubin MXFP8 backward configuration. The mode
  returns operands only; dense `dW1`/`dW2` computation remains an explicit
  grouped-wgrad call by the integration layer.
- CUDA Graph capture with `generate_c=True`, same-process concurrent EP
  subgroups, expert bias, shared/dense experts inside this operator, implicit
  top-k normalization, capacity-factor routing, and implicit route drop.
- Dense weight-gradient returns and an integrated `torch.autograd.Function`
  wrapper.

Source/packaging tests separately require the vendored Rubin
`training/mega/fwd_glu` and `training/mega/bwd_dglu` packages, reject sibling
`cutedsl_megamoe` dependencies and `kernel_src.blackwell` imports, and validate
isolated-wheel imports. These checks validate packaging rather than additional
device capabilities.

Future format families must add the same CUDA/NVSHMEM, stress, graph, and
isolated-package matrix before their capability gates are removed.

## Current first-version boundaries

These choices should remain explicit until there is a concrete model requiring
more surface area:

- contiguous, equal expert partition only;
- no expert bias;
- no shared/dense expert inside this operator;
- no implicit top-k normalization;
- no capacity factor or implicit route drop;
- backward semantics are fixed by `MoeEpReference.backward` (consuming the
  `generate_c=True` stash); opt-in wgrad operands are exposed, while dense
  weight-gradient returns and a `torch.autograd` wrapper are not part of this
  API;
- logical scales at the Python boundary, backend swizzle internally.
