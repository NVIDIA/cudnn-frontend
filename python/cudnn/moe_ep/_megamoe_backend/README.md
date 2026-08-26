# MegaMoE backend capabilities

This private backend implements the public `cudnn.moe_ep` contract with Rubin
SM107 CuTeDSL products. Public validation and backend capability checks are
separate: a request may be valid for `MoeEp` but unavailable in this backend.

## Implemented forward paths

- MXFP8 inputs use the training MegaMoE forward GLU product.
- Plain BF16, FP16, and FP32 operands are quantized into the same logical
  MXFP8 representation before launch.
- Block-scaled operands must use the public MXFP8 representation. Native NVFP4
  operands are part of the public contract but are not executable in this
  backend.
- Combine format may be BF16 or MXFP8. Final output format is BF16.
- MXFP8 combine quantizes each FP32 route accumulator directly.
- Rubin training execution requires `apply_topk_in_fc1=True`.
- Forward with `generate_c=False` supports CUDA Graph capture after warmup;
  `generate_c=True` is eager-only.

## Explicit Rubin limits

- CUDA compute capability must be 10.7.
- `max_tokens_per_rank` must be positive and explicit.
- `hidden_size` must be divisible by 128, and `intermediate_size` must be
  divisible by 256.
- `top_k` must not exceed 32.
- EP sizes above 16 use a generated vector peer-offset table; EP sizes through
  16 use the fixed 128-byte by-value table.

These are backend limits, not additional public `MoeEp` semantics. They remain
precise, product-specific capability gates rather than hidden padding or a
silent numerical fallback.

## Backward status

`MoeEp.backward` has a validated backend seam and requires a forward stash from
`generate_c=True`. In the default `backward_wgrad_mode="none"`, the restricted
Rubin MXFP8 path returns
`(grad_activation, grad_topk_weights)` for any positive EP size with BF16 or
MXFP8 combine, BF16 output, `apply_topk_in_fc1=True`, optional
`gate_up_clamp`, and eager execution. It uses `fc1_c` and `route_metadata` to reconstruct an
external pool-layout `fc1_preact` tensor and converts `grad_output` to FP32
before re-dispatching it for semantic dprob. The kernel's source-domain dprob
plane is symmetric and reset before every launch; the public router-weight
gradient remains an FP32 semantic recomputation. Default mode does not accept
or retain the forward activation, does not produce FC1/FC2 wgrad operands, and
does not depend on "most recent forward" state.

### Opt-in grouped-wgrad operands

Constructing with `backward_wgrad_mode="operands"` requires
`generate_c=True`, `token_padding_size=256`, and `sf_padding_size=128`.
Forward then returns
`(output, fc1_c, route_metadata, wgrad_forward_stash)`. The fourth value is a
`MoeEpWgradForwardStash` for that exact routed call: MXFP8 `x.T` data/scales,
cumulative padded expert offsets, valid route counts, and the same route
metadata.

Backward takes the stash by keyword:

```python
grad_activation, grad_topk_weights, operands = op.backward(
    grad_output,
    fc1_weight,
    fc2_weight,
    topk_idx,
    topk_weights,
    fc1_c,
    route_metadata,
    wgrad_forward_stash=wgrad_forward_stash,
)
```

`MoeEpWgradOperands` is directly shaped for the grouped-wgrad Tensor2D ABI:

- `fc1_a=(H,Kp)`, `fc1_b=(Kp,2I)` represent
  `dW1 = x.T @ dC`;
- `fc2_a=(I,Kp)`, `fc2_b=(Kp,H)` represent the upstream factorization
  `dW2 = (p * h).T @ dY`;
- every local expert's valid rows precede zero padding to 256 routes;
  `expert_offsets` contains cumulative padded ends and may repeat for empty
  experts, while `valid_route_counts` excludes padding;
- E4M3 data uses unit stride on K. E8M0 scales represent logical 1x32 blocks
  assembled into the grouped kernel's physical 128x4 layout.

The device order is deliberate. Forward first MXFP8-stages `x` along H, then
column-requantizes routed/padded rows along K. Backward follows the upstream
FC2-gradient factorization: the recomputed `h` export carries the route weight,
while the token-axis `grad_y2` export is the unweighted routed `dY`. Their
grouped product is therefore `dW2 = (p*h).T @ dY`. Staged `dY` and `W2.T`
produce `dH`, the route weight is applied before the SwiGLU derivative, and
`dC` is directly column-requantized along K. All three backward auxiliary
scale outputs use the upstream MN-major 128-column by 4-token-block atom
layout.

The forward stash and backward operand tensors are caller-owned fresh
allocations, not views of reusable execution-plan workspace. Callers must keep
the forward stash alive through its matching backward call and keep returned
operands alive until external grouped-wgrad work completes. Later operator
calls do not overwrite them. Route identity/count validation prevents mixing
stashes from different forwards.

This mode only produces operands; it does not launch grouped wgrad or return
dense `dW1`/`dW2`. It remains eager-only and inherits the restricted Rubin
MXFP8 backward gates (SM107, BF16 output,
`apply_topk_in_fc1=True`, and BF16/MXFP8 combine). End-to-end operand
production still requires SM107 acceptance. Direct FC1/FC2 consumer execution
has been validated separately on SM100 with reference-generated operands.

The dGLU product emits BF16 `grad_activation`; the backend converts it to FP32
for the public return. This is a documented BF16-rounded numerical limitation,
not strict FP32 dgrad parity. `apply_topk_in_fc1=False`, NVFP4 operands or
combine, non-BF16 output, and backward CUDA Graph capture remain
capability-gated.

## Validation boundary

L0 tests cover public validation, plain-to-MXFP8 staging, compile/cache keys,
workspace sizing, combine semantics, overflow audit behavior, and backward
layout/dispatch capability gates. Current CUDA Graph acceptance covers EP1
forward. The single-node eager forward suite defines WORLD EP2/EP3/EP4, and
the current SM107 L1 hardware run establishes PASS for all three EP sizes.
The torchrun-native multi-node suite balances NVSHMEM PEs across participating
nodes: EP7 uses a WORLD14 subgroup over seven nodes with two workers per node,
EP12 uses WORLD12 over three nodes with four workers per node, EP15 uses a
WORLD20 subgroup over five nodes with four workers per node, and EP16 uses
WORLD16 over four nodes with four workers per node. Multi-node collection or
skip results do not establish a hardware PASS. Current hardware runs establish
PASS for EP12 and EP16; EP7 and EP15 remain pending.

End-to-end device forward/backward parity requires SM107 hardware and the
`moe_ep` optional runtime dependencies, including a CuTeDSL installation that
provides `cutlass.utils.rubin_helpers`. Backward acceptance remains limited to
EP1/EP2/EP4; larger EP sizes are enabled but not covered by current backward
hardware acceptance.
