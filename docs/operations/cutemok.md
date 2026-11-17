# CuTe MoK

`cudnn.moe_ep.cutemok` exposes the experimental CuTe MoK API: one expert-parallel
mixture-of-experts layer. Each rank's tokens are dispatched to the ranks that own
their top-k experts, the routed experts' SwiGLU MLPs (and an optional shared
expert's) are computed, and the results are combined back on the tokens' rank.
The forward and the backward are each one persistent CuTe DSL megakernel that
overlaps the cross-rank token traffic with the expert GEMMs, plus a few small
launches (route all-gather, scheduler, staging copies, barriers, epilogue).
Routed weights are BF16 or MXFP8. The operator is derived from
[Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens) by Cursor
Research, whose megakernels are written in CUDA C++. It is separate from `MoeEp`
and from Cake MoK (`cudnn.moe_ep.cake`).

Requirements: SM100 or SM103 (B200, B300, GB200, GB300); `nvidia-cutlass-dsl[cu13]
>= 4.8.0` in its cu13 build (a lower version or the cu12 build is declined with an
error naming it); PyTorch 2.9 or later built for CUDA 13; an initialized NCCL
process group of 1, 4, 8, 16, 32 or 64 ranks whose GPUs share PyTorch symmetric
memory.

```python
import cudnn

config = cudnn.ops.CuteMoKConfig(fwd_num_comm_sms=40, bwd_num_comm_sms=28,
                                 minibatch_size=4096, macrobatch_size=131072)
plan = cudnn.ops.cutemok_prepare(  # once; serves every layer of this configuration
    config, group, device=device, max_tokens_per_rank=16384, hidden_size=H,
    intermediate_size=I, num_local_experts=E_local, topk=K,
)
weights = (shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down)
y, schedule, saved = cudnn.ops.cutemok_forward(
    x, top_experts, router_weights, *weights, plan=plan, config=config,
)
grads = cudnn.ops.cutemok_backward(
    dy, x, router_weights, *weights, schedule, saved, plan=plan, config=config,
)
```

`x[T,H]` and `dy[T,H]` are BF16 with T a multiple of 256 in
`[512, max_tokens_per_rank]`. `top_experts[T,K]` is int32 global expert ids, `-1`
for an unused slot; `router_weights[T,K]` is FP32. Shared gate/up are `[I,H]`,
shared down `[H,I]`, or three `None` for a plan prepared with `has_shared=False`;
routed weights add an expert axis of `E_local`. H and I are multiples of 128 and
at least 256; top-k is in `[1, 255]`. Activations and routing are contiguous;
another dtype or layout, or routed weights in another form than the plan's, is
declined with `NotImplementedError` naming the tensor before anything launches.
Rank r owns experts `[r * E_local, (r + 1) * E_local)`; the output is the shared
expert's plus the router-weighted sum of the routed experts' outputs.
The forward returns a `TupleDict` with `output`, `schedule` and `forward_context`;
the backward's keys are `d_x`, `d_router_weights`, `d_w_routed_gate`,
`d_w_routed_up`, `d_w_routed_down`, `d_w_shared_gate`, `d_w_shared_up`,
`d_w_shared_down` and `d_w_output_gate` (`None` without a shared output gate).
Outputs and weight gradients are BF16, router-weight gradients FP32 (zero at
unused slots).

`cutemok_prepare`, collective over the group and outside graph capture, compiles
every kernel the forward and backward launch and allocates the plan's workspace:
PyTorch symmetric memory sized for `max_tokens_per_rank` (T_max). Two BF16
`[T_max * K, H]` buffers, the routed rows' expert outputs and input gradients
(`4 * T_max * K * H` bytes together), dominate it; the rest are two BF16
`[T_max, H]` staging buffers, the FP32 `[T_max, K]` router weights and their
gradients, the int32 `[ranks, T_max, K]` gathered routes and a barrier word. The
plan also holds the megakernels' counters and scratch (sized by the config's
macrobatch).

Forward and backward are allocating eager calls: they allocate their results with
`torch.empty` on the current stream's device on every call, nothing cached across
calls, and return them; they compile nothing and never read device memory to the
host. The first call over new `CuteMoKSplitWeight` address tables also allocates
and builds their TMA descriptors, once per set of tables. The schedule and the
forward context hold the backward's inputs: keep them until the backward.

The plan owns its workspace; nothing caches or shares it. One plan can serve every
layer of the same configuration: pass it to each of them (two plans have two
workspaces). Calls on one plan share its buffers: issue them in one stream order,
on every rank in the same order with the same T. Retain the plan until all eager
calls and graph replays that use it have finished on every rank, then drop it on
every rank, outside graph capture, to free the workspace: synchronize first (for
example `torch.cuda.synchronize()` and a barrier over the group), since the other
ranks write into its buffers.

Warm each variant up with an eager forward and backward before capturing it in a
CUDA graph; a variant the plan did not compile raises `RuntimeError` instead of
compiling. Captured calls allocate their results in the graph's private pool,
which the graph retains; keep the graph and its plan alive for all replays. A
graph replays its own T and input addresses; their contents, routing included,
may change between replays.

The token count and the `CuteMoKConfig` runtime fields (communication SMs per
direction, minibatch sizes, a macrobatch up to the plan's) may change on every
call and compile nothing. Variants are compile-time choices of prepare:
`swiglu_clamped` (forward and backward then take `swiglu_limit=C`:
`silu(min(gate, C)) * clip(up, -C, C)`), `situ` (SiTU-GLU, `situ_beta`),
`has_shared`, `shared_output_gate`, `split_routed_weights`,
`main_grad_dtype`, `routed_precision`, `native_columnwise` and the config's
`gate_up_context`, `comm_tile_shape` and `epilogue_combine`. A plan prepared
with `shared_output_gate=True` takes `shared_output_gate_weight` (BF16 `[1,H]`)
in every forward and backward, and a plan without it declines one: the shared
output is scaled by `sigmoid(x @ w^T)`, and the backward returns the weight's
FP32 gradient as `d_w_output_gate`, or adds it into
`shared_output_gate_main_grad` (FP32 or BF16 `[1,H]`). The gated backward runs
fused Triton kernels. One
`[E_local, 2I, H]` tensor passed as both routed gate and up is a combined FC1.
The routed rows a rank receives, padded to 256 per local expert, must fit the
schedule of `T * K * max(2, ceil(ranks * config.schedule_capacity_multiplier))`
rows; a schedule that does not fit stops the kernel with a device-side trap.

For MXFP8 routed weights, prepare with `routed_precision="mxfp8"` and call
`cudnn.ops.cutemok_quantize_weights(gate, up, down)` after every weight update. It
returns a `TupleDict` of `forward_weights` and `backward_weights`, each splatted as
the routed weights of its call. I and H must be multiples of 128; shared weights,
activations and gradients stay BF16. `config.gate_up_context` keeps the routed
gate/up activations as MXFP8 (default) or BF16.

Routed experts held in separate allocations are passed as
`CuteMoKSplitWeight.from_experts(tensors)`, one per routed matrix (one
`[2I, H]` tensor per expert for a combined FC1). Prepare with
`split_routed_weights=True` and `main_grad_dtype`; the backward then takes
`main_grads` (FP32 or BF16 buffers it accumulates into, instead of returning
new weight gradients) and `main_grad_storage_tables`, the per-expert main-grad
addresses as int64 CUDA tensors. Build the weights and tables once; again only
when an allocation moves. A backward from a forward context without its routed
activations is not part of this API.

The public module is `python/cudnn/moe_ep/cutemok.py`; the implementation is the
private package `python/cudnn/moe_ep/_cutemok/` (`functional.py`, `ops.py`,
`cutedsl/` with the kernels and hosts, `cutedsl/_compat/` with helpers adapted
from QuACK). `_cutemok/UPSTREAM.md` records the sources. Tests are in
`test/python/moe_ep/cutemok/test_cutemok.py`; they run on one GPU under pytest and
expert-parallel under `torchrun`.
