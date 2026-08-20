# Framework integration performance: getting the best per-call performance from cuDNN Frontend

When cuDNN Frontend (FE) is driven op-by-op from a framework (PyTorch eager, a custom
autograd op, a JAX/`tvm-ffi` bridge, a scheduler that issues one graph per layer), the
kernels are the same ones cuDNN always runs — but the *host* path around each `execute`
can quietly dominate small and medium ops. This guide lists the host-overhead traps and
how to avoid them, so an integration reaches the kernel-time performance cuDNN is capable of.

Every number below is measured on a B200 (cuDNN 9.26). They are illustrative magnitudes,
not promises — measure your own shapes — but the *ranking* is stable across GPUs.

## TL;DR checklist

- [ ] **Build and plan once per shape; cache by shape.** Never rebuild the graph per call.
- [ ] **Pin the plan and call `execute_plan_at_index`, not the generic `execute`.** (~5 µs/call)
- [ ] **Set the stream once; call `set_stream` only when the stream actually changes.** (~5 µs/call)
- [ ] **Reuse the variant-pack dict and output buffers; avoid per-call Python object churn.** (~2-3 µs/call)
- [ ] **Bind transposed weights as strided `.t()` views, never `.t().contiguous()`.** (avoids a whole transpose kernel)
- [ ] **Fuse in the graph** (activation as a GEMM epilogue/prologue) — real kernel-time wins.
- [ ] **Don't interleave cuDNN `execute` with cuBLAS/other-library calls in a hot eager loop** if you can batch by phase.
- [ ] **CUDA-graph the steady-state loop** — it captures all host dispatch once; replay is kernel-time only.
- [ ] **Benchmark with CUDA-graph replay** when judging kernel quality; eager best-of-N includes host bubbles.

## Why this matters: the host path, not the kernel

For a 256³ bf16 matmul (kernel ≈ free), µs/call is pure host dispatch cost:

| path | µs/call |
|---|---|
| `torch.mm` (cuBLAS, the reference) | 7.6 |
| cuDNN `execute_plan_at_index`, pinned plan, stream fixed, prebuilt variant pack | **8.4** |
| cuDNN generic `execute`, prebuilt variant pack | 13.2 |
| a naive wrapper (per-call `set_stream` + dict build + generic execute) | 21.0 |

**The cuDNN backend dispatch is already at cuBLAS parity (8.4 vs 7.6 µs).** Everything above
that floor is avoidable wrapper cost. A GEMM-heavy layer issues 6-8 of these per step, so a
naive wrapper turns a per-GEMM parity into a multi-µs-per-GEMM tax that shows up as a whole-
layer regression — even though the kernels are identical.

## The traps, and the fix for each

### 1. Rebuilding the graph per call
`validate → build_operation_graph → create_execution_plans → check_support → build_plans`
is build-time work. Do it once, key an entry by `(shapes, strides, dtypes)`, and reuse it.
Per-call, only the variant pack (device pointers) and `execute` should run.

### 2. The generic `execute` re-does per-call work — pin the plan instead
`execute` re-normalizes the variant pack and re-selects among built plans on every call
(~5 µs here). Autotune once (build all plans, time them), remember the winning index, and
call `execute_plan_at_index(var_pack, workspace, index=best, handle=h)` in the hot path.
You also get deterministic kernel selection for free.

```python
# build-time (once per shape)
g.build_plans(cudnn.build_plan_policy.ALL)
best = argmin_over_plans(lambda i: time(g.execute_plan_at_index(vp, ws, index=i, handle=h)))
# hot path
g.execute_plan_at_index(vp, ws, index=best, handle=h)
```

### 3. `set_stream` on every execute
`cudnn.set_stream` costs ~5 µs and is almost always redundant — the stream rarely changes
between calls. Track the last stream and re-set only on change. (Also: querying
`torch.cuda.current_stream().cuda_stream` is itself ~2.6 µs — cache it too.)

```python
if stream != _last_stream:
    cudnn.set_stream(handle=h, stream=stream); _last_stream = stream
```

### 4. Per-call Python object churn
Rebuilding the variant-pack `dict`, `unsqueeze`-ing inputs, allocating the output with
`torch.empty`, and recomputing a `(shape, stride)` cache key each call each cost ~1-3 µs.
Reuse a mutable variant-pack dict, reuse output buffers where lifetimes allow, and use the
cheapest sufficient cache key. (This is the same fast-path treatment already shipped for the
grouped-GEMM and cuTeDSL paths.)

### 5. Materializing transposed weights
An `nn.Linear` weight is `[out, in]`; the GEMM needs it transposed. Binding
`W.t().contiguous()` runs a full transpose kernel every call — for a 5120×17408 bf16 weight
that copy (~0.8 ms) costs *more than the GEMM it feeds*. cuDNN matmul reads column-major
operands directly: declare the tensor with the `.t()` view's stride and bind the `.t()` view.

```python
WG = g.tensor(dim=[1, H, I], stride=list(Wg.t().unsqueeze(0).stride()), data_type=BF16)
# hot path binds Wg.t().unsqueeze(0)  -- a view, no copy
```

### 6. Not fusing what the graph will fuse
Fusing the activation into the GEMM is a genuine kernel-time win, not just a launch saving.
A SwiGLU MLP forward — `gate_gemm + up_gemm + SiLU + mul` — compiles to ONE cuDNN kernel and
runs 1.05-1.20× a 2-cuBLAS-GEMM + torch-activation baseline (eager and under graph replay,
every token count). That same kernel also emits its two pre-activations as extra outputs, so
the backward reads them instead of recomputing two GEMMs. The backward's
`matmul(dout,Wd) + dSwiGLU` uses a B200-tuned 2-CTA FROST kernel, bringing the bare GEMM to
nvjet parity and turning the avoided `dh` HBM round-trip into a whole-step win. At
M=8192, H=5120, I=17408, a balanced run measured 1.077× backward and 1.109× fresh
forward+backward versus eager torch. The earlier 1-CTA result did not show this because its
tile sweep held CTA group and cluster shape fixed. Always measure the whole step rather than
extrapolating from an isolated fused stage.

### 7. Mixing libraries in a hot eager loop
Interleaving cuDNN `execute` with cuBLAS/`torch.mm`/other-library calls, op by op, is the
*worst* eager pattern measured here — worse than either library alone — because each switch
pays host and library-state cost. If you can, keep a phase on one library, or (better) put
the whole step in a CUDA graph.

### 8. Measuring host bubbles instead of kernels
Eager best-of-N CUDA-event timing includes the host dispatch bubble between events; it can
make a faster kernel look slower purely from wrapper cost. When you want to compare *kernel*
quality, capture the region in a CUDA graph and time `replay()` — that removes the host
dispatch bubbles and reports the captured-workload GPU time (kernels plus the graph's own
in-graph launch overhead), which is close to kernel time but not a pure kernel-only profile;
for that, use a kernel profiler (Nsight/`torch.profiler` device time). Use eager timing to
measure your *integration's* per-call overhead (a real cost too), graph replay for the
captured GPU time, and a profiler when you need per-kernel numbers.

## The ceiling: CUDA graphs

For a steady-state training or inference loop with fixed shapes, capture the step (or the
whole iteration) into a CUDA graph once and replay it. This removes *all* per-call host
overhead in this document at a stroke — `set_stream`, plan selection, variant-pack building,
Python churn — leaving essentially the captured GPU work. It is the single highest-leverage step
when the shape is stable. See `docs/cuda-graphs.md`. cuDNN `execute` is capture-safe;
pre-build graphs and pre-allocate workspaces/outputs so nothing allocates during capture.

## What "good" looks like

A correctly-integrated single cuDNN GEMM lands at ~8-9 µs/call eager (cuBLAS parity), and a
CUDA-graphed layer runs at its captured GPU time where the in-graph fusions (SwiGLU forward, dACT
epilogue) are net wins. If you see a GEMM-heavy layer regress when routed through cuDNN, the
cause is almost always one of traps 1-4 above, not the kernels.

---
Companion runnable prototype: `samples/python/gemm_swiglu_mlp_fusion.py` (a fused dense
SwiGLU-MLP autograd op that applies items 2, 5, 6). Measurement probes that produced the
numbers here live in the cudnn-frontend perf investigation (`fla_shim_proto/qwen_ab/`).
