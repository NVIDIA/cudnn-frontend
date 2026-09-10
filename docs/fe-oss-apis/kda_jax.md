# Kimi Delta Attention in JAX (draft)

`cudnn.jax.kimi_delta_attention` returns `(output, final_state_or_None)` and
supports `jax.jit` and first-order reverse-mode differentiation. It uses the
existing Frost KDA kernels, including their backward, checkpoint recompute,
split scheduler, gate gradients, and grouped-head reductions.

This draft is pinned to Frost. Other engines have different launch/runtime
contracts; accepting their graph plans would not make them executable through
this bridge. The existing torch API retains its engine routing.

## Inputs and outputs

`T` is the packed token count, `N` the sequence count, and `HO = max(HQ, HV)`.

| Tensor | Shape | Dtype |
|---|---|---|
| `q` | `[T, HQ, K]` | BF16 or FP16 |
| `k` | `[T, HK, K]` | Same as q |
| `v` | `[T, HV, V]` | Same as q |
| `g` | `[T, HO, K]` | FP32, BF16, or FP16 |
| `beta` | `[T, HO]` | FP32 or q dtype |
| `cu_seqlens` | `[N+1]` | INT32 |
| `initial_state` (optional) | `[N, HO, V, K]` | FP32 or BF16 |
| `a_log` (optional) | `[HO]` | FP32, BF16, or FP16 |
| `dt_bias` (optional) | `[HO, K]` | FP32, BF16, or FP16 |
| output | `[T, HO, V]` | Same as q |
| final state (optional) | `[N, HO, V, K]` | Initial state dtype; otherwise FP32 |

`cu_seqlens` must start at zero, be nondecreasing, and end at `T`. Values are
runtime device data: changing sequence lengths without changing tensor shapes
does not require recompilation. The wrapper validates metadata, not boundary
values; it never copies boundaries to the host. Callers must supply valid
boundaries. Empty individual sequences preserve their initial state (or zero).

The state is V-major. It is a differentiable array, not a mutable cache. Pass the
returned state explicitly to the next call, including inside `jax.lax.scan`.

## Usage

Install GPU-enabled JAX and CuTeDSL separately from the framework-neutral package:

```bash
pip install 'jax[cuda13]' 'nvidia-cutlass-dsl[cu13]>=4.7.0'
```

Select one visible supported GPU before starting Python. This draft requires
one local GPU to avoid guessing a target from tracers. The tested combination is
JAX 0.11.1, CuTeDSL 4.7.1, SM100; broader version/architecture qualification is
still required.

```python
import jax
import jax.numpy as jnp
from cudnn.jax import kimi_delta_attention as kda

q = jnp.full((32, 1, 64), 0.05, jnp.bfloat16)
k, v = q, jnp.ones_like(q)
g = jnp.full(q.shape, -0.1, jnp.float32)
beta = jnp.full((32, 1), 0.5, jnp.float32)
cu = jnp.array([0, 16, 32], jnp.int32)
state = jnp.zeros((2, 1, 64, 64), jnp.float32)

# Eager.
o, final = kda(q, k, v, g, beta, cu, initial_state=state,
               output_final_state=True)

# Options are static; arrays, boundaries and state remain runtime inputs.
def attend(q, k, v, g, beta, cu, state):
    return kda(q, k, v, g, beta, cu, initial_state=state,
               output_final_state=True)

o, final = jax.jit(attend)(q, k, v, g, beta, cu, state)

def loss(q, k, v, g, beta, state):
    o, final = attend(q, k, v, g, beta, cu, state)
    return o.astype(jnp.float32).sum() + final.sum()

grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5)))(
    q, k, v, g, beta, state)
```

Safe-gate parameters are ordinary differentiable arguments:

```python
a_log = jnp.zeros((1,), jnp.float32)
dt_bias = jnp.zeros((1, 64), jnp.float32)

def gate_loss(a_log, dt_bias):
    o, _ = kda(q, k, v, g, beta, cu, safe_gate=True,
               a_log=a_log, dt_bias=dt_bias,
               use_beta_sigmoid_in_kernel=True)
    return o.astype(jnp.float32).sum()

da_log, ddt_bias = jax.jit(jax.grad(gate_loss, argnums=(0, 1)))(a_log, dt_bias)
```

For explicit backward:

```python
from cudnn.jax import kimi_delta_attention_fwd, kimi_delta_attention_bwd

o, final, residual = kimi_delta_attention_fwd(
    q, k, v, g, beta, cu, initial_state=state,
    output_final_state=True, checkpoint_every_n_tokens=16)
grads = kimi_delta_attention_bwd(
    residual, jnp.ones_like(o), d_final_state=jnp.ones_like(final))
# grads.dq, dk, dv, dg, dbeta, d_initial_state, d_a_log, d_dt_bias
```

An omitted final-state cotangent means zero. Gradients for absent initial state
or parameters are `None`. Boundaries and static options are not differentiable.
The explicit APIs do not themselves provide autodiff rules; the high-level API
installs `custom_vjp` using the existing backward kernels. Higher derivatives and
forward-mode differentiation are unsupported.

## Bounded support

| Feature | Draft contract |
|---|---|
| Engine / GPU | Frost; SM100 or SM103; one visible GPU. SM103 not yet tested here. |
| Head dimensions | K and V independently 64 or 128 |
| Heads | HQ:HV ratios 1, 2, 4, or 8 in either direction; HK equals HQ or HV |
| Sequences | Positive static T/N; packed THD; empty individual sequences allowed |
| State | Optional initial/final state; gradients through both; recurrent scan |
| Gates | Natural-log decay by default; safe gate; optional a_log/dt_bias |
| Safe gate | `lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`; default bound -5, allowed [-5, 0); omitted parameters mean 0 |
| Beta | Direct write strength, or sigmoid of logits; `allow_neg_eigval=True` multiplies sigmoid by 2 and requires sigmoid enabled |
| Q/K normalization | Optional in-kernel L2 normalization |
| Scheduling | Default Frost split schedule; `batch_invariant=True` disables splitting |
| Checkpoints | Cadence 0 (backward recomputes) or 16 (saved for backward) |
| Transformations | Eager, jit, first-order grad/vjp, recurrent scan |

Checkpoint capacity is `[max(T // 16 + N, 1), HO, V, K]` in q dtype. The valid
prefix contains `sum(ceil(sequence_length / 16))` rows; padding is initialized to
zero. The residual holds primals, static configuration and optional checkpoints.
It owns no graph handle, workspace or executable. Do not fabricate or edit
residuals: checkpoints must correspond to their saved primals and configuration.

Deferred: cuTile/other engine routing, SM90/SM120/SM107, coarse checkpoints,
zero total tokens, arbitrary physical strides, `vmap`, JVP, higher derivatives,
sharding/multiple GPUs, portable executable export, and full FLA compatibility.
JAX supplies compact row-major operands to the custom call; upstream transposes
may require materialization.

## Execution and review notes

`linear_attention/jax_api.py` owns metadata, argument binding and autodiff.
`linear_attention/frost/kda_launch.py` composes the native launch sequence.
The existing engine owns workspace layout; the split scheduler owns launch
geometry, shared by torch and JAX.

Graph tensors come directly from JAX shape/dtype metadata; the existing
`graph.kda` / `graph.kda_bwd` methods infer output shapes and validate the graph.
Frost workspace layouts are reused. The JAX launcher composes existing CuTe host
functions into one native XLA FFI call per forward/backward. That call receives XLA's stream and launches the complete
sequence, including runtime TMA descriptors and PDL dependencies. No Python
callback, torch tensor, DLPack handoff, or KDA math implementation is introduced.

XLA allocates outputs and per-invocation byte workspace. Backward carves the
same scheduler, descriptor, recompute, gate-reduction and head-reduction regions
as torch. Workspace is never cached across calls. Graph/shape/launch metadata
is cached at trace time; executable caching belongs to the CuTeDSL/JAX bridge.
Checkpoints use the shared bridge's initialized-output support, with aliased
outputs restored to their declared argument positions.

Alternatives considered: a native C++ XLA FFI wrapper around the engine would
require exporting its Python-compiled launch artifacts and maintaining a new
ABI; Python callbacks or torch/DLPack would add host overhead and weaken stream
and capture guarantees. The composite CuTe path is the smallest working bridge.

Before production release, qualify customer shapes, CUDA/JAX/CuTeDSL versions,
SM103, true independent-stream stress and long-sequence numerical limits. The
10–20 us CPU target needs warm dispatch measurements separately from blocking
end-to-end latency and GPU kernel time. Compare against the same Frost schedule
and checkpoint policy. This draft does not assert that target for every shape.
BSA and quantized grouped GEMM are separate work.

Run coverage with:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m pytest test/jax/test_call.py test/jax/test_kda.py
```

## Container tests

The NGC JAX container can build the frontend from this checkout and run the full
JAX KDA and shared-call suites without installing torch. Use an SM100/SM103 GPU:

```bash
set -o pipefail
git archive HEAD | docker run --rm -i --gpus device=0 --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=0 -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e JAX_PLATFORMS=cuda -e CMAKE_BUILD_PARALLEL_LEVEL=8 \
  --entrypoint bash nvcr.io/nvidia/jax:26.07-py3 -lc '
    set -euo pipefail
    mkdir -p /workspace
    cd /workspace
    tar -xf -
    python -m pip install ".[cutedsl]" "nvidia-cutlass-dsl[cu13]==4.7.1" pytest
    python -c "import importlib.util; assert importlib.util.find_spec(\"torch\") is None"
    python -m pytest -s -q test/jax
  '
```

The archive contains committed files only. The container builds its own frontend
extension; no host virtualenv or compiled extension is mounted. GPU CI must
invoke the pytest command explicitly: an import-only check does not run KDA
forward/backward kernels.

Validated on SM100 with `nvcr.io/nvidia/jax:26.07-py3`: the frontend source build
succeeded and all **21 tests passed, zero skipped**. The container used Python
3.12.3, JAX `0.10.2.dev20260630+3757395a28`, CUDA 13.3, cuDNN 9.24.0 and
CuTeDSL 4.7.1, with no PyTorch installation. This includes the subprocess that
rejects torch imports while compiling and executing a jitted KDA gradient.

## Measurements and remaining blocker

Measured on SM100 (148 SMs), Python 3.14, JAX 0.11.1, CuTeDSL 4.7.1,
torch 2.14.0+cu130, driver 580.159.03. BF16 THD inputs, H=4, K=V=128,
one sequence, split scheduling, no recurrent state or optional gate parameters,
checkpoint interval zero. Backward includes checkpoint recomputation. All
forward outputs and five explicit-backward gradients matched torch exactly at
T=128, 1024 and 4096; captured and repeatedly replayed outputs also matched.

At **T=1024**, raw full-sequence GPU time was **45.1 us forward / 128.8 us
backward**. Unprofiled medians of 200 calls after 20 warmups:

| Path | Forward blocking us | Backward blocking us | Forward minus raw us | Backward minus raw us |
|---|---:|---:|---:|---:|
| JAX jit + command buffers | 126.4 | 223.9 | 81.3 | 95.1 |
| torch eager custom ops | 118.9 | 218.0 | 73.8 | 89.2 |
| Fixed-buffer graph replay | 52.8 | 136.1 | 7.8 | 7.3 |

Host dispatch was **45.0 / 63.4 us for JAX**, **95.0 / 129.4 us for torch**,
and **2.3 / 2.3 us for fixed-buffer graph replay** (forward/backward).
**Neither framework meets the 10–20 us CPU target. Torch eager overhead is
also high:** its total-minus-raw gap is 74 / 89 us, whereas fixed-buffer replay
adds only about 8 / 7 us. The eager path performs custom-op dispatch, validation,
output allocation, graph binding and multiple host launches each call. Replay
bypasses that path; this comparison does not isolate each component's cost.
Torch workspace is cached; JAX owns per-invocation workspace and outputs.

With default XLA settings, JAX at T=1024 measured 51.4 / 74.7 us host dispatch
and 113.1 / 216.3 us blocking latency. Command buffers reduced host dispatch but
did not reduce blocking latency in these runs. Nsight shows graph-child updates
when output/workspace addresses change; resolving this overhead remains a
production release blocker. These measurements do not establish a universal
winner or qualify every shape.

Additional command-buffer cases (us; totals include synchronization):

| T | Direction | Raw GPU | JAX total | torch total | JAX minus raw | torch minus raw |
|---:|---|---:|---:|---:|---:|---:|
| 128 | fwd | 24.9 | 94.6 | 105.0 | 69.7 | 80.1 |
| 128 | bwd | 56.6 | 137.0 | 142.6 | 80.4 | 86.0 |
| 4096 | fwd | 64.0 | 140.3 | 140.3 | 76.2 | 76.3 |
| 4096 | bwd | 162.2 | 255.8 | 252.2 | 93.6 | 90.1 |

A separate Nsight trace confirmed five forward and seven backward kernels per
call. Over ten measured T=1024 calls, median GPU time from the first kernel's
start to the last kernel's end was **45.3 / 132.3 us for JAX**, **52.8 / 133.2 us
for torch eager**, and **45.3 / 128.3 us for fixed-buffer replay**. This includes
inter-kernel gaps and preserves overlap. JAX's device sequence is close to the
captured Frost baseline; its large end-to-end gap remains outside that interval.
Graph-child parameter updates were visible on every measured JAX call. These
profiled GPU spans are separate from the unprofiled latency measurements above.

For a standalone composite call, XLA may otherwise decline capture because it
counts only one custom-call operation. The validated capture configuration is:

```python
compiled = jax.jit(attend, compiler_options={
    "xla_gpu_enable_command_buffer": "CUSTOM_CALL",
    "xla_gpu_graph_min_graph_size": 1,
})
```

These are application compiler options; the library does not change global XLA
settings. Capture was confirmed in Nsight, including replay after pointer changes.

Reproduce warm forward and explicit-backward measurements on an otherwise idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python test/jax/benchmark_kda.py --command-buffer --repetitions 200 \
  --output /tmp/kda-benchmark.json
```

Remove `--command-buffer` for default XLA settings. `--tokens`, `--heads`, and
`--dim` control the shape. The benchmark intentionally imports both frameworks;
it checks JAX and raw replay against torch for the output and all five input
gradients, including after repeated batched replay.

- **Host dispatch:** CPU interval around the warm API call, with prior GPU work
  drained. Compilation and input creation are excluded.
- **Blocking latency:** the same call followed by device-wide synchronization,
  using the same synchronization primitive for JAX and torch.
- **Raw GPU sequence:** CUDA-event elapsed time for a graph containing 32 complete
  Frost calls, divided by 32. Capture fixes outputs and workspace; replay bypasses
  Python/custom-op dispatch and allocation. This includes scheduling, prologues,
  main kernels, inter-kernel dependencies and backward checkpoint recomputation.
  It amortizes graph submission/event overhead and preserves PDL overlap; it is
  not a sum of individual kernel durations or a main-kernel-only measurement.
- **Blocking minus raw:** a diagnostic of host/launch/allocation/synchronization
  overhead and GPU scheduling gaps. It is not pure CPU time. Host dispatch and GPU
  execution overlap, so subtracting raw GPU time from host dispatch is invalid.

Each path has 20 warmup calls; reported latencies are medians. The fixed-buffer,
one-call graph replay also reports host and blocking latency as a control. This
is a stateless BF16 microbenchmark with fixed inputs, checkpoint interval zero,
precomputed explicit-backward residuals, and no optional gate parameters. It does
not measure compilation, a complete training step, or all supported features.

To inspect individual kernels and graph updates, wrap a shorter run with:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
  -o /tmp/kda-profile python test/jax/benchmark_kda.py \
  --command-buffer --repetitions 10 --warmup 2 --raw-batch-size 4
```

Use unprofiled runs for host latency. Concurrent GPU profiling produced roughly
2 ms synchronization latency even for a trivial graph on this machine; that run
was discarded and measurements repeated after the other process exited.

Memory-only Compute Sanitizer coverage passed five forward/backward and concurrent
replay cases with zero errors using `--tool memcheck --report-api-errors no`.
API reporting was disabled for that run because cuda-python probes API versions
newer than the local driver's 13.0 interface; the unfiltered run reported 68 such
probe errors, without invalid device-buffer accesses. Upgrade the driver before
claiming an unfiltered clean sanitizer run on the customer stack.
