# MegaMoE backend

The private backend provides Rubin SM107 MXFP8 execution for `cudnn.moe_ep`.

## Capability

- CUDA Rubin SM107
- BF16 output with BF16 or MXFP8 combine
- `hidden_size % 128 == 0`
- `intermediate_size % 256 == 0`
- `top_k <= min(32, num_experts)`
- positive `max_tokens_per_rank`
- `apply_topk_in_fc1=True`

Inference accepts BF16/FP16/FP32 or MXFP8 operands. Training accepts
BF16/FP32 or MXFP8 activation and grad-output, contiguous Int32 routing
indices, contiguous FP32 routing weights, and independent native forward and
backward weight packs. Every active routing index is a valid global expert ID;
the `-1` sentinel is reserved for private capacity-tail staging.

## Execution state

`MoeEp.prepare_training` creates one private `Mxfp8TrainingState`, which owns
only:

- prepared forward/backward kernels and compile caches;
- NVSHMEM/runtime handles;
- one local and symmetric scratch slab for the instance;
- private fixed-capacity transport and routing scratch used only during a call.

It does not own or retain caller weights, output bundles, saved forward state,
WGrad operands, or weight-staging bundles. No slot is exposed by the public
API.

## Native weights

Training has explicit contiguous and discrete native-weight specializations.
Contiguous execution accepts `MoeEpNativeForwardWeights` and
`MoeEpNativeBackwardWeights`; validation checks the exact versioned
`layout_id`, shape, stride, dtype, alignment, and device.

Discrete execution accepts `MoeEpNativeDiscreteForwardWeights` and
`MoeEpNativeDiscreteBackwardWeights`. Each role supplies caller-owned CUDA
`int64[E_local]` payload and scale pointer tables. The launch adapter passes
their addresses directly to the specialized kernel without allocation, copy,
refresh, or persistent binding. Pointee shape/stride/extent and table/pointee
lifetime remain caller contracts. Discrete backward uses dedicated layout IDs
because upstream consumes physical row-major GEMM `(N, K)` payloads rather
than the contiguous ABI's transposed tensor layouts.

`materialize_forward` and `materialize_backward` are allocation-free fallback
transforms. They write only caller-provided staging bundles and return native
packs that alias those destinations.

## Inputs and outputs

Plain training inputs use `Mxfp8TrainingStager`. MXFP8
`BlockScaledTensor` inputs bypass quantization and copy their payload/scales
only into the symmetric transport plane required for peer addressing.

Caller outputs are borrowed for one launch:

- required FC1 preactivation is passed directly to forward and backward
  kernels;
- all forward and backward WGrad payloads, scales, and route metadata are
  required after `prepare_training()` and passed directly to the kernels;
- primary forward output, grad-activation, and dprob are stable views of the
  instance-owned symmetric storage because remote ranks address that plane;
- callers bind those exposed views into the required output bundles.

Standalone top-k reduction uses persistent pre-reduction data and scale
planes without clearing them between launches. Dense, non-overflow routing
must completely overwrite every active `(token, top-k slot)` before reduction.
Capacity-tail rows are unspecified and never copied to caller outputs.

The producing kernels already expose the final grouped-WGrad scale carriers
when token and scale-factor padding are both 128. Caller E8M0 matrices are
viewed through the producer's flat or matrix signature, so no scale expansion
kernel is launched. FC1-B remains gate/up-interleaved, and FC1-A/FC2-A use
legal transpose views without physical transpose copies.

## Overflow and distributed ordering

Each phase keeps overflow state private and applies the configured policy
before returning. The transport always truncates safely and derives the same
group-wide overflow bit on every rank from the route totals already exchanged
by the kernel, without a separate scalar collective. There is no public
`finalize_overflow`.

One `MoeEp` instance supports only sequential work on one CUDA stream. The
implementation does not bind or validate that stream, and stable symmetric
addresses do not retain results across calls. A later call may overwrite
earlier output, grad-activation, dprob, routing, and finalizer contents.
Applications needing parallel resource isolation use multiple instances and
must still submit distributed forward/backward launches in the same instance
order with the required caller-owned CUDA-event dependencies on every EP rank.

## CUDA Graph

Preparation and first-time compilation happen before capture. Training calls
require every destination advertised by `prepare_training()` to be
caller-owned. Every input, output, saved-state, native weight, and staging
address referenced by a graph remains stable until that graph executable is
destroyed. Routing values remain dense and valid on every replay; replay adds
no value-validation work. Eager calls may change addresses between
invocations. Multiple graphs for one instance must be captured sequentially
and replayed sequentially on the same stream; they share and may overwrite the
same symmetric results.
