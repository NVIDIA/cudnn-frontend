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
backward weight packs.

## Execution state

`MoeEp.prepare_training` creates one private `Mxfp8TrainingState`, which owns
only:

- prepared forward/backward kernels and compile caches;
- NVSHMEM/runtime handles;
- one local and symmetric scratch slab per execution lane;
- private fixed-capacity transport and routing scratch used only during a call.

It does not own or retain caller weights, output bundles, saved forward state,
WGrad operands, or weight-staging bundles. No slot is exposed by the public
API.

## Native weights

Training execution accepts only `MoeEpNativeForwardWeights` or
`MoeEpNativeBackwardWeights`. Validation checks the exact versioned
`layout_id`, shape, stride, dtype, alignment, and device. The launch adapter
creates aliases to payload and blocked E8M0 scale tensors without allocation,
copy, refresh, or persistent binding.

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
- combine output and dprob first land in private symmetric buffers, then copy
  to caller buffers because remote ranks address the symmetric plane;
- primary forward/backward outputs are required caller-owned destinations.

The producing kernels already expose the final grouped-WGrad scale carriers
when token and scale-factor padding are both 128. Caller E8M0 matrices are
viewed through the producer's flat or matrix signature, so no scale expansion
kernel is launched. FC1-B remains gate/up-interleaved, and FC1-A/FC2-A use
legal transpose views without physical transpose copies.

## Overflow and distributed ordering

Each phase keeps overflow state private and applies the configured policy
before returning. EP2+ performs the scalar MAX needed for a rank-consistent
decision. There is no public `finalize_overflow`.

One lane is exclusive to one active stream. Every EP rank must submit
distributed forward/backward launches in identical order. Distinct lanes do
not make unordered collective-kernel overlap valid.

## CUDA Graph

Preparation and first-time compilation happen before capture. Training calls
require every destination advertised by `prepare_training()` to be
caller-owned. Every input, output, saved-state, native weight, and staging
address referenced by a graph remains stable until that graph executable is
destroyed. Eager calls may change addresses between invocations.
