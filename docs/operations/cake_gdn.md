# Cake GDN training

This experimental engine provides deterministic packed GDN forward and backward
through the existing graph API and `cudnn.ops.gated_delta_net`.

Set `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` and pin `plan_name="gdn_cake_det"`.
It requires SM100, CUDA Toolkit 13.4 or later, BF16 contiguous THD Q/K/V,
FP32 activated log-decay gates and beta, and contiguous int32 packed prefixes.
K and V dimensions are 128. Q/K head counts match; V heads are a positive
multiple of Q heads. Optional Q/K L2 normalization runs inside the preparation
kernel. All operand pointers must be 16-byte aligned.

Each document starts from zero state. Set `output_final_state=False` and
`checkpoint_every_n_tokens=0`; recurrent states, checkpoints, state pools,
fused gate/beta activation, and context parallelism decline explicitly.
The packed prefixes must start at zero, be nondecreasing, and end at the
declared token count. Prefix values remain on device; the caller owns this
metadata contract. Empty documents are supported inside a nonempty pack.

```python
from cudnn.ops import gated_delta_net

out, _ = gated_delta_net(
    q, k, v, g, beta, cu_seqlens,
    output_final_state=False,
    use_qk_l2norm_in_kernel=True,
    checkpoint_every_n_tokens=0,
    plan_name="gdn_cake_det",
)
```

The forward uses four launches: metadata, preparation, WY, and fused recurrent
scan/output. Backward recomputes those intermediates, then runs four gradient
stages. Both use fixed-order reductions and one caller stream. Serial scans
are the initial supported schedule; long documents may run slower than the
segmented upstream planner.

Graph execution uses only caller-owned outputs and scratch reported by
`graph.get_workspace_size()`. It allocates no device memory and reads no tensor
values on the host. The Torch operation allocates its outputs and workspace
at the caller layer. Overlapping outputs, inputs, and scratch are rejected.

Kernel sources and build manifests are copied from
[miles-ext-kernels f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce).
`cudnn/_cake/_vendor/VENDORED_GDN.json` records the upstream file identities.
The existing `kda_cake` C16 engine is separate.
