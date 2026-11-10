# Cake chunk-64 KDA training

Pin `plan_name="kda_cake_det"` in `cudnn.ops.kimi_delta_attention` and set
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` to use the current Cake chunk-64 chain.
The existing `kda_cake` C16 engine stays separate.

The initial contract is SM100 with CUDA Toolkit 13.4 or later:

- Contiguous BF16 THD Q/K/V, with K=V=128 and 16, 32, or 64 equal heads.
- Caller-normalized Q/K, FP32 activated per-channel log gates, BF16 activated beta.
- Contiguous int32 packed prefixes, zero initial state, no final state, and
  `checkpoint_every_n_tokens=0`.
- Nonempty token pack; empty documents and partial chunks are valid.

Packed prefixes must start at zero, be nondecreasing, and end at the declared
token count. The caller owns this device metadata contract. Inputs and outputs
must have 16-byte-aligned addresses. The adapter rejects overlapping writes and
scratch before launch.

```python
out, _ = cudnn.ops.kimi_delta_attention(
    q, k, v, g, beta, cu_seqlens,
    output_final_state=False,
    checkpoint_every_n_tokens=0,
    plan_name="kda_cake_det",
)
```

The upstream host planner selects serial recurrence variants and workspace
lifetimes at build time. Graph execution uses caller-owned storage and the
execution stream. It does not allocate device memory, read tensor values on the
host, or compile. Backward recomputes forward intermediates, then applies the
fixed-order gradient chain. Torch imports are confined to upstream host planning;
Torch operations do not run inside the graph executor.

Grouped heads, fused gate activation, fused Q/K normalization, recurrent states,
checkpoints, context parallelism, and the segmented low-head schedule decline.
These are bounded initial contracts, with no throughput claim.

Sources and build records come from
[miles-ext-kernels f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce).
`cudnn/_cake/_vendor/VENDORED_KDA.json` lists the copied files and selected records.
