# Python API Naming Conventions

These rules apply to every **new** public op API: any op name added to
`test/python/api_index/api_index.txt`. The graph API (`cudnn.pygraph` and its
enums) is out of scope. Existing names keep working; renaming them is part of
the API pruning plan, not this document.

## Name pattern

```
cudnn.ops.<op_name>_<direction>               # PyTorch
cudnn.ops.<op_name>_<direction>_jax           # JAX
cudnn.ops.<op_name>_<direction>_standalone    # no framework
```

- `<op_name>`: the operation in snake_case, such as `grouped_gemm_glu` or `sdpa`.
- `jax`: marks the JAX variant. The PyTorch variant has no framework token.
- `standalone`: marks the framework-free variant, which takes raw device
  pointers and a stream instead of framework tensors.
- `<direction>`: `forward` or `backward`. Every op has a direction, including an
  op without a backward.

Each name is a function exported from `cudnn.ops`. Ops are imperative: forward
and backward are separate calls.

## Rules

1. **Functions only.** Do not export op classes. Exported op classes such as
   `GroupedGemmSwigluSm100` are legacy.
2. **No `wrapper` in names.** The function is the framework op; `wrapper` adds
   nothing.
3. **Spell out the direction.** Use `forward` / `backward`, never `fwd` / `bwd`.
4. **Keep implementation details out of names.** Do not add an architecture
   (`sm90`, `sm100`, ...) or an implementation technology (`dsl`, `cutedsl`,
   `frost`, `triton`, ...). The function dispatches internally and raises a
   clear error for an unsupported configuration.
5. **Ops live in `cudnn.ops`.** Do not add op APIs to the top-level `cudnn`
   package, `cudnn.torch`, `cudnn.jax`, or an op package such as `cudnn.gemm`; a
   JAX variant is marked by a `_jax` suffix. Existing exports keep working until
   the API pruning plan retires them.

## Examples

| Existing name | New API name |
| --- | --- |
| `cudnn.gemm.cutedsl.grouped.grouped_gemm_glu_wrapper_sm100` | `cudnn.ops.grouped_gemm_glu_forward` |
| `cudnn.gemm.cutedsl.grouped.GroupedGemmGluSm100` | none; use `cudnn.ops.grouped_gemm_glu_forward` |
| `cudnn.gemm.cutedsl.grouped.grouped_gemm_glu_jax_sm100` | `cudnn.ops.grouped_gemm_glu_forward_jax` |
| `cudnn.sdpa.fwd.sdpa_fwd_wrapper_dsl_sm100` | `cudnn.ops.sdpa_forward` |
| `cudnn.deepseek_sparse_attention.sparse_attention_backward_wrapper` | `cudnn.ops.sparse_attention_backward` |
| `cudnn.flex_attention.FlexAttentionBwd` | `cudnn.ops.flex_attention_backward` |
| `cudnn.rmsnorm_rht_amax.rmsnorm_rht_amax_wrapper_sm100` | `cudnn.ops.rmsnorm_rht_amax_forward` |
| `cudnn.torch.block_sparse_attention_forward` | `cudnn.ops.block_sparse_attention_forward` |
| `cudnn.jax.block_sparse_attention_forward` | `cudnn.ops.block_sparse_attention_forward_jax` |
| `cudnn.jax.kimi_delta_attention_fwd` | `cudnn.ops.kimi_delta_attention_forward_jax` |
| `cudnn.ops.causal_conv1d` | `cudnn.ops.causal_conv1d_forward` |
| `cudnn.causal_conv1d_forward` (raw pointers) | `cudnn.ops.causal_conv1d_forward_standalone` |
