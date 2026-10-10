# Python API Naming Conventions

These rules apply to every **new** public op API: any op name added to
`test/python/api_index/api_index.txt`. The graph API (`cudnn.pygraph` and its
enums) is out of scope. Existing names keep working; renaming them is part of
the API pruning plan, not this document.

## Name pattern

```
cudnn.<op_name>_<direction>               # PyTorch
cudnn.<op_name>_jax_<direction>           # JAX
cudnn.<op_name>_standalone_<direction>    # no framework
```

- `<op_name>`: the operation in snake_case, such as `grouped_gemm_swiglu` or `sdpa`.
- `jax`: marks the JAX variant. The PyTorch variant has no framework token.
- `standalone`: marks the framework-free variant, which takes raw device
  pointers and a stream instead of framework tensors.
- `<direction>`: `forward` or `backward`. Every op has a direction, including an
  op without a backward.

Each name is a function exported from the top-level `cudnn` package.

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
5. **No namespaces for ops.** Do not add op APIs under `cudnn.torch`,
   `cudnn.jax`, or an op package such as `cudnn.gemm`; a JAX variant is marked
   by `_jax` in the name. Existing exports keep working until the API pruning
   plan retires them.

## Examples

| Existing name | New API name |
| --- | --- |
| `cudnn.gemm.cutedsl.grouped.grouped_gemm_swiglu_wrapper_sm100` | `cudnn.grouped_gemm_swiglu_forward` |
| `cudnn.gemm.cutedsl.grouped.GroupedGemmSwigluSm100` | none; use `cudnn.grouped_gemm_swiglu_forward` |
| `cudnn.gemm.cutedsl.grouped.grouped_gemm_glu_jax_sm100` | `cudnn.grouped_gemm_glu_jax_forward` |
| `cudnn.sdpa.fwd.sdpa_fwd_wrapper_dsl_sm100` | `cudnn.sdpa_forward` |
| `cudnn.deepseek_sparse_attention.sparse_attention_backward_wrapper` | `cudnn.sparse_attention_backward` |
| `cudnn.flex_attention.FlexAttentionBwd` | `cudnn.flex_attention_backward` |
| `cudnn.rmsnorm_rht_amax.rmsnorm_rht_amax_wrapper_sm100` | `cudnn.rmsnorm_rht_amax_forward` |
| `cudnn.torch.block_sparse_attention_forward` | `cudnn.block_sparse_attention_forward` |
| `cudnn.jax.block_sparse_attention_forward` | `cudnn.block_sparse_attention_jax_forward` |
| `cudnn.jax.kimi_delta_attention_fwd` | `cudnn.kimi_delta_attention_jax_forward` |
| `cudnn.causal_conv1d_forward` (raw pointers) | `cudnn.causal_conv1d_standalone_forward` |
