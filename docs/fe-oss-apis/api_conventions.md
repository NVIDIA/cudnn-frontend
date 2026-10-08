# Python API Naming Conventions

These rules apply to every **new** public Python API: any name that is added to
`test/python/api_index/api_index.txt`. Existing names keep working; renaming them
is part of the API pruning plan, not this document.

## Terms

- **Implementation**: the `APIBase` subclass that checks support, compiles, and
  executes a kernel.
- **Wrapper**: the top-level framework API that wraps an implementation as a
  framework op. It takes framework tensors, allocates outputs, reuses compiled
  implementations, and returns the results. "Wrapper" names this role; it is not
  part of the API name.

## Rules

1. **Name the operation, not the role.** Do not put `wrapper` in a public name.
   The wrapper is `<op>` and its implementation is `<Op>`.
2. **Spell out the direction.** Use `_forward` / `_backward` in function names
   and `Forward` / `Backward` in class names. Do not use `_fwd`, `_bwd`, `Fwd`,
   or `Bwd`. An op without a backward takes no direction suffix.
3. **Keep implementation details out of names.** Do not add an architecture
   (`_sm90`, `_sm100`, `Sm100`, ...) or an implementation technology (`_dsl`,
   `_cutedsl`, `_frost`, `_triton`, ...). The API dispatches internally, and an
   unsupported configuration fails `check_support()` with a clear error.
4. **No framework namespaces for kernels.** Do not add kernel APIs under
   `cudnn.jax` or `cudnn.torch`. The PyTorch API takes the plain name, and a JAX
   variant appends `_jax`. Existing `cudnn.jax` / `cudnn.torch` kernel exports
   keep working until the API pruning plan retires them.

## Examples

| Existing name | New API name |
| --- | --- |
| `grouped_gemm_swiglu_wrapper_sm100` | `grouped_gemm_swiglu` |
| `GroupedGemmSwigluSm100` | `GroupedGemmSwiglu` |
| `sparse_attention_backward_wrapper` | `sparse_attention_backward` |
| `sdpa_fwd_wrapper_dsl_sm100` | `sdpa_forward` |
| `HSTUFwdSm100` / `HSTUBwdSm100` | `HSTUForward` / `HSTUBackward` |
| `FlexAttentionBwd` | `FlexAttentionBackward` |
| `grouped_gemm_glu_jax_sm100` | `grouped_gemm_glu_jax` |
| `cudnn.jax.kimi_delta_attention_fwd` | `kimi_delta_attention_forward_jax` |
| `cudnn.torch.block_sparse_attention_forward` | `block_sparse_attention_forward` |
