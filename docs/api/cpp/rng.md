# rng

Append a random-number generation operation to the graph.

## C++ synopsis

```cpp
auto Y = graph.rng(seed, offset,
                   fe::graph::Rng_attributes()
                       .set_distribution(fe::RngDistribution_t::BERNOULLI)
                       .set_bernoulli_probability(0.9)
                       .set_dim({B, H, S, T})
                       .set_stride({H * S * T, S * T, T, 1}));
```

## Parameters

- `seed` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required unless `attributes.set_seed(...)` is used): Runtime seed tensor. When `Rng_attributes::seed` is set to a compile-time integer value, the backend operation uses that value and does not require a seed tensor descriptor.

- `offset` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required unless `attributes.set_seed(...)` is used): Runtime offset tensor used with runtime seed mode.

- `attributes` (`fe::graph::Rng_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_distribution(fe::RngDistribution_t)`: Distribution selector. The current node implementation accepts only `BERNOULLI`.

- `attributes.set_bernoulli_probability(double)`: Bernoulli success probability. It must be specified when `distribution` is `BERNOULLI`.

- `attributes.set_dim(std::vector<int64_t>)` and `attributes.set_stride(std::vector<int64_t>)`: Optional output shape and stride. If the returned output tensor does not set a shape, `set_dim(...)` is used. If no stride is provided, the frontend generates NHWC-style strides from the output dimensions.

- `attributes.set_seed(std::optional<int64_t>)`: Optional compile-time seed value. When set, the backend uses this seed value instead of seed and offset tensor descriptors.

## Returns

- `Y` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Output tensor attributes for generated random values. Mark it as an output when the random tensor should be written to a user buffer.

## Errors

`Graph::rng()` appends a node and returns tensor attributes. Shape inference failures, missing output dimensions, unsupported distributions other than Bernoulli, missing Bernoulli probability, invalid seed or offset tensors, and backend failures are reported by later graph lifecycle `error_t` statuses.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** 8.5+ for the C++ frontend; Python wrapper paths require 9.12+
- **Family CUDA:** A CUDA toolkit supported by the installed cuDNN release
- **Family GPU:** Architecture supported by the selected cuDNN engine
- **Family platforms:** Linux (executed); Windows (source-supported, not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** Python and C++ representative samples executed with CUDA 13, cuDNN 9.15.1, and SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

This is a low-level C++ graph helper. It is used internally by SDPA dropout paths to generate dropout masks. For SDPA, prefer the higher-level `set_dropout(...)` and `set_rng_dump(...)` attributes unless you are constructing a custom graph.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node_interface.h#L486) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node/rng.h#L171).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/advanced_operations.cc)

<Markdown src="/snippets/generated/cpp/advanced_operations/rng.mdx" />

## Appendix: Python equivalent

There is no direct Python graph binding documented for `rng()` in `docs/api/python`.
