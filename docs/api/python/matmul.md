# matmul

Append a matrix multiplication operation to a PyGraph.

## Python synopsis

```python
C = graph.matmul(
    A,
    B,
    compute_data_type=cudnn.data_type.NOT_SET,
    padding=0.0,
    name="",
)
```

For the innermost two matrix dimensions, `matmul` computes:

```text
C[..., m, n] = sum_k A[..., m, k] * B[..., k, n]
```

Leading dimensions are batch dimensions and must be compatible with the cuDNN
backend plan selected during graph build.

## Parameters

- `A` (`cudnn.tensor`, required): Left input tensor. The last logical dimension
  is the contracting `K` dimension. The preceding dimension is the `M`
  dimension. Rank-2 and batched rank-3 or higher forms are accepted when the
  backend supports the corresponding layout.

- `B` (`cudnn.tensor`, required): Right input tensor. The second-to-last logical
  dimension is the contracting `K` dimension and must match `A`'s `K`. The last
  logical dimension is `N`.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Accumulation and
  compute type for the operation. Common examples use `FLOAT` accumulation for
  FP16, BF16, FP8, and INT8 inputs. If `NOT_SET`, the graph-level compute type is
  used when available.

- `padding` (`float`, default `0.0`): Value used by the low-level matmul
  attributes for padded elements in layouts that require backend padding. Most
  ordinary dense examples leave this at `0.0`.

- `name` (`str`, default `""`): Operation name used to name the generated output
  tensor.

## Returns

- `C` (`cudnn.tensor`): Graph tensor for the matmul output. The logical shape is
  inferred from `A` and `B`, usually `broadcast_batch_shape + (M, N)`. Set
  `C.set_output(True)` and an output dtype when it should be materialized in the
  variant pack.

## Raises

- `TypeError`: Raised immediately when `A`, `B`, `compute_data_type`, `padding`,
  or `name` cannot be converted to the bound C++ types.

- `ValueError`: Raised by `graph.validate()` for incompatible tensor metadata,
  invalid shape deduction, or missing output metadata required by later nodes.

- `cudnn.cudnnGraphNotSupportedError`: Raised by `graph.check_support()` or
  `graph.build_plans()` when no backend engine supports the dtype, layout,
  alignment, architecture, or dimensions.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
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

`A`, `B`, and `C` do not need identical strides. Each tensor only needs strides
accepted by the selected matmul engine. Examples in this repository use
row-major dense tensors, column-major weight tensors, sliced tensors, and
casted pointwise intermediates before matmul.

The output dtype is not necessarily the compute dtype. For mixed precision,
set `compute_data_type=cudnn.data_type.FLOAT` and set the returned output tensor
to the desired IO dtype.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L693) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L693).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/matmul_basic.py)

<Code
  src="../../samples/python/matmul_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::matmul` reference](/api/cpp/matmul) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
