# Transpose

The transpose operation permutes the dimensions of a tensor according to a **permutation** vector. For each output axis `i`, the value `permutation[i]` is the source axis taken from the input (so `output_dim[i] = input_dim[permutation[i]]`, and strides are permuted the same way).

On supported cuDNN versions, the graph lowers to the backend **transpose** operation (`CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR`).

## Requirements

- Native backend transpose support requires **cuDNN 9.22.0** or newer. Older toolkits may report `GRAPH_NOT_SUPPORTED` for transpose nodes.

## FROST GEMM operand views (Python)

With `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`, the FROST GEMM provider can absorb
identity or **last-two-axis** transposes feeding a single plain matmul. Rank-2,
rank-3 and higher-rank matrices are supported: `[1, 0]` for rank-2, `[0, 2, 1]`
for rank-3, `[0, 1, 3, 2]` for rank-4, and so on. Leading batch axes stay in order.
Transposes may precede, follow or separate supported operand pointwise operations;
successive transposes compose. For example, `transpose(relu(X))` is lowered as
`relu` on a transposed input view, with no transpose kernel or intermediate storage.

Rank-2 inputs get an implicit unit batch axis. Higher-rank leading axes are folded
into one batch only when their strides describe an **affine** linear batch pitch.
Gaps between matrices, padded row pitches and transposed matrix layouts are allowed;
no copying reshape is used. Each operand's batch axes must either match all output
batch axes or all broadcast from one. Partial/crossed batch broadcasting such as
`[2,1,M,K] @ [1,3,K,N]` is declined because one batch stride cannot express it.
The prepared batch shape and original rank stay fixed; M/N/K can be rebound.
Rank-normalized outputs must have non-overlapping storage.

Each pointwise node keeps its compute dtype, output dtype, attributes and rounding
boundary. Transpose itself preserves dtype. Bind the **original graph inputs** in
their declared coordinates, including when supplying raw addresses or live
shape/stride overrides. No caller-side transpose or contiguous conversion is needed.
Scalar and supported per-K operands are projected into the same coordinates.
Existing GEMM layout, alignment, architecture and pointwise support gates still apply.

This lowering does not require the backend transpose descriptor, but it requires
a supported FROST GEMM engine. It rejects materialized operand intermediates,
epilogue/unrelated transposes, permutations moving batch axes, dtype-changing transposes,
MoE, block-scale dequantization and multi-GEMM graphs. General reshape normalization
is not included. Rank-normalized (rank-2/rank-N) graphs currently support pointwise
operand and epilogue fusion, not quantization or reductions; existing rank-3 routes
retain their support gates. Both GEMM operands must have the same original rank.

See the self-checking [transpose mainloop example](https://github.com/NVIDIA/cudnn-frontend/blob/develop/samples/frost/gemm/08_matmul_transpose_mainloop.py).

## C++ API

```cpp
std::shared_ptr<Tensor_attributes>
transpose(std::shared_ptr<Tensor_attributes> input, Transpose_attributes attributes);
```

`Transpose_attributes` provides:

```cpp
Transpose_attributes&
set_permutation(std::vector<int64_t> const& value);

Transpose_attributes&
set_name(std::string const& value);

Transpose_attributes&
set_compute_data_type(DataType_t value);  // optional; types usually follow the input
```

The permutation must have length equal to the input rank, contain each index `0 .. rank-1` exactly once, and describe where each **output** axis reads from in the **input**.

## Python API

- `pygraph.transpose(input, permutation, name="")`
  - **input**: source tensor.
  - **permutation**: list of ints, same semantics as C++.
  - **name**: optional node name.

## Example (C++)

```cpp
auto X = graph.tensor(/* ... */);  // e.g. dims {2, 4}, row-major strides
auto Y = graph.transpose(
    X,
    cudnn_frontend::graph::Transpose_attributes()
        .set_name("t")
        .set_permutation({1, 0}));  // dims become {4, 2}
Y->set_output(true);
```
