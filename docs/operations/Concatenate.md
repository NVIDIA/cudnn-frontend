# Concatenate

The Concatenate operation merges two or more tensors into one, along the specified axis.  The user may also specify an in-place merge.  The operation provides the capabilities of the {ref}`cudnn backend's concatenate operation                          <CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR>`.

## C++ API

```
std::shared_ptr<Tensor_attributes>
concatenate(std::vector<std::shared_ptr<Tensor_attributes>>, Concatenate_attributes);
```

Concatenate attributes is a lightweight structure with inputs, outputs, and setters:

```
std::vector<std::shared_ptr<Tensor_attributes>> inputs;

std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

Concatenate_attributes&
set_axis(int64_t const value)

Concatenate_attributes&
set_in_place_index(int64_t const value)
```

## In-place concat (optional)

`set_in_place_index` is **optional**. When unset, the backend writes the concatenation result only into the output tensor. When set to index `k`, the concatenate operation may reuse the buffer of input `k` as documented for `CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX` (see the cuDNN backend concatenate descriptor). Callers must still provide a distinct output tensor in the graph API; layout and lifetime rules follow backend requirements for the chosen index.

## Python API

- `pygraph.concatenate(inputs, axis, name="", in_place_index=None)`
  - **inputs**: list of tensors to concatenate.
  - **axis**: dimension index to concatenate along.
  - **in_place_index**: optional `int`; omit for out-of-place behavior.

## Requirements

Graph concatenate lowering requires a sufficiently new cuDNN backend (see frontend release notes for the minimum version).
