# Slice

The slice operation extracts a strided window of a tensor along each dimension:

$$ Y = X[\text{start}_0:\text{limit}_0:\text{step}_0,\ \ldots] $$

where each dimension uses a half-open interval `[start, limit)` and an optional **slice stride** (step) per axis. The output length along axis `i` is `ceil((limit_i - start_i) / step_i)` when `step_i > 0`.

## Backend operation (cuDNN 9.22+)

On **cuDNN 9.22.0** and newer, the node can lower to the native backend **slice** operation and materializes both input and output tensor descriptors. On older versions, the frontend may use a pointer-offset style lowering for compatible cases.

## C++ API

```cpp
std::shared_ptr<Tensor_attributes>
Slice(std::shared_ptr<Tensor_attributes> input, Slice_attributes);
```

`Slice_attributes` setters:

```cpp
Slice_attributes&
set_slices(std::vector<std::pair<int64_t, int64_t>> const value);  // [start, limit) per dim

Slice_attributes&
set_strides(std::vector<int64_t> const value);  // step per dim; must be > 0; defaults to 1

Slice_attributes&
set_name(std::string const&);

Slice_attributes&
set_compute_data_type(DataType_t value);
```

If `set_strides` is omitted, every step defaults to `1`. If provided, the vector must cover each sliced dimension consistently with `set_slices`.

## Python API

- `slice(input, slices=[], compute_data_type=cudnn.data_type.NOT_SET, name="")`
  - **input**: tensor to slice.
  - **slices**: list of `slice` objects, one per dimension. Per-axis step is taken from each slice’s `step` (default `1`), after normalization for that axis length (same idea as `sequence[sl]` in Python).

Example:

```python
input_tensor = graph.tensor(dim=[4, 8, 16], stride=[128, 16, 1], data_type=cudnn.data_type.float)
sliced_tensor = graph.slice(
    input_tensor,
    slices=[slice(1, 3), slice(2, 6, 2), slice(0, 16)],
    name="my_slice",
    compute_data_type=cudnn.data_type.float,
)
```
