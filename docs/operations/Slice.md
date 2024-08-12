## Table of Contents
1. [Slice](#Slice)

### Slice
Slice operation extracts a portion of a tensor:
$$ Y = X[start_0:end_0, start_1:end_1, ..., start_n:end_n] $$
Where $X$ is the input tensor, $Y$ is the output tensor, and $start_i$ and $end_i$ are the start and end indices for the $i$-th dimension.

The operation allows for flexible slicing across any number of dimensions, supporting Python-style slice syntax including start, stop, and step parameters.

The API to achieve the above is:
```cpp
std::shared_ptr<Tensor_attributes>
Slice(std::shared_ptr<Tensor_attributes> input, Slice_attributes);
```

Slice attributes is a lightweight structure with setters:
```cpp
Slice_attributes&
set_slices(std::vector<std::pair<int64_t, int64_t>> const value)

Slice_attributes&
set_name(std::string const&)

Slice_attributes&
set_compute_data_type(DataType_t value)
```


### Python API:
- slice
    - input
        - The input tensor to be sliced
    - slices
        - A list of Python slice objects, one for each dimension
    - name
        - Optional name for the operation
    - compute_data_type
        - Optional compute data type for the operation

Example usage:

```python
# Create an input tensor

input_tensor = graph.tensor(dims = [4, 8, 16])

# Perform slicing
sliced_tensor = graph.slice(input_tensor, 
                            slices=[slice(1, 3), slice(2, 6), slice(0, 16)],
                            name="my_slice",
                            compute_data_type=cudnn.float32)
```
