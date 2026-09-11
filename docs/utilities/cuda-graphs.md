# CUDA Graphs

## `populate_cuda_graph`

The `populate_cuda_graph` function is a member function of the `Graph` class. It is used to populate a CUDA graph with the necessary data and operations.

### Parameters

- `handle`: A cuDNN handle.
- `uid_to_device_ptrs`: A map of tensor UIDs to device pointers.
- `workspace`: A pointer to the workspace memory.
- `cudnn_cuda_graph`: A pointer to the CUDA graph.

### Return Value

- An `error_t` object indicating the success or failure of the function.

## `update_cuda_graph`

The `update_cuda_graph` function is a member function of the `Graph` class. It is used to update a CUDA graph with the necessary data and operations.

### Parameters

- `handle`: A cuDNN handle.
- `uid_to_device_ptrs`: A map of tensor UIDs to device pointers.
- `workspace`: A pointer to the workspace memory.
- `cudnn_cuda_graph`: A pointer to the CUDA graph.

### Return Value

- An `error_t` object indicating the success or failure of the function.
