# Deviceless Ahead-of-time Compilation

## Ahead-of-time Compilation
Since cuDNN 9.8, customers are allowed to create and finalize an execution plan devicelessly with a device property descriptor. This helps customers to cover the plan build time ahead of the execution.

Typical workflow:
- Create a device property descriptor from the device, serialize it out. This requires the device.
- Deserialize the device property, create an execution plan from it as well as the computation graph, serialize the plan out. This doesn't require the device.
- Deserialize and execute the execution plan on devices with the same properties. This requires the device.

Refer to the corresponding C++ sample in [samples/cpp/misc/deviceless_aot_compilation.cpp](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/cpp/misc/deviceless_aot_compilation.cpp).

### What "deviceless" means at each phase

| Phase | What is needed |
|---|---|
| Plan deserialization (`Graph::deserialize(blob)`) | A `DeviceProperties` descriptor — no cuDNN handle required. |
| Execution (`Graph::execute(handle, ...)`) | A cuDNN handle and a compatible device. |

One shared, read-only `DeviceProperties` descriptor can be passed concurrently to `deserialize` from multiple threads, enabling efficient parallel deserialization of cached plans without creating per-thread cuDNN handles. (Each `deserialize` produces an independent `Graph` object; they do not share mutable state.)

### C++ API

```cpp
// 1. Deserialize the plan using a device properties descriptor (no handle).
auto graph_deser = std::make_shared<cudnn_frontend::graph::Graph>();
graph_deser->set_device_properties(device_prop_deserialized);
REQUIRE(graph_deser->deserialize(data_graph).is_good());

// 2. Create a handle only when executing.
cudnnHandle_t handle;
REQUIRE(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);
REQUIRE(graph_deser->execute(handle, variant_pack, workspace).is_good());
cudnnDestroy(handle);
```

The `deserialize(blob)` overload (no handle) is available since cuDNN 9.8 at the API level.
Runtime test coverage follows the existing deviceless sample policy and gates at 9.11.

### Python API

```python
device_prop = cudnn.create_device_properties(0)

graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
    device_property=device_prop,
)
# ... add ops, build, serialize ...
blob = graph.serialize()

# Deserialize handle-less; no handle is created.
graph_deser = cudnn.pygraph(device_property=device_prop)
graph_deser.deserialize(blob)

# Execute with a handle.
graph_deser.execute({X: x_gpu, W: w_gpu, Y: y_gpu}, workspace, handle=handle)
```

## cuDNN Device Properties
cuDNN device property descriptor describes the properties of a GPU device, is serializable and can be used to query cuDNN heuristics / create an execution plan directly without the device to be available.

The API to create a device property descriptor is:
```cpp
auto device_prop = std::make_shared<cudnn_frontend::DeviceProperties>();
```

Ways to initialize the device properties:
```cpp
set_handle(cudnnHandle_t handle);  // initialize from a cuDNN handle
set_device_id(int32_t device_id);  // initialize from a specific device
deserialize(const std::vector<uint8_t>& serialized_buf);  // deserialize from json
```

The API to set a device property descriptor is:
```cpp
graph.set_device_properties(device_prop)
```
