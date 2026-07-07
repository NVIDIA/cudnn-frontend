# Deviceless Ahead-of-time Compilation

## Ahead-of-time Compilation
Since cuDNN 9.8, customers are allowed to create and finalize an execution plan devicelessly with a device property descriptor. This helps customers to cover the plan build time ahead of the exection.

Typical workflow:
- Create a device property descriptor from device, serialize it out. This requires the device.
- Deserialize the device property, create an execution plan from it as well as the computation graph, serialize the plan out. This doesn't require the device.
- Deserialize and execute the execution plan on devices with the same properties. This requires the device.

Refer to the corresponding C++ sample in [samples/cpp/misc/deviceless_aot_compilation.cpp](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/cpp/misc/deviceless_aot_compilation.cpp).

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
