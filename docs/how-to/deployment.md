# Serialize, precompile, and deploy graphs

Prepare graphs on one system and load them in another only when the serialization format and target properties are compatible.

## Before you begin

Begin with a graph that already builds and passes numerical checks. Record the
frontend, backend, CUDA, and target-device versions before treating serialized
bytes as a deployable artifact.

## Choose a representation

`serialize()` preserves the cuDNN backend graph representation. Create a compatible `pygraph`, then call its `deserialize()` method to reconstruct that representation. Python engine objects and arbitrary inspection dictionaries are not part of the format.

<Code
  src="../samples/python/serialize_basic.py"
  language="python"
  maxLines={0}
/>

## Record target properties

Capture frontend and backend versions, CUDA runtime, GPU architecture, device properties, graph key, operation attributes, and build policy. Structure-free or deviceless ahead-of-time compilation must receive explicit target SM and device properties instead of reading the current device implicitly.

## Run the three-stage deviceless flow

1. On a representative target device, construct `DeviceProperties`, initialize it with `set_device_id(device)` or `set_handle(handle)`, call `build()`, and serialize the descriptor. This stage requires the device.
2. On the build host, deserialize those bytes into a new `DeviceProperties`, attach it with `graph.set_device_properties(...)`, describe the graph, build a plan with the no-handle lifecycle, and serialize the graph or plan. This stage is intended to run without the target device.
3. On a device with matching properties, deserialize with a live cuDNN handle, query workspace, bind variant-pack pointers, and execute. Reject a property mismatch instead of assuming that the serialized plan will re-specialize.

Device-property bytes and graph bytes are separate artifacts. Version them together with the target commit, cuDNN backend version, build policy, and operation metadata. A serialized `DeviceProperties` object describes the target; it does not prove that every operation has a deviceless heuristic or compiler path.

## Control warmup

Use warmup to materialize lazy kernels or caches before latency-sensitive service. Bind representative, valid buffers and workspace. Warmup does not replace correctness testing and may not cover every dynamic specialization.

## Validate on load

Reject incompatible metadata before execution. Rebuild when serialization or device compatibility cannot be established. Keep a source-graph path available for cache misses and deployment upgrades.

References: [`serialize`](/api/python/serialize), [`deserialize`](/api/python/deserialize), [`warmup`](/api/cpp/warmup), and [`set_device_properties`](/api/cpp/set-device-properties).

## Verification limits

The recorded sample verified serialization in one process on a single-GPU
host. Cross-process artifact transfer and a genuinely deviceless build stage
were not exercised, so that run does not establish deployment isolation.

## Expected result

The serialization sample round-trips a graph and executes it on a compatible
target. Production loading must reject incompatible metadata and fall back to
rebuilding; the published verification does not establish a cross-process or
genuinely deviceless deployment.
