# serialize

Serialize graph and plan state to a Python list of bytes.

## Python synopsis

```python
data = graph.serialize()
```

## Parameters

- This method takes no Python arguments.

## Returns

- `data` (`list[int]`): Serialized graph data. The list contains byte values and can be passed to `graph.deserialize(...)` in a compatible process and build.

## Raises

- `TypeError`: Raised immediately when Python arguments cannot be converted to the bound C++ types, for example an invalid enum, a non-sequence list argument, or a non-integer plan index.

- `ValueError`: Raised for frontend validation errors that map to `std::invalid_argument`, such as missing tensor attributes, shape-deduction failures, invalid tensor names, or invalid variant-pack metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised when a well-formed graph or plan candidate is not supported by the selected heuristics, backend implementation, architecture, cuDNN version, or graph format.

- `RuntimeError`: Raised for cuDNN backend API failures, CUDA API failures, handle errors, graph execution failures, invalid values reported by the frontend, NVRTC compilation failures, or invalid device state.

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

Serialization records graph structure, selected plan data, behavior notes, variant-pack UIDs, pass-by-value data, frontend workspace metadata, and tensor metadata needed by deserialization. Builds compiled without the JSON library report serialization as unsupported.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L2597) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L2597).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/serialize_basic.py)

<Code
  src="../../samples/python/serialize_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::serialize` reference](/api/cpp/serialize) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
