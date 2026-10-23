# tensor_like

Create a graph tensor by cloning existing tensor metadata or by importing metadata from a DLPack object.

## Python synopsis

```python
Y = graph.tensor_like(input, name="")
Z = graph.tensor_like(dlpack_object)
```

The first overload copies metadata from another `cudnn.tensor`. The second overload creates a graph tensor from an object that exposes `__dlpack__`, such as a PyTorch tensor.

## Parameters

- `input` (`cudnn.tensor`, required for the first overload): Existing graph tensor attributes to copy. The copy keeps shape, stride, dtype, virtual state, pass-by-value state, ragged-offset metadata, and reordering metadata, but receives a fresh frontend UID and the supplied `name`.

- `name` (`str`, default `""`): Name assigned to the cloned tensor for the `tensor_like(input, name)` overload. Names are not copied from `input`.

- `dlpack_object` (object with `__dlpack__`, required for the second overload): Runtime object whose DLPack capsule supplies shape, stride, dtype, and device metadata. If the DLPack strides pointer is `NULL`, the binding treats the object as compact row-major and generates row-major strides. CPU DLPack tensors are marked pass-by-value; CUDA and CUDA-managed tensors are regular runtime-buffer tensors.

## Returns

- `Y` or `Z` (`cudnn.tensor`): A graph tensor attribute registered as a graph input. A tensor imported from DLPack is non-virtual and infers dtype, dimensions, and strides from the capsule.

## Raises

- `TypeError`: Raised immediately when the arguments do not match either overload or when `name` cannot be converted to `str`.

- `ValueError`: Raised when the DLPack object does not expose `__dlpack__`, the capsule is invalid, the DLPack device type is not CPU, CUDA host, CUDA device, or CUDA managed memory, or later validation finds malformed tensor metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by later support checking or plan building when the imported dtype, shape, stride, or consuming operation is unsupported.

- `RuntimeError`: Raised by later lifecycle methods for CUDA, cuDNN backend, handle, invalid value, or execution failures.

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

`tensor_like(input, name)` is useful when two graph inputs share shape and layout but must remain distinct variant-pack entries. The clone intentionally does not copy the UID or name.

`tensor_like(dlpack_object)` is a metadata convenience. It still does not make the graph own the runtime buffer; the object or another buffer with compatible layout must be supplied when executing the graph.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L479) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L479).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/tensor_like_basic.py)

<Code
  src="../../samples/python/tensor_like_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::tensor_like` reference](/api/cpp/tensor-like) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
