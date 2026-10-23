# find_tensor

Find a graph tensor by its name or numeric UID.

## Python synopsis

```python
graph.find_tensor(name_or_uid: str | int) -> Optional[cudnn.Tensor]
```

## Parameters

- `name_or_uid`: An exact tensor name or integer UID. Names and UIDs are graph-local identifiers.

## Returns

The matching `cudnn.Tensor`, or `None` when the graph has no match.

## Raises

- `TypeError`: Raised when the argument is neither `str` nor `int`.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Python IR routing and custom engines**

- **API disposition:** Unreleased
- **API stability:** Stable
- **API language:** python
- **API-specific availability:** Current checkout
- **Family release status:** Unreleased
- **Family cuDNN:** 9.12+ when falling back to the backend
- **Family CUDA:** CUDA 13 tested for this checkout
- **Family GPU:** Depends on the selected engine
- **Family platforms:** Linux (executed); Windows (not executed)
- **Family backends:** Python engine; cuDNN fallback
- **Family languages:** Python
- **Family optional and runtime dependencies:** Python engine implementation
- **Recorded verification:** Automatic manifest discovery, ranked selection, and backend fallback are covered by the engine-selection and core lifecycle samples
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

This lookup does not create tensors, create the backend graph representation, or freeze the graph. Prefer a direct tensor variable when it is already available.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L878) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L878).
{/* END GENERATED API SOURCES */}

## Examples

<Markdown src="/snippets/generated/python/custom_engine/inspection.mdx" />

## Appendix: C++ equivalent

Use `Graph::query_tensor_attributes_of_uid` for a C++ UID lookup after the operation graph is built.
