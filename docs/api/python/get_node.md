# get_node

Find a graph node by its exact name.

## Python synopsis

```python
graph.get_node(name: str) -> Optional[cudnn.Node]
```

## Parameters

- `name`: Exact node name assigned by an operation builder.

## Returns

The matching `cudnn.Node`, or `None` when no node has that name.

## Raises

- `TypeError`: Raised when `name` is not a string.

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

Node inspection is most useful to engine authors. Ordinary graph execution should retain operation outputs instead of depending on node internals.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L886) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L886).
{/* END GENERATED API SOURCES */}

## Examples

<Markdown src="/snippets/generated/python/custom_engine/inspection.mdx" />

## Appendix: C++ equivalent

The public C++ Graph API does not provide an equivalent name-based node lookup.
