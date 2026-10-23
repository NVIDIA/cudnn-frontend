# tensors

Return graph tensors indexed by their unique names.

## Python synopsis

```python
graph.tensors -> dict[str, cudnn.Tensor]
```

## Parameters

This read-only property takes no parameters.

## Returns

A new dictionary mapping each assigned tensor name to the graph-owned tensor object.

## Raises

Property access does not raise under normal use. Duplicate names are rejected when tensors are added or renamed.

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

Mutating the returned dictionary does not mutate the graph. Tensor objects remain graph-owned and become immutable when planning freezes the graph.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L869) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L869).
{/* END GENERATED API SOURCES */}

## Examples

<Markdown src="/snippets/generated/python/custom_engine/inspection.mdx" />

## Appendix: C++ equivalent

C++ callers retain `shared_ptr<Tensor_attributes>` objects; the public Graph API does not expose a name-indexed tensor dictionary.
