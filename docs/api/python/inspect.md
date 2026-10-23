# inspect

Return a JSON-compatible description of the Python graph IR.

## Python synopsis

```python
graph.inspect() -> dict[str, Any]
```

## Parameters

This method takes no parameters.

## Returns

A dictionary containing graph context, tensor records, node records, and lifecycle state suitable for diagnostics and engine support checks.

## Raises

The method does not normally raise. Values supplied by custom node parameters must still be representable by the caller if it serializes the returned dictionary.

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

The structure is an inspection view, not the stable cuDNN serialization format. Do not persist it as a replacement for `serialize()`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L899) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L899).
{/* END GENERATED API SOURCES */}

## Examples

<Markdown src="/snippets/generated/python/custom_engine/inspection.mdx" />

## Appendix: C++ equivalent

C++ `Graph::print()` and stream output provide diagnostic text; `serialize()` provides the backend persistence format.
