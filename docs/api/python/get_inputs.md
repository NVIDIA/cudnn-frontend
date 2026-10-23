# get_inputs

Return the graph tensors that must be supplied as execution inputs.

## Python synopsis

```python
graph.get_inputs() -> list[cudnn.Tensor]
```

## Parameters

This method takes no parameters.

## Returns

Non-virtual tensors consumed by a node and not produced by another node. The returned list is ordered by graph tensor insertion order.

## Raises

The method does not raise under normal use.

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

Pass-by-value and ragged-offset tensors can also require execution bindings even when they are not conventional data inputs. Use the variant-pack UID query for the exact backend binding set in the graph representation.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L890) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L890).
{/* END GENERATED API SOURCES */}

## Examples

<Markdown src="/snippets/generated/python/custom_engine/inspection.mdx" />

## Appendix: C++ equivalent

C++ callers normally retain the input `Tensor_attributes` returned during graph construction and bind them in the variant pack.
