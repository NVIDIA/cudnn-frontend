# context

Return the graph-level data-type and target-device configuration.

## Python synopsis

```python
graph.context -> cudnn.GraphContext
```

## Parameters

This read-only property takes no parameters.

## Returns

The graph's `GraphContext` object. It records IO, intermediate, and compute data types together with optional SM targeting values.

## Raises

Property access does not raise under normal use.

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

The context supplies defaults during validation. Per-operation compute types and per-tensor data types can override the relevant graph defaults.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L874) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L874).
{/* END GENERATED API SOURCES */}

## Examples

<Markdown src="/snippets/generated/python/custom_engine/inspection.mdx" />

## Appendix: C++ equivalent

C++ exposes the corresponding defaults through `Graph::set_io_data_type`, `set_intermediate_data_type`, and `set_compute_data_type`.
