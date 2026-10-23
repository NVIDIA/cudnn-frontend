# select_plan

Select one entry from the ranked routed-plan list.

## Python synopsis

```python
graph.select_plan(index: int) -> cudnn.pygraph
```

## Parameters

- `index`: Zero-based index into `graph.plans`, after `create_execution_plans()`.

## Returns

The graph, enabling fluent calls.

## Raises

- `RuntimeError`: Raised before routed plans are created.
- `IndexError`: Raised when `index` is outside `0 <= index < len(graph.plans)`.

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

Changing the routed plan invalidates its built state. Call `check_support()` and `build_plans()` for the newly selected entry before execution. Backend sub-plan indices use separate delegated APIs.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L1437) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L1437).
{/* END GENERATED API SOURCES */}

## Examples

<Markdown src="/snippets/generated/python/custom_engine/routing.mdx" />

## Appendix: C++ equivalent

Use `Graph::build_plan_at_index` and `Graph::execute_plan_at_index` for C++ backend plan indices; the index spaces are not interchangeable.
