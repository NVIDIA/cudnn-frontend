# Run dynamic shapes with a kernel cache

Build a graph that accepts documented shape or stride overrides and attach a kernel cache so compatible specializations can be reused.

## Before you begin

Use C++ with cuDNN 9.23 or newer for the override-aware workspace query. Start
from a graph that passes at fixed shapes and choose explicit minimum, maximum,
and incompatible cases for every dimension you intend to vary.

## Enable the mode

Set dynamic-shape or override-shape mode before validation. They are graph construction properties and cannot be enabled after creating the graph representation or after planning freezes the graph.

## Identify overridden tensors

Supply corresponding arrays of UIDs, shapes, and strides at execution. Each UID must belong to the graph, each shape must have the expected rank, and each stride must describe the supplied storage. Omitted tensors retain compiled metadata.

The three arrays must have identical lengths and matching positions. Override every materialized tensor whose metadata changes, including outputs and auxiliary statistics. A changed input shape does not implicitly rewrite the output shape or stride supplied to the backend.

## Query runtime workspace before allocation

For every override set, call the overload of `get_workspace_size(handle, ..., override_uids, override_shapes, override_strides)` before allocating workspace. Then execute with the same three arrays. Runtime specialization can require more or less workspace than the cache shape used to build the graph, so a buffer sized only from parameterless `get_workspace_size()` is not a general bound.

The C++ override workspace query requires cuDNN 9.23 in the current implementation. On older backends, do not assume the build-shape query is sufficient for arbitrary runtime shapes. Either restrict overrides to a separately established bound, rebuild per shape, or upgrade before deploying an unbounded override workflow.

Allocate one workspace per overlapping execution or use a stream-aware pool. Querying the right byte count does not make one mutable buffer safe for concurrent streams.

## Attach a kernel cache

Create a cache with `cudnn.create_kernel_cache()` or the C++ `KernelCache`, attach it before build, and keep it alive while graphs use it. Serialized cache data is backend- and compatibility-sensitive; validate device and backend properties before reuse.

## Bound specialization

Dynamic support is operation and engine specific. Test minimum, maximum, boundary, and incompatible shapes. A successful override on one plan does not establish support for every candidate.

Relevant references: [`set_dynamic_shape_enabled`](/api/cpp/set-dynamic-shape-enabled), [`set_override_shape_enabled`](/api/cpp/set-override-shape-enabled), and [`set_kernel_cache`](/api/cpp/set-kernel-cache).

## Verification limits

No documentation-owned dynamic-shape execution sample is published. The
recorded cuDNN 9.15.1 backend is below the 9.23 gate for the override-aware
workspace query, so the ordering contract was verified from source rather than
by runtime execution.

## Expected result

For every accepted override, the runtime workspace query and execution use the
same UID, shape, and stride arrays, and compatible specializations reuse the
attached cache. This repository does not publish a runtime example, so those
properties must be established on the deployment backend.
