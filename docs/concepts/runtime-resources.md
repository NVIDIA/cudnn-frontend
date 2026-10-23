# Runtime resources and asynchronous execution

Graph construction describes reusable computation. Execution supplies context and storage for one invocation.

## Handles and streams

A cuDNN handle selects a CUDA stream and carries backend context. Changing its stream affects subsequent work. The high-level Python wrapper binds its managed handle to PyTorch's current stream; explicit APIs use caller-supplied handles.

## Variant packs

A variant pack maps every required non-virtual UID to a device pointer and includes workspace. Tensor descriptors do not own those buffers. Pass-by-value and ragged-offset values can add bindings beyond obvious inputs and outputs.

## Workspace

Workspace is temporary storage sized for a built plan. It can be reused after prior asynchronous work completes. Concurrent executions need independent storage unless the plan contract explicitly permits sharing.

## Lifetime and synchronization

Execution normally enqueues work and returns before completion. Keep graph, plan, handle, input, output, scalar, auxiliary, and workspace resources alive until the stream reaches the operation. Synchronize only for host access, timing, error localization, or dependency not already expressed by streams and events.

## CUDA Graph interaction

CUDA Graph capture adds stronger address and lifetime requirements. Resources captured by address must remain stable across replay; unsupported changes require repopulation.
