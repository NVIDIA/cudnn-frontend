# Glossary

**Backend graph**: the cuDNN operation graph representation created from frontend tensor and operation attributes.

**Block scaling**: quantization in which one scale applies to a fixed-size block of values rather than an entire tensor.

**Compute data type**: precision used by an operation's computation, often FP32 even when I/O is FP16 or BF16.

**Execution plan**: an engine and configuration selected for a validated graph.

**Graph representation creation**: conversion from frontend tensor and operation attributes into the cuDNN backend operation graph.

**Frontend workspace**: temporary storage owned by graph execution in addition to any backend plan workspace.

**Intermediate data type**: default type for virtual tensors between operations.

**I/O data type**: default type for non-virtual graph inputs and outputs.

**Pass-by-value tensor**: a scalar embedded or supplied from host memory rather than a device allocation.

**Ragged offset**: a tensor describing variable-length storage offsets, optionally scaled by an element multiplier.

**Routed plan**: a top-level Python `PlanConfig` selecting an in-tree Python engine or a cuDNN backend candidate; its combined-list index is not a backend plan index.

**Tensor UID**: stable integer used to match graph tensors to runtime pointers.

**Variant pack**: the runtime mapping from tensor descriptors or UIDs to device buffers plus workspace.

**Virtual tensor**: an internal graph value that does not require a caller-owned device buffer.
