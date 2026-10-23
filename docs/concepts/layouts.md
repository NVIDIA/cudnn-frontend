# Shapes, strides, broadcasting, and data types

Correct graph construction requires both a logical view and an exact storage description.

## Shapes and strides

`dim[i]` is the extent of logical axis `i`. `stride[i]` is the distance in elements between adjacent indices on that axis. Contiguous row-major `[d0,d1,d2]` uses `[d1*d2,d2,1]`. Channels-last, transposed, sliced, and broadcast layouts use different strides without renaming logical axes.

Alignment, vector count, vector dimension, and reorder type can further constrain which engines accept storage. A zero or overlapping stride is legal only where the individual API and engine permit it.

## Broadcasting

Pointwise Python IR aligns shapes from the right. Equal extents are compatible; an extent of `1` broadcasts. Matmul broadcasts leading batch dimensions but treats its final two dimensions as matrix axes. Operation-specific broadcasting can be narrower than general framework rules.

## Data types

IO data type defaults external tensors. Intermediate data type defaults virtual values. Compute data type controls accumulation or arithmetic. An operation-level compute type overrides the graph default, and an output tensor's data type still controls storage.

Low-precision formats add scale layouts, block sizes, amax, reordering, and architecture constraints. Boolean, integer index, and pass-by-value tensors have operation-specific conversion rules.

## Ragged and packed layouts

Ragged offsets describe variable sequence starts, optionally multiplied by an element unit. THD attention packs tokens from several sequences. Block masks, page tables, and block scales use packed representations whose integer entries or bytes must be interpreted by the owning API, not by generic shape rules.
