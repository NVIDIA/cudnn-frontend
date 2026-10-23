# Reshape, transpose, slice, and concatenate

Choose whether a shape change is metadata-only or requires an executable graph operation, then preserve the resulting layout explicitly.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph) and review
[logical shapes and physical strides](/concepts/layouts). For each
operation, derive the expected output dimensions, strides, and values before
building the graph. The standalone transpose, slice, and concatenate examples
use the SM100 memory-bound path and require CUDA 13.1 and cuDNN 9.22 or newer.

## Choose an operation

### Reshape

`reshape` changes dimensions according to `reshape_mode`. A reshape is view-like only when the new dimensions and strides describe the same storage; other modes can materialize or infer a supported layout.

### Transpose

`transpose` permutes logical axes, and its output dimensions and strides follow
the permutation. A framework transpose can be a metadata-only view; the Graph
operation shown below binds a separate output buffer.

### Slice

Slices use start, stop, and step values with half-open intervals. The output extent follows Python-style slice arithmetic in the Python IR. Non-unit or negative steps require explicit support from the selected operation path.

### Concatenate

Concatenate joins compatible inputs along one axis. Non-concatenated extents must match. In-place and out-of-place forms have different aliasing and lifetime constraints; bind every externally materialized input and output.

## Transpose example

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/transpose_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/transpose_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
## Slice example

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/slice_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/slice_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
## Concatenate example

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/concatenate_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/concatenate_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
The [reshape sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/reshape_basic.py) is Python-only. The
current C++ terminal-view modes have backend-version-specific limitations, so
the documentation does not imply a portable C++ equivalent.

## Expected result

On a compatible SM100 system, every paired sample executes and compares output
dimensions and values with an independently constructed result. In the
recorded cuDNN 9.15.1 environment, these six samples were compiled or
syntax-checked because standalone memory-bound execution requires cuDNN 9.22.
View-like reshape or framework transpose does not imply that an executable
Graph transpose, slice, or concatenate aliases its input.
