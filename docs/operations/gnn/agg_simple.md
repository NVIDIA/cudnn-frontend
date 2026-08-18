# GNN simple aggregation

`cudnn.gnn.agg_simple` performs neighborhood aggregation over
a graph stored in compressed sparse column (CSC) format. It supports node
features, edge features, and per-destination features that are appended to the
result without aggregation.

The implementation calls cuDNN's `cudnnGnnAggSimpleForward` and
`cudnnGnnAggSimpleBackward` APIs. The first call for each feature/index dtype
combination may JIT-compile an NVRTC kernel; later calls reuse a process-wide
cache.

## Mathematical formula

For destination vertex  incoming CSC neighbors (v) node features
 edge features  and optional destination features  the
operation computes

```text
y_v = concat(reduce({x_u | u in N(v)}),
             reduce({e_uv | u in N(v)}),
             c_v)
```

where `reduce` is SUM, MEAN, MAX, or MIN. Omitted feature groups are omitted
from the concatenation.

## Convention and notation

The graph is directed and represented in CSC destination-major order.
`offsets[v]:offsets[v + 1]` selects the incoming edges for destination `v`;
`indices` stores their source vertices. `map_csc_to_coo`, when present, maps
each CSC edge position to the corresponding edge-feature row.

## Example

```python
import torch
from cudnn.gnn import CscGraph, agg_simple

offsets = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
indices = torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32)
graph = CscGraph(offsets, indices, num_src_nodes=3)

node_features = torch.randn(3, 16, device="cuda", requires_grad=True)
output = agg_simple(graph, node_features=node_features, aggr="mean")
output.sum().backward()
```

## API

```python
agg_simple(
    graph,
    *,
    node_features=None,
    edge_features=None,
    concat_features=None,
    aggr="sum",
)
```

- `graph` is a `CscGraph` containing CUDA `int32` or `int64` offsets and
  indices. `num_src_nodes` is explicit because it cannot always be inferred
  from CSC metadata without synchronizing the GPU.
- `node_features` has shape `(num_src_nodes, node_feat_dim)`.
- `edge_features` has shape `(num_edges, edge_feat_dim)`.
- `concat_features` has shape `(num_dst_nodes, concat_feat_dim)` and is copied
  to the final columns of the output.
- `aggr` is `"sum"`, `"mean"`, `"max"`, or `"min"`.
- The output shape is
  `(num_dst_nodes, node_feat_dim + edge_feat_dim + concat_feat_dim)`.

At least one of `node_features` and `edge_features` is required. Feature
tensors must use the same CUDA device and one of FP32, FP16, or BF16.

If edge-feature rows are not already in CSC order, set
`CscGraph.map_csc_to_coo` to an index tensor of shape `(num_edges,)` that maps
each CSC edge position to its edge-feature row.

Compatibility helpers matching the common `pylibcugraphops` entry points are
also available:

```python
agg_simple_n2n(node_features, graph, aggr="sum")
agg_simple_e2n(edge_features, graph, aggr="sum")
agg_simple_n2n_e2n(node_features, edge_features, graph, aggr="sum")
```

Weighted aggregation is not included because it is a separate operation and
is not part of the cuDNN AggSimple C API.

## Autograd and determinism

The Python API registers a PyTorch autograd formula. For max/min reductions,
forward saves the winning source/edge positions required by backward.

Backward uses atomic additions for node gradients and for concat gradients to
support aliasing those backend output buffers. The Python frontend temporarily
initializes all gradient outputs before calling the backend; this workaround
can be removed once cuDNN performs that initialization internally. Atomic
accumulation is nondeterministic. Higher-order gradients are not supported.

## Support

### cuDNN backend

The bindings are compiled only when the selected cuDNN headers declare both
AggSimple APIs, and they additionally verify that the loaded cuDNN library
exports both symbols at runtime:

```python
import cudnn
print(cudnn.is_gnn_agg_simple_available())
```

For development against a local cuDNN build, select both its headers and
libraries before installing cuDNN Frontend:

```bash
export CUDNN_INCLUDE_PATH=/workspace/repo/cudnn/include
export CUDNN_LIBRARY_PATH=/workspace/repo/cudnn/out/build/linux-x86_64-13.2--debug
export LD_LIBRARY_PATH="$CUDNN_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pip install -e /workspace/repo/cudnn-frontend
```

The current backend requires SM 8.0 or newer. A graph with zero destination
nodes is handled by the Python wrapper. A graph with destination nodes but no
edges is not yet supported by the backend binding.


## Benchmark

The benchmark separates reusable model shapes, execution, and command-line
orchestration under `benchmark/gnn/agg_simple`:

```bash
python benchmark/gnn/agg_simple/harness.py \
    --shape medium \
    --dtype float32 \
    --aggr sum \
    --backward
```

It reports mean forward latency, optional backward latency, and forward
edge throughput after configurable warmup iterations.

## Blog

See the [cuDNN Frontend project blog](https://nvidia.github.io/cudnn-frontend/)
for performance articles and release updates.
