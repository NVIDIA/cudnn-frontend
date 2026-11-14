# GNN heterogeneous basis aggregation (post)

`cudnn.gnn.agg_hg_basis_post` aggregates source-node features by edge type,
then combines the per-type aggregates into basis outputs. It calls cuDNN's
`cudnnGnnAggHgBasisPostForward` and `cudnnGnnAggHgBasisPostBackward` APIs.
The name "post" means that feature transformations may be applied after this
aggregation, as in a basis-regularized relational graph convolution.

## Mathematical formula

For destination `v` and edge type `t`, let `N_t(v)` be its incoming source
nodes connected by edges of type `t`. The operation computes

```text
a[v, t] = sum(x[u] for u in N_t(v))
```

With `norm_by_edge_type=True`, divide each nonempty per-type sum by
`len(N_t(v))`. A type with no incoming edges contributes zeros.
When combination weights are supplied:

```text
y[v, b] = sum(weights_comb[t, b] * a[v, t] for t in range(num_edge_types))
```

Otherwise, `y[v, t] = a[v, t]`. The output concatenates these basis groups.
With `concat_own=True`, it appends `x[v]` after the basis groups.

## Convention and notation

The graph is directed and stored in destination-major CSC order.
`edge_types[e]` describes CSC edge `e`, with values in
`[0, num_edge_types)`. Edge types always use **direct CSC order**;
`graph.map_csc_to_coo` does not remap them. Reverse-CSC metadata is not used.
The caller is responsible for valid graph indices and edge-type values;
the wrapper validates tensor metadata without scanning tensor contents.

## Example

```python
import torch
from cudnn.gnn import CscGraph, agg_hg_basis_post

graph = CscGraph(
    torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
    torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
    num_src_nodes=3,
)
edge_types = torch.tensor([0, 1, 1, 0], device="cuda", dtype=torch.int32)
node_features = torch.randn(3, 16, device="cuda", requires_grad=True)
weights_comb = torch.randn(2, 3, device="cuda", requires_grad=True)
output = agg_hg_basis_post(
    graph,
    node_features=node_features,
    edge_types=edge_types,
    num_edge_types=2,
    weights_comb=weights_comb,
    concat_own=True,
    norm_by_edge_type=True,
)
output.sum().backward()
```

## API

```python
agg_hg_basis_post(
    graph,
    *,
    node_features,
    edge_types,
    num_edge_types,
    weights_comb=None,
    concat_own=False,
    norm_by_edge_type=False,
)
```

- `graph` is a `CscGraph` with CUDA INT32 or INT64 offsets and indices.
- `node_features` is FP32 with shape `(num_src_nodes, node_feat_dim)`.
- `edge_types` is INT32 with shape `(num_edges,)`, even for INT64 graphs.
- `num_edge_types` is positive and includes any unused edge types.
- Optional FP32 `weights_comb` has shape `(num_edge_types, num_bases)`.
  Its second dimension determines the number of bases. Without weights, the
  number of bases equals `num_edge_types`.
- `concat_own=True` appends `node_features[:num_dst_nodes]`. It requires
  `num_dst_nodes <= num_src_nodes` and assumes destination nodes occupy that
  source-feature prefix. Without concatenation, no prefix correspondence is
  required.
- `norm_by_edge_type=True` averages incoming features separately for each
  destination and edge type. Its default is False (sum).
- The output has shape
  `(num_dst_nodes, node_feat_dim * (effective_bases + int(concat_own)))`.

All tensors must be on the graph's CUDA device. Noncontiguous input tensors
are accepted and converted to contiguous buffers before backend calls.

## Autograd and determinism

The wrapper registers first-order PyTorch autograd for node features and
optional combination weights. Either gradient may be requested separately.
Graph metadata and edge types are not differentiable.

The backend initializes requested gradient buffers before atomic accumulation.
Backward does not guarantee repeatable gradients and has no deterministic
mode. Higher-order gradients are not supported. Features, combination weights,
outputs, and gradients are FP32; there is no gradient-precision flag.

## Support

The loaded backend must provide the AggHgBasisPost APIs in cuDNN 9.29 or
newer. Dynamic loading permits building the Python bindings with older
supported cuDNN headers. The operation requires SM 8.0 (Ampere) or newer and
is unavailable on Windows. Some large edge-type/basis combinations exceed the
GPU's shared-memory capacity and are rejected by the backend.

A graph with no destination nodes is handled by the wrapper and produces
empty output and zero input gradients. A nonempty destination set with no
edges is unsupported. Zero-degree destinations within a graph that has edges
produce zero basis groups and retain the optional own-feature group.
