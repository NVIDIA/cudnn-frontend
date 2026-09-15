# GNN GAT and GATv2 attention

`cudnn.gnn.mha_gat` and `cudnn.gnn.mha_gat_v2` apply multi-head graph attention to homogeneous or bipartite graphs in compressed sparse column (CSC) format. They call the cuDNN `cudnnGnnMhaGat*` and `cudnnGnnMhaGatV2*` APIs and provide PyTorch autograd formulas.

## Example

```python
import torch
from cudnn.gnn import CscGraph, mha_gat

offsets = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
indices = torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32)
graph = CscGraph(offsets, indices, num_src_nodes=3)

src_features = torch.randn(3, 16, device="cuda", requires_grad=True)
attn_weights = torch.randn(32, device="cuda", requires_grad=True)
output, attention = mha_gat(
    graph,
    src_features,
    attn_weights,
    num_heads=2,
    return_attention_weights=True,
)
output.sum().backward()
```

## API

```python
mha_gat(
    graph,
    features,
    attn_weights,
    *,
    edge_features=None,
    dropout_mask=None,
    num_heads=1,
    concat_heads=True,
    activation="leaky_relu",
    activation_alpha=0.2,
    return_attention_weights=False,
    deterministic=False,
)
```

`mha_gat_v2` has the same signature.

Both operations require:

- `features` as a source tensor with shape `(num_src_nodes, dim_node)` for homogeneous graphs, or a `(src_features, dst_features)` tuple with shapes `(num_src_nodes, dim_node)` and `(num_dst_nodes, dim_node)` for bipartite graphs. `dim_node` must be divisible by `num_heads`.
- CUDA FP32, FP16, or BF16 features and weights on the graph's device.

GAT uses flat `attn_weights` with layout `source | destination | edge` and length `2 * dim_node + dim_edge`. Its optional `edge_features` has shape `(num_edges, dim_edge)`, where `dim_edge` is divisible by `num_heads`.

GATv2 uses `attn_weights` with shape `(dim_node,)`. Its optional `edge_features` must have shape `(num_edges, dim_node)`.

`activation` accepts `"linear"`, `"relu"`, `"sigmoid"`, `"tanh"`, `"elu"`, `"scalar"`, or `"leaky_relu"`. `activation_alpha` is the negative slope for leaky ReLU and the scale for the scalar activation.

When `concat_heads=True`, the output has shape `(num_dst_nodes, dim_node)`. Otherwise, the heads are averaged and the output has shape `(num_dst_nodes, dim_node / num_heads)`.

## Dropout and returned attention

`dropout_mask` is an optional CUDA FP32 tensor with shape `(num_heads, num_edges)`. It contains inverted-dropout factors: zero for a dropped coefficient and `1 / (1 - p)` for a retained coefficient. The operation does not generate random masks, which lets the caller control RNG and reuse masks during recomputation.

With `return_attention_weights=True`, the result is `(output, attention)`. `attention` has shape `(num_heads, num_edges)`, uses FP32, and contains post-softmax, post-dropout coefficients. If `map_csc_to_coo` is set on the graph, attention and dropout entries use mapped edge-feature order.

Gradients from both `output` and returned `attention` are included in backward. `dropout_mask` is not differentiable.

## Support and determinism

The bindings are compiled with cuDNN 9.28 or newer headers on non-Windows platforms and resolve backend entry points when called. The backend requires SM 8.0 or newer.

By default, backward uses atomic accumulation for node-feature and attention-weight gradients. For deterministic backward, construct reverse-CSC metadata and request the deterministic path explicitly:

```python
graph = graph.with_reverse_csc()
output = mha_gat(graph, src_features, attn_weights, deterministic=True)
```

`with_reverse_csc()` returns the original graph when reverse metadata is already present. Otherwise, it uses PyTorch operations on the current CUDA stream to create `csc_rev_offsets` with shape `(num_src_nodes + 1,)` and `map_rev_to_coo` with shape `(num_edges,)`. Applications that already own reverse-CSC tensors can pass them directly to `CscGraph`; the two tensors must be supplied together and must match the graph index dtype and device.

Deterministic backward allocates temporary feature and weight reduction workspaces. Their shapes are `(num_edges, dim_node)` and `(num_dst_nodes, 2 * dim_node + dim_edge)` for GAT, or `(num_dst_nodes, dim_node)` for GATv2.

The first call for each supported dtype/index/device configuration may JIT-compile an NVRTC kernel; later calls reuse the backend cache.
