---
title: Simple multi-head attention
---

# Simple multi-head attention

`cudnn.gnn.mha_simple` applies scaled dot-product multi-head attention to a
homogeneous or bipartite graph represented by `cudnn.gnn.CscGraph`.

For every CSC edge from source node `j` to destination node `i`, it computes

```text
score[h, i, j] = dot(query[i, h], key[j, h] + edge[i, j, h])
```

An optional per-head edge weight multiplies the score, optional dimensional
normalization divides it by `sqrt(dim_head)`, and an optional edge bias is
added. Softmax is applied across the incoming neighbors of each destination.
The normalized coefficient aggregates `value[j, h] + edge[i, j, h]`.

```python
from cudnn.gnn import CscGraph, mha_simple

output, attention = mha_simple(
    graph,
    key_features,
    query_features,
    value_features,
    edge_features=edge_features,
    num_heads=4,
    return_attention_weights=True,
)
```

## Inputs and options

- `key_features` and `value_features` have shape
  `(num_src_nodes, dim_node)`.
- `query_features` has shape `(num_dst_nodes, dim_node)`.
- Optional `edge_features` has shape `(num_edges, dim_node)`.
- `dim_node` must be divisible by `num_heads`.
- `concat_heads=True` concatenates the head outputs. When false, the heads are
  averaged.
- `norm_by_dim=True` scales logits by `1 / sqrt(dim_head)`.
- Optional `score_bias` and `score_weight` are FP32 tensors with shape
  `(num_heads, num_edges)` in direct CSC edge order.
- Optional `dropout_mask` contains caller-generated FP32 inverted-dropout
  factors with shape `(num_heads, num_edges)`. It uses mapped edge order when
  `graph.map_csc_to_coo` is present.

When `return_attention_weights=True`, the function returns
`(output, attention)`. `attention` contains normalized pre-dropout coefficients
with shape `(num_heads, num_edges)` and FP32 dtype. It remains differentiable,
so a loss may depend on the output, the returned attention, or both.

## Backward options

Set `deterministic=True` to use deterministic source-node reductions. The graph
must first contain reverse-CSC metadata, for example:

```python
graph = graph.with_reverse_csc()
```

Backward with the default `deterministic=False` does not guarantee repeatable
gradients. For repeatability, pass `deterministic=True` with reverse-CSC
metadata. `torch.use_deterministic_algorithms(True)` does not automatically
enable this option.

For FP16 or BF16 inputs, `high_precision_dgrad=True` asks the backend to compute
key, query, value, and edge gradients in FP32. PyTorch separately controls the
dtype used to accumulate leaf gradients.

## Availability

The loaded cuDNN backend must be 9.29 or newer and provide
`cudnnGnnMhaSimpleForward` and `cudnnGnnMhaSimpleBackward`. The operation
requires an Ampere-class GPU or newer and is not available on Windows.
