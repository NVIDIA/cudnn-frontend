# FE - Programming Samples

## Python Interface Samples
Samples leveraging FE's Python interface are located in [samples/python](python/).
* [01_epilogue](python/01_matmul_bias.ipynb)
    Shows how to fuse elementwise functions to a GEMM graph.

* [02_serialization](python/02_sdpa_graph_serialization.ipynb)
    Shows how to serialize and deserialize a graph for future execution.

* [03_mixed_precision](python/03_mixed_precision_matmul.ipynb)
    Shows how to mutiply tensors of different data types.

* [50_sdpa](python/50_scaled_dot_product_attention.ipynb)
    Shows how to run causal self attention with dropout in forward pass.

* [51_sdpa](python/51_scaled_dot_product_attention_backward.ipynb)
    Shows how to run causal self attention in bprop.

## C++ Interface Samples
Samples leveraging FE's C++ interface are located in [samples/cpp](cpp/).

## [Deprecated] C++ v0.x Interface Samples
Samples leveraging FE's C++ 0.x interface are located in [samples/legacy_samples](legacy_samples/).
