# Sample guide

Use these complete, short examples as starting points. Each row links directly
to source code; read the associated tutorial or how-to guide for the contract
and required environment.

| Task | Python | C++ | Explanation |
| --- | --- | --- | --- |
| Build a graph | [`matmul.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/matmul.py) | [`matmul.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/matmul.cc) | [Quick start](/getting-started/quickstart) |
| Extend a graph with fusion | [`fusion.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/fusion.py) | [`fusion.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/fusion.cc) | [Build and extend a graph](/tutorials/first-graph) |
| Pointwise addition | [`add_basic.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/add_basic.py) | [`add_basic.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/add_basic.cc) | [Pointwise and reduction](/tutorials/pointwise-reduction) |
| Broadcast addition | [`add_broadcast.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/add_broadcast.py) | [`add_broadcast.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/add_broadcast.cc) | [Pointwise and reduction](/tutorials/pointwise-reduction) |
| Reduction | [`reduction_basic.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/reduction_basic.py) | [`reduction_basic.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/reduction_basic.cc) | [Pointwise and reduction](/tutorials/pointwise-reduction) |
| Convolution forward | [`conv_fprop_basic.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/conv_fprop_basic.py) | [`conv_fprop_basic.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/conv_fprop_basic.cc) | [Train a convolution](/tutorials/convolution) |
| Normalization forward | [`layernorm_basic.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/layernorm_basic.py) | [`layernorm_basic.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/layernorm_basic.cc) | [Train with normalization](/tutorials/normalization) |
| Normalization backward | [`layernorm_backward_basic.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/layernorm_backward_basic.py) | [`layernorm_backward_basic.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/layernorm_backward_basic.cc) | [Train with normalization](/tutorials/normalization) |
| Causal SDPA forward and backward | [`sdpa_bf16_causal.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/sdpa_bf16_causal.py) | [`sdpa_bf16_causal.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_bf16_causal.cc) | [Scaled dot-product attention](/tutorials/attention) |
| Block-scaled matmul | [`block_scaled_matmul.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/block_scaled_matmul.py) | [`block_scaled_matmul.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/block_scaled_matmul.cc) | [Block scaling](/tutorials/block-scaling) |
| MoE grouped matmul | [`moe_grouped_matmul.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/moe_grouped_matmul.py) | [`moe_grouped_matmul_basic.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/moe_grouped_matmul_basic.cc) | [MoE grouped matmul](/tutorials/moe-grouped-matmul) |
| Serialize a graph | [`serialize_basic.py`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/serialize_basic.py) | [`serialize_basic.cc`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/serialize_basic.cc) | [Serialize and deploy](/how-to/deployment) |

For exhaustive generated records, use the [full sample catalog](/verified-samples/catalog). The
curated guide stays small enough to scan while the catalog records every
published sample and its verification status.
