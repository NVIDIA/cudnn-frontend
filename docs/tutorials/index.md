# Tutorials

Each tutorial builds one capability at a time. Before you start, it is
recommended to complete the [quick start](/getting-started/quickstart).

1. [Build and extend a graph](/tutorials/first-graph) — Construct a multi-operation
   graph, select and build an execution plan, then execute and reuse it.
2. [Scaled dot-product attention](/tutorials/attention) — Build causal grouped-query
   attention forward and backward graphs and compare their results with PyTorch.
3. [Compose pointwise and reduction expressions](/tutorials/pointwise-reduction) — Learn
   broadcasting and reduction shape rules by executing addition and
   channel-reduction graphs.
4. [Train a convolution](/tutorials/convolution) — Connect convolution forward,
   data-gradient, and weight-gradient graphs through their shared tensor
   metadata.
5. [Train with normalization](/tutorials/normalization) — Build normalization forward
   and backward graphs while preserving the statistics required for training.
6. [Integrate with PyTorch](/tutorials/pytorch-integration) — Cache shape-specialized
   cuDNN graphs, preserve PyTorch autograd behavior, and learn when
   `torch.compile` requires a registered custom operator.
7. [Block-scaled matmul](/tutorials/block-scaling) (Advanced) — Quantize values by block,
   execute an MXFP8 matrix multiplication, and manage its scale tensors on
   Blackwell.
8. [MoE grouped matmul](/tutorials/moe-grouped-matmul) (Advanced) — Execute routed expert
   matrix multiplications from routing metadata without building a separate
   graph for each expert.

Most tutorials present Python and C++ examples side by side. Exact parameter
and support constraints remain on the individual API pages.
