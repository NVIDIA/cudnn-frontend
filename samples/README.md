# FE - Programming Samples

## Python Interface Samples
Samples leveraging FE's Python interface are located in [samples/python](python/). Where a notebook needs a specific GPU, cuDNN version or driver, it checks this itself and skips the affected cells; those requirements are noted below.
* [00_introduction](python/00_introduction.ipynb)
    Introduction to the cuDNN Frontend Python API.

* [01_graph_building](python/01_graph_building.ipynb)
    Overview of the `cudnn.Graph` wrapper. Uses CuPy in addition to PyTorch.

* [02_low_level_api](python/02_low_level_api.ipynb)
    Overview of the lower-level `pygraph` binding API.

* [03_localized_tensors](python/03_localized_tensors.ipynb)
    Runs a graph on tensors placed in one locality domain of the GPU. Needs a GPU that reports at least two locality domains (for example Rubin) and a CUDA 13.4 or newer driver.

* [10_matrix_multiplication](python/10_matrix_multiplication.ipynb)
    Matrix multiplication with the graph API.

* [20_layernorm](python/20_layernorm_forward.ipynb), [21_layernorm](python/21_layernorm_backward.ipynb), [22_layernorm](python/22_layernorm_inference.ipynb)
    LayerNorm forward training, backward and inference.

* [23_layer_norm](python/23_layernorm_with_pointwise_add_fusion.ipynb)
    Shows how to run pointwise add and layer norm fusion with intermediate bfloat16 output.

* [24_layernorm](python/24_layernorm_zero_centered_gamma_forward_training_and_backward.ipynb), [25_layernorm](python/25_layernorm_zero_centered_gamma_inference.ipynb)
    LayerNorm with zero-centered gamma: forward training and backward, and inference.

* [26_layer_norm](python/26_layernorm_forward_training_and_backward_with_relu_bitmask.ipynb)
    Shows how to use layer norm with fusion pattern for relu using a bitmask.

* [27_adaptive_layernorm](python/27_adaptive_layernorm_forward_training_and_backward.ipynb), [28_adaptive_layernorm](python/28_adaptive_layernorm_inference.ipynb)
    Adaptive LayerNorm: forward training and backward, and inference.

* [29_rmsnorm](python/29_rmsnorm.ipynb)
    RMSNorm forward and backward.

* [30_batchnorm](python/30_batchnorm.ipynb)
    BatchNorm forward, fused with a ReLU and a comparison mask.

* [31_instancenorm](python/31_instancenorm.ipynb), [32_instancenorm](python/32_instancenorm_fusion.ipynb)
    Instance norm, and instance norm fused with an add and a ReLU.

* [50_sdpa](python/50_sdpa_forward.ipynb)
    Shows how to run causal self attention with dropout in forward pass.

* [51_sdpa](python/51_sdpa_backward.ipynb)
    Shows how to run causal self attention in bprop.

* [52_sdpa](python/52_sdpa_with_paged_caches.ipynb)
    Shows how to run scaled dot product attention (prefill phase) where the K and V caches are stored in non contiguous memory.

* [53_sdpa](python/53_sdpa_decode_with_paged_caches.ipynb)
    Shows how to run scaled dot product attention (decode phase) where the K and V caches are stored in non contiguous memory.

* [60_causal_conv1d](python/60_causal_conv1d_forward.ipynb), [61_causal_conv1d](python/61_causal_conv1d_backward.ipynb)
    Causal Conv1D forward and backward.

* [62_causal_conv1d_nwh](python/62_causal_conv1d_nwh_forward.ipynb), [63_causal_conv1d_nwh](python/63_causal_conv1d_nwh_backward.ipynb)
    Causal Conv1D in the NWH layout, forward and backward.

* [64_b2b_causal_conv1d](python/64_b2b_causal_conv1d_forward.ipynb), [65_b2b_causal_conv1d](python/65_b2b_causal_conv1d_backward.ipynb)
    Back-to-back Causal Conv1D, forward and backward.

* [66_fft_causal_conv1d](python/66_fft_causal_conv1d_forward.ipynb), [67_fft_causal_conv1d](python/67_fft_causal_conv1d_backward.ipynb)
    FFT-based Causal Conv1D, forward and backward. Needs cuDNN 9.26.0 or newer.

* [70_boolean_cmp_logic](python/70_boolean_cmp_logic.ipynb)
    Boolean comparison and logic fusion. The cuDNN graph cells need a data-center Blackwell GPU.

* [71_custom_reduction_argmax](python/71_custom_reduction_argmax.ipynb)
    Shows how to pick LLM tokens with the custom-reduction argmax helper: greedy decoding, the NaN rules, speculative-decoding verify, masked argmax and Gumbel-max sampling (cuDNN 9.29 or later, data center Blackwell).

* [gemm_swiglu_mlp_fusion.py](python/gemm_swiglu_mlp_fusion.py)
    The dense bf16 SwiGLU-MLP autograd op (`cudnn.gemm.ops.swiglu_mlp`): gate and up GEMMs, SiLU and multiply fused into one forward kernel. Needs an SM100 or newer (Blackwell) GPU.

## FROST Python Samples
Standalone scripts for the FROST (CuTeDSL) engines, pure cuDNN frontend Python API, each checked against a torch or fp64 reference:
* [frost/gemm](frost/gemm/): matmul across dtypes, fused epilogues, mainloop fusion, mixed input types, dual-GEMM SwiGLU, MoE grouped matmul. Needs a data-center Blackwell-family GPU (`100 <= SM < 120`); on other parts the scripts print `[NN] SKIP  needs a data-center Blackwell-family GPU (100 <= SM < 120)` and exit 0.
* [frost/linear_attention](frost/linear_attention/): GDN, KDA, GDN-2, GDP and Mamba-2 prefill and backward, plus the per-span summary nodes (forward and backward) used for context parallelism.
* [frost/gated_attention_block](frost/gated_attention_block/): the gated attention block's training forward (`save_for_backward=True`) -- the caller-owned `SavedForBackward` record and the zero-copy `saved_slab_views` of its projection slab, checked against torch (Rubin only).

```
python samples/frost/linear_attention/01_gdn_prefill.py
```

## Model Samples
End-to-end samples of open-weight model families in [samples/models](models/): one process, a printed table of which provider serves every block of the model, and a self-check against a plain-PyTorch reference. They are not a serving engine and carry no performance table; see [models/README.md](models/README.md).
* [models/qwen_dense](models/qwen_dense/): the dense Qwen 3.5-architecture model. `00_decode.py` runs a plain greedy decode on the reduced seeded model (2 layers at the published widths, no checkpoint, no extra package), token ids in and token ids out, with the linear-attention step on `cudnn.GdnDecodeFused`. Needs a GPU of compute capability 10.x or 12.x; elsewhere it prints `[00] SKIP <requirement>` and exits 0.

```
python samples/models/qwen_dense/00_decode.py
```

## C++ Interface Samples
Samples leveraging FE's C++ interface are located in [samples/cpp](cpp/).

### Building the samples

```
mkdir build
cd build
cmake -DCUDNN_PATH=/path/to/cudnn -DCUDAToolkit_ROOT=/path/to/cuda  ../
cmake --build . -j16
bin/samples
```

To run a single sample, for eg. `TEST_CASE("Cached sdpa", "[graph][sdpa][flash]")`

```
./bin/samples "Cached sdpa"
```

### Scaled dot product attention SDPA examples

##### [samples/cpp/sdpa](cpp/sdpa) shows how to use cudnn's sdpa operation.

- [Cached SDPA](cpp/sdpa/fp16_cached.cpp)

Users are expected to build a graph once and then execute it multiple times. This example shows how to cache cudnn sdpa graph building. 

- [Fwd SDPA](cpp/sdpa/fp16_fwd.cpp) and [Bwd SDPA](cpp/sdpa/fp16_bwd.cpp)

cudnn's sdpa operation enables various customizations on itself. These examples show how to build a graph with sdpa operation for your own custom sdpa needs.

- [Fwd SDPA with paged caches](cpp/sdpa/fp16_fwd_with_paged_caches.cpp)

Similar to [Fwd SDPA](cpp/sdpa/fp16_fwd.cpp), but here with the ability to use non contiguous K and V caches in combination with page tables, as described in the [PagedAttention paper](https://arxiv.org/abs/2309.06180).

- [Fwd FP8 SDPA](cpp/sdpa/fp8_fwd.cpp) and [Bwd SDPA](cpp/sdpa/fp8_bwd.cpp)

Extends the sdpa sample to fp8 precision.

- [Fwd SDPA with CUDA graph](cpp/sdpa/fp16_fwd_with_cudagraphs.cpp)

Demonstrates the building and execution of a CUDA graph representing the SDPA operation, followed by the update (and another execution) of the CUDA graph with new variant pointers.

- [Bwd SDPA with CUDA graph](cpp/sdpa/fp16_bwd_with_cudagraphs.cpp)

The same CUDA graph build-and-update flow for the SDPA backward pass. Needs cuDNN 9.6.0 or newer.

- [Fwd SDPA with flexible graphs](cpp/sdpa/fp16_fwd_with_flexible_graphs.cpp) and [Bwd SDPA with flexible graphs](cpp/sdpa/fp16_bwd_with_flexible_graphs.cpp)

SDPA with score modifiers (softcap and bias) built from graph operations. Needs cuDNN 9.4.0 or newer; the backward sample needs Hopper or newer and is not yet supported on SM120 and above.

- [Fwd SDPA with max and sum-exp outputs](cpp/sdpa/fp16_fwd_with_max_and_sum_exp.cpp)

Also returns the row max and the sum of exponentials of the attention scores. Needs cuDNN 9.13 or newer.

- [Fwd SDPA with sink token](cpp/sdpa/fp16_fwd_with_sink_token.cpp) and [Bwd SDPA with sink token](cpp/sdpa/fp16_bwd_with_sink_token.cpp)

Attention with a learned sink token. Needs cuDNN 9.13 or newer.

- [Fwd SDPA with custom dropout](cpp/sdpa/fp16_fwd_with_custom_dropout.cpp)

Passes a caller-provided dropout mask and scale. Not supported on Blackwell.

- [Fwd SDPA with cu_seq_len](cpp/sdpa/fp16_fwd_with_cu_seq_len.cpp)

Variable-length sequences described by cumulative sequence lengths. Needs cuDNN 9.24.0 or newer.

- [Fwd SDPA paged decode and prefill](cpp/sdpa/fp16_fwd_paged_decode_and_prefill.cpp)

Decode and prefill over the same paged K/V caches. Needs cuDNN 9.8.0 or newer.

- [Fwd SDPA with block mask](cpp/sdpa/fp16_fwd_with_block_mask.cpp)

Block-sparse masking. Needs compute capability 10.x and cuDNN 9.26 or newer.

- [SDPA with dynamic shapes](cpp/sdpa/fp16_dynamic_shapes.cpp)

One graph built with shape overrides enabled and executed for different problem sizes. Needs cuDNN 9.21 or newer at build time.

- [SDPA benchmark](cpp/sdpa/fp16_benchmark.cpp)

Measures graph API runtimes. Long-running, so it is off by default.

- [Fwd FP8 SDPA with bottom-right causal mask](cpp/sdpa/fp8_fwd_bottom_right_causal_mask.cpp) and [Bwd FP8 SDPA with bottom-right causal mask](cpp/sdpa/fp8_bwd_bottom_right_causal_mask.cpp)

FP8 attention with a bottom-right aligned causal mask. The forward sample needs a Hopper or data-center Blackwell GPU; the backward sample needs compute capability 10.x.

- [Fwd FP8 SDPA with current scaling](cpp/sdpa/fp8_fwd_current_scaling.cpp) and [Bwd FP8 SDPA with current scaling](cpp/sdpa/fp8_bwd_with_current_scaling.cpp)

FP8 attention whose scales are computed from the current tensors. Needs compute capability 10.x and cuDNN 9.13.0 or newer.

- [Fwd MXFP8 SDPA](cpp/sdpa/mxfp8_fwd.cpp) and [Bwd MXFP8 SDPA](cpp/sdpa/mxfp8_bwd.cpp)

MXFP8 block-scaled attention; the backward file covers MHA, GQA and MLA. Needs a data-center Blackwell GPU and cuDNN 9.21.0 or newer.

### Convolution fusion examples

##### [samples/cpp/convolution](cpp/convolution/) shows how to use cudnn fprop, dgrad, wgrad operation and some fusions with them.

- [Fprop](cpp/convolution/fprop.cpp)

Showcases a simple fprop, fprop with pointwise fusion of scale bias and relu, fprop with bias and relu for channels first layout and fusions before convolution in the form of scale bias relu conv and stats.  Also epilogue fusion of concatenate.

- [Fp8 fprop](cpp/convolution/fp8_fprop.cpp)

Showcases fp8 convolution with scaling and amax reduction.

- [Int8 fprop](cpp/convolution/int8_fprop.cpp)

Showcases Int8 convolution.

- [Dgrad](cpp/convolution/dgrads.cpp)

Has samples for simple dgrad, fusion for dgrad + drelu and Dgrad + Drelu + DBNweight fused operation.

- [Wgrad](cpp/convolution/wgrads.cpp)

Similar to dgrad was simple wgrad and scale+bias+relu+wgrad fused operation.

- [Dynamic shape benchmark](cpp/convolution/conv_dynamic_shape_benchmark.cpp)

Measures convolution graph API runtimes across shapes. This long-running benchmark is included in the samples executable and runs in an unfiltered invocation on supported setups; select tests explicitly to avoid running it unintentionally.

### Matmul fusion examples

##### [Matmul](cpp/matmul/) showcases different matmul samples.

- [Matmul fusion](cpp/matmul/matmuls.cpp) 

Has samples for simple Matmul, matmul fusions like matmul+abs, matmul+bias and matmul+scale+bias+relu operation.

- [Fp8 Matmul](cpp/matmul/fp8_matmul.cpp)

Showcases fp8 matmul with scaling and amax reduction.

- [Int8 Matmul](cpp/matmul/int8_matmul.cpp)

Showcases Int8 matmul.

- [Mixed precision matmul](cpp/matmul/mixed_matmul.cpp)

Mixed precision multiplication between int8 and bf16 data-type with int8 operand being upcasted to bf16

- [Complex FP32 matmul](cpp/matmul/complex_fp32_matmul.cpp)

Complex-valued FP32 matmul. Needs cuDNN 9.14.0 or newer.

- [General block scale matmul](cpp/matmul/general_block_scale_matmul.cpp)

Matmul with block-scaled operands. Needs cuDNN 9.11.0 or newer.

- [Blackwell NVFP4/MXFP8 block scale matmul](cpp/matmul/blackwell_nvfp4_mxfp8_block_scale_matmul.cpp)

Hardware-accelerated NVFP4 and MXFP8 block-scaled matmul, with quantize and SwiGLU epilogues and dynamic shape overrides. Needs Blackwell or newer; the quantize epilogue needs compute capability 10.x.

### Normalization examples

##### [Norm](cpp/norm/) showcases different normalization samples.

- [LayerNorm](cpp/norm/layernorm.cpp)

Eg for layernorm training, inference and back propagation

- [AdaLayerNorm](cpp/norm/adaptive_layernorm.cpp)

Eg for adaptive layernorm training, inference and back propagation

- [RMSNorm](cpp/norm/rmsnorm.cpp)

Eg for rmsnorm training, inference and back propagation. Needs Ampere or newer.

- [BatchNorm](cpp/norm/batchnorm.cpp)

Shows different fusions in batch norm fprop and bprop. And split batch norm fusions.

- [Block scale quantize](cpp/norm/norm_block_scale.cpp)

Showcases normalization with block scale quantize epilogue fusion.

- [Norm zero centered gamma](cpp/norm/norm_zero_centered_gamma.cpp)

Showcases layer normalization with zero centered gamma usage.

- [Layer norm with bitmask relu](cpp/norm/layernorm_bitmask_relu.cpp)

Showcases layer normalization and relu with bitmask.

### Custom reduction examples

##### [samples/cpp/membound](cpp/membound/) shows how to use the custom reduction (cuDNN 9.29 or later, data center Blackwell).

- [Custom reduction](cpp/membound/custom_reduction.cpp): a sum with a helper, mixed precisions, several streams, a body with a constant, and plan serialization.
- [argmax and argmin rules](cpp/membound/custom_reduction_argmax.cpp): `reduction_modifiers::argmax` and `argmin` with each NaN rule, checked against a host reference.
- [LLM token selection](cpp/membound/custom_reduction_sampling.cpp): greedy decoding, speculative-decoding verify, masked argmax and Gumbel-max sampling.
- [XLA HLO examples](cpp/membound/custom_reduction_xla.cpp): the argmax that XLA emits, at all 31 distinct argmax operand shapes of an XLA HLO corpus, with the XLA and the torch NaN rule. The shapes and their source models are listed in [docs/operations/CustomReduction.md](../docs/operations/CustomReduction.md#xla-hlo-examples).

### Miscellaneous examples

##### [Misc](cpp/misc/) Miscellaneous samples

- [Pointwise fusions](cpp/misc/pointwise.cpp)

pointwise fusions with scalar are shown in this sample.

- [Resample](cpp/misc/resample.cpp)

resample fprop operation with different resampling modes.

- [Serialization](cpp/misc/serialization.cpp)

How to serialize a graph into a file and read it back on another thread/process. 

- [Autotuning](cpp/misc/autotuning.cpp)

How to choose the best performing plan among multiple plans suggested by the heuristics.

- [Cuda Graphs](cpp/misc/cudagraphs.cpp)

Shows how to use the native cuda graph API. The samples show how to create cudnn's cuda graph, and how to repeatedly update it with new device buffers for multiple execution.

- [SM Carveout](cpp/misc/sm_carveout.cpp)

Showcases a Batch norm example, where only a partial number of SMs participate in executing the kernel.

- [Deviceless ahead-of-time compilation](cpp/misc/deviceless_aot_compilation.cpp)

Showcases how to do deviceless ahead-of-time compilation with the device property descriptor (instead of a cuDNN handle).

- [Custom plan](cpp/misc/custom_plan.cpp)

Enumerates the engines and their knobs and builds a plan from an explicit engine configuration instead of the heuristics.

- [Parallel compilation](cpp/misc/parallel_compilation.cpp)

Builds a graph's plans from several threads. Long-running, so it is off by default.

- [Slice](cpp/misc/slice.cpp)

A slice of an input tensor feeding a matmul.

- [Localized tensors](cpp/misc/localized_tensors.cpp)

Matmul on tensors placed in one locality domain of the GPU. Needs CUDA 13.4 or newer and a driver that provides locality domains.

- [Compile-time constants](cpp/misc/compile_time_constant_example.cpp)

Scalars baked into the graph at compile time. Needs cuDNN 9.22.0 or newer and a data-center Blackwell GPU.

- [CUDA graph topology](cpp/misc/cudagraph_topology.cpp)

Reproducer for the CUDA graph topology and lifecycle questions in #186. The multi-device cases need two visible GPUs.

### Membound examples

##### [Membound](cpp/membound/) shows the TensorIR memory-bound engine. These samples need a data-center Blackwell GPU and cuDNN 9.22.0 or newer.

- [Reshape](cpp/membound/reshape.cpp), [Transpose](cpp/membound/transpose.cpp), [Slice](cpp/membound/slice.cpp) and [Concatenate](cpp/membound/concat.cpp)

Single data-movement operations.

- [Membound fusion](cpp/membound/membound_fusion.cpp)

Data movement fused with pointwise operations (reshape then ReLU, transpose then bias add).

- [Boolean fusion](cpp/membound/boolean_fusion.cpp)

Comparison and logical operations fused into one kernel.

### Causal conv1d examples

##### [Causal conv1d](cpp/causal_conv1d/) shows the causal conv1d operations. They are not supported on Windows (MSVC).

- [Causal conv1d](cpp/causal_conv1d/causal_conv1d.cpp) and [Causal conv1d NWH](cpp/causal_conv1d/causal_conv1d_nwh.cpp)

Forward and backward in the default and NWH layouts. The NWH sample needs cuDNN 9.24.0 or newer.

- [Back-to-back causal conv1d](cpp/causal_conv1d/b2b_causal_conv1d.cpp)

Two causal conv1d operations fused back to back, forward and backward. Needs cuDNN 9.24.0 or newer.

- [FFT causal conv1d](cpp/causal_conv1d/fft_causal_conv1d.cpp)

FFT-based causal conv1d for medium and long sequences, forward and backward. Needs cuDNN 9.26.0 or newer.

### MoE grouped matmul examples

- [MoE grouped matmul](cpp/moe_grouped_matmul/moe_grouped_matmul.cpp)

Weight-only-quantized grouped matmul over experts, and the BF16 backward. Needs cuDNN 9.18.0 or newer (9.22.0 for the backward).

## [Deprecated] C++ v0.x Interface Samples
Samples leveraging FE's C++ 0.x interface are located in [samples/legacy_samples](legacy_samples/).
