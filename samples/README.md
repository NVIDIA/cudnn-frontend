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

- [Fwd FP8 SDPA](cpp/sdpa/fp8_fwd.cpp) and [Bwd SDPA](cpp/sdpa/fp8_bwd.cpp)

Extends the sdpa sample to fp8 precision.

### Convolution fusion examples

##### [samples/cpp/convolution](cpp/convolution/) shows how to use cudnn fprop, dgrad, wgrad operation and some fusions with them.

- [Fprop](cpp/convolution/fprop.cpp)

Show cases a simple fprop, fprop with pointwise fusion of scale bias and relu, fprop with bias and relu for channels first layout and fusions before convolution in the form of scale bias relu conv and stats.

- [Fp8 fprop](cpp/convolution/fp8_fprop.cpp)

Showcases fp8 convolution with scaling and amax reduction.

- [Int8 fprop](cpp/convolution/int8_fprop.cpp)

Showcases Int8 convolution.

- [Dgrad](cpp/convolution/dgrads.cpp)

Has samples for simple dgrad, fusion for dgrad + drelu and Dgrad + Drelu + DBNweight fused operation.

- [Wgrad](cpp/convolution/wgrads.cpp)

Similar to dgrad was simple wgrad and scale+bias+relu+wgrad fused operation.

### Matmul fusion examples

##### [Matmul](cpp/matmul/) showcases different matmul samples.

- [Matmul fusion](cpp/matmul/matmuls.cpp) 

Has samples for simple Matmul, matmul fusions like matmul+abs, matmul+bias and matmul+scale+bias+relu operation.

- [Fp8 Matmul](cpp/matmul/fp8_matmul.cpp)

Showcases fp8 matmul with scaling and amax reduction.

- [Int8 Matmul](cpp/matmul/int8_matmul.cpp)

Showcases Int8 mamtul.

- [Mixed precision matmul](cpp/matmul/mixed_matmul.cpp)

Mixed precision multiplication between int8 and bf16 data-type with int8 operand being upcasted to bf16

### Normaliization examples

##### [Norm](cpp/norm/) showcases different matmul samples.

- [LayerNorm](cpp/norm/layernorm.cpp)

Eg for layernorm training, inference and back propagation

- [RMSNorm](cpp/norm/layernorm.cpp)

Eg for rmsnorm training, inference and back propagation

- [BatchNorm](cpp/norm/batchnorm.cpp)

Shows different fusions in batch norm fprop and bprop. And split batch norm fusions.

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

## [Deprecated] C++ v0.x Interface Samples
Samples leveraging FE's C++ 0.x interface are located in [samples/legacy_samples](legacy_samples/).
