## Table of Contents
1. [Scaled Dot Product Attention](#scaled-dot-product-attention)
2. [Scaled Dot Product Attention Backward](#scaled-dot-product-attention-backward)
3. [Scaled Dot Product Attention FP8](#scaled-dot-product-attention-fp8)
4. [Scaled Dot Product Attention Backward FP8](#scaled-dot-product-attention-backward-fp8)
5. Appendices
    - [Tensor Layouts](#appendix-a)
    - [Workspace limits and Performance](#appendix-b)
    - [RNG dump](#appendix-c)
6. [Miscellaneous](#miscellaneous)

### Scaled Dot Product Attention

This operation computes the scaled dot product attention, as

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$

using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). It is applicable for both training and inference phases, with an option to generate a stats tensor to be used for backwards training computation.

- Python sample: [samples/python/50_scaled_dot_product_attention.ipynb](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/50_scaled_dot_product_attention.ipynb)

- C++ sample: [samples/cpp/mha.cpp](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/mha.cpp)

- Python tests: [test/python_fe/test_mhas.py](https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python_fe/test_mhas.py)

#### Configurable Options:

- Attention scale (`attn_scale`): Applies a scaling factor to attention scores before the softmax, such as $\frac{1}{\sqrt{\text{d}}}$. Set to 1.0 by default.
- Bias mask: Applies an additive bias mask to attention scores. Users must pass a bias tensor as specified in the tensors section below.
- Alibi mask: Attention with Linear Biases (ALiBi) is an additive mask applied to the attention scores as described in the paper [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).
- Padding mask: Also called variable sequence length, this option masks out padded time steps to ignore them in computation. Users must pass a per-batch sequence length as specified in the tensors section below.
- Causal mask: Fills the upper triangular matrix of attention scores with negative infinity.
- Dropout: Randomly zeros some of the attention weights after the softmax as a form of regularization.
  Users can configure dropout in two ways:
  - To use the more performant Philox RNG dropout implementation, users must provide:
    - An RNG seed, passed as a cudnn tensor.
    - An RNG offset, passed as a cudnn tensor.
    - A float representing the dropout probability, which is the probability that any given weight is set to zero.
  - To use an user-provided dropout mask, users must provide:
    - `dropout mask` that matches the attention weights' dimensions, indicating which weights to drop.
    - `dropout scale` used to adjust the scale of the remaining weights accordingly, such as $1 / (1 - \text{dropout probability})$.
- Ragged tensor: allows the query, key, value, and output tensor to be [ragged tensors](https://www.tensorflow.org/guide/ragged_tensor), which are tensors with nested variable length lists as inner dimensions. Users must pass another tensor called ragged offset tensor using the `Tensor_attributes.set_ragged_offset()` method as specified in the tensors section below.

#### Tensors:

- Query tensor should have dimensions $(B, H_{q}, S_{q}, D_{qk})$ with input/output datatype.
- Key tensor should have dimensions $(B, H_{k}, S_{kv}, D_{qk})$ with input/output datatype.
- Value tensor should have dimensions $(B, H_{v}, S_{kv}, D_{v})$ with input/output datatype.
- Output tensor should have dimensions $(B, H_{q}, S_{q}, D_{v})$ with input/output datatype.
- (Optional) When `is_inference` is false, the stats tensor should have dimensions $(B, H_{q}, S_{q}, 1)$ with float32 datatype.
- (Optional) When bias mask is enabled, the bias tensor has dimensions $(1, 1, S_{q}, S_{kv})$, $(1, H_{q}, S_{q}, S_{kv})$, $(B, 1, S_{q}, S_{kv})$, or $(B, H_{q}, S_{q}, S_{kv})$ with input/output datatype.  
The dimensions that are passed as 1 will apply a broadcasted mask over attention scores.
- (Optional) When padding mask is enabled, the sequence length q, and sequence length kv tensors should have shape $(B, 1, 1, 1)$ with int32 datatype.
- (Optional) When philox RNG dropout mask is enabled, the RNG seed and offset tensors should have size $(1, 1, 1, 1)$ with int32 or int64 datatype as either a CPU or GPU tensor.
- (Optional) When a user provided dropout mask is enabled, a dropout mask tensor should have shape $(1, 1, S_{q}, S_{kv})$, $(1, H_{q}, S_{q}, S_{kv})$, $(B, 1, S_{q}, S_{kv})$, or $(B, H_{q}, S_{q}, S_{kv})$ with input/output datatype.  
The dimensions that are passed as 1 will apply a broadcasted mask over attention weights.
- (Optional) When query, key, value, and output tensors are ragged tensors, the ragged offset tensor must be a tensor of size $(B + 1, 1, 1, 1)$ that contains the nested tensor's offset in terms of number of elements (not bytes). The last value of the offset tensor specifies the offset of the past-the-end element of the ragged tensor.

Where,

- $B$ is the batch size
- $H_{q}$ is the number of query heads
- $H_{k}$ is the number of key heads
- $H_{v}$ is the number of value heads
- $S_{q}$ is the sequence length of the query
- $S_{kv}$ is the sequence length of the key and value
- $D_{qk}$ is the embedding dimension per head of query and key
- $D_{v}$ is the embedding dimension per head of value

#### Group-query attention (GQA) and Multi-query attention (MQA)

- As described in the paper [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245),
- When $H_{k}$ and $H_{v}$ is less than $H_{q}$ and factors of $H_{q}$, this operation will perform group-query attention (GQA) computation.
- When $H_{k}$ and $H_{v}$ are both set to 1, this operation perform multi-query attention (MQA) computation.

#### Limitations:

- All input and output tensor datatypes must be float16 or bfloat16 datatype except the softmax stats output tensor, which must be float32.
- The dimension of the embedding dimension per head $D_{qk}$ and $D_{v}$ must be a multiple of 8 with maximum value 128.
- the stride of the embedding dimension per head $D_{qk}$ and $D_{v}$ for all the tensors above must be 1.
- this operation is only supported on GPUs with NVIDIA Ampere architecture (SM80) or newer.

#### C++ API:

```cpp
// returns [output, softmax_stats]
std::array<std::shared_ptr<Tensor_attributes>, 2> 
sdpa(std::shared_ptr<Tensor_attributes> q,
     std::shared_ptr<Tensor_attributes> k,
     std::shared_ptr<Tensor_attributes> v,
     SDPA_attributes options);
```

The `options` parameter of type `SDPA_attributes` is used to control the attributes of the forward operation, as detailed below:

```cpp
SDPA_attributes&
set_is_inference(bool const value);

SDPA_attributes&
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes&
set_attn_scale(float const value);

SDPA_attributes&
set_bias(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes&
set_alibi_mask(bool const value);

SDPA_attributes&
set_padding_mask(bool const value);

SDPA_attributes&
set_seq_len_q(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes&
set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes&
set_causal_mask(bool const value);

SDPA_attributes&
set_dropout(float const probability,
            std::shared_ptr<Tensor_attributes> seed,
            std::shared_ptr<Tensor_attributes> offset);

SDPA_attributes&
set_dropout(std::shared_ptr<Tensor_attributes> mask,
            std::shared_ptr<Tensor_attributes> scale);

SDPA_attributes&
set_compute_data_type(DataType_t value);
```

#### Python API:

```
Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    is_inference (bool): Whether it is an inference step or training step.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    bias (Optional[cudnn_tensor]): The bias data for attention. Default is None.
    use_alibi_mask (Optional[bool]): Whether to use alibi mask. Default is False.
    use_padding_mask (Optional[bool]): Whether to use padding mask. Default is False.
    seq_len_q (Optional[cudnn_tensor]): The sequence length of the query.
    seq_len_kv (Optional[cudnn_tensor]): The sequence length of the key.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.

Returns:
    o (cudnn_tensor): The output data.
    stats (Optional[cudnn_tensor]): The softmax statistics in case the operation is in a training step.
```

### Scaled Dot Product Attention Backward

This operation computes gradient tensors for scaled dot product attention using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). The user is required to pass the stats tensor from the forward operation to the backward operation as input.

- Python sample: [samples/python/51_scaled_dot_product_attention_backward.ipynb](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/51_scaled_dot_product_attention_backward.ipynb)

- C++ sample: [samples/cpp/mha.cpp](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/mha.cpp)

- Python tests: [test/python_fe/test_mhas.py](https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python_fe/test_mhas.py)

#### Configurable Options:

All the options mentioned in the forward operation, including ragged tensors and GQA/MQA, are applicable in the backward operation as well.

#### Tensors:

All the tensor requirements described in the forward operation are applicable in the backward operation as well. The gradient tensors for query, key, value, output, and bias should have the same properites as their non-gradient counterparts.

#### Limitations:

All the limitations mentioned in the forward operation are applicable in the backward operation as well.

#### C++ API:
```cpp
// returns [dQ, dK, dV]
std::array<std::shared_ptr<Tensor_attributes>, 3>
sdpa_backward(std::shared_ptr<Tensor_attributes> q,
              std::shared_ptr<Tensor_attributes> k,
              std::shared_ptr<Tensor_attributes> v,
              std::shared_ptr<Tensor_attributes> o,
              std::shared_ptr<Tensor_attributes> dO,
              std::shared_ptr<Tensor_attributes> stats,
              SDPA_backward_attributes);
```

The `options` parameter of type `SDPA_backward_attributes` is used to control the attributes of backward operation, as detailed below:

```cpp
SDPA_backward_attributes&
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_attn_scale(float const value);

SDPA_backward_attributes&
set_bias(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_dbias(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_alibi_mask(bool const value);

SDPA_backward_attributes&
set_padding_mask(bool const value);

SDPA_backward_attributes&
set_seq_len_q(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_causal_mask(bool const value);

SDPA_backward_attributes&
set_dropout(float const probability,
            std::shared_ptr<Tensor_attributes> seed,
            std::shared_ptr<Tensor_attributes> offset);

SDPA_backward_attributes&
set_dropout(std::shared_ptr<Tensor_attributes> mask,
            std::shared_ptr<Tensor_attributes> scale,
            std::shared_ptr<Tensor_attributes> scale_inv);

SDPA_backward_attributes&
set_compute_data_type(DataType_t const value);
```

#### Python API: 

```
Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    o (cudnn_tensor): The output data.
    dO (cudnn_tensor): The output loss gradient.
    stats (cudnn_tensor): The softmax statistics from the forward pass.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    bias (Optional[cudnn_tensor]): The bias data for attention. Default is None.
    dBias (Optional[cudnn_tensor]): The dBias output for attention. Default is None.
    use_alibi_mask (Optional[bool]): Whether to use alibi mask. Default is False.
    use_padding_mask (Optional[bool]): Whether to use padding mask. Default is False.
    seq_len_q (Optional[cudnn_tensor]): The sequence length of the query.
    seq_len_kv (Optional[cudnn_tensor]): The sequence length of the key.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.

Returns:
    dQ (cudnn_tensor): The query gradient data.
    dK (cudnn_tensor): The key gradient data.
    dV (cudnn_tensor): The value gradient data.
```

### Scaled Dot Product Attention FP8

This operation computes the scaled dot product attention in the FP8 (8-bit floating point) datatype, using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). It is applicable for both training and inference phases, with an option to generate a stats tensor to be used for backwards training computation.

The FP8 datatype consists of two encodings:
- `FP8_E4M3` (1 sign bit, 4 exponent bits, and 3 mantissa bits)
- `FP8_E5M2` (1 sign bit, 5 exponent bits, 2 mantissa bits).

Due to the limited numerical precision of FP8 data type, for practical use cases, users must scale values computed in FP32 format before storing them in FP8 format, and descale the values stored in FP8 format before performing computations on them. For more information, refer to [the Transformer Engine FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

The suggested value for the scaling factor is computed as: (Max representable value in the fp8 format) / (Max absolute value seen in the tensor for the previous layer).
- For E4M3, the suggested scaling factor is `448.f/ prev_layer_tensor_amax` (rounded to the nearest lower power of two)
- For E5M2, the suggested scaling factor is `57344.f/ prev_layer_tensor_amax` (rounded to the nearest lower power of two)

The suggested value for the descale factor is the reciprocal of the scale factor.

Since scaling and descaling are critical for convergence with FP8 datatype, users are required to pass scaling and descaling input tensors, as well as amax output tensors.

#### Configurable Options

The current FP8 support is a subset of the options supported in FP16 and BF16 support. We are actively working on expanding the support for FP8.
- Attention scale (`attn_scale`): Applies a scaling factor to attention scores before the softmax, such as $\frac{1}{\sqrt{\text{d}}}$. Set to 1.0 by default.
- Causal mask: Fills the upper triangular matrix of attention scores with negative infinity.

#### Tensors

The tensors in forward operation are defined as the following:

$P = QK^T$

$S = \text{softmax}(P)$

$O = SV$

##### Input Tensors

| Tensor Name           | Device     | Data Type    | Dimensions                   |
|-----------------------|------------|--------------|------------------------------|
| Q                     | GPU        | E4M3 or E5M2 | $(B, H_{q}, S_{q}, D_{qk})$  |
| K                     | GPU        | E4M3 or E5M2 | $(B, H_{k}, S_{kv}, D_{qk})$ |
| V                     | GPU        | E4M3 or E5M2 | $(B, H_{v}, S_{kv}, D_{v})$  |
| Descale Q             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale K             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale V             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale S             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Scale S               | GPU        | FP32         | $(1, 1, 1, 1)$               |

##### Output Tensors

| Tensor Name           | Device     | Data Type    | Dimensions                   |
|-----------------------|------------|--------------|------------------------------|
| O                     | GPU        | E4M3 or E5M2 | $(B, H_{q}, S_{q}, D_{v})$   |
| Stats (training only) | GPU        | FP32         | $(B, H_{q}, S_{q}, 1)$       |
| AMax S                | GPU        | FP32         | $(1, 1, 1, 1)$               |
| AMax O                | GPU        | FP32         | $(1, 1, 1, 1)$               |

Where,

- $B$ is the batch size
- $H_{q}$ is the number of query heads
- $H_{k}$ is the number of key heads
- $H_{v}$ is the number of value heads
- $S_{q}$ is the sequence length of the query
- $S_{kv}$ is the sequence length of the key and value
- $D_{qk}$ is the embedding dimension per head of query and key
- $D_{v}$ is the embedding dimension per head of value

#### Group-query attention (GQA) and Multi-query attention (MQA)

- As described in the paper [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245),
- When $H_{k}$ and $H_{v}$ is less than $H_{q}$ and factors of $H_{q}$, this operation will perform group-query attention (GQA) computation.
- When $H_{k}$ and $H_{v}$ are both set to 1, this operation perform multi-query attention (MQA) computation.

#### Limitations:
- The dimension of the embedding dimension per head $D_{qk}$ and $D_{v}$ must be a multiple of 8 with maximum value 128.
- the stride of the embedding dimension per head $D_{qk}$ and $D_{v}$ for all the tensors above must be 1.
- this operation is only supported on GPUs with NVIDIA Hopper architecture (SM90) or newer.

#### C++ API:
```cpp
// returns [o, stats, amax_s, amax_o]
std::array<std::shared_ptr<Tensor_attributes>, 4>
Graph::sdpa_fp8(std::shared_ptr<Tensor_attributes> q,
                std::shared_ptr<Tensor_attributes> k,
                std::shared_ptr<Tensor_attributes> v,
                std::shared_ptr<Tensor_attributes> descale_q,
                std::shared_ptr<Tensor_attributes> descale_k,
                std::shared_ptr<Tensor_attributes> descale_v,
                std::shared_ptr<Tensor_attributes> descale_s,
                std::shared_ptr<Tensor_attributes> scale_s,
                std::shared_ptr<Tensor_attributes> scale_o,
                SDPA_fp8_attributes attributes);
```

The `options` parameter of type `SDPA_fp8_attributes` is used to control the attributes of the forward operation, as detailed below:


```cpp
SDPA_fp8_attributes&
set_is_inference(bool const value);

SDPA_fp8_attributes&
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_attributes&
set_attn_scale(float const value);

SDPA_fp8_attributes&
set_causal_mask(bool const value);
```

#### Python API: 
```
Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    descale_q (cudnn_tensor): Descale factor for query.
    descale_k (cudnn_tensor): Descale factor for key.
    descale_v (cudnn_tensor): Descale factor for value.
    descale_s (cudnn_tensor): Descale factor for S tensor.
    scale_s (cudnn_tensor): Scale factor for S tensor.
    scale_o (cudnn_tensor): Scale factor for output.
    is_inference (bool): Whether it is an inference step or training step.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.

Returns:
    o (cudnn_tensor): The output data.
    stats (Optional[cudnn_tensor]): The softmax statistics in case the operation is in a training step.
    amax_s (cudnn_tensor): The absolute maximum of S tensor.
    amax_o (cudnn_tensor): The absolute maximum of output tensor.
```

### Scaled Dot Product Attention Backward FP8

This operation computes the gradients for scaled dot product attention in the FP8 (8-bit floating point) datatype, using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). The user is required to pass the stats tensor from the forward operation to the backward operation as input.

#### Configurable Options:

All the options mentioned in the forward FP8 operation, including ragged tensors and GQA/MQA, are applicable in the backward operation as well.

#### Tensors

The tensors in backward operation are defined as the following:

$dV = S^TdO$

$dS = dOV^T$

$dP = \text{dSoftmax}(dS)$

$dQ = dPK$

$dK = QdP$

##### Input Tensors

| Tensor Name           | Device     | Data Type    | Dimensions                   |
|-----------------------|------------|--------------|------------------------------|
| Q                     | GPU        | E4M3 or E5M2 | $(B, H_{q}, S_{q}, D_{qk})$  |
| K                     | GPU        | E4M3 or E5M2 | $(B, H_{k}, S_{kv}, D_{qk})$ |
| V                     | GPU        | E4M3 or E5M2 | $(B, H_{v}, S_{kv}, D_{v})$  |
| O                     | GPU        | E4M3 or E5M2 | $(B, H_{q}, S_{q}, D_{v})$   |
| dO                    | GPU        | E4M3 or E5M2 | $(B, H_{q}, S_{q}, D_{v})$   |
| Descale Q             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale K             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale V             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale O             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale dO            | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale S             | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Descale dP            | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Scale S               | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Scale dQ              | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Scale dK              | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Scale dV              | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Scale dP              | GPU        | FP32         | $(1, 1, 1, 1)$               |

##### Output Tensors

| Tensor Name           | Device     | Data Type    | Dimensions                   |
|-----------------------|------------|--------------|------------------------------|
| dQ                    | GPU        | E4M3 or E5M2 | $(B, H_{q}, S_{q}, D_{qk})$  |
| dK                    | GPU        | E4M3 or E5M2 | $(B, H_{k}, S_{kv}, D_{qk})$ |
| dV                    | GPU        | E4M3 or E5M2 | $(B, H_{v}, S_{kv}, D_{v})$  |
| Amax dQ               | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Amax dK               | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Amax dV               | GPU        | FP32         | $(1, 1, 1, 1)$               |
| Amax dP               | GPU        | FP32         | $(1, 1, 1, 1)$               |

Where,

- $B$ is the batch size
- $H_{q}$ is the number of query heads
- $H_{k}$ is the number of key heads
- $H_{v}$ is the number of value heads
- $S_{q}$ is the sequence length of the query
- $S_{kv}$ is the sequence length of the key and value
- $D_{qk}$ is the embedding dimension per head of query and key
- $D_{v}$ is the embedding dimension per head of value

#### Limitations:
All the limitations mentioned in the forward operation are applicable in the backward operation as well.

#### C++ API:
```cpp
// returns [dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP]
std::array<std::shared_ptr<Tensor_attributes>, 7>
Graph::sdpa_fp8_backward(std::shared_ptr<Tensor_attributes> q,
                         std::shared_ptr<Tensor_attributes> k,
                         std::shared_ptr<Tensor_attributes> v,
                         std::shared_ptr<Tensor_attributes> o,
                         std::shared_ptr<Tensor_attributes> dO,
                         std::shared_ptr<Tensor_attributes> Stats,
                         std::shared_ptr<Tensor_attributes> descale_q,
                         std::shared_ptr<Tensor_attributes> descale_k,
                         std::shared_ptr<Tensor_attributes> descale_v,
                         std::shared_ptr<Tensor_attributes> descale_o,
                         std::shared_ptr<Tensor_attributes> descale_do,
                         std::shared_ptr<Tensor_attributes> descale_s,
                         std::shared_ptr<Tensor_attributes> descale_dp,
                         std::shared_ptr<Tensor_attributes> scale_s,
                         std::shared_ptr<Tensor_attributes> scale_dq,
                         std::shared_ptr<Tensor_attributes> scale_dk,
                         std::shared_ptr<Tensor_attributes> scale_dv,
                         std::shared_ptr<Tensor_attributes> scale_dp,
                         SDPA_fp8_backward_attributes attributes);
```

The `options` parameter of type `SDPA_fp8_backward_attributes` is used to control the attributes of the forward operation, as detailed below:


```
SDPA_fp8_backward_attributes&
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_backward_attributes&
set_attn_scale(float const value);

SDPA_fp8_backward_attributes&
set_causal_mask(bool const value);
```

#### Python API: 
```
Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    o (cudnn_tensor): The output data.
    dO (cudnn_tensor): The output gradient data.
    stats (cudnn_tensor): The softmax statistics in case the operation is in a training step.
    descale_q (cudnn_tensor): Descale factor for query.
    descale_k (cudnn_tensor): Descale factor for key.
    descale_v (cudnn_tensor): Descale factor for value.
    descale_o (cudnn_tensor): Descale factor for output.
    descale_dO (cudnn_tensor): Descale factor for output gradient.
    descale_s (cudnn_tensor): Descale factor for S tensor.
    descale_dP (cudnn_tensor): Descale factor for P gradient tensor.
    scale_s (cudnn_tensor): Scale factor for S tensor.
    scale_dQ (cudnn_tensor): Scale factor for query gradient.
    scale_dK (cudnn_tensor): Scale factor for key gradient.
    scale_dV (cudnn_tensor): Scale factor for value gradient.
    scale_dP (cudnn_tensor): Scale factor for dP gradient.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.

Returns:
    dQ (cudnn_tensor): The query gradient data.
    dK (cudnn_tensor): The key gradient data.
    dV (cudnn_tensor): The value gradient data.
    amax_dQ (cudnn_tensor): The absolute maximum of query gradient tensor.
    amax_dK (cudnn_tensor): The absolute maximum of key gradient tensor.
    amax_dV (cudnn_tensor): The absolute maximum of value gradient tensor.
    amax_dP (cudnn_tensor): The absolute maximum of dP tensor.
```

### Appendix A 
Tensor Layouts:
Q, K, V, O and corresponding gradients layout support. cuDNN API expresses the layout of tensors based on strides.

For example, let Q have dimensions = [5, 7, 4, 3], and strides = [84, 12, 3, 1]
An element at index [i, j, k, l] can be accessed at the position of Q_ptr + i * 84 + j * 12 + k * 3 + l * 1

Notice how the strides are multiplied to the indices to get the position of all elements.
Below we will go through the standard usage of the attention tensors and how they can be expressed in cuDNN.

  1. Q, K, V are different matrices with strided layout
  This is the basic case where the user can specify dims and strides for each of Q, K and V and it works as the example given above.
  The only limitation is that stride corresponding to the hidden dimension per head (d, last dim in Q) needs to be 1.

  2. Q, K, V are interleaved 
  This is a special case of (1) and can be described in a strided layout as well. 
  For example, Q, K and V can be a single matrix of dims (batch (b), number_of_heads (h), sequence_length (s), 3, hidden_dim_per_head(d))
  Strides of Q can be defined as [h * s * 3 * d, s * 3 * d, 3 * d, 1]
  Notice how the 3 is multiplied to the strides corresponding to b, h and s because of the interleaving.

  3. There are some special cases when all tokens are not valid and Q, K, V can be in special layouts
    Let Q tensor have two sequences (i.e batch = 2, number_of_heads = 1) with max_seq_len = 8 and actual_seq_len = [2, 3]
    Conider two tokens "aa" & "bbb".
      - Fully padded layout

        aa000000
        bbb00000
        Dims = [b=2, h=1, s=8, d=64]
        Strides = [512, 512, 64, 1]
        
        CUDNN gets indication of the actual sequence lengths using the seq_len_q and the seq_len_kv and cuts the computation at these values. Please enable use_padding_mask also for this case. CUDNN reads the data based on the strides.

      - Fully packed layout
        aabbb000
        00000000
        Dims = [b=2, h=1, s=8, d=64]
        Strides = [512, 512, 64, 1]

        The strides remain the same but they are incorrect as the second batch begins at 64*2. Therefore, we have an API called "ragged_offset" which is a b+1 size tensor telling where each batch begins. The b+1 element is where the last batch ends.
        Users can set <tensor>.set_ragged_offset(<ragged_offset_tensor>)
        For this example ragged_offset = [0, 128, 320]
        Actual sequence length still have to be provided with padding mask.

      - Valid tokens in a batch are packed together
        aa00bbb0
        00000000

        User just needs to update the ragged offset to = [0, 256, 448]

      - Valid tokens are not packed together
        a0abbb00
        bb000000
        
        Ragged offset is insufficient to represent this. This case is NOT supported.

### Appendix B
Workspace limit:
Scaled Dot Product Attention Backward improves performance by using an optional dP workspace tensor. This tensor's memory consumption increases quadratically with the sequence length. The following describes the behavior of the `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT` environment variable, which allows the user to change the GPU memory limit for this workspace tensor:
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = unset`  
    The optimization will utilize workspace memory until reaching the default limit of 256MB.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = -1`  
    Workspace optimization is always enabled, regardless of memory usage.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = 0`  
    Workspace optimization is always disabled, avoiding the additional memory usage.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = n`  
    Allows workspace optimization up to a user-defined limit of n bytes, accommodating systems with varying GPU memory capacities.

### Appendix C
To dump the dropout mask generated by the Philox RNG dropout implementation for debugging purposes, users can use the `rng_dump` option. This option requires users to pass a tensor of dimensions $(B, H_{q}, S_{q}, S_{kv})$ 

### Miscellaneous
- FE provides shadow enums which help avoid users to workaround having different enums for different cudnn versions.
- The cudnn backend enums are changed as follows:
    - `cudnnBackend<enum_name>` -> `cudnn_frontend::<enum_name>`
    - `cudnn<enum_name>` -> `cudnn_frontend::<enum_name>`