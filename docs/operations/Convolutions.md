
# Convolutions

## Convolution Fprop

Convolution fprop computes:

$$ response = image * filter $$

### C++ API


```
std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_fprop_attributes);
```

Conv_fprop_attributes is a lightweight structure with setters:  
```
Conv_fprop_attributes&
set_padding(std::vector<int64_t>)

Conv_fprop_attributes&
set_stride(std::vector<int64_t>)

Conv_fprop_attributes&
set_dilation(std::vector<int64_t>)

Conv_fprop_attributes&
set_name(std::string const&)

Conv_fprop_attributes&
set_compute_data_type(DataType_t value)

Conv_fprop_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_fprop
    - image
    - weight
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.

## CuTe DSL Fused Conv3D + Post-Operations

The experimental direct Python APIs `Conv3dRmsNormSiluSm100` and
`Conv3dRmsNormSiluPadSm100` fuse Conv3D with bias, optional residual addition,
channel RMS normalization, affine scale, and SiLU. Related variants provide
raw Conv3D, bias/residual/spatial padding, and causal input packing plus Conv3D.
These APIs are not selected automatically by `cudnn.pygraph`.

### Current Limitations

- SM100 or SM103 GPUs, `nvidia-cutlass-dsl >= 4.9`, and BF16 inference only;
  no backward implementation.
- Fixed `3x3x3` convolution, unit stride and dilation, and one group. The
  valid-convolution APIs consume already-padded input when padding is needed.
- Selected channel pairs from the WAN VAE `base_dim=160` family (160/320/640),
  not arbitrary channels or the 128/256/512 family. Integrated normalization
  supports only `160->160`, `160->320`, and `320->320`; C640 output uses raw
  convolution followed by standalone normalization. The causal input variant
  currently supports only `12->160`. Unsupported configurations have no fallback.
- Contiguous NTHWC activations, except `CausalConv3dWithCacheSm100`, which accepts
  strided NCTHW input. Weights require variant-specific packing before execution.
- Normalization uses an L2-norm clamp followed by `sqrt(C)` scaling, with fixed
  BF16 rounding points; it is not arbitrary-epsilon RMSNorm.
- Padding/cache layouts are fixed for streaming convolution. The fused
  normalization-and-padding variant writes current frames and zeros but leaves
  existing-history interiors for the caller to fill.

See the [fused Conv3D + post-operations API reference](../fe-oss-apis/conv3d_postops.md)
for the complete per-variant channel table, tensor shapes, normalization math,
weight formats, and history/stream contracts.

## Convolution Dgrad

Convolution dgrad computes data gradient during backpropagation.

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_dgrad_attributes);
```

Conv_dgrad_attributes is a lightweight structure with setters:  
```
Conv_dgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_dgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_dgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_dgrad_attributes&
set_name(std::string const&)

Conv_dgrad_attributes&
set_compute_data_type(DataType_t value)

Conv_dgrad_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_dgrad
    - loss
    - filter
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.

## Convolution Wgrad

Convolution wgrad computes weight gradient during backpropagation.

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_wgrad_attributes);
```

Conv_wgrad_attributes is a lightweight structure with setters:  
```
Conv_wgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_wgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_wgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_wgrad_attributes&
set_name(std::string const&)

Conv_wgrad_attributes&
set_compute_data_type(DataType_t value)

Conv_wgrad_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_wgrad
    - image
    - loss
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.
