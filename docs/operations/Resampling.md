
# Resampling

(resampling-forward)=
## Resampling Forward

The resample operation represents the resampling of the spatial dimensions of an image to a desired value.

The output array contains two tensors:

- The resampled output tensor.
- The computed index tensor.

```{note} The index tensor is only output in training mode of max pooling. It can be fed to backward pass for faster performance.
```

### Resample Attributes

The Resample_attributes class is used to configure the resampling operation. It provides the following setters:

```
# The resampling mode, such as average pooling, max pooling, bi-linear, or cubic.
auto set_resampling_mode(ResampleMode_t const& value) -> Resample_attributes&;

# The padding mode, such as zero or neg infinity.
auto set_padding_mode(PaddingMode_t const& value) -> Resample_attributes&;

# The window size to be used for the resampling operation.
auto set_window(std::vector<int64_t> const& value) -> Resample_attributes&;
auto set_window(std::vector<cudnnFraction_t> const& value) -> Resample_attributes&;

# The stride values to be used for the resampling operation.
auto set_stride(std::vector<int64_t> const& value) -> Resample_attributes&;
auto set_stride(std::vector<cudnnFraction_t> const& value) -> Resample_attributes&;

# The padding values to be applied before and after the resampling input.
auto set_pre_padding(std::vector<int64_t> const& value) -> Resample_attributes&;
auto set_pre_padding(std::vector<cudnnFraction_t> const& value) -> Resample_attributes&;
auto set_post_padding(std::vector<int64_t> const& value) -> Resample_attributes&;
auto set_post_padding(std::vector<cudnnFraction_t> const& value) -> Resample_attributes&;

# A flag indicating whether the resampling is being performed during inference. 
auto set_is_inference(bool const value) -> Resample_attributes&;
```

For more information on exact support surfaces across different versions, refer to {ref}`resamplefwd-runtime-fusion-engine` in the *Frontend Developer Guide*.

Python API for resampling forward will be supported soon.

(resampling-backward)=
## Resampling Backward

To be supported soon.

