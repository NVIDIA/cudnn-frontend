
# Convolutions

(convolution-fprop)=
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
    - padding
    - stride
    - dilation
    - compute_data_type
    - name

(convolution-dgrad)=
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
    - filter
    - loss
    - padding
    - stride
    - dilation
    - compute_data_type
    - name

(convolution-wgrad)=
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
    - padding
    - stride
    - dilation
    - compute_data_type
    - name
