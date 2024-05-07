
## Table of Contents
1. [Fprop](#Convolution Fprop)
2. [Dgrad](#Convolution Dgrad)
3. [Wgrad](#Convolution Wgrad)

### Convolution Fprop
Convolution fprop computes:
$$ response = image * filter $$

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_fprop_attributes);
```

Conv_fprop attributes is a lightweight structure with setters:  
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
```

Python API: 
- conv_fprop
    - image
    - weight
    - padding
    - stride
    - dilation
    - compute_data_type
    - name

### Convolution Dgrad
Convolution dgrad computes data gradient during backpropagation.

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_dgrad_attributes);
```

Conv_dgrad attributes is a lightweight structure with setters:  
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
```

Python API: 
- conv_dgrad
    - filter
    - loss
    - padding
    - stride
    - dilation
    - compute_data_type
    - name

### Convolution Wgrad
Convolution wgrad computes weight gradient during backpropagation.

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_wgrad_attributes);
```

Conv_wgrad attributes is a lightweight structure with setters:  
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
```

Python API: 
- conv_wgrad
    - image
    - loss
    - padding
    - stride
    - dilation
    - compute_data_type
    - name
