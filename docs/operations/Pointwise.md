## Table of Contents
1. [Pointwise](#Pointwise)
2. [Reduction](#Reduction)

### Pointwise
Pointwise performs an elementwise operation between two tensors. The operation used is controlled by pointwise mode `cudnn_frontend::PointwiseMode_t`.   

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);

std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);

std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);
```
where the pointwise mode dictates the API among the choices above.
Please refer to documentation of `cudnn_frontend::PointwiseMode_t` for details.

Pointwise attributes is a lightweight structure with setters:  
```
Pointwise_attributes&
set_mode(PointwiseMode_t)

Pointwise_attributes&
set_axis(int64_t)

Pointwise_attributes&
set_relu_lower_clip(float)

Pointwise_attributes&
set_relu_upper_clip(float)

Pointwise_attributes&
set_relu_lower_clip_slope(float)

Pointwise_attributes&
set_name(std::string const&)

Pointwise_attributes&
set_compute_data_type(DataType_t value)
```

Python API: 
- add
    - a
    - b
    - compute_data_type
    - name
- bias
    - input
    - bias
    - compute_data_type
    - name
- rsqrt
    - input
    - compute_data_type
    - name
- sub
    - a
    - b
    - compute_data_type
    - name
- mul
    - a
    - b
    - compute_data_type
    - name
- scale
    - input
    - scale
    - compute_data_type
    - name
- relu
    - input
    - compute_data_type
    - name
- gelu
    - input
    - compute_data_type
    - name
- elu
    - input
    - compute_data_type
    - name
- cmp_gt
    - input
    - comparison
    - compute_data_type
    - name

### Reduction
Reduction operation reduces an input tensor using an operation controlled by `cudnn_frontend::ReductionMode_t`.
The dimensions in input tensors to reduce are deduced using output tensor dimensions.

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes>
reduction(std::shared_ptr<Tensor_attributes> input, Reduction_attributes);
```

Reduction attributes is a lightweight structure with setters:  
```
Reduction_attributes&
set_mode(ReductionMode_t)

Reduction_attributes&
set_name(std::string const&)

Reduction_attributes&
set_compute_data_type(DataType_t value)
```
