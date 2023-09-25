# cuDNN FrontEnd(FE) v1.0 API

## Table of Contents
1. [Introduction](#Introduction)
2. [Workflow](#Workflow)
3. [APIs](#APIs)
4. [Samples](#Samples)
5. [Operations](#Operations)
6. [Miscellaneous](#Miscellaneous)

## Introduction
FE v1.0 API is aimed to extend functionality and usage exposed by the [cuDNN C backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api). Both C++ and python APIs are provided with both having functional parity.  
For a general introduction to FE, please first refer README.md

## Workflow
The steps involved in building and running a cudnn graph are as follows:
1. Create a cudnn graph and specify the global properties. The global properties like compute precision and input/output data type help infer properties that are not explicitly mentioned.
2. Create and add the input tensors.
3. Create and add the operation nodes. The outputs of these operation are of tensor type and can be sequentially used as inputs to the next node.
4. Validate the operation graph. This step makes sure the graph is well built and does not have hanging tensors or node.
5. Build the cudnn operation graph. This step lowers the graph into cudnn dialect.
6. Get the execution plan, based on the heuristics type of your choice.
7. [Optional] Check support of the operation graph.
8. [Optional] Filter out the plans by your custom criteria (Optional).
9. [Optional] Run autotuning on the filter plan (Optional). 
10. Set the execution plan of choice back into the graph.
11. Execute the graph with the relevant data pointers.
    
## APIs
FE v1.0 API follows a functional style of building a graph. Operations take in input tensors and return output tensors. This also allows composition of operations. 

| Purpose                 | C++ API                                                   | Python API   |
| ---                     | ---                                                       | ---          |
| Create tensor           | tensor                                                    | tensor       |
| Convolution Fprop       | conv_fprop <br>Conv_fprop_attributes                      | conv_fprop   |
| Convolution Dgrad       | conv_dgrad <br>Conv_dgrad_attributes                      | conv_dgrad   |
| Convolution Wgrad       | conv_wgrad <br>Conv_wgrad_attributes                      | conv_wgrad   |
| Matrix Multiplication   | matmul <br> Matmul_attributes                             | matmul       |
| Pointwise Operations    | pointwise <br> Pointwise_attributes                       | - add<br>- bias<br>- rqsrt<br>- sub<br>- mul<br>- scale<br>- relu<br>- elu<br>- gelu<br>- cmp_gt       |
| Batch Normalization     | batchnorm <br>Batchnorm_attributes                        | batchnorm    |
| Batch Norm bprop        | batchnorm_backward <br>Batchnorm_backward_attributes      | batchnorm_backward    |
| Generate stats of output| genstats <br>Genstats_attributes                          | genstats     |
| BN Finalize of stats    | bn_finalize <br>BN_finalize_attributes                    | bn_finalize  |
| Dbn weight              | dbn_weight <br>DBN_weight_attributes                      | dbn_weight   |
| Scale dot product flash attention | scaled_dot_product_flash_attention<br> Scaled_dot_product_flash_attention_attributes | scaled_dot_product_flash_attention |
| Scale dot product flash attention_backward | scaled_dot_product_flash_attention_backward<br> Scaled_dot_product_flash_attention_backward_attributes | scaled_dot_product_flash_attention_backward |

### Create Graph
Instantiate an object of class `cudnn_frontend::graph::Graph` which will house tensors and operations.  

Optional graph level attributes can be set on the object:
- `cudnn_frontend::graph::Graph& set_io_data_type(cudnn_frontend::DataType_t)`
- `cudnn_frontend::graph::Graph& set_intermediate_data_type(cudnn_frontend::DataType_t)`
- `cudnn_frontend::graph::Graph& set_compute_data_type(cudnn_frontend::DataType_t)`
These attributes are meant to used as default in case they are not provided for constituent tensors and operations.

### Define Tensors
Users create input tensors to provide to operations within a graph. To add tensors in a graph, use:  
`std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> cudnn_frontend::graph::tensor(cudnn_frontend::graph::Tensor_attributes)`.  
As the API returns a shared pointer, both the user and FE graph are owners of the tensor.  

Tensor attributes is a lighweight structure with setters for each attribute.  
- `cudnn_frontend::graph::Tensor_attributes& set_data_type(cudnn_frontend::DataType_t)`
- `cudnn_frontend::graph::Tensor_attributes& set_dim(std::vector<int64_t>&)`
- `cudnn_frontend::graph::Tensor_attributes& set_stride(std::vector<int64_t>&)`
- `cudnn_frontend::graph::Tensor_attributes& set_is_virtual(bool)`
- `cudnn_frontend::graph::Tensor_attributes& set_is_pass_by_value(bool)`
- `cudnn_frontend::graph::Tensor_attributes& set_reordering_type(cudnn_frontend::TensorReordering_t)`
- `cudnn_frontend::graph::Tensor_attributes& set_name(std::string&)`

### Define Operations
Operations take in mandatory input tensor via positional arguments. Optional input tensors are provided using corresponding setters in operation attributes. 

Operations return an ordered array of output tensors. Any optional outputs if not present will have their shared pointers pointing to `std::nullptr`.

Please looks at [operations](#Operations) section for more details. 

### Validate graph
Validate API ensures API usage is sound, checks against dangling tensors, etc.
Internally, any unspecified properties like dimensions, strides, etc are inferred.

```
cudnn_frontend::error_t cudnn_frontend::graph::Graph::validate()
```

### Build cudnn backend graph
This method creates cudnn backend descriptors for all constituents of the graph.

```
cudnn_frontend::error_t cudnn_frontend::graph::Graph::build_operation_graph(cudnnHandle_t handle)
```

### Get Execution plans
This method returns a list of execution plans that can potentially run the FE graph.

```
cudnn_frontend::graph::Plans cudnn_frontend::graph::Graph::get_execution_plans(heur_mode_t)
```

### Filter plans
Users can filter out plans against numerical, behavioral notes, or plans that do not provide desired functional correctness.

```
cudnn_frontend::graph::Plans& cudnn_frontend::graph::Plans::filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const&);
cudnn_frontend::graph::Plans& cudnn_frontend::graph::Plans::filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const&);
cudnn_frontend::graph::Plans& cudnn_frontend::graph::Plans::filter_out_workspace_greater_than(int64_t max_allowed_workspace);
```

### Check graph support
This method guarantees that executing the graph using plans queried will succeed.

```
cudnn_frontend::error_t Plans::check_support();
```

### Autotune

Autotuning provides a way to execute different execution plans for a given graph and measure their relative performance under run time conditions.
This generally helps validate and improve upon the results provided by the heuristics.

The current API to perform the autotuning on the filtered plans:
```
    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> variants,
             void *workspace,
             void *user_impl = nullptr);

```

### Set Execution plans
After checking support, filtering and/or autotuning, execution plans can be set in descending order of preference.

```
cudnn_frontend::error_t
cudnn_frontend::graph::Graph::set_execution_plans(cudnn_frontend::::graph::Plans const&)
```

### Execute
Executing graph requires device pointers to all input output tensors and a user alloaction device workspace pointer.

```
cudnn_frontend::error_t
cudnn_frontend::graph::Graph::execute(cudnnHandle_t handle,
                                        std::unordered_map<std::shared_ptr<Tensor>, void *> var_pack,
                                        void* workspace);
```

## Samples
Samples are meant to illustrate FE v1.0 API usage to users.  
- `samples/cpp` contains samples that use C++ API.
- `samples/python` contains samples that use python API.

C++ samples are written using [Catch2](https://github.com/catchorg/Catch2) test framework.  
Python samples are written using [pytest](https://github.com/pytest-dev/pytest) and [pytorch](https://pytorch.org), with both requiring external installation.

## Operations

Please look at docs/operations for APIs of different operation types.
