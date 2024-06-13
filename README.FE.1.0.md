# cuDNN FrontEnd(FE) v1.0 API

## Table of Contents
1. [Introduction](#Introduction)
2. [Workflow](#Workflow)
3. [APIs](#APIs)
4. [Samples](#Samples)
5. [Operations](#Operations)
6. [Miscellaneous](#Miscellaneous)

## Introduction
FE v1.0 API is aimed to extend functionality and usage exposed by the [cuDNN C backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api). Both C++ and python APIs are provided, and both have functional parity.  
For a general introduction to FE, please start with README.md.

In the frontend v1 API, you can describe multiple operations that form subgraphs through a persistent cudnn_frontend::graph::Graph object. Unlike the frontend v0.x API, you don't have to worry about specifying shapes and sizes of the intermediate virtual tensors. The frontend v1 API extends the groundwork of earlier versions and introduces a new set of APIs to further simplify the workflow. 

Additionally, the frontend v1 API provides Python bindings to all API. Refer to samples/cpp and samples/python for more details on its usage.
With the release of v1, we are bumping up the minimum supported cuDNN version to 8.5.0.

## Workflow
The steps involved in building and running a cudnn graph are as follows:
1. Create a cudnn graph and specify the global properties. The global properties like compute precision and input/output data type help infer properties that are not explicitly mentioned.
2. Create and add the input tensors.
3. Create and add the operation nodes. The outputs of these operation are of tensor type and can be sequentially used as inputs to the next node.
4. Validate the operation graph. This step makes sure the graph is well built and does not have hanging tensors or node.
5. Build the cudnn operation graph. This step lowers the graph into cudnn dialect.
6. Create the execution plan, based on the heuristics type of your choice.
7. Check support of the operation graph.
8. [Optional] Filter out the plans by your custom criteria (Optional).
9. Build (one or all) the execution plans.
10. [Optional] Run autotuning on the filtered plan (Optional).
11. Execute the graph with the relevant data pointers.
    
## APIs
FE v1.0 API follows a functional style of building a graph. Operations take in input tensors and return output tensors. This also allows composition of operations. 

| Purpose                                                                  | C++ API                                              | Python API                                                                                       |
|--------------------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Create tensor                                                            | tensor                                               | tensor                                                                                           |
| [Convolution Fprop](docs/operations/Convolutions.md)                     | conv_fprop <br>Conv_fprop_attributes                 | conv_fprop                                                                                       |
| [Convolution Dgrad](docs/operations/Convolutions.md)                     | conv_dgrad <br>Conv_dgrad_attributes                 | conv_dgrad                                                                                       |
| [Convolution Wgrad](docs/operations/Convolutions.md)                     | conv_wgrad <br>Conv_wgrad_attributes                 | conv_wgrad                                                                                       |
| [Matrix Multiplication](docs/operations/Matmul.md)                       | matmul <br> Matmul_attributes                        | matmul                                                                                           |
| [Pointwise Operations](docs/operations/Pointwise.md)                     | pointwise <br> Pointwise_attributes                  | - add<br>- bias<br>- rqsrt<br>- sub<br>- mul<br>- scale<br>- relu<br>- elu<br>- gelu<br>- cmp_gt |
| [Batch Normalization](docs/operations/Normalizations.md)                 | batchnorm <br>Batchnorm_attributes                   | batchnorm                                                                                        |
| [Batch Norm bprop](docs/operations/Normalizations.md)                    | batchnorm_backward <br>Batchnorm_backward_attributes | batchnorm_backward                                                                               |
| Generate stats of output                                                 | genstats <br>Genstats_attributes                     | genstats                                                                                         |
| BN Finalize of stats                                                     | bn_finalize <br>BN_finalize_attributes               | bn_finalize                                                                                      |
| Dbn weight                                                               | dbn_weight <br>DBN_weight_attributes                 | dbn_weight                                                                                       |
| [Resampling](docs/operations/Resampling.md)                              | resample <br>Resample_attributes                     | resample                                                                                         |
| [Scale dot product attention](docs/operations/Attention.md)              | sdpa<br> SDPA_attributes                             | sdpa                                                                                             |
| [Scale dot product attention backward](docs/operations/Attention.md)     | sdpa_backward<br> SDPA_backward_attributes           | sdpa_backward                                                                                    |
| [Scale dot product attention FP8](docs/operations/Attention.md)          | sdpa_fp8<br> SDPA_fp8_attributes                     | sdpa_fp8                                                                                         |
| [Scale dot product attention backward FP8](docs/operations/Attention.md) | sdpa_fp8_backward<br> SDPA_fp8_backward_attributes   | sdpa_fp8_backward                                                                                |

### Creating the Graph
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

Tensor attributes is a lightweight structure with setters for each attribute.
- `cudnn_frontend::graph::Tensor_attributes& set_data_type(cudnn_frontend::DataType_t)`
- `cudnn_frontend::graph::Tensor_attributes& set_dim(std::vector<int64_t>&)`
- `cudnn_frontend::graph::Tensor_attributes& set_stride(std::vector<int64_t>&)`
- `cudnn_frontend::graph::Tensor_attributes& set_is_virtual(bool)`
- `cudnn_frontend::graph::Tensor_attributes& set_is_pass_by_value(bool)`
- `cudnn_frontend::graph::Tensor_attributes& set_reordering_type(cudnn_frontend::TensorReordering_t)`
- `cudnn_frontend::graph::Tensor_attributes& set_name(std::string&)`

### Defining Operations
Operations take in mandatory input tensor via positional arguments. Optional input tensors are provided using corresponding setters in operation attributes. 

Operations return an ordered array of output tensors. Any optional outputs if not present will have their shared pointers pointing to `std::nullptr`.

Please looks at [operations](#Operations) section for more details. 

### Validating the Graph
Validate API ensures API usage is sound, checks against dangling tensors, etc.
Internally, any unspecified properties like dimensions, strides, etc are inferred.

```
cudnn_frontend::error_t cudnn_frontend::graph::Graph::validate()
```

### Building the Backend Graph
This method creates cudnn backend descriptors for all constituents of the graph.

```
cudnn_frontend::error_t cudnn_frontend::graph::Graph::build_operation_graph(cudnnHandle_t handle)
```

### Creating the Execution Plan
This method internally queries the heuristics for engine configs for the given heuristics modes.

```
cudnn_frontend::error_t cudnn_frontend::graph::Graph::get_execution_plans(std::vector<heur_mode_t>)
```

### Getting the Execution Plan Count
This method returns the number of execution plans returned by cudnn heuristics. Each plan gets an index from 0 to #plans-1, with 0 having top priority.

```
cudnn_frontend::int64_t
cudnn_frontend::Graph::get_execution_plan_count() const;
```

### Checking Graph Support
This method guarantees that executing the graph using plans queried will succeed.

```
cudnn_frontend::error_t cudnn_frontend::graph::Graph::check_support(cudnnHandle_t h);
```

###  Building the Execution Plan

This function builds execution plans queried with `create_execution_plan(...)` API.

There are two flavours of this API:

Use this method to build execution plans according to a policy. Suitable when trusting cudnn heuristics to return nest suitable execution plan with top priority.
```
cudnn_frontend::error_t
cudnn_frontend::graph::Graph::build_plan(
    cudnnHandle_t const &handle, 
    cudnn_frontend::BuildPlanPolicy_t const policy, 
    bool const do_multithreaded_builds
);
```

Use this method to build individual plan indices. Main use case is to build execution plans in parallel when autotuning.
Plan index to be used here can be queried with `get_execution_plan_count(...)` API.
```
cudnn_frontend::error_t
cudnn_frontend::Graph::build_plan_at_index(
    cudnnHandle_t const &handle,
    int64_t plan_index
);
```
### Filtering Plans (Optional)
Users can filter plans on numerical, behavioral notes, or plans that do not provide desired functional correctness.

```
cudnn_frontend::graph::Graph& cudnn_frontend::graph::Plans::select_numeric_notes(std::vector<cudnn_frontend::NumericalNote_t> const&);
cudnn_frontend::graph::Graph& cudnn_frontend::graph::Plans::select_behavior_notes(std::vector<cudnn_frontend::BehaviorNote_t> const&);

cudnn_frontend::graph::Graph& cudnn_frontend::graph::Plans::deselect_numeric_notes(std::vector<cudnn_frontend::NumericalNote_t> const&);
cudnn_frontend::graph::Graph& cudnn_frontend::graph::Plans::deselect_behavior_notes(std::vector<cudnn_frontend::BehaviorNote_t> const&);
cudnn_frontend::graph::Graph& cudnn_frontend::graph::Plans::deselect_workspace_greater_than(int64_t const workspace);
```

### Autotuning

Autotuning provides a way to execute different execution plans for a given graph and measure their relative performance under run time conditions.
This generally helps validate and improve upon the results provided by the heuristics. Please refer to [samples](samples/cpp/autotuning.cpp)

### Executing the Graph
Executing the graph requires device pointers to all input output tensors and a user allocated device workspace pointer.

Two flavours of execute exists, corresponding to `build_plans(...)` API.

This API already has a candidate execution plan set. Candidate execution plan get internally set either:
- if build_policy_t::HEURISTIC_CHOICE is used, or
- as the last plan built that got built.

```
cudnn_frontend::error_t
cudnn_frontend::graph::Graph::execute(
    cudnnHandle_t handle,
    std::unordered_map<std::shared_ptr<Tensor>, void *> var_pack,
    void* workspace
);
```

execute API also takes a plan index to target a specific plan. This may be used when autotuning, in conjunction with `build_plan_at_index(...)` API.
```
cudnn_frontend::error_t
cudnn_frontend::graph::Graph::execute(
    cudnnHandle_t handle,
    std::unordered_map<std::shared_ptr<Tensor>, void *> var_pack,
    void* workspace,
    int64_t plan_index
);
```

### Miscellaneous APIs

Get workspace to execute the current selected execution plan.

Can also take in a plan index to query workspace for. This may be used when autotuning, in conjunction with `build_plan_at_index(...)` API.

`int64_t get_workspace_size() const`
`int64_t get_workspace_size_plan_index(int64_t plan_index) const`

Get workspace to run autotune on all plans.

`get_autotune_workspace_size() const`

### Error handling

C++ API returns a error object which has a error code and error message. 

Python API throws an exception with similar error message to be handled in python API.

## Samples
Samples are meant to illustrate FE v1.0 API usage to users.  
- `samples/cpp` contains samples that use C++ API.
- `samples/python` contains samples that use python API.

Python samples are jupyter notebooks with step by step guide on using FE v1 API.

## Operations

Please look at [docs/operations](docs/operations/) for APIs of different operation types.