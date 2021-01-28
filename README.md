# cuDNN Frontend API

## Introduction
The cuDNN Frontend API is a C++ header-only library that demonstrates how to use the cuDNN C backend API. The cuDNN C backend API is documented in the [cuDNN developer guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html). 

## Usage
In order to include the entire library, include the cudnn_frontend header file `cudnn_frontend.h` into your compilation unit.

## Organization
Each `cudnnBackendDescriptorType_t` documented in the enum is organized into its header file.
- cudnn_frontend_Tensor.h         -> CUDNN_BACKEND_TENSOR_DESCRIPTOR
- cudnn_frontend_ConvDesc.h       -> CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR
- cudnn_frontend_PointWiseDesc.h  -> CUDNN_BACKEND_POINTWISE_DESCRIPTOR
- cudnn_frontend_Operation.h      -> CUDNN_BACKEND_OPERATION_*_DESCRIPTOR
- cudnn_frontend_OperationGraph.h -> CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
- cudnn_frontend_Heuristics.h     -> CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR
- cudnn_frontend_Engine.h         -> CUDNN_BACKEND_ENGINE_DESCRIPTOR
- cudnn_frontend_EngineConfig.h   -> CUDNN_BACKEND_ENGINECFG_DESCRIPTOR
- cudnn_frontend_ExecutionPlan.h  -> CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
- cudnn_frontend_VariantPack.h    -> CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR

### Utility Functions
- cudnn_frontend_find_plan.h -> Implements the `cudnnFindPlan` function
- cudnn_frontend_get_plan.h  -> Implements the `cudnnGetPlan` function
- cudnn_frontend_Filters.h   -> List of helpful utility functions to filter out execution plans

### Error Handling 
- cudnn_frontend_utils.h

### Fallback Lists
- cudnn_frontend_EngineFallbackList.h -> Provides a fallback engine id if the heuristics do not provide an executable engine.

## Samples
Multiple samples of convolution, dgrad, wgrad and convBiasAct are added in `samples/test_list.cpp` and `samples/conv_sample.cpp`.  

Sample tests are written using the [Catch2](https://github.com/catchorg/Catch2) C++ test framework.

### How to build samples:
     - CUDA_PATH has the cuda installation. 
        - Include files are in CUDA_PATH/include
        - Link files are in CUDA_PATH/lib64
     - CUDNN_WRAP_PATH has the wrapper header files.

     make CUDA_PATH=/usr/local/cuda CUDNN_WRAP_PATH=/usr/local/include/
    
## cudnnFindPlan and cudnnGetPlan:
Prior to cuDNN V8, cuDNN provided `cudnnFindConvolution*` and `cudnnGetConvolution*` functions, which provided a way to sample all the algorithms for a given problem and study the run times. This can be further used to cache the best algorithms for a given problem.  In cuDNN V8, this has been replaced with `cudnnFindPlan` and `cudnnGetPlan`.

In order to use `cudnnFindPlan`, a user needs to provide:
- Source for a pruned list of `engineConfig`s for the given problem statement
- Filter function to Filter out the execution plan based on the prerequisite conditions

The `cudnnFindPlan` in turn
- Creates a set of execution plans that are supported
- Execute each filtered plan and ranks them in order of the execution plan runtime

The most common `engineConfig` generation is the built-in heuristics of cuDNN V8. Generally, this is appended with the fallback list. An example of usage can be seen in `run_from_cudnn_find(...)` function in `conv_sample.cpp`.

## Feedback
Support, resources, and information about cuDNN can be found online at https://developer.nvidia.com/cudnn. 

For questions or to provide feedback, please contact cuDNN@nvidia.com.
