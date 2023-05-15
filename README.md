# cuDNN Frontend API

## Introduction
The cuDNN frontend API is a C++ header-only library that wraps the [cuDNN C backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api). Both the frontend and backend APIs are entry points to the same set of functionality that we commonly refer to as the "[graph API](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion)".

While there are two entry points to the graph API (i.e. backend and frontend), we expect that most users will use the frontend entry point because:

- It is less verbose without loss of control. All functionality accessible through the backend API is also accessible through the frontend API.
- It adds functionality on top of the backend API, like errata filters and autotuning.

Also, for those who want to use the backend API, the frontend source can serve as a reference implementation.

## Usage
In order to include the entire library, include the cudnn_frontend header file `cudnn_frontend.h` into your compilation unit.

## Organization
Each `cudnnBackendDescriptorType_t` documented in the enum is organized into its header file.
- cudnn_frontend_Tensor.h         -> CUDNN_BACKEND_TENSOR_DESCRIPTOR
- cudnn_frontend_ConvDesc.h       -> CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR
- cudnn_frontend_PointWiseDesc.h  -> CUDNN_BACKEND_POINTWISE_DESCRIPTOR
- cudnn_frontend_MatMulDesc.h     -> CUDNN_BACKEND_MATMUL_DESCRIPTOR
- cudnn_frontend_ReductionDesc.h  -> CUDNN_BACKEND_REDUCTION_DESCRIPTOR
- cudnn_frontend_Operation.h      -> CUDNN_BACKEND_OPERATION_*_DESCRIPTOR
- cudnn_frontend_OperationGraph.h -> CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
- cudnn_frontend_Heuristics.h     -> CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR
- cudnn_frontend_Engine.h         -> CUDNN_BACKEND_ENGINE_DESCRIPTOR
- cudnn_frontend_EngineConfig.h   -> CUDNN_BACKEND_ENGINECFG_DESCRIPTOR
- cudnn_frontend_ExecutionPlan.h  -> CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
- cudnn_frontend_ExecutionPlan.h  -> CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
- cudnn_frontend_VariantPack.h    -> CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR

### Utility Functions
- cudnn_frontend_find_plan.h          -> Implements the `cudnnFindPlan` function
- cudnn_frontend_get_plan.h           -> Implements the `cudnnGetPlan` function
- cudnn_frontend_Filters.h            -> List of helpful utility functions to filter out execution plans
- cudnn_frontend_ExecutionPlanCache.h -> Describes and implements the execution plan caching.

### Logging
- cudnn_frontend_Logging.h -> Implements a basic logging framework for cudnn_frontend

### Error Handling 
- cudnn_frontend_utils.h

### Fallback Lists
- cudnn_frontend_EngineFallbackList.h -> Provides a fallback engine id if the heuristics do not provide an executable engine.

## Samples
Multiple samples of convolution, dgrad, wgrad and convBiasAct are added in `samples/test_list.cpp` and `samples/conv_sample.cpp`.  
Samples of runtime fusion are added in `samples/test_list.cpp` and `samples/fusion_sample.cpp`.  

Sample tests are written using the [Catch2](https://github.com/catchorg/Catch2) C++ test framework.

### How to build samples:
     - Provide CUDA according to: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
     - CUDNN_PATH has the cudnn installation.
        - Headers are in CUDNN_PATH/include
        - Libraries are in CUDNN_PATH/lib or CUDNN_PATH/lib64 or CUDNN_PATH/lib/x64

     From Project Root,

     mkdir build; cd build
     cmake -DCUDNN_PATH=/path/to/cudnn -DCUDAToolkit_ROOT=/path/to/cuda  ../
     cmake --build . -j16
     bin/samples

     - You can skip building samples by providing CUDNN_FRONTEND_BUILD_SAMPLES=0 to the cmake.
    
## cudnnFindPlan and cudnnGetPlan:
Prior to cuDNN V8, cuDNN provided `cudnnFindConvolution*` and `cudnnGetConvolution*` functions, which provided a way to sample all the algorithms for a given problem and study the run times. This can be further used to cache the best algorithms for a given problem.  In cuDNN V8, this has been replaced with `cudnnFindPlan` and `cudnnGetPlan`.

In order to use `cudnnFindPlan`, a user needs to provide:
- Source for a pruned list of `engineConfig`s for the given problem statement
- Filter function to Filter out the execution plan based on the prerequisite conditions

The `cudnnFindPlan` in turn
- Creates a set of execution plans that are supported
- Execute each filtered plan and ranks them in order of the execution plan runtime

The most common `engineConfig` generation is the built-in heuristics of cuDNN V8. Generally, this is appended with the fallback list. An example of usage can be seen in `run_from_cudnn_find(...)` function in `conv_sample.cpp`.

## Errata Filter:
Errata filter gives the cuDNN team an opportunity to block certain faulty kernels from being executed. cuDNN team can eitherprovide a json file which blocks certain engine configs from being executed. The users can augment to this list if they find certain characteristics to be undesirable (Eg. Bad memory access, Execution plan failure). Users can either declare the json file statically or load from a file during runtime using the environment variable "CUDNN_ERRATA_JSON_FILE".

#### Json format
    version             : 1    - Mandatory. Tells the format version of the json.
    rules               : []   - Mandatory. Array of rule object which identifies the engine config
    rule_id             : ""   - Optional.  Used to uniquely identify a rule. Has no purpose other than being easy to debug.
    operation           : ""   - Mandatory. Stringified version of the operation graph.
    engine              : ""   - Mandatory. Stringified version of the engine ID.
    knob                : ""   - Optional.  Stringified version of the knob. If specified only the engineConfig for the engine matching the knobs will be blocked. Else, all possible combination of knobs for the engine will be blocked.
    input_shape         : []   - Optional. Array of input shape for kernel (ex. [64, 32, 128, 128]) to be filtered out. Use -1 if you don't want to filter that dimension. (ex. [-1, -1, 128, 128] to only filter HxW for NCHW format)
    filter_shape        : []   - Optional. Array of kernel/filter shape for kernel (ex. [32, 32, 5, 5]) to be filtered out. Use -1 if you don't want to filter that dimension. (ex. [-1, -1, 5, 5] to only filter 5x5 filter sizes)
    shape_format        : ""   - Mandatory if input_shape and/or kernel_shape is present. Optional otherwise. Shape format of tensors as a string. (Ex. "NCHW", "NHWC").
    cudnn_version_start : 0    - Optional. Denotes the cudnn version after which the engine started having issues.
    cudnn_version_end   : -1   - Optional. Denotes the cudnn_version when the issue was fixed. "-1" denotes its an ongoing issue.
    arch                : ""   - Optional. Architectures where this kernel might be faulty.

PS: The errata filter note is still in beta version. We may add/modify certain features as necessary.

## Execution Plan Caching
cuDNN through heuristics provides a way to query a list of good engine configs. Based on this query we build the cudnn_frontend_find_plan function which runs all the engineConfig(s) on the given user system and returns a sorted list of plans. This process of running multiple plans through several iterations is time consuming. The ExecutionPlanCache allows the user to build a cache with operation graph as the key to query an execution plan. It is the responsibilty of the user to maintain different caches for different types of operation_graphs (For eg. different cache for convolutionForward compared to Dgrad or Wgrad). The `is_fastest_plan_stable` builds on top of this by making sure the same plan is chosen by the cudnnFind multiple times.

### API:
    - void add_plan_to_cache(const cudnn_frontend::OperationGraph &op_graph, const cudnn_frontend::ExecutionPlan &plan) : Creates a mapping between the operation graph and executionPlan
    - bool get_plan_from_cache(const cudnn_frontend::OperationGraph &op_graph, const cudnn_frontend::ExecutionPlan *&plan) : Sets the executionPlan in the plan pointer and returns true if found.
    - cudnnFindPlanAndCache(cudnnHandle_t handle, cudnn_frontend::OperationGraph &opGraph, cudnn_frontend::VariantPack const &variantPack, cudnn_frontend::ExecutionPlanCache &cache, Predicate pred) -> cudnn_frontend::ExecutionPlan
      The above API chains the output of cudnn_frontend_find_plan and caches the result for future usage. 


PS: ExecutionPlanCaching today supports only single operation operation_graphs.

## Execution Plan Serialization and Deserialization (Experimental)
cuDNN v8.4 and above provides exeuction plan serialization and deserialization to save the execution plan as a string in JSON format. The execution plan can be then restored from that string at a later point, and this also saves compilation time compared to rebuilding the plan from scratch. Currently, this is an experimental feature that only supports the runtime fusion engine. No forward/backward or cross-device compatibility guarantee is offered at this time.

### API:
    - std::string cudnn_frontend::ExecutionPlan_v8::getJsonRepresentation() : Serialize the execution plan into a string in JSON format.
    - cudnn_frontend::ExecutionPlan_v8&& cudnn_frontend::ExecutionPlanBuilder_v8::loadFromJson(const std::string &json_plan) : Deserialize from a string containing the JSON representation of the execution plan.

## Logging
cuDNN Frontend API logging records execution flow through cuDNN frontend API. This functionality is disabled by default, and can be enabled through methods described in this section.

### Method 1: Using Environment Variables:
| Environment variables                             | CUDNN_FRONTEND_LOG_INFO=0 | CUDNN_FRONTEND_LOG_INFO=1 |
| ------------------------------------------------- | ------------------------- | -----------               |
| CUDNN_FRONTEND_LOG_FILE not set                   | No Logging                | No Logging                |
| CUDNN_FRONTEND_LOG_FILE set to stdout or stderr   | No Logging                | Logging to cout or cerr   |
| CUDNN_FRONTEND_LOG_FILE set to filename.txt       | No Logging                | Logging to the filename   |

### Method 2: Using API calls:
Calling `cudnn_frontend::isLoggingEnabled() = true|false` has same effect of setting the environment variable.
Calling `cudnn_frontend::getStream() = stream_name` can be used to assign the output stream directly. 

## Documentation
Documentation can be found at https://nvidia.github.io/cudnn-frontend/

## Contributing:
At this point we are not accepting any external PRs. Please create an issue in github and we will get to it.

## Feedback
Support, resources, and information about cuDNN can be found online at https://developer.nvidia.com/cudnn. 

For questions or to provide feedback, please contact cuDNN@nvidia.com.
