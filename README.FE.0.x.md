# cuDNN FE 0.x API

## Introduction
FE v0.x API is wraps [cuDNN C backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api) in C++ APIs.  
For a general introduction to FE, please first refer README.md.

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

## Samples

Samples are meant to illustrate FE v0.x API usage to users.  
- `samples/conv_samples.cpp` contains conv/dgrad/wgrad-fusion samples.  
- `samples/norm_samples.cpp` contains batch normalization-fusion samples.
- `samples/fusion_samples.cpp` contains fusion samples that use cudnn's runtime fusion engine.
- `samples/fused_mha_samples.cpp` contains flash attention sample.

Sample tests are written using [Catch2](https://github.com/catchorg/Catch2) test framework and are controlled by `samples/test_list.cpp`.  
    
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

## Deprecation
v0.x API may be deprecated in version 2.0 of the API. Please, consider adopting 1.0 API. If there are any issues, or missing functionalities in v1.0 API, please create a gitlab issue for this.