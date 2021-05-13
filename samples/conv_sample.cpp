/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "conv_sample.h"
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>

namespace {

bool
isNonDeterministic(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(engine_config);
}

bool
isReducedPrecisionReduction(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION>(engine_config);
}

bool
isDownConvertingInputs(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(engine_config);
}

bool
isNonDeterministicOrisDownConverting(cudnnBackendDescriptor_t engine_config) {
    return isNonDeterministic(engine_config) || isDownConvertingInputs(engine_config);
}

bool
allowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

bool allowErrata(int64_t *padA) {
    return std::all_of(padA,padA + 2, [](int64_t pad) {
            return pad == 0;});
}

}
enum {
    X_TENSOR,
    Y_TENSOR,
    W_TENSOR,
    Z_TENSOR,
    B_TENSOR,
    AFTERADD_TENSOR,
    AFTERBIAS_TENSOR,
    AFTERCONV_TENSOR,
};

using common_conv_descriptors =
    std::tuple<cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::ConvDesc>;

using common_convbias_descriptors = std::tuple<cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor>;

common_convbias_descriptors
create_conv_add_bias_act_descriptors(int64_t* x_dim_padded,
                                     int64_t* padA,
                                     int64_t* convstrideA,
                                     int64_t* dilationA,
                                     int64_t* w_dim_padded,
                                     int64_t* y_dim_padded,
                                     cudnnDataType_t dataType) {
    (void)padA;
    (void)convstrideA;
    (void)dilationA;
    int64_t b_dim_padded[4];
    b_dim_padded[0] = y_dim_padded[0];
    b_dim_padded[1] = y_dim_padded[1];
    b_dim_padded[2] = 1;
    b_dim_padded[3] = 1;

    int64_t x_stride_padded[4];
    int64_t y_stride_padded[4];
    int64_t w_stride_padded[4];
    int64_t b_stride_padded[4];

    generateStrides(w_dim_padded, w_stride_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(x_dim_padded, x_stride_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(y_dim_padded, y_stride_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(b_dim_padded, b_stride_padded, 4, CUDNN_TENSOR_NCHW);

    return common_convbias_descriptors(cudnn_frontend::TensorBuilder()
                                           .setDim(4, x_dim_padded)
                                           .setStrides(4, x_stride_padded)
                                           .setId('x')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('y')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, w_dim_padded)
                                           .setStrides(4, w_stride_padded)
                                           .setId('w')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('z')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, b_dim_padded)
                                           .setStrides(4, b_stride_padded)
                                           .setId('b')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setVirtual()
                                           .setId('A')  // after add
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setVirtual()
                                           .setId('B')  // after bias
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('C')  // after conv
                                           .setAlignment(4)
                                           .setVirtual()
                                           .setDataType(dataType)
                                           .build());
}

common_conv_descriptors
create_common_descriptors(int64_t* x_dim_padded,
                          int64_t* padA,
                          int64_t* convstrideA,
                          int64_t* dilationA,
                          int64_t* w_dim_padded,
                          int64_t* y_dim_padded,
                          cudnnDataType_t dataType,
                          cudnnConvolutionMode_t mode) {
    const int convDim = 2;

    int64_t strideA_padded[4];
    int64_t outstrideA_padded[4];
    int64_t filterstrideA_padded[4];

    generateStrides(w_dim_padded, filterstrideA_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(x_dim_padded, strideA_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(y_dim_padded, outstrideA_padded, 4, CUDNN_TENSOR_NCHW);

    return common_conv_descriptors(cudnn_frontend::TensorBuilder()
                                       .setDim(4, x_dim_padded)
                                       .setStrides(4, strideA_padded)
                                       .setId('x')
                                       .setAlignment(4)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::TensorBuilder()
                                       .setDim(4, y_dim_padded)
                                       .setStrides(4, outstrideA_padded)
                                       .setId('y')
                                       .setAlignment(4)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::TensorBuilder()
                                       .setDim(4, w_dim_padded)
                                       .setStrides(4, filterstrideA_padded)
                                       .setId('w')
                                       .setAlignment(4)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::ConvDescBuilder()
                                       .setDataType(dataType)
                                       .setMathMode(mode)
                                       .setNDims(convDim)
                                       .setStrides(convDim, convstrideA)
                                       .setPrePadding(convDim, padA)
                                       .setPostPadding(convDim, padA)
                                       .setDilation(convDim, dilationA)
                                       .build());
}

cudnn_frontend::OperationGraph
create_operation_graph(common_conv_descriptors& descriptors, cudnnBackendDescriptorType_t mode, cudnnHandle_t handle_) {
    float alpha = 1.0f;
    float beta  = 0.0;

    auto op = cudnn_frontend::OperationBuilder(mode)
                  .setxDesc(std::get<X_TENSOR>(descriptors))
                  .setyDesc(std::get<Y_TENSOR>(descriptors))
                  .setwDesc(std::get<W_TENSOR>(descriptors))
                  .setcDesc(std::get<3>(descriptors))
                  .setAlpha(alpha)
                  .setBeta(beta)
                  .build();

    std::cout << "Operation is " << op.describe() << std::endl;

    std::array<cudnn_frontend::Operation const*, 1> ops = {&op};

    return cudnn_frontend::OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();
}

// Method for engine config generator based on heuristics
auto heurgen_method = [](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(opGraph)
                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                          .build();
    std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;

    auto &engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    cudnn_frontend::EngineConfigList filtered_configs;
    cudnn_frontend::filter(engine_configs, filtered_configs, ::allowAll);
    return filtered_configs;
};

// Method for engine config generator based on fallback list
auto fallback_method = [](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                        .setOperationGraph(opGraph)
                        .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                        .build();
    auto &fallback_list = fallback.getFallbackList();

    cudnn_frontend::EngineConfigList filtered_configs;
    // We create this filter to pre-remove configs being passed to cudnnFind.
    // This is just a sample and is not necessary
    cudnn_frontend::filter(fallback_list, filtered_configs, ::isNonDeterministic);

    return filtered_configs;
};

void
run_from_heuristics(int64_t* x_dim_padded,
                    int64_t* padA,
                    int64_t* convstrideA,
                    int64_t* dilationA,
                    int64_t* w_dim_padded,
                    int64_t* y_dim_padded,
                    cudnnDataType_t dataType,
                    cudnnConvolutionMode_t mode,
                    float* devPtrX,
                    float* devPtrW,
                    float* devPtrY,
                    cudnnBackendHeurMode_t heur_mode) {
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_conv_descriptors descriptors = create_common_descriptors(
            x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                              .setOperationGraph(opGraph)
                              .setHeurMode(heur_mode)
                              .build();

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto& engine_config = heuristics.getEngineConfig();

        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config[0], opGraph.getTag()).build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    if (handle_) cudnnDestroy(handle_);
    return;
}

void
run_from_global_index(int64_t* x_dim_padded,
                      int64_t* padA,
                      int64_t* convstrideA,
                      int64_t* dilationA,
                      int64_t* w_dim_padded,
                      int64_t* y_dim_padded,
                      cudnnDataType_t dataType,
                      cudnnConvolutionMode_t mode,
                      float* devPtrX,
                      float* devPtrW,
                      float* devPtrY) {
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_conv_descriptors descriptors = create_common_descriptors(
            x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph = create_operation_graph(
            descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        // We have to randomly pick one engine from [0, total_engines)
        // Selecting "0" by default
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }

        if (knobs.begin() != knobs.end()) {
            std::cout << "Updated knob choice" << std::endl;
            knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
            std::cout << knobs.begin()->describe() << std::endl;
        }
        auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
        std::cout << engine_config.describe() << std::endl;
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(nullptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    if (handle_) cudnnDestroy(handle_);
}

void
run_with_external_config(int64_t* x_dim_padded,
                         int64_t* padA,
                         int64_t* convstrideA,
                         int64_t* dilationA,
                         int64_t* w_dim_padded,
                         int64_t* y_dim_padded,
                         cudnnDataType_t dataType,
                         cudnnConvolutionMode_t mode,
                         float* devPtrX,
                         float* devPtrW,
                         float* devPtrY) {
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_conv_descriptors descriptors = create_common_descriptors(
            x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                              .setOperationGraph(opGraph)
                              .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                              .build();

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

        auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                            .setOperationGraph(opGraph)
                            .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                            .build();
        auto& fallback_list = fallback.getFallbackList();
        std::cout << "Fallback List has " << fallback_list.size() << " configurations " << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        cudnn_frontend::filter(engine_config, filtered_configs, ::isNonDeterministicOrisDownConverting);
        cudnn_frontend::filter(fallback_list, filtered_configs, ::isNonDeterministic);

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        std::cout << "Fallback List has " << fallback_list.size() << " configurations " << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan =
            cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        std::cout << plan.describe() << std::endl;
        void* data_ptrs[]   = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]      = {'x', 'y', 'w'};
        auto variantPack    = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(nullptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    if (handle_) cudnnDestroy(handle_);

    return;
}

// create_plan(std::vector<cudnnBackendDescriptor_t> &)
void
run_conv_add_bias_activation(int64_t* x_dim_padded,
                             int64_t* pad,
                             int64_t* convstride,
                             int64_t* dilation,
                             int64_t* w_dim_padded,
                             int64_t* y_dim_padded,
                             cudnnDataType_t dataType,
                             float* devPtrX,
                             float* devPtrW,
                             float* devPtrY,
                             float* devPtrZ,
                             float* devPtrB) {
    cudnnHandle_t handle_;
    try {
        int convDim = 2;
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        common_convbias_descriptors tensors = create_conv_add_bias_act_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, dataType);
        std::cout << std::get<X_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<Z_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<B_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERADD_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERBIAS_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERCONV_TENSOR>(tensors).describe() << std::endl;

        // Define the add operation
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // Define the bias operation
        auto addDesc2 = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << addDesc2.describe() << std::endl;

        // Define the activation operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(dataType)
                            .setMathMode(CUDNN_CONVOLUTION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha  = 1.0f;
        float alpha2 = 0.5f;
        float beta   = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(std::get<X_TENSOR>(tensors))
                           .setwDesc(std::get<W_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Add Node with scaling parameters.
        auto add_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(std::get<Z_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERADD_TENSOR>(tensors))
                           .setpwDesc(addDesc)
                           .setAlpha(alpha)
                           .setAlpha2(alpha2)
                           .build();
        std::cout << add_op1.describe() << std::endl;

        // Create a Bias Node.
        auto add_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op1.getOutputTensor())
                           .setbDesc(std::get<B_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERBIAS_TENSOR>(tensors))
                           .setpwDesc(addDesc2)
                           .build();
        std::cout << add_op2.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_op2.getOutputTensor())
                          .setyDesc(std::get<Y_TENSOR>(tensors))
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &add_op1, &add_op2, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // How many engines support this operation graph ?
        auto total_engines = opGraph.getEngineCount();
        std::cout << "conv_add_bias_activation " << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
        // We have to randomly pick one engine from [0, total_engines)
        // Selecting "0" by default
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }
        if (knobs.begin() != knobs.end()) {
            std::cout << "Updated knob choice" << std::endl;
            knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
            std::cout << knobs.begin()->describe() << std::endl;
        }

        // Create the requisite engine config
        auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
        std::cout << engine_config.describe() << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB};
        int64_t uids[]    = {'x', 'y', 'w', 'z', 'b'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(5, data_ptrs)
                               .setUids(5, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_from_cudnn_find(int64_t* x_dim_padded,
                    int64_t* padA,
                    int64_t* convstrideA,
                    int64_t* dilationA,
                    int64_t* w_dim_padded,
                    int64_t* y_dim_padded,
                    cudnnDataType_t dataType,
                    cudnnConvolutionMode_t mode,
                    float* devPtrX,
                    float* devPtrW,
                    float* devPtrY) {
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_conv_descriptors descriptors = create_common_descriptors(
            x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder().setDataPointers(3, data_ptrs).setUids(3, uids).build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        auto sample_predicate_function = [](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            return plan.getWorkspaceSize() != 0;
        };

        std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
        cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());

        auto options = generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            handle_, std::move(opGraph), variantPack, sample_predicate_function);

        std::for_each(options.begin(), options.end(), [](struct cudnn_frontend::executionOption& opt) {
            std::cout << "Plan tag: " << opt.plan.getTag() << " finished in " << opt.time_ms << " ms,"
                      << " workspace: " << opt.plan.getWorkspaceSize() << " bytes" << std::endl;
        });

        cudnnStatus_t status =
            cudnnBackendExecute(handle_, options.front().plan.get_raw_desc(), variantPack.get_raw_desc());

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");
    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    if (handle_) cudnnDestroy(handle_);
    return;
}

void
run_conv_add_bias_activation_with_cudnn_find(int64_t* x_dim_padded,
                                             int64_t* pad,
                                             int64_t* convstride,
                                             int64_t* dilation,
                                             int64_t* w_dim_padded,
                                             int64_t* y_dim_padded,
                                             cudnnDataType_t dataType,
                                             float* devPtrX,
                                             float* devPtrW,
                                             float* devPtrY,
                                             float* devPtrZ,
                                             float* devPtrB) {
    cudnnHandle_t handle_;
    try {
        int convDim = 2;
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        common_convbias_descriptors tensors = create_conv_add_bias_act_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, dataType);
        std::cout << std::get<X_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<Z_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<B_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERADD_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERBIAS_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERCONV_TENSOR>(tensors).describe() << std::endl;

        // Define the add operation
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // Define the bias operation
        auto addDesc2 = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << addDesc2.describe() << std::endl;

        // Define the activation operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(dataType)
                            .setMathMode(CUDNN_CONVOLUTION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha  = 1.0f;
        float alpha2 = 0.5f;
        float beta   = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(std::get<X_TENSOR>(tensors))
                           .setwDesc(std::get<W_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Add Node with scaling parameters.
        auto add_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(std::get<Z_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERADD_TENSOR>(tensors))
                           .setpwDesc(addDesc)
                           .setAlpha(alpha)
                           .setAlpha2(alpha2)
                           .build();
        std::cout << add_op1.describe() << std::endl;

        // Create a Bias Node.
        auto add_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op1.getOutputTensor())
                           .setbDesc(std::get<B_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERBIAS_TENSOR>(tensors))
                           .setpwDesc(addDesc2)
                           .build();
        std::cout << add_op2.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_op2.getOutputTensor())
                          .setyDesc(std::get<Y_TENSOR>(tensors))
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &add_op1, &add_op2, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto max_workspace_size = 10 * 1024 * 1024;  // 10 MiB
        void* workspace_ptr = nullptr;
        checkCudaErr(cudaMalloc(&workspace_ptr, max_workspace_size));

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB};
        int64_t uids[]    = {'x', 'y', 'w', 'z', 'b'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(5, data_ptrs)
                               .setUids(5, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        auto sample_predicate_function = [=](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            return plan.getWorkspaceSize() > max_workspace_size;
        };

        std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
        cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());

        auto options = generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            handle_, std::move(opGraph), variantPack, sample_predicate_function);

        std::for_each(options.begin(), options.end(), [](struct cudnn_frontend::executionOption& opt) {
            std::cout << "Plan tag: " << opt.plan.getTag() << " finished in " << opt.time_ms << " ms,"
                      << " workspace: " << opt.plan.getWorkspaceSize() << " bytes" << std::endl;
        });

        cudnnStatus_t status =
            cudnnBackendExecute(handle_, options.front().plan.get_raw_desc(), variantPack.get_raw_desc());

        checkCudaErr(cudaFree(workspace_ptr));
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_from_cudnn_get(int64_t* x_dim_padded,
                   int64_t* padA,
                   int64_t* convstrideA,
                   int64_t* dilationA,
                   int64_t* w_dim_padded,
                   int64_t* y_dim_padded,
                   cudnnDataType_t dataType,
                   cudnnConvolutionMode_t mode,
                   float* devPtrX,
                   float* devPtrW,
                   float* devPtrY) {
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_conv_descriptors descriptors = create_common_descriptors(
            x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder().setDataPointers(3, data_ptrs).setUids(3, uids).build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        auto sample_predicate_function = [](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            return plan.getWorkspaceSize() != 0;
        };

        std::array<cudnn_frontend::GeneratorSource const, 1> sources = {heurgen_method};
        cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());

        auto plans = generator.cudnnGetPlan(handle_, std::move(opGraph), sample_predicate_function);

        std::for_each(plans.begin(), plans.end(), [](cudnn_frontend::ExecutionPlan& plan) {
            std::cout << "Plan tag: " << plan.getTag() << " workspace: " << plan.getWorkspaceSize() << " bytes"
                      << std::endl;
        });

        cudnnStatus_t status = cudnnBackendExecute(handle_, plans.front().get_raw_desc(), variantPack.get_raw_desc());

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");
    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    if (handle_) cudnnDestroy(handle_);
    return;
}

void
block_using_errata(int64_t* x_dim_padded,
                   int64_t* padA,
                   int64_t* convstrideA,
                   int64_t* dilationA,
                   int64_t* w_dim_padded,
                   int64_t* y_dim_padded,
                   cudnnDataType_t dataType,
                   cudnnConvolutionMode_t mode,
                   float* devPtrX,
                   float* devPtrW,
                   float* devPtrY) {
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_conv_descriptors descriptors = create_common_descriptors(
            x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        (void)devPtrX;
        (void)devPtrY;
        (void)devPtrW;

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph = create_operation_graph(
            descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        // We have to randomly pick one engine from [0, total_engines)
        // Selecting "0" by default
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }

        if (knobs.begin() != knobs.end()) {
            std::cout << "Updated knob choice" << std::endl;
            knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
            std::cout << knobs.begin()->describe() << std::endl;
        }
        auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
        std::cout << engine_config.describe() << std::endl;
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        /// Please note that the json string mentioned below is just an example and is
        /// not actually a buggy engine config (kernel).
        auto json_handle = json::parse(R"(
            { "version" : 1, 
              "rules"   : 
                [ 
                    { "rule_id"             : "ConvBwdData_eng1_k2=2_k3=0", 
                      "operation"           : "ConvBwdData",
                      "engine"              : "eng1", 
                      "knob"                : ["k2=4", "k3=0"],
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : -1 
                    }, 
                    { "rule_id"             : "ConvBwdFilter_eng0",
                      "operation"           : "ConvBwdFilter",
                      "engine"              : "eng0", 
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : -1 
                    } 
                ] 
            })");

        auto fn = std::bind(::allowErrata, padA);
        bool is_plan_blocked = cudnn_frontend::check_errata<decltype(fn)>(json_handle, plan.getTag(), handle_, fn);
        CHECK(is_plan_blocked);

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    if (handle_) cudnnDestroy(handle_);
}
