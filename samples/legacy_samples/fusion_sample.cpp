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

#include "fusion_sample.h"
#include <cudnn_frontend.h>
#include "../utils/error_util.h"

bool
allowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

#if (CUDNN_VERSION >= 8200)
bool
isRuntimeCompilation(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION>(engine_config);
}
#endif

cudnn_frontend::ExecutionPlan
get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_) {
#if (CUDNN_VERSION >= 8200)
    {
        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                              .setOperationGraph(opGraph)
                              .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                              .build();

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

        // Try engine configs returned by the heuristics and pick up the first one that works.
        for (auto& ecfg : engine_config) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle_)
                                .setEngineConfig(ecfg, opGraph.getTag())
                                .build();
                return plan;
            } catch (cudnn_frontend::cudnnException& e) {
                continue;
            }
        }
    }
#endif

    {
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<1>(
            {"heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (auto status : statuses) {
            std::cout << cudnn_frontend::to_string(status) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        return cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle_)
            .setEngineConfig(filtered_configs[0], opGraph.getTag())
            .build();
    }
}

void
run_conv_scale_bias_add_leaky_relu(int64_t* x_dim,
                                   int64_t* w_dim,
                                   int64_t* y_dim,
                                   int64_t* s_dim,
                                   int64_t* b_dim,
                                   int64_t* a_dim,
                                   cudnnDataType_t dataType,
                                   int convDim,
                                   int64_t* conv_padA,
                                   int64_t* conv_dilationA,
                                   int64_t* conv_strideA,
                                   void* devPtrX,
                                   void* devPtrW,
                                   void* devPtrY,
                                   void* devPtrS,
                                   void* devPtrB,
                                   void* devPtrA) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(a_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto aTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, a_dim)
                           .setStride(4, stride)
                           .setId('a')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterAddTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, y_dim)
                                  .setStride(4, stride)
                                  .setId('D')  // after add
                                  .setAlignment(16)
                                  .setVirtual()
                                  .setDataType(dataType)
                                  .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << aTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterAddTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the add descriptor
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .setReluLowerClipSlope(0.01)  // leaky relu
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(conv_op.getOutputTensor())
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(scale_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create a Add Node.
        auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setbDesc(aTensor)
                          .setyDesc(afterAddTensor)
                          .setpwDesc(addDesc)
                          .build();
        std::cout << add_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution scale bias add activation
        std::array<cudnn_frontend::Operation const*, 5> ops = {&conv_op, &scale_op, &bias_op, &add_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrS, devPtrB, devPtrA};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'b', 'a'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(6, data_ptrs)
                               .setUids(6, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere cards
        if (prop.major < 8 && e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
        } else {
#if (CUDNN_VERSION == 8600) || (CUDNN_VERSION == 8700)
            if (prop.major == 9) {
                std::cout << "Hopper GPUs does not have float fused operations support yet\n";
                return;
            }
#endif
#if (CUDNN_VERSION >= 8300)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif
        }
    }
}

void
run_conv_bias_scale_relu(int64_t* x_dim,
                         int64_t* w_dim,
                         int64_t* y_dim,
                         int64_t* b_dim,
                         int64_t* s_dim,
                         cudnnDataType_t dataType,
                         int convDim,
                         int64_t* conv_padA,
                         int64_t* conv_dilationA,
                         int64_t* conv_strideA,
                         void* devPtrX,
                         void* devPtrW,
                         void* devPtrY,
                         void* devPtrB,
                         void* devPtrS) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('C')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(bias_op.getOutputTensor())
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(scale_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &bias_op, &scale_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrB, devPtrS};
        int64_t uids[]    = {'x', 'y', 'w', 'b', 's'};
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

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        // this example is only for Ampere cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
            CHECK(false);
#endif
        }
    }
}

void
run_serialization_conv_bias_scale_relu(int64_t* x_dim,
                                       int64_t* w_dim,
                                       int64_t* y_dim,
                                       int64_t* b_dim,
                                       int64_t* s_dim,
                                       cudnnDataType_t dataType,
                                       int convDim,
                                       int64_t* conv_padA,
                                       int64_t* conv_dilationA,
                                       int64_t* conv_strideA,
                                       void* devPtrX,
                                       void* devPtrW,
                                       void* devPtrY,
                                       void* devPtrB,
                                       void* devPtrS) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('C')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(bias_op.getOutputTensor())
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(scale_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &bias_op, &scale_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        std::string plan_json;
        {
            // Suppose this is how execution plans are normally created
            auto plan_tmp = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);
            // Generate a JSON serialization of the execution plan
            plan_json = plan_tmp.getJsonRepresentation();
            // Optionally save to a file, etc...
            // std::ofstream output_file("execution_plan.json");
            // output_file << plan_json;
            // The temporary execution plan can now be discarded.
        }
        // Load the plan from a JSON string.
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).loadFromJson(plan_json);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrB, devPtrS};
        int64_t uids[]    = {'x', 'y', 'w', 'b', 's'};
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

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        // this example is only for Ampere cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
#if (CUDNN_VERSION >= 8400)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif
        }
    }
}

void
run_conv_scale_bias_relu_gen_index_selection(int64_t* x_dim,
                                             int64_t* w_dim,
                                             int64_t* y_dim,
                                             int64_t* s_dim,
                                             int64_t* b_dim,
                                             int64_t* threshold_dim,
                                             cudnnDataType_t dataType,
                                             int convDim,
                                             int64_t* conv_padA,
                                             int64_t* conv_dilationA,
                                             int64_t* conv_strideA,
                                             int axis,
                                             void* devPtrX,
                                             void* devPtrW,
                                             void* devPtrY,
                                             void* devPtrS,
                                             void* devPtrB,
                                             void* devPtrTopThreshold,
                                             void* devPtrBottomThreshold) {
    cudnnHandle_t handle_;
    (void)handle_;
    (void)x_dim;
    (void)w_dim;
    (void)y_dim;
    (void)s_dim;
    (void)b_dim;
    (void)threshold_dim;
    (void)dataType;
    (void)convDim;
    (void)conv_padA;
    (void)conv_dilationA;
    (void)conv_strideA;
    (void)axis;
    (void)devPtrX;
    (void)devPtrW;
    (void)devPtrY;
    (void)devPtrS;
    (void)devPtrB;
    (void)devPtrTopThreshold;
    (void)devPtrBottomThreshold;
    try {
#if (CUDNN_VERSION >= 8400)
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("turing") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_ARCH_MISMATCH,
                "run_conv_scale_bias_relu_gen_index_selection: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(CUDNN_DATA_FLOAT)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();

        auto afterActivationTensor = cudnn_frontend::TensorBuilder()
                                         .setDim(4, y_dim)
                                         .setStride(4, stride)
                                         .setId('D')  // after activation
                                         .setAlignment(16)
                                         .setVirtual()
                                         .setDataType(CUDNN_DATA_FLOAT)
                                         .build();

        auto genIndexTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, y_dim)
                                  .setStride(4, stride)
                                  .setId('I')  // output of the gen index operation
                                  .setAlignment(16)
                                  .setVirtual()
                                  .setDataType(CUDNN_DATA_INT32)
                                  .build();

        auto maskTopTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(4, y_dim)
                                 .setStride(4, stride)
                                 .setId('m')  // top half of the mask created after the less than
                                 .setAlignment(16)
                                 .setVirtual()
                                 .setDataType(CUDNN_DATA_BOOLEAN)
                                 .build();

        auto maskBottomTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('n')  // bottom half of the mask
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(CUDNN_DATA_BOOLEAN)
                                    .build();

        auto maskTensor = cudnn_frontend::TensorBuilder()
                              .setDim(4, y_dim)
                              .setStride(4, stride)
                              .setId('M')  // OR of the top and bottom masks
                              .setAlignment(16)
                              .setVirtual()
                              .setDataType(CUDNN_DATA_BOOLEAN)
                              .build();

        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(threshold_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto thresholdTopTensor = cudnn_frontend::TensorBuilder()
                                      .setDim(4, threshold_dim)
                                      .setStride(4, stride)
                                      .setId('t')  // threshold for creating the top mask
                                      .setAlignment(16)
                                      .setDataType(CUDNN_DATA_INT32)
                                      .build();

        auto thresholdBottomTensor = cudnn_frontend::TensorBuilder()
                                         .setDim(4, threshold_dim)
                                         .setStride(4, stride)
                                         .setId('u')  // threshold for creating the bottom mask
                                         .setAlignment(16)
                                         .setDataType(CUDNN_DATA_INT32)
                                         .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << afterActivationTensor.describe() << std::endl;
        std::cout << genIndexTensor.describe() << std::endl;
        std::cout << maskTopTensor.describe() << std::endl;
        std::cout << maskBottomTensor.describe() << std::endl;
        std::cout << maskTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;
        std::cout << thresholdTopTensor.describe() << std::endl;
        std::cout << thresholdBottomTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the genIndex descriptor
        auto genIndexDesc = cudnn_frontend::PointWiseDescBuilder()
                                .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setAxis(axis)
                                .build();
        std::cout << genIndexDesc.describe() << std::endl;

        // Define the lessThan descriptor
        auto lessThanDesc = cudnn_frontend::PointWiseDescBuilder()
                                .setMode(CUDNN_POINTWISE_CMP_LT)
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .build();
        std::cout << lessThanDesc.describe() << std::endl;

        // Define the greaterThan descriptor
        auto greaterThanDesc = cudnn_frontend::PointWiseDescBuilder()
                                   .setMode(CUDNN_POINTWISE_CMP_GT)
                                   .setComputeType(CUDNN_DATA_FLOAT)
                                   .build();
        std::cout << greaterThanDesc.describe() << std::endl;

        // Define the logical_or descriptor
        auto logicalOrDesc = cudnn_frontend::PointWiseDescBuilder()
                                 .setMode(CUDNN_POINTWISE_LOGICAL_OR)
                                 .setComputeType(CUDNN_DATA_BOOLEAN)
                                 .build();
        std::cout << logicalOrDesc.describe() << std::endl;

        // Define the binary_selection descriptor
        auto selectionDesc = cudnn_frontend::PointWiseDescBuilder()
                                 .setMode(CUDNN_POINTWISE_BINARY_SELECT)
                                 .setComputeType(CUDNN_DATA_FLOAT)
                                 .build();
        std::cout << selectionDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(afterConvTensor)
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(afterScaleTensor)
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(afterBiasTensor)
                          .setyDesc(afterActivationTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create a Gen_Index Node.
        auto genIndex_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(afterActivationTensor)
                               .setyDesc(genIndexTensor)
                               .setpwDesc(genIndexDesc)
                               .build();
        std::cout << genIndex_op.describe() << std::endl;

        // Create a LessThan Node.
        auto lessThan_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(genIndexTensor)
                               .setbDesc(thresholdTopTensor)
                               .setyDesc(maskTopTensor)
                               .setpwDesc(lessThanDesc)
                               .build();
        std::cout << lessThan_op.describe() << std::endl;

        // Create a GreaterThan Node.
        auto greaterThan_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                  .setxDesc(genIndexTensor)
                                  .setbDesc(thresholdBottomTensor)
                                  .setyDesc(maskBottomTensor)
                                  .setpwDesc(greaterThanDesc)
                                  .build();
        std::cout << greaterThan_op.describe() << std::endl;

        // Create a LogicalOr Node.
        auto logicalOr_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                .setxDesc(maskTopTensor)
                                .setbDesc(maskBottomTensor)
                                .setyDesc(maskTensor)
                                .setpwDesc(logicalOrDesc)
                                .build();
        std::cout << logicalOr_op.describe() << std::endl;

        // Create a Binary_Selection Node.
        auto selection_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                .setxDesc(afterConvTensor)
                                .setbDesc(afterActivationTensor)
                                .settDesc(maskTensor)
                                .setyDesc(yTensor)
                                .setpwDesc(selectionDesc)
                                .build();
        std::cout << selection_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 9> ops = {&conv_op,
                                                               &scale_op,
                                                               &bias_op,
                                                               &act_op,
                                                               &genIndex_op,
                                                               &lessThan_op,
                                                               &greaterThan_op,
                                                               &logicalOr_op,
                                                               &selection_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // How many engines support this operation graph ?
        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrS, devPtrB, devPtrTopThreshold, devPtrBottomThreshold};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'b', 't', 'u'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
#endif

    } catch (cudnn_frontend::cudnnException& e) {
        if (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH) {
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_conv_scale_bias_relu_int8(int64_t* x_dim,
                              int64_t* w_dim,
                              int64_t* y_dim,
                              int64_t* s_dim,
                              int64_t* b_dim,
                              int convDim,
                              int64_t* conv_padA,
                              int64_t* conv_dilationA,
                              int64_t* conv_strideA,
                              void* devPtrX,
                              void* devPtrW,
                              void* devPtrY,
                              void* devPtrS,
                              void* devPtrB) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        if (check_device_arch_newer_than("turing") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_ARCH_MISMATCH,
                "run_conv_scale_bias_relu_int8: Sample requires Turing or above GPU");
        }

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(CUDNN_DATA_INT8)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_INT8)
                           .build();
        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_INT32)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(CUDNN_DATA_FLOAT)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_INT8)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_INT32)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(conv_op.getOutputTensor())
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(scale_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &scale_op, &bias_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // How many engines support this operation graph ?
        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrS, devPtrB};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'b'};
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

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        // this example is only for Turing and later cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Turing GPUs" << std::endl;
        } else {
#if (CUDNN_VERSION == 8600)
            if (prop.major == 9) {
                std::cout << "Hopper GPUs does not have int8 fused operations support yet\n";
                return;
            }
#endif
#if (CUDNN_VERSION >= 8300)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif
        }
    }
}

void
run_pool_scale_bias_relu_int8(int64_t* x_dim,
                              int64_t* y_dim,
                              int64_t* s_dim,
                              int64_t* b_dim,
                              void* devPtrX,
                              void* devPtrY,
                              void* devPtrS,
                              void* devPtrB,
                              cudnnDataType_t compType,
                              cudnnNanPropagation_t const nanOpt,
                              cudnn_frontend::ResampleMode_t const mode,
                              cudnn_frontend::PaddingMode_t const padding_mode,
                              int64_t nbSpatialDims,
                              double alpha,
                              double beta,
                              int64_t* windowDimA,
                              int64_t* prePaddingA,
                              int64_t* postPaddingA,
                              int64_t* strideA) {
    cudnnHandle_t handle_;
    (void)nbSpatialDims;
    (void)alpha;
    (void)beta;
    (void)windowDimA;
    (void)prePaddingA;
    (void)postPaddingA;
    (void)strideA;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t strideTensor[4];
        generateStrides(x_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, strideTensor)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(CUDNN_DATA_INT8)
                           .build();
        generateStrides(s_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, strideTensor)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(b_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, strideTensor)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(y_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto afterPoolTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, strideTensor)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(compType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, strideTensor)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(compType)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, strideTensor)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(compType)
                                   .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, strideTensor)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_INT8)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterPoolTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setNanPropagation(nanOpt)
                            .setResampleMode(mode)
                            .setPaddingMode(padding_mode)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setSpatialStride(nbSpatialDims, strideA)
                            .setPrePadding(nbSpatialDims, prePaddingA)
                            .setPostPadding(nbSpatialDims, postPaddingA)
                            .build();
        std::cout << "Initialized Pool Desc" << std::endl;
        std::cout << poolDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc =
            cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_MUL).setComputeType(compType).build();
        std::cout << "Initialized Scale Desc" << std::endl;
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc =
            cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_ADD).setComputeType(compType).build();
        std::cout << "Initialized Bias Desc" << std::endl;
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc =
            cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_RELU_FWD).setComputeType(compType).build();
        std::cout << "Initialized Activation Desc" << std::endl;
        std::cout << actDesc.describe() << std::endl;

#if (CUDNN_VERSION >= 8500)
        // Create a Resample Node
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setyDesc(afterPoolTensor)
                           .setResampleDesc(poolDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << pool_op.describe() << std::endl;
#endif
        // Create a Multiplication Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
#if (CUDNN_VERSION >= 8500)
                            .setxDesc(pool_op.getOutputTensor())
#endif
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(scale_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

#if (CUDNN_VERSION >= 8500)
        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&pool_op, &scale_op, &bias_op, &act_op};
#else
        std::array<cudnn_frontend::Operation const*, 3> ops = {&scale_op, &bias_op, &act_op};
#endif
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // How many engines support this operation graph ?
        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        // Create the variant pack and associate with the data pointers
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrS, devPtrB};
        int64_t uids[]    = {'x', 'y', 's', 'b'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // Trigger the execute operation
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        checkCudnnErr(cudnnDestroy(handle_));
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        std::cout << "EXECUTE SUCCESS" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Sample not executed for cuDNN version " << CUDNN_VERSION << std::endl;
        // this example is only for Ampere cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
#if (CUDNN_VERSION >= 8500)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif
        }
    }
}

void
run_matmul_bias_gelu(int64_t* a_dim,
                     int64_t* b_dim,
                     int64_t* c_dim,
                     int64_t* z_dim,
                     cudnnDataType_t dataType,
                     void* devPtrA,
                     void* devPtrB,
                     void* devPtrC,
                     void* devPtrZ,
                     void* devPtrAfterZ) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("ampere") == false && dataType == CUDNN_DATA_FLOAT) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, CUDNN_STATUS_ARCH_MISMATCH, "run_matmul_bias_gelu: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[3];
        // the intension is to compute stride for a [1, M, K] matrix with K in the inner most dimension, and
        // CUDNN_TENSOR_NCHW is a borrowed notation
        generateStrides(a_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto aMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, a_dim)
                                 .setStride(3, stride)
                                 .setId('a')
                                 .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                 .setDataType(dataType)
                                 .build();
        generateStrides(b_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto bMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, b_dim)
                                 .setStride(3, stride)
                                 .setId('b')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        generateStrides(z_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto biasTensor = cudnn_frontend::TensorBuilder()
                              .setDim(3, z_dim)
                              .setStride(3, stride)
                              .setId('z')
                              .setAlignment(16)
                              .setDataType(dataType)
                              .build();

        generateStrides(c_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto afterMatMulTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, c_dim)
                                     .setStride(3, stride)
                                     .setId('A')  // after matmul
                                     .setAlignment(16)
                                     .setVirtual()
                                     .setDataType(dataType)
                                     .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(3, c_dim)
                                   .setStride(3, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setDataType(dataType)
                                   .build();
        auto outputTensor = cudnn_frontend::TensorBuilder()
                                .setDim(3, c_dim)
                                .setStride(3, stride)
                                .setId('c')  // output after gelu
                                .setAlignment(16)
                                .setDataType(dataType)
                                .build();

        std::cout << aMatrixTensor.describe() << std::endl;
        std::cout << bMatrixTensor.describe() << std::endl;
        std::cout << biasTensor.describe() << std::endl;
        std::cout << afterMatMulTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << outputTensor.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
#if (CUDNN_VERSION >= 8500)
                           .setMode(CUDNN_POINTWISE_GELU_APPROX_TANH_FWD)
#else
                           .setMode(CUDNN_POINTWISE_GELU_FWD)
#endif
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the matmul desc
        auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
        std::cout << matmulDesc.describe() << std::endl;

        // Create a matmul Node
        auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                             .setaMatDesc(aMatrixTensor)
                             .setbMatDesc(bMatrixTensor)
                             .setcMatDesc(afterMatMulTensor)
                             .setmatmulDesc(matmulDesc)
                             .build();
        std::cout << matmul_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(matmul_op.getOutputTensor())
                           .setbDesc(biasTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setyDesc(outputTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is matmul bias activation
        std::array<cudnn_frontend::Operation const*, 3> ops = {&matmul_op, &bias_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrA, devPtrB, devPtrC, devPtrZ, devPtrAfterZ};
        int64_t uids[]    = {'a', 'b', 'c', 'z', 'B'};
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

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED || e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH)) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
            CHECK(false);
#endif
        }
    }
}

void
run_conv_drelu(int64_t* x_dim,
               int64_t* pad,
               int64_t* convstride,
               int64_t* dilation,
               int64_t* w_dim,
               int64_t* y_dim,
               cudnnDataType_t dataType,
               void* dev_ptr_x,
               void* dev_ptr_w,
               void* dev_ptr_y,
               void* dev_ptr_bwd_act_x) {
    cudnnHandle_t handle_;
    try {
        int convDim = 2;

        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, CUDNN_STATUS_ARCH_MISMATCH, "run_conv_drelu: Sample requires Ampere or above GPU");
        }
        int64_t x_id         = 101;
        int64_t w_id         = 102;
        int64_t bwd_act_x_id = 201;
        int64_t y_id         = 301;

        int64_t after_conv_id = 1001;

        int64_t x_stride_padded[4];
        int64_t y_stride_padded[4];
        int64_t w_stride_padded[4];

        generateStrides(w_dim, w_stride_padded, 4, CUDNN_TENSOR_NHWC);
        generateStrides(x_dim, x_stride_padded, 4, CUDNN_TENSOR_NHWC);
        generateStrides(y_dim, y_stride_padded, 4, CUDNN_TENSOR_NHWC);

        auto x_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, x_dim)
                            .setStride(4, x_stride_padded)
                            .setId(x_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto w_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, w_dim)
                            .setStride(4, w_stride_padded)
                            .setId(w_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto after_conv_tensor = cudnn_frontend::TensorBuilder()
                                     .setDim(4, y_dim)
                                     .setStride(4, y_stride_padded)
                                     .setId(after_conv_id)
                                     .setAlignment(4)
                                     .setVirtual()
                                     .setDataType(CUDNN_DATA_FLOAT)
                                     .build();

        auto bwd_act_x_tensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, y_stride_padded)
                                    .setId(bwd_act_x_id)
                                    .setAlignment(4)
                                    .setDataType(dataType)
                                    .build();

        auto after_activation_tensor = cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim)
                                           .setStride(4, y_stride_padded)
                                           .setId(y_id)
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build();

        std::cout << x_tensor.describe() << std::endl;
        std::cout << w_tensor.describe() << std::endl;
        std::cout << after_conv_tensor.describe() << std::endl;
        std::cout << bwd_act_x_tensor.describe() << std::endl;
        std::cout << after_activation_tensor.describe() << std::endl;

        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(x_tensor)
                           .setwDesc(w_tensor)
                           .setyDesc(after_conv_tensor)
                           .setcDesc(convDesc)
                           .setAlpha(1.0f)
                           .setBeta(0.0f)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_BWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setdyDesc(after_conv_tensor)
                          .setxDesc(bwd_act_x_tensor)
                          .setdxDesc(after_activation_tensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {dev_ptr_x, dev_ptr_w, dev_ptr_bwd_act_x, dev_ptr_y};
        int64_t uids[]    = {x_id, w_id, bwd_act_x_id, y_id};

        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        if (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH) {
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;

        CHECK(false);
    }
}

void
run_dgrad_drelu(int64_t* dx_dim,
                int64_t* pad,
                int64_t* convstride,
                int64_t* dilation,
                int64_t* w_dim,
                int64_t* dy_dim,
                cudnnDataType_t dataType,
                void* dev_ptr_dx,
                void* dev_ptr_w,
                void* dev_ptr_dy,
                void* dev_ptr_bwd_act_x) {
    cudnnHandle_t handle_;
    try {
        int convDim = 2;
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, CUDNN_STATUS_ARCH_MISMATCH, "run_dgrad_drelu: Sample requires Ampere or above GPU");
        }
        checkCudnnErr(cudnnCreate(&handle_));

        int64_t dx_id        = 101;
        int64_t w_id         = 102;
        int64_t bwd_act_x_id = 201;
        int64_t dy_id        = 301;

        int64_t after_dgrad_id = 1001;

        int64_t dx_stride[4];
        int64_t dy_stride[4];
        int64_t w_stride[4];

        generateStrides(w_dim, w_stride, 4, CUDNN_TENSOR_NHWC);
        generateStrides(dx_dim, dx_stride, 4, CUDNN_TENSOR_NHWC);
        generateStrides(dy_dim, dy_stride, 4, CUDNN_TENSOR_NHWC);

        auto after_dgrad_dx_tensor = cudnn_frontend::TensorBuilder()
                                         .setDim(4, dx_dim)
                                         .setStride(4, dx_stride)
                                         .setId(after_dgrad_id)
                                         .setAlignment(4)
                                         .setVirtual()
                                         .setDataType(dataType)
                                         .build();

        auto w_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, w_dim)
                            .setStride(4, w_stride)
                            .setId(w_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto dy_tensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, dy_dim)
                             .setStride(4, dy_stride)
                             .setId(dy_id)
                             .setAlignment(4)
                             .setDataType(dataType)
                             .build();

        auto bwd_act_x_tensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, dx_dim)
                                    .setStride(4, dx_stride)
                                    .setId(bwd_act_x_id)
                                    .setAlignment(4)
                                    .setDataType(dataType)
                                    .build();

        auto after_bwd_activation_dx_tensor = cudnn_frontend::TensorBuilder()
                                                  .setDim(4, dx_dim)
                                                  .setStride(4, dx_stride)
                                                  .setId(dx_id)
                                                  .setAlignment(4)
                                                  .setDataType(dataType)
                                                  .build();

        std::cout << after_dgrad_dx_tensor.describe() << std::endl;
        std::cout << w_tensor.describe() << std::endl;
        std::cout << dy_tensor.describe() << std::endl;
        std::cout << bwd_act_x_tensor.describe() << std::endl;
        std::cout << after_bwd_activation_dx_tensor.describe() << std::endl;

        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                           .setdyDesc(dy_tensor)
                           //    .setyDesc(dy_tensor)
                           .setwDesc(w_tensor)
                           .setdxDesc(after_dgrad_dx_tensor)
                           //    .setxDesc(after_dgrad_dx_tensor)
                           .setcDesc(convDesc)
                           .setAlpha(1.0f)
                           .setBeta(0.0f)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_BWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setdyDesc(after_dgrad_dx_tensor)
                          .setxDesc(bwd_act_x_tensor)
                          .setdxDesc(after_bwd_activation_dx_tensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {dev_ptr_dx, dev_ptr_w, dev_ptr_bwd_act_x, dev_ptr_dy};
        int64_t uids[]    = {dx_id, w_id, bwd_act_x_id, dy_id};

        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        if (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH) {
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_matmul_dgelu_dbias(const int64_t* dy_dim,
                       const int64_t* w_dim,
                       const int64_t* dx_dim,
                       const int64_t* dbias_dim,
                       cudnnDataType_t dataType,
                       void* dev_ptr_dy,
                       void* dev_ptr_w,
                       void* dev_ptr_bwd_act_x,
                       void* dev_ptr_dx,
                       void* dev_ptr_dbias) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, CUDNN_STATUS_ARCH_MISMATCH, "run_matmul_dgelu_dbias: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[3];

        // Use the following UIDs for tensors
        int64_t dy_uid        = 101;
        int64_t w_uid         = 102;
        int64_t bwd_act_x_uid = 103;
        int64_t dx_uid        = 104;
        int64_t dbias_uid     = 105;

        // Create tensor descriptor for DY matrix
        generateStrides(dy_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto dyMatrixTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(3, dy_dim)
                                  .setStride(3, stride)
                                  .setId(dy_uid)
                                  .setAlignment(16)
                                  .setDataType(dataType)
                                  .build();
        std::cout << dyMatrixTensor.describe() << std::endl;

        // Create tensor descriptor for weight matrix
        generateStrides(w_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto wMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, w_dim)
                                 .setStride(3, stride)
                                 .setId(w_uid)
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();
        std::cout << wMatrixTensor.describe() << std::endl;

        // Create tensor descriptor for dx matrix
        generateStrides(dx_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto dataGrad1MatrixTensor = cudnn_frontend::TensorBuilder()
                                         .setDim(3, dx_dim)
                                         .setStride(3, stride)
                                         .setId('X')
                                         .setAlignment(16)
                                         .setDataType(dataType)
                                         .setVirtual(true)
                                         .build();
        std::cout << dataGrad1MatrixTensor.describe() << std::endl;

        // Create tensor descriptor for geluInput matrix
        generateStrides(dx_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto geluInputMatrixTensor = cudnn_frontend::TensorBuilder()
                                         .setDim(3, dx_dim)
                                         .setStride(3, stride)
                                         .setId(bwd_act_x_uid)
                                         .setAlignment(16)
                                         .setDataType(dataType)
                                         .build();
        std::cout << geluInputMatrixTensor.describe() << std::endl;

        // Create tensor descriptor for output of backwardGelu matrix
        generateStrides(dx_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto backwardGeluMatrixTensor = cudnn_frontend::TensorBuilder()
                                            .setDim(3, dx_dim)
                                            .setStride(3, stride)
                                            .setId(dx_uid)
                                            .setAlignment(16)
                                            .setDataType(dataType)
                                            .build();
        std::cout << backwardGeluMatrixTensor.describe() << std::endl;

        // Create tensor descriptor for output of biasGrad(reduction) matrix
        generateStrides(dbias_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto backwardBiasMatrixTensor = cudnn_frontend::TensorBuilder()
                                            .setDim(3, dbias_dim)
                                            .setStride(3, stride)
                                            .setId(dbias_uid)
                                            .setAlignment(16)
                                            .setDataType(CUDNN_DATA_FLOAT)
                                            .build();
        std::cout << backwardBiasMatrixTensor.describe() << std::endl;

        auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
        std::cout << matmulDesc.describe() << std::endl;

        auto geluDesc = cudnn_frontend::PointWiseDescBuilder()
#if (CUDNN_VERSION >= 8500)
                            .setMode(CUDNN_POINTWISE_GELU_APPROX_TANH_BWD)
#else
                            .setMode(CUDNN_POINTWISE_GELU_BWD)
#endif
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << geluDesc.describe() << std::endl;

        // Define the reduction descriptor
        auto reductionDesc = cudnn_frontend::ReductionDescBuilder()
                                 .setComputeType(CUDNN_DATA_FLOAT)
                                 .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                 .build();
        std::cout << reductionDesc.describe() << std::endl;

        // Create a matmul Node for Dgrad
        auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dyMatrixTensor)
                            .setbMatDesc(wMatrixTensor)
                            .setcMatDesc(dataGrad1MatrixTensor)
                            .setmatmulDesc(matmulDesc)
                            .build();
        std::cout << matmulOp.describe() << std::endl;

        // Create a matmul Node for dGeLU
        auto geluOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setdyDesc(matmulOp.getOutputTensor())
                          .setxDesc(geluInputMatrixTensor)
                          .setdxDesc(backwardGeluMatrixTensor)
                          .setpwDesc(geluDesc)
                          .build();
        std::cout << geluOp.describe() << std::endl;

        // Create a reduction add Node.
        auto reduction_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(backwardGeluMatrixTensor)
                                .setyDesc(backwardBiasMatrixTensor)
                                .setreductionDesc(reductionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph.
        std::array<cudnn_frontend::Operation const*, 3> ops = {&matmulOp, &geluOp, &reduction_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {dev_ptr_dy, dev_ptr_w, dev_ptr_dx, dev_ptr_bwd_act_x, dev_ptr_dbias};
        int64_t uids[]    = {dy_uid, w_uid, dx_uid, bwd_act_x_uid, dbias_uid};
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

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED || e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH)) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }
}

void
run_conv_reduction(int64_t* x_dim,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   int64_t* r_dim,
                   cudnnDataType_t dataType,
                   int convDim,
                   int64_t* conv_padA,
                   int64_t* conv_dilationA,
                   int64_t* conv_strideA,
                   void* devPtrX,
                   void* devPtrW,
                   void* devPtrR) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, CUDNN_STATUS_ARCH_MISMATCH, "run_conv_reduction: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(r_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto rTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, r_dim)
                           .setStride(4, stride)
                           .setId('r')  // output
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('y')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << rTensor.describe() << std::endl;

        std::cout << afterConvTensor.describe() << std::endl;

        // Define the reduction descriptor
        auto redunctionDesc = cudnn_frontend::ReductionDescBuilder()
                                  .setComputeType(CUDNN_DATA_FLOAT)
                                  .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                  .build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a reduction add Node.
        auto reduction_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(conv_op.getOutputTensor())
                                .setyDesc(rTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution reduction add
        std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &reduction_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrR};
        int64_t uids[]    = {'x', 'w', 'r'};
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

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        if (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH) {
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

cudnnStatus_t
run_bn_conv_gen_stat(int64_t* xTensorDim,
                     int64_t* wTensorDim,
                     int64_t* yTensorDim,
                     int64_t* scaleTensorDim,
                     int convDim,
                     int64_t* conv_padA,
                     int64_t* conv_dilationA,
                     int64_t* conv_strideA,
                     void* XdevPtr,
                     void* WdevPtr,
                     void* YdevPtr,
                     void* scaledevPtr,
                     void* biasdevPtr,
                     void* sumdevPtr,
                     void* sqSumdevPtr) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(xTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, xTensorDim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(CUDNN_DATA_HALF)
                           .build();

        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, xTensorDim)
                                    .setStride(4, stride)
                                    .setId('d')
                                    .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                    .setDataType(CUDNN_DATA_FLOAT)
                                    .setVirtual()
                                    .build();

        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, xTensorDim)
                                   .setStride(4, stride)
                                   .setId('e')
                                   .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .setVirtual()
                                   .build();

        auto afterReluTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, xTensorDim)
                                   .setStride(4, stride)
                                   .setId('f')
                                   .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .setVirtual()
                                   .build();

        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto scaleTensor = cudnn_frontend::TensorBuilder()
                               .setDim(4, scaleTensorDim)
                               .setStride(4, stride)
                               .setId('s')
                               .setAlignment(16)
                               .setDataType(CUDNN_DATA_HALF)
                               .build();

        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto biasTensor = cudnn_frontend::TensorBuilder()
                              .setDim(4, scaleTensorDim)
                              .setStride(4, stride)
                              .setId('b')
                              .setAlignment(16)
                              .setDataType(CUDNN_DATA_HALF)
                              .build();
        generateStrides(wTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, wTensorDim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_HALF)
                           .build();

        generateStrides(yTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, yTensorDim)
                           .setStride(4, stride)
                           .setId('y')  // after conv
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_HALF)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .setReluLowerClipSlope(0.01)  // leaky relu
                           .build();
        std::cout << actDesc.describe() << std::endl;
        std::cout << "Creating OPs " << std::endl;
        // Create a Multiplication Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(xTensor)
                            .setbDesc(scaleTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(afterScaleTensor)
                           .setbDesc(biasTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(afterBiasTensor)
                          .setyDesc(afterReluTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;
        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(afterReluTensor)
                           .setwDesc(wTensor)
                           .setyDesc(yTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sumTensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, scaleTensorDim)
                             .setStride(4, stride)
                             .setId('u')
                             .setAlignment(16)
                             .setDataType(CUDNN_DATA_FLOAT)
                             .build();

        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sqsumTensor = cudnn_frontend::TensorBuilder()
                               .setDim(4, scaleTensorDim)
                               .setStride(4, stride)
                               .setId('v')
                               .setAlignment(16)
                               .setDataType(CUDNN_DATA_FLOAT)
                               .build();

        // Create a genstats node
        auto genstat_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR)
                              .setxDesc(yTensor)
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
                              .setSumDesc(sumTensor)
                              .setSqSumDesc(sqsumTensor)
                              .build();
        std::cout << genstat_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is scale bias Relu conv gen_stats
        std::array<cudnn_frontend::Operation const*, 5> ops = {&scale_op, &bias_op, &conv_op, &act_op, &genstat_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        cudnn_frontend::ManagedOpaqueDescriptor plan_desc = nullptr;
        int64_t workspace_size                            = 0;
        cudnnStatus_t st                                  = CUDNN_STATUS_SUCCESS;
        for (auto& config : filtered_configs) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle_)
                                .setEngineConfig(config, opGraph.getTag())
                                .build();
                std::cout << "Plan tag: " << plan.getTag() << std::endl;

                workspace_size = plan.getWorkspaceSize();
                std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
                plan_desc = plan.get_desc();
            } catch (cudnn_frontend::cudnnException& e) {
                st = e.getCudnnStatus();
                continue;
            }
        }
        if (plan_desc == nullptr) {
            std::cout << "No plan found implementing the operation graph" << std::endl;
            return st;
        }

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        void* data_ptrs[] = {XdevPtr, WdevPtr, YdevPtr, scaledevPtr, biasdevPtr, sumdevPtr, sqSumdevPtr};
        int64_t uids[]    = {'x', 'w', 'y', 's', 'b', 'u', 'v'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status =
            cudnnBackendExecute(handle_, plan_desc->get_backend_descriptor(), variantPack.get_raw_desc());

        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

        return status;

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere cards
        if (prop.major < 8 && e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
            CHECK(false);
#endif
        }
        return CUDNN_STATUS_SUCCESS;
    }
}

void
run_bn_finalize(int64_t* perChannelSum,
                int64_t* epsilon,

                void* YSumdevPtr,
                void* YSqSumdevPtr,
                void* scaledevPtr,
                void* biasdevPtr,
                void* in_meandevPtr,
                void* in_vardevPtr,
                void* out_meandevPtr,
                void* out_vardevPtr,
                void* saved_meandevPtr,
                void* saved_inv_vardevPtr,
                void* eq_scaledevPtr,
                void* eq_biasdevPtr,

                double epsilon_val,
                double exponential_decay_factor,
                int64_t accumCnt_val) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(perChannelSum, stride, 4, CUDNN_TENSOR_NHWC);

        auto tensor_create = [&stride, &perChannelSum](cudnnDataType_t type, int64_t id) {
            return cudnn_frontend::TensorBuilder()
                .setDim(4, perChannelSum)
                .setStride(4, stride)
                .setId(id)
                .setAlignment(16)
                .setDataType(type)
                .build();
        };

        auto sumTensor         = tensor_create(CUDNN_DATA_FLOAT, 100);
        auto sqSumTensor       = tensor_create(CUDNN_DATA_FLOAT, 101);
        auto scaleTensor       = tensor_create(CUDNN_DATA_FLOAT, 102);
        auto biasTensor        = tensor_create(CUDNN_DATA_FLOAT, 103);
        auto inMeanTensor      = tensor_create(CUDNN_DATA_FLOAT, 104);
        auto inVarTensor       = tensor_create(CUDNN_DATA_FLOAT, 105);
        auto outMeanTensor     = tensor_create(CUDNN_DATA_FLOAT, 106);
        auto outVarTensor      = tensor_create(CUDNN_DATA_FLOAT, 107);
        auto savedMeanTensor   = tensor_create(CUDNN_DATA_FLOAT, 108);
        auto savedInvVarTensor = tensor_create(CUDNN_DATA_FLOAT, 109);
        auto outEqScaleTensor  = tensor_create(CUDNN_DATA_FLOAT, 200);
        auto outEqBiasTensor   = tensor_create(CUDNN_DATA_FLOAT, 201);

        int64_t epsilon_stride[4];
        generateStrides(epsilon, epsilon_stride, 4, CUDNN_TENSOR_NHWC);
        auto scalar_tensor_create = [&epsilon_stride, &epsilon](cudnnDataType_t type, int64_t id) {
            return cudnn_frontend::TensorBuilder()
                .setDim(4, epsilon)
                .setStride(4, epsilon_stride)
                .setId(id)
                .setAlignment(16)
                .setDataType(type)
                .setByValue(true)
                .build();
        };

        auto epsilonTensor    = scalar_tensor_create(CUDNN_DATA_DOUBLE, 300);
        auto expDecayTensor   = scalar_tensor_create(CUDNN_DATA_DOUBLE, 301);
        auto accumCountTensor = scalar_tensor_create(CUDNN_DATA_INT64, 302);

        // Create a Finalize node
        auto finalize_stat_op =
            cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
                .setComputeType(CUDNN_DATA_FLOAT)
                .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
                .setSumDesc(sumTensor)
                .setSqSumDesc(sqSumTensor)
                .setScaleAndBias(scaleTensor, biasTensor)
                .setEqScaleAndBias(outEqScaleTensor, outEqBiasTensor)
                .setPrevRunningMeanAndVar(inMeanTensor, inVarTensor)
                .setNextRunningMeanAndVar(outMeanTensor, outVarTensor)
                .setSavedMeanAndInvVar(savedMeanTensor, savedInvVarTensor)
                .setEpsilonTensor(epsilonTensor)
                .setAccumCountTensor(accumCountTensor)
                .setExpDecayFactorTensor(expDecayTensor)
                .build();

        std::array<cudnn_frontend::Operation const*, 1> ops = {&finalize_stat_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan_builder = [&filtered_configs, &opGraph, &handle_]() {
            for (size_t i = 0; i < filtered_configs.size(); i++) {
                try {
                    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                    .setHandle(handle_)
                                    .setEngineConfig(filtered_configs[i], opGraph.getTag())
                                    .build();
                    return plan;
                } catch (cudnn_frontend::cudnnException&) {
                    continue;
                }
            }
            return cudnn_frontend::ExecutionPlanBuilder()
                .setHandle(handle_)
                .setEngineConfig(filtered_configs[0], opGraph.getTag())
                .build();
        };

        REQUIRE(filtered_configs.size() > 0);
        auto plan = plan_builder();
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        void* data_ptrs[15] = {YSumdevPtr,
                               YSqSumdevPtr,
                               scaledevPtr,
                               biasdevPtr,
                               in_meandevPtr,
                               in_vardevPtr,
                               out_meandevPtr,
                               out_vardevPtr,
                               saved_meandevPtr,
                               saved_inv_vardevPtr,
                               eq_scaledevPtr,
                               eq_biasdevPtr,
                               &epsilon_val,
                               &exponential_decay_factor,
                               &accumCnt_val};
        int64_t uids[15]    = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 200, 201, 300, 301, 302};
        auto variantPack    = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(15, data_ptrs)
                               .setUids(15, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

        std::cout << "BN Finalize run completed successfully" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
#if (CUDNN_VERSION >= 8400)
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
#endif
    }
}

cudnnStatus_t
run_dsbar(int64_t* Y_dim,
          int64_t* scaleTensorDim,
          void* RP_YdevPtr,
          void* RP_scaleDevPtr,
          void* RP_biasDevPtr,
          void* DP_YdevPtr,
          void* DP_scaleDevPtr,
          void* DP_biasDevPtr,
          void* YdevPtr,
          cudnnDataType_t op_data_type) {
    cudnnHandle_t handle_;

    try {
        // Create a handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Create tensor descriptors
        int64_t stride[4];

        // RP_Y tensor
        generateStrides(Y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto RP_yTensor = cudnn_frontend::TensorBuilder()
                              .setDim(4, Y_dim)
                              .setStride(4, stride)
                              .setId('y')
                              .setAlignment(16)  // 16 byte alignment
                              .setDataType(CUDNN_DATA_HALF)
                              .build();

        // RP_scale tensor
        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto RP_scaleTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, scaleTensorDim)
                                  .setStride(4, stride)
                                  .setId('s')
                                  .setAlignment(16)  // 16 byte alignment
                                  .setDataType(CUDNN_DATA_FLOAT)
                                  .build();

        // After RP scale tensor (RP_yTensor * RP_scaleTensor)
        generateStrides(Y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto RP_afterScaleTensor = cudnn_frontend::TensorBuilder()
                                       .setDim(4, Y_dim)
                                       .setStride(4, stride)
                                       .setId('d')
                                       .setVirtual()
                                       .setAlignment(16)  // 16 byte alignment
                                       .setDataType(CUDNN_DATA_FLOAT)
                                       .build();

        // RP_bias tensor
        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto RP_biasTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(4, scaleTensorDim)
                                 .setStride(4, stride)
                                 .setId('b')
                                 .setAlignment(16)  // 16 byte alignment
                                 .setDataType(CUDNN_DATA_FLOAT)
                                 .build();

        // After RP bias tensor (RP_afterScaleTensor + RP_biasTensor)
        generateStrides(Y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto RP_afterBiasTensor = cudnn_frontend::TensorBuilder()
                                      .setDim(4, Y_dim)
                                      .setStride(4, stride)
                                      .setId('e')
                                      .setVirtual()
                                      .setAlignment(16)  // 16 byte alignment
                                      .setDataType(CUDNN_DATA_FLOAT)
                                      .build();

        // DP_Y tensor
        generateStrides(Y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto DP_yTensor = cudnn_frontend::TensorBuilder()
                              .setDim(4, Y_dim)
                              .setStride(4, stride)
                              .setId('a')
                              .setAlignment(16)  // 16 byte alignment
                              .setDataType(CUDNN_DATA_HALF)
                              .build();

        // DP_scale tensor
        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto DP_scaleTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, scaleTensorDim)
                                  .setStride(4, stride)
                                  .setId('h')
                                  .setAlignment(16)  // 16 byte alignment
                                  .setDataType(CUDNN_DATA_FLOAT)
                                  .build();

        // After DP scale tensor (DP_yTensor * DP_scaleTensor)
        generateStrides(Y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto DP_afterScaleTensor = cudnn_frontend::TensorBuilder()
                                       .setDim(4, Y_dim)
                                       .setStride(4, stride)
                                       .setId('p')
                                       .setVirtual()
                                       .setAlignment(16)  // 16 byte alignment
                                       .setDataType(CUDNN_DATA_FLOAT)
                                       .build();

        // DP_bias tensor
        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto DP_biasTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(4, scaleTensorDim)
                                 .setStride(4, stride)
                                 .setId('t')
                                 .setAlignment(16)  // 16 byte alignment
                                 .setDataType(CUDNN_DATA_FLOAT)
                                 .build();

        // After DP bias tensor (DP_afterScaleTensor + DP_biasTensor)
        generateStrides(Y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto DP_afterBiasTensor = cudnn_frontend::TensorBuilder()
                                      .setDim(4, Y_dim)
                                      .setStride(4, stride)
                                      .setId('n')
                                      .setVirtual()
                                      .setAlignment(16)  // 16 byte alignment
                                      .setDataType(CUDNN_DATA_FLOAT)
                                      .build();

        // After add RP_bias and DP_bias tensor (RP_afterBiasTensor + DP_afterBiasTensor)
        generateStrides(Y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterAddTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, Y_dim)
                                  .setStride(4, stride)
                                  .setId('m')
                                  .setVirtual()
                                  .setAlignment(16)  // 16 byte alignment
                                  .setDataType(CUDNN_DATA_FLOAT)
                                  .build();

        // Final output tensor after ReLU
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, Y_dim)
                           .setStride(4, stride)
                           .setId('f')
                           .setAlignment(16)  // 16 byte alignment
                           .setDataType(op_data_type)
                           .build();

        std::cout << RP_yTensor.describe() << std::endl;
        std::cout << DP_yTensor.describe() << std::endl;

        // Create the scale, add, and relu problems
        // Scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Bias (add) descriptor
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // ReLU descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;
        std::cout << "Creating Operations now!" << std::endl;

        // Create RP scaling operation
        auto RP_scaleOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                              .setxDesc(RP_yTensor)
                              .setbDesc(RP_scaleTensor)
                              .setyDesc(RP_afterScaleTensor)
                              .setpwDesc(scaleDesc)
                              .build();
        std::cout << RP_scaleOp.describe() << std::endl;

        // Create RP bias operation
        auto RP_biasOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(RP_afterScaleTensor)
                             .setbDesc(RP_biasTensor)
                             .setyDesc(RP_afterBiasTensor)
                             .setpwDesc(addDesc)
                             .build();
        std::cout << RP_biasOp.describe() << std::endl;

        // Create DP scaling operation
        auto DP_scaleOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                              .setxDesc(DP_yTensor)
                              .setbDesc(DP_scaleTensor)
                              .setyDesc(DP_afterScaleTensor)
                              .setpwDesc(scaleDesc)
                              .build();
        std::cout << DP_scaleOp.describe() << std::endl;

        // Create DP bias operation
        auto DP_biasOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(DP_afterScaleTensor)
                             .setbDesc(DP_biasTensor)
                             .setyDesc(DP_afterBiasTensor)
                             .setpwDesc(addDesc)
                             .build();
        std::cout << DP_biasOp.describe() << std::endl;

        // Create add operation
        auto addOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(RP_afterBiasTensor)
                         .setbDesc(DP_afterBiasTensor)
                         .setyDesc(afterAddTensor)
                         .setpwDesc(addDesc)
                         .build();
        std::cout << addOp.describe() << std::endl;

        // Create ReLU operation
        auto actOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(afterAddTensor)
                         .setyDesc(yTensor)
                         .setpwDesc(actDesc)
                         .build();
        std::cout << actOp.describe() << std::endl;
        std::cout << "Creating operation graph now!" << std::endl;

        // Create an Operation Graph. In this case it is:
        // RP_scaleOp -> RP_biasOp -> DP_scaleOp -> DP_biasOp -> addOp -> reluOp
        std::array<cudnn_frontend::Operation const*, 6> ops = {
            &RP_scaleOp, &RP_biasOp, &DP_scaleOp, &DP_biasOp, &addOp, &actOp};
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        void* data_ptrs[] = {
            RP_YdevPtr, DP_YdevPtr, RP_scaleDevPtr, DP_scaleDevPtr, RP_biasDevPtr, DP_biasDevPtr, YdevPtr};
        int64_t uids[]   = {'y', 'a', 's', 'h', 'b', 't', 'f'};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

        return status;

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED || e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH)) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
            return e.getCudnnStatus();
        }
#if (CUDNN_VERSION >= 8300)
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
#endif
        return e.getCudnnStatus();
    }
}

cudnnStatus_t
run_conv_two_global_scales(int64_t* xTensorDim,
                           int64_t* wTensorDim,
                           int64_t* yTensorDim,
                           int64_t* scaleTensorDim,
                           int convDim,
                           int64_t* conv_padA,
                           int64_t* conv_dilationA,
                           int64_t* conv_strideA,
                           void* devPtrX,
                           void* devPtrW,
                           void* devPtrScale1,
                           void* devPtrScale2,
                           void* devPtrOutput,
                           void* afterConv) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, CUDNN_STATUS_ARCH_MISMATCH, "run_conv_two_global_scales: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(xTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, xTensorDim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(CUDNN_DATA_HALF)
                           .build();

        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto scale1Tensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, scaleTensorDim)
                                .setStride(4, stride)
                                .setId('s')
                                .setAlignment(16)
                                .setDataType(CUDNN_DATA_FLOAT)
                                .build();

        auto scale2Tensor = cudnn_frontend::TensorBuilder().cloneFrom(scale1Tensor, 'b').build();

        generateStrides(wTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, wTensorDim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_HALF)
                           .build();

        generateStrides(yTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, yTensorDim)
                                   .setStride(4, stride)
                                   .setId('a')  // after conv
                                   .setAlignment(16)
                                   .setDataType(CUDNN_DATA_HALF)
                                   .build();

        auto afterScale1Tensor = cudnn_frontend::TensorBuilder().cloneFrom(afterConvTensor, 'v').setVirtual().build();

        auto finalOutputTensor =
            cudnn_frontend::TensorBuilder().cloneFrom(afterConvTensor, 'y').setVirtual(false).build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << finalOutputTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        std::cout << "Creating OPs " << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto scale1_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(afterConvTensor)
                             .setbDesc(scale1Tensor)
                             .setyDesc(afterScale1Tensor)
                             .setpwDesc(scaleDesc)
                             .build();
        std::cout << scale1_op.describe() << std::endl;

        auto scale2_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(afterScale1Tensor)
                             .setbDesc(scale2Tensor)
                             .setyDesc(finalOutputTensor)
                             .setpwDesc(scaleDesc)
                             .build();
        std::cout << scale2_op.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is scale bias Relu conv gen_stats
        std::array<cudnn_frontend::Operation const*, 3> ops = {&conv_op, &scale1_op, &scale2_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        cudnn_frontend::ManagedOpaqueDescriptor plan_desc = nullptr;
        int64_t workspace_size                            = 0;
        for (auto& config : filtered_configs) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle_)
                                .setEngineConfig(config, opGraph.getTag())
                                .build();
                std::cout << "Plan tag: " << plan.getTag() << std::endl;

                workspace_size = plan.getWorkspaceSize();
                std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
                plan_desc = plan.get_desc();
            } catch (cudnn_frontend::cudnnException&) {
                continue;
            }
        }
        if (plan_desc == nullptr) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_NOT_SUPPORTED,
                "run_conv_two_global_scales: No plan found to be implementing this operation graph");
        }

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        void* data_ptrs[] = {devPtrX, devPtrW, devPtrScale1, devPtrScale2, devPtrOutput, afterConv};
        int64_t uids[]    = {'x', 'w', 's', 'b', 'y', 'a'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(6, data_ptrs)
                               .setUids(6, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status =
            cudnnBackendExecute(handle_, plan_desc->get_backend_descriptor(), variantPack.get_raw_desc());

        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        return status;
    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED || e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH)) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
            return e.getCudnnStatus();
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
            CHECK(false);
#endif
            return e.getCudnnStatus();
        }
    }
}

#if (CUDNN_VERSION >= 8600)
void
run_maxpool_with_idx(int64_t* x_dim,
                     int64_t* y_dim,
                     int64_t* idx_dim,
                     void* devPtrdX,
                     void* devPtrdY,
                     void* devPtrIdx,
                     cudnnDataType_t tensorType,
                     cudnnNanPropagation_t const nanOpt,
                     cudnn_frontend::ResampleMode_t mode,
                     cudnn_frontend::PaddingMode_t const padding_mode,
                     int32_t nbSpatialDims,
                     int64_t* windowDimA,
                     int64_t* prePaddingA,
                     int64_t* postPaddingA,
                     int64_t* strideA) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t strideTensor[4];
        generateStrides(x_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, strideTensor)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(tensorType)
                           .build();

        generateStrides(y_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStrides(4, strideTensor)
                           .setId('y')  // after conv
                           .setAlignment(16)
                           .setDataType(tensorType)
                           .build();

        generateStrides(idx_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto idxTensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, idx_dim)
                             .setStrides(4, strideTensor)
                             .setId('i')
                             .setAlignment(16)
                             .setDataType(CUDNN_DATA_INT8)
                             .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;
        std::cout << idxTensor.describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setNanPropagation(nanOpt)
                            .setResampleMode(mode)
                            .setPaddingMode(padding_mode)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setSpatialStride(nbSpatialDims, strideA)
                            .setPrePadding(nbSpatialDims, prePaddingA)
                            .setPostPadding(nbSpatialDims, postPaddingA)
                            .build();
        std::cout << "Initialized Pool Desc" << std::endl;
        std::cout << poolDesc.describe() << std::endl;

        // Create a maxpooling Resample Node with index tensor
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setyDesc(yTensor)
                           .setidxDesc(idxTensor)
                           .setResampleDesc(poolDesc)
                           .build();
        std::cout << pool_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 1> ops = {&pool_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        // Create the variant pack and associate with the data pointers
        void* data_ptrs[] = {devPtrdX, devPtrdY, devPtrIdx};
        int64_t uids[]    = {'x', 'y', 'i'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // Trigger the execute operation
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        checkCudnnErr(cudnnDestroy(handle_));
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        std::cout << "EXECUTE SUCCESS" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif

#if (CUDNN_VERSION >= 8600)
void
run_backward_avgpool(int64_t* dx_dim,
                     int64_t* dy_dim,
                     void* devPtrdX,
                     void* devPtrdY,
                     cudnnDataType_t tensorType,
                     cudnnNanPropagation_t const nanOpt,
                     cudnn_frontend::ResampleMode_t mode,
                     cudnn_frontend::PaddingMode_t const padding_mode,
                     int32_t nbSpatialDims,
                     int64_t* windowDimA,
                     int64_t* prePaddingA,
                     int64_t* postPaddingA,
                     int64_t* strideA) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t strideTensor[4];
        generateStrides(dy_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto dyTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, dy_dim)
                            .setStrides(4, strideTensor)
                            .setId('y')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(tensorType)
                            .build();

        generateStrides(dx_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto dxTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, dx_dim)
                            .setStrides(4, strideTensor)
                            .setId('x')  // after conv
                            .setAlignment(16)
                            .setDataType(tensorType)
                            .build();

        std::cout << dyTensor.describe() << std::endl;
        std::cout << dxTensor.describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setNanPropagation(nanOpt)
                            .setResampleMode(mode)
                            .setPaddingMode(padding_mode)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setSpatialStride(nbSpatialDims, strideA)
                            .setPrePadding(nbSpatialDims, prePaddingA)
                            .setPostPadding(nbSpatialDims, postPaddingA)
                            .build();
        std::cout << "Initialized Pool Desc" << std::endl;
        std::cout << poolDesc.describe() << std::endl;

        // Create an average pooling Resample Node
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                           .setdxDesc(dxTensor)
                           .setdyDesc(dyTensor)
                           .setResampleDesc(poolDesc)
                           .build();
        std::cout << pool_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 1> ops = {&pool_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        // Create the variant pack and associate with the data pointers
        void* data_ptrs[] = {devPtrdX, devPtrdY};
        int64_t uids[]    = {'x', 'y'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(2, data_ptrs)
                               .setUids(2, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // Trigger the execute operation
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        checkCudnnErr(cudnnDestroy(handle_));
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        std::cout << "EXECUTE SUCCESS" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere cards
        if (prop.major < 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif

#if (CUDNN_VERSION >= 8600)
void
run_backward_maxpool(int64_t* dx_dim,
                     int64_t* dy_dim,
                     int64_t* idx_dim,
                     void* devPtrdX,
                     void* devPtrdY,
                     void* devPtrIdx,
                     cudnnDataType_t tensorType,
                     cudnnNanPropagation_t const nanOpt,
                     cudnn_frontend::ResampleMode_t mode,
                     cudnn_frontend::PaddingMode_t const padding_mode,
                     int32_t nbSpatialDims,
                     int64_t* windowDimA,
                     int64_t* prePaddingA,
                     int64_t* postPaddingA,
                     int64_t* strideA) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t strideTensor[4];
        generateStrides(dy_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto dyTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, dy_dim)
                            .setStrides(4, strideTensor)
                            .setId('y')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(tensorType)
                            .build();

        generateStrides(dx_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto dxTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, dx_dim)
                            .setStrides(4, strideTensor)
                            .setId('x')  // after conv
                            .setAlignment(16)
                            .setDataType(tensorType)
                            .build();

        generateStrides(idx_dim, strideTensor, 4, CUDNN_TENSOR_NHWC);
        auto idxTensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, idx_dim)
                             .setStrides(4, strideTensor)
                             .setId('i')
                             .setAlignment(16)
                             .setDataType(CUDNN_DATA_INT8)
                             .build();

        std::cout << dyTensor.describe() << std::endl;
        std::cout << dxTensor.describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setNanPropagation(nanOpt)
                            .setResampleMode(mode)
                            .setPaddingMode(padding_mode)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setSpatialStride(nbSpatialDims, strideA)
                            .setPrePadding(nbSpatialDims, prePaddingA)
                            .setPostPadding(nbSpatialDims, postPaddingA)
                            .build();
        std::cout << "Initialized Pool Desc" << std::endl;
        std::cout << poolDesc.describe() << std::endl;

        // Create a maxpooling Resample Node with index tensor
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                           .setdxDesc(dxTensor)
                           .setdyDesc(dyTensor)
                           .setidxDesc(idxTensor)
                           .setResampleDesc(poolDesc)
                           .build();
        std::cout << pool_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 1> ops = {&pool_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        // Create the variant pack and associate with the data pointers
        void* data_ptrs[] = {devPtrdX, devPtrdY, devPtrIdx};
        int64_t uids[]    = {'x', 'y', 'i'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // Trigger the execute operation
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        checkCudnnErr(cudnnDestroy(handle_));
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        std::cout << "EXECUTE SUCCESS" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere cards
        if (prop.major != 8 &&
            (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif

#if (CUDNN_VERSION >= 8400)
void
run_bn_bwd_weight(int64_t* xDim,
                  int64_t* dyDim,
                  int64_t* wDim,
                  int64_t* scaleDim,
                  void* x_bn_fwd,
                  void* w_fwd,
                  void* dy,
                  void* dy_bn,
                  void* mean,
                  void* inv_var,
                  void* scale,
                  void* bias,
                  void* d_scale,
                  void* d_bias,
                  void* eqscale_dy,
                  void* eqscale_x,
                  void* eqbias) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_ARCH_MISMATCH,
                "run_conv_scale_bias_relu_gen_index_selection: Sample requires Ampere or above GPU");
        }

        cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

        // Creates the necessary tensor descriptors
        int64_t xstrideTensor[4];
        int64_t dystrideTensor[4];
        int64_t wstrideTensor[4];
        generateStrides(xDim, xstrideTensor, 4, CUDNN_TENSOR_NHWC);
        generateStrides(dyDim, dystrideTensor, 4, CUDNN_TENSOR_NHWC);
        generateStrides(wDim, wstrideTensor, 4, CUDNN_TENSOR_NHWC);

        int64_t perChannelStride[4];
        generateStrides(scaleDim, perChannelStride, 4, CUDNN_TENSOR_NHWC);

        auto tensor_create = [](int64_t* stride, int64_t* dim, cudnnDataType_t type, int64_t id, bool is_virtual) {
            return cudnn_frontend::TensorBuilder()
                .setDim(4, dim)
                .setStride(4, stride)
                .setId(id)
                .setAlignment(16)
                .setDataType(type)
                .setVirtual(is_virtual)
                .build();
        };

        auto pointwise_create = [](cudnnPointwiseMode_t mode) {
            return cudnn_frontend::PointWiseDescBuilder().setMode(mode).setComputeType(CUDNN_DATA_FLOAT).build();
        };

        auto pointwise_op_create = [](cudnn_frontend::Tensor& x,
                                      cudnn_frontend::Tensor& s,
                                      cudnn_frontend::Tensor& y,
                                      cudnn_frontend::PointWiseDesc& pw) {
            return cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(x)
                .setbDesc(s)
                .setyDesc(y)
                .setpwDesc(pw)
                .build();
        };

        auto x_tensor_bn_fwd = tensor_create(xstrideTensor, xDim, CUDNN_DATA_HALF, 100, false);
        auto w_tensor        = tensor_create(wstrideTensor, wDim, CUDNN_DATA_HALF, 101, false);
        auto dy_tensor       = tensor_create(dystrideTensor, dyDim, CUDNN_DATA_HALF, 102, false);
        auto dy_bn_tensor    = tensor_create(xstrideTensor, xDim, CUDNN_DATA_HALF, 103, false);

        auto scaleTensor  = tensor_create(perChannelStride, scaleDim, computeType, 200, false);
        auto biasTensor   = tensor_create(perChannelStride, scaleDim, computeType, 201, false);
        auto meanTensor   = tensor_create(perChannelStride, scaleDim, computeType, 202, false);
        auto invVarTensor = tensor_create(perChannelStride, scaleDim, computeType, 203, false);

        auto d_scaleTensor    = tensor_create(perChannelStride, scaleDim, computeType, 300, false);
        auto d_biasTensor     = tensor_create(perChannelStride, scaleDim, computeType, 301, false);
        auto eqscale_dyTensor = tensor_create(perChannelStride, scaleDim, computeType, 302, false);
        auto eqscale_xTensor  = tensor_create(perChannelStride, scaleDim, computeType, 303, false);
        auto eqbiasTensor     = tensor_create(perChannelStride, scaleDim, computeType, 304, false);

        auto after_scaleTensor  = tensor_create(xstrideTensor, xDim, computeType, 400, true);
        auto after_biasTensor   = tensor_create(xstrideTensor, xDim, computeType, 401, true);
        auto after_meanTensor   = tensor_create(xstrideTensor, xDim, computeType, 402, true);
        auto after_invVarTensor = tensor_create(xstrideTensor, xDim, computeType, 403, true);

        auto after_dgrad_tensor = tensor_create(xstrideTensor, xDim, CUDNN_DATA_HALF, 500, true);

        // Define the pointwise descriptor
        auto scaleDesc   = pointwise_create(CUDNN_POINTWISE_MUL);
        auto biasDesc    = pointwise_create(CUDNN_POINTWISE_ADD);
        auto addDesc     = pointwise_create(CUDNN_POINTWISE_ADD);
        auto mulDesc     = pointwise_create(CUDNN_POINTWISE_MUL);
        auto bwdReluDesc = pointwise_create(CUDNN_POINTWISE_RELU_BWD);

        // Create Pointwise Operations
        auto addOpDesc     = pointwise_op_create(x_tensor_bn_fwd, meanTensor, after_meanTensor, addDesc);
        auto mulOpDesc     = pointwise_op_create(after_meanTensor, invVarTensor, after_invVarTensor, mulDesc);
        auto scaleOpDesc   = pointwise_op_create(after_invVarTensor, scaleTensor, after_scaleTensor, scaleDesc);
        auto biasOpDesc    = pointwise_op_create(after_scaleTensor, biasTensor, after_biasTensor, biasDesc);
        auto bwdReluOpDesc = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                 .setdyDesc(after_dgrad_tensor)
                                 .setxDesc(after_biasTensor)
                                 .setdxDesc(dy_bn_tensor)
                                 .setpwDesc(bwdReluDesc)
                                 .build();

        // Create dgrad desc and operation
        int64_t convDim      = 2;
        int64_t padding[]    = {1, 1};
        int64_t dilation[]   = {1, 1};
        int64_t convstride[] = {1, 1};

        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(computeType)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, convstride)
                            .setPrePadding(convDim, padding)
                            .setPostPadding(convDim, padding)
                            .setDilation(convDim, dilation)
                            .build();

        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                           .setdyDesc(dy_tensor)
                           .setwDesc(w_tensor)
                           .setdxDesc(after_dgrad_tensor)
                           .setcDesc(convDesc)
                           .setAlpha(1.0f)
                           .setBeta(0.0f)
                           .build();

        auto bn_bwd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
                             .setComputeType(computeType)
                             .setxDesc(x_tensor_bn_fwd)
                             .setSavedMeanAndInvVar(meanTensor, invVarTensor)
                             .setScale(scaleTensor)
                             .setdyDesc(dy_bn_tensor)
                             .setEqScalesAndBias(eqscale_dyTensor, eqscale_xTensor, eqbiasTensor)
                             .setDScaleAndDBias(d_scaleTensor, d_biasTensor)
                             .build();

        // Create an Operation Graph. In this case it is convolution scale bias add activation
        std::array<cudnn_frontend::Operation const*, 7> ops = {
            &conv_op, &addOpDesc, &mulOpDesc, &scaleOpDesc, &biasOpDesc, &bwdReluOpDesc, &bn_bwd_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }

        void* data_ptrs[] = {
            x_bn_fwd, w_fwd, dy, dy_bn, scale, bias, mean, inv_var, d_scale, d_bias, eqscale_dy, eqscale_x, eqbias};
        int64_t uids[]   = {100, 101, 102, 103, 200, 201, 202, 203, 300, 301, 302, 303, 304};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(13, data_ptrs)
                               .setUids(13, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        checkCudnnErr(cudnnDestroy(handle_));

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        // this example is only for Ampere and Hopper cards
        bool is_supported_on_ampere = is_ampere_arch();
        bool is_supported_on_hopper = is_hopper_arch() && (cudnnGetVersion() >= 8900);
        if (((!is_supported_on_hopper) && (!is_supported_on_ampere)) &&
            (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            SKIP("Example is only supported for Ampere and Hopper GPUs");
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif
