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
#include "error_util.h"

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
        auto statuses = 
            cudnn_frontend::get_heuristics_list<1>({
            "heuristics_fallback"
            }, opGraph,::allowAll, filtered_configs, true);
        
        std::cout << "get_heuristics_list Statuses: ";
        for (auto i = 0 ; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
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
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStrides(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStrides(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(a_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto aTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, a_dim)
                           .setStrides(4, stride)
                           .setId('a')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStrides(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterAddTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, y_dim)
                                  .setStrides(4, stride)
                                  .setId('D')  // after add
                                  .setAlignment(16)
                                  .setVirtual()
                                  .setDataType(dataType)
                                  .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStrides(4, stride)
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
                             .setMathPrecision(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the add descriptor
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .setReluLowerClipSlope(0.01)  // leaky relu
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
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
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
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
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStrides(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStrides(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStrides(4, stride)
                                    .setId('C')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStrides(4, stride)
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
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
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
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStrides(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStrides(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStrides(4, stride)
                                    .setId('C')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStrides(4, stride)
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
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
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
#if (CUDNN_VERSION >= 8400)
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
    try {
#if (CUDNN_VERSION >= 8400)
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStrides(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStrides(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStrides(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(CUDNN_DATA_FLOAT)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();

        auto afterActivationTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('D')  // after activation
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();

        auto genIndexTensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, y_dim)
                                .setStrides(4, stride)
                                .setId('I')  // output of the gen index operation
                                .setAlignment(16)
                                .setVirtual()
                                .setDataType(CUDNN_DATA_INT32)
                                .build();

        auto maskTopTensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, y_dim)
                                .setStrides(4, stride)
                                .setId('m')  // top half of the mask created after the less than
                                .setAlignment(16)
                                .setVirtual()
                                .setDataType(CUDNN_DATA_BOOLEAN)
                                .build();

        auto maskBottomTensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, y_dim)
                                .setStrides(4, stride)
                                .setId('n')  // bottom half of the mask
                                .setAlignment(16)
                                .setVirtual()
                                .setDataType(CUDNN_DATA_BOOLEAN)
                                .build();

        auto maskTensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, y_dim)
                                .setStrides(4, stride)
                                .setId('M')  // OR of the top and bottom masks
                                .setAlignment(16)
                                .setVirtual()
                                .setDataType(CUDNN_DATA_BOOLEAN)
                                .build();

        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStrides(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(threshold_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto thresholdTopTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, threshold_dim)
                           .setStrides(4, stride)
                           .setId('t')  // threshold for creating the top mask
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_INT32)
                           .build();

        auto thresholdBottomTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, threshold_dim)
                           .setStrides(4, stride)
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
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the genIndex descriptor
        auto genIndexDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_GEN_INDEX)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .setAxis(axis)
                           .build();
        std::cout << genIndexDesc.describe() << std::endl;

        // Define the lessThan descriptor
        auto lessThanDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_CMP_LT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << lessThanDesc.describe() << std::endl;

        // Define the greaterThan descriptor
        auto greaterThanDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_CMP_GT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << greaterThanDesc.describe() << std::endl;

        // Define the logical_or descriptor
        auto logicalOrDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_LOGICAL_OR)
                           .setMathPrecision(CUDNN_DATA_BOOLEAN)
                           .build();
        std::cout << logicalOrDesc.describe() << std::endl;

        // Define the binary_selection descriptor
        auto selectionDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_BINARY_SELECT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
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
        std::array<cudnn_frontend::Operation const*, 9> ops = {&conv_op, &scale_op, &bias_op, &act_op, &genIndex_op, &lessThan_op, &greaterThan_op, &logicalOr_op, &selection_op};

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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
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

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(CUDNN_DATA_INT8)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_INT8)
                           .build();
        generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStrides(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStrides(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_INT32)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStrides(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(CUDNN_DATA_FLOAT)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStrides(4, stride)
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
                            .setDataType(CUDNN_DATA_INT32)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
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
        checkCudaErrors(cudaGetDeviceProperties( &prop, 0 ));
        // this example is only for Ampere cards
        if (prop.major < 8 && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl; 
        }  else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
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
                     void* devPtrZ) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[3];
        // the intension is to compute stride for a [1, M, K] matrix with K in the inner most dimension, and
        // CUDNN_TENSOR_NCHW is a borrowed notation
        generateStrides(a_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto aMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, a_dim)
                                 .setStrides(3, stride)
                                 .setId('a')
                                 .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                 .setDataType(dataType)
                                 .build();
        generateStrides(b_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto bMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, b_dim)
                                 .setStrides(3, stride)
                                 .setId('b')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        generateStrides(z_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto biasTensor = cudnn_frontend::TensorBuilder()
                              .setDim(3, z_dim)
                              .setStrides(3, stride)
                              .setId('z')
                              .setAlignment(16)
                              .setDataType(dataType)
                              .build();

        generateStrides(c_dim, stride, 3, CUDNN_TENSOR_NCHW);
        auto afterMatMulTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, c_dim)
                                     .setStrides(3, stride)
                                     .setId('A')  // after matmul
                                     .setAlignment(16)
                                     .setVirtual()
                                     .setDataType(dataType)
                                     .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(3, c_dim)
                                   .setStrides(3, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto outputTensor = cudnn_frontend::TensorBuilder()
                                .setDim(3, c_dim)
                                .setStrides(3, stride)
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
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_GELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the matmul desc
        auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setMathPrecision(CUDNN_DATA_FLOAT).build();
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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }
        void* data_ptrs[] = {devPtrA, devPtrB, devPtrC, devPtrZ};
        int64_t uids[]    = {'a', 'b', 'c', 'z'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
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
                            .setStrides(4, x_stride_padded)
                            .setId(x_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto w_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, w_dim)
                            .setStrides(4, w_stride_padded)
                            .setId(w_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto after_conv_tensor = cudnn_frontend::TensorBuilder()
                                     .setDim(4, y_dim)
                                     .setStrides(4, y_stride_padded)
                                     .setId(after_conv_id)
                                     .setAlignment(4)
                                     .setVirtual()
                                     .setDataType(dataType)
                                     .build();

        auto bwd_act_x_tensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStrides(4, y_stride_padded)
                                    .setId(bwd_act_x_id)
                                    .setAlignment(4)
                                    .setDataType(dataType)
                                    .build();

        auto after_activation_tensor = cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim)
                                           .setStrides(4, y_stride_padded)
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
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
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
                           .setMathPrecision(CUDNN_DATA_FLOAT)
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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
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
                                         .setStrides(4, dx_stride)
                                         .setId(after_dgrad_id)
                                         .setAlignment(4)
                                         .setVirtual()
                                         .setDataType(dataType)
                                         .build();

        auto w_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, w_dim)
                            .setStrides(4, w_stride)
                            .setId(w_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto dy_tensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, dy_dim)
                             .setStrides(4, dy_stride)
                             .setId(dy_id)
                             .setAlignment(4)
                             .setDataType(dataType)
                             .build();

        auto bwd_act_x_tensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, dx_dim)
                                    .setStrides(4, dx_stride)
                                    .setId(bwd_act_x_id)
                                    .setAlignment(4)
                                    .setDataType(dataType)
                                    .build();

        auto after_bwd_activation_dx_tensor = cudnn_frontend::TensorBuilder()
                                                  .setDim(4, dx_dim)
                                                  .setStrides(4, dx_stride)
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
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
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
                           .setMathPrecision(CUDNN_DATA_FLOAT)
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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
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
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
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

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(r_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto rTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, r_dim)
                           .setStrides(4, stride)
                           .setId('r')  // output
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
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
                                  .setMathPrecision(CUDNN_DATA_FLOAT)
                                  .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                  .build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
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
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
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
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_bn_conv_gen_stat(int64_t* xTensorDim, 
                    int64_t* wTensorDim, 
                    int64_t* yTensorDim,
                    int64_t* scaleTensorDim,  
                    int convDim, 
                    int64_t *conv_padA, 
                    int64_t* conv_dilationA, 
                    int64_t* conv_strideA, 
                    void *XdevPtr, 
                    void *WdevPtr, 
                    void *YdevPtr,
                    void *scaledevPtr, 
                    void *biasdevPtr, 
                    void *sumdevPtr, 
                    void *sqSumdevPtr) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(xTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, xTensorDim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(CUDNN_DATA_HALF)
                           .build();
        
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, xTensorDim)
                                .setStrides(4, stride)
                                .setId('d')
                                .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                .setDataType(CUDNN_DATA_FLOAT)
                                .setVirtual()
                                .build();

        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, xTensorDim)
                                .setStrides(4, stride)
                                .setId('e')
                                .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                .setDataType(CUDNN_DATA_FLOAT)
                                .setVirtual()
                                .build();

        auto afterReluTensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, xTensorDim)
                                .setStrides(4, stride)
                                .setId('f')
                                .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                .setDataType(CUDNN_DATA_FLOAT)
                                .setVirtual()
                                .build();

        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto scaleTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, scaleTensorDim)
                           .setStrides(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_HALF)
                           .build();

        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto biasTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, scaleTensorDim)
                           .setStrides(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_HALF)
                           .build();
        generateStrides(wTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, wTensorDim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_HALF)
                           .build();

        generateStrides(yTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto yTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, yTensorDim)
                                   .setStrides(4, stride)
                                   .setId('y')  // after conv
                                   .setAlignment(16)
                                   .setDataType(CUDNN_DATA_HALF)
                                   .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

                // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(CUDNN_DATA_FLOAT)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
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
                           .setStrides(4, stride)
                           .setId('u')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        generateStrides(scaleTensorDim, stride, 4, CUDNN_TENSOR_NHWC);
        auto sqsumTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, scaleTensorDim)
                           .setStrides(4, stride)
                           .setId('v')
                           .setAlignment(16)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();

        //Create a genstats node
        auto genstat_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR)
                    .setxDesc(yTensor)
                    .setMathPrecision(CUDNN_DATA_FLOAT)
                    .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
                    .setSumDesc(sumTensor)
                    .setSqSumDesc(sqsumTensor)
                    .build();
        std::cout << genstat_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is scale bias Relu conv gen_stats
        std::array<cudnn_frontend::Operation const*, 5> ops = {&scale_op, &bias_op, &conv_op, &act_op, &genstat_op};
        auto opGraph = cudnn_frontend::OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = 
            cudnn_frontend::get_heuristics_list<2>({"heuristics_instant" 
            , "heuristics_fallback"
            }, opGraph,::allowAll, filtered_configs, true);
        
        std::cout << "get_heuristics_list Statuses: ";
        for (auto i = 0 ; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan =
            cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }

        void* data_ptrs[] = {XdevPtr, WdevPtr, YdevPtr, scaledevPtr, biasdevPtr, sumdevPtr, sqSumdevPtr};
        int64_t uids[]    = {'x', 'w', 'y', 's', 'b', 'u', 'v'};
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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

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
    }
}

void
run_bn_finalize( 
    int64_t *perChannelSum, 
    int64_t *epsilon,

    void *YSumdevPtr, 
    void *YSqSumdevPtr, 
    void *scaledevPtr, 
    void *biasdevPtr, 
    void *in_meandevPtr, 
    void *in_vardevPtr, 
    void *out_meandevPtr, 
    void *out_vardevPtr,
    void *saved_meandevPtr, 
    void *saved_inv_vardevPtr, 
    void *eq_scaledevPtr, 
    void *eq_biasdevPtr,

    double epsilon_val,
    double exponential_decay_factor,
    int64_t accumCnt_val
) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(perChannelSum, stride, 4, CUDNN_TENSOR_NHWC);

        auto tensor_create = [&stride, &perChannelSum](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, perChannelSum)
                   .setStrides(4, stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .build();
        };

        auto sumTensor           = tensor_create(CUDNN_DATA_FLOAT, 100);
        auto sqSumTensor         = tensor_create(CUDNN_DATA_FLOAT, 101);
        auto scaleTensor         = tensor_create(CUDNN_DATA_FLOAT, 102);
        auto biasTensor          = tensor_create(CUDNN_DATA_FLOAT, 103);
        auto inMeanTensor        = tensor_create(CUDNN_DATA_FLOAT, 104);
        auto inVarTensor         = tensor_create(CUDNN_DATA_FLOAT, 105);
        auto outMeanTensor       = tensor_create(CUDNN_DATA_FLOAT, 106);
        auto outVarTensor        = tensor_create(CUDNN_DATA_FLOAT, 107);
        auto savedMeanTensor     = tensor_create(CUDNN_DATA_FLOAT, 108);
        auto savedInvVarTensor   = tensor_create(CUDNN_DATA_FLOAT, 109);
        auto outEqScaleTensor    = tensor_create(CUDNN_DATA_FLOAT, 200);
        auto outEqBiasTensor     = tensor_create(CUDNN_DATA_FLOAT, 201);

        int64_t epsilon_stride[4];
        generateStrides(epsilon, epsilon_stride, 4, CUDNN_TENSOR_NHWC);
        auto scalar_tensor_create = [&epsilon_stride, &epsilon](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, epsilon)
                   .setStrides(4, epsilon_stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .setByValue(true)
                   .build();
        };

        auto epsilonTensor       = scalar_tensor_create(CUDNN_DATA_DOUBLE, 300);
        auto expDecayTensor      = scalar_tensor_create(CUDNN_DATA_DOUBLE, 301);
        auto accumCountTensor    = scalar_tensor_create(CUDNN_DATA_INT64,  302);

        //Create a Finalize node
        auto finalize_stat_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
                    .setMathPrecision(CUDNN_DATA_FLOAT)
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
        auto opGraph = cudnn_frontend::OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = 
            cudnn_frontend::get_heuristics_list<2>({"heuristics_instant" 
            , "heuristics_fallback"
            }, opGraph,::allowAll, filtered_configs, true);
        
        std::cout << "get_heuristics_list Statuses: ";
        for (auto i = 0 ; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }

        void* data_ptrs[15] = {YSumdevPtr, YSqSumdevPtr, scaledevPtr, biasdevPtr, 
                               in_meandevPtr, in_vardevPtr, out_meandevPtr, out_vardevPtr,
                               saved_meandevPtr, saved_inv_vardevPtr, eq_scaledevPtr, eq_biasdevPtr,
                               &epsilon_val, &exponential_decay_factor, &accumCountTensor};
        int64_t uids[15]    = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 200, 201, 300, 301, 302};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
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
        
    } catch (cudnn_frontend::cudnnException &e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
#if (CUDNN_VERSION >= 8303)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif   
    }
}
