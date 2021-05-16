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

#if (CUDNN_VERSION >= 8200)
bool
isRuntimeCompilation(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION>(engine_config);
}
#endif

void
run_conv_scale_bias_add_relu(int64_t* x_dim,
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

        // How many engines support this operation graph ?
        auto total_engines = opGraph.getEngineCount();
        std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }

        // Create the requisite engine config
        auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
        std::cout << engine_config.describe() << std::endl;
#if (CUDNN_VERSION >= 8200)
        if (isRuntimeCompilation(engine_config.get_raw_desc())) {
            std::cout << "The engine is runtime compilation" << std::endl;
        } else {
            std::cout << "The engine is not runtime compilation" << std::endl;
        }
#endif

        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();

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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
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

        // How many engines support this operation graph ?
        auto total_engines = opGraph.getEngineCount();
        std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
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

        // How many engines support this operation graph ?
        auto total_engines = opGraph.getEngineCount();
        std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
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

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
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

        // How many engines support this operation graph ?
        auto total_engines = opGraph.getEngineCount();
        std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
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

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
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

        // How many engines support this operation graph ?
        auto total_engines = opGraph.getEngineCount();
        std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}
