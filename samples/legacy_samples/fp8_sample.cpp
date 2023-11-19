#include "fp8_sample.h"
#include <cudnn_frontend.h>
#include "../utils/error_util.h"

using namespace cudnn_frontend;

ExecutionPlan_v8
get_exec_plan_from_heuristics(OperationGraph_v8&& opGraph, cudnnHandle_t handle) {
    auto heuristics = EngineHeuristicsBuilder().setOperationGraph(opGraph).setHeurMode(CUDNN_HEUR_MODE_INSTANT).build();

    auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    auto plan_builder = [&]() -> ExecutionPlan {
        for (auto& ecfg : engine_config) {
            try {
                auto plan = ExecutionPlanBuilder().setHandle(handle).setEngineConfig(ecfg, opGraph.getTag()).build();
                return plan;
            } catch (cudnnException& e) {
                continue;
            }
        }
        return ExecutionPlanBuilder().setHandle(handle).setEngineConfig(engine_config[0], opGraph.getTag()).build();
    };

    return plan_builder();
}

#if (CUDNN_VERSION >= 8600)
void
run_fp8_conv_scale(int64_t* x_dim,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   int64_t* scale_dim,
                   cudnnDataType_t dataType,
                   int convDim,
                   int64_t* conv_padA,
                   int64_t* conv_dilationA,
                   int64_t* conv_strideA,
                   void* devPtrX,
                   void* devPtrW,
                   void* devPtrY,
                   void* devPtrScale) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, CUDNN_STATUS_ARCH_MISMATCH, "run_fp8_conv_scale: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        ::generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        ::generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        ::generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('y')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();

        auto afterScaleTensor =
            TensorBuilder().cloneFrom(afterConvTensor, 'a').setVirtual(false).setDataType(dataType).build();

        ::generateStrides(scale_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto scaleTensor = TensorBuilder()
                               .setDim(4, scale_dim)
                               .setStrides(4, stride)
                               .setId('s')  // after conv
                               .setAlignment(16)
                               .setDataType(CUDNN_DATA_FLOAT)
                               .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << scaleTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = PointWiseDescBuilder().setMode(CUDNN_POINTWISE_MUL).setMathPrecision(CUDNN_DATA_FLOAT).build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = ConvDescBuilder()
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
        auto conv_op = OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto scale_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(conv_op.getOutputTensor())
                            .setbDesc(scaleTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution scale
        std::array<Operation const*, 2> ops = {&conv_op, &scale_op};

        auto opGraph = OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();

        auto plan = get_exec_plan_from_heuristics(std::move(opGraph), handle_);
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrY, devPtrScale};
        int64_t uids[]    = {'x', 'w', 'a', 's'};
        auto variantPack  = VariantPackBuilder()
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

        throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnnException& e) {
        // this example is only for Hopper cards
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 9 &&
            (e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED || e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH)) {
            std::cout << "Fusion with fp8 inputs is only supported on Hopper or later" << std::endl;
            return;
        }

        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_fp8_conv_descale_descale_amax_scale(int64_t* x_dim,
                                        int64_t* w_dim,
                                        int64_t* y_dim,
                                        int64_t* r_dim,
                                        int64_t* scale_dim,
                                        cudnnDataType_t dataType,
                                        int convDim,
                                        int64_t* conv_padA,
                                        int64_t* conv_dilationA,
                                        int64_t* conv_strideA,
                                        void* devPtrX,
                                        void* devPtrW,
                                        void* devPtrR,
                                        void* devPtrOutput,
                                        void* devPtrDescale1,
                                        void* devPtrDescale2,
                                        void* devPtrScale) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_ARCH_MISMATCH,
                "run_fp8_conv_descale_descale_amax_scale: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        ::generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        ::generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        ::generateStrides(r_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto amaxTensor = TensorBuilder()
                              .setDim(4, r_dim)
                              .setStrides(4, stride)
                              .setId('r')  // output
                              .setAlignment(16)
                              .setDataType(CUDNN_DATA_FLOAT)
                              .build();

        ::generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvTensor = TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('y')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();

        auto afterDescale1Tensor = TensorBuilder().cloneFrom(afterConvTensor, 'a').build();

        auto afterDescale2Tensor = TensorBuilder().cloneFrom(afterConvTensor, 'b').build();

        auto fp8OutputTensor =
            TensorBuilder().cloneFrom(afterConvTensor, 'c').setVirtual(false).setDataType(dataType).build();

        ::generateStrides(scale_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto descaleTensor1 = TensorBuilder()
                                  .setDim(4, scale_dim)
                                  .setStrides(4, stride)
                                  .setId('s')
                                  .setAlignment(16)
                                  .setDataType(CUDNN_DATA_FLOAT)
                                  .build();

        auto descaleTensor2 = TensorBuilder().cloneFrom(descaleTensor1, 't').build();

        auto scaleTensor = TensorBuilder().cloneFrom(descaleTensor1, 'u').build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << scaleTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = PointWiseDescBuilder().setMode(CUDNN_POINTWISE_MUL).setMathPrecision(CUDNN_DATA_FLOAT).build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the reduction descriptor
        auto redunctionDesc =
            ReductionDescBuilder().setMathPrecision(CUDNN_DATA_FLOAT).setReductionOp(CUDNN_REDUCE_TENSOR_AMAX).build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = ConvDescBuilder()
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
        auto conv_op = OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto descale_op1 = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(afterConvTensor)
                               .setbDesc(descaleTensor1)
                               .setyDesc(afterDescale1Tensor)
                               .setpwDesc(scaleDesc)
                               .build();
        std::cout << descale_op1.describe() << std::endl;

        auto descale_op2 = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(afterDescale1Tensor)
                               .setbDesc(descaleTensor2)
                               .setyDesc(afterDescale2Tensor)
                               .setpwDesc(scaleDesc)
                               .build();
        std::cout << descale_op2.describe() << std::endl;

        auto scale_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(afterDescale2Tensor)
                            .setbDesc(scaleTensor)
                            .setyDesc(fp8OutputTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a reduction add Node.
        auto reduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterDescale2Tensor)
                                .setyDesc(amaxTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution descale descale amax scale
        std::array<Operation const*, 5> ops = {&conv_op, &descale_op1, &descale_op2, &scale_op, &reduction_op};

        auto opGraph = OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();

        auto plan = get_exec_plan_from_heuristics(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrR, devPtrDescale1, devPtrDescale2, devPtrScale, devPtrOutput};
        int64_t uids[]    = {'x', 'w', 'r', 's', 't', 'u', 'c'};
        auto variantPack  = VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudaErr(cudaDeviceSynchronize());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        checkCudnnErr(cudnnDestroy(handle_));

        throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnnException& e) {
        // this example is only for Hopper cards
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 9 &&
            (e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED || e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH)) {
            std::cout << "Fusion with fp8 inputs is only supported on Hopper or later" << std::endl;
            return;
        }

        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_tranpose_scale_convert_fp16_fp8_amax(int64_t* x_dim,
                                         int64_t* y_dim,
                                         int64_t* r_dim,
                                         int64_t* scale_dim,
                                         cudnnDataType_t dataType,
                                         void* devPtrX,
                                         void* devPtrR,
                                         void* devPtrOutput,
                                         void* devPtrScale) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_ARCH_MISMATCH,
                "run_tranpose_scale_convert_fp16_fp8_amax: Sample requires Ampere or above GPU");
        }

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        ::generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto xTensor = TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)              // 16B alignment is needed to run a tensor core engine
                           .setDataType(CUDNN_DATA_HALF)  // Half as input
                           .build();

        ::generateStrides(scale_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto scaleTensor = TensorBuilder()
                               .setDim(4, scale_dim)
                               .setStrides(4, stride)
                               .setId('s')
                               .setAlignment(16)
                               .setDataType(CUDNN_DATA_FLOAT)
                               .build();

        ::generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterScaleTensor = TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStrides(4, stride)
                                    .setId('a')  // after transpose + convert
                                    .setAlignment(16)
                                    .setDataType(CUDNN_DATA_FLOAT)  // Transpose + convert to FP8
                                    .setVirtual()
                                    .build();

        // Tranposed from NWHC to CHWN
        ::generate4dTransposeStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterConvertTensor = TensorBuilder()
                                      .setDim(4, y_dim)
                                      .setStrides(4, stride)
                                      .setId('y')  // after transpose + convert
                                      .setAlignment(16)
                                      .setDataType(dataType)  // Transpose + convert to FP8
                                      .build();

        ::generateStrides(r_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto amaxTensor = TensorBuilder()
                              .setDim(4, r_dim)
                              .setStrides(4, stride)
                              .setId('r')  // output
                              .setAlignment(16)
                              .setDataType(CUDNN_DATA_FLOAT)
                              .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << scaleTensor.describe() << std::endl;
        std::cout << afterConvertTensor.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = PointWiseDescBuilder().setMode(CUDNN_POINTWISE_MUL).setMathPrecision(CUDNN_DATA_FLOAT).build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the convert descriptor
        auto identityDesc =
            PointWiseDescBuilder().setMode(CUDNN_POINTWISE_IDENTITY).setMathPrecision(CUDNN_DATA_FLOAT).build();
        std::cout << identityDesc.describe() << std::endl;

        // Define the reduction descriptor
        auto redunctionDesc =
            ReductionDescBuilder().setMathPrecision(CUDNN_DATA_FLOAT).setReductionOp(CUDNN_REDUCE_TENSOR_AMAX).build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto scale_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(xTensor)
                            .setbDesc(scaleTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create transpose + convert node
        auto convert_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                              .setxDesc(afterScaleTensor)
                              .setyDesc(afterConvertTensor)
                              .setpwDesc(identityDesc)
                              .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a reduction add Node.
        auto reduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(xTensor)
                                .setyDesc(amaxTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is scale transpose amax
        std::array<Operation const*, 3> ops = {&scale_op, &convert_op, &reduction_op};

        auto opGraph = OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();

        auto plan = get_exec_plan_from_heuristics(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrX, devPtrR, devPtrScale, devPtrOutput};
        int64_t uids[]    = {'x', 'r', 's', 'y'};
        auto variantPack  = VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        checkCudaErr(cudaDeviceSynchronize());
        checkCudnnErr(cudnnDestroy(handle_));

        throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnnException& e) {
        // this example is only for Hopper cards
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 9 &&
            (e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED || e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH)) {
            std::cout << "Fusion with fp8 inputs is only supported on Hopper or later" << std::endl;
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_fp8_dgrad_descale_descale_amax_scale(int64_t* dx_dim,
                                         int64_t* w_dim,
                                         int64_t* dy_dim,
                                         int64_t* r_dim,
                                         int64_t* scale_dim,
                                         cudnnDataType_t dataType,
                                         int convDim,
                                         int64_t* conv_padA,
                                         int64_t* conv_dilationA,
                                         int64_t* conv_strideA,
                                         void* devPtrdX,
                                         void* devPtrW,
                                         void* devPtrR,
                                         void* devPtrdY,
                                         void* devPtrDescale1,
                                         void* devPtrDescale2,
                                         void* devPtrScale) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_ARCH_MISMATCH,
                "run_fp8_dgrad_descale_descale_amax_scale: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];

        ::generateStrides(dy_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto dyTensor = TensorBuilder()
                            .setDim(4, dy_dim)
                            .setStrides(4, stride)
                            .setId('y')  // after conv
                            .setAlignment(16)
                            .setDataType(dataType)
                            .build();

        ::generate4dTransposeStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        ::generateStrides(dx_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto dxTensor = TensorBuilder()
                            .setDim(4, dx_dim)
                            .setStrides(4, stride)
                            .setId('x')
                            .setVirtual()      // after dgrad is virtual
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(CUDNN_DATA_FLOAT)
                            .build();

        ::generateStrides(scale_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto dyDescaleTensor = TensorBuilder()
                                   .setDim(4, scale_dim)
                                   .setStrides(4, stride)
                                   .setId('s')
                                   .setAlignment(16)
                                   .setDataType(CUDNN_DATA_FLOAT)
                                   .build();

        auto afterDescale1Tensor = TensorBuilder().cloneFrom(dxTensor, 'a').build();

        auto wDescaleTensor = TensorBuilder().cloneFrom(dyDescaleTensor, 't').build();

        auto afterDescale2Tensor = TensorBuilder().cloneFrom(dxTensor, 'b').build();

        auto dxScaleTensor = TensorBuilder().cloneFrom(dyDescaleTensor, 'u').build();

        auto fp8OutputTensor = TensorBuilder().cloneFrom(dxTensor, 'c').setVirtual(false).setDataType(dataType).build();

        ::generateStrides(r_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto amaxTensor = TensorBuilder()
                              .setDim(4, r_dim)
                              .setStrides(4, stride)
                              .setId('r')  // output
                              .setAlignment(16)
                              .setDataType(CUDNN_DATA_FLOAT)
                              .build();

        std::cout << dxTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << dxScaleTensor.describe() << std::endl;
        std::cout << dyTensor.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = PointWiseDescBuilder().setMode(CUDNN_POINTWISE_MUL).setMathPrecision(CUDNN_DATA_FLOAT).build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the reduction descriptor
        auto redunctionDesc =
            ReductionDescBuilder().setMathPrecision(CUDNN_DATA_FLOAT).setReductionOp(CUDNN_REDUCE_TENSOR_AMAX).build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = ConvDescBuilder()
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
        auto dgrad_op = OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                            .setdyDesc(dyTensor)
                            .setwDesc(wTensor)
                            .setdxDesc(dxTensor)
                            .setcDesc(convDesc)
                            .setAlpha(alpha)
                            .setBeta(beta)
                            .build();
        std::cout << dgrad_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        auto descale_op1 = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(dxTensor)
                               .setbDesc(dyDescaleTensor)
                               .setyDesc(afterDescale1Tensor)
                               .setpwDesc(scaleDesc)
                               .build();
        std::cout << descale_op1.describe() << std::endl;

        auto descale_op2 = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(afterDescale1Tensor)
                               .setbDesc(wDescaleTensor)
                               .setyDesc(afterDescale2Tensor)
                               .setpwDesc(scaleDesc)
                               .build();
        std::cout << descale_op2.describe() << std::endl;

        auto scale_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(afterDescale2Tensor)
                            .setbDesc(dxScaleTensor)
                            .setyDesc(fp8OutputTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a reduction add Node.
        auto reduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterDescale2Tensor)
                                .setyDesc(amaxTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is dgrad descale descale amax scale
        std::array<Operation const*, 5> ops = {&dgrad_op, &descale_op1, &descale_op2, &scale_op, &reduction_op};

        auto opGraph = OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();

        auto plan = get_exec_plan_from_heuristics(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
        }
        void* data_ptrs[] = {devPtrdX, devPtrW, devPtrR, devPtrDescale1, devPtrDescale2, devPtrScale, devPtrdY};
        int64_t uids[]    = {'c', 'w', 'r', 's', 't', 'u', 'y'};
        auto variantPack  = VariantPackBuilder()
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

        throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnnException& e) {
        // this example is only for Hopper cards
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 9 &&
            (e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED || e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH)) {
            std::cout << "Fusion with fp8 inputs is only supported on Hopper or later" << std::endl;
            return;
        }

        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}
#endif
