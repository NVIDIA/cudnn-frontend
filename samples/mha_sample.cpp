/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "cudnn_frontend.h"
#include "cudnn_frontend_utils.h"
#include "error_util.h"
#include "mha_sample.h"

using namespace cudnn_frontend;

ExecutionPlan_v8
get_plan_from_heuristics(OperationGraph_v8 &opGraph, cudnnHandle_t handle) {
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(opGraph)
                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                          .build();

    auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    auto plan_builder = [&]() -> cudnn_frontend::ExecutionPlan {
        for (auto &ecfg : engine_config) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(ecfg, opGraph.getTag())
                                .build();
                return plan;
            } catch (cudnn_frontend::cudnnException &e) {
                continue;
            }
        }
        return cudnn_frontend::ExecutionPlanBuilder()
                   .setHandle(handle)
                   .setEngineConfig(engine_config[0], opGraph.getTag())
                   .build();
    };

    return plan_builder();
}

class MHAParams {
   public:
    friend class MHA_intputProjLayer;
    friend class MHA_outputProjLayer;
    friend class MHA_attentionLayer;
    friend class MHAParamsBuilder;

    MHAParams(MHAParams &&from) = default;
    MHAParams &
    operator= (MHAParams &&from) = default;

    ~MHAParams() = default;

   private:
    int64_t inputSize  = 0;
    int64_t outputSize = 0;
    int64_t headSize   = 0;
    int64_t numHeads   = 1;
    int64_t seqLength  = 1;
    int64_t batchSize  = 1;

    cudnnDataType_t dataType = CUDNN_DATA_HALF;
    cudnnDataType_t mathPrec = CUDNN_DATA_FLOAT;

    MHAParams()                  = default;
    MHAParams(MHAParams const &) = delete;
    MHAParams &
    operator=(MHAParams const &) = delete;
};

///
/// MHAParamsBuilder Class
/// Helper class used to build MHAParams class
class MHAParamsBuilder {
   public:
    /** @defgroup MHAParamsBuilder
     *  Set individual property of MHAParams class
     *  @{
     */

    //! Set input vector size
    auto
    setInputSize(int64_t inputSize_) -> MHAParamsBuilder & {
        mhaParams.inputSize = inputSize_;
        return *this;
    }
    /** @} */

    //! Set output vector size
    auto
    setOutputSize(int64_t outputSize_) -> MHAParamsBuilder & {
        mhaParams.outputSize = outputSize_;
        return *this;
    }
    /** @} */

    //! Set attention head size
    auto
    setHeadSize(int64_t headSize_) -> MHAParamsBuilder & {
        mhaParams.headSize = headSize_;
        return *this;
    }
    /** @} */

    //! Set number of heads
    auto
    setNumHeads(int64_t numHeads_) -> MHAParamsBuilder & {
        mhaParams.numHeads = numHeads_;
        return *this;
    }
    /** @} */

    //! Set input sequence length
    auto
    setSeqLength(int64_t seqLength_) -> MHAParamsBuilder & {
        mhaParams.seqLength = seqLength_;
        return *this;
    }
    /** @} */

    //! Set input batch size
    auto
    setBatchSize(int64_t batchSize_) -> MHAParamsBuilder & {
        mhaParams.batchSize = batchSize_;
        return *this;
    }
    /** @} */

    //! Set input data type
    auto
    setDataType(cudnnDataType_t dataType_) -> MHAParamsBuilder & {
        mhaParams.dataType = dataType_;
        return *this;
    }
    /** @} */

    //! Set math precision
    auto
    setMathPrec(cudnnDataType_t mathPrec_) -> MHAParamsBuilder & {
        mhaParams.mathPrec = mathPrec_;
        return *this;
    }
    /** @} */

    //! constructs the MHAParams by calling the cudnn API
    //! Throws the appropriate error message
    MHAParams &&
    build() {
        if (mhaParams.inputSize <= 0) {
            set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_BAD_PARAM,
                "MHA: Check and Set the input vector size to valid value");
            return std::move(mhaParams);
        }
        if (mhaParams.outputSize <= 0) {
            set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_BAD_PARAM,
                "MHA: Check and Set the output vector size to valid value");
            return std::move(mhaParams);
        }
        if (mhaParams.headSize <= 0) {
            set_error_and_throw_exception(
                nullptr,
                CUDNN_STATUS_BAD_PARAM,
                "MHA: Check and Set the head size to valid value");
            return std::move(mhaParams);
        }

        return std::move(mhaParams);
    }

    explicit MHAParamsBuilder()                = default;
    ~MHAParamsBuilder()                        = default;
    MHAParamsBuilder(MHAParamsBuilder &&)      = delete;
    MHAParamsBuilder(MHAParamsBuilder const &) = delete;
    MHAParamsBuilder &
    operator=(MHAParamsBuilder const &) = delete;

   private:
    MHAParams mhaParams;
};

class MHA_intputProjLayer {
   public:
    MHA_intputProjLayer(MHA_intputProjLayer &&from) = default;
    MHA_intputProjLayer &
    operator= (MHA_intputProjLayer &&from) = default;

    ~MHA_intputProjLayer() = default;

    MHA_intputProjLayer(cudnnHandle_t handle, MHAParams &mhaParams) {
        const int64_t inputSize  = mhaParams.inputSize;
        const int64_t headSize   = mhaParams.headSize;
        const int64_t numHeads   = mhaParams.numHeads;
        const int64_t seqLength  = mhaParams.seqLength;
        const int64_t batchSize  = mhaParams.batchSize;

        const cudnnDataType_t dataType = mhaParams.dataType;

        const int64_t wDim[3]    = {3, inputSize, headSize * numHeads};
        const int64_t wStride[3] = {headSize * numHeads * inputSize, headSize * numHeads, 1};

        auto wMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, wDim)
                                 .setStride(3, wStride)
                                 .setId('W')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        // Create tensor descriptor for input matrix
        const int64_t iDim[3]    = {1, seqLength * batchSize, inputSize};
        const int64_t iStride[3] = {inputSize * seqLength * batchSize, inputSize, 1};

        auto iMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, iDim)
                                 .setStride(3, iStride)
                                 .setId('i')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        // Create tensor descriptor for QKVBias Matrix
        const int64_t bDim[3]    = {3, 1, headSize * numHeads};
        const int64_t bStride[3] = {headSize * numHeads, headSize * numHeads, 1};

        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(3, bDim)
                           .setStride(3, bStride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        // Create tensor descriptors for embeddings Q, K, V
        const int64_t oDim[3]    = {3, seqLength * batchSize, headSize * numHeads};
        const int64_t oStride[3] = {headSize * numHeads * seqLength * batchSize, headSize * numHeads, 1};

        auto beforeBiasMatrixTensor = cudnn_frontend::TensorBuilder()
                                          .setDim(3, oDim)
                                          .setStride(3, oStride)
                                          .setId('a')
                                          .setAlignment(16)
                                          .setVirtual()
                                          .setDataType(dataType)
                                          .build();

        auto oMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, oDim)
                                 .setStride(3, oStride)
                                 .setId('o')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        // Define the matmul descriptor for Q, K, V = W * Input
        auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .build();
        // Define the Bias descriptor for Q, K, V = W * Input + Bias
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        // Create a matmul Node for Q, K, V = W * Input
        auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(iMatrixTensor)
                            .setbMatDesc(wMatrixTensor)
                            .setcMatDesc(beforeBiasMatrixTensor)
                            .setmatmulDesc(matmulDesc)
                            .build();
        // Create a Bias Node for Q, K, V = W * Input + Bias
        auto biasOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(matmulOp.getOutputTensor())
                          .setbDesc(bTensor)
                          .setyDesc(oMatrixTensor)
                          .setpwDesc(biasDesc)
                          .build();
        // Create an Operation Graphs
        std::array<cudnn_frontend::Operation const*, 2> ops = {&matmulOp, &biasOp};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_plan_from_heuristics(opGraph, handle);
        inputProjLayerPlan = std::make_shared<cudnn_frontend::ExecutionPlan>(std::move(plan));

        std::cout << "[INFO] Execution Plan tag for input projection layer: " << inputProjLayerPlan->getTag() << std::endl;
    };

    cudnnStatus_t
    execute(cudnnHandle_t handle, 
            void const *devPtrIn,
            void const *devPtrWeight,
            void const *devPtrBias,
            void *devPtrQKV) {
        void* data_ptrs[] = {const_cast<void*>(devPtrIn),
                             const_cast<void*>(devPtrWeight),
                             const_cast<void*>(devPtrBias),
                             devPtrQKV};
        int64_t uids[] = {'i', 'W', 'b', 'o'};

        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();

        return cudnnBackendExecute(handle, inputProjLayerPlan->get_raw_desc(), variantPack.get_raw_desc());
    };

   private:
    std::shared_ptr<cudnn_frontend::ExecutionPlan> inputProjLayerPlan;

    MHA_intputProjLayer()                            = delete;
    MHA_intputProjLayer(MHA_intputProjLayer const &) = delete;
    MHA_intputProjLayer &
    operator=(MHA_intputProjLayer const &) = delete;
};

class MHA_outputProjLayer {
   public:
    MHA_outputProjLayer(MHA_outputProjLayer &&from) = default;
    MHA_outputProjLayer &
    operator= (MHA_outputProjLayer &&from) = default;

    ~MHA_outputProjLayer() = default;

    MHA_outputProjLayer(cudnnHandle_t handle, MHAParams &mhaParams) {
        const int64_t outputSize = mhaParams.outputSize;
        const int64_t headSize   = mhaParams.headSize;
        const int64_t numHeads   = mhaParams.numHeads;
        const int64_t seqLength  = mhaParams.seqLength;
        const int64_t batchSize  = mhaParams.batchSize;

        const cudnnDataType_t dataType = mhaParams.dataType;

        // Create tensor descriptor for OWeight Matrix
        const int64_t wDim[3]    = {1, headSize * numHeads, outputSize};
        const int64_t wStride[3] = {outputSize * headSize * numHeads, outputSize, 1};

        auto wMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, wDim)
                                 .setStride(3, wStride)
                                 .setId('W')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        // Create tensor descriptor for Y^T matrix
        const int64_t iDim[3]    = {1, seqLength * batchSize, headSize * numHeads};
        const int64_t iStride[3] = {seqLength * batchSize * headSize * numHeads, headSize * numHeads, 1};

        auto iMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, iDim)
                                 .setStride(3, iStride)
                                 .setId('i')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        // Create tensor descriptor for QKVBias Matrix
        const int64_t bDim[3]    = {1, 1, outputSize};
        const int64_t bStride[3] = {outputSize, outputSize, 1};

        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(3, bDim)
                           .setStride(3, bStride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        // Create tensor descriptor for Output before bias add
        const int64_t oDim[3]    = {1, seqLength * batchSize, outputSize};
        const int64_t oStride[3] = {seqLength * batchSize * outputSize, outputSize, 1};

        auto beforeBiasMatrixTensor = cudnn_frontend::TensorBuilder()
                                          .setDim(3, oDim)
                                          .setStride(3, oStride)
                                          .setId('a')
                                          .setAlignment(16)
                                          .setVirtual()
                                          .setDataType(CUDNN_DATA_FLOAT)
                                          .build();
        auto oMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, oDim)
                                 .setStride(3, oStride)
                                 .setId('o')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        // Define the matmul descriptor
        auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .build();
        // Define the Bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
        // Create a matmul Node
        auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(iMatrixTensor)
                            .setbMatDesc(wMatrixTensor)
                            .setcMatDesc(beforeBiasMatrixTensor)
                            .setmatmulDesc(matmulDesc)
                            .build();
        // Create a Bias Node
        auto biasOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(matmulOp.getOutputTensor())
                          .setbDesc(bTensor)
                          .setyDesc(oMatrixTensor)
                          .setpwDesc(biasDesc)
                          .build();
        // Create an Operation Graphs
        std::array<cudnn_frontend::Operation const*, 2> ops = {&matmulOp, &biasOp};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_plan_from_heuristics(opGraph, handle);
        outputProjLayerPlan = std::make_shared<cudnn_frontend::ExecutionPlan>(std::move(plan));

        std::cout << "[INFO] Execution Plan tag for output projection layer: " << outputProjLayerPlan->getTag() << std::endl;
    };

    cudnnStatus_t
    execute(cudnnHandle_t handle, 
            void const *devPtrIn,
            void const *devPtrWeight,
            void const *devPtrBias,
            void *devPtrOut) {
        void* data_ptrs[] = {const_cast<void*>(devPtrIn),
                              const_cast<void*>(devPtrWeight),
                              const_cast<void*>(devPtrBias),
                              devPtrOut};
        int64_t uids[] = {'i', 'W', 'b', 'o'};

        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();

        return cudnnBackendExecute(handle, outputProjLayerPlan->get_raw_desc(), variantPack.get_raw_desc());
    };

   private:
    std::shared_ptr<cudnn_frontend::ExecutionPlan> outputProjLayerPlan;

    MHA_outputProjLayer()                             = delete;
    MHA_outputProjLayer(MHA_outputProjLayer const &) = delete;
    MHA_outputProjLayer &
    operator=(MHA_outputProjLayer const &) = delete;
};

class MHA_attentionLayer {
   public:
    MHA_attentionLayer(MHA_attentionLayer &&from) = default;
    MHA_attentionLayer &
    operator= (MHA_attentionLayer &&from) = default;

    ~MHA_attentionLayer() = default;

    MHA_attentionLayer(cudnnHandle_t handle, MHAParams &mhaParams) {
        (void)handle;
        (void)mhaParams;
#if (CUDNN_VERSION >= 8303)
        const int64_t headSize   = mhaParams.headSize;
        const int64_t numHeads   = mhaParams.numHeads;
        const int64_t seqLength  = mhaParams.seqLength;
        const int64_t batchSize  = mhaParams.batchSize;

        const cudnnDataType_t dataType = mhaParams.dataType;
        {

            // Create tensor descriptor for embedding Q
            const int64_t qDim[3]    = {batchSize * numHeads, headSize, seqLength};
            const int64_t qStride[3] = {headSize, 1, headSize * numHeads * batchSize};

            auto qMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, qDim)
                                     .setStride(3, qStride)
                                     .setId('q')
                                     .setAlignment(16)
                                     .setDataType(dataType)
                                     .build();
            // Create tensor descriptor for embedding K^T
            const int64_t kDim[3]    = {batchSize * numHeads, seqLength, headSize};
            const int64_t kStride[3] = {headSize, headSize * numHeads * batchSize, 1};

            auto kMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, kDim)
                                     .setStride(3, kStride)
                                     .setId('k')
                                     .setAlignment(16)
                                     .setDataType(dataType)
                                     .build();

            // Create tensor descriptor for S = K^T * Q
            const int64_t sDim[3]    = {batchSize * numHeads, seqLength, seqLength};
            const int64_t sStride[3] = {seqLength * seqLength, seqLength, 1};

            auto sMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, sDim)
                                     .setStride(3, sStride)
                                     .setId('s')
                                     .setAlignment(16)
                                     .setVirtual()
                                     .setDataType(CUDNN_DATA_FLOAT)
                                     .build();
            // Create tensor descriptor for Z = softmaxScaler * S
            auto zMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, sDim)
                                     .setStride(3, sStride)
                                     .setId('z')
                                     .setAlignment(16)
                                     .setDataType(dataType)
                                     .build();
            // Create tensor descriptor for E = exp(Z)
            auto eMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, sDim)
                                     .setStride(3, sStride)
                                     .setId('e')
                                     .setAlignment(16)
                                     .setVirtual()
                                     .setDataType(CUDNN_DATA_FLOAT)
                                     .build();
            // Create tensor descriptor for softmaxScaler
            const int64_t scalerDim[3]    = {1, 1, 1};
            const int64_t scalerStride[3] = {1, 1, 1};

            auto softmaxScalerTensor = cudnn_frontend::TensorBuilder()
                                           .setDim(3, scalerDim)
                                           .setStride(3, scalerStride)
                                           .setId('m')
                                           .setAlignment(16)
                                           .setDataType(CUDNN_DATA_FLOAT)
                                           .build();
            // Create tensor descriptor for Col-Reduction of E
            const int64_t cDim[3]    = {batchSize * numHeads, 1, seqLength};
            const int64_t cStride[3] = {seqLength, seqLength, 1};

            auto cTensor = cudnn_frontend::TensorBuilder()
                               .setDim(3, cDim)
                               .setStride(3, cStride)
                               .setId('c')
                               .setAlignment(16)
                               .setDataType(CUDNN_DATA_FLOAT)
                               .build();
            // Define the matmul descriptor for S = (Q^T * K)
            auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
            // Define the scale descriptor for S' = softmaxScaler * S
            auto softmaxScalerDesc = cudnn_frontend::PointWiseDescBuilder()
                                         .setMode(CUDNN_POINTWISE_MUL)
                                         .setComputeType(CUDNN_DATA_FLOAT)
                                         .build();
            // Define the activation descriptor for E = exp(S')
            auto expDesc = cudnn_frontend::PointWiseDescBuilder()
                               .setMode(CUDNN_POINTWISE_EXP)
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .build();
            // Define the reduction descriptor
            auto colRedunctionDesc = cudnn_frontend::ReductionDescBuilder()
                                         .setComputeType(CUDNN_DATA_FLOAT)
                                         .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                         .build();
            // Create a matmul Node for S = (Q^T * K)
            auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                                .setaMatDesc(kMatrixTensor)
                                .setbMatDesc(qMatrixTensor)
                                .setcMatDesc(sMatrixTensor)
                                .setmatmulDesc(matmulDesc)
                                .build();
            // Create a scale Node for S' = softmaxScaler * S
            auto softmaxScalerOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                       .setxDesc(matmulOp.getOutputTensor())
                                       .setbDesc(softmaxScalerTensor)
                                       .setyDesc(zMatrixTensor)
                                       .setpwDesc(softmaxScalerDesc)
                                       .build();
            // Create a EXP Node for E = exp(S')
            auto expOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(softmaxScalerOp.getOutputTensor())
                             .setyDesc(eMatrixTensor)
                             .setpwDesc(expDesc)
                             .build();
            // Create a row-reduction Node
            auto colReductionOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                       .setxDesc(expOp.getOutputTensor())
                                       .setyDesc(cTensor)
                                       .setreductionDesc(colRedunctionDesc)
                                       .build();

            // Create an Operation Graphs
            std::array<cudnn_frontend::Operation const*, 4> ops = {&matmulOp, &softmaxScalerOp, &expOp, &colReductionOp};

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                               .setHandle(handle)
                               .setOperationGraph(ops.size(), ops.data())
                               .build();

            auto plan = get_plan_from_heuristics(opGraph, handle);
            attentionLayerPlan0 = std::make_shared<cudnn_frontend::ExecutionPlan>(std::move(plan));
        }

        {
            // Create tensor descriptor for embedding V^T
            const int64_t vDim[3]    = {batchSize * numHeads, seqLength, headSize};
            const int64_t vStride[3] = {headSize, headSize * numHeads * batchSize, 1};

            auto vMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, vDim)
                                     .setStride(3, vStride)
                                     .setId('v')
                                     .setAlignment(16)
                                     .setDataType(dataType)
                                     .build();
            // Create tensor descriptor for E^T
            const int64_t eDim[3]    = {batchSize * numHeads, seqLength, seqLength};
            const int64_t eStride[3] = {seqLength * seqLength, 1, seqLength};

            auto sMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, eDim)
                                     .setStride(3, eStride)
                                     .setId('s')
                                     .setAlignment(16)
                                     .setDataType(dataType)
                                     .build();

            auto eMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, eDim)
                                     .setStride(3, eStride)
                                     .setId('e')
                                     .setAlignment(16)
                                     .setVirtual()
                                     .setDataType(CUDNN_DATA_FLOAT)
                                     .build();

            // Create tensor descriptor for P = Y / R
            auto pMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, eDim)
                                     .setStride(3, eStride)
                                     .setId('p')
                                     .setAlignment(16)
                                     .setVirtual()
                                     .setDataType(dataType)
                                     .build();

            // Create tensor descriptor for Y = E^T * V^T
            const int64_t yDim[3]    = {batchSize * numHeads, seqLength, headSize};
            const int64_t yStride[3] = {headSize, headSize * numHeads * batchSize, 1};

            auto yMatrixTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, yDim)
                                     .setStride(3, yStride)
                                     .setId('y')
                                     .setAlignment(16)
                                     .setDataType(dataType)
                                     .build();

            // Create tensor descriptor for Row-broadcast
            const int64_t rDim[3]    = {batchSize * numHeads, seqLength, 1};
            const int64_t rStride[3] = {seqLength, 1, seqLength};

            auto rTensor = cudnn_frontend::TensorBuilder()
                               .setDim(3, rDim)
                               .setStride(3, rStride)
                               .setId('r')
                               .setAlignment(16)
                               .setDataType(CUDNN_DATA_FLOAT)
                               .build();
            // Define the activation descriptor
            auto expDesc = cudnn_frontend::PointWiseDescBuilder()
                               .setMode(CUDNN_POINTWISE_EXP)
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .build();
            // Define the row-broadcast descriptor for P = Y / R
            auto rowBroadcastDesc = cudnn_frontend::PointWiseDescBuilder()
                                        .setMode(CUDNN_POINTWISE_DIV)
                                        .setComputeType(CUDNN_DATA_FLOAT)
                                        .build();
            // Define the matmul descriptor for Y = E * V^T
            auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();

            // Create a EXP Node for E = exp(S')
            auto expOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(sMatrixTensor)
                             .setyDesc(eMatrixTensor)
                             .setpwDesc(expDesc)
                             .build();
            // Create a row-broadcast Node for P = Y / R
            auto rowBroadcastOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                       .setxDesc(expOp.getOutputTensor())
                                       .setbDesc(rTensor)
                                       .setyDesc(pMatrixTensor)
                                       .setpwDesc(rowBroadcastDesc)
                                       .build();
            // Create a matmul Node for Y = E * V^T
            auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                                .setaMatDesc(rowBroadcastOp.getOutputTensor())
                                .setbMatDesc(vMatrixTensor)
                                .setcMatDesc(yMatrixTensor)
                                .setmatmulDesc(matmulDesc)
                                .build();

            // Create an Operation Graphs
            std::array<cudnn_frontend::Operation const*, 3> ops = {&expOp, &rowBroadcastOp, &matmulOp};

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                               .setHandle(handle)
                               .setOperationGraph(ops.size(), ops.data())
                               .build();

            auto plan = get_plan_from_heuristics(opGraph, handle);
            attentionLayerPlan1 = std::make_shared<cudnn_frontend::ExecutionPlan>(std::move(plan));
        }

        std::cout << "[INFO] Execution Plan tag for attention layer, part 0: " << attentionLayerPlan0->getTag() << std::endl;
        std::cout << "[INFO] Execution Plan tag for attention layer, part 1: " << attentionLayerPlan1->getTag() << std::endl;
#endif
    };

    cudnnStatus_t
    execute(cudnnHandle_t handle, 
            void const *devPtrQ,
            void const *devPtrK,
            void const *devPtrV,
            void const *devPtrScaler,
            void *devPtrE,
            void *devPtrR,
            void *devPtrX) {
        void* data_ptrs0[] = {const_cast<void *>(devPtrQ),
                              const_cast<void *>(devPtrK),
                              const_cast<void *>(devPtrE),
                              devPtrR,
                              const_cast<void *>(devPtrScaler)};
        int64_t uids0[] = {'q', 'k', 'z', 'c', 'm'};

        void* data_ptrs1[] = {devPtrE,
                              const_cast<void *>(devPtrV),
                              devPtrR,
                              devPtrX};
        int64_t uids1[] = {'s', 'v', 'r', 'y'};

        auto variantPack0 = cudnn_frontend::VariantPackBuilder()
                                .setDataPointers(5, data_ptrs0)
                                .setUids(5, uids0)
                                .build();
        auto variantPack1 = cudnn_frontend::VariantPackBuilder()
                                .setDataPointers(4, data_ptrs1)
                                .setUids(4, uids1)
                                .build();

        cudnnStatus_t status = cudnnBackendExecute(handle, attentionLayerPlan0->get_raw_desc(), variantPack0.get_raw_desc());
        if (status != CUDNN_STATUS_SUCCESS) {
            return status;
        }
        return cudnnBackendExecute(handle, attentionLayerPlan1->get_raw_desc(), variantPack1.get_raw_desc());
    };

   private:
    std::shared_ptr<cudnn_frontend::ExecutionPlan> attentionLayerPlan0;
    std::shared_ptr<cudnn_frontend::ExecutionPlan> attentionLayerPlan1;

    MHA_attentionLayer()                           = delete;
    MHA_attentionLayer(MHA_attentionLayer const &) = delete;
    MHA_attentionLayer &
    operator=(MHA_attentionLayer const &) = delete;
};

void
multiHeadAttention(const int64_t inputSize,
                   const int64_t headSize,
                   const int64_t seqLength,
                   const int64_t numHeads,
                   const int64_t batchSize,
                   const int64_t outputSize,
                   cudnnDataType_t dataType,
                   void const *devPtrIn,
                   void const *devPtrQKVWeight,
                   void const *devPtrOWeight,
                   void const *devPtrQKVBias,
                   void const *devPtrOBias,
                   void *devPtrOut) {
    (void)inputSize;
    (void)headSize;
    (void)seqLength;
    (void)numHeads;
    (void)batchSize;
    (void)outputSize;
    (void)dataType;
    (void)devPtrIn;
    (void)devPtrQKVWeight;
    (void)devPtrOWeight;
    (void)devPtrQKVBias;
    (void)devPtrOBias;
    (void)devPtrOut;
#if (CUDNN_VERSION >= 8301)
    cudnnHandle_t handle_;

    /*
      Suppose the following layouts:
          - input data layout: [seqLength, batchSize, inputSize]

          - QKV Weight layout: [3, projSize * numHeads, inputSize]
              Q-Weight followed by K-Weight followed by V-Weight

          - QKV Bias layout: [3,  projSize * numHeads, 1]
              Q-Bias followed by K-Bias followed by V-Bias

          - O Weight layout: [1, projSize * numHeads, outputSize]

          - O Bias layout: [1, 1, outputSize]

          - output data layout: [seqLength, batchSize, outputSize]

      All tensors are fully-packed.
    */

    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

        if (check_device_arch_newer_than("ampere") == false) {
            set_error_and_throw_exception(
                    nullptr,
                    CUDNN_STATUS_ARCH_MISMATCH,
                    "MHA: Sample requires Ampere or above GPU");
        }

        // Softmax scaler
        const float softmaxScaler   = static_cast<float>(1.0 / sqrt(static_cast<double>(headSize)));
        const int64_t embeddingSize = batchSize * numHeads * headSize * seqLength;
        (void)softmaxScaler;

        void *devPtrQKV    = nullptr;
        void *devPtrE      = nullptr;
        void *devPtrR      = nullptr;
        void *devPtrX      = nullptr;
        void *devPtrScaler = nullptr;

        // Device Memory to store embeddings Q, K, V
        checkCudaErr(cudaMalloc(&devPtrQKV, (size_t)(embeddingSize * sizeof(half) * 3)));
        // Device Memory to store internal data E = exp(softmaxScaler * (Q * K^T))
        checkCudaErr(cudaMalloc(&devPtrE, (size_t)(batchSize * numHeads * seqLength * seqLength * sizeof(half))));
        // Device Memory to store internal data R = row-reduction of E
        checkCudaErr(cudaMalloc(&devPtrR, (size_t)(batchSize * numHeads * seqLength * sizeof(float))));
        // Device Memory to store the output from attention layer
        checkCudaErr(cudaMalloc(&devPtrX, (size_t)(batchSize * numHeads * seqLength * headSize * sizeof(half))));
        // Device memory for softmax scaler parameter
        checkCudaErr(cudaMalloc(&devPtrScaler, sizeof(float)));

        // Copy softmax scaler to device memory
        checkCudaErr(cudaMemcpy(devPtrScaler, &softmaxScaler, sizeof(float), cudaMemcpyHostToDevice));

        void *devPtrQ = reinterpret_cast<char *>(devPtrQKV);
        void *devPtrK = reinterpret_cast<char *>(devPtrQ) + embeddingSize * sizeof(half);
        void *devPtrV = reinterpret_cast<char *>(devPtrK) + embeddingSize * sizeof(half);

        auto mhaParams = MHAParamsBuilder().setInputSize(inputSize)
                                           .setOutputSize(outputSize)
                                           .setHeadSize(headSize)
                                           .setNumHeads(numHeads)
                                           .setSeqLength(seqLength)
                                           .setBatchSize(batchSize)
                                           .setDataType(dataType)
                                           .setMathPrec(CUDNN_DATA_FLOAT)
                                           .build();

        // Build input projection layer to generate embeddings Q, K, V
        // Q = Wq * input + Bq
        // K = Wk * input + Bk
        // V = Wv * input + Bv
        auto inputProjLayer = MHA_intputProjLayer(handle_, mhaParams);

        // Build attention layer
        // S = softmax(softmaxScaler * (Q^T * K)) * V concatenated across all heads
        auto attnLayer = MHA_attentionLayer(handle_, mhaParams);

        // Build output projection layer to generate final output
        // Output = Wo * S + Bo
        auto outputProjLayer = MHA_outputProjLayer(handle_, mhaParams);

        status = inputProjLayer.execute(handle_, devPtrIn, devPtrQKVWeight, devPtrQKVBias, devPtrQKV);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "input layer execution error", status);

        status = attnLayer.execute(handle_, devPtrQ, devPtrK, devPtrV, devPtrScaler, devPtrE, devPtrR, devPtrX);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "attention layer execution error", status);

        status = outputProjLayer.execute(handle_, devPtrX, devPtrOWeight, devPtrOBias, devPtrOut);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "output layer execution error", status);

        checkCudaErr(cudaDeviceSynchronize());
        cudaFree(devPtrQKV);
        cudaFree(devPtrE);
        cudaFree(devPtrR);
        cudaFree(devPtrX);
        cudaFree(devPtrScaler);

    } catch (cudnn_frontend::cudnnException &e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, 0 ));
        // this example is only for Ampere cards
        if (prop.major < 8 && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl; 
        }  else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8400)
            CHECK(false);
#endif
        }
    }
#endif
}
