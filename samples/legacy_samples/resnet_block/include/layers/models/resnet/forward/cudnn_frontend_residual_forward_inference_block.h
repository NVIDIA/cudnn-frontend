#pragma once

#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <tuple>
#include <functional>
#include <unordered_map>
#include <cudnn.h>

#include "layers/common/cudnn_frontend_resnet_block_helpers.h"

#include "layers/models/resnet/block_params/cudnn_frontend_residual_block_params.h"
#include "layers/models/resnet/block_device_pointer_stores/cudnn_frontend_residual_block_dev_ptr_store.h"

namespace cudnn_frontend {

/**
 * A mixed-precision residual block for the forward inference pass. This class acts as the basic building block for a
 * Residaul Network such as a ResNet34. The residual forward block consists of a convolutional layer, a batch
 * normalization layer, and a ReLu actuvation stacked together 3 times. There is also a residual connection between the
 * input and the output of the block, which can optionally have a 1x1 convolution + batch normalization layer before the
 * addition, or just a direct connection to the addition.
 */
class ResidualForwardInferenceBlock : public IBlock {
   public:
    /**
     * Constructor for the ResidualForwardInferenceBlock.
     * @param handle A handle to the cudnn library.
     * @param params The parameters for the block. See cudnn_frontend_residual_block_params.h for more details. The
     * ResidualBlockParams object configures all the parameters for the convolutions, batch norms, ReLus, etc.
     */
    ResidualForwardInferenceBlock(cudnnHandle_t &handle, ResidualBlockParams const &params)
        : IBlock(IBlock::FORWARD_INFERENCE), params_(params) {
        (void)handle;
        getLogger() << "[cudnn_frontend] "
                    << "INFO: Creating ResidualForwardInferenceBlock" << std::endl;
        // Set the block name to be used for debugging
        blockName = "Residual Forward Block";

        // Creates the necessary tensors for the forward pass (Done for the user)
        status_ = createTensors();

        // Configures all the problem descriptors necessary for the forward pass
        status_ = configureProblems();

        // Creates the subgraphs (see cudnn_frontend_layer_interface.h for more details) which is nothing but a vector
        // of Nodes to form an operation graph.
        createSubGraphs();

        // Calculate intermediate tensor size
        calculateIntermidiateTensorsSize();
    };

    cudnnStatus_t
    createVariantPacks(ResidualBlockDevPtrStore *devPtrStore) override {
        // Set the internal dev ptr store to the passed in dev ptr store by the user to be used by execute

        getLogger() << "[cudnn_frontend] "
                    << "INFO: Creating variant packs for convs" << std::endl;

        // Loops through all the convolution + gen stats + BN Finalize params and creates a variant pack for each of
        // them.
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (params_.skip_residual_convolution(i)) {
                continue;
            }

            void *conv_data_ptrs[] = {devPtrStore->XDevPtrs[i],
                                      devPtrStore->weight_nhwc_device_pointers[i],
                                      devPtrStore->YDevPtrs[i],
                                      devPtrStore->XDescaleDevPtrs[i],
                                      devPtrStore->WDescaleDevPtrs[i],
                                      devPtrStore->YScaleDevPtrs[i],
                                      devPtrStore->YAmaxDevPtrs[i]};

            int64_t const conv_tensor_uids[] = {
                params_.conv_params[i].uids[convolution_params::UIDs::INPUT_UID],
                params_.conv_params[i].uids[convolution_params::UIDs::WEIGHT_UID],
                params_.conv_params[i].uids[convolution_params::UIDs::OUTPUT_UID],
                params_.conv_params[i].uids[convolution_params::UIDs::INPUT_DESCALE_UID],
                params_.conv_params[i].uids[convolution_params::UIDs::WEIGHT_DESCALE_UID],
                params_.conv_params[i].uids[convolution_params::UIDs::OUTPUT_SCALE_UID],
                params_.conv_params[i].uids[convolution_params::UIDs::OUTPUT_AMAX_UID]};

            auto convVariantPack = VariantPackBuilder()
                                       .setWorkspacePointer(plan_execution_workspace_pointer)
                                       .setDataPointers(7, conv_data_ptrs)
                                       .setUids(7, conv_tensor_uids)
                                       .build();

            if (checkErrorStatusAndLog(convVariantPack.get_status(),
                                       blockName,
                                       "variant pack",
                                       "conv_descale_descale_scale_amax",
                                       error_message,
                                       convVariantPack.get_error()))
                return convVariantPack.get_status();

            updateVariantPackforPlan("conv_descale_descale_scale_amax" + std::to_string(i), convVariantPack);
        }

        getLogger() << "[cudnn_frontend] "
                    << "INFO: Creating variant packs for BNs" << std::endl;
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (params_.skip_residual_convolution(i)) {
                continue;
            }

            if (i == ResidualBlockParams::ForwardLocation::TWO) {
                void *zptr = params_.skip_residual_convolution(ResidualBlockParams::ForwardLocation::RESIDUAL)
                                 ? devPtrStore->XDevPtrs[ResidualBlockParams::ForwardLocation::ZERO]
                                 : intermediate_tensor_workspace_pointer;

                void *bn_data_ptrs[] = {devPtrStore->YDevPtrs[i],
                                        devPtrStore->BNXDescaleDevPtrs[i],
                                        devPtrStore->BNYDevPtrs[i],
                                        devPtrStore->BNYScaleDevPtrs[i],
                                        devPtrStore->BNYAMaxDevPtrs[i],
                                        devPtrStore->scaleDevPtrs[i],
                                        devPtrStore->biasDevPtrs[i],
                                        devPtrStore->running_mean_DevPtrs[i],
                                        devPtrStore->running_var_DevPtrs[i],
                                        zptr,                   // Z-ptr
                                        devPtrStore->zDescale,  // Z-descale
                                        &(devPtrStore->BN_epsilons[i])};

                int64_t const bn_tensor_uids[] = {
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_DESCALE_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_SCALE_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_AMAX_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_SCALE_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_BIAS_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::MEAN_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::VAR_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::ADD_TENSOR_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::ADD_TENSOR_DESCALE_UID],
                    params_.bn_params[i].uids[bn_fusion_params::UIDs::EPSILON_UID]};

                auto bnVariantPack = VariantPackBuilder()
                                         .setWorkspacePointer(plan_execution_workspace_pointer)
                                         .setDataPointers(12, bn_data_ptrs)
                                         .setUids(12, bn_tensor_uids)
                                         .build();

                updateVariantPackforPlan("descale_bn_add_relu_scale_amax" + std::to_string(i), bnVariantPack);

            } else if (i == ResidualBlockParams::ForwardLocation::RESIDUAL) {
                void *bn_data_ptrs[] = {devPtrStore->YDevPtrs[i],
                                        devPtrStore->BNXDescaleDevPtrs[i],
                                        intermediate_tensor_workspace_pointer,
                                        devPtrStore->BNYScaleDevPtrs[i],
                                        devPtrStore->BNYAMaxDevPtrs[i],
                                        devPtrStore->scaleDevPtrs[i],
                                        devPtrStore->biasDevPtrs[i],
                                        devPtrStore->running_mean_DevPtrs[i],
                                        devPtrStore->running_var_DevPtrs[i],
                                        &(devPtrStore->BN_epsilons[i])};

                int64_t const bn_tensor_uids[] = {params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_DESCALE_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_SCALE_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_AMAX_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_SCALE_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_BIAS_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::MEAN_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::VAR_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::EPSILON_UID]};

                auto bnVariantPack = VariantPackBuilder()
                                         .setWorkspacePointer(plan_execution_workspace_pointer)
                                         .setDataPointers(10, bn_data_ptrs)
                                         .setUids(10, bn_tensor_uids)
                                         .build();

                updateVariantPackforPlan("descale_bn_scale_amax" + std::to_string(i), bnVariantPack);
            }  // Required pesky add operation
            else {
                void *bn_data_ptrs[] = {devPtrStore->YDevPtrs[i],
                                        devPtrStore->BNXDescaleDevPtrs[i],
                                        devPtrStore->BNYDevPtrs[i],
                                        devPtrStore->BNYScaleDevPtrs[i],
                                        devPtrStore->BNYAMaxDevPtrs[i],
                                        devPtrStore->scaleDevPtrs[i],
                                        devPtrStore->biasDevPtrs[i],
                                        devPtrStore->running_mean_DevPtrs[i],
                                        devPtrStore->running_var_DevPtrs[i],
                                        &(devPtrStore->BN_epsilons[i])};

                int64_t const bn_tensor_uids[] = {params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_DESCALE_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_SCALE_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::OUTPUT_AMAX_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_SCALE_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::INPUT_BIAS_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::MEAN_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::VAR_UID],
                                                  params_.bn_params[i].uids[bn_fusion_params::UIDs::EPSILON_UID]};

                auto bnVariantPack = VariantPackBuilder()
                                         .setWorkspacePointer(plan_execution_workspace_pointer)
                                         .setDataPointers(10, bn_data_ptrs)
                                         .setUids(10, bn_tensor_uids)
                                         .build();

                updateVariantPackforPlan("descale_bn_relu_scale_amax" + std::to_string(i), bnVariantPack);
            }
        }

        return CUDNN_STATUS_SUCCESS;
    };

   private:
    // Handle, params, and dev ptr store object for the block
    ResidualBlockParams params_;
    ResidualBlockDevPtrStore devPtrStore_;

    // Map of Problems (Problem name -> problem descriptor)
    std::unordered_map<std::string, std::shared_ptr<ConvDesc>> convolution_problems;
    std::unordered_map<std::string, std::shared_ptr<PointwiseDesc>> pointwise_problems;
    std::unordered_map<std::string, std::shared_ptr<ReductionDesc>> reduction_problems;

    cudnnStatus_t
    createTensors() {
        getLogger() << "[cudnn_frontend] "
                    << "INFO: Creating Tensors for Residual Forward Block!" << std::endl;

        status_ = createConvTensors();
        if (status_ != CUDNN_STATUS_SUCCESS) {
            getLogger() << "[cudnn_frontend] " << error_message << std::endl;
            return status_;
        }

        status_ = createBNInferenceTensors();
        if (status_ != CUDNN_STATUS_SUCCESS) {
            getLogger() << "[cudnn_frontend] " << error_message << std::endl;
            return status_;
        }
        return CUDNN_STATUS_SUCCESS;
    };

    cudnnStatus_t
    createConvTensors() {
        auto status = CUDNN_STATUS_SUCCESS;
        getLogger() << "[cudnn_frontend] "
                    << "INFO: Creating Conv Tensors for Residual Forward Block!" << std::endl;
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (params_.skip_residual_convolution(i)) {
                continue;
            }

            getLogger() << "[cudnn_frontend] "
                        << "INFO: Input Residual Forward Block " << i << std::endl;
            auto &tensor_params = params_.conv_params[i];
            generateStrides(
                tensor_params.input_dim, tensor_params.input_stride, tensor_params.dim_count, CUDNN_TENSOR_NHWC);
            auto input = TensorBuilder()
                             .setDim(tensor_params.dim_count, tensor_params.input_dim)
                             .setStrides(tensor_params.dim_count, tensor_params.input_stride)
                             .setId(tensor_params.uids[convolution_params::UIDs::INPUT_UID])
                             .setAlignment(16)
                             .setDataType(tensor_params.tensor_data_type)
                             .setVirtual(false)
                             .setByValue(false)
                             .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: Weight Residual Forward Block " << i << std::endl;

            generateStrides(
                tensor_params.weight_dim, tensor_params.weight_stride, tensor_params.dim_count, CUDNN_TENSOR_NHWC);
            auto weight = TensorBuilder()
                              .setDim(tensor_params.dim_count, tensor_params.weight_dim)
                              .setStrides(tensor_params.dim_count, tensor_params.weight_stride)
                              .setId(tensor_params.uids[convolution_params::UIDs::WEIGHT_UID])
                              .setAlignment(16)
                              .setDataType(tensor_params.tensor_data_type)
                              .setVirtual(false)
                              .setByValue(false)
                              .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: afterConv Residual Forward Block " << i << std::endl;
            generateStrides(
                tensor_params.output_dim, tensor_params.output_stride, tensor_params.dim_count, CUDNN_TENSOR_NHWC);
            auto afterConv = TensorBuilder()
                                 .setDim(tensor_params.dim_count, tensor_params.output_dim)
                                 .setStrides(tensor_params.dim_count, tensor_params.output_stride)
                                 .setId(tensor_params.uids[convolution_params::UIDs::AFTER_CONV_UID])
                                 .setAlignment(16)
                                 .setDataType(tensor_params.compute_type)
                                 .setVirtual(true)
                                 .setByValue(false)
                                 .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: Amax Residual Forward Block " << i << std::endl;
            auto tensor_amax = TensorBuilder()
                                   .setDim(amax_dim_stride.size(), amax_dim_stride.data())
                                   .setStrides(amax_dim_stride.size(), amax_dim_stride.data())
                                   .setId(tensor_params.uids[convolution_params::UIDs::OUTPUT_AMAX_UID])
                                   .setAlignment(16)
                                   .setDataType(tensor_params.compute_type)
                                   .setVirtual(false)
                                   .setByValue(false)
                                   .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: Input Descale Forward Block " << i << std::endl;
            auto tensor_x_descale = TensorBuilder()
                                        .setDim(amax_dim_stride.size(), amax_dim_stride.data())
                                        .setStrides(amax_dim_stride.size(), amax_dim_stride.data())
                                        .setId(tensor_params.uids[convolution_params::UIDs::INPUT_DESCALE_UID])
                                        .setAlignment(16)
                                        .setDataType(tensor_params.compute_type)
                                        .setVirtual(false)
                                        .setByValue(false)
                                        .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: Weight Descale Forward Block " << i << std::endl;
            auto tensor_w_descale = TensorBuilder()
                                        .setDim(amax_dim_stride.size(), amax_dim_stride.data())
                                        .setStrides(amax_dim_stride.size(), amax_dim_stride.data())
                                        .setId(tensor_params.uids[convolution_params::UIDs::WEIGHT_DESCALE_UID])
                                        .setAlignment(16)
                                        .setDataType(tensor_params.compute_type)
                                        .setVirtual(false)
                                        .setByValue(false)
                                        .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: Weight Descale Forward Block " << i << std::endl;
            auto tensor_y_scale = TensorBuilder()
                                      .setDim(amax_dim_stride.size(), amax_dim_stride.data())
                                      .setStrides(amax_dim_stride.size(), amax_dim_stride.data())
                                      .setId(tensor_params.uids[convolution_params::UIDs::OUTPUT_SCALE_UID])
                                      .setAlignment(16)
                                      .setDataType(tensor_params.compute_type)
                                      .setVirtual(false)
                                      .setByValue(false)
                                      .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: AFTER_INPUT_DESCALE_UID Residual Forward Block " << i << std::endl;
            auto afterxDescale = TensorBuilder()
                                     .setDim(tensor_params.dim_count, tensor_params.output_dim)
                                     .setStrides(tensor_params.dim_count, tensor_params.output_stride)
                                     .setId(tensor_params.uids[convolution_params::UIDs::AFTER_INPUT_DESCALE_UID])
                                     .setAlignment(16)
                                     .setDataType(tensor_params.compute_type)
                                     .setVirtual(true)
                                     .setByValue(false)
                                     .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: AFTER_WEIGHT_DESCALE_UID Residual Forward Block " << i << std::endl;
            auto afterwDescale = TensorBuilder()
                                     .setDim(tensor_params.dim_count, tensor_params.output_dim)
                                     .setStrides(tensor_params.dim_count, tensor_params.output_stride)
                                     .setId(tensor_params.uids[convolution_params::UIDs::AFTER_WEIGHT_DESCALE_UID])
                                     .setAlignment(16)
                                     .setDataType(tensor_params.compute_type)
                                     .setVirtual(true)
                                     .setByValue(false)
                                     .build();
            getLogger() << "[cudnn_frontend] "
                        << "INFO: Output Residual Forward Block " << i << std::endl;
            auto output = TensorBuilder()
                              .setDim(tensor_params.dim_count, tensor_params.output_dim)
                              .setStrides(tensor_params.dim_count, tensor_params.output_stride)
                              .setId(tensor_params.uids[convolution_params::UIDs::OUTPUT_UID])
                              .setAlignment(16)
                              .setDataType(tensor_params.tensor_data_type)
                              .setVirtual(false)
                              .setByValue(false)
                              .build();

            addTensor("CONV::X" + std::to_string(i), std::make_shared<Tensor>(std::move(input)));
            addTensor("CONV::W" + std::to_string(i), std::make_shared<Tensor>(std::move(weight)));
            addTensor("CONV::Y" + std::to_string(i), std::make_shared<Tensor>(std::move(output)));
            addTensor("CONV::AMax" + std::to_string(i), std::make_shared<Tensor>(std::move(tensor_amax)));
            addTensor("CONV::XDescale" + std::to_string(i), std::make_shared<Tensor>(std::move(tensor_x_descale)));
            addTensor("CONV::WDescale" + std::to_string(i), std::make_shared<Tensor>(std::move(tensor_w_descale)));
            addTensor("CONV::YScale" + std::to_string(i), std::make_shared<Tensor>(std::move(tensor_y_scale)));
            addTensor("CONV::AfterConv" + std::to_string(i), std::make_shared<Tensor>(std::move(afterConv)));
            addTensor("CONV::AfterXDescale" + std::to_string(i), std::make_shared<Tensor>(std::move(afterxDescale)));
            addTensor("CONV::AfterWDescale" + std::to_string(i), std::make_shared<Tensor>(std::move(afterwDescale)));
        }
        return status;
    }

    cudnnStatus_t
    createBNInferenceTensors() {
        auto status = CUDNN_STATUS_SUCCESS;
        getLogger() << "[cudnn_frontend] "
                    << "INFO: Creating BN Fusion Tensors for Residual Forward Block!" << std::endl;
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (params_.skip_residual_convolution(i)) {
                continue;
            }

            getLogger() << "[cudnn_frontend] "
                        << "INFO: Input Residual Forward Block " << i << std::endl;
            auto &tensor_params = params_.bn_params[i];
            generateStrides(
                tensor_params.input_dim, tensor_params.input_strides, tensor_params.dim_count, CUDNN_TENSOR_NHWC);
            generateStrides(tensor_params.per_channel_dim,
                            tensor_params.per_channel_strides,
                            tensor_params.dim_count,
                            CUDNN_TENSOR_NHWC);

            auto input = TensorBuilder()
                             .setDim(tensor_params.dim_count, tensor_params.input_dim)
                             .setStrides(tensor_params.dim_count, tensor_params.input_strides)
                             .setId(tensor_params.uids[bn_fusion_params::UIDs::INPUT_UID])
                             .setAlignment(16)
                             .setDataType(tensor_params.data_type)
                             .setVirtual(false)
                             .setByValue(false)
                             .build();

            auto output = TensorBuilder()
                              .setDim(tensor_params.dim_count, tensor_params.input_dim)
                              .setStrides(tensor_params.dim_count, tensor_params.input_strides)
                              .setId(tensor_params.uids[bn_fusion_params::UIDs::OUTPUT_UID])
                              .setAlignment(16)
                              .setDataType(tensor_params.data_type)
                              .setVirtual(false)
                              .setByValue(false)
                              .build();

            auto input_tensor_create = [&tensor_params](bn_fusion_params::UIDs uid, bool is_virtual) {
                return cudnn_frontend::TensorBuilder()
                    .setDim(tensor_params.dim_count, tensor_params.input_dim)
                    .setStride(tensor_params.dim_count, tensor_params.input_strides)
                    .setId(tensor_params.uids[uid])
                    .setAlignment(16)
                    .setDataType(tensor_params.compute_type)
                    .setVirtual(is_virtual)
                    .setByValue(false)
                    .build();
            };

            auto input_hp       = input_tensor_create(bn_fusion_params::UIDs::INPUT_HP_UID, true);
            auto input_sub_mean = input_tensor_create(bn_fusion_params::UIDs::INPUT_SUB_MEAN_UID, true);
            auto norm_input     = input_tensor_create(bn_fusion_params::UIDs::NORM_INPUT_UID, true);
            auto mul_scale      = input_tensor_create(bn_fusion_params::UIDs::INPUT_MUL_SCALE_UID, true);
            auto add_bias       = input_tensor_create(bn_fusion_params::UIDs::INPUT_ADD_BIAS_UID, true);
            auto output_hp      = input_tensor_create(bn_fusion_params::UIDs::BN_OUTPUT_UID, true);

            auto tensor_create = [&tensor_params](bn_fusion_params::UIDs uid, bool is_virtual) {
                return cudnn_frontend::TensorBuilder()
                    .setDim(tensor_params.dim_count, tensor_params.per_channel_dim)
                    .setStride(tensor_params.dim_count, tensor_params.per_channel_strides)
                    .setId(tensor_params.uids[uid])
                    .setAlignment(16)
                    .setDataType(tensor_params.compute_type)
                    .setVirtual(is_virtual)
                    .setByValue(false)
                    .build();
            };
            auto scaleTensor     = tensor_create(bn_fusion_params::UIDs::INPUT_SCALE_UID, false);
            auto biasTensor      = tensor_create(bn_fusion_params::UIDs::INPUT_BIAS_UID, false);
            auto MeanTensor      = tensor_create(bn_fusion_params::UIDs::MEAN_UID, false);
            auto VarTensor       = tensor_create(bn_fusion_params::UIDs::VAR_UID, false);
            auto VarAddEpsTensor = tensor_create(bn_fusion_params::UIDs::VAR_ADD_EPS_UID, true);
            auto RsqrtVarTensor  = tensor_create(bn_fusion_params::UIDs::RSQRT_VAR_UID, true);

            auto scalar_tensor_create = [&tensor_params](bn_fusion_params::UIDs uid, bool is_pass_by_value) {
                return cudnn_frontend::TensorBuilder()
                    .setDim(amax_dim_stride.size(), amax_dim_stride.data())
                    .setStride(amax_dim_stride.size(), amax_dim_stride.data())
                    .setId(tensor_params.uids[uid])
                    .setAlignment(16)
                    .setDataType(tensor_params.compute_type)
                    .setVirtual(false)
                    .setByValue(is_pass_by_value)
                    .build();
            };
            auto input_descale = scalar_tensor_create(bn_fusion_params::UIDs::INPUT_DESCALE_UID, false);
            auto output_scale  = scalar_tensor_create(bn_fusion_params::UIDs::OUTPUT_SCALE_UID, false);
            auto output_amax   = scalar_tensor_create(bn_fusion_params::UIDs::OUTPUT_AMAX_UID, false);
            auto epsilonTensor = scalar_tensor_create(bn_fusion_params::UIDs::EPSILON_UID, true);

            addTensor("BN::X_" + std::to_string(i), std::make_shared<Tensor>(std::move(input)));
            addTensor("BN::X_DESCALE_" + std::to_string(i), std::make_shared<Tensor>(std::move(input_descale)));
            addTensor("BN::X_HP_" + std::to_string(i), std::make_shared<Tensor>(std::move(input_hp)));
            addTensor("BN::X_SUB_MeanTensor_" + std::to_string(i), std::make_shared<Tensor>(std::move(input_sub_mean)));
            addTensor("BN::VAR_ADD_EPS_" + std::to_string(i), std::make_shared<Tensor>(std::move(VarAddEpsTensor)));
            addTensor("BN::RSQRT_VAR_" + std::to_string(i), std::make_shared<Tensor>(std::move(RsqrtVarTensor)));
            addTensor("BN::N_X_" + std::to_string(i), std::make_shared<Tensor>(std::move(norm_input)));
            addTensor("BN::N_X_MUL_S_" + std::to_string(i), std::make_shared<Tensor>(std::move(mul_scale)));
            addTensor("BN::Y_" + std::to_string(i), std::make_shared<Tensor>(std::move(output)));
            addTensor("BN::Y_HP_" + std::to_string(i), std::make_shared<Tensor>(std::move(output_hp)));
            addTensor("BN::Y_SCALE_" + std::to_string(i), std::make_shared<Tensor>(std::move(output_scale)));
            addTensor("BN::Y_AMAX_" + std::to_string(i), std::make_shared<Tensor>(std::move(output_amax)));
            addTensor("BN::EPSILON_" + std::to_string(i), std::make_shared<Tensor>(std::move(epsilonTensor)));
            addTensor("BN::scaleTensor_" + std::to_string(i), std::make_shared<Tensor>(std::move(scaleTensor)));
            addTensor("BN::biasTensor_" + std::to_string(i), std::make_shared<Tensor>(std::move(biasTensor)));
            addTensor("BN::MeanTensor_" + std::to_string(i), std::make_shared<Tensor>(std::move(MeanTensor)));
            addTensor("BN::VarTensor_" + std::to_string(i), std::make_shared<Tensor>(std::move(VarTensor)));

            if (tensor_params.has_relu) {
                getLogger() << "[cudnn_frontend] "
                            << "INFO: Input Residual Forward Block After Relu" << i << std::endl;
                auto afterRelu = input_tensor_create(bn_fusion_params::UIDs::AFTER_ACTIVATION_UID, true);
                addTensor("BN::afterRelu_" + std::to_string(i), std::make_shared<Tensor>(std::move(afterRelu)));
            }

            if (tensor_params.has_add_relu) {
                getLogger() << "[cudnn_frontend] "
                            << "INFO: Input Residual Forward Block BN-ADD-Relu" << i << std::endl;

                auto add_input = TensorBuilder()
                                     .setDim(tensor_params.dim_count, tensor_params.input_dim)
                                     .setStrides(tensor_params.dim_count, tensor_params.input_strides)
                                     .setId(tensor_params.uids[bn_fusion_params::UIDs::ADD_TENSOR_UID])
                                     .setAlignment(16)
                                     .setDataType(tensor_params.data_type)
                                     .setVirtual(false)
                                     .setByValue(false)
                                     .build();
                auto add_ip_descale = scalar_tensor_create(bn_fusion_params::UIDs::ADD_TENSOR_DESCALE_UID, false);
                auto add_input_hp   = input_tensor_create(bn_fusion_params::UIDs::ADD_TENSOR_HP_UID, true);
                auto sum_hp         = input_tensor_create(bn_fusion_params::UIDs::BEFORE_ACTIVATION_UID, true);

                addTensor("BN::Z_" + std::to_string(i), std::make_shared<Tensor>(std::move(add_input)));
                addTensor("BN::Z_HP_" + std::to_string(i), std::make_shared<Tensor>(std::move(add_input_hp)));
                addTensor("BN::Z_DESCALE_" + std::to_string(i), std::make_shared<Tensor>(std::move(add_ip_descale)));
                addTensor("BN::SUM_HP_" + std::to_string(i), std::make_shared<Tensor>(std::move(sum_hp)));
            }
        }
        return status;
    }

    cudnnStatus_t
    configureProblems() {
        // Use the same scale descriptor for all fp8 related pointwise multiplications
        auto scale_descriptor =
            PointwiseDescBuilder().setMode(CUDNN_POINTWISE_MUL).setMathPrecision(CUDNN_DATA_FLOAT).build();
        addPointwiseProblem("scale", std::make_shared<PointwiseDesc>(std::move(scale_descriptor)));

        // Use the same reduction descriptor for all fp8 related amaxs'.
        auto reduction_descriptor =
            ReductionDescBuilder().setReductionOp(CUDNN_REDUCE_TENSOR_AMAX).setMathPrecision(CUDNN_DATA_FLOAT).build();
        addReductionProblem("amax", std::make_shared<ReductionDesc>(std::move(reduction_descriptor)));

        // Relu problem is common wherever it exists.
        auto relu = PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_RELU_FWD)
                        .setComputeType(params_.bn_params[ResidualBlockParams::ForwardLocation::ZERO].compute_type)
                        .build();
        addPointwiseProblem("ReLU", std::make_shared<PointwiseDesc>(std::move(relu)));

        auto add = PointWiseDescBuilder().setMode(CUDNN_POINTWISE_ADD).setComputeType(CUDNN_DATA_FLOAT).build();
        addPointwiseProblem("Add", std::make_shared<PointwiseDesc>(std::move(add)));

        auto sub = PointWiseDescBuilder().setMode(CUDNN_POINTWISE_SUB).setComputeType(CUDNN_DATA_FLOAT).build();
        addPointwiseProblem("sub", std::make_shared<PointwiseDesc>(std::move(sub)));

        auto rsqrt = PointWiseDescBuilder().setMode(CUDNN_POINTWISE_RSQRT).setComputeType(CUDNN_DATA_FLOAT).build();
        addPointwiseProblem("rsqrt", std::make_shared<PointwiseDesc>(std::move(rsqrt)));

        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (params_.skip_residual_convolution(i)) {
                continue;
            }

            auto &tensor_params = params_.conv_params[i];

            // First convolution descriptor
            auto dim_count               = tensor_params.dim_count;
            auto spatial_dim_count       = dim_count - 2;
            int64_t const *conv_stride   = tensor_params.stride;
            int64_t const *conv_padding  = tensor_params.padding;
            int64_t const *conv_dilation = tensor_params.dilation;
            auto convolution_descriptor  = cudnn_frontend::ConvDescBuilder()
                                              .setComputeType(tensor_params.compute_type)
                                              .setMathMode(CUDNN_CROSS_CORRELATION)
                                              .setSpatialDimCount(spatial_dim_count)
                                              .setSpatialStride(spatial_dim_count, conv_stride)
                                              .setPrePadding(spatial_dim_count, conv_padding)
                                              .setPostPadding(spatial_dim_count, conv_padding)
                                              .setDilation(spatial_dim_count, conv_dilation)
                                              .build();

            addConvProblem(std::string("conv") + std::to_string(i),
                           std::make_shared<ConvDesc>(std::move(convolution_descriptor)));
        }

        return CUDNN_STATUS_SUCCESS;
    };

    /**
     * @brief The meat of the Residual forward block. Creates all the subgraphs needed for the forward pass. From a
     * cudnn perspective, we know what can be fused and what can't be fused. We also know what all the tensors are going
     * to be used and the connectivity between the tensors. Thus, we create the subgraphs internally for the user.
     *
     */
    void
    createSubGraphs() {
        getLogger() << "[cudnn_frontend] "
                    << "INFO: Creating sub graph" << std::endl;
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (params_.skip_residual_convolution(i)) {
                continue;
            }

            SubGraph convolution_subgraph = {
                {"conv" + std::to_string(i),
                 cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                 "conv" + std::to_string(i),
                 {"CONV::X" + std::to_string(i), "CONV::W" + std::to_string(i), "CONV::AfterConv" + std::to_string(i)}},
                {"conv_input_descale" + std::to_string(i),
                 cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                 "scale",
                 {"CONV::AfterConv" + std::to_string(i),
                  "CONV::XDescale" + std::to_string(i),
                  "CONV::AfterXDescale" + std::to_string(i)}},
                {"conv_weight_descale" + std::to_string(i),
                 cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                 "scale",
                 {"CONV::AfterXDescale" + std::to_string(i),
                  "CONV::WDescale" + std::to_string(i),
                  "CONV::AfterWDescale" + std::to_string(i)}},
                {"conv_output_scale" + std::to_string(i),
                 cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                 "scale",
                 {"CONV::AfterWDescale" + std::to_string(i),
                  "CONV::YScale" + std::to_string(i),
                  "CONV::Y" + std::to_string(i)}},
                {"amax" + std::to_string(i),
                 cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
                 "amax",
                 {"CONV::AfterWDescale" + std::to_string(i), "CONV::AMax" + std::to_string(i)}}};

            addSubOpGraph("conv_descale_descale_scale_amax" + std::to_string(i), convolution_subgraph);

            if (i == ResidualBlockParams::ForwardLocation::ZERO || i == ResidualBlockParams::ForwardLocation::ONE) {
                SubGraph bn_subgraph = {
                    {"BN_input_descale" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::X_" + std::to_string(i),
                      "BN::X_DESCALE_" + std::to_string(i),
                      "BN::X_HP_" + std::to_string(i)}},
                    {"BN_sub_mean" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "sub",
                     {"BN::X_HP_" + std::to_string(i),
                      "BN::MeanTensor_" + std::to_string(i),
                      "BN::X_SUB_MeanTensor_" + std::to_string(i)}},
                    {"BN_eps_add" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "Add",
                     {"BN::VarTensor_" + std::to_string(i),
                      "BN::EPSILON_" + std::to_string(i),
                      "BN::VAR_ADD_EPS_" + std::to_string(i)}},
                    {"BN_rsqrt_var" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "rsqrt",
                     {"BN::VAR_ADD_EPS_" + std::to_string(i), "BN::RSQRT_VAR_" + std::to_string(i)}},
                    {"BN_N_X" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::X_SUB_MeanTensor_" + std::to_string(i),
                      "BN::RSQRT_VAR_" + std::to_string(i),
                      "BN::N_X_" + std::to_string(i)}},
                    {"BN_MUL_S" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::N_X_" + std::to_string(i),
                      "BN::scaleTensor_" + std::to_string(i),
                      "BN::N_X_MUL_S_" + std::to_string(i)}},
                    {"BN_ADD_B" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "Add",
                     {"BN::N_X_MUL_S_" + std::to_string(i),
                      "BN::biasTensor_" + std::to_string(i),
                      "BN::Y_HP_" + std::to_string(i)}},
                    {"BN_op_relu" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "ReLU",
                     {"BN::Y_HP_" + std::to_string(i), "BN::afterRelu_" + std::to_string(i)}},
                    {"BN_op_scale" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::afterRelu_" + std::to_string(i),
                      "BN::Y_SCALE_" + std::to_string(i),
                      "BN::Y_" + std::to_string(i)}},
                    {"BN_amax" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
                     "amax",
                     {"BN::afterRelu_" + std::to_string(i), "BN::Y_AMAX_" + std::to_string(i)}}};
                addSubOpGraph("descale_bn_relu_scale_amax" + std::to_string(i), bn_subgraph);

            } else if (i == ResidualBlockParams::ForwardLocation::TWO) {
                SubGraph bn_subgraph = {
                    {"BN_input_descale" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::X_" + std::to_string(i),
                      "BN::X_DESCALE_" + std::to_string(i),
                      "BN::X_HP_" + std::to_string(i)}},
                    {"BN_sub_mean" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "sub",
                     {"BN::X_HP_" + std::to_string(i),
                      "BN::MeanTensor_" + std::to_string(i),
                      "BN::X_SUB_MeanTensor_" + std::to_string(i)}},
                    {"BN_eps_add" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "Add",
                     {"BN::VarTensor_" + std::to_string(i),
                      "BN::EPSILON_" + std::to_string(i),
                      "BN::VAR_ADD_EPS_" + std::to_string(i)}},
                    {"BN_rsqrt_var" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "rsqrt",
                     {"BN::VAR_ADD_EPS_" + std::to_string(i), "BN::RSQRT_VAR_" + std::to_string(i)}},
                    {"BN_N_X" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::X_SUB_MeanTensor_" + std::to_string(i),
                      "BN::RSQRT_VAR_" + std::to_string(i),
                      "BN::N_X_" + std::to_string(i)}},
                    {"BN_MUL_S" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::N_X_" + std::to_string(i),
                      "BN::scaleTensor_" + std::to_string(i),
                      "BN::N_X_MUL_S_" + std::to_string(i)}},
                    {"BN_ADD_B" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "Add",
                     {"BN::N_X_MUL_S_" + std::to_string(i),
                      "BN::biasTensor_" + std::to_string(i),
                      "BN::Y_HP_" + std::to_string(i)}},
                    {"BN_op_descale" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::Z_" + std::to_string(i),
                      "BN::Z_DESCALE_" + std::to_string(i),
                      "BN::Z_HP_" + std::to_string(i)}},
                    {"BN_op_add" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "Add",
                     {"BN::Y_HP_" + std::to_string(i),
                      "BN::Z_HP_" + std::to_string(i),
                      "BN::SUM_HP_" + std::to_string(i)}},
                    {"BN_op_relu" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "ReLU",
                     {"BN::SUM_HP_" + std::to_string(i), "BN::afterRelu_" + std::to_string(i)}},
                    {"BN_op_scale" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::afterRelu_" + std::to_string(i),
                      "BN::Y_SCALE_" + std::to_string(i),
                      "BN::Y_" + std::to_string(i)}},
                    {"BN_amax" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
                     "amax",
                     {"BN::afterRelu_" + std::to_string(i), "BN::Y_AMAX_" + std::to_string(i)}}};
                addSubOpGraph("descale_bn_add_relu_scale_amax" + std::to_string(i), bn_subgraph);

            } else if (i == ResidualBlockParams::ForwardLocation::RESIDUAL) {
                SubGraph bn_subgraph = {
                    {"BN_input_descale" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::X_" + std::to_string(i),
                      "BN::X_DESCALE_" + std::to_string(i),
                      "BN::X_HP_" + std::to_string(i)}},
                    {"BN_sub_mean" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "sub",
                     {"BN::X_HP_" + std::to_string(i),
                      "BN::MeanTensor_" + std::to_string(i),
                      "BN::X_SUB_MeanTensor_" + std::to_string(i)}},
                    {"BN_eps_add" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "Add",
                     {"BN::VarTensor_" + std::to_string(i),
                      "BN::EPSILON_" + std::to_string(i),
                      "BN::VAR_ADD_EPS_" + std::to_string(i)}},
                    {"BN_rsqrt_var" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "rsqrt",
                     {"BN::VAR_ADD_EPS_" + std::to_string(i), "BN::RSQRT_VAR_" + std::to_string(i)}},
                    {"BN_N_X" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::X_SUB_MeanTensor_" + std::to_string(i),
                      "BN::RSQRT_VAR_" + std::to_string(i),
                      "BN::N_X_" + std::to_string(i)}},
                    {"BN_MUL_S" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::N_X_" + std::to_string(i),
                      "BN::scaleTensor_" + std::to_string(i),
                      "BN::N_X_MUL_S_" + std::to_string(i)}},
                    {"BN_ADD_B" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "Add",
                     {"BN::N_X_MUL_S_" + std::to_string(i),
                      "BN::biasTensor_" + std::to_string(i),
                      "BN::Y_HP_" + std::to_string(i)}},
                    {"BN_op_scale" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                     "scale",
                     {"BN::Y_HP_" + std::to_string(i),
                      "BN::Y_SCALE_" + std::to_string(i),
                      "BN::Y_" + std::to_string(i)}},
                    {"BN_amax" + std::to_string(i),
                     cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
                     "amax",
                     {"BN::Y_HP_" + std::to_string(i), "BN::Y_AMAX_" + std::to_string(i)}}};
                addSubOpGraph("descale_bn_scale_amax" + std::to_string(i), bn_subgraph);
            }
        }
    };

    void
    calculateIntermidiateTensorsSize() {
        // Plans are executed serially. Each plan has atmost one intermidaite tensor as i/o.
        // Hence, just the maximum of all intermediate tensor sizes should be sufficient.
        intermediate_tensor_workspace_size = 0;

        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (params_.skip_residual_convolution(i)) {
                continue;
            }

            intermediate_tensor_workspace_size =
                std::max(intermediate_tensor_workspace_size,
                         compute_tensor_size(tensor_map["BN::X_" + std::to_string(i)]->getDim(), 4));
        }

        getLogger() << "[cudnn_frontend] "
                    << "INFO: Intermidiate tensors require " << intermediate_tensor_workspace_size << " bytes."
                    << std::endl;
    }

    cudnnStatus_t
    buildOpGraph(cudnnHandle_t &handle, SubGraph const &sub_graph, const std::string &graph_name) override {
        getLogger() << "[cudnn_frontend] "
                    << "INFO: Building Operation Graph for Residual Forward Block." << graph_name << std::endl;

        // Instantiate a vector of operations for the OperationGraphBuilder,
        std::vector<Operation> ops;

        // We loop through all the nodes in the subgraph
        for (auto &node : sub_graph) {
            getLogger() << "[cudnn_frontend] "
                        << "INFO: Creating Operation for node: " << node.op_name << std::endl;

            // The way the subgraph is constructed, the edges represent connectivity with respect to tensors.
            // In this case, a convolution node edge list represents the connectivity between
            // the input tensor, the filter tensor, and the output tensor. We check if these tensors exist in the tensor
            // map.
            for (auto &edge : node.edges) {
                if (tensor_map.find(edge) == tensor_map.end()) {
                    error_message = "ERROR: Tensor " + edge + " not found!";
                    return CUDNN_STATUS_EXECUTION_FAILED;
                }
            }
            // Lets check the backend descriptor type to see what operation we're dealing with.
            switch (node.type) {
                // If the node is a forward convolutional node, we create a convolution operation.
                case cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR: {
                    float alpha = 1.0f;
                    float beta  = 0.0f;

                    // A node also has a member called problem name, used to identify the problem descriptor. We use
                    // this to get the problem descriptor from the problem map. See `configureProblems()` for
                    // implementation details.
                    if (convolution_problems.find(node.problem_name) == convolution_problems.end()) {
                        error_message = "ERROR: Convolution problem " + node.problem_name + " not found!";
                        return CUDNN_STATUS_EXECUTION_FAILED;
                    }

#ifndef NV_CUDNN_DISABLE_EXCEPTION
                    try {
#endif

                        // Create the convolution operation.
                        auto convOp =
                            cudnn_frontend::OperationBuilder(node.type)     // the backened descriptor type
                                .setxDesc(*(tensor_map.at(node.edges[0])))  // Input tensor from edge list
                                .setwDesc(*(tensor_map.at(node.edges[1])))  // Filter tensor from edge list
                                .setyDesc(*(tensor_map.at(node.edges[2])))  // Output tensor from edge list
                                .setcDesc(*(convolution_problems.at(node.problem_name)))  // Conv Problem descriptor
                                .setAlpha(alpha)
                                .setBeta(beta)
                                .build();

                        if (checkErrorStatusAndLog(convOp.get_status(),
                                                   blockName,
                                                   "operation descriptor",
                                                   node.op_name,
                                                   error_message,
                                                   convOp.get_error()))
                            return convOp.get_status();

                        ops.emplace_back(std::move(convOp));

#ifndef NV_CUDNN_DISABLE_EXCEPTION
                    } catch (cudnn_frontend::cudnnException &e) {
                        logErrorMessage(e.getCudnnStatus(),
                                        blockName,
                                        "operation descriptor",
                                        node.op_name,
                                        error_message,
                                        e.what());
                        throw cudnnException(error_message.c_str(), e.getCudnnStatus());
                    }
#endif
                    getLogger() << "[cudnn_frontend] "
                                << "INFO: Built Convolution " << node.op_name << std::endl;
                } break;

                // A pointwise descriptor can be a scale, a bias, or an activation (ReLu) operation.
                case cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: {
                    // Check if the problem name exists in the problem map
                    if (pointwise_problems.find(node.problem_name) == pointwise_problems.end()) {
                        error_message = "ERROR: Pointwise problem " + node.problem_name + " not found!";
                        return CUDNN_STATUS_EXECUTION_FAILED;
                    }

#ifndef NV_CUDNN_DISABLE_EXCEPTION
                    try {
#endif

                        auto pointwise_descriptor = pointwise_problems.at(node.problem_name);
                        if (pointwise_descriptor->getPointWiseMode() == cudnn_frontend::PointwiseMode_t::IDENTITY ||
                            pointwise_descriptor->getPointWiseMode() == cudnn_frontend::PointwiseMode_t::RELU_FWD ||
                            pointwise_descriptor->getPointWiseMode() == cudnn_frontend::PointwiseMode_t::RSQRT) {
                            // Create a Scale or Bias Node for Weight1 * Input + Bias
                            auto pwOp =
                                cudnn_frontend::OperationBuilder(node.type)
                                    .setxDesc(
                                        *(tensor_map.at(node.edges[0])))  // Input tensor (output of prev conv node)
                                    .setyDesc(*(tensor_map.at(node.edges[1])))  // Output tensor
                                    .setpwDesc(*pointwise_descriptor)           // PW problem descriptor
                                    .build();

                            if (checkErrorStatusAndLog(pwOp.get_status(),
                                                       blockName,
                                                       "operation descriptor",
                                                       node.op_name,
                                                       error_message,
                                                       pwOp.get_error()))
                                return pwOp.get_status();

                            ops.emplace_back(std::move(pwOp));
                        } else {
                            // Create a Scale or Bias Node for Weight1 * Input + Bias
                            auto pwOp =
                                cudnn_frontend::OperationBuilder(node.type)
                                    .setxDesc(
                                        *(tensor_map.at(node.edges[0])))  // Input tensor (output of prev conv node)
                                    .setbDesc(
                                        *(tensor_map.at(node.edges[1])))  // Scale/bias tensor (output of prev BN node)
                                    .setyDesc(*(tensor_map.at(node.edges[2])))  // Output tensor (most likely virtual)
                                    .setpwDesc(*pointwise_descriptor)           // PW problem descriptor
                                    .build();

                            if (checkErrorStatusAndLog(pwOp.get_status(),
                                                       blockName,
                                                       "operation descriptor",
                                                       node.op_name,
                                                       error_message,
                                                       pwOp.get_error()))
                                return pwOp.get_status();

                            ops.emplace_back(std::move(pwOp));
                        }

#ifndef NV_CUDNN_DISABLE_EXCEPTION
                    } catch (cudnn_frontend::cudnnException &e) {
                        logErrorMessage(e.getCudnnStatus(),
                                        blockName,
                                        "operation descriptor",
                                        node.op_name,
                                        error_message,
                                        e.what());
                        throw cudnnException(error_message.c_str(), e.getCudnnStatus());
                    }
#endif
                    getLogger() << "[cudnn_frontend] "
                                << "INFO: Built Pointwise " << node.op_name << std::endl;
                } break;

                // A reduction descriptor can be a scale, a bias, or an activation (ReLu) operation.
                case cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR: {
                    // Check if the problem name exists in the problem map
                    if (reduction_problems.find(node.problem_name) == reduction_problems.end()) {
                        error_message = "ERROR: Reduction problem " + node.problem_name + " not found!";
                        return CUDNN_STATUS_EXECUTION_FAILED;
                    }

#ifndef NV_CUDNN_DISABLE_EXCEPTION
                    try {
#endif

                        // Create a Scale or Bias Node for Weight1 * Input + Bias
                        auto reduction_operation =
                            cudnn_frontend::OperationBuilder(node.type)
                                .setxDesc(*(tensor_map.at(node.edges[0])))  // Input tensor (output of prev conv node)
                                .setyDesc(*(tensor_map.at(node.edges[1])))  // amax
                                .setreductionDesc(
                                    *(reduction_problems.at(node.problem_name)))  // reduction problem descriptor
                                .build();

                        if (checkErrorStatusAndLog(reduction_operation.get_status(),
                                                   blockName,
                                                   "operation descriptor",
                                                   node.op_name,
                                                   error_message,
                                                   reduction_operation.get_error()))
                            return reduction_operation.get_status();

                        ops.emplace_back(std::move(reduction_operation));

#ifndef NV_CUDNN_DISABLE_EXCEPTION
                    } catch (cudnn_frontend::cudnnException &e) {
                        logErrorMessage(e.getCudnnStatus(),
                                        blockName,
                                        "operation descriptor",
                                        node.op_name,
                                        error_message,
                                        e.what());
                        throw cudnnException(error_message.c_str(), e.getCudnnStatus());
                    }
#endif
                    getLogger() << "[cudnn_frontend] "
                                << "INFO: Built Reduction " << node.op_name << std::endl;
                } break;
                default:
                    getLogger() << "[cudnn_frontend] "
                                << "WARN: Unhandled Node type for node: " << node.op_name << std::endl;
                    break;
            }
        }

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Builds the operation graph with the vector of operations
            auto opGraph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(ops).build();

            if (checkErrorStatusAndLog(opGraph.get_status(), blockName, "operation graph", graph_name, error_message))
                return opGraph.get_status();

            // Add the op graph to the vector of operation graphs
            op_graphs.emplace(graph_name, std::move(opGraph));

            getLogger() << "[cudnn_frontend] "
                        << "Done building Operation Graph!" << std::endl;

            return CUDNN_STATUS_SUCCESS;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            logErrorMessage(e.getCudnnStatus(), blockName, "operation graph", graph_name, error_message, e.what());
            throw cudnnException(error_message.c_str(), e.getCudnnStatus());
        }
#endif
    };

    void
    addConvProblem(const std::string &name, std::shared_ptr<ConvDesc> const convolution_problem) {
        convolution_problems.emplace(name, convolution_problem);
    };

    void
    addPointwiseProblem(const std::string &name, std::shared_ptr<PointwiseDesc> const pointwise_problem) {
        pointwise_problems.emplace(name, pointwise_problem);
    };

    void
    addReductionProblem(const std::string &name, std::shared_ptr<ReductionDesc> const reduction_problem) {
        reduction_problems.emplace(name, reduction_problem);
    };
};

}  // namespace cudnn_frontend