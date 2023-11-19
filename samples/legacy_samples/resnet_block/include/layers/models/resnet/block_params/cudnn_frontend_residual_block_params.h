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

namespace cudnn_frontend {
/**
 * @brief A class that encapsulates the parameters of a residual block. Has a corresponding builder
 * class that can be used to construct an instance of this class and set up the parameters from user side.
 *
 */
class ResidualBlockParams {
   public:
    friend class ResidualForwardBlock;
    friend class ResidualForwardInferenceBlock;
    friend class ResidualBlockParamsBuilder;

    ResidualBlockParams(ResidualBlockParams&& from) = default;
    ResidualBlockParams&
    operator=(ResidualBlockParams&& from)           = default;
    ResidualBlockParams(ResidualBlockParams const&) = default;
    ResidualBlockParams()                           = default;
    ResidualBlockParams&
    operator=(ResidualBlockParams const&) = default;

    ~ResidualBlockParams() = default;
    enum ForwardLocation { ZERO, ONE, RESIDUAL, TWO, COUNT };

    bool
    skip_residual_convolution(int loc) {
        return (loc == ForwardLocation::RESIDUAL) && (false == has_residual_convolution);
    }

   private:
    bool has_residual_convolution = false;

    template <typename T, size_t N = ForwardLocation::COUNT>
    using parameter_container = std::array<T, N>;

    // Used for fprop, dgrad, wgrad
    parameter_container<convolution_params> conv_params;

    // Used for BN, BN-Relu, BN-Add-Relu
    parameter_container<bn_fusion_params> bn_params;

    // Used for transpose
    parameter_container<pointwise_parameters> weight_transpose_params;
    parameter_container<pointwise_parameters> x_transpose_params;
    parameter_container<pointwise_parameters> dy_transpose_params;
};

/// ResidualBlockParamsBuilder Class
/// Helper class used to build ResidualBlockParams class.
/// NOTE: The number of conv dims and number of input dims defaults to 2 and 4 respectively. If you want to change it,
/// set the conv dims and input dims BEFORE you set the input size, filter size, stride size, etc.
class ResidualBlockParamsBuilder {
   private:
    std::vector<int64_t> input_size_;
    std::array<std::vector<int64_t>, ResidualBlockParams::ForwardLocation::COUNT> convolution_params_;

   public:
    ResidualBlockParamsBuilder&
    setInputDim(std::vector<int64_t> const& input_size) {
        input_size_ = input_size;
        return *this;
    }

    ResidualBlockParamsBuilder&
    setConvolutionParams(ResidualBlockParams::ForwardLocation loc, std::vector<int64_t> const& convolution_params) {
        if (loc == ResidualBlockParams::ForwardLocation::RESIDUAL) {
            residualBlockParams.has_residual_convolution = true;
        }

        convolution_params_[loc] = convolution_params;
        return *this;
    }

    ResidualBlockParams
    build() {
        validate_();
        return residualBlockParams;
    }

    explicit ResidualBlockParamsBuilder()                         = default;
    ~ResidualBlockParamsBuilder()                                 = default;
    ResidualBlockParamsBuilder(ResidualBlockParamsBuilder&&)      = delete;
    ResidualBlockParamsBuilder(ResidualBlockParamsBuilder const&) = delete;
    ResidualBlockParamsBuilder&
    operator=(ResidualBlockParamsBuilder const&) = default;

   private:
    ResidualBlockParams residualBlockParams;

    void
    validate_() {
        // Set convolution parameters
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (residualBlockParams.skip_residual_convolution(i)) {
                continue;
            }

            auto& convolution_params = residualBlockParams.conv_params[i];

            // first and residual convolution take input size
            if (i == ResidualBlockParams::ForwardLocation::RESIDUAL ||
                i == ResidualBlockParams::ForwardLocation::ZERO) {
                // Create parameters with default values
                convolution_params = make_convolution_params(i * 100,
                                                             input_size_[0],
                                                             input_size_[1],
                                                             input_size_[2],
                                                             input_size_[3],
                                                             convolution_params_[i][0],
                                                             convolution_params_[i][1],
                                                             convolution_params_[i][2],
                                                             convolution_params_[i][3],
                                                             convolution_params_[i][4]);
            } else {
                auto& previous_convolution_params =
                    residualBlockParams.conv_params[i == ResidualBlockParams::ForwardLocation::ONE
                                                        ? ResidualBlockParams::ForwardLocation::ZERO
                                                        : ResidualBlockParams::ForwardLocation::ONE];
                convolution_params = make_convolution_params(i * 100,
                                                             previous_convolution_params.output_dim[0],
                                                             previous_convolution_params.output_dim[1],
                                                             previous_convolution_params.output_dim[2],
                                                             previous_convolution_params.output_dim[3],
                                                             convolution_params_[i][0],
                                                             convolution_params_[i][1],
                                                             convolution_params_[i][2],
                                                             convolution_params_[i][3],
                                                             convolution_params_[i][4]);
            }
        }

        // Set batch norm parameters
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (residualBlockParams.skip_residual_convolution(i)) {
                continue;
            }

            auto& batch_norm_params = residualBlockParams.bn_params[i];

            // Create parameters with default values
            batch_norm_params = make_bn_fusion_params(1000 + i * 100,
                                                      residualBlockParams.conv_params[i].output_dim[0],
                                                      residualBlockParams.conv_params[i].output_dim[1],
                                                      residualBlockParams.conv_params[i].output_dim[2],
                                                      residualBlockParams.conv_params[i].output_dim[3]);

            // Set relu for first, second, third
            if (i != ResidualBlockParams::ForwardLocation::RESIDUAL) {
                batch_norm_params.has_relu = true;
            }

            // Set add relu for third
            if (i == ResidualBlockParams::ForwardLocation::TWO) {
                batch_norm_params.has_add_relu = true;
            }
        }

        // Set weight transpose parameters for dgrad
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (residualBlockParams.skip_residual_convolution(i)) {
                continue;
            }

            auto& weight_transpose_params = residualBlockParams.weight_transpose_params[i];

            // Create parameters with default values
            weight_transpose_params = make_pointwise_parameters(2000 + i * 100,
                                                                residualBlockParams.conv_params[i].weight_dim[0],
                                                                residualBlockParams.conv_params[i].weight_dim[1],
                                                                residualBlockParams.conv_params[i].weight_dim[2],
                                                                residualBlockParams.conv_params[i].weight_dim[3]);
        }

        // Set x transpose parameters for wgrad
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (residualBlockParams.skip_residual_convolution(i)) {
                continue;
            }

            auto& x_transpose_params = residualBlockParams.x_transpose_params[i];

            // Create parameters with default values
            x_transpose_params = make_pointwise_parameters(3000 + i * 100,
                                                           residualBlockParams.conv_params[i].input_dim[0],
                                                           residualBlockParams.conv_params[i].input_dim[1],
                                                           residualBlockParams.conv_params[i].input_dim[2],
                                                           residualBlockParams.conv_params[i].input_dim[3]);
        }

        // Set dy transpose parameters for wgrad
        for (int i = ResidualBlockParams::ForwardLocation::ZERO; i < ResidualBlockParams::ForwardLocation::COUNT; i++) {
            if (residualBlockParams.skip_residual_convolution(i)) {
                continue;
            }

            auto& dy_transpose_params = residualBlockParams.dy_transpose_params[i];

            // Create parameters with default values
            dy_transpose_params = make_pointwise_parameters(4000 + i * 100,
                                                            residualBlockParams.conv_params[i].output_dim[0],
                                                            residualBlockParams.conv_params[i].output_dim[1],
                                                            residualBlockParams.conv_params[i].output_dim[2],
                                                            residualBlockParams.conv_params[i].output_dim[3]);
        }
    }

    void
    setErrorAndThrowException(cudnnStatus_t status, const std::string& message) {
        std::string error;
        logErrorMessage(
            status, "Residual Block Params Builder", "params builder", "residual params builder", error, message);
        getLogger() << "[cudnn_frontend] " << error << std::endl;
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(error.c_str(), status);
#endif
    }
};

}  // namespace cudnn_frontend