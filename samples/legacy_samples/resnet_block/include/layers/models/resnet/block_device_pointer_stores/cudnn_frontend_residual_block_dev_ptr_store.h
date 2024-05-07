#pragma once

#include <vector>
#include <utility>

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

#ifdef WIN32
#define strncasecmp strnicmp
#endif

namespace cudnn_frontend {

/**
 * @brief A class that stores the device pointers of the residual block. The user is able to set the device
 * pointers needed for the block, or can autoallocate if needed.
 *
 */
class ResidualBlockDevPtrStore {
    friend class ResidualForwardInferenceBlock;
    friend class ResidualForwardBlock;
    friend class ResidualBlockParams;

   public:
    ResidualBlockDevPtrStore() = default;

    ResidualBlockDevPtrStore&
    setXDevPtr(void* const devPtr) {
        XDevPtr = devPtr;
        return *this;
    }

    ResidualBlockDevPtrStore&
    setYDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            YDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setWeightNHWCDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            weight_nhwc_device_pointers[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNXDescaleDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            BNXDescaleDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNYDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            BNYDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNYScaleDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            BNYScaleDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNYAMaxDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            BNYAMaxDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNInScaleDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            scaleDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNInBiasDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            biasDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNSavedMeanDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            savedMeanDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNSavedInvVarDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            savedInvVarDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNRunningMeanDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            running_mean_DevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setBNRunningVarDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            running_var_DevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setXDescaleDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            XDescaleDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setWDescaleDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            WDescaleDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setYScaleDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            YScaleDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    ResidualBlockDevPtrStore&
    setYAmaxDevPtrs(std::vector<std::pair<ResidualBlockParams::ForwardLocation, void* const>> const& devPtrs) {
        for (auto const& p : devPtrs) {
            YAmaxDevPtrs[p.first] = p.second;
        }
        return *this;
    }

    auto
    setBNZDeScale(void* ptr) -> ResidualBlockDevPtrStore& {
        zDescale = ptr;
        return *this;
    }

    auto
    setBNEpsilons(std::vector<std::pair<ResidualBlockParams::ForwardLocation, float>> const& epsilons)
        -> ResidualBlockDevPtrStore& {
        for (auto const& p : epsilons) {
            BN_epsilons[p.first] = p.second;
        }
        return *this;
    }
    auto
    setBNExponentialAverageFactors(std::vector<std::pair<ResidualBlockParams::ForwardLocation, float>> const&
                                       exp_avg_factors) -> ResidualBlockDevPtrStore& {
        for (auto const& p : exp_avg_factors) {
            BN_exp_avg_factors[p.first] = p.second;
        }
        return *this;
    }

    cudnnStatus_t
    finalize() {
        // Inputs to convolutions are the output from previous batch norm subgraphs
        XDevPtrs[ResidualBlockParams::ForwardLocation::ONE] = BNYDevPtrs[ResidualBlockParams::ForwardLocation::ZERO];
        XDevPtrs[ResidualBlockParams::ForwardLocation::TWO] = BNYDevPtrs[ResidualBlockParams::ForwardLocation::ONE];

        // Set inputs to residual conv and first one to be the input to block
        XDevPtrs[ResidualBlockParams::ForwardLocation::ZERO]     = XDevPtr;
        XDevPtrs[ResidualBlockParams::ForwardLocation::RESIDUAL] = XDevPtr;

        // The same DQ of block input X is applied on residual conv too
        XDescaleDevPtrs[ResidualBlockParams::ForwardLocation::RESIDUAL] =
            XDescaleDevPtrs[ResidualBlockParams::ForwardLocation::ZERO];

        return CUDNN_STATUS_SUCCESS;
    }

    /**
     * @brief Get the cudnnStatus_t of the builder
     *
     * @return cudnnStatus_t The current status of the builder
     */
    cudnnStatus_t
    getStatus() const {
        return status_;
    }

    /**
     * @brief Get the error message of the block
     *
     * @return const std::string& The error message of the block as a string
     */
    const std::string&
    getErrorMessage() const {
        return errorMessage;
    }

    cudnnStatus_t
    setWorkspace(const std::string& blockType, void* workspace) {
        if (strncasecmp(blockType.c_str(), "forward", blockType.size()) == 0) {
            workspace_forward = workspace;
            return CUDNN_STATUS_SUCCESS;
        } else if (strncasecmp(blockType.c_str(), "backward", blockType.size()) == 0) {
            workspace_backward = workspace;
            return CUDNN_STATUS_SUCCESS;
        } else if (strncasecmp(blockType.c_str(), "forward_inference", blockType.size()) == 0) {
            workspace_forward_inference = workspace;
            return CUDNN_STATUS_SUCCESS;
        } else if (strncasecmp(blockType.c_str(), "backward_mixed_precision", blockType.size()) == 0) {
            workspace_backward_mixed_precision = workspace;
            return CUDNN_STATUS_SUCCESS;
        }

        return CUDNN_STATUS_NOT_SUPPORTED;
    }

    void*
    getWorkspace(const IBlock::Direction& direction) {
        if (direction == IBlock::Direction::FORWARD) {
            return workspace_forward;
        } else if (direction == IBlock::Direction::BACKWARD) {
            return workspace_backward;
        } else if (direction == IBlock::Direction::FORWARD_INFERENCE) {
            return workspace_forward_inference;
        } else if (direction == IBlock::Direction::BACKWARD_MIXED_PRECISION) {
            return workspace_backward_mixed_precision;
        }
        return nullptr;
    }

   private:
    int NUM_CONV_NODES = 4;

    // Conv
    void* XDevPtr;
    std::vector<void*> XDevPtrs                    = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> weight_nhwc_device_pointers = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> YDevPtrs                    = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> XDescaleDevPtrs             = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> WDescaleDevPtrs             = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> YScaleDevPtrs               = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> YAmaxDevPtrs                = std::vector<void*>(NUM_CONV_NODES, nullptr);

    std::vector<void*> BNXDescaleDevPtrs = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> BNYDevPtrs        = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> BNYScaleDevPtrs   = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> BNYAMaxDevPtrs    = std::vector<void*>(NUM_CONV_NODES, nullptr);

    std::vector<void*> scaleDevPtrs         = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> biasDevPtrs          = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> running_mean_DevPtrs = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> running_var_DevPtrs  = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> savedMeanDevPtrs     = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<void*> savedInvVarDevPtrs   = std::vector<void*>(NUM_CONV_NODES, nullptr);
    std::vector<float> BN_epsilons          = std::vector<float>(NUM_CONV_NODES, 0);
    std::vector<float> BN_exp_avg_factors   = std::vector<float>(NUM_CONV_NODES, 0);

    void* zDescale;

    ///////////////////////////////
    // Workspaces
    ///////////////////////////////

    void* workspace_forward;
    void* workspace_backward;
    void* workspace_backward_mixed_precision;
    void* workspace_forward_inference;

    std::string errorMessage = "SUCCESS";
    cudnnStatus_t status_    = CUDNN_STATUS_SUCCESS;

    /**
     * @brief Set the error and throw exception object for the params builder
     *
     * @param status a cudnnStatus_t value indicating the current status of the builder
     * @param message An error message to log to the user
     */
    void
    setErrorAndThrowException(cudnnStatus_t status, const std::string& message) {
        status_ = status;

        logErrorMessage(status,
                        "Residual Block Dev Ptr Store",
                        "device pointer store",
                        "residual block dev ptr store",
                        errorMessage,
                        message);

        getLogger() << "[cudnn_frontend] " << errorMessage << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(errorMessage.c_str(), status);
#endif
    }
};

}  // namespace cudnn_frontend

#undef strncasecmp