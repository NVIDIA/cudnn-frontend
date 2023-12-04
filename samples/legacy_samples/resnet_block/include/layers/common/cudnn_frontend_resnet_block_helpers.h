#pragma once

#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Forward declare block classes

namespace cudnn_frontend {

class ResidualForwardInferenceBlock;
class ResidualForwardBlock;
class ResidualBlockParamsBuilder;
class ResidualBlockDevPtrStore;

static size_t
compute_tensor_size(int64_t const *const arr, int64_t const n) {
    size_t initialProduct = 1;
    return accumulate(arr, arr + n, initialProduct, std::multiplies<int64_t>());
}

static void
generateStrides(const int64_t *dimA, int64_t *strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat) {
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strideA[nbDims - 1] = 1;
        for (int64_t d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1]          = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int64_t d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

static void
generate4dTransposeStrides(const int64_t *dimA, int64_t *strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat) {
    (void)nbDims;
    (void)filterFormat;
    // Here we assume that the format is NWHC getting tranposed to CHWN
    strideA[0] = 1;                     // N has stride 1
    strideA[3] = strideA[0] * dimA[0];  // W has stride strideN * dimN
    strideA[2] = strideA[3] * dimA[3];  // H has stride strideW * dimW
    strideA[1] = strideA[2] * dimA[2];  // C has stride strideH * dimH
}
static constexpr std::array<int64_t, 4> amax_dim_stride = {1, 1, 1, 1};
static constexpr cudnnDataType_t data_type              = CUDNN_DATA_FLOAT;

/**
 * A struct to store the parameters of a convolution node in the block.
 *
 */

struct ConvParams {
    int64_t xDim[CUDNN_DIM_MAX + 1] = {-1};
    int64_t wDim[CUDNN_DIM_MAX + 1] = {-1};
    int64_t yDim[CUDNN_DIM_MAX + 1] = {-1};

    // Conv dim (1D or 2D or 3D etc.) Defaults to 2D
    int64_t nConvDims                       = 2;
    cudnnConvolutionMode_t convMode         = CUDNN_CROSS_CORRELATION;
    int64_t convPad[CUDNN_DIM_MAX + 1]      = {-1};
    int64_t convStride[CUDNN_DIM_MAX + 1]   = {-1};
    int64_t convDilation[CUDNN_DIM_MAX + 1] = {-1};

    int64_t perChannelScaleBiasDim[CUDNN_DIM_MAX + 1] = {-1};

    cudnnDataType_t dataType = CUDNN_DATA_HALF;
};

typedef struct convolution_params_ {
    // Main convolution
    int64_t dim_count;
    int64_t input_dim[CUDNN_DIM_MAX];
    int64_t weight_dim[CUDNN_DIM_MAX];
    int64_t output_dim[CUDNN_DIM_MAX];

    int64_t input_stride[CUDNN_DIM_MAX];
    int64_t weight_stride[CUDNN_DIM_MAX];
    int64_t output_stride[CUDNN_DIM_MAX];

    int64_t padding[CUDNN_DIM_MAX];
    int64_t stride[CUDNN_DIM_MAX];
    int64_t dilation[CUDNN_DIM_MAX];

    cudnnDataType_t tensor_data_type;
    cudnnDataType_t compute_type;

    enum UIDs {
        INPUT_UID,
        WEIGHT_UID,
        OUTPUT_UID,

        INPUT_DESCALE_UID,
        WEIGHT_DESCALE_UID,
        OUTPUT_SCALE_UID,

        OUTPUT_AMAX_UID,

        /* VIRTUAL */
        AFTER_CONV_UID,
        AFTER_INPUT_DESCALE_UID,
        AFTER_WEIGHT_DESCALE_UID,

        UID_COUNT
    };

    using uids_t = int64_t;
    uids_t uids[UIDs::UID_COUNT];

    /*
       Sizes of input, weight and output tensor
       Assuming non-overlapping strides and fully packed tensor.
    */

    size_t
    getInputTensorSize() {
        return compute_tensor_size(input_dim, dim_count);
    }

    size_t
    getWeightTensorSize() {
        return compute_tensor_size(weight_dim, dim_count);
    }

    size_t
    getOutputTensorSize() {
        return compute_tensor_size(output_dim, dim_count);
    }

} convolution_params;

static convolution_params
make_convolution_params(convolution_params::uids_t const uid_offset,
                        int64_t const N,
                        int64_t const C,
                        int64_t const H,
                        int64_t const W,
                        int64_t const K,
                        int64_t const R,
                        int64_t const S,
                        int64_t const pad,
                        int64_t const stride) {
    return {/* dim count  */ 4,
            /* input dim  */ {N, C, H, W},
            /* weight dim */ {K, C, R, S},
            /* output dim */
            {N,
             K,
             static_cast<int64_t>(std::ceil((H + (2 * pad) - R + 1) / stride)),
             static_cast<int64_t>(std::ceil((W + (2 * pad) - S + 1) / stride))},
            /* input stride  */ {-1, -1, -1, -1},  // Will be auto calculated later when creating Tensor
            /* weight stride */ {-1, -1, -1, -1},  // Will be auto calculated later when creating Tensor
            /* output stride */ {-1, -1, -1, -1},  // Will be auto calculated later when creating Tensor
            /* padding */ {pad, pad},
            /* stride */ {stride, stride},
            /* dilation */ {1, 1},
            /* tensor data type */ CUDNN_DATA_FP8_E4M3,
            /* compute type */ CUDNN_DATA_FLOAT,
            /* uids */
            {
                uid_offset + convolution_params::UIDs::INPUT_UID,
                uid_offset + convolution_params::UIDs::WEIGHT_UID,
                uid_offset + convolution_params::UIDs::OUTPUT_UID,
                uid_offset + convolution_params::UIDs::INPUT_DESCALE_UID,
                uid_offset + convolution_params::UIDs::WEIGHT_DESCALE_UID,
                uid_offset + convolution_params::UIDs::OUTPUT_SCALE_UID,
                uid_offset + convolution_params::UIDs::OUTPUT_AMAX_UID,
                uid_offset + convolution_params::UIDs::AFTER_CONV_UID,
                uid_offset + convolution_params::UIDs::AFTER_INPUT_DESCALE_UID,
                uid_offset + convolution_params::UIDs::AFTER_WEIGHT_DESCALE_UID,
            }};
}

typedef struct bn_fusion_params_ {
    int64_t dim_count;
    int64_t input_dim[CUDNN_DIM_MAX];
    int64_t per_channel_dim[CUDNN_DIM_MAX];
    int64_t input_strides[CUDNN_DIM_MAX];
    int64_t per_channel_strides[CUDNN_DIM_MAX];

    size_t
    getInputTensorSize() {
        return compute_tensor_size(input_dim, dim_count);
    }
    size_t
    getOutputTensorSize() {
        return compute_tensor_size(input_dim, dim_count);
    }
    size_t
    getPerChannelTensorSize() {
        return compute_tensor_size(per_channel_dim, dim_count);
    }

    bool has_add_relu;
    bool has_relu;

    cudnnDataType_t compute_type;
    cudnnDataType_t data_type;

    enum UIDs {
        INPUT_UID,
        INPUT_HP_UID,
        INPUT_DESCALE_UID,
        WEIGHT_HP_UID,
        WEIGHT_UID,
        WEIGHT_DESCALE_UID,
        INPUT_SCALE_UID,  // This is per channel scale not FP8
        INPUT_BIAS_UID,
        OUTPUT_MEAN_UID,
        OUTPUT_INV_VAR_UID,
        IN_RUNNING_MEAN_UID,
        IN_RUNNING_INV_VAR_UID,
        OUT_RUNNING_MEAN_UID,
        OUT_RUNNING_INV_VAR_UID,
        OUTPUT_DSCALE_UID,
        OUTPUT_DBIAS_UID,
        OUTPUT_UID,        // FP8 OUT Tensor
        OUTPUT_HP_UID,     // FP32 OUT Tensor
        OUTPUT_SCALE_UID,  // This is an input
        OUTPUT_AMAX_UID,
        EPSILON_UID,
        EXP_AVG_FACTOR_UID,

        ADD_TENSOR_UID,
        ADD_TENSOR_DESCALE_UID,
        /* VIRTUAL */
        BEFORE_ACTIVATION_UID,
        AFTER_ACTIVATION_UID,
        BN_OUTPUT_UID,  // UID of the HP BN OUTPUT
        ADD_TENSOR_HP_UID,

        INPUT_SUB_MEAN_UID,
        NORM_INPUT_UID,
        INPUT_MUL_SCALE_UID,
        INPUT_ADD_BIAS_UID,
        MEAN_UID,
        VAR_UID,
        VAR_ADD_EPS_UID,
        RSQRT_VAR_UID,

        UID_COUNT
    };

    using uids_t = int64_t;
    uids_t uids[UIDs::UID_COUNT];
} bn_fusion_params;

static bn_fusion_params
make_bn_fusion_params(bn_fusion_params::uids_t uid_offset,
                      int64_t const N,
                      int64_t const C,
                      int64_t const H,
                      int64_t const W) {
    return {/* dim count  */ 4,
            /* input dim  */ {N, C, H, W},
            /* channel dim  */ {1, C, 1, 1},
            /* input stride */ {-1, -1, -1, -1},
            /* channel stride */ {-1, -1, -1, -1},
            /* has add relu */ false,
            /* has relu */ false,
            /* compute_type */ CUDNN_DATA_FLOAT,
            /* data_type */ CUDNN_DATA_FP8_E4M3,
            /* uids */
            {uid_offset + bn_fusion_params::UIDs::INPUT_UID,
             uid_offset + bn_fusion_params::UIDs::INPUT_HP_UID,
             uid_offset + bn_fusion_params::UIDs::INPUT_DESCALE_UID,
             uid_offset + bn_fusion_params::UIDs::WEIGHT_HP_UID,
             uid_offset + bn_fusion_params::UIDs::WEIGHT_UID,
             uid_offset + bn_fusion_params::UIDs::WEIGHT_DESCALE_UID,
             uid_offset + bn_fusion_params::UIDs::INPUT_SCALE_UID,
             uid_offset + bn_fusion_params::UIDs::INPUT_BIAS_UID,
             uid_offset + bn_fusion_params::UIDs::OUTPUT_MEAN_UID,
             uid_offset + bn_fusion_params::UIDs::OUTPUT_INV_VAR_UID,
             uid_offset + bn_fusion_params::UIDs::IN_RUNNING_MEAN_UID,
             uid_offset + bn_fusion_params::UIDs::IN_RUNNING_INV_VAR_UID,
             uid_offset + bn_fusion_params::UIDs::OUT_RUNNING_MEAN_UID,
             uid_offset + bn_fusion_params::UIDs::OUT_RUNNING_INV_VAR_UID,
             uid_offset + bn_fusion_params::UIDs::OUTPUT_DSCALE_UID,
             uid_offset + bn_fusion_params::UIDs::OUTPUT_DBIAS_UID,
             uid_offset + bn_fusion_params::UIDs::OUTPUT_UID,
             uid_offset + bn_fusion_params::UIDs::OUTPUT_HP_UID,
             uid_offset + bn_fusion_params::UIDs::OUTPUT_SCALE_UID,
             uid_offset + bn_fusion_params::UIDs::OUTPUT_AMAX_UID,
             uid_offset + bn_fusion_params::UIDs::EPSILON_UID,
             uid_offset + bn_fusion_params::UIDs::EXP_AVG_FACTOR_UID,
             uid_offset + bn_fusion_params::UIDs::ADD_TENSOR_UID,
             uid_offset + bn_fusion_params::UIDs::ADD_TENSOR_DESCALE_UID,
             uid_offset + bn_fusion_params::UIDs::BEFORE_ACTIVATION_UID,
             uid_offset + bn_fusion_params::UIDs::AFTER_ACTIVATION_UID,
             uid_offset + bn_fusion_params::UIDs::BN_OUTPUT_UID,
             uid_offset + bn_fusion_params::UIDs::ADD_TENSOR_HP_UID,
             uid_offset + bn_fusion_params::UIDs::INPUT_SUB_MEAN_UID,
             uid_offset + bn_fusion_params::UIDs::NORM_INPUT_UID,
             uid_offset + bn_fusion_params::UIDs::INPUT_MUL_SCALE_UID,
             uid_offset + bn_fusion_params::UIDs::INPUT_ADD_BIAS_UID,
             uid_offset + bn_fusion_params::UIDs::MEAN_UID,
             uid_offset + bn_fusion_params::UIDs::VAR_UID,
             uid_offset + bn_fusion_params::UIDs::VAR_ADD_EPS_UID,
             uid_offset + bn_fusion_params::UIDs::RSQRT_VAR_UID}};
}

typedef struct pointwise_parameters_ {
    int64_t dim_count;

    int64_t input_dim[CUDNN_DIM_MAX];
    int64_t weight_dim[CUDNN_DIM_MAX];
    int64_t output_dim[CUDNN_DIM_MAX];

    int64_t input_stride[CUDNN_DIM_MAX];
    int64_t weight_stride[CUDNN_DIM_MAX];
    int64_t output_stride[CUDNN_DIM_MAX];

    cudnnDataType_t tensor_data_type;
    cudnnDataType_t compute_type;

    enum UIDs {
        INPUT_UID,
        WEIGHT_UID,
        OUTPUT_UID,

        INPUT_DESCALE_UID,
        WEIGHT_DESCALE_UID,
        OUTPUT_SCALE_UID,

        OUTPUT_AMAX_UID,

        /* VIRTUAL */
        AFTER_POINTWISE_UID,
        AFTER_INPUT_DESCALE_UID,
        AFTER_WEIGHT_DESCALE_UID,

        UID_COUNT
    };

    using uids_t = int64_t;
    uids_t uids[UIDs::UID_COUNT];
} pointwise_parameters;

static pointwise_parameters
make_pointwise_parameters(pointwise_parameters::uids_t uid_offset,
                          int64_t const N,
                          int64_t const C,
                          int64_t const H,
                          int64_t const W) {
    return {/* dim count  */ 4,
            /* input dim  */ {N, C, H, W},
            /* weight dim  */ {-1, -1, -1, -1},
            /* output dim  */ {N, C, H, W},
            /* input stride */ {-1, -1, -1, -1},
            /* weight stride */ {-1, -1, -1, -1},
            /* output stride */ {-1, -1, -1, -1},
            /* data_type */ CUDNN_DATA_FP8_E4M3,
            /* compute_type */ CUDNN_DATA_FLOAT,
            /* uids */
            {uid_offset + pointwise_parameters::UIDs::INPUT_UID,
             uid_offset + pointwise_parameters::UIDs::WEIGHT_UID,
             uid_offset + pointwise_parameters::UIDs::OUTPUT_UID,
             uid_offset + pointwise_parameters::UIDs::INPUT_DESCALE_UID,
             uid_offset + pointwise_parameters::UIDs::WEIGHT_DESCALE_UID,
             uid_offset + pointwise_parameters::UIDs::OUTPUT_SCALE_UID,
             uid_offset + pointwise_parameters::UIDs::OUTPUT_AMAX_UID,
             uid_offset + pointwise_parameters::UIDs::AFTER_POINTWISE_UID,
             uid_offset + pointwise_parameters::UIDs::AFTER_INPUT_DESCALE_UID,
             uid_offset + pointwise_parameters::UIDs::AFTER_WEIGHT_DESCALE_UID}};
}

/**
 * A struct for the Gen Stats/BN finalize nodes in the network.
 */

struct StatsParams {
    int64_t sumDim[CUDNN_DIM_MAX + 1]       = {-1};
    int64_t accumCnt                        = 0;
    int64_t epsilonDim[CUDNN_DIM_MAX + 1]   = {-1};
    double epsilon                          = 0.05;
    double expDecayFactor                   = 0.9;
    cudnnGenStatsMode_t mode                = CUDNN_GENSTATS_SUM_SQSUM;
    cudnnBnFinalizeStatsMode_t finalizeMode = CUDNN_BN_FINALIZE_STATISTICS_TRAINING;
    cudnnDataType_t dataType                = CUDNN_DATA_FLOAT;
};

struct PoolingParams {
    cudnn_frontend::ResampleMode_t mode        = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
    cudnn_frontend::PaddingMode_t padding_mode = cudnn_frontend::PaddingMode_t::ZERO_PAD;
    cudnnNanPropagation_t nanOpt               = CUDNN_PROPAGATE_NAN;

    int32_t nbSpatialDims              = 2;
    int64_t windowDim[CUDNN_DIM_MAX]   = {-1};
    int64_t prePadding[CUDNN_DIM_MAX]  = {0, 0};
    int64_t postPadding[CUDNN_DIM_MAX] = {0, 0};
    int64_t stride[CUDNN_DIM_MAX]      = {-1};
    int64_t outputDim[CUDNN_DIM_MAX]   = {-1};
    double alpha                       = 1.0;
    double beta                        = 0.0;
};

/**
 * An enum that represents the different conv nodes in the residual block.
 * The first 3 are the direct path conv nodes in order of 0, 1, 2. The last one is the residual conv node.
 * This allows for easy indexing into a vector of conv nodes with this format.
 */
enum ConvNode { DP_FIRST_NODE = 0, DP_SECOND_NODE = 1, DP_THIRD_NODE = 2, RESIDUAL_NODE = 3 };

/**
 * A struct containing hardcoded UIDs for all the necessary tensors in the Residual Block.
 *
 */
struct ResidualUIDs {
    std::vector<int64_t> xUIDs              = {4, 5, 6, 7};
    std::vector<int64_t> yUIDs              = {8, 9, 10, 11};
    std::vector<int64_t> wUIDs              = {12, 13, 14, 15};
    std::vector<int64_t> sumUIDs            = {16, 17, 18, 19};
    std::vector<int64_t> sqSumUIDs          = {20, 21, 22, 23};
    std::vector<int64_t> rpUIDs             = {24, 25};
    std::vector<int64_t> dpUIDs             = {26, 27};
    std::vector<int64_t> outScaleUIDs       = {28, 29, 30, 31};
    std::vector<int64_t> outBiasUIDs        = {32, 33, 34, 35};
    std::vector<int64_t> epsilonUIDs        = {36, 37, 38, 39};
    std::vector<int64_t> expDecayFactorUIDs = {40, 41, 42, 43};
    std::vector<int64_t> accumCntUIDs       = {44, 45, 46, 47};

    std::vector<int64_t> inScaleUIDs     = {48, 49, 50, 51};
    std::vector<int64_t> inBiasUIDs      = {52, 53, 54, 55};
    std::vector<int64_t> inMeanUIDs      = {56, 57, 58, 59};
    std::vector<int64_t> inVarUIDs       = {60, 61, 62, 63};
    std::vector<int64_t> outMeanUIDs     = {64, 65, 66, 67};
    std::vector<int64_t> outVarUIDs      = {68, 69, 70, 71};
    std::vector<int64_t> savedMeanUIDs   = {72, 73, 74, 75};
    std::vector<int64_t> savedInvVarUIDs = {76, 77, 78, 79};

    std::vector<int64_t> afterScaleUIDs = {80, 81, 82, 83};
    std::vector<int64_t> afterBiasUIDs  = {84, 85, 86, 87};

    std::vector<int64_t> afterConvDataGradUIDs = {92, 93, 94, 95};
    std::vector<int64_t> afterConvWGradUIDs    = {96, 97, 98, 99};
    std::vector<int64_t> afterReluGradUIDs     = {100, 101, 102, 103};
    std::vector<int64_t> inputReluGradUIDs     = {104, 105, 106, 107};
    std::vector<int64_t> afterBNUIDs           = {108, 109, 110, 111};
    std::vector<int64_t> afterBNGradUIDs       = {112, 113, 114, 115};
    std::vector<int64_t> afterReluUIDs         = {116, 117, 118, 119};
    std::vector<int64_t> reluBitmaskUIDs       = {120, 121, 122, 123};

    int64_t inputGradUID           = 199;
    int64_t afterAddUID            = 200;
    int64_t finalOutputUID         = 201;
    int64_t finalBackwardOutputUID = 202;
    int64_t zeroUID                = 203;
};

/**
 * @brief Helper function to calculate output size of pooling
 *
 * @param input_size Input size to the pooling operation
 * @param window_size Window size used for the pooling
 * @param stride Stride for the pooling
 * @param padding Padding for the pooling (usually 0)
 * @param dilation Dilation for the pooling
 * @return int64_t
 */
static inline int64_t
calculatePoolingOutputSize(const int64_t &input_size,
                           const int64_t &window_size,
                           const int64_t &stride,
                           const int64_t &padding) {
    return (input_size + 2 * padding - window_size) / stride + 1;
}

// Helper functions to calculate device ptr sizes.
static inline int64_t
calculateSize(const int64_t *dims, int numDims) {
    int64_t size = 1;
    for (int i = 0; i < numDims; i++) {
        size *= dims[i];
    }
    return size;
}

static inline bool
isDimensionMatch(const int64_t *A, const int64_t *B, int numDims) {
    for (int i = 0; i < numDims; i++) {
        if (A[i] != B[i]) return false;
    }
    return true;
}

/**
 * @brief C++ implementation of Python instanceof method. Checks to see whether or not an object is instance of some
 * class.
 *
 * @tparam Base Base class to compare object against
 * @tparam T Generic type of object to be compared
 * @param object The actual object to be compared with type T. Must pass in a pointer to the object.
 * @return true True if object is of type Base class. False if object is not of type Base class.
 */
template <typename Base, typename T>
static inline bool instanceof (const T *object) {
    return dynamic_cast<const Base *>(object) != nullptr;
}

/**
 * @brief Helper error logging function Uses a generic type T for the block calling the error, that way the user knows
 * what block the error is happening. It sets a passed in reference error_message to be used later by the user or
 * internally.
 *
 * @param blockName The name of the block as a string.
 * @param descriptorType A string denoting a descriptor type. This allows for more specific error messages. Descriptor
 * type examples include "tensor", "problem descriptor", "operation graph", "execution plan" etc.
 * @param descriptorName A name of the descriptor. Example: "Input X tensor" with descriptor type "tensor"
 * @param error_message A reference to a string to log the error message to.
 * @param extraMessage Any extra message to be appeneded to the error message. Usually this is an exception message, or
 * CUDNN descriptor get_error() string. Defaults to empty string.
 * @return bool Returns true if there was an error. The error message gets set. Returns false if there is no error (e.g.
 * status == CUDNN_STATUS_SUCCESS)
 */
static inline void
logErrorMessage(cudnnStatus_t status,
                const std::string &blockName,
                const std::string &descriptorType,
                const std::string &descriptorName,
                std::string &errorMessage,
                const std::string &extraMessage = "") {
    if (descriptorType == "execution plan") {
        errorMessage =
            "[ERROR]: CUDNN Error in exeucting plan with name: " + descriptorName + "! This error happens in the " +
            blockName +
            "! Double check your parameters/device pointers. CUDNN error string: " + cudnnGetErrorString(status) + " " +
            extraMessage;

    } else if (descriptorType == "params builder") {
        errorMessage = "[ERROR]: Error in creating : " + descriptorType + " with name: " + descriptorName +
                       "! This error happens in the " + blockName +
                       "! Double check your parameters and read error string to see what's wrong. Error string: " +
                       cudnnGetErrorString(status) + " " + extraMessage;

    } else if (descriptorType == "device pointer store") {
        errorMessage = "[ERROR]: Error in creating : " + descriptorType + " with name: " + descriptorName +
                       "! This error happens in the " + blockName +
                       "! Double check that your device pointers are set correctly and all necessary device pointers "
                       "are set. Read error string to see what's wrong. Error string: " +
                       cudnnGetErrorString(status) + " " + extraMessage;

    } else {
        errorMessage =
            "[ERROR]: CUDNN Error in creating descriptor type: " + descriptorType + " with name: " + descriptorName +
            "! This error happens in the " + blockName +
            "! Double check your parameters/device pointers. CUDNN error string: " + cudnnGetErrorString(status) + " " +
            extraMessage;
    }
}

/**
 * @brief Helper error logging function to check for errors based on a cudnnStatus_t. Used internally when
 * NV_CUDNN_DISABLE_EXCEPTION is enabled since exceptions won't be thrown. Instead, this function is used to log
 * helpeful error messages. Uses a generic type T for the block calling the error, that way the user knows what block
 * the error is happening. It sets a passed in reference error_message to be used later by the user or internally.
 *
 * @param status A cudnnStatus_t to check.
 * @param blockName The name of the block as a string.
 * @param descriptorType A string denoting a descriptor type. This allows for more specific error messages. Descriptor
 * type examples include "tensor", "problem descriptor", "operation graph", "execution plan" etc.
 * @param descriptorName A name of the descriptor. Example: "Input X tensor" with descriptor type "tensor"
 * @param errorMessage A reference to a string to log the error message to.
 * @param extraMessage Any extra message to be appeneded to the error message. Usually this is an exception message, or
 * CUDNN descriptor get_error() string. Defaults to empty string.
 * @return bool Returns true if there was an error. The error message gets set. Returns false if there is no error (e.g.
 * status == CUDNN_STATUS_SUCCESS)
 */
static inline bool
checkErrorStatusAndLog(cudnnStatus_t status,
                       const std::string &blockName,
                       const std::string &descriptorType,
                       const std::string &descriptorName,
                       std::string &errorMessage,
                       const std::string &extraMessage = "") {
    if (status != CUDNN_STATUS_SUCCESS) {
        logErrorMessage(status, blockName, descriptorType, descriptorName, errorMessage, extraMessage);
        return true;
    }
    return false;
}

}  // namespace cudnn_frontend
