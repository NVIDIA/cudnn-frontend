#pragma once

#include <unordered_map>
#include <vector>

#include <iomanip>
#include <unordered_set>
#include <algorithm>
#include <string>

namespace cudnn_frontend {

enum class [[nodiscard]] error_code_t{OK,
                                      ATTRIBUTE_NOT_SET,
                                      SHAPE_DEDUCTION_FAILED,
                                      INVALID_TENSOR_NAME,
                                      INVALID_VARIANT_PACK,
                                      GRAPH_NOT_SUPPORTED,
                                      GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                      GRAPH_EXECUTION_FAILED,
                                      HEURISTIC_QUERY_FAILED,
                                      UNSUPPORTED_GRAPH_FORMAT,
                                      CUDA_API_FAILED,
                                      CUDNN_BACKEND_API_FAILED,
                                      INVALID_CUDA_DEVICE,
                                      HANDLE_ERROR};

typedef struct [[nodiscard]] error_object {
    error_code_t code;
    std::string err_msg;
    error_object() : code(error_code_t::OK), err_msg(""){};
    error_object(error_code_t err, std::string msg) : code(err), err_msg(msg){};

    error_code_t
    get_code() {
        return code;
    }

    std::string
    get_message() {
        return err_msg;
    }

    bool
    is_good() const {
        return code == error_code_t::OK;
    }

    bool
    is_bad() const {
        return !is_good();
    }

    bool
    operator==(error_code_t compare_code) {
        return code == compare_code;
    }

    bool
    operator!=(error_code_t compare_code) {
        return code != compare_code;
    }

} error_t;

#ifdef WIN32
#define CUDNN_FRONTEND_WHILE_FALSE \
    __pragma(warning(push)) __pragma(warning(disable : 4127)) while (0) __pragma(warning(pop))
#else
#define CUDNN_FRONTEND_WHILE_FALSE while (0)
#endif

#define CHECK_CUDNN_FRONTEND_ERROR(x)                                                                              \
    do {                                                                                                           \
        if (auto retval = x; retval.is_bad()) {                                                                    \
            getLogger() << "[cudnn_frontend] ERROR: " << #x << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return retval;                                                                                         \
        }                                                                                                          \
    }                                                                                                              \
    CUDNN_FRONTEND_WHILE_FALSE

#define RETURN_CUDNN_FRONTEND_ERROR_IF(cond, retval, message)                                                        \
    do {                                                                                                             \
        if (cond) {                                                                                                  \
            if (retval == error_code_t::OK) {                                                                        \
                getLogger() << "[cudnn_frontend] INFO: ";                                                            \
            } else {                                                                                                 \
                getLogger() << "[cudnn_frontend] ERROR: ";                                                           \
            }                                                                                                        \
            getLogger() << message << ". " << retval << " because (" << #cond ") at " << __FILE__ << ":" << __LINE__ \
                        << "\n";                                                                                     \
            return {retval, message};                                                                                \
        }                                                                                                            \
    }                                                                                                                \
    CUDNN_FRONTEND_WHILE_FALSE

#define CHECK_CUDNN_ERROR(x)                                                                                      \
    do {                                                                                                          \
        if (auto cudnn_retval = x; cudnn_retval != CUDNN_STATUS_SUCCESS) {                                        \
            std::stringstream error_msg;                                                                          \
            error_msg << #x << " failed with " << cudnnGetErrorString(cudnn_retval);                              \
            getLogger() << "[cudnn_frontend] ERROR: " << error_msg.str() << " at " << __FILE__ << ":" << __LINE__ \
                        << std::endl;                                                                             \
            return {error_code_t::CUDNN_BACKEND_API_FAILED, error_msg.str()};                                     \
        }                                                                                                         \
    }                                                                                                             \
    CUDNN_FRONTEND_WHILE_FALSE

#define CHECK_CUDA_ERROR(x)                                                                                       \
    do {                                                                                                          \
        if (auto cuda_retval = x; cuda_retval != cudaSuccess) {                                                   \
            std::stringstream error_msg;                                                                          \
            error_msg << #x << " failed with " << cudaGetErrorString(cuda_retval);                                \
            getLogger() << "[cudnn_frontend] ERROR: " << error_msg.str() << " at " << __FILE__ << ":" << __LINE__ \
                        << std::endl;                                                                             \
            return {error_code_t::CUDA_API_FAILED, error_msg.str()};                                              \
        }                                                                                                         \
    }                                                                                                             \
    CUDNN_FRONTEND_WHILE_FALSE

NLOHMANN_JSON_SERIALIZE_ENUM(error_code_t,
                             {
                                 {error_code_t::OK, "OK"},
                                 {error_code_t::ATTRIBUTE_NOT_SET, "ATTRIBUTE_NOT_SET"},
                                 {error_code_t::SHAPE_DEDUCTION_FAILED, "SHAPE_DEDUCTION_FAILED"},
                                 {error_code_t::INVALID_TENSOR_NAME, "INVALID_TENSOR_NAME"},
                                 {error_code_t::INVALID_VARIANT_PACK, "INVALID_VARIANT_PACK"},
                                 {error_code_t::GRAPH_NOT_SUPPORTED, "GRAPH_NOT_SUPPORTED"},
                                 {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                  "GRAPH_EXECUTION_PLAN_CREATION_FAILED"},
                                 {error_code_t::GRAPH_EXECUTION_FAILED, "GRAPH_EXECUTION_FAILED"},
                                 {error_code_t::HEURISTIC_QUERY_FAILED, "HEURISTIC_QUERY_FAILED"},
                                 {error_code_t::CUDNN_BACKEND_API_FAILED, "CUDNN_BACKEND_API_FAILED"},
                                 {error_code_t::CUDA_API_FAILED, "CUDA_API_FAILED"},
                                 {error_code_t::INVALID_CUDA_DEVICE, "INVALID_CUDA_DEVICE"},
                                 {error_code_t::UNSUPPORTED_GRAPH_FORMAT, "UNSUPPORTED_GRAPH_FORMAT"},
                                 {error_code_t::HANDLE_ERROR, "HANDLE_ERROR"},
                             })

static inline std::ostream&
operator<<(std::ostream& os, const error_code_t& mode) {
    os << json{mode};
    return os;
}

static inline std::ostream&
operator<<(std::ostream& os, cudnn_frontend::error_object& err) {
    os << err.get_code() << err.get_message();
    return os;
}

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

namespace detail {

inline bool
is_activation_backward_mode(PointwiseMode_t const mode) {
    return ((mode == PointwiseMode_t::RELU_BWD) || (mode == PointwiseMode_t::TANH_BWD) ||
            (mode == PointwiseMode_t::SIGMOID_BWD) || (mode == PointwiseMode_t::ELU_BWD) ||
            (mode == PointwiseMode_t::GELU_BWD) || (mode == PointwiseMode_t::GELU_APPROX_TANH_BWD) ||
            (mode == PointwiseMode_t::SOFTPLUS_BWD) || (mode == PointwiseMode_t::SWISH_BWD));
}

// Creates dense, non-overlapping strides from given dim and stride_order.
// For example, if a is a 4D tensor with dimensions labeled NCHW, then strided(a, (3, 0, 2, 1)) produces
// strides where the C dimension has a corresponding stride of one.
inline std::vector<int64_t>
generate_stride(std::vector<int64_t> const& dim, std::vector<int64_t> const& stride_order) {
    size_t num_dims = dim.size();
    std::vector<int64_t> stride(num_dims);

    // Sort the dimensions according to strides from least to greatest.
    // Example, dim = (2, 3, 4, 5) stride_order = (3, 1, 2, 0)
    // sorted_stride_order = ((0, (3, 5)), (1, (1, 3)), (2, (2, 4)), (3, (0, 2)))
    std::vector<std::pair<int64_t, std::pair<size_t, size_t>>> sorted_stride_order;
    for (size_t i = 0; i < num_dims; ++i) {
        sorted_stride_order.push_back({stride_order[i], {i, dim[i]}});
    }
    std::sort(sorted_stride_order.begin(), sorted_stride_order.end());

    // As dims have now been ordered starting from fastest changing,
    // just fill in strides by iterating linearly over them.
    int64_t product = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        stride[sorted_stride_order[i].second.first] = product;
        product *= sorted_stride_order[i].second.second;
    }

    return stride;
}

// Generate NHWC stride_order
inline std::vector<int64_t>
generate_NHWC_stride_order(int64_t const num_dims) {
    std::vector<int64_t> stride_order(num_dims);

    int64_t order   = 0;
    stride_order[1] = order++;
    for (size_t i = num_dims - 1; i > 1; --i) {
        stride_order[i] = order++;
    }
    stride_order[0] = order;

    return stride_order;
}

// Generate row major stride_order for matrices
// dim = (*, M, N) where * is batch dimsensions
// strides should be (..., N, 1)
inline std::vector<int64_t>
generate_row_major_stride_order(int64_t const num_dims) {
    std::vector<int64_t> stride_order(num_dims);

    int64_t order = num_dims - 1;
    std::generate(stride_order.begin(), stride_order.end(), [&order] { return order--; });

    return stride_order;
}

// Generate column major stride_order for matrices
// dim = (M, N)
// strides should be (1, M)
inline std::vector<int64_t>
generate_column_major_stride_order(int64_t const num_dims) {
    std::vector<int64_t> stride_order(num_dims);

    int64_t order = 1;
    std::generate(stride_order.begin(), stride_order.end(), [&order] { return order++; });

    return stride_order;
}

}  // namespace detail

}  // namespace cudnn_frontend