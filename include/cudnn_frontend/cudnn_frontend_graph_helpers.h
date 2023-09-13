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
                                      INVALID_CUDA_DEVICE,
                                      HANDLE_ERROR};

typedef struct error_object {
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

#define CHECK_CUDNN_FRONTEND_ERROR(x)                                                                                \
    do {                                                                                                             \
        if (x.is_bad()) {                                                                                            \
            getLogger() << "[cudnn_frontend] ERROR: " << #x << " code " << x.get_code() << " at " << __FILE__ << ":" \
                        << __LINE__ << std::endl;                                                                    \
            return x;                                                                                                \
        }                                                                                                            \
    } while (0)

#define RETURN_CUDNN_FRONTEND_ERROR_IF(cond, retval)                                                              \
    do {                                                                                                          \
        if (cond) {                                                                                               \
            if (retval.get_code() == error_code_t::OK) {                                                          \
                getLogger() << "[cudnn_frontend] INFO: " << #cond << " returned " << retval.get_code() << " at "  \
                            << __FILE__ << ":" << __LINE__ << std::endl;                                          \
            } else {                                                                                              \
                getLogger() << "[cudnn_frontend] ERROR: " << #cond << " returned " << retval.get_code() << " at " \
                            << __FILE__ << ":" << __LINE__ << std::endl;                                          \
            }                                                                                                     \
            return {retval};                                                                                      \
        }                                                                                                         \
    } while (0)

static inline std::ostream&
operator<<(std::ostream& os, const error_code_t& mode) {
    switch (mode) {
        case error_code_t::OK:
            os << "OK";
            break;
        case error_code_t::ATTRIBUTE_NOT_SET:
            os << "ATTRIBUTE_NOT_SET";
            break;
        case error_code_t::SHAPE_DEDUCTION_FAILED:
            os << "SHAPE_DEDUCTION_FAILED";
            break;
        case error_code_t::INVALID_TENSOR_NAME:
            os << "INVALID_TENSOR_NAME";
            break;
        case error_code_t::INVALID_VARIANT_PACK:
            os << "INVALID_VARIANT_PACK";
            break;
        case error_code_t::GRAPH_NOT_SUPPORTED:
            os << "GRAPH_NOT_SUPPORTED";
            break;
        case error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
            os << "GRAPH_EXECUTION_PLAN_CREATION_FAILED";
            break;
        case error_code_t::GRAPH_EXECUTION_FAILED:
            os << "GRAPH_EXECUTION_FAILED";
            break;
        case error_code_t::HEURISTIC_QUERY_FAILED:
            os << "HEURISTIC_QUERY_FAILED";
            break;
        case error_code_t::INVALID_CUDA_DEVICE:
            os << "INVALID_CUDA_DEVICE";
            break;
        case error_code_t::UNSUPPORTED_GRAPH_FORMAT:
            os << "UNSUPPORTED_GRAPH_FORMAT";
            break;
        case error_code_t::HANDLE_ERROR:
            os << "HANDLE_ERROR";
            break;
    }
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

class Context {
    DataType_t compute_data_type      = DataType_t::NOT_SET;
    DataType_t intermediate_data_type = DataType_t::NOT_SET;
    DataType_t io_data_type           = DataType_t::NOT_SET;

   public:
    Context&
    set_intermediate_data_type(DataType_t const type) {
        intermediate_data_type = type;
        return *this;
    }

    Context&
    set_io_data_type(DataType_t const type) {
        io_data_type = type;
        return *this;
    }

    Context&
    set_compute_data_type(DataType_t const type) {
        compute_data_type = type;
        return *this;
    }

    DataType_t
    get_io_data_type() const {
        return io_data_type;
    }

    DataType_t
    get_intermediate_data_type() const {
        return intermediate_data_type;
    }

    DataType_t
    get_compute_data_type() const {
        return compute_data_type;
    }

    Context&
    fill_missing_properties(Context const& global_context) {
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(global_context.get_compute_data_type());
        }
        if (get_intermediate_data_type() == DataType_t::NOT_SET) {
            set_intermediate_data_type(global_context.get_intermediate_data_type());
        }
        if (get_io_data_type() == DataType_t::NOT_SET) {
            set_io_data_type(global_context.get_io_data_type());
        }
        return *this;
    }
};

// Always generates NCHW (4d/5d tensors) or Col major (matrices)
inline std::vector<int64_t>
generate_stride(std::vector<int64_t> const& dim) {
    std::vector<int64_t> stride(dim.size(), 1);

    stride[dim.size() - 1] = stride[1] * dim[1];
    for (int64_t d = dim.size() - 2; d >= 2; d--) {
        stride[d] = stride[d + 1] * dim[d + 1];
    }
    stride[0] = stride[2] * dim[2];

    return stride;
}

}  // namespace detail

}  // namespace cudnn_frontend