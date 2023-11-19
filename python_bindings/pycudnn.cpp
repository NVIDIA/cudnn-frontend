#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend {
namespace python_bindings {

// Raise C++ exceptions corresponding to C++ FE error codes.
// Pybinds will automatically convert C++ exceptions to pythpn exceptions.
void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const &error_msg) {
    if (cond == false) return;

    switch (error_code) {
        case cudnn_frontend::error_code_t::OK:
            return;
        case cudnn_frontend::error_code_t::ATTRIBUTE_NOT_SET:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::SHAPE_DEDUCTION_FAILED:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::INVALID_TENSOR_NAME:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::INVALID_VARIANT_PACK:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_EXECUTION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::HEURISTIC_QUERY_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::CUDNN_BACKEND_API_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::CUDA_API_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::INVALID_CUDA_DEVICE:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::UNSUPPORTED_GRAPH_FORMAT:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_NOT_SUPPORTED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::HANDLE_ERROR:
            throw std::runtime_error(error_msg);
    }
}

// pybinds for pygraph class
void
init_pygraph_submodule(py::module_ &);

// pybinds for all properties and helpers
void
init_properties(py::module_ &);

PYBIND11_MODULE(cudnn, m) {
    m.def("backend_version", &cudnnGetVersion);

    init_properties(m);
    init_pygraph_submodule(m);
}

}  // namespace python_bindings
}  // namespace cudnn_frontend