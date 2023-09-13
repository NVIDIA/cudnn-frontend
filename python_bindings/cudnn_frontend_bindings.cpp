#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cudnn_frontend;

namespace cudnn_frontend {
namespace python_bindings {

// pybinds for pygraph class
void
init_pygraph_submodule(py::module_ &);

// pybinds for all properties and helpers
void
init_properties(py::module_ &);

void *
create_handle();

void
destroy_handle(void *);

PYBIND11_MODULE(cudnn, m) {
    m.def("backend_version", &cudnnGetVersion);
    m.def("create_handle", &create_handle);
    m.def("destroy_handle", &destroy_handle);

    py::enum_<cudnn_frontend::DataType_t>(m, "data_type")
        .value("FLOAT", cudnn_frontend::DataType_t::FLOAT)
        .value("DOUBLE", cudnn_frontend::DataType_t::DOUBLE)
        .value("HALF", cudnn_frontend::DataType_t::HALF)
        .value("INT8", cudnn_frontend::DataType_t::INT8)
        .value("INT32", cudnn_frontend::DataType_t::INT32)
        .value("INT8x4", cudnn_frontend::DataType_t::INT8x4)
        .value("UINT8", cudnn_frontend::DataType_t::UINT8)
        .value("UINT8x4", cudnn_frontend::DataType_t::UINT8x4)
        .value("INT8x32", cudnn_frontend::DataType_t::INT8x32)
        .value("BFLOAT16", cudnn_frontend::DataType_t::BFLOAT16)
        .value("INT64", cudnn_frontend::DataType_t::INT64)
        .value("BOOLEAN", cudnn_frontend::DataType_t::BOOLEAN)
        .value("FP8_E4M3", cudnn_frontend::DataType_t::FP8_E4M3)
        .value("FP8_E5M2", cudnn_frontend::DataType_t::FP8_E5M2)
        .value("FAST_FLOAT_FOR_FP8", cudnn_frontend::DataType_t::FAST_FLOAT_FOR_FP8)
        .value("NOT_SET", cudnn_frontend::DataType_t::NOT_SET);

    py::enum_<cudnn_frontend::NormFwdPhase_t>(m, "norm_forward_phase")
        .value("INFERENCE", cudnn_frontend::NormFwdPhase_t::INFERENCE)
        .value("TRAINING", cudnn_frontend::NormFwdPhase_t::TRAINING)
        .value("NOT_SET", cudnn_frontend::NormFwdPhase_t::NOT_SET);

    py::enum_<cudnn_frontend::ReductionMode_t>(m, "reduction_mode")
        .value("ADD", cudnn_frontend::ReductionMode_t::ADD)
        .value("MUL", cudnn_frontend::ReductionMode_t::MUL)
        .value("MIN", cudnn_frontend::ReductionMode_t::MIN)
        .value("MAX", cudnn_frontend::ReductionMode_t::MAX)
        .value("AMAX", cudnn_frontend::ReductionMode_t::AMAX)
        .value("AVG", cudnn_frontend::ReductionMode_t::AVG)
        .value("NORM1", cudnn_frontend::ReductionMode_t::NORM1)
        .value("NORM2", cudnn_frontend::ReductionMode_t::NORM2)
        .value("MUL_NO_ZEROS", cudnn_frontend::ReductionMode_t::MUL_NO_ZEROS);

    init_pygraph_submodule(m);
    init_properties(m);
}

}  // namespace python_bindings
}  // namespace cudnn_frontend