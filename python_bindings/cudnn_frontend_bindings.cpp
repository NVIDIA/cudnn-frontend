#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cudnn_frontend;

// pybinds for pygraph class
void
init_pygraph_submodule(py::module_ &);

// pybinds for all properties and helpers
void
init_properties(py::module_ &);

PYBIND11_MODULE(cudnn, m) {
    m.def("get_cudnn_version", &cudnnGetVersion);

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

    init_pygraph_submodule(m);
    init_properties(m);
}
