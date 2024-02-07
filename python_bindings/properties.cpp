#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend {

namespace python_bindings {

void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const& error_msg);

class HandleManagement {
   public:
    static void*
    create_handle() {
        cudnnHandle_t handle;
        cudnnCreate(&handle);
        return (void*)handle;
    }

    static void
    destroy_handle(void* handle) {
        auto status = cudnnDestroy((cudnnHandle_t)handle);
        throw_if(
            status != CUDNN_STATUS_SUCCESS, cudnn_frontend::error_code_t::HANDLE_ERROR, "cudnnHandle Destroy failed");
    }

    static void
    set_stream(void* handle, void* stream) {
        auto status = cudnnSetStream((cudnnHandle_t)handle, (cudaStream_t)stream);
        throw_if(status != CUDNN_STATUS_SUCCESS, cudnn_frontend::error_code_t::HANDLE_ERROR, "cudnnSetStream failed");
    }

    static void
    get_stream(void* handle, void* streamId) {
        auto status = cudnnGetStream((cudnnHandle_t)handle, (cudaStream_t*)streamId);
        throw_if(status != CUDNN_STATUS_SUCCESS, cudnn_frontend::error_code_t::HANDLE_ERROR, "cudnnGetStream failed");
    }
};

void
init_properties(py::module_& m) {
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

    py::class_<cudnn_frontend::graph::Tensor_attributes, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>(
        m, "tensor")
        .def(py::init<>())
        .def("get_name", &cudnn_frontend::graph::Tensor_attributes::get_name)
        .def("set_name", &cudnn_frontend::graph::Tensor_attributes::set_name)
        .def("get_data_type", &cudnn_frontend::graph::Tensor_attributes::get_data_type)
        .def("set_data_type", &cudnn_frontend::graph::Tensor_attributes::set_data_type)
        .def("get_dim", &cudnn_frontend::graph::Tensor_attributes::get_dim)
        .def("set_dim", &cudnn_frontend::graph::Tensor_attributes::set_dim)
        .def("get_stride", &cudnn_frontend::graph::Tensor_attributes::get_stride)
        .def("set_stride", &cudnn_frontend::graph::Tensor_attributes::set_stride)
        .def("get_is_virtual", &cudnn_frontend::graph::Tensor_attributes::get_is_virtual)
        .def("set_is_virtual", &cudnn_frontend::graph::Tensor_attributes::set_is_virtual)
        .def(
            "set_output",
            [](cudnn_frontend::graph::Tensor_attributes& self,
               bool const is_output) -> cudnn_frontend::graph::Tensor_attributes& {
                self.set_is_virtual(!is_output);
                return self;
            },
            py::return_value_policy::reference)  // NOTICE THATS ITS JUST ANOTHER NAME FOR SET_IS_VIRTUAL
        .def("get_is_pass_by_value", &cudnn_frontend::graph::Tensor_attributes::get_is_pass_by_value)
        .def("set_is_pass_by_value", &cudnn_frontend::graph::Tensor_attributes::set_is_pass_by_value)
        .def("get_uid", &cudnn_frontend::graph::Tensor_attributes::get_uid)
        .def("set_uid", &cudnn_frontend::graph::Tensor_attributes::set_uid)
        .def("set_ragged_offset", &cudnn_frontend::graph::Tensor_attributes::set_ragged_offset)
        .def("__repr__", [](cudnn_frontend::graph::Tensor_attributes const& props) {
            std::ostringstream out;
            out << json{props};
            return out.str();
        });

    m.def("create_handle", &HandleManagement::create_handle);
    m.def("destroy_handle", &HandleManagement::destroy_handle);
    m.def("get_stream", &HandleManagement::get_stream);
    m.def(
        "set_stream",
        [](void* handle, int64_t stream) { return HandleManagement::set_stream(handle, (void*)stream); },
        py::arg("handle"),
        py::arg("stream"));

    py::enum_<cudnn_frontend::NormFwdPhase_t>(m, "norm_forward_phase")
        .value("INFERENCE", cudnn_frontend::NormFwdPhase_t::INFERENCE)
        .value("TRAINING", cudnn_frontend::NormFwdPhase_t::TRAINING)
        .value("NOT_SET", cudnn_frontend::NormFwdPhase_t::NOT_SET);

    py::enum_<cudnn_frontend::HeurMode_t>(m, "heur_mode")
        .value("A", cudnn_frontend::HeurMode_t::A)
        .value("B", cudnn_frontend::HeurMode_t::B)
        .value("FALLBACK", cudnn_frontend::HeurMode_t::FALLBACK);

    py::enum_<cudnn_frontend::ReductionMode_t>(m, "reduction_mode")
        .value("ADD", cudnn_frontend::ReductionMode_t::ADD)
        .value("MUL", cudnn_frontend::ReductionMode_t::MUL)
        .value("MIN", cudnn_frontend::ReductionMode_t::MIN)
        .value("MAX", cudnn_frontend::ReductionMode_t::MAX)
        .value("AMAX", cudnn_frontend::ReductionMode_t::AMAX)
        .value("AVG", cudnn_frontend::ReductionMode_t::AVG)
        .value("NORM1", cudnn_frontend::ReductionMode_t::NORM1)
        .value("NORM2", cudnn_frontend::ReductionMode_t::NORM2)
        .value("MUL_NO_ZEROS", cudnn_frontend::ReductionMode_t::MUL_NO_ZEROS)
        .value("NOT_SET", cudnn_frontend::ReductionMode_t::NOT_SET);

    py::enum_<cudnn_frontend::BuildPlanPolicy_t>(m, "build_plan_policy")
        .value("HEURISTICS_CHOICE", cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE)
        .value("ALL", cudnn_frontend::BuildPlanPolicy_t::ALL);

    py::enum_<cudnn_frontend::NumericalNote_t>(m, "numerical_note")
        .value("TENSOR_CORE", cudnn_frontend::NumericalNote_t::TENSOR_CORE)
        .value("DOWN_CONVERT_INPUTS", cudnn_frontend::NumericalNote_t::DOWN_CONVERT_INPUTS)
        .value("REDUCED_PRECISION_REDUCTION", cudnn_frontend::NumericalNote_t::REDUCED_PRECISION_REDUCTION)
        .value("FFT", cudnn_frontend::NumericalNote_t::FFT)
        .value("NONDETERMINISTIC", cudnn_frontend::NumericalNote_t::NONDETERMINISTIC)
        .value("WINOGRAD", cudnn_frontend::NumericalNote_t::WINOGRAD)
        .value("WINOGRAD_TILE_4x4", cudnn_frontend::NumericalNote_t::WINOGRAD_TILE_4x4)
        .value("WINOGRAD_TILE_6x6", cudnn_frontend::NumericalNote_t::WINOGRAD_TILE_6x6)
        .value("WINOGRAD_TILE_13x13", cudnn_frontend::NumericalNote_t::WINOGRAD_TILE_13x13);

    py::enum_<cudnn_frontend::BehaviorNote_t>(m, "behavior_note")
        .value("RUNTIME_COMPILATION", cudnn_frontend::BehaviorNote_t::RUNTIME_COMPILATION)
        .value("REQUIRES_FILTER_INT8x32_REORDER", cudnn_frontend::BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER)
        .value("REQUIRES_BIAS_INT8x32_REORDER", cudnn_frontend::BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER);
}

}  // namespace python_bindings
}  // namespace cudnn_frontend