#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend {

namespace python_bindings {

void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const& error_msg);

class HandleManagement {
   public:
    static std::intptr_t
    create_handle() {
        cudnnHandle_t handle;
        auto status = detail::create_handle(&handle);
        throw_if(
            status != CUDNN_STATUS_SUCCESS, cudnn_frontend::error_code_t::HANDLE_ERROR, "cudnnHandle Create failed");
        return reinterpret_cast<std::intptr_t>(handle);
    }

    static void
    destroy_handle(std::intptr_t handle) {
        auto status = detail::destroy_handle((cudnnHandle_t)handle);
        throw_if(
            status != CUDNN_STATUS_SUCCESS, cudnn_frontend::error_code_t::HANDLE_ERROR, "cudnnHandle Destroy failed");
    }

    static void
    set_stream(std::intptr_t handle, std::intptr_t stream) {
        auto status = detail::set_stream((cudnnHandle_t)handle, (cudaStream_t)stream);
        throw_if(status != CUDNN_STATUS_SUCCESS, cudnn_frontend::error_code_t::HANDLE_ERROR, "cudnnSetStream failed");
    }

    static std::intptr_t
    get_stream(std::intptr_t handle) {
        cudaStream_t streamId = nullptr;
        auto status           = detail::get_stream((cudnnHandle_t)handle, &streamId);
        throw_if(status != CUDNN_STATUS_SUCCESS, cudnn_frontend::error_code_t::HANDLE_ERROR, "cudnnGetStream failed");

        return reinterpret_cast<std::intptr_t>(streamId);
    }
};

std::shared_ptr<cudnn_frontend::KernelCache>
create_kernel_cache_helper() {
    auto kernel_cache = std::make_shared<cudnn_frontend::KernelCache>();
    throw_if(kernel_cache == nullptr, cudnn_frontend::error_code_t::INVALID_VALUE, "kernel cache creation failed");
    return kernel_cache;
}

std::string
kernel_cache_to_json_helper(std::shared_ptr<cudnn_frontend::KernelCache> const& kernel_cache) {
    std::string str_json;
    auto err = kernel_cache->to_json(str_json);
    throw_if(err.is_bad(), err.code, err.get_message());
    return str_json;
}

void
kernel_cache_from_json_helper(std::shared_ptr<cudnn_frontend::KernelCache> kernel_cache, std::string const& str_json) {
    auto err = kernel_cache->from_json(str_json);
    throw_if(err.is_bad(), err.code, err.get_message());
}

static std::string
get_last_error_string() {
    return detail::get_last_error_string_();
}

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
        .value("FP8_E8M0", cudnn_frontend::DataType_t::FP8_E8M0)
        .value("FP4_E2M1", cudnn_frontend::DataType_t::FP4_E2M1)
        .value("INT4", cudnn_frontend::DataType_t::INT4)
        .value("NOT_SET", cudnn_frontend::DataType_t::NOT_SET);

    py::enum_<cudnn_frontend::TensorReordering_t>(m, "tensor_reordering")
        .value("NONE", cudnn_frontend::TensorReordering_t::NONE)
        .value("INT8x32", cudnn_frontend::TensorReordering_t::INT8x32)
        .value("F16x16", cudnn_frontend::TensorReordering_t::F16x16)
        .value("F8_128x4", cudnn_frontend::TensorReordering_t::F8_128x4);

    py::class_<cudnn_frontend::graph::Tensor_attributes, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>(
        m, "tensor")
        .def(py::init<>())
        .def("get_name", &cudnn_frontend::graph::Tensor_attributes::get_name)
        .def("set_name", &cudnn_frontend::graph::Tensor_attributes::set_name)
        .def("get_data_type", &cudnn_frontend::graph::Tensor_attributes::get_data_type)
        .def("_set_data_type", &cudnn_frontend::graph::Tensor_attributes::set_data_type)
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

    py::enum_<cudnn_frontend::KnobType_t>(m, "knob_type")
        .value("NOT_SET", cudnn_frontend::KnobType_t::NOT_SET)
        .value("SWIZZLE", cudnn_frontend::KnobType_t::SWIZZLE)
        .value("TILE_SIZE", cudnn_frontend::KnobType_t::TILE_SIZE)
        .value("EDGE", cudnn_frontend::KnobType_t::EDGE)
        .value("MULTIPLY", cudnn_frontend::KnobType_t::MULTIPLY)
        .value("SPLIT_K_BUF", cudnn_frontend::KnobType_t::SPLIT_K_BUF)
        .value("TILEK", cudnn_frontend::KnobType_t::TILEK)
        .value("STAGES", cudnn_frontend::KnobType_t::STAGES)
        .value("REDUCTION_MODE", cudnn_frontend::KnobType_t::REDUCTION_MODE)
        .value("SPLIT_K_SLC", cudnn_frontend::KnobType_t::SPLIT_K_SLC)
        .value("IDX_MODE", cudnn_frontend::KnobType_t::IDX_MODE)
        .value("SPECFILT", cudnn_frontend::KnobType_t::SPECFILT)
        .value("KERNEL_CFG", cudnn_frontend::KnobType_t::KERNEL_CFG)
        .value("WORKSPACE", cudnn_frontend::KnobType_t::WORKSPACE)
        .value("TILE_CGA_M", cudnn_frontend::KnobType_t::TILE_CGA_M)
        .value("TILE_CGA_N", cudnn_frontend::KnobType_t::TILE_CGA_N)
        .value("BLOCK_SIZE", cudnn_frontend::KnobType_t::BLOCK_SIZE)
        .value("OCCUPANCY", cudnn_frontend::KnobType_t::OCCUPANCY)
        .value("ARRAY_SIZE_PER_THREAD", cudnn_frontend::KnobType_t::ARRAY_SIZE_PER_THREAD)
        .value("SPLIT_COLS", cudnn_frontend::KnobType_t::SPLIT_COLS)
        .value("TILE_ROWS", cudnn_frontend::KnobType_t::TILE_ROWS)
        .value("TILE_COLS", cudnn_frontend::KnobType_t::TILE_COLS)
        .value("LOAD_SIZE", cudnn_frontend::KnobType_t::LOAD_SIZE)
        .value("CTA_COUNT", cudnn_frontend::KnobType_t::CTA_COUNT)
        .value("STREAM_K", cudnn_frontend::KnobType_t::STREAM_K)
        .value("SPLIT_P_SLC", cudnn_frontend::KnobType_t::SPLIT_P_SLC)
        .value("TILE_M", cudnn_frontend::KnobType_t::TILE_M)
        .value("TILE_N", cudnn_frontend::KnobType_t::TILE_N)
        .value("WARP_SPEC_CFG", cudnn_frontend::KnobType_t::WARP_SPEC_CFG);

    py::class_<cudnn_frontend::Knob, std::shared_ptr<cudnn_frontend::Knob>>(m, "knob")
        .def(py::init<cudnn_frontend::KnobType_t, int64_t, int64_t, int64_t>(),
             py::arg_v("knob_type", cudnn_frontend::KnobType_t::NOT_SET),
             py::arg_v("max_value", py::none()),
             py::arg_v("min_value", py::none()),
             py::arg_v("stride", py::none()))
        .def_readonly("type", &cudnn_frontend::Knob::type)
        .def_readonly("max_value", &cudnn_frontend::Knob::maxValue)
        .def_readonly("min_value", &cudnn_frontend::Knob::minValue)
        .def_readonly("stride", &cudnn_frontend::Knob::stride)
        .def("__repr__", [](cudnn_frontend::Knob const& knob) {
            std::stringstream ss;
            json j;
            j["knob_type"] = knob.type;
            j["max_value"] = knob.maxValue;
            j["min_value"] = knob.minValue;
            j["stride"]    = knob.stride;
            ss << j.dump();
            return ss.str();
        });

    m.def("get_last_error_string", &get_last_error_string);

    py::class_<cudnn_frontend::KernelCache, std::shared_ptr<cudnn_frontend::KernelCache>>(m, "kernel_cache")
        .def("serialize", &kernel_cache_to_json_helper)
        .def("deserialize", &kernel_cache_from_json_helper);
    m.def("create_kernel_cache", &create_kernel_cache_helper);

    m.def("create_handle", &HandleManagement::create_handle);
    m.def("destroy_handle", &HandleManagement::destroy_handle);
    m.def("get_stream", &HandleManagement::get_stream);
    m.def("set_stream", &HandleManagement::set_stream, py::arg("handle"), py::arg("stream"));

    py::enum_<cudnn_frontend::NormFwdPhase_t>(m, "norm_forward_phase")
        .value("INFERENCE", cudnn_frontend::NormFwdPhase_t::INFERENCE)
        .value("TRAINING", cudnn_frontend::NormFwdPhase_t::TRAINING)
        .value("NOT_SET", cudnn_frontend::NormFwdPhase_t::NOT_SET);

    py::enum_<cudnn_frontend::HeurMode_t>(m, "heur_mode")
        .value("A", cudnn_frontend::HeurMode_t::A)
        .value("B", cudnn_frontend::HeurMode_t::B)
        .value("FALLBACK", cudnn_frontend::HeurMode_t::FALLBACK);

    py::enum_<cudnn_frontend::ConvolutionMode_t>(m, "convolution_mode")
        .value("CONVOLUTION", cudnn_frontend::ConvolutionMode_t::CONVOLUTION)
        .value("CROSS_CORRELATION", cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION);

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
        .value("WINOGRAD_TILE_13x13", cudnn_frontend::NumericalNote_t::WINOGRAD_TILE_13x13)
        .value("STRICT_NAN_PROP", cudnn_frontend::NumericalNote_t::STRICT_NAN_PROP);

    py::enum_<cudnn_frontend::BehaviorNote_t>(m, "behavior_note")
        .value("RUNTIME_COMPILATION", cudnn_frontend::BehaviorNote_t::RUNTIME_COMPILATION)
        .value("REQUIRES_FILTER_INT8x32_REORDER", cudnn_frontend::BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER)
        .value("REQUIRES_BIAS_INT8x32_REORDER", cudnn_frontend::BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER)
        .value("SUPPORTS_CUDA_GRAPH_NATIVE_API", cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API);

    py::enum_<cudnn_frontend::DiagonalAlignment_t>(m, "diagonal_alignment")
        .value("TOP_LEFT", cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
        .value("BOTTOM_RIGHT", cudnn_frontend::DiagonalAlignment_t::BOTTOM_RIGHT);
}

}  // namespace python_bindings
}  // namespace cudnn_frontend

// namespace pybind11 {
//     namespace detail {
//     template <> struct type_caster<std::shared_ptr<cudnn_frontend::KernelCache>> {
//     public:
//         PYBIND11_TYPE_CASTER(std::shared_ptr<cudnn_frontend::KernelCache>, _("KernelCachePtr"));

//         bool load(handle , bool) {
//             return false; // Prevent Python -> C++ conversion
//         }

//         static handle cast(std::shared_ptr<cudnn_frontend::KernelCache> src, return_value_policy, handle) {
//             if (!src) return none().release();
//             return capsule(new std::shared_ptr<cudnn_frontend::KernelCache>(std::move(src)),
//                            [](void *ptr) { delete static_cast<std::shared_ptr<cudnn_frontend::KernelCache>*>(ptr);
//                            }).release();
//         }
//     };
// }} // namespace pybind11::detail