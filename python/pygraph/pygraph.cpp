#include <utility>
#include <unordered_map>
#include <vector>

#include "dlpack/dlpack.h"

// Part of the Array API specification.
#define CUDNN_FRONTEND_DLPACK_CAPSULE_NAME "dltensor"
#define CUDNN_FRONTEND_DLPACK_USED_CAPSULE_NAME "used_dltensor"

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"
#include "pygraph.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend::python_bindings {

void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const& error_msg);

void
init_pygraph_norm_submodule(py::class_<PyGraph>&);

void
init_pygraph_sdpa_submodule(py::class_<PyGraph>&);

void
init_pygraph_pointwise_submodule(py::class_<PyGraph>&);

cudnn_frontend::DataType_t
convert_to_cudnn_data_type(const DLDataType& dtype) {
    switch (dtype.code) {
        case DLDataTypeCode::kDLUInt:
            switch (dtype.bits) {
                case 8:
                    return cudnn_frontend::DataType_t::UINT8;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (dtype.bits) {
                case 8:
                    return cudnn_frontend::DataType_t::INT8;
                case 32:
                    return cudnn_frontend::DataType_t::INT32;
                case 64:
                    return cudnn_frontend::DataType_t::INT64;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (dtype.bits) {
                case 16:
                    return cudnn_frontend::DataType_t::HALF;
                case 32:
                    return cudnn_frontend::DataType_t::FLOAT;
                case 64:
                    return cudnn_frontend::DataType_t::DOUBLE;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (dtype.bits) {
                case 16:
                    return cudnn_frontend::DataType_t::BFLOAT16;
            }
            break;
        case DLDataTypeCode::kDLBool:
            switch (dtype.bits) {
                case 8:
                    return cudnn_frontend::DataType_t::BOOLEAN;
            }
            break;
    }
    return cudnn_frontend::DataType_t::NOT_SET;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor(std::vector<int64_t> const& dim,
                std::vector<int64_t> const& stride,
                cudnn_frontend::DataType_t const& data_type,
                bool const& is_virtual,
                bool const& is_pass_by_value,
                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& ragged_offset,
                cudnn_frontend::TensorReordering_t const reordering_type,
                std::string const& name,
                int64_t const& uid) {
    auto props = cudnn_frontend::graph::Tensor_attributes()
                     .set_data_type(data_type)
                     .set_is_virtual(is_virtual)
                     .set_is_pass_by_value(is_pass_by_value)
                     .set_dim(dim)
                     .set_stride(stride)
                     .set_ragged_offset(ragged_offset)
                     .set_reordering_type(reordering_type)
                     .set_name(name);

    if (uid != -1) {
        props.set_uid(uid);
    }

    return graph->tensor(props);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_like(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& tensor, std::string const& name) {
    return graph->tensor_like(tensor, name);
}

static std::intptr_t
extract_data_pointer(py::object const& obj) {
    throw_if(!py::hasattr(obj, "__dlpack__"),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Object does not have the __dlpack__() method");

    py::capsule capsule = obj.attr("__dlpack__")();
    throw_if(capsule.is_none(),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Failed to retrieve the DLPack capsule.");

    DLManagedTensor* managed =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), CUDNN_FRONTEND_DLPACK_CAPSULE_NAME));
    throw_if(managed == nullptr, cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Invalid DLPack capsule.");

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    throw_if(
        device_type != kDLCPU && device_type != kDLCUDAHost && device_type != kDLCUDA && device_type != kDLCUDAManaged,
        cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
        "Invalid device type.");

    void* p     = (char*)managed->dl_tensor.data + managed->dl_tensor.byte_offset;
    auto result = reinterpret_cast<std::intptr_t>(p);
    return result;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_like(py::object const& pyobj) {
    throw_if(!py::hasattr(pyobj, "__dlpack__"),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Object does not have the __dlpack__() method");

    py::capsule capsule = pyobj.attr("__dlpack__")();
    throw_if(capsule.is_none(),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Failed to retrieve the DLPack capsule.");

    DLManagedTensor* managed =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), CUDNN_FRONTEND_DLPACK_CAPSULE_NAME));
    throw_if(managed == nullptr, cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Invalid DLPack capsule.");

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    throw_if(
        device_type != kDLCPU && device_type != kDLCUDAHost && device_type != kDLCUDA && device_type != kDLCUDAManaged,
        cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
        "Invalid device type.");

    auto ndim = managed->dl_tensor.ndim;
    std::vector<int64_t> dim(managed->dl_tensor.shape, managed->dl_tensor.shape + ndim);

    auto props = cudnn_frontend::graph::Tensor_attributes()
                     .set_data_type(convert_to_cudnn_data_type(managed->dl_tensor.dtype))
                     .set_is_virtual(false)
                     .set_is_pass_by_value(managed->dl_tensor.device.device_type == kDLCPU)
                     .set_dim(dim);

    if (managed->dl_tensor.strides == nullptr) {
        // dlpack says "can be NULL, indicating tensor is compact and row-majored"
        auto stride_order = detail::generate_row_major_stride_order(ndim);
        props.set_stride(detail::generate_stride(dim, stride_order));
    } else {
        std::vector<int64_t> stride(managed->dl_tensor.strides, managed->dl_tensor.strides + ndim);
        props.set_stride(stride);
    }

    return graph->tensor(props);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::slice(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
               std::vector<py::slice> const& slices,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
    auto input_dim = input->get_dim();

    std::vector<std::pair<int64_t, int64_t>> start_end_indices;
    for (size_t i = 0; i < slices.size(); ++i) {
        int64_t start, stop, step, length;
        if (!slices[i].compute(input_dim[i], &start, &stop, &step, &length)) {
            throw std::runtime_error("Invalid slice");
        }
        start_end_indices.push_back({start, stop});
    }

    auto attributes = cudnn_frontend::graph::Slice_attributes()
                          .set_slices(start_end_indices)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto output = graph->slice(input, attributes);
    return output;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_fprop(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::ConvolutionMode_t const& conv_mode,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_fprop_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_convolution_mode(conv_mode)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto Y = graph->conv_fprop(image, weight, attributes);
    return Y;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_dgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& filter,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::ConvolutionMode_t const& conv_mode,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_dgrad_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_convolution_mode(conv_mode)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    auto DX = graph->conv_dgrad(loss, filter, attributes);
    return DX;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_wgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::ConvolutionMode_t const& conv_mode,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_wgrad_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_convolution_mode(conv_mode)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    auto DW = graph->conv_wgrad(loss, image, attributes);
    return DW;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::block_scale_dequantize(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale,
                                std::vector<int32_t> const& block_size,
                                cudnn_frontend::DataType_t const& compute_data_type,
                                std::string const& name) {
    auto attributes = cudnn_frontend::graph::Block_scale_dequantize_attributes()
                          .set_block_size(block_size)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    if (compute_data_type != cudnn_frontend::DataType_t::NOT_SET) {
        attributes.set_compute_data_type(compute_data_type);
    }
    auto output = graph->block_scale_dequantize(input, descale, attributes);
    return output;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::matmul(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& A,
                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& B,
                cudnn_frontend::DataType_t const& compute_data_type,
                double const padding,
                std::string const& name) {
    auto attributes = cudnn_frontend::graph::Matmul_attributes()
                          .set_compute_data_type(compute_data_type)
                          .set_name(name)
                          .set_padding(padding);

    auto C = graph->matmul(A, B, attributes);
    return C;
}

std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2UL>
PyGraph::genstats(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name) {
    auto attributes =
        cudnn_frontend::graph::Genstats_attributes().set_compute_data_type(compute_data_type).set_name(name);

    auto [SUM, SQ_SUM] = graph->genstats(input, attributes);
    return {SUM, SQ_SUM};
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::reduction(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                   cudnn_frontend::ReductionMode_t const mode,
                   cudnn_frontend::DataType_t const& compute_data_type,
                   std::string const& name) {
    auto attributes = cudnn_frontend::graph::Reduction_attributes()
                          .set_mode(mode)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto OUT_0 = graph->reduction(input, attributes);
    return OUT_0;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::reshape(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input, std::string const& name) {
    auto attributes = cudnn_frontend::graph::Reshape_attributes().set_name(name);

    auto OUT_0 = graph->reshape(input, attributes);
    return OUT_0;
}

void
PyGraph::validate() {
    auto status = graph->validate();
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

size_t
PyGraph::key() {
    return graph->key();
}

void
PyGraph::build_operation_graph() {
    auto status = graph->build_operation_graph(handle);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

std::vector<BehaviorNote_t>
PyGraph::get_behavior_notes() {
    std::vector<BehaviorNote_t> notes;
    auto status = graph->get_behavior_notes(notes);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return notes;
}

std::vector<BehaviorNote_t>
PyGraph::get_behavior_notes_for_plan_at_index(int64_t const index) {
    std::vector<BehaviorNote_t> notes;
    auto status = graph->get_behavior_notes_for_plan_at_index(index, notes);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return notes;
}

void
PyGraph::create_execution_plans(std::vector<cudnn_frontend::HeurMode_t> const& modes) {
    auto status = graph->create_execution_plans(modes);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const& knobs) {
    auto status = graph->create_execution_plan(engine_id, knobs);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

int64_t
PyGraph::get_engine_count() {
    int64_t engine_count = 0;
    auto status          = graph->get_engine_count(engine_count);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return engine_count;
}

std::vector<Knob>
PyGraph::get_knobs_for_engine(int64_t const engine_id) {
    std::vector<Knob> knobs;
    auto status = graph->get_knobs_for_engine(engine_id, knobs);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return knobs;
}

void
PyGraph::build_plans(BuildPlanPolicy_t const policy) {
    // TODO: Add multithreaded support in python
    auto status = graph->build_plans(policy, false);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::build_plan_at_index(int64_t const index) {
    auto status = graph->build_plan_at_index(index);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::build(std::vector<cudnn_frontend::HeurMode_t> const& modes) {
    validate();
    build_operation_graph();
    create_execution_plans(modes);
    check_support();
    build_plans(cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE);
}

void
PyGraph::build() {
    validate();
    build_operation_graph();
}

void
PyGraph::check_support() {
    auto status = graph->check_support();
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

int64_t
PyGraph::get_workspace_size() {
    int64_t workspace = 0;

    auto status = graph->get_workspace_size(workspace);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return workspace;
}

int64_t
PyGraph::get_workspace_size_plan_at_index(int64_t index) {
    int64_t workspace;

    auto status = graph->get_workspace_size_plan_at_index(index, workspace);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return workspace;
}

std::vector<uint8_t>
PyGraph::serialize() const {
    std::vector<uint8_t> data;
    auto status = graph->serialize(data);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return data;
}

void
PyGraph::deserialize(std::optional<std::intptr_t> handle_, py::object const& pyobj) {
    if (py::isinstance<py::str>(pyobj)) {
        json j = json::parse(pyobj.cast<std::string>());

        auto status = graph->deserialize(j);

        throw_if(status.is_bad(), status.get_code(), status.get_message());

    } else {
        // If handle is provided, use it (AoT compilation).
        cudnnHandle_t handle =
            handle_.has_value() ? static_cast<cudnnHandle_t>((void*)(handle_.value())) : this->handle;

        std::vector<uint8_t> data = pyobj.cast<std::vector<uint8_t>>();
        auto status               = graph->deserialize(handle, data);

        throw_if(status.is_bad(), status.get_code(), status.get_message());
    }
}

void
PyGraph::deserialize(py::object const& pyobj) {
    // Call the overloaded version with default handle (nullopt)
    deserialize(std::nullopt, pyobj);
}

void
PyGraph::update_cuda_graph(std::intptr_t handle,
                           std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, std::intptr_t> var_pack,
                           std::intptr_t workspace,
                           std::intptr_t cuda_graph) {
    std::unordered_map<int64_t, void*> var_pack_;
    var_pack_.reserve(var_pack.size());
    for (auto const& [uid, device_pointer] : var_pack) {
        var_pack_.emplace(uid, (void*)device_pointer);
    }

    auto status = graph->update_cuda_graph(reinterpret_cast<cudnnHandle_t>(handle),
                                           var_pack_,
                                           reinterpret_cast<void*>(workspace),
                                           reinterpret_cast<cudaGraph_t>(cuda_graph));
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return;
}

void
PyGraph::populate_cuda_graph(
    std::intptr_t handle,
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, std::intptr_t> var_pack,
    std::intptr_t workspace,
    std::intptr_t cuda_graph) {
    std::unordered_map<int64_t, void*> var_pack_;
    var_pack_.reserve(var_pack.size());
    for (auto const& [uid, device_pointer] : var_pack) {
        var_pack_.emplace(uid, (void*)device_pointer);
    }

    auto status = graph->populate_cuda_graph(reinterpret_cast<cudnnHandle_t>(handle),
                                             var_pack_,
                                             reinterpret_cast<void*>(workspace),
                                             reinterpret_cast<cudaGraph_t>(cuda_graph));
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return;
}

void
PyGraph::execute(std::unordered_map<int64_t, std::intptr_t> var_pack,
                 std::intptr_t workspace,
                 std::optional<std::intptr_t> exec_handle) {
    std::unordered_map<int64_t, void*> var_pack_;
    var_pack_.reserve(var_pack.size());
    for (auto const& [uid, device_pointer] : var_pack) {
        var_pack_.emplace(uid, (void*)device_pointer);
    }

    auto workspace_ptr = (void*)workspace;

    cudnnHandle_t handle_ = exec_handle.has_value() ? static_cast<cudnnHandle_t>((void*)(exec_handle.value())) : handle;

    auto status = graph->execute(handle_, var_pack_, workspace_ptr);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return;
}

void
PyGraph::execute_plan_at_index(std::unordered_map<int64_t, std::intptr_t> var_pack,
                               std::intptr_t workspace,
                               int64_t index,
                               std::optional<std::intptr_t> exec_handle) {
    std::unordered_map<int64_t, void*> var_pack_;
    for (auto const& [uid, device_pointer] : var_pack) {
        var_pack_.emplace(uid, (void*)device_pointer);
    }

    auto workspace_ptr = (void*)workspace;

    cudnnHandle_t handle_ = exec_handle.has_value() ? static_cast<cudnnHandle_t>((void*)(exec_handle.value())) : handle;

    auto status = graph->execute_plan_at_index(handle_, var_pack_, workspace_ptr, index);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return;
}

std::shared_ptr<graph::Tensor_attributes>
PyGraph::query_tensor_attributes_of_uid(int64_t const uid) const {
    graph::Tensor_attributes tensor;
    auto status = graph->query_tensor_attributes_of_uid(uid, tensor);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return std::make_shared<graph::Tensor_attributes>(tensor);
}

std::string
PyGraph::get_plan_name_at_index(int64_t index) {
    std::string plan_name;
    auto status = graph->get_plan_name_at_index(index, plan_name);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return plan_name;
}

std::vector<int64_t>
default_vector(void) {
    return {};
}

void
init_pygraph_submodule(py::module_& m) {
    py::class_<PyGraph> pygraph_(m, "pygraph");
    pygraph_
        .def(py::init<std::string const&,
                      cudnn_frontend::DataType_t,
                      cudnn_frontend::DataType_t,
                      cudnn_frontend::DataType_t,
                      std::optional<std::intptr_t>,
                      py::object,
                      py::object,
                      std::shared_ptr<KernelCache>,
                      std::shared_ptr<cudnn_frontend::DeviceProperties>>(),
             py::arg_v("name", "test_graph"),
             py::arg_v("io_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("intermediate_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("handle", std::nullopt),
             py::arg_v("sm_count", py::none()),
             py::arg_v("sm_version", py::none()),
             py::arg_v("kernel_cache", nullptr),
             py::arg_v("device_property", nullptr))
        .def("tensor_like",
             py::overload_cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const&, std::string const&>(
                 &PyGraph::tensor_like),
             py::arg("input"),
             py::arg_v("name", ""))
        .def("tensor_like", py::overload_cast<py::object const&>(&PyGraph::tensor_like))
        .def("_make_tensor",
             &PyGraph::tensor,
             py::arg{"dim"},
             py::arg{"stride"},
             py::arg_v("data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"is_virtual", false},
             py::arg_v{"is_pass_by_value", false},
             py::arg_v{"ragged_offset", nullptr},
             py::arg_v{"reordering_type", cudnn_frontend::TensorReordering_t::NONE},
             py::arg_v("name", ""),
             py::arg_v("uid", -1))
        .def("genstats",
             &PyGraph::genstats,
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("slice",
             &PyGraph::slice,
             py::arg("input"),
             py::arg_v{"slices", default_vector()},
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Perform slice operation on the given input tensor.

                Args:
                    input (cudnn_tensor): The input tensor to be sliced.
                    slices (List[slice]): A list of Python slice objects, one for each dimension.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation.
                        Default is NOT_SET.
                    name (Optional[str]): A name for the slice operation.

                Returns:
                    cudnn_tensor: The resulting sliced tensor.

                Example:
                    >>> input_tensor = graph.tensor([4, 8, 16])
                    >>> sliced_tensor = graph.slice(input_tensor, [slice(0, 2), slice(1, 5), slice(0, 16)])
            )pbdoc")
        .def(
            "conv_fprop",
            [](PyGraph& self,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const convolution_mode,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_fprop(
                    image, weight, padding, padding, stride, dilation, convolution_mode, compute_data_type, name);
            },
            py::arg("image"),
            py::arg("weight"),
            py::arg_v{"padding", default_vector()},
            py::arg_v{"stride", default_vector()},
            py::arg_v{"dilation", default_vector()},
            py::arg_v{"convolution_mode", cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION},
            py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
            py::arg_v("name", ""))
        .def("conv_fprop",
             &PyGraph::conv_fprop,
             py::arg("image"),
             py::arg("weight"),
             py::arg_v{"pre_padding", default_vector()},
             py::arg_v{"post_padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()},
             py::arg_v{"convolution_mode", cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION},
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Perform convolution operation with the given inputs.

                Args:
                    image (cudnn_tensor): The image tensor.
                    weight (cudnn_tensor): The weight tensor.
                    pre_padding (Optional[List[int]]): The pre padding values for the operation. Default is an empty list.
                    post_padding (Optional[List[int]]): The post padding values for the operation. Default is an empty list.
                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def(
            "conv_wgrad",
            [](PyGraph& self,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const convolution_mode,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_wgrad(
                    image, loss, padding, padding, stride, dilation, convolution_mode, compute_data_type, name);
            },
            py::arg("image"),
            py::arg("loss"),
            py::arg_v{"padding", default_vector()},
            py::arg_v{"stride", default_vector()},
            py::arg_v{"dilation", default_vector()},
            py::arg_v{"convolution_mode", cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION},
            py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
            py::arg_v("name", ""))
        .def("conv_wgrad",
             &PyGraph::conv_wgrad,
             py::arg("image"),
             py::arg("loss"),
             py::arg_v{"pre_padding", default_vector()},
             py::arg_v{"post_padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()},
             py::arg_v{"convolution_mode", cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION},
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Compute weight gradients using the given inputs and loss.

                Args:
                    image (cudnn_tensor): The image tensor.
                    loss (cudnn_tensor): The loss tensor.
                    pre_padding (Optional[List[int]]): The pre padding values for the operation. Default is an empty list.
                    post_padding (Optional[List[int]]): The post padding values for the operation. Default is an empty list.                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def(
            "conv_dgrad",
            [](PyGraph& self,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& filter,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const convolution_mode,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_dgrad(
                    loss, filter, padding, padding, stride, dilation, convolution_mode, compute_data_type, name);
            },
            py::arg("loss"),
            py::arg("filter"),
            py::arg_v{"padding", default_vector()},
            py::arg_v{"stride", default_vector()},
            py::arg_v{"dilation", default_vector()},
            py::arg_v{"convolution_mode", cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION},
            py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
            py::arg_v("name", ""))
        .def("conv_dgrad",
             &PyGraph::conv_dgrad,
             py::arg("loss"),
             py::arg("filter"),
             py::arg_v{"pre_padding", default_vector()},
             py::arg_v{"post_padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()},
             py::arg_v{"convolution_mode", cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION},
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Compute filter gradients using the given inputs and loss.

                Args:
                    loss (cudnn_tensor): The loss tensor.
                    filter (cudnn_tensor): The filter tensor.
                    pre_padding (Optional[List[int]]): The pre padding values for the operation. Default is an empty list.
                    post_padding (Optional[List[int]]): The post padding values for the operation. Default is an empty list.
                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[pycudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def("matmul",
             &PyGraph::matmul,
             py::arg("A"),
             py::arg("B"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("padding", 0.0),
             py::arg_v("name", ""),
             R"pbdoc(
                Perform matrix multiplication of two tensors A and B.

                Args:
                    A (cudnn_tensor): The first tensor.
                    B (cudnn_tensor): The second matrix tensor.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The result of the matrix multiplication.
            )pbdoc")
        .def("reduction",
             &PyGraph::reduction,
             py::arg("input"),
             py::arg("mode"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Reduce an input tensor along certain dimensions. These dimensions to reduce on are inferred from output tensor shape.

                Args:
                    input (cudnn_tensor): The input tensor.
                    mode (cudnn.reduction_mode): The mode to use to reduce along a dimension.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The result of reduction operation.
            )pbdoc")
        .def("block_scale_dequantize",
             &PyGraph::block_scale_dequantize,
             py::arg("input"),
             py::arg("descale"),
             py::arg("block_size"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Dequantize an input tensor to other dimensions without changing the actual memory layout.
            )pbdoc")
        .def("reshape",
             &PyGraph::reshape,
             py::arg("input"),
             py::arg_v("name", ""),
             R"pbdoc(
                Reshape an input tensor to other dimensions without changing the actual memory layout.
                These dimensions to reshape to are inferred from output tensor shape.

                Args:
                    input (cudnn_tensor): The input tensor.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The result of reshape operation. Please set the dims for the output tensor.
            )pbdoc")
        .def("get_behavior_notes", &PyGraph::get_behavior_notes)
        .def("get_behavior_notes_for_plan_at_index", &PyGraph::get_behavior_notes_for_plan_at_index)
        .def("deselect_engines", &PyGraph::deselect_engines)
        .def("deselect_numeric_notes", &PyGraph::deselect_numeric_notes)
        .def("deselect_behavior_notes", &PyGraph::deselect_behavior_notes)
        .def("select_numeric_notes", &PyGraph::select_numeric_notes)
        .def("select_behavior_notes", &PyGraph::select_behavior_notes)
        .def("deselect_workspace_greater_than", &PyGraph::deselect_workspace_greater_than)
        .def("validate", &PyGraph::validate)
        .def("key", &PyGraph::key)
        .def("build_operation_graph", &PyGraph::build_operation_graph)
        .def("create_execution_plans", &PyGraph::create_execution_plans)
        .def("create_execution_plan",
             &PyGraph::create_execution_plan,
             R"pbdoc(
                Gets the knob configurations available for the given engine.
                Args:
                    engine_id (int): The ID of the engine to create the execution plan on.
                    knobs (dict[Knob, int]): The map of knobs to knob values.
            )pbdoc")
        .def("get_engine_count", &PyGraph::get_engine_count)
        .def("get_knobs_for_engine",
             &PyGraph::get_knobs_for_engine,
             R"pbdoc(
                Gets the knob configurations available for the given engine.
                Args:
                    engine_id (int): The ID of the engine to query knob configurations for.
            )pbdoc")
        .def("check_support", &PyGraph::check_support)
        .def("build_plans",
             &PyGraph::build_plans,
             py::arg("policy") = cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE)
        .def("build_plan_at_index",
             &PyGraph::build_plan_at_index,
             py::arg("index"),
             R"pbdoc(
                Build a plan at the given index.
                Args:
                    index (int): The index of the plan to build.
            )pbdoc")
        .def("build", (void(PyGraph::*)(std::vector<cudnn_frontend::HeurMode_t> const&)) & PyGraph::build)
        .def("build", (void(PyGraph::*)()) & PyGraph::build)
        .def("get_execution_plan_count",
             &PyGraph::get_execution_plan_count,
             R"pbdoc(
                Get the number of execution plan candidates.
            )pbdoc")
        .def("get_workspace_size", &PyGraph::get_workspace_size)
        .def("get_workspace_size_plan_at_index",
             &PyGraph::get_workspace_size_plan_at_index,
             py::arg("index"),
             R"pbdoc(
                Get workspace for a plan at the given index.
                Args:
                    index (int): The index of the plan to get workspace from.
                    If the graph is not built at the index, this will return 0.
            )pbdoc")
        .def("query_tensor_attributes_of_uid",
             &PyGraph::query_tensor_attributes_of_uid,
             py::arg("uid"),
             R"pbdoc(
                    Get tensor_attributes for a given UID
                    Args:
                    uid (int): The uid of tensor to be queried
                    If the graph does not have the UID, this will raise an error
                )pbdoc")
        .def("get_plan_name_at_index",
             &PyGraph::get_plan_name_at_index,
             py::arg("index"),
             R"pbdoc(
                    Get the name for a plan at the given index.
                    Args:
                    index (int): The index of the plan to get workspace from.
                )pbdoc")
        .def("_execute", &PyGraph::execute)
        .def("populate_cuda_graph", &PyGraph::populate_cuda_graph)
        .def("update_cuda_graph", &PyGraph::update_cuda_graph)
        .def("serialize", &PyGraph::serialize)
        .def("deserialize",
             (void(PyGraph::*)(std::optional<std::intptr_t>, py::object const&)) & PyGraph::deserialize,
             py::arg("handle_"),
             py::arg("pyobj"))
        .def("deserialize", (void(PyGraph::*)(py::object const&)) & PyGraph::deserialize, py::arg("pyobj"))
        .def("_execute_plan_at_index", &PyGraph::execute_plan_at_index)
        .def("__repr__", [](PyGraph const& pygraph) {
            std::stringstream ss;
            json j = pygraph.graph;
            ss << j.dump(4);
            return ss.str();
        });

    m.def("_get_data_ptr", &extract_data_pointer);

    init_pygraph_norm_submodule(pygraph_);
    init_pygraph_sdpa_submodule(pygraph_);
    init_pygraph_pointwise_submodule(pygraph_);
}

}  // namespace cudnn_frontend::python_bindings
