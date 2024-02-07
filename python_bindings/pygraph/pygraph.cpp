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

char*
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

    return (char*)managed->dl_tensor.data + managed->dl_tensor.byte_offset;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor(std::vector<int64_t> const& dim,
                std::vector<int64_t> const& stride,
                cudnn_frontend::DataType_t const& data_type,
                bool const& is_virtual,
                bool const& is_pass_by_value,
                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& ragged_offset,
                std::string const& name) {
    auto props = cudnn_frontend::graph::Tensor_attributes()
                     .set_data_type(data_type)
                     .set_is_virtual(is_virtual)
                     .set_is_pass_by_value(is_pass_by_value)
                     .set_dim(dim)
                     .set_stride(stride)
                     .set_ragged_offset(ragged_offset)
                     .set_name(name);

    return graph.tensor(props);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_like(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& tensor, std::string const& name) {
    return graph.tensor_like(tensor, name);
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
        auto stride_order = cudnn_frontend::detail::generate_row_major_stride_order(ndim);
        props.set_stride(cudnn_frontend::detail::generate_stride(dim, stride_order));
    } else {
        std::vector<int64_t> stride(managed->dl_tensor.strides, managed->dl_tensor.strides + ndim);
        props.set_stride(stride);
    }

    return graph.tensor(props);
}
std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_fprop(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_fprop_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto Y = graph.conv_fprop(image, weight, attributes);
    return Y;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_dgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& filter,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_dgrad_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    auto DX = graph.conv_dgrad(loss, filter, attributes);
    return DX;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_wgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_wgrad_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    auto DW = graph.conv_wgrad(loss, image, attributes);
    return DW;
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

    auto C = graph.matmul(A, B, attributes);
    return C;
}

std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2UL>
PyGraph::genstats(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name) {
    auto attributes =
        cudnn_frontend::graph::Genstats_attributes().set_compute_data_type(compute_data_type).set_name(name);

    auto [SUM, SQ_SUM] = graph.genstats(input, attributes);
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

    auto OUT_0 = graph.reduction(input, attributes);
    return OUT_0;
}

void
PyGraph::validate() {
    auto status = graph.validate();
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

size_t
PyGraph::key() {
    return graph.key();
}

void
PyGraph::build_operation_graph() {
    auto status = graph.build_operation_graph(handle);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::create_execution_plans(std::vector<cudnn_frontend::HeurMode_t> const& modes) {
    auto status = graph.create_execution_plans(modes);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::build_plans(BuildPlanPolicy_t const policy) {
    // TODO: Add multithreaded support in python
    auto status = graph.build_plans(handle, policy, false);
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
PyGraph::check_support() {
    auto status = graph.check_support(handle);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

int64_t
PyGraph::get_workspace_size() {
    return graph.get_workspace_size();
}

void
PyGraph::execute(std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, py::object> var_pack,
                 py::object workspace) {
    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> var_pack_;
    for (auto const& [tensor, pyobject] : var_pack) {
        // Its alright for the user to pass in None objects as key
        // FE will just ignore them
        if (tensor) {
            var_pack_.emplace(tensor, extract_data_pointer(pyobject));
        }
    }

    void* workspace_ptr = extract_data_pointer(workspace);

    // TODO: Probably concatenate in a macro?
    auto status = graph.execute(handle, var_pack_, workspace_ptr);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return;
}

void
PyGraph::execute(std::unordered_map<int64_t, py::object> var_pack, py::object workspace) {
    std::unordered_map<int64_t, void*> var_pack_;
    for (auto const& [uid, pyobject] : var_pack) {
        var_pack_.emplace(uid, extract_data_pointer(pyobject));
    }

    void* workspace_ptr = extract_data_pointer(workspace);

    // TODO: Probably concatenate in a macro?
    auto status = graph.execute(handle, var_pack_, workspace_ptr);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return;
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
                      void*>(),
             py::arg_v("name", "test_graph"),
             py::arg_v("io_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("intermediate_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("handle", nullptr))
        .def("tensor_like",
             py::overload_cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const&, std::string const&>(
                 &PyGraph::tensor_like),
             py::arg("input"),
             py::arg_v("name", ""))
        .def("tensor_like", py::overload_cast<py::object const&>(&PyGraph::tensor_like))
        .def("tensor",
             &PyGraph::tensor,
             py::arg{"dim"},
             py::arg{"stride"},
             py::arg_v("data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"is_virtual", false},
             py::arg_v{"is_pass_by_value", false},
             py::arg_v{"ragged_offset", nullptr},
             py::arg_v("name", ""),
             R"pbdoc(
                Create a tensor.

                Args:
                    dim (List[int]): The dimensions of the tensor.
                    stride (List[int]): The strides of the tensor.
                    data_type (cudnn.data_type): The data type of the tensor. Default is cudnn.data_type.NOT_SET.
                    is_virtual (bool): Flag indicating if the tensor is virtual. Default is False.
                    is_pass_by_value (bool): Flag indicating if the tensor is passed by value. Default is False.
                    ragged_offset (cudnn_tensor): The ragged offset tensor. Default is nullptr.
                    name (Optional[str]): The name of the tensor.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def("genstats",
             &PyGraph::genstats,
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def(
            "conv_fprop",
            [](PyGraph& self,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_fprop(image, weight, padding, padding, stride, dilation, compute_data_type, name);
            },
            py::arg("image"),
            py::arg("weight"),
            py::arg_v{"padding", default_vector()},
            py::arg_v{"stride", default_vector()},
            py::arg_v{"dilation", default_vector()},
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
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_wgrad(image, loss, padding, padding, stride, dilation, compute_data_type, name);
            },
            py::arg("image"),
            py::arg("loss"),
            py::arg_v{"padding", default_vector()},
            py::arg_v{"stride", default_vector()},
            py::arg_v{"dilation", default_vector()},
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
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_dgrad(loss, filter, padding, padding, stride, dilation, compute_data_type, name);
            },
            py::arg("loss"),
            py::arg("filter"),
            py::arg_v{"padding", default_vector()},
            py::arg_v{"stride", default_vector()},
            py::arg_v{"dilation", default_vector()},
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
        .def("deselect_numeric_notes", &PyGraph::deselect_numeric_notes)
        .def("deselect_behavior_notes", &PyGraph::deselect_behavior_notes)
        .def("deselect_workspace_greater_than", &PyGraph::deselect_workspace_greater_than)
        .def("validate", &PyGraph::validate)
        .def("key", &PyGraph::key)
        .def("build_operation_graph", &PyGraph::build_operation_graph)
        .def("create_execution_plans", &PyGraph::create_execution_plans)
        .def("check_support", &PyGraph::check_support)
        .def("build_plans",
             &PyGraph::build_plans,
             py::arg("policy") = cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE)
        .def("build", &PyGraph::build)
        .def("get_workspace_size", &PyGraph::get_workspace_size)
        .def(
            "execute",
            static_cast<void (PyGraph::*)(
                std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, py::object>, py::object)>(
                &PyGraph::execute))
        .def("execute",
             static_cast<void (PyGraph::*)(std::unordered_map<int64_t, py::object>, py::object)>(&PyGraph::execute))
        .def("__repr__", [](PyGraph const& pygraph) {
            std::stringstream ss;
            json j = pygraph.graph;
            ss << j.dump(4);
            return ss.str();
        });

    init_pygraph_norm_submodule(pygraph_);
    init_pygraph_sdpa_submodule(pygraph_);
    init_pygraph_pointwise_submodule(pygraph_);
}

}  // namespace cudnn_frontend::python_bindings