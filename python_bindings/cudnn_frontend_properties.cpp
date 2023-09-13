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

void * 
create_handle() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    return (void *)handle;
}

void
destroy_handle(void *handle) {
    auto status = cudnnDestroy((cudnnHandle_t)handle);
    throw_if(status != CUDNN_STATUS_SUCCESS, cudnn_frontend::error_code_t::HANDLE_ERROR, "cudnnHandle Destroy failed");
}

void
init_properties(py::module_& m) {
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
        .def("__repr__", [](cudnn_frontend::graph::Tensor_attributes const& props) {
            std::ostringstream out;
            out << json{props};
            return out.str();
        });

}

}
}