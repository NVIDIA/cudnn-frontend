#include <utility>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"
#include "pygraph.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend {

namespace python_bindings {

std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
PyGraph::batchnorm(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& in_running_mean,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& in_running_var,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& momentum,
                   std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& peer_stats,
                   cudnn_frontend::DataType_t const& compute_data_type,
                   std::string const& name) {
    auto attributes = cudnn_frontend::graph::Batchnorm_attributes()
                          .set_compute_data_type(compute_data_type)
                          .set_epsilon(epsilon)
                          .set_previous_running_stats(in_running_mean, in_running_var, momentum)
                          .set_peer_stats(peer_stats)
                          .set_name(name);

    auto [Y, mean, inv_var, next_running_mean, next_running_var] = graph.batchnorm(x, scale, bias, attributes);
    return {Y, mean, inv_var, next_running_mean, next_running_var};
}

std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
PyGraph::layernorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
                   cudnn_frontend::DataType_t const& compute_data_type,
                   std::string const& name) {
    auto attributes = cudnn_frontend::graph::Layernorm_attributes()
                          .set_forward_phase(forward_phase)
                          .set_compute_data_type(compute_data_type)
                          .set_epsilon(epsilon)
                          .set_name(name);

    auto [Y, mean, inv_var] = graph.layernorm(x, scale, bias, attributes);
    return {Y, mean, inv_var};
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::batchnorm_inference(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& mean,
                             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& inv_variance,
                             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                             cudnn_frontend::DataType_t const& compute_data_type,
                             std::string const& name) {
    auto attributes =
        cudnn_frontend::graph::Batchnorm_inference_attributes().set_compute_data_type(compute_data_type).set_name(name);

    return graph.batchnorm_inference(x, mean, inv_variance, scale, bias, attributes);
}

std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
PyGraph::layernorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& mean,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                            cudnn_frontend::DataType_t const& compute_data_type,
                            std::string const& name) {
    auto attributes = cudnn_frontend::graph::Layernorm_backward_attributes()
                          .set_saved_mean_and_inv_variance(mean, inv_variance)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto [DX, DScale, DBias] = graph.layernorm_backward(dy, x, scale, attributes);
    return {DX, DScale, DBias};
}

std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
PyGraph::batchnorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& mean,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                            std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& peer_stats,
                            cudnn_frontend::DataType_t const& compute_data_type,
                            std::string const& name) {
    auto attributes = cudnn_frontend::graph::Batchnorm_backward_attributes()
                          .set_saved_mean_and_inv_variance(mean, inv_variance)
                          .set_peer_stats(peer_stats)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto [DX, DScale, DBias] = graph.batchnorm_backward(dy, x, scale, attributes);
    return {DX, DScale, DBias};
}

std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
PyGraph::rmsnorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
                 cudnn_frontend::DataType_t const& compute_data_type,
                 std::string const& name) {
    auto attributes = cudnn_frontend::graph::Rmsnorm_attributes()
                          .set_forward_phase(forward_phase)
                          .set_compute_data_type(compute_data_type)
                          .set_bias(bias)
                          .set_epsilon(epsilon)
                          .set_name(name);

    auto [Y, inv_var] = graph.rmsnorm(x, scale, attributes);
    return {Y, inv_var};
}

std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
PyGraph::rmsnorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                          std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                          std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                          std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                          bool const has_dbias,
                          cudnn_frontend::DataType_t const& compute_data_type,
                          std::string const& name) {
    auto attributes = cudnn_frontend::graph::Rmsnorm_backward_attributes()
                          .has_dbias(has_dbias)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto [DX, DScale, DBias] = graph.rmsnorm_backward(dy, x, scale, inv_variance, attributes);
    return {DX, DScale, DBias};
}

std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
PyGraph::instancenorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
                      cudnn_frontend::DataType_t const& compute_data_type,
                      std::string const& name) {
    auto attributes = cudnn_frontend::graph::Instancenorm_attributes()
                          .set_forward_phase(forward_phase)
                          .set_compute_data_type(compute_data_type)
                          .set_epsilon(epsilon)
                          .set_name(name);

    auto [Y, mean, inv_var] = graph.instancenorm(x, scale, bias, attributes);
    return {Y, mean, inv_var};
}

std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
PyGraph::instancenorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& mean,
                               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                               cudnn_frontend::DataType_t const& compute_data_type,
                               std::string const& name) {
    auto attributes = cudnn_frontend::graph::Instancenorm_backward_attributes()
                          .set_saved_mean_and_inv_variance(mean, inv_variance)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto [DX, DScale, DBias] = graph.instancenorm_backward(dy, x, scale, attributes);
    return {DX, DScale, DBias};
}

void
init_pygraph_norm_submodule(py::class_<PyGraph>& m) {
    m.def("batchnorm",
          &PyGraph::batchnorm,
          py::arg("input"),
          py::arg("scale"),
          py::arg("bias"),
          py::arg("in_running_mean"),
          py::arg("in_running_var"),
          py::arg("epsilon"),
          py::arg("momentum"),
          py::arg_v("peer_stats", std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>()),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""))
        .def("layernorm",
             &PyGraph::layernorm,
             py::arg("norm_forward_phase"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg("epsilon"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("batchnorm_inference",
             &PyGraph::batchnorm_inference,
             py::arg("input"),
             py::arg("mean"),
             py::arg("inv_variance"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("batchnorm_backward",
             &PyGraph::batchnorm_backward,
             py::arg("grad"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("mean"),
             py::arg("inv_variance"),
             py::arg_v("peer_stats", std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>()),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("layernorm_backward",
             &PyGraph::layernorm_backward,
             py::arg("grad"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("mean"),
             py::arg("inv_variance"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("rmsnorm",
             &PyGraph::rmsnorm,
             py::arg("norm_forward_phase"),
             py::arg("input"),
             py::arg("scale"),
             py::arg_v("bias", nullptr),
             py::arg("epsilon"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("rmsnorm_backward",
             &PyGraph::rmsnorm_backward,
             py::arg("grad"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("inv_variance"),
             py::arg("has_dbias"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))

        .def("instancenorm",
             &PyGraph::instancenorm,
             py::arg("norm_forward_phase"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg("epsilon"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))

        .def("instancenorm_backward",
             &PyGraph::instancenorm_backward,
             py::arg("grad"),
             py::arg("input"),
             py::arg("scale"),
             py::arg_v("mean", nullptr),
             py::arg_v("inv_variance", nullptr),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""));
}

}  // namespace python_bindings

}  // namespace cudnn_frontend