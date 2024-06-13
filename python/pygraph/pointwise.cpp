#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"
#include "pygraph.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend::python_bindings {

template <cudnn_frontend::PointwiseMode_t MODE>
std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::pointwise_ternary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& b,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& c,
                           cudnn_frontend::DataType_t const& compute_data_type,
                           std::string const& name) {
    auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                          .set_mode(MODE)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    return graph.pointwise(a, b, c, attributes);
}

template <cudnn_frontend::PointwiseMode_t MODE>
std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::pointwise_binary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                          std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& b,
                          cudnn_frontend::DataType_t const& compute_data_type,
                          std::string const& name) {
    auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                          .set_mode(MODE)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    return graph.pointwise(a, b, attributes);
}

template <cudnn_frontend::PointwiseMode_t MODE>
std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::pointwise_unary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                         cudnn_frontend::DataType_t const& compute_data_type,
                         std::string const& name) {
    auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                          .set_mode(MODE)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    return graph.pointwise(a, attributes);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::relu(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
              std::optional<float> const& negative_slope,
              std::optional<float> const& lower_clip,
              std::optional<float> const& upper_clip,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name) {
    auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                          .set_compute_data_type(compute_data_type)
                          .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                          .set_name(name);

    if (negative_slope.has_value()) {
        attributes.set_relu_lower_clip_slope(negative_slope.value());
    }

    if (lower_clip.has_value()) {
        attributes.set_relu_lower_clip(lower_clip.value());
    }

    if (upper_clip.has_value()) {
        attributes.set_relu_upper_clip(upper_clip.value());
    }

    auto OUT_0 = graph.pointwise(input, attributes);
    return OUT_0;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::gen_index(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                   int64_t const axis,
                   cudnn_frontend::DataType_t const& compute_data_type,
                   std::string const& name) {
    auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                          .set_compute_data_type(compute_data_type)
                          .set_mode(cudnn_frontend::PointwiseMode_t::GEN_INDEX)
                          .set_axis(axis)
                          .set_name(name);

    auto OUT_0 = graph.pointwise(input, attributes);
    return OUT_0;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::relu_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                       std::optional<float> const& negative_slope,
                       std::optional<float> const& lower_clip,
                       std::optional<float> const& upper_clip,
                       cudnn_frontend::DataType_t const& compute_data_type,
                       std::string const& name) {
    auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                          .set_compute_data_type(compute_data_type)
                          .set_mode(cudnn_frontend::PointwiseMode_t::RELU_BWD)
                          .set_name(name);

    if (negative_slope.has_value()) {
        attributes.set_relu_lower_clip_slope(negative_slope.value());
    }

    if (lower_clip.has_value()) {
        attributes.set_relu_lower_clip(lower_clip.value());
    }

    if (upper_clip.has_value()) {
        attributes.set_relu_upper_clip(upper_clip.value());
    }

    auto OUT_0 = graph.pointwise(loss, input, attributes);
    return OUT_0;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::leaky_relu_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                             float const negative_slope,
                             cudnn_frontend::DataType_t const& compute_data_type,
                             std::string const& name) {
    return relu_backward(loss, input, negative_slope, std::nullopt, std::nullopt, compute_data_type, name);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::leaky_relu(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                    float const negative_slope,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    return relu(input, negative_slope, std::nullopt, std::nullopt, compute_data_type, name);
}

void
init_pygraph_pointwise_submodule(py::class_<PyGraph>& m) {
    m.def("add",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::ADD>,
          py::arg("a"),
          py::arg("b"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Adds two cudnn tensors.

            Args:
                a (cudnn_tensor): The first tensor.
                b (cudnn_tensor): The second tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of addition operation.
        )pbdoc");
    m.def("bias",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::ADD>,
          py::arg("input"),
          py::arg("bias"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Add bias to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                bias (cudnn_tensor): The bias tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of adding bias to the input.
        )pbdoc");
    m.def("mul",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MUL>,
          py::arg("a"),
          py::arg("b"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Computes elementwise multiplication of two cudnn tensors.

        Args:
            a (cudnn_tensor): The first tensor.
            b (cudnn_tensor): The second tensor.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of the elementwise multiplication operation.
            )pbdoc");
    m.def("scale",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MUL>,
          py::arg("input"),
          py::arg("scale"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Scale the input.

            Args:
                input (cudnn_tensor): The input tensor.
                scale (cudnn_tensor): The scale tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the scaling operation.
        )pbdoc");

    m.def("sqrt",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SQRT>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Square root of the input tensor is computed

        Args:
            input (cudnn_tensor): The input tensor.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: pointwise square root of the input tensor is computed
        )pbdoc");

    m.def("max",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MAX>,
          py::arg("input0"),
          py::arg("input1"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Max of the input tensors is computed

        Args:
            input (cudnn_tensor): The input tensor 0.
            input (cudnn_tensor): The input tensor 1.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: a pointwise maximum is taken between two tensors.
        )pbdoc");
    m.def("min",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MIN>,
          py::arg("input0"),
          py::arg("input1"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Max of the input tensors is computed

        Args:
            input (cudnn_tensor): The input tensor 0.
            input (cudnn_tensor): The input tensor 1.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: a pointwise minimum is taken between two tensors.
        )pbdoc");

    m.def("gen_index",
          &PyGraph::gen_index,
          py::arg("input"),
          py::arg("axis"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Generates pointwise index value of the input tensor is generated along a given axis.

        Args:
            input (cudnn_tensor): The input tensor.
            axis (int): The axis to generate index for.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result tensor containing the indices
        )pbdoc");

    // forward activations
    m.def("relu",
          &PyGraph::relu,
          py::arg("input"),
          py::arg_v("negative_slope", py::none()),
          py::arg_v("lower_clip", py::none()),
          py::arg_v("upper_clip", py::none()),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Apply the Rectified Linear Unit (ReLU) activation function to the input.

        Args:
            input (cudnn_tensor): The input tensor.
            negative_slope (Optional[float]): Sets the lower clip slope value for ReLU.
            lower_clip (Optional[float]): Sets the lower clip value for ReLU.
            upper_clip (Optional[float]): Sets the upper clip value for ReLU.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of the ReLU activation.
        )pbdoc");
    m.def("leaky_relu",
          &PyGraph::leaky_relu,
          py::arg("input"),
          py::arg("negative_slope"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Apply the Leaky Rectified Linear Unit (Leaky ReLU) activation function to the input.

        Args:
            input (cudnn_tensor): The input tensor.
            negative_slope (float): The slope of the activation for negative inputs.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of the Leaky ReLU activation.
        )pbdoc");
    m.def("tanh",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::TANH_FWD>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        tanh activation of the input tensors is computed

        Args:
            input (cudnn_tensor): The input tensor.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: Result of tanh activation
        )pbdoc");
    m.def("elu",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::ELU_FWD>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Exponential Linear Unit (ELU) activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the ELU activation.
        )pbdoc");
    m.def("gelu",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::GELU_FWD>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Gaussian Error Linear Unit (GELU) activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the GELU activation.
        )pbdoc");
    m.def("sigmoid",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SIGMOID_FWD>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the sigmoid activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the sigmoid activation.
        )pbdoc");
    m.def("swish",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SWISH_FWD>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Swish activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the Swish activation.
        )pbdoc");
    m.def("softplus",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SOFTPLUS_FWD>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Softplus activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the Softplus activation.
        )pbdoc");
    m.def("gelu_approx_tanh",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::GELU_APPROX_TANH_FWD>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Approximate GELU activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the Approximate GELU activation.
        )pbdoc");
    // End of forward activations

    // Backward activations
    m.def("relu_backward",
          &PyGraph::relu_backward,
          py::arg("loss"),
          py::arg("input"),
          py::arg_v("negative_slope", py::none()),
          py::arg_v("lower_clip", py::none()),
          py::arg_v("upper_clip", py::none()),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on Rectified Linear Unit (ReLU) activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                negative_slope (Optional[float]): Sets the lower clip slope value for ReLU.
                lower_clip (Optional[float]): Sets the lower clip value for ReLU.
                upper_clip (Optional[float]): Sets the upper clip value for ReLU.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of ReLU activation.
        )pbdoc");
    m.def("leaky_relu_backward",
          &PyGraph::leaky_relu_backward,
          py::arg("loss"),
          py::arg("input"),
          py::arg("negative_slope"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on Leaky Rectified Linear Unit (Leaky ReLU) activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                negative_slope (float): The slope of the activation for negative inputs.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of Leaky ReLU activation.
        )pbdoc");
    m.def("tanh_backward",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::TANH_BWD>,
          py::arg("loss"),
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on tanh activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of tanh activation.
        )pbdoc");
    m.def("sigmoid_backward",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::SIGMOID_BWD>,
          py::arg("loss"),
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on sigmoid activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of sigmoid activation.
        )pbdoc");
    m.def("elu_backward",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::ELU_BWD>,
          py::arg("loss"),
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on elu activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of elu activation.
        )pbdoc");
    m.def("gelu_backward",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::GELU_BWD>,
          py::arg("loss"),
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on gelu activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of gelu activation.
        )pbdoc");
    m.def("softplus_backward",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::SOFTPLUS_BWD>,
          py::arg("loss"),
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on softplus activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of softplus activation.
        )pbdoc");
    m.def("swish_backward",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::SWISH_BWD>,
          py::arg("loss"),
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on swish activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of swish activation.
        )pbdoc");
    m.def("gelu_approx_tanh_backward",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::GELU_APPROX_TANH_BWD>,
          py::arg("loss"),
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply backpropagation on approximate gelu activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of approximate gelu activation.
        )pbdoc");
    // End of backward activation functions
    m.def("erf",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::ERF>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute erf of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of erf of input.
        )pbdoc");
    m.def("identity",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::IDENTITY>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Copy input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The copy of input.
        )pbdoc");

    m.def("exp",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::EXP>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute exponential of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of exponential of input.
        )pbdoc");
    m.def("log",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::LOG>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute natural logarithm of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of natural logarithm of input.
        )pbdoc");
    m.def("neg",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::NEG>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute numerical negative of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of numerical sign negation of input.
        )pbdoc");
    m.def("mod",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MOD>,
          py::arg("input0"),
          py::arg("input1"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            In this mode, a pointwise floating-point remainder of the first tensor's division by the second tensor is computed.

            Args:
                input0 (cudnn_tensor): The input tensor.
                input1 (cudnn_tensor): The divisor tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of pointwise floating-point remainder of the input0 tensor's division by the input1 tensor
        )pbdoc");
    m.def("pow",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::POW>,
          py::arg("input0"),
          py::arg("input1"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            In this mode, a pointwise value from the first tensor to the power of the second tensor is computed.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of first tensor to the power of the second tensor.
        )pbdoc");
    m.def("abs",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::ABS>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Absolute value of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of absolute value of input.
        )pbdoc");
    m.def("ceil",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::CEIL>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            A pointwise ceiling of the input tensor is computed.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of ceil of input.
        )pbdoc");
    m.def("floor",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::FLOOR>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute floor of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of floor of input.
        )pbdoc");
    m.def("rsqrt",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::RSQRT>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute reciprocal square root of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of reciprocal square root of input.
        )pbdoc");
    m.def("reciprocal",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::RECIPROCAL>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute reciprocal input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of reciprocal of input.
        )pbdoc");
    m.def("sin",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SIN>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute Sine of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of sine of input.
        )pbdoc");
    m.def("cos",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::COS>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute Cosine of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of cosine of input.
        )pbdoc");
    m.def("tan",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::TAN>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Compute Tangent of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of tangent of input.
        )pbdoc");
    m.def("logical_not",
          &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::LOGICAL_NOT>,
          py::arg("input"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Compute logical_not of input tensor.

        Args:
            input (cudnn_tensor): The input tensor.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of logical_not of input.
    )pbdoc");
    m.def("logical_and",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::LOGICAL_AND>,
          py::arg("a"),
          py::arg("b"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Computes logical and of two tensors.

        Args:
            a (cudnn_tensor): The tensor to subtract from.
            b (cudnn_tensor): The tensor to subtract with.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of logical and between two tensors.
    )pbdoc");
    m.def("logical_or",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::LOGICAL_OR>,
          py::arg("a"),
          py::arg("b"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Computes logical or of two tensors.

        Args:
            a (cudnn_tensor): The tensor to subtract from.
            b (cudnn_tensor): The tensor to subtract with.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of logical or between two tensors.
    )pbdoc");

    m.def("sub",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::SUB>,
          py::arg("a"),
          py::arg("b"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
        Computes subtraction of two tensors.

        Args:
            a (cudnn_tensor): The tensor to subtract from.
            b (cudnn_tensor): The tensor to subtract with.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of subtration.
    )pbdoc");
    m.def("div",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::DIV>,
          py::arg("a"),
          py::arg("b"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Computes Division of two tensors.

            Args:
                a (cudnn_tensor): The tensor to subtract from.
                b (cudnn_tensor): The tensor to subtract with.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of Division.
        )pbdoc");
    m.def("add_square",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::ADD_SQUARE>,
          py::arg("a"),
          py::arg("b"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            a pointwise addition between the first tensor and the square of the second tensor is computed.

            Args:
                a (cudnn_tensor): The tensor to subtract from.
                b (cudnn_tensor): The tensor to subtract with.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of a pointwise addition between the first tensor and the square of the second tensor is computed.
        )pbdoc");

    m.def("cmp_eq",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_EQ>,
          py::arg("input"),
          py::arg("comparison"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Compare Equal to Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    m.def("cmp_neq",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_NEQ>,
          py::arg("input"),
          py::arg("comparison"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Compare Not equal to Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    m.def("cmp_gt",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_GT>,
          py::arg("input"),
          py::arg("comparison"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Compare Greater Than Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    m.def("cmp_ge",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_GE>,
          py::arg("input"),
          py::arg("comparison"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Compare Greater Than or Equal Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    m.def("cmp_lt",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_LT>,
          py::arg("input"),
          py::arg("comparison"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Compare Lesser Than Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    m.def("cmp_le",
          &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_LE>,
          py::arg("input"),
          py::arg("comparison"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Apply the Compare Lesser Than or Equal Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    m.def("binary_select",
          &PyGraph::pointwise_ternary<cudnn_frontend::PointwiseMode_t::BINARY_SELECT>,
          py::arg("input0"),
          py::arg("input1"),
          py::arg("mask"),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
            Selects between input0 or input1 based on the mask

            Args:
                input0 (cudnn_tensor): The input tensor0.
                input1 (cudnn_tensor): The input tensor1.
                mask (cudnn_tensor): The mask tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
}

}  // namespace cudnn_frontend::python_bindings