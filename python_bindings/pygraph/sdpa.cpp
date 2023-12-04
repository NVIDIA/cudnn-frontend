#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"
#include "pygraph.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend::python_bindings {

std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2>
PyGraph::sdpa(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
              bool const is_inference,
              py::object const& attn_scale,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
              bool const use_alibi_mask,
              bool const use_padding_mask,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
              bool const use_causal_mask,
              py::object const& dropout,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name) {
    auto attributes = cudnn_frontend::graph::SDPA_attributes()
                          .set_is_inference(is_inference)
                          .set_bias(bias)
                          .set_alibi_mask(use_alibi_mask)
                          .set_padding_mask(use_padding_mask)
                          .set_seq_len_q(seq_len_q)
                          .set_seq_len_kv(seq_len_kv)
                          .set_causal_mask(use_causal_mask)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    if (!attn_scale.is_none()) {
        if (py::isinstance<py::float_>(attn_scale)) {
            auto const attn_scale_value = attn_scale.cast<float>();
            attributes.set_attn_scale(attn_scale_value);
        } else {
            auto const attn_scale_tensor = attn_scale.cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            if (!attn_scale_tensor) {
                throw std::runtime_error("attn_scale must be a cudnn_tensor or float.");
            }
            attributes.set_attn_scale(attn_scale_tensor);
        }
    }

    if (!dropout.is_none()) {
        py::tuple dropout_tuple = dropout.cast<py::tuple>();
        if ((!dropout_tuple) || (dropout_tuple.size() != 3 && dropout_tuple.size() != 2)) {
            throw std::runtime_error(
                "dropout must be a tuple of (float probability, a seed tensor, and an offset tensor) or (mask "
                "tensor, scale tensor)");
        }
        if (py::isinstance<py::float_>(dropout_tuple[0])) {
            auto const probability = dropout_tuple[0].cast<float>();
            auto const seed        = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            if (!seed) {
                throw std::runtime_error("dropout seed must be a cudnn_tensor.");
            }

            auto const offset = dropout_tuple[2].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            if (!offset) {
                throw std::runtime_error("dropout offset must be a cudnn_tensor.");
            }

            attributes.set_dropout(probability, seed, offset);
            if (rng_dump) {
                attributes.set_rng_dump(rng_dump);
            }
        } else {
            auto const mask = dropout_tuple[0].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            if (!mask) {
                throw std::runtime_error("dropout mask must be a cudnn_tensor.");
            }

            auto const scale = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            if (!scale) {
                throw std::runtime_error("dropout scale must be a cudnn_tensor.");
            }

            attributes.set_dropout(mask, scale);
        }
    }

    auto [O, Stats] = graph.sdpa(q, k, v, attributes);
    return {O, Stats};
}

std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 3>
PyGraph::sdpa_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& o,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& dO,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& stats,
                       py::object const& attn_scale,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& dBias,
                       bool const use_alibi_mask,
                       bool const use_padding_mask,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
                       bool const use_causal_mask,
                       py::object const& dropout,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
                       cudnn_frontend::DataType_t const& compute_data_type,
                       std::string const& name) {
    auto attributes = cudnn_frontend::graph::SDPA_backward_attributes()
                          .set_bias(bias)
                          .set_dbias(dBias)
                          .set_alibi_mask(use_alibi_mask)
                          .set_padding_mask(use_padding_mask)
                          .set_seq_len_q(seq_len_q)
                          .set_seq_len_kv(seq_len_kv)
                          .set_causal_mask(use_causal_mask)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    py::object cudnn_tensor_type = py::module_::import("cudnn").attr("tensor");

    if (!attn_scale.is_none()) {
        if (py::isinstance<py::float_>(attn_scale)) {
            auto const attn_scale_value = attn_scale.cast<float>();
            attributes.set_attn_scale(attn_scale_value);
        } else {
            auto const attn_scale_tensor = attn_scale.cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            if (!attn_scale_tensor) {
                throw std::runtime_error("attn_scale must be a cudnn_tensor or float.");
            }
            attributes.set_attn_scale(attn_scale_tensor);
        }
    }

    if (!dropout.is_none()) {
        if (!py::isinstance<py::tuple>(dropout)) {
            throw std::runtime_error(
                "dropout must be a tuple of (float probability, a seed tensor"
                ", and an offset tensor) or (mask tensor, scale tensor)");
        }
        py::tuple dropout_tuple = dropout.cast<py::tuple>();
        if (dropout_tuple.size() != 3) {
            throw std::runtime_error(
                "dropout must be a tuple of (float probability, a seed tensor"
                ", and an offset tensor) or (mask tensor, scale tensor)");
        }

        if (py::isinstance<py::float_>(dropout_tuple[0]) && py::isinstance(dropout_tuple[1], cudnn_tensor_type) &&
            py::isinstance(dropout_tuple[2], cudnn_tensor_type)) {
            auto const probability = dropout_tuple[0].cast<float>();
            auto const seed        = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            auto const offset      = dropout_tuple[2].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            attributes.set_dropout(probability, seed, offset);
            if (rng_dump) {
                attributes.set_rng_dump(rng_dump);
            }
        } else if (py::isinstance(dropout_tuple[0], cudnn_tensor_type) &&
                   py::isinstance(dropout_tuple[1], cudnn_tensor_type) &&
                   py::isinstance(dropout_tuple[2], cudnn_tensor_type)) {
            auto const mask      = dropout_tuple[0].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            auto const scale     = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            auto const scale_inv = dropout_tuple[2].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
            attributes.set_dropout(mask, scale, scale_inv);
        } else {
            throw std::runtime_error(
                "dropout must be a tuple of (float probability, a seed tensor"
                ", and an offset tensor) or (mask tensor, scale tensor)");
        }
    }

    auto [dQ, dK, dV] = graph.sdpa_backward(q, k, v, o, dO, stats, attributes);
    return {dQ, dK, dV};
}

void
init_pygraph_sdpa_submodule(py::class_<PyGraph>& m) {
    m.def("sdpa",
          &PyGraph::sdpa,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("is_inference"),
          py::arg_v("attn_scale", py::none()),
          py::arg_v("bias", nullptr),
          py::arg_v("use_alibi_mask", false),
          py::arg_v("use_padding_mask", false),
          py::arg_v("seq_len_q", nullptr),
          py::arg_v("seq_len_kv", nullptr),
          py::arg_v("use_causal_mask", false),
          py::arg_v("dropout", py::none()),
          py::arg_v("rng_dump", nullptr),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
                Perform scaled dot product attention.

                Args:
                    q (cudnn_tensor): The query data.
                    k (cudnn_tensor): The key data.
                    v (cudnn_tensor): The value data.
                    is_inference (bool): Whether it is an inference step or training step.
                    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
                    bias (Optional[cudnn_tensor]): The bias data for attention. Default is None.
                    use_alibi_mask (Optional[bool]): Whether to use alibi mask. Default is False.
                    use_padding_mask (Optional[bool]): Whether to use padding mask. Default is False.
                    seq_len_q (Optional[cudnn_tensor]): The sequence length of the query.
                    seq_len_kv (Optional[cudnn_tensor]): The sequence length of the key.
                    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
                    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): The name of the operation.

                Returns:
                    o (cudnn_tensor): The result of scaled dot-product flash attention.
                    stats (Optional[cudnn_tensor]): The softmax statistics in case the operation is in a training step.
            )pbdoc")
        .def("sdpa_backward",
             &PyGraph::sdpa_backward,
             py::arg("q"),
             py::arg("k"),
             py::arg("v"),
             py::arg("o"),
             py::arg("dO"),
             py::arg("stats"),
             py::arg_v("attn_scale", py::none()),
             py::arg_v("bias", nullptr),
             py::arg_v("dBias", nullptr),
             py::arg_v("use_alibi_mask", false),
             py::arg_v("use_padding_mask", false),
             py::arg_v("seq_len_q", nullptr),
             py::arg_v("seq_len_kv", nullptr),
             py::arg_v("use_causal_mask", false),
             py::arg_v("dropout", py::none()),
             py::arg_v("rng_dump", nullptr),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Compute the key, query, value gradients of scaled dot product attention.

                Args:
                    q (cudnn_tensor): The query data.
                    k (cudnn_tensor): The key data.
                    v (cudnn_tensor): The value data.
                    o (cudnn_tensor): The output data.
                    dO (cudnn_tensor): The output loss gradient.
                    stats (cudnn_tensor): The softmax statistics from the forward pass.
                    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
                    bias (Optional[cudnn_tensor]): The bias data for attention. Default is None.
                    dBias (Optional[cudnn_tensor]): The dBias data for attention. Default is None.
                    use_alibi_mask (Optional[bool]): Whether to use alibi mask. Default is False.
                    use_padding_mask (Optional[bool]): Whether to use padding mask. Default is False.
                    seq_len_q (Optional[cudnn_tensor]): The sequence length of the query.
                    seq_len_kv (Optional[cudnn_tensor]): The sequence length of the key.
                    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
                    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): The name of the operation.

                Returns:
                    dQ (cudnn_tensor): The query gradient tensor of scaled dot-product flash attention.
                    dK (cudnn_tensor): The key gradient tensor of scaled dot-product flash attention.
                    dV (cudnn_tensor): The value gradient tensor of scaled dot-product flash attention.
            )pbdoc");
    m.attr("scaled_dot_product_flash_attention")          = m.attr("sdpa");
    m.attr("scaled_dot_product_flash_attention_backward") = m.attr("sdpa_backward");
}

}  // namespace cudnn_frontend::python_bindings