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
              bool const use_causal_mask_bottom_right,
              py::object const& sliding_window_length,
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
                          .set_causal_mask_bottom_right(use_causal_mask_bottom_right)
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

    if (!sliding_window_length.is_none()) {
        int const sliding_window_value = sliding_window_length.cast<int>();
        attributes.set_sliding_window_length(sliding_window_value);
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
                       bool const use_causal_mask_bottom_right,
                       py::object const& sliding_window_length,
                       py::object const& dropout,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
                       bool const use_deterministic_algorithm,
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
                          .set_causal_mask_bottom_right(use_causal_mask_bottom_right)
                          .set_deterministic_algorithm(use_deterministic_algorithm)
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

    if (!sliding_window_length.is_none()) {
        int const sliding_window_value = sliding_window_length.cast<int>();
        attributes.set_sliding_window_length(sliding_window_value);
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

std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 4>
PyGraph::sdpa_fp8(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_q,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_k,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_v,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_s,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_s,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_o,
                  bool const is_inference,
                  py::object const& attn_scale,
                  bool const use_causal_mask,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name) {
    auto attributes = cudnn_frontend::graph::SDPA_fp8_attributes()
                          .set_is_inference(is_inference)
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

    auto [o, stats, amax_s, amax_o] =
        graph.sdpa_fp8(q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, attributes);
    return {o, stats, amax_s, amax_o};
}

std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 7>
PyGraph::sdpa_fp8_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& o,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& dO,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& stats,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_q,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_k,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_v,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_o,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_dO,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_s,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_dP,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_s,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_dQ,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_dK,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_dV,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_dP,
                           py::object const& attn_scale,
                           bool const use_causal_mask,
                           cudnn_frontend::DataType_t const& compute_data_type,
                           std::string const& name) {
    auto attributes = cudnn_frontend::graph::SDPA_fp8_backward_attributes()
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

    auto [dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP] = graph.sdpa_fp8_backward(q,
                                                                                    k,
                                                                                    v,
                                                                                    o,
                                                                                    dO,
                                                                                    stats,
                                                                                    descale_q,
                                                                                    descale_k,
                                                                                    descale_v,
                                                                                    descale_o,
                                                                                    descale_dO,
                                                                                    descale_s,
                                                                                    descale_dP,
                                                                                    scale_s,
                                                                                    scale_dQ,
                                                                                    scale_dK,
                                                                                    scale_dV,
                                                                                    scale_dP,
                                                                                    attributes);
    return {dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP};
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
          py::arg_v("use_causal_mask_bottom_right", false),
          py::arg_v("sliding_window_length", py::none()),
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
                    use_causal_mask_bottom_right (Optional[bool]): Whether to use bottom right aligned causal mask. Default is False.
                    sliding_window_length (Optional[int]): The length of sliding window. Default is None.
                    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
                    rng_dump (Optional[cudnn_tensor]): Debug tensor to dump the Philox RNG dropout mask. Default is None.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): The name of the operation.

                Returns:
                    o (cudnn_tensor): The output data.
                    stats (Optional[cudnn_tensor]): The softmax statistics in case the operation is in a training step.
            )pbdoc");
    m.def("sdpa_backward",
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
          py::arg_v("use_causal_mask_bottom_right", false),
          py::arg_v("sliding_window_length", py::none()),
          py::arg_v("dropout", py::none()),
          py::arg_v("rng_dump", nullptr),
          py::arg_v("use_deterministic_algorithm", false),
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
                    use_causal_mask_bottom_right (Optional[bool]): Whether to use bottom right aligned causal mask. Default is False.
                    sliding_window_length (Optional[int]): The length of sliding window. Default is None.
                    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
                    rng_dump (Optional[cudnn_tensor]): Debug tensor to dump the Philox RNG dropout mask. Default is None.
                    use_deterministic_algorithm (Optional[bool]): Whether to always use deterministic algorithm. Default is False.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): The name of the operation.

                Returns:
                    dQ (cudnn_tensor): The query gradient data.
                    dK (cudnn_tensor): The key gradient data.
                    dV (cudnn_tensor): The value gradient data.
            )pbdoc");
    m.def("sdpa_fp8",
          &PyGraph::sdpa_fp8,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("descale_q"),
          py::arg("descale_k"),
          py::arg("descale_v"),
          py::arg("descale_s"),
          py::arg("scale_s"),
          py::arg("scale_o"),
          py::arg("is_inference"),
          py::arg_v("attn_scale", py::none()),
          py::arg_v("use_causal_mask", false),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
                Perform scaled dot product attention with fp8 datatype inputs and outputs.

                Args:
                    q (cudnn_tensor): The query data.
                    k (cudnn_tensor): The key data.
                    v (cudnn_tensor): The value data.
                    descale_q (cudnn_tensor): Descale factor for query.
                    descale_k (cudnn_tensor): Descale factor for key.
                    descale_v (cudnn_tensor): Descale factor for value.
                    descale_s (cudnn_tensor): Descale factor for S tensor.
                    scale_s (cudnn_tensor): Scale factor for S tensor.
                    scale_o (cudnn_tensor): Scale factor for output.
                    is_inference (bool): Whether it is an inference step or training step.
                    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
                    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): The name of the operation.

                Returns:
                    o (cudnn_tensor): The output data.
                    stats (Optional[cudnn_tensor]): The softmax statistics in case the operation is in a training step.
                    amax_s (cudnn_tensor): The absolute maximum of S tensor.
                    amax_o (cudnn_tensor): The absolute maximum of output tensor.
            )pbdoc");
    m.def("sdpa_fp8_backward",
          &PyGraph::sdpa_fp8_backward,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("o"),
          py::arg("dO"),
          py::arg("stats"),
          py::arg("descale_q"),
          py::arg("descale_k"),
          py::arg("descale_v"),
          py::arg("descale_o"),
          py::arg("descale_dO"),
          py::arg("descale_s"),
          py::arg("descale_dP"),
          py::arg("scale_s"),
          py::arg("scale_dQ"),
          py::arg("scale_dK"),
          py::arg("scale_dV"),
          py::arg("scale_dP"),
          py::arg_v("attn_scale", py::none()),
          py::arg_v("use_causal_mask", false),
          py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
                Compute the key, query, value gradients of scaled dot product attention with fp8 datatype inputs and outputs.

                Args:
                    q (cudnn_tensor): The query data.
                    k (cudnn_tensor): The key data.
                    v (cudnn_tensor): The value data.
                    o (cudnn_tensor): The output data.
                    dO (cudnn_tensor): The output gradient data.
                    stats (cudnn_tensor): The softmax statistics in case the operation is in a training step.
                    descale_q (cudnn_tensor): Descale factor for query.
                    descale_k (cudnn_tensor): Descale factor for key.
                    descale_v (cudnn_tensor): Descale factor for value.
                    descale_o (cudnn_tensor): Descale factor for output.
                    descale_dO (cudnn_tensor): Descale factor for output gradient.
                    descale_s (cudnn_tensor): Descale factor for S tensor.
                    descale_dP (cudnn_tensor): Descale factor for P gradient tensor.
                    scale_s (cudnn_tensor): Scale factor for S tensor.
                    scale_dQ (cudnn_tensor): Scale factor for query gradient.
                    scale_dK (cudnn_tensor): Scale factor for key gradient.
                    scale_dV (cudnn_tensor): Scale factor for value gradient.
                    scale_dP (cudnn_tensor): Scale factor for dP gradient.
                    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
                    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): The name of the operation.

                Returns:
                    dQ (cudnn_tensor): The query gradient data.
                    dK (cudnn_tensor): The key gradient data.
                    dV (cudnn_tensor): The value gradient data.
                    amax_dQ (cudnn_tensor): The absolute maximum of query gradient tensor.
                    amax_dK (cudnn_tensor): The absolute maximum of key gradient tensor.
                    amax_dV (cudnn_tensor): The absolute maximum of value gradient tensor.
                    amax_dP (cudnn_tensor): The absolute maximum of dP tensor.
            )pbdoc");
    m.attr("scaled_dot_product_flash_attention")          = m.attr("sdpa");
    m.attr("scaled_dot_product_flash_attention_backward") = m.attr("sdpa_backward");
}

}  // namespace cudnn_frontend::python_bindings