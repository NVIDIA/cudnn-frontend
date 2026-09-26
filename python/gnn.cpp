/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <stdexcept>
#include <string>

#include "pybind11/pybind11.h"

#include "cudnn_frontend.h"

namespace py = pybind11;

namespace cudnn_frontend::python_bindings {

namespace {

// cuDNN GNN APIs are not supported on Windows. Use frontend-owned ABI
// structures so bindings built with older cuDNN headers remain available when
// loaded with a newer backend.
#if !defined(_WIN32)
detail::gnn_csc_graph_t
make_csc_graph(std::intptr_t csc_offsets,
               std::intptr_t csc_indices,
               std::intptr_t map_csc_to_coo,
               std::int64_t n_src_nodes,
               std::int64_t n_dst_nodes,
               std::int64_t n_indices,
               int idx_type,
               std::intptr_t csc_rev_offsets = 0,
               std::intptr_t map_rev_to_coo  = 0) {
    detail::gnn_csc_graph_t graph{};
    graph.csc_offsets     = reinterpret_cast<const void *>(csc_offsets);
    graph.csc_indices     = reinterpret_cast<const void *>(csc_indices);
    graph.map_csc_to_coo  = reinterpret_cast<const void *>(map_csc_to_coo);
    graph.map_rev_to_coo  = reinterpret_cast<const void *>(map_rev_to_coo);
    graph.n_src_nodes     = n_src_nodes;
    graph.n_dst_nodes     = n_dst_nodes;
    graph.n_indices       = n_indices;
    graph.idx_type        = static_cast<cudnnDataType_t>(idx_type);
    graph.csc_rev_offsets = reinterpret_cast<const void *>(csc_rev_offsets);
    return graph;
}

void
require_gnn_backend_version(size_t minimum_version, char const *operation) {
    auto const backend_version = detail::get_backend_version();
    if (backend_version < minimum_version) {
        auto const message = std::string(operation) + " requires cuDNN " +
                             detail::convert_version_to_str(minimum_version) + " or newer; loaded " +
                             detail::convert_version_to_str(backend_version);
        throw std::runtime_error(message);
    }
}

void
throw_if_gnn_failed(cudnnStatus_t status, char const *operation) {
    if (status == CUDNN_STATUS_SUCCESS) return;

    std::string message = std::string(operation) + " failed: " + detail::get_error_string(status);
    if (status == CUDNN_STATUS_BAD_PARAM) {
        throw std::invalid_argument(message);
    }
    if (status == CUDNN_STATUS_NOT_SUPPORTED || status == CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH) {
        throw std::runtime_error(message);
    }
    throw std::runtime_error(message);
}

void
ensure_cuda_runtime_context() {
    // AggSimple's NVRTC path currently requires a CUDA context to already be
    // current on the calling thread. cudaFree(nullptr) initializes the CUDA
    // Runtime's primary context without allocating or freeing device memory.
    auto const status = detail::cuda_free(nullptr);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to initialize the CUDA Runtime context: ") +
                                 detail::cuda_get_error_string(status));
    }
}
#endif

}  // namespace

void
init_gnn_submodule([[maybe_unused]] py::module_ &m) {
    // maybe_unused: on Windows the binding bodies compile away entirely.
#if CUDNN_VERSION >= 92600 && !defined(_WIN32)
    py::enum_<cudnnGnnAggOp_t>(m, "gnn_agg_op")
        .value("SUM", CUDNN_GNN_AGG_SUM)
        .value("MEAN", CUDNN_GNN_AGG_MEAN)
        .value("MAX", CUDNN_GNN_AGG_MAX)
        .value("MIN", CUDNN_GNN_AGG_MIN);

    m.def(
        "gnn_agg_simple_forward",
        [](std::intptr_t stream,
           std::intptr_t csc_offsets,
           std::intptr_t csc_indices,
           std::intptr_t map_csc_to_coo,
           std::int64_t n_src_nodes,
           std::int64_t n_dst_nodes,
           std::int64_t n_indices,
           int idx_type,
           std::intptr_t node_features,
           std::intptr_t edge_features,
           std::intptr_t concat_features,
           std::intptr_t output,
           std::intptr_t out_positions,
           int node_feat_dim,
           int edge_feat_dim,
           int concat_feat_dim,
           int data_type,
           int agg_op) {
            ensure_cuda_runtime_context();
            auto graph =
                make_csc_graph(csc_offsets, csc_indices, map_csc_to_coo, n_src_nodes, n_dst_nodes, n_indices, idx_type);
            auto status = detail::gnn_agg_simple_forward(reinterpret_cast<cudaStream_t>(stream),
                                                         &graph,
                                                         reinterpret_cast<const void *>(node_features),
                                                         reinterpret_cast<const void *>(edge_features),
                                                         reinterpret_cast<const void *>(concat_features),
                                                         reinterpret_cast<void *>(output),
                                                         reinterpret_cast<void *>(out_positions),
                                                         node_feat_dim,
                                                         edge_feat_dim,
                                                         concat_feat_dim,
                                                         static_cast<cudnnDataType_t>(data_type),
                                                         static_cast<cudnnGnnAggOp_t>(agg_op));
            throw_if_gnn_failed(status, "cudnnGnnAggSimpleForward");
        },
        py::arg("stream"),
        py::arg("csc_offsets"),
        py::arg("csc_indices"),
        py::arg("map_csc_to_coo"),
        py::arg("n_src_nodes"),
        py::arg("n_dst_nodes"),
        py::arg("n_indices"),
        py::arg("idx_type"),
        py::arg("node_features"),
        py::arg("edge_features"),
        py::arg("concat_features"),
        py::arg("output"),
        py::arg("out_positions"),
        py::arg("node_feat_dim"),
        py::arg("edge_feat_dim"),
        py::arg("concat_feat_dim"),
        py::arg("data_type"),
        py::arg("agg_op"));

    m.def(
        "gnn_agg_simple_backward",
        [](std::intptr_t stream,
           std::intptr_t csc_offsets,
           std::intptr_t csc_indices,
           std::intptr_t map_csc_to_coo,
           std::int64_t n_src_nodes,
           std::int64_t n_dst_nodes,
           std::int64_t n_indices,
           int idx_type,
           std::intptr_t grad_output,
           std::intptr_t out_positions,
           std::intptr_t grad_node_features,
           std::intptr_t grad_edge_features,
           std::intptr_t grad_concat_features,
           int node_feat_dim,
           int edge_feat_dim,
           int concat_feat_dim,
           int data_type,
           int agg_op) {
            ensure_cuda_runtime_context();
            auto graph =
                make_csc_graph(csc_offsets, csc_indices, map_csc_to_coo, n_src_nodes, n_dst_nodes, n_indices, idx_type);
            auto status = detail::gnn_agg_simple_backward(reinterpret_cast<cudaStream_t>(stream),
                                                          &graph,
                                                          reinterpret_cast<const void *>(grad_output),
                                                          reinterpret_cast<const void *>(out_positions),
                                                          reinterpret_cast<void *>(grad_node_features),
                                                          reinterpret_cast<void *>(grad_edge_features),
                                                          reinterpret_cast<void *>(grad_concat_features),
                                                          node_feat_dim,
                                                          edge_feat_dim,
                                                          concat_feat_dim,
                                                          static_cast<cudnnDataType_t>(data_type),
                                                          static_cast<cudnnGnnAggOp_t>(agg_op));
            throw_if_gnn_failed(status, "cudnnGnnAggSimpleBackward");
        },
        py::arg("stream"),
        py::arg("csc_offsets"),
        py::arg("csc_indices"),
        py::arg("map_csc_to_coo"),
        py::arg("n_src_nodes"),
        py::arg("n_dst_nodes"),
        py::arg("n_indices"),
        py::arg("idx_type"),
        py::arg("grad_output"),
        py::arg("out_positions"),
        py::arg("grad_node_features"),
        py::arg("grad_edge_features"),
        py::arg("grad_concat_features"),
        py::arg("node_feat_dim"),
        py::arg("edge_feat_dim"),
        py::arg("concat_feat_dim"),
        py::arg("data_type"),
        py::arg("agg_op"));
#endif

#if !defined(_WIN32)
    py::enum_<detail::gnn_activation_op_t>(m, "gnn_activation_op")
        .value("LINEAR", detail::gnn_activation_op_t::LINEAR)
        .value("RELU", detail::gnn_activation_op_t::RELU)
        .value("SIGMOID", detail::gnn_activation_op_t::SIGMOID)
        .value("TANH", detail::gnn_activation_op_t::TANH)
        .value("ELU", detail::gnn_activation_op_t::ELU)
        .value("SCALAR", detail::gnn_activation_op_t::SCALAR)
        .value("LEAKY_RELU", detail::gnn_activation_op_t::LEAKY_RELU);

    m.def(
        "gnn_mha_gat_forward",
        [](std::intptr_t stream,
           std::intptr_t csc_offsets,
           std::intptr_t csc_indices,
           std::intptr_t map_csc_to_coo,
           std::int64_t n_src_nodes,
           std::int64_t n_dst_nodes,
           std::int64_t n_indices,
           int idx_type,
           std::intptr_t src_features,
           std::intptr_t dst_features,
           std::intptr_t edge_features,
           std::intptr_t attn_weights,
           std::intptr_t dropout_mask,
           std::intptr_t output,
           std::intptr_t sm_scores,
           int node_feat_dim,
           int edge_feat_dim,
           int activation,
           float activation_alpha,
           int num_heads,
           bool concat_heads,
           int data_type) {
            require_gnn_backend_version(92800, "cudnnGnnMhaGatForward");
            ensure_cuda_runtime_context();
            auto graph =
                make_csc_graph(csc_offsets, csc_indices, map_csc_to_coo, n_src_nodes, n_dst_nodes, n_indices, idx_type);
            detail::gnn_mha_params_t params{static_cast<detail::gnn_activation_op_t>(activation),
                                            activation_alpha,
                                            num_heads,
                                            concat_heads ? 1 : 0};
            auto status = detail::gnn_mha_gat_forward(reinterpret_cast<cudaStream_t>(stream),
                                                      &graph,
                                                      reinterpret_cast<const void *>(src_features),
                                                      reinterpret_cast<const void *>(dst_features),
                                                      reinterpret_cast<const void *>(edge_features),
                                                      reinterpret_cast<const void *>(attn_weights),
                                                      reinterpret_cast<const float *>(dropout_mask),
                                                      reinterpret_cast<void *>(output),
                                                      reinterpret_cast<void *>(sm_scores),
                                                      node_feat_dim,
                                                      edge_feat_dim,
                                                      &params,
                                                      static_cast<cudnnDataType_t>(data_type));
            throw_if_gnn_failed(status, "cudnnGnnMhaGatForward");
        },
        py::arg("stream"),
        py::arg("csc_offsets"),
        py::arg("csc_indices"),
        py::arg("map_csc_to_coo"),
        py::arg("n_src_nodes"),
        py::arg("n_dst_nodes"),
        py::arg("n_indices"),
        py::arg("idx_type"),
        py::arg("src_features"),
        py::arg("dst_features"),
        py::arg("edge_features"),
        py::arg("attn_weights"),
        py::arg("dropout_mask"),
        py::arg("output"),
        py::arg("sm_scores"),
        py::arg("node_feat_dim"),
        py::arg("edge_feat_dim"),
        py::arg("activation"),
        py::arg("activation_alpha"),
        py::arg("num_heads"),
        py::arg("concat_heads"),
        py::arg("data_type"));

    m.def(
        "gnn_mha_gat_backward",
        [](std::intptr_t stream,
           std::intptr_t csc_offsets,
           std::intptr_t csc_indices,
           std::intptr_t map_csc_to_coo,
           std::int64_t n_src_nodes,
           std::int64_t n_dst_nodes,
           std::int64_t n_indices,
           int idx_type,
           std::intptr_t grad_output,
           std::intptr_t src_features,
           std::intptr_t dst_features,
           std::intptr_t edge_features,
           std::intptr_t attn_weights,
           std::intptr_t sm_scores,
           std::intptr_t dropout_mask,
           std::intptr_t grad_attention,
           std::intptr_t grad_src_features,
           std::intptr_t grad_dst_features,
           std::intptr_t grad_edge_features,
           std::intptr_t grad_weights,
           std::intptr_t grad_sm_scores,
           int node_feat_dim,
           int edge_feat_dim,
           int activation,
           float activation_alpha,
           int num_heads,
           bool concat_heads,
           int data_type,
           std::intptr_t csc_rev_offsets,
           std::intptr_t map_rev_to_coo,
           std::intptr_t grad_workspace_features,
           std::intptr_t grad_workspace_weights,
           int grad_data_type,
           int grad_weight_type) {
            require_gnn_backend_version(92800, "cudnnGnnMhaGatBackward");
            ensure_cuda_runtime_context();
            auto graph = make_csc_graph(csc_offsets,
                                        csc_indices,
                                        map_csc_to_coo,
                                        n_src_nodes,
                                        n_dst_nodes,
                                        n_indices,
                                        idx_type,
                                        csc_rev_offsets,
                                        map_rev_to_coo);
            detail::gnn_mha_params_t params{static_cast<detail::gnn_activation_op_t>(activation),
                                            activation_alpha,
                                            num_heads,
                                            concat_heads ? 1 : 0};
            auto status = detail::gnn_mha_gat_backward(
                reinterpret_cast<cudaStream_t>(stream),
                &graph,
                reinterpret_cast<const void *>(grad_output),
                reinterpret_cast<const void *>(src_features),
                reinterpret_cast<const void *>(dst_features),
                reinterpret_cast<const void *>(edge_features),
                reinterpret_cast<const void *>(attn_weights),
                reinterpret_cast<const void *>(sm_scores),
                reinterpret_cast<const float *>(dropout_mask),
                reinterpret_cast<const float *>(grad_attention),
                reinterpret_cast<void *>(grad_src_features),
                reinterpret_cast<void *>(grad_dst_features),
                reinterpret_cast<void *>(grad_edge_features),
                reinterpret_cast<void *>(grad_weights),
                reinterpret_cast<void *>(grad_sm_scores),
                reinterpret_cast<void *>(grad_workspace_features),
                reinterpret_cast<void *>(grad_workspace_weights),
                node_feat_dim,
                edge_feat_dim,
                &params,
                static_cast<cudnnDataType_t>(data_type),
                static_cast<cudnnDataType_t>(grad_data_type < 0 ? data_type : grad_data_type),
                static_cast<cudnnDataType_t>(grad_weight_type < 0 ? data_type : grad_weight_type));
            throw_if_gnn_failed(status, "cudnnGnnMhaGatBackward");
        },
        py::arg("stream"),
        py::arg("csc_offsets"),
        py::arg("csc_indices"),
        py::arg("map_csc_to_coo"),
        py::arg("n_src_nodes"),
        py::arg("n_dst_nodes"),
        py::arg("n_indices"),
        py::arg("idx_type"),
        py::arg("grad_output"),
        py::arg("src_features"),
        py::arg("dst_features"),
        py::arg("edge_features"),
        py::arg("attn_weights"),
        py::arg("sm_scores"),
        py::arg("dropout_mask"),
        py::arg("grad_attention"),
        py::arg("grad_src_features"),
        py::arg("grad_dst_features"),
        py::arg("grad_edge_features"),
        py::arg("grad_weights"),
        py::arg("grad_sm_scores"),
        py::arg("node_feat_dim"),
        py::arg("edge_feat_dim"),
        py::arg("activation"),
        py::arg("activation_alpha"),
        py::arg("num_heads"),
        py::arg("concat_heads"),
        py::arg("data_type"),
        py::arg("csc_rev_offsets"),
        py::arg("map_rev_to_coo"),
        py::arg("grad_workspace_features"),
        py::arg("grad_workspace_weights"),
        py::arg("grad_data_type")   = -1,
        py::arg("grad_weight_type") = -1);

    m.def(
        "gnn_mha_gat_v2_forward",
        [](std::intptr_t stream,
           std::intptr_t csc_offsets,
           std::intptr_t csc_indices,
           std::intptr_t map_csc_to_coo,
           std::int64_t n_src_nodes,
           std::int64_t n_dst_nodes,
           std::int64_t n_indices,
           int idx_type,
           std::intptr_t src_features,
           std::intptr_t dst_features,
           std::intptr_t edge_features,
           std::intptr_t attn_weights,
           std::intptr_t dropout_mask,
           std::intptr_t output,
           std::intptr_t sm_scores,
           std::intptr_t act_scores,
           int node_feat_dim,
           int activation,
           float activation_alpha,
           int num_heads,
           bool concat_heads,
           int data_type) {
            require_gnn_backend_version(92800, "cudnnGnnMhaGatV2Forward");
            ensure_cuda_runtime_context();
            auto graph =
                make_csc_graph(csc_offsets, csc_indices, map_csc_to_coo, n_src_nodes, n_dst_nodes, n_indices, idx_type);
            detail::gnn_mha_params_t params{static_cast<detail::gnn_activation_op_t>(activation),
                                            activation_alpha,
                                            num_heads,
                                            concat_heads ? 1 : 0};
            auto status = detail::gnn_mha_gat_v2_forward(reinterpret_cast<cudaStream_t>(stream),
                                                         &graph,
                                                         reinterpret_cast<const void *>(src_features),
                                                         reinterpret_cast<const void *>(dst_features),
                                                         reinterpret_cast<const void *>(edge_features),
                                                         reinterpret_cast<const void *>(attn_weights),
                                                         reinterpret_cast<const float *>(dropout_mask),
                                                         reinterpret_cast<void *>(output),
                                                         reinterpret_cast<void *>(sm_scores),
                                                         reinterpret_cast<void *>(act_scores),
                                                         node_feat_dim,
                                                         &params,
                                                         static_cast<cudnnDataType_t>(data_type));
            throw_if_gnn_failed(status, "cudnnGnnMhaGatV2Forward");
        },
        py::arg("stream"),
        py::arg("csc_offsets"),
        py::arg("csc_indices"),
        py::arg("map_csc_to_coo"),
        py::arg("n_src_nodes"),
        py::arg("n_dst_nodes"),
        py::arg("n_indices"),
        py::arg("idx_type"),
        py::arg("src_features"),
        py::arg("dst_features"),
        py::arg("edge_features"),
        py::arg("attn_weights"),
        py::arg("dropout_mask"),
        py::arg("output"),
        py::arg("sm_scores"),
        py::arg("act_scores"),
        py::arg("node_feat_dim"),
        py::arg("activation"),
        py::arg("activation_alpha"),
        py::arg("num_heads"),
        py::arg("concat_heads"),
        py::arg("data_type"));

    m.def(
        "gnn_mha_gat_v2_backward",
        [](std::intptr_t stream,
           std::intptr_t csc_offsets,
           std::intptr_t csc_indices,
           std::intptr_t map_csc_to_coo,
           std::int64_t n_src_nodes,
           std::int64_t n_dst_nodes,
           std::int64_t n_indices,
           int idx_type,
           std::intptr_t grad_output,
           std::intptr_t src_features,
           std::intptr_t dst_features,
           std::intptr_t edge_features,
           std::intptr_t attn_weights,
           std::intptr_t sm_scores,
           std::intptr_t act_scores,
           std::intptr_t dropout_mask,
           std::intptr_t grad_attention,
           std::intptr_t grad_src_features,
           std::intptr_t grad_dst_features,
           std::intptr_t grad_edge_features,
           std::intptr_t grad_weights,
           std::intptr_t grad_sm_scores,
           int node_feat_dim,
           int activation,
           float activation_alpha,
           int num_heads,
           bool concat_heads,
           int data_type,
           std::intptr_t csc_rev_offsets,
           std::intptr_t map_rev_to_coo,
           std::intptr_t grad_workspace_features,
           std::intptr_t grad_workspace_weights,
           int grad_type) {
            require_gnn_backend_version(92800, "cudnnGnnMhaGatV2Backward");
            ensure_cuda_runtime_context();
            auto graph = make_csc_graph(csc_offsets,
                                        csc_indices,
                                        map_csc_to_coo,
                                        n_src_nodes,
                                        n_dst_nodes,
                                        n_indices,
                                        idx_type,
                                        csc_rev_offsets,
                                        map_rev_to_coo);
            detail::gnn_mha_params_t params{static_cast<detail::gnn_activation_op_t>(activation),
                                            activation_alpha,
                                            num_heads,
                                            concat_heads ? 1 : 0};
            auto status =
                detail::gnn_mha_gat_v2_backward(reinterpret_cast<cudaStream_t>(stream),
                                                &graph,
                                                reinterpret_cast<const void *>(grad_output),
                                                reinterpret_cast<const void *>(src_features),
                                                reinterpret_cast<const void *>(dst_features),
                                                reinterpret_cast<const void *>(edge_features),
                                                reinterpret_cast<const void *>(attn_weights),
                                                reinterpret_cast<const void *>(sm_scores),
                                                reinterpret_cast<const void *>(act_scores),
                                                reinterpret_cast<const float *>(dropout_mask),
                                                reinterpret_cast<const float *>(grad_attention),
                                                reinterpret_cast<void *>(grad_src_features),
                                                reinterpret_cast<void *>(grad_dst_features),
                                                reinterpret_cast<void *>(grad_edge_features),
                                                reinterpret_cast<void *>(grad_weights),
                                                reinterpret_cast<void *>(grad_sm_scores),
                                                reinterpret_cast<void *>(grad_workspace_features),
                                                reinterpret_cast<void *>(grad_workspace_weights),
                                                node_feat_dim,
                                                &params,
                                                static_cast<cudnnDataType_t>(data_type),
                                                static_cast<cudnnDataType_t>(grad_type < 0 ? data_type : grad_type));
            throw_if_gnn_failed(status, "cudnnGnnMhaGatV2Backward");
        },
        py::arg("stream"),
        py::arg("csc_offsets"),
        py::arg("csc_indices"),
        py::arg("map_csc_to_coo"),
        py::arg("n_src_nodes"),
        py::arg("n_dst_nodes"),
        py::arg("n_indices"),
        py::arg("idx_type"),
        py::arg("grad_output"),
        py::arg("src_features"),
        py::arg("dst_features"),
        py::arg("edge_features"),
        py::arg("attn_weights"),
        py::arg("sm_scores"),
        py::arg("act_scores"),
        py::arg("dropout_mask"),
        py::arg("grad_attention"),
        py::arg("grad_src_features"),
        py::arg("grad_dst_features"),
        py::arg("grad_edge_features"),
        py::arg("grad_weights"),
        py::arg("grad_sm_scores"),
        py::arg("node_feat_dim"),
        py::arg("activation"),
        py::arg("activation_alpha"),
        py::arg("num_heads"),
        py::arg("concat_heads"),
        py::arg("data_type"),
        py::arg("csc_rev_offsets"),
        py::arg("map_rev_to_coo"),
        py::arg("grad_workspace_features"),
        py::arg("grad_workspace_weights"),
        py::arg("grad_type") = -1);
#endif
}

}  // namespace cudnn_frontend::python_bindings
