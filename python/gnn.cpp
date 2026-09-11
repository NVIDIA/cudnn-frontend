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

// cuDNN GNN APIs are introduced in 9.26 and are not supported on Windows.
#if CUDNN_VERSION >= 92600 && !defined(_WIN32)
cudnnGnnCscGraph_t
make_csc_graph(std::intptr_t csc_offsets,
               std::intptr_t csc_indices,
               std::intptr_t map_csc_to_coo,
               std::int64_t n_src_nodes,
               std::int64_t n_dst_nodes,
               std::int64_t n_indices,
               int idx_type,
               std::intptr_t csc_rev_offsets = 0,
               std::intptr_t map_rev_to_coo  = 0) {
    cudnnGnnCscGraph_t graph{};
    graph.cscOffsets  = reinterpret_cast<const void *>(csc_offsets);
    graph.cscIndices  = reinterpret_cast<const void *>(csc_indices);
    graph.mapCscToCoo = reinterpret_cast<const void *>(map_csc_to_coo);
#if CUDNN_VERSION >= 92700
    graph.cscRevOffsets = reinterpret_cast<const void *>(csc_rev_offsets);
    graph.mapRevToCoo   = reinterpret_cast<const void *>(map_rev_to_coo);
#else
    (void)csc_rev_offsets;
    (void)map_rev_to_coo;
#endif
    graph.nSrcNodes = n_src_nodes;
    graph.nDstNodes = n_dst_nodes;
    graph.nIndices  = n_indices;
    graph.idxType   = static_cast<cudnnDataType_t>(idx_type);
    return graph;
}

void
throw_if_gnn_failed(cudnnStatus_t status, char const *operation) {
    if (status == CUDNN_STATUS_SUCCESS) return;

    std::string message = std::string(operation) + " failed: " + detail::get_error_string(status);
    if (status == CUDNN_STATUS_BAD_PARAM) {
        throw std::invalid_argument(message);
    }
    if (status == CUDNN_STATUS_NOT_SUPPORTED || status == CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH) {
        throw cudnnGraphNotSupportedException(message.c_str());
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
    // maybe_unused: with pre-9.26 cuDNN headers (or on Windows) the #if body
    // below compiles away entirely and -Werror=unused-parameter breaks the
    // build (seen with the pip source build inside containers shipping older
    // cuDNN headers).
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

#if CUDNN_VERSION >= 92700
    py::enum_<cudnnGnnActivationOp_t>(m, "gnn_activation_op")
        .value("LINEAR", CUDNN_GNN_ACT_LINEAR)
        .value("RELU", CUDNN_GNN_ACT_RELU)
        .value("SIGMOID", CUDNN_GNN_ACT_SIGMOID)
        .value("TANH", CUDNN_GNN_ACT_TANH)
        .value("ELU", CUDNN_GNN_ACT_ELU)
        .value("SCALAR", CUDNN_GNN_ACT_SCALAR)
        .value("LEAKY_RELU", CUDNN_GNN_ACT_LEAKY_RELU);

    m.def("gnn_mha_gat_forward",
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
              ensure_cuda_runtime_context();
              auto graph = make_csc_graph(
                  csc_offsets, csc_indices, map_csc_to_coo, n_src_nodes, n_dst_nodes, n_indices, idx_type);
              cudnnGnnMhaParams_t params{
                  static_cast<cudnnGnnActivationOp_t>(activation), activation_alpha, num_heads, concat_heads ? 1 : 0};
              auto status = detail::gnn_mha_gat_forward(reinterpret_cast<cudaStream_t>(stream),
                                                        &graph,
                                                        reinterpret_cast<const void *>(node_features),
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
          });

    m.def("gnn_mha_gat_backward",
          [](std::intptr_t stream,
             std::intptr_t csc_offsets,
             std::intptr_t csc_indices,
             std::intptr_t map_csc_to_coo,
             std::int64_t n_src_nodes,
             std::int64_t n_dst_nodes,
             std::int64_t n_indices,
             int idx_type,
             std::intptr_t grad_output,
             std::intptr_t node_features,
             std::intptr_t edge_features,
             std::intptr_t attn_weights,
             std::intptr_t sm_scores,
             std::intptr_t dropout_mask,
             std::intptr_t grad_attention,
             std::intptr_t grad_node_features,
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
             std::intptr_t grad_workspace_weights) {
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
              cudnnGnnMhaParams_t params{
                  static_cast<cudnnGnnActivationOp_t>(activation), activation_alpha, num_heads, concat_heads ? 1 : 0};
              auto status = detail::gnn_mha_gat_backward(reinterpret_cast<cudaStream_t>(stream),
                                                         &graph,
                                                         reinterpret_cast<const void *>(grad_output),
                                                         reinterpret_cast<const void *>(node_features),
                                                         reinterpret_cast<const void *>(edge_features),
                                                         reinterpret_cast<const void *>(attn_weights),
                                                         reinterpret_cast<const void *>(sm_scores),
                                                         reinterpret_cast<const float *>(dropout_mask),
                                                         reinterpret_cast<const float *>(grad_attention),
                                                         reinterpret_cast<void *>(grad_node_features),
                                                         reinterpret_cast<void *>(grad_edge_features),
                                                         reinterpret_cast<void *>(grad_weights),
                                                         reinterpret_cast<void *>(grad_sm_scores),
                                                         reinterpret_cast<void *>(grad_workspace_features),
                                                         reinterpret_cast<void *>(grad_workspace_weights),
                                                         node_feat_dim,
                                                         edge_feat_dim,
                                                         &params,
                                                         static_cast<cudnnDataType_t>(data_type),
                                                         static_cast<cudnnDataType_t>(data_type),
                                                         static_cast<cudnnDataType_t>(data_type));
              throw_if_gnn_failed(status, "cudnnGnnMhaGatBackward");
          });

    m.def("gnn_mha_gat_v2_forward",
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
              ensure_cuda_runtime_context();
              auto graph = make_csc_graph(
                  csc_offsets, csc_indices, map_csc_to_coo, n_src_nodes, n_dst_nodes, n_indices, idx_type);
              cudnnGnnMhaParams_t params{
                  static_cast<cudnnGnnActivationOp_t>(activation), activation_alpha, num_heads, concat_heads ? 1 : 0};
              auto status = detail::gnn_mha_gat_v2_forward(reinterpret_cast<cudaStream_t>(stream),
                                                           &graph,
                                                           reinterpret_cast<const void *>(node_features),
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
          });

    m.def("gnn_mha_gat_v2_backward",
          [](std::intptr_t stream,
             std::intptr_t csc_offsets,
             std::intptr_t csc_indices,
             std::intptr_t map_csc_to_coo,
             std::int64_t n_src_nodes,
             std::int64_t n_dst_nodes,
             std::int64_t n_indices,
             int idx_type,
             std::intptr_t grad_output,
             std::intptr_t node_features,
             std::intptr_t edge_features,
             std::intptr_t attn_weights,
             std::intptr_t sm_scores,
             std::intptr_t act_scores,
             std::intptr_t dropout_mask,
             std::intptr_t grad_attention,
             std::intptr_t grad_node_features,
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
             std::intptr_t grad_workspace_weights) {
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
              cudnnGnnMhaParams_t params{
                  static_cast<cudnnGnnActivationOp_t>(activation), activation_alpha, num_heads, concat_heads ? 1 : 0};
              auto status = detail::gnn_mha_gat_v2_backward(reinterpret_cast<cudaStream_t>(stream),
                                                            &graph,
                                                            reinterpret_cast<const void *>(grad_output),
                                                            reinterpret_cast<const void *>(node_features),
                                                            reinterpret_cast<const void *>(edge_features),
                                                            reinterpret_cast<const void *>(attn_weights),
                                                            reinterpret_cast<const void *>(sm_scores),
                                                            reinterpret_cast<const void *>(act_scores),
                                                            reinterpret_cast<const float *>(dropout_mask),
                                                            reinterpret_cast<const float *>(grad_attention),
                                                            reinterpret_cast<void *>(grad_node_features),
                                                            reinterpret_cast<void *>(grad_edge_features),
                                                            reinterpret_cast<void *>(grad_weights),
                                                            reinterpret_cast<void *>(grad_sm_scores),
                                                            reinterpret_cast<void *>(grad_workspace_features),
                                                            reinterpret_cast<void *>(grad_workspace_weights),
                                                            node_feat_dim,
                                                            &params,
                                                            static_cast<cudnnDataType_t>(data_type),
                                                            static_cast<cudnnDataType_t>(data_type));
              throw_if_gnn_failed(status, "cudnnGnnMhaGatV2Backward");
          });
#endif
#endif
}

}  // namespace cudnn_frontend::python_bindings
