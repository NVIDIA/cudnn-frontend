/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
Reproducer for the CUDA graph topology / lifecycle investigation tracked in
NVIDIA/cudnn-frontend#186.

Run it with:
  bin/samples "[cudagraph_186]" -s
  bin/samples "[cudagraph_186_multidevice]" -s        # needs 2 visible devices

This file deliberately does not assert a fixed number of backend root or kernel
nodes. It dumps what the driver reports for each encapsulation layer and only
compares layers relatively (for example: "the caller parent adds exactly one
more child graph level than the frontend graph it embeds").

Paths covered (see the execution plan for #186):
  P0  plain execution of the same plan (numerical baseline)
  P1  current frontend populate_cuda_graph -> instantiate -> launch
  P2  P1's frontend graph embedded as the child of a caller-owned parent graph
  P3  direct population of an empty caller graph (populate_cuda_graph_direct)
  P4  frontend auxiliary memcpy node plus the backend graph (SDPA bwd + alibi)

Correctness / lifecycle matrix covered: G01..G10.

Set CUDNN186_LOG=<path> to also append every printed line to a raw log file.
*/

#include "../utils/helpers.h"
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <cudnn_frontend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace fe = cudnn_frontend;

std::shared_ptr<fe::graph::Graph>
create_sdpa_backward_graph(int64_t const b,
                           int64_t const h_q,
                           int64_t const h_k,
                           int64_t const h_v,
                           int64_t const s_q,
                           int64_t const s_kv,
                           int64_t const d_qk,
                           int64_t const d_v,
                           float const attn_scale,
                           bool const generate_stats,
                           bool const causal_mask,
                           bool const alibi_mask,
                           bool const padding_mask,
                           bool has_attn_bias,
                           bool is_deterministic);

// Graph builders already defined by other translation units of the same
// `samples` binary. Reused here so this reproducer exercises the same frontend
// graphs as the shipped samples instead of a private look-alike.
std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph(int64_t const b,
                          int64_t const h_q,
                          int64_t const h_k,
                          int64_t const h_v,
                          int64_t const s_q,
                          int64_t const s_kv,
                          int64_t const d_qk,
                          int64_t const d_v,
                          float const attn_scale,
                          bool const generate_stats,
                          bool const causal_mask,
                          bool const padding_mask);

namespace {

// ===========================================================================
// Evidence logging
// ===========================================================================

std::ofstream &
evidence_file() {
    static std::ofstream file = []() -> std::ofstream {
        char const *path = std::getenv("CUDNN186_LOG");
        if (path == nullptr || path[0] == '\0') {
            return std::ofstream();
        }
        return std::ofstream(path, std::ios::app);
    }();
    return file;
}

void
ev(std::string const &line) {
    std::cout << line << '\n';
    auto &file = evidence_file();
    if (file.is_open()) {
        file << line << '\n';
        file.flush();
    }
}

void
ev_header(std::string const &title) {
    ev("");
    ev("[186] ==================== " + title + " ====================");
}

std::string
fmt(double value, int precision = 3) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return std::string(buffer);
}

std::string
note_name(fe::BehaviorNote_t note) {
    switch (note) {
        case fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API:
            return "SUPPORTS_CUDA_GRAPH_NATIVE_API";
        default:
            return "OTHER";
    }
}

size_t
host_rss_bytes() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    long total_pages = 0;
    long rss_pages   = 0;
    if (!(statm >> total_pages >> rss_pages)) {
        return 0;
    }
    (void)total_pages;
    return static_cast<size_t>(rss_pages) * static_cast<size_t>(sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

// Returns {free_bytes, total_bytes}.
std::pair<size_t, size_t>
device_memory() {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return {0, 0};
    }
    return {free_bytes, total_bytes};
}

// ===========================================================================
// CUDA graph topology dump
// ===========================================================================

char const *
node_type_name(cudaGraphNodeType type) {
    switch (type) {
        case cudaGraphNodeTypeKernel:
            return "kernel";
        case cudaGraphNodeTypeMemcpy:
            return "memcpy";
        case cudaGraphNodeTypeMemset:
            return "memset";
        case cudaGraphNodeTypeHost:
            return "host";
        case cudaGraphNodeTypeGraph:
            return "child_graph";
        case cudaGraphNodeTypeEmpty:
            return "empty";
        case cudaGraphNodeTypeWaitEvent:
            return "wait_event";
        case cudaGraphNodeTypeEventRecord:
            return "event_record";
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11040
        case cudaGraphNodeTypeExtSemaphoreSignal:
            return "ext_semaphore_signal";
        case cudaGraphNodeTypeExtSemaphoreWait:
            return "ext_semaphore_wait";
#endif
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        case cudaGraphNodeTypeMemAlloc:
            return "mem_alloc";
        case cudaGraphNodeTypeMemFree:
            return "mem_free";
#endif
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        case cudaGraphNodeTypeConditional:
            return "conditional";
#endif
        default:
            return "unknown";
    }
}

struct TopoSummary {
    std::string label;
    size_t root_nodes      = 0;  // roots at the top level only
    size_t top_level_nodes = 0;  // nodes at the top level only
    size_t total_nodes     = 0;  // recursive
    size_t kernels         = 0;  // recursive
    size_t memcpys         = 0;
    size_t memsets         = 0;
    size_t child_graphs    = 0;
    size_t empty_nodes     = 0;
    size_t other_nodes     = 0;
    int max_child_depth    = 0;  // 0 == no child graph node
    std::vector<std::string> lines;
};

void
walk_graph(cudaGraph_t graph, std::string const &path, int depth, TopoSummary &summary) {
    size_t node_count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &node_count) != cudaSuccess) {
        summary.lines.push_back(path + ": <cudaGraphGetNodes failed>");
        return;
    }
    std::vector<cudaGraphNode_t> nodes(node_count);
    if (node_count > 0) {
        if (cudaGraphGetNodes(graph, nodes.data(), &node_count) != cudaSuccess) {
            summary.lines.push_back(path + ": <cudaGraphGetNodes(2) failed>");
            return;
        }
    }
    size_t root_count = 0;
    if (cudaGraphGetRootNodes(graph, nullptr, &root_count) != cudaSuccess) {
        summary.lines.push_back(path + ": <cudaGraphGetRootNodes failed>");
        return;
    }
    if (depth == 0) {
        summary.root_nodes      = root_count;
        summary.top_level_nodes = node_count;
    }
    summary.total_nodes += node_count;
    summary.lines.push_back(path + ": nodes=" + std::to_string(node_count) + " roots=" + std::to_string(root_count));

    std::unordered_map<cudaGraphNode_t, size_t> index_of;
    for (size_t i = 0; i < node_count; ++i) {
        index_of.emplace(nodes[i], i);
    }

    for (size_t i = 0; i < node_count; ++i) {
        cudaGraphNodeType type = cudaGraphNodeTypeCount;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) {
            continue;
        }
        switch (type) {
            case cudaGraphNodeTypeKernel:
                summary.kernels++;
                break;
            case cudaGraphNodeTypeMemcpy:
                summary.memcpys++;
                break;
            case cudaGraphNodeTypeMemset:
                summary.memsets++;
                break;
            case cudaGraphNodeTypeGraph:
                summary.child_graphs++;
                break;
            case cudaGraphNodeTypeEmpty:
                summary.empty_nodes++;
                break;
            default:
                summary.other_nodes++;
                break;
        }

        std::string dependents = "[";
        size_t dependent_count = 0;
        if (cudnn_frontend::detail::cuda_graph_node_get_dependent_nodes(nodes[i], nullptr, &dependent_count) ==
            cudaSuccess) {
            std::vector<cudaGraphNode_t> dependent_nodes(dependent_count);
            if (dependent_count > 0 && cudnn_frontend::detail::cuda_graph_node_get_dependent_nodes(
                                           nodes[i], dependent_nodes.data(), &dependent_count) == cudaSuccess) {
                for (size_t d = 0; d < dependent_count; ++d) {
                    auto it = index_of.find(dependent_nodes[d]);
                    dependents +=
                        (it == index_of.end()) ? std::string("?") : (path + ".n" + std::to_string(it->second));
                    dependents += " ";
                }
            }
        }
        dependents += "]";

        std::string dependencies = "[";
        size_t dependency_count  = 0;
        if (cudnn_frontend::detail::cuda_graph_node_get_dependencies(nodes[i], nullptr, &dependency_count) ==
            cudaSuccess) {
            std::vector<cudaGraphNode_t> dependency_nodes(dependency_count);
            if (dependency_count > 0 && cudnn_frontend::detail::cuda_graph_node_get_dependencies(
                                            nodes[i], dependency_nodes.data(), &dependency_count) == cudaSuccess) {
                for (size_t d = 0; d < dependency_count; ++d) {
                    auto it = index_of.find(dependency_nodes[d]);
                    dependencies +=
                        (it == index_of.end()) ? std::string("?") : (path + ".n" + std::to_string(it->second));
                    dependencies += " ";
                }
            }
        }
        dependencies += "]";

        std::string const node_path = path + ".n" + std::to_string(i);
        summary.lines.push_back(node_path + " type=" + node_type_name(type) + " dependencies=" + dependencies +
                                " dependents=" + dependents);

        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child = nullptr;
            if (cudaGraphChildGraphNodeGetGraph(nodes[i], &child) == cudaSuccess) {
                if (depth + 1 > summary.max_child_depth) {
                    summary.max_child_depth = depth + 1;
                }
                walk_graph(child, node_path + "/child", depth + 1, summary);
            }
        }
    }
}

TopoSummary
analyze_graph(cudaGraph_t graph, std::string const &label) {
    TopoSummary summary;
    summary.label = label;
    walk_graph(graph, "L0", 0, summary);
    return summary;
}

void
print_topology(TopoSummary const &summary, bool print_nodes) {
    ev("[186][TOPO] " + summary.label);
    if (print_nodes) {
        for (auto const &line : summary.lines) {
            ev("[186][TOPO]   " + line);
        }
    }
    ev("[186][TOPO] " + summary.label + " SUMMARY root_nodes=" + std::to_string(summary.root_nodes) +
       " top_level_nodes=" + std::to_string(summary.top_level_nodes) +
       " recursive_nodes=" + std::to_string(summary.total_nodes) + " kernel_nodes=" + std::to_string(summary.kernels) +
       " memcpy_nodes=" + std::to_string(summary.memcpys) + " memset_nodes=" + std::to_string(summary.memsets) +
       " child_graph_nodes=" + std::to_string(summary.child_graphs) +
       " empty_nodes=" + std::to_string(summary.empty_nodes) + " other_nodes=" + std::to_string(summary.other_nodes) +
       " max_child_depth=" + std::to_string(summary.max_child_depth));
}

std::vector<cudaGraphNode_t>
top_level_nodes_of_type(cudaGraph_t graph, cudaGraphNodeType wanted) {
    std::vector<cudaGraphNode_t> result;
    size_t node_count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &node_count) != cudaSuccess || node_count == 0) {
        return result;
    }
    std::vector<cudaGraphNode_t> nodes(node_count);
    if (cudaGraphGetNodes(graph, nodes.data(), &node_count) != cudaSuccess) {
        return result;
    }
    for (size_t i = 0; i < node_count; ++i) {
        cudaGraphNodeType type = cudaGraphNodeTypeCount;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) {
            continue;
        }
        if (type == wanted) {
            result.push_back(nodes[i]);
        }
    }
    return result;
}

// Returns the graph held by the first child graph node found at the top level,
// or nullptr. Inspection-only: the library never dispatches on this.
cudaGraph_t
first_child_graph(cudaGraph_t graph) {
    auto nodes = top_level_nodes_of_type(graph, cudaGraphNodeTypeGraph);
    if (nodes.empty()) {
        return nullptr;
    }
    cudaGraph_t child = nullptr;
    if (cudaGraphChildGraphNodeGetGraph(nodes[0], &child) != cudaSuccess) {
        return nullptr;
    }
    return child;
}

size_t
count_nodes(cudaGraph_t graph) {
    size_t node_count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &node_count) != cudaSuccess) {
        return 0;
    }
    return node_count;
}

// ===========================================================================
// Plan preparation
// ===========================================================================

struct PlanStatus {
    bool built      = false;
    bool native_api = false;
    std::string reason;
};

PlanStatus
prepare_native_plan(std::shared_ptr<fe::graph::Graph> const &graph, cudnnHandle_t handle) {
    PlanStatus status;
    auto result = graph->validate();
    if (result.is_bad()) {
        status.reason = "validate(): " + result.get_message();
        return status;
    }
    result = graph->build_operation_graph(handle);
    if (result.is_bad()) {
        status.reason = "build_operation_graph(): " + result.get_message();
        return status;
    }
    result = graph->create_execution_plans({fe::HeurMode_t::A});
    if (result.is_bad()) {
        status.reason = "create_execution_plans(): " + result.get_message();
        return status;
    }
    graph->select_behavior_notes({fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
    result = graph->check_support();
    if (result.is_bad()) {
        status.reason = "check_support(SUPPORTS_CUDA_GRAPH_NATIVE_API): " + result.get_message();
        return status;
    }
    result = graph->build_plans();
    if (result.is_bad()) {
        status.reason = "build_plans(): " + result.get_message();
        return status;
    }
    std::vector<fe::BehaviorNote_t> notes;
    if (graph->get_behavior_notes(notes).is_good()) {
        for (auto note : notes) {
            if (note == fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API) {
                status.native_api = true;
            }
        }
    }
    status.built  = true;
    status.reason = status.native_api ? "native CUDA graph API supported" : "plan built but native note absent";
    return status;
}

void
print_plan_identity(std::shared_ptr<fe::graph::Graph> const &graph, std::string const &label) {
    std::string candidate_name;
    auto name_status = graph->get_plan_name(candidate_name);

    std::string plan0_name;
    auto plan0_status = graph->get_plan_name_at_index(0, plan0_name);

    int64_t workspace_size = 0;
    auto ws_status         = graph->get_workspace_size(workspace_size);
    std::vector<fe::BehaviorNote_t> notes;
    auto notes_status = graph->get_behavior_notes_for_plan_at_index(0, notes);

    std::string note_names;
    for (auto note : notes) {
        if (!note_names.empty()) {
            note_names += ",";
        }
        note_names += note_name(note);
    }

    ev("[186][PLAN] " + label + " cudnn_version=" + std::to_string(cudnnGetVersion()) + " cudart_version=" +
       std::to_string(cudnnGetCudartVersion()) + " plan_count=" + std::to_string(graph->get_execution_plan_count()) +
       " candidate_plan_name=" + (name_status.is_good() ? candidate_name : std::string("<error>")) +
       " plan0_name=" + (plan0_status.is_good() ? plan0_name : std::string("<error>")) +
       " total_workspace_bytes=" + (ws_status.is_good() ? std::to_string(workspace_size) : std::string("<error>")) +
       " plan0_behavior_notes=" + (notes_status.is_good() ? note_names : std::string("<error>")));
}

int64_t
workspace_size_of(std::shared_ptr<fe::graph::Graph> const &graph) {
    int64_t size = 0;
    auto status  = graph->get_workspace_size(size);
    REQUIRE(status.is_good());
    return size;
}

// ---------------------------------------------------------------------------
// Direct-population facade.
//
// The opt-in direct entry points are added by the same change that adds this
// reproducer. Defining CUDNN186_BASELINE_ONLY builds the very same file against
// a tree that only has the shipping populate/update API, so a paired baseline
// can be produced from one source instead of a forked copy: the facade then
// routes to the wrapped path and reports direct population as unsupported.
// ---------------------------------------------------------------------------
#if defined(CUDNN186_BASELINE_ONLY)
constexpr bool kDirectPopulationSupported = false;
#else
constexpr bool kDirectPopulationSupported = true;
#endif

bool
direct_population_supported() {
    return kDirectPopulationSupported;
}

std::string
direct_path_label() {
    return kDirectPopulationSupported ? std::string("P3") : std::string("P3_absent_from_this_tree");
}

bool
direct_population_eligible(std::shared_ptr<fe::graph::Graph> const &graph) {
#if defined(CUDNN186_BASELINE_ONLY)
    (void)graph;
    return false;
#else
    return graph->cuda_graph_direct_population_eligible();
#endif
}

fe::error_t
populate_graph(std::shared_ptr<fe::graph::Graph> const &graph,
               cudnnHandle_t handle,
               std::unordered_map<int64_t, void *> &variant_pack,
               void *workspace,
               cudaGraph_t target) {
#if defined(CUDNN186_BASELINE_ONLY)
    return graph->populate_cuda_graph(handle, variant_pack, workspace, target);
#else
    return graph->populate_cuda_graph_direct(handle, variant_pack, workspace, target);
#endif
}

fe::error_t
update_graph(std::shared_ptr<fe::graph::Graph> const &graph,
             cudnnHandle_t handle,
             std::unordered_map<int64_t, void *> &variant_pack,
             void *workspace,
             cudaGraph_t target) {
#if defined(CUDNN186_BASELINE_ONLY)
    return graph->update_cuda_graph(handle, variant_pack, workspace, target);
#else
    return graph->update_cuda_graph_direct(handle, variant_pack, workspace, target);
#endif
}

void
log_environment(std::string const &title) {
    int device = 0;
    cudaDeviceProp properties{};
    int visible = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    (void)cudaGetDeviceCount(&visible);
    cudaUUID_t uuid = properties.uuid;
    char uuid_string[64];
    std::snprintf(uuid_string,
                  sizeof(uuid_string),
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  static_cast<unsigned>(uuid.bytes[0]),
                  static_cast<unsigned>(uuid.bytes[1]),
                  static_cast<unsigned>(uuid.bytes[2]),
                  static_cast<unsigned>(uuid.bytes[3]),
                  static_cast<unsigned>(uuid.bytes[4]),
                  static_cast<unsigned>(uuid.bytes[5]),
                  static_cast<unsigned>(uuid.bytes[6]),
                  static_cast<unsigned>(uuid.bytes[7]),
                  static_cast<unsigned>(uuid.bytes[8]),
                  static_cast<unsigned>(uuid.bytes[9]),
                  static_cast<unsigned>(uuid.bytes[10]),
                  static_cast<unsigned>(uuid.bytes[11]),
                  static_cast<unsigned>(uuid.bytes[12]),
                  static_cast<unsigned>(uuid.bytes[13]),
                  static_cast<unsigned>(uuid.bytes[14]),
                  static_cast<unsigned>(uuid.bytes[15]));

    auto memory = device_memory();

    ev_header(title);
    ev("[186][ENV] cudnn_version=" + std::to_string(cudnnGetVersion()) +
       " cudart_version=" + std::to_string(cudnnGetCudartVersion()) +
       " compiled_cudart=" + std::to_string(CUDART_VERSION) + " device_index=" + std::to_string(device) +
       " device_name=" + std::string(properties.name) + " device_uuid=" + std::string(uuid_string) +
       " cc=" + std::to_string(properties.major) + "." + std::to_string(properties.minor) +
       " visible_device_count=" + std::to_string(visible) + " mem_free_bytes=" + std::to_string(memory.first) +
       " mem_total_bytes=" + std::to_string(memory.second) + " host_rss_bytes=" + std::to_string(host_rss_bytes()));
}

// ===========================================================================
// matmul + add fixture (same shape as the existing misc/cudagraphs.cpp)
// ===========================================================================

constexpr int64_t kUidA = 0;
constexpr int64_t kUidB = 1;
constexpr int64_t kUidC = 2;
constexpr int64_t kUidD = 3;

std::shared_ptr<fe::graph::Graph>
make_matmul_add_graph(int64_t b, int64_t m, int64_t n, int64_t k, float scale_value) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph->tensor(
        fe::graph::Tensor_attributes().set_name("A").set_dim({b, m, k}).set_stride({m * k, k, 1}).set_uid(kUidA));

    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto S             = graph->pointwise(A, graph->tensor(scale_value), scale_options);
    S->set_data_type(fe::DataType_t::HALF);

    auto B = graph->tensor(
        fe::graph::Tensor_attributes().set_name("B").set_dim({b, k, n}).set_stride({n * k, n, 1}).set_uid(kUidB));
    auto T = graph->matmul(S, B, fe::graph::Matmul_attributes());

    auto C = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("C")
                               .set_dim({1, 1, 1})
                               .set_stride({1, 1, 1})
                               .set_is_pass_by_value(true)
                               .set_uid(kUidC));

    auto add_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto D           = graph->pointwise(T, C, add_options);
    D->set_output(true).set_uid(kUidD);
    return graph;
}

// ===========================================================================
// Legacy Slice aliasing (a frontend/backend pair below cuDNN 9.22)
// ===========================================================================

constexpr int64_t kUidSliceX = 21;
constexpr int64_t kUidSliceY = 22;
constexpr int64_t kUidSliceZ = 23;
constexpr int64_t kUidSliceW = 24;

// Y = Slice(X)[:, 1:2, :], Z = Y @ W, all rank-3.  When Slice cannot be
// expressed as a backend operation the frontend records a variant-pack
// replacement instead: Y has to be bound to X plus a byte offset, and the
// caller supplies the source pointer.  Rank-3 on purpose: this is the shape the
// legacy lane builds on machines where a rank-2 slice feeding a pointwise has
// no engine at all (measured on cuDNN 9.21.1 / SM100), so the replacement is
// exercised rather than skipped.
std::shared_ptr<fe::graph::Graph>
make_slice_matmul_graph(int64_t b, int64_t s, int64_t d, int64_t n) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph->tensor(
        fe::graph::Tensor_attributes().set_name("X").set_dim({b, s, d}).set_stride({s * d, d, 1}).set_uid(kUidSliceX));
    auto Y = graph->slice(X, fe::graph::Slice_attributes().set_slices({{0, b}, {1, 2}, {0, d}}));
    Y->set_data_type(fe::DataType_t::HALF).set_uid(kUidSliceY);

    auto W = graph->tensor(
        fe::graph::Tensor_attributes().set_name("W").set_dim({b, d, n}).set_stride({d * n, n, 1}).set_uid(kUidSliceW));
    auto Z = graph->matmul(Y, W, fe::graph::Matmul_attributes());
    Z->set_data_type(fe::DataType_t::HALF).set_output(true).set_uid(kUidSliceZ);
    return graph;
}

// Fill values are powers of two / small integers so that the expected result is
// exactly representable and can be compared without a tolerance.
class MatmulAddFixture {
   public:
    MatmulAddFixture(int64_t b,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     float scale,
                     half a_fill,
                     half b_fill,
                     half bias,
                     int64_t workspace_bytes,
                     half output_fill)
        : k_(k),
          scale_(scale),
          a_fill_(a_fill),
          b_fill_(b_fill),
          bias_(bias),
          a_(static_cast<size_t>(b * m * k), a_fill),
          bmat_(static_cast<size_t>(b * k * n), b_fill),
          d_(static_cast<size_t>(b * m * n), output_fill),
          workspace_(static_cast<size_t>(std::max<int64_t>(workspace_bytes, 1))) {
        variant_pack_ = {{kUidA, a_.devPtr}, {kUidB, bmat_.devPtr}, {kUidC, &bias_}, {kUidD, d_.devPtr}};
    }

    std::unordered_map<int64_t, void *> &
    variant_pack() {
        return variant_pack_;
    }

    half
    expected() const {
        return __float2half(scale_ * __half2float(a_fill_) * static_cast<float>(k_) * __half2float(b_fill_) +
                            __half2float(bias_));
    }

    std::vector<half>
    read_output() const {
        std::vector<half> host(d_.size);
        CUDA_CHECK(cudaMemcpy(host.data(), d_.devPtr, sizeof(half) * d_.size, cudaMemcpyDeviceToHost));
        return host;
    }

    bool
    output_all_equal(half value) const {
        for (auto element : read_output()) {
            if (element != value) {
                return false;
            }
        }
        return true;
    }

    void *
    output_ptr() const {
        return d_.devPtr;
    }

    void *
    workspace_ptr() const {
        return workspace_.devPtr;
    }

    void
    fill_output(half value) {
        fillImage(d_.devPtr, d_.size, value);
    }

    int64_t
    output_size() const {
        return d_.size;
    }

   private:
    int64_t k_   = 0;
    float scale_ = 1.0f;
    half a_fill_ = __float2half(0.f);
    half b_fill_ = __float2half(0.f);
    half bias_   = __float2half(0.f);
    Surface<half> a_;
    Surface<half> bmat_;
    Surface<half> d_;
    Surface<int8_t> workspace_;
    std::unordered_map<int64_t, void *> variant_pack_;
};

// ===========================================================================
// SDPA forward + alibi fixture (the frontend auxiliary node case)
// ===========================================================================

// helpers.h' initImage() advances a file-static PRNG, so two fixtures built in
// the same process would otherwise hold different data. This fill depends only
// on (seed, index) and makes the two fixtures bitwise comparable.
void
fill_deterministic_half(half *device_ptr, size_t count, unsigned seed) {
    std::vector<half> host(count);
    unsigned state = seed;
    for (size_t i = 0; i < count; ++i) {
        state   = (1103515245u * state + 12345u) & 0xffffffffu;
        float v = (static_cast<float>(state) * 2.3283064e-10f) - 0.5f;
        host[i] = cpu_float2half_rn(v);
    }
    CUDA_CHECK(cudaMemcpy(device_ptr, host.data(), sizeof(half) * count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

constexpr int64_t kSdpaQ     = 1;
constexpr int64_t kSdpaK     = 2;
constexpr int64_t kSdpaV     = 3;
constexpr int64_t kSdpaO     = 4;
constexpr int64_t kSdpaStats = 5;
constexpr int64_t kSdpaDo    = 101;
constexpr int64_t kSdpaDq    = 102;
constexpr int64_t kSdpaDk    = 103;
constexpr int64_t kSdpaDv    = 104;

// A flash-attention forward graph with an alibi mask. The alibi slopes are not
// a caller-provided tensor: the frontend caches them and emits its own
// workspace memcpy node, which is exactly the auxiliary-node case P4/G06 needs.
std::shared_ptr<fe::graph::Graph>
make_sdpa_fwd_alibi_graph(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, float attn_scale) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(kSdpaQ)
                               .set_dim({b, h, s_q, d})
                               .set_stride({h * s_q * d, s_q * d, d, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(kSdpaK)
                               .set_dim({b, h, s_kv, d})
                               .set_stride({h * s_kv * d, s_kv * d, d, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(kSdpaV)
                               .set_dim({b, h, s_kv, d})
                               .set_stride({h * s_kv * d, s_kv * d, d, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention_alibi")
                            .set_generate_stats(false)
                            .set_alibi_mask(true)
                            .set_diagonal_band_right_bound(0)
                            .set_attn_scale(attn_scale);

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;
    O->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}).set_uid(kSdpaO);
    return graph;
}

class SdpaFwdFixture {
   public:
    SdpaFwdFixture(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, int64_t workspace_bytes)
        : q_(static_cast<size_t>(b * h * s_q * d), __float2half(0.f)),
          k_(static_cast<size_t>(b * h * s_kv * d), __float2half(0.f)),
          v_(static_cast<size_t>(b * h * s_kv * d), __float2half(0.f)),
          o_(static_cast<size_t>(b * h * s_q * d), __float2half(0.f)),
          workspace_(static_cast<size_t>(std::max<int64_t>(workspace_bytes, 1))) {
        fill_deterministic_half(q_.devPtr, q_.size, 11u);
        fill_deterministic_half(k_.devPtr, k_.size, 22u);
        fill_deterministic_half(v_.devPtr, v_.size, 33u);
        variant_pack_ = {{kSdpaQ, q_.devPtr}, {kSdpaK, k_.devPtr}, {kSdpaV, v_.devPtr}, {kSdpaO, o_.devPtr}};
    }

    std::unordered_map<int64_t, void *> &
    variant_pack() {
        return variant_pack_;
    }

    void *
    workspace_ptr() const {
        return workspace_.devPtr;
    }

    std::vector<half>
    read_output() const {
        std::vector<half> host(o_.size);
        CUDA_CHECK(cudaMemcpy(host.data(), o_.devPtr, sizeof(half) * o_.size, cudaMemcpyDeviceToHost));
        return host;
    }

    void
    reset_output() {
        fillImage(o_.devPtr, o_.size, __float2half(0.f));
    }

    bool
    output_is_all_zero() const {
        for (auto element : read_output()) {
            if (__half2float(element) != 0.0f) {
                return false;
            }
        }
        return true;
    }

   private:
    Surface<half> q_;
    Surface<half> k_;
    Surface<half> v_;
    Surface<half> o_;
    Surface<int8_t> workspace_;
    std::unordered_map<int64_t, void *> variant_pack_;
};

// Minimal SDPA backward variant pack plus a wrapped population, used only to
// count the frontend's own CUDA graph nodes for the eligibility discussion.
struct SdpaBwdProbe {
    Surface<half> q_;
    Surface<half> k_;
    Surface<half> v_;
    Surface<half> o_;
    Surface<half> d_o_;
    Surface<float> stats_;
    Surface<half> d_q_;
    Surface<half> d_k_;
    Surface<half> d_v_;
    Surface<int8_t> workspace_;
    std::unordered_map<int64_t, void *> variant_pack_;
    bool populated               = false;
    std::string populate_message = "<not attempted>";
    size_t top_level_nodes       = 0;
    size_t memcpy_nodes          = 0;
    size_t memset_nodes          = 0;
    size_t child_graph_nodes     = 0;

    SdpaBwdProbe(int64_t b,
                 int64_t h,
                 int64_t s_q,
                 int64_t s_kv,
                 int64_t d,
                 int64_t workspace_bytes,
                 cudnnHandle_t handle,
                 std::shared_ptr<fe::graph::Graph> const &graph)
        : q_(static_cast<size_t>(b * h * s_q * d), __float2half(0.f)),
          k_(static_cast<size_t>(b * h * s_kv * d), __float2half(0.f)),
          v_(static_cast<size_t>(b * h * s_kv * d), __float2half(0.f)),
          o_(static_cast<size_t>(b * h * s_q * d), __float2half(0.f)),
          d_o_(static_cast<size_t>(b * h * s_q * d), __float2half(0.f)),
          stats_(static_cast<size_t>(b * h * s_q)),
          d_q_(static_cast<size_t>(b * h * s_q * d), __float2half(0.f)),
          d_k_(static_cast<size_t>(b * h * s_kv * d), __float2half(0.f)),
          d_v_(static_cast<size_t>(b * h * s_kv * d), __float2half(0.f)),
          workspace_(static_cast<size_t>(std::max<int64_t>(workspace_bytes, 1))) {
        fill_deterministic_half(q_.devPtr, q_.size, 11u);
        fill_deterministic_half(k_.devPtr, k_.size, 22u);
        fill_deterministic_half(v_.devPtr, v_.size, 33u);
        fill_deterministic_half(o_.devPtr, o_.size, 44u);
        fill_deterministic_half(d_o_.devPtr, d_o_.size, 55u);
        variant_pack_ = {{kSdpaQ, q_.devPtr},
                         {kSdpaK, k_.devPtr},
                         {kSdpaV, v_.devPtr},
                         {kSdpaO, o_.devPtr},
                         {kSdpaDo, d_o_.devPtr},
                         {kSdpaStats, stats_.devPtr},
                         {kSdpaDq, d_q_.devPtr},
                         {kSdpaDk, d_k_.devPtr},
                         {kSdpaDv, d_v_.devPtr}};

        cudaGraph_t populated_graph = nullptr;
        if (cudaGraphCreate(&populated_graph, 0) != cudaSuccess) {
            populate_message = "<cudaGraphCreate failed>";
            return;
        }
        auto status      = graph->populate_cuda_graph(handle, variant_pack_, workspace_.devPtr, populated_graph);
        populated        = status.is_good();
        populate_message = status.get_message();
        if (populated) {
            top_level_nodes   = count_nodes(populated_graph);
            memcpy_nodes      = top_level_nodes_of_type(populated_graph, cudaGraphNodeTypeMemcpy).size();
            memset_nodes      = top_level_nodes_of_type(populated_graph, cudaGraphNodeTypeMemset).size();
            child_graph_nodes = top_level_nodes_of_type(populated_graph, cudaGraphNodeTypeGraph).size();
        }
        cudaGraphDestroy(populated_graph);
    }
};

// ===========================================================================
// Timing helpers
// ===========================================================================

struct Distribution {
    std::vector<double> samples_us;

    void
    add(double value) {
        samples_us.push_back(value);
    }

    double
    quantile(double q) const {
        if (samples_us.empty()) {
            return 0.0;
        }
        std::vector<double> sorted = samples_us;
        std::sort(sorted.begin(), sorted.end());
        size_t index = static_cast<size_t>(q * static_cast<double>(sorted.size() - 1));
        return sorted[index];
    }

    double
    median() const {
        return quantile(0.5);
    }
    double
    p90() const {
        return quantile(0.9);
    }
    double
    best() const {
        return quantile(0.0);
    }
    size_t
    size() const {
        return samples_us.size();
    }
};

void
report(char const *phase, std::string const &path, Distribution const &distribution, std::string const &mechanism) {
    ev(std::string("[186][TIMING] phase=") + phase + " path=" + path + " mechanism=" + mechanism +
       " n=" + std::to_string(distribution.size()) + " median_us=" + fmt(distribution.median()) +
       " p90_us=" + fmt(distribution.p90()) + " best_us=" + fmt(distribution.best()));
}

template <typename Fn>
Distribution
time_host(int reps, int warmup, Fn &&fn) {
    Distribution distribution;
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    for (int i = 0; i < reps; ++i) {
        auto start = std::chrono::steady_clock::now();
        fn();
        auto stop = std::chrono::steady_clock::now();
        distribution.add(std::chrono::duration<double, std::micro>(stop - start).count());
    }
    return distribution;
}

Distribution
time_device_event(cudaGraphExec_t exec, cudaStream_t stream, int reps, int warmup) {
    Distribution distribution;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        distribution.add(static_cast<double>(elapsed_ms) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return distribution;
}

// Time `inner` back-to-back launches between two events and divide by `inner`,
// which removes most of the per-sample event overhead.
Distribution
time_device_batched(cudaGraphExec_t exec, cudaStream_t stream, int outer, int inner, int warmup_inner) {
    Distribution distribution;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup_inner; ++i) {
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < outer; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int j = 0; j < inner; ++j) {
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        distribution.add(static_cast<double>(elapsed_ms) * 1000.0 / static_cast<double>(inner));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return distribution;
}

Distribution
time_device_execute(std::shared_ptr<fe::graph::Graph> const &graph,
                    cudnnHandle_t handle,
                    std::unordered_map<int64_t, void *> &variant_pack,
                    void *workspace,
                    cudaStream_t stream,
                    int reps,
                    int warmup) {
    Distribution distribution;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) {
        auto status = graph->execute(handle, variant_pack, workspace);
        REQUIRE(status.is_good());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        auto status = graph->execute(handle, variant_pack, workspace);
        REQUIRE(status.is_good());
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        distribution.add(static_cast<double>(elapsed_ms) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return distribution;
}

}  // namespace

// ===========================================================================
// Topology baseline
// ===========================================================================

TEST_CASE("186 topology baseline: encapsulation layers of one native-capable plan", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
        return;
    }
    log_environment("topology baseline (matmul+add)");

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    float const scale = 0.5f;
    half const a_fill = __float2half(1.0f);
    half const b_fill = __float2half(1.0f);
    half const bias   = __float2half(2.0f);

    auto graph  = make_matmul_add_graph(b, m, n, k, scale);
    auto status = prepare_native_plan(graph, handle);
    print_plan_identity(graph, "P0/P1/P2/P3 matmul+add");
    ev("[186][PLAN] prepare_native_plan built=" + std::string(status.built ? "true" : "false") +
       " native_api=" + std::string(status.native_api ? "true" : "false") + " reason=" + status.reason);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture fixture(b, m, n, k, scale, a_fill, b_fill, bias, workspace_bytes, __float2half(0.f));
    auto variant_pack = fixture.variant_pack();

    // ---- P1: the shipping frontend path -------------------------------------
    cudaGraph_t graph_p1 = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_p1, 0));
    REQUIRE(graph->populate_cuda_graph(handle, variant_pack, fixture.workspace_ptr(), graph_p1).is_good());

    auto topology_p1 = analyze_graph(graph_p1, "P1 wrapped frontend populate (caller graph = frontend graph)");
    print_topology(topology_p1, true);

    cudaGraph_t backend_graph = first_child_graph(graph_p1);
    REQUIRE(backend_graph != nullptr);
    auto topology_backend = analyze_graph(backend_graph, "P1 inner backend graph (child of the frontend graph)");
    print_topology(topology_backend, true);

    // ---- P2: caller-owned parent embedding P1's frontend graph --------------
    cudaGraph_t parent_graph          = nullptr;
    cudaGraphNode_t parent_child_node = nullptr;
    CUDA_CHECK(cudaGraphCreate(&parent_graph, 0));
    CUDA_CHECK(cudaGraphAddChildGraphNode(&parent_child_node, parent_graph, nullptr, 0, graph_p1));
    auto topology_p2 = analyze_graph(parent_graph, "P2 caller parent child graph embedding P1's frontend graph");
    print_topology(topology_p2, true);

    // ---- relative comparison only, no hardcoded backend numbers -------------
    ev("[186][TOPO] RELATIVE P2 minus P1: top_level_nodes_delta=" +
       std::to_string(static_cast<long long>(topology_p2.top_level_nodes) -
                      static_cast<long long>(topology_p1.top_level_nodes)) +
       " recursive_nodes_delta=" +
       std::to_string(static_cast<long long>(topology_p2.total_nodes) -
                      static_cast<long long>(topology_p1.total_nodes)) +
       " kernel_nodes_delta=" +
       std::to_string(static_cast<long long>(topology_p2.kernels) - static_cast<long long>(topology_p1.kernels)) +
       " child_depth_delta=" + std::to_string(topology_p2.max_child_depth - topology_p1.max_child_depth));

    REQUIRE(topology_p2.kernels == topology_p1.kernels);
    REQUIRE(topology_p2.max_child_depth == topology_p1.max_child_depth + 1);
    REQUIRE(topology_p2.total_nodes == topology_p1.total_nodes + 1);

    ev("[186][ELIGIBILITY] matmul+add P1 memcpy_nodes=" + std::to_string(topology_p1.memcpys) + " memset_nodes=" +
       std::to_string(topology_p1.memsets) + " child_graph_nodes=" + std::to_string(topology_p1.child_graphs) +
       " direct_population_eligible=" + std::string(direct_population_eligible(graph) ? "true" : "false") +
       " total_workspace_bytes=" + std::to_string(workspace_bytes) +
       " direct_population_supported=" + std::string(direct_population_supported() ? "true" : "false"));
    REQUIRE(topology_p1.memcpys == 0);
    REQUIRE(topology_p1.memsets == 0);
    REQUIRE(topology_p1.child_graphs == 1);
    REQUIRE(direct_population_eligible(graph) == direct_population_supported());
    CUDA_CHECK(cudaGraphDestroy(parent_graph));
    CUDA_CHECK(cudaGraphDestroy(graph_p1));
#endif
}

// ===========================================================================
// P0 / P1 / P2 / P3 numerical equivalence
// ===========================================================================

TEST_CASE("186 P0 P1 P2 P3 produce identical results for one plan", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    log_environment("P0/P1/P2/P3 numerical equivalence (matmul+add)");

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 64, n = 32, k = 16;
    float const scale = 0.5f;
    half const a_fill = __float2half(2.0f);
    half const b_fill = __float2half(1.0f);
    half const bias   = __float2half(3.0f);

    auto graph  = make_matmul_add_graph(b, m, n, k, scale);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }
    print_plan_identity(graph, "P0/P1/P2/P3 matmul+add");

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture fixture(b, m, n, k, scale, a_fill, b_fill, bias, workspace_bytes, __float2half(0.f));
    auto variant_pack   = fixture.variant_pack();
    half const expected = fixture.expected();
    ev("[186][DATA] expected_value=" + fmt(__half2float(expected), 6) + " k=" + std::to_string(k) +
       " scale=" + fmt(static_cast<double>(scale), 4));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ---- P0: plain execution -------------------------------------------------
    fixture.fill_output(__float2half(0.f));
    REQUIRE(graph->execute(handle, variant_pack, fixture.workspace_ptr()).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(fixture.output_all_equal(expected));
    ev("[186][PATH] P0 plain execute -> output correct");

    // ---- P1: frontend populate -> instantiate -> launch ----------------------
    fixture.fill_output(__float2half(0.f));
    cudaGraph_t graph_p1 = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_p1, 0));
    REQUIRE(graph->populate_cuda_graph(handle, variant_pack, fixture.workspace_ptr(), graph_p1).is_good());
    cudaGraphExec_t exec_p1 = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_p1, graph_p1, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec_p1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    REQUIRE(fixture.output_all_equal(expected));
    ev("[186][PATH] P1 wrapped populate/instantiate/launch -> output correct");

    // ---- P2: P1's frontend graph inside a caller-owned parent ----------------
    fixture.fill_output(__float2half(0.f));
    cudaGraph_t parent_graph          = nullptr;
    cudaGraphNode_t parent_child_node = nullptr;
    CUDA_CHECK(cudaGraphCreate(&parent_graph, 0));
    CUDA_CHECK(cudaGraphAddChildGraphNode(&parent_child_node, parent_graph, nullptr, 0, graph_p1));
    cudaGraphExec_t exec_p2 = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_p2, parent_graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec_p2, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    REQUIRE(fixture.output_all_equal(expected));
    ev("[186][PATH] P2 caller parent child wrapping -> output correct");

    // ---- P3: direct population into an empty caller graph --------------------
    fixture.fill_output(__float2half(0.f));
    cudaGraph_t graph_p3 = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_p3, 0));
    auto direct_status = populate_graph(graph, handle, variant_pack, fixture.workspace_ptr(), graph_p3);
    ev("[186][PATH] P3 populate_cuda_graph_direct status=" + std::string(direct_status.is_good() ? "OK" : "ERROR") +
       " message=" + direct_status.get_message());
    REQUIRE(direct_status.is_good());
    cudaGraphExec_t exec_p3 = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_p3, graph_p3, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec_p3, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    REQUIRE(fixture.output_all_equal(expected));
    ev("[186][PATH] P3 direct populate/instantiate/launch -> output correct");

    auto topology_p3 = analyze_graph(graph_p3, "P3 direct populate (caller graph = backend graph)");
    print_topology(topology_p3, true);
    ev("[186][TOPO] RELATIVE P3 vs P1: top_level_nodes_p1=" + std::to_string(count_nodes(graph_p1)) +
       " top_level_nodes_p3=" + std::to_string(topology_p3.top_level_nodes) + " p3_child_graph_nodes=" +
       std::to_string(topology_p3.child_graphs) + " p3_kernel_nodes=" + std::to_string(topology_p3.kernels) +
       " p3_max_child_depth=" + std::to_string(topology_p3.max_child_depth));

    CUDA_CHECK(cudaGraphExecDestroy(exec_p3));
    CUDA_CHECK(cudaGraphDestroy(graph_p3));
    CUDA_CHECK(cudaGraphExecDestroy(exec_p2));
    CUDA_CHECK(cudaGraphDestroy(parent_graph));
    CUDA_CHECK(cudaGraphExecDestroy(exec_p1));
    CUDA_CHECK(cudaGraphDestroy(graph_p1));
    CUDA_CHECK(cudaStreamDestroy(stream));
#endif
}

// ===========================================================================
// G01 .. G05, G10
// ===========================================================================

TEST_CASE("186 G01: direct populate and replay without frontend auxiliary nodes", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }

    ev_header("G01 direct populate / replay without frontend auxiliary nodes");
    REQUIRE(direct_population_eligible(graph) == direct_population_supported());

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture fixture(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, __float2half(0.f));
    auto variant_pack = fixture.variant_pack();

    cudaGraph_t graph_direct = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_direct, 0));
    REQUIRE(populate_graph(graph, handle, variant_pack, fixture.workspace_ptr(), graph_direct).is_good());
    auto direct_topology = analyze_graph(graph_direct, "G01 direct backend graph");
    print_topology(direct_topology, true);

    cudaGraph_t graph_wrapped = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_wrapped, 0));
    REQUIRE(graph->populate_cuda_graph(handle, variant_pack, fixture.workspace_ptr(), graph_wrapped).is_good());
    auto wrapped_topology = analyze_graph(graph_wrapped, "G01 wrapped frontend graph");
    print_topology(wrapped_topology, true);

    cudaGraph_t inner_backend = first_child_graph(graph_wrapped);
    REQUIRE(inner_backend != nullptr);
    auto inner_topology = analyze_graph(inner_backend, "G01 wrapped inner backend graph");
    print_topology(inner_topology, false);

    ev("[186][G01] direct_top_level_nodes=" + std::to_string(direct_topology.top_level_nodes) +
       " wrapped_top_level_nodes=" + std::to_string(wrapped_topology.top_level_nodes) +
       " direct_kernel_nodes=" + std::to_string(direct_topology.kernels) +
       " wrapped_inner_kernel_nodes=" + std::to_string(inner_topology.kernels) +
       " wrapped_child_graph_nodes=" + std::to_string(wrapped_topology.child_graphs));
    REQUIRE(direct_topology.kernels == inner_topology.kernels);
    REQUIRE(direct_topology.child_graphs == (direct_population_supported() ? 0u : 1u));
    REQUIRE(wrapped_topology.child_graphs == 1);

    cudaGraphExec_t exec_direct = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_direct, graph_direct, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay) {
        fixture.fill_output(__float2half(0.f));
        CUDA_CHECK(cudaGraphLaunch(exec_direct, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(fixture.output_all_equal(fixture.expected()));
    }
    ev("[186][G01] PASS 3 replays of the direct graph produced the expected values");

    CUDA_CHECK(cudaGraphExecDestroy(exec_direct));
    CUDA_CHECK(cudaGraphDestroy(graph_direct));
    CUDA_CHECK(cudaGraphDestroy(graph_wrapped));
#endif
}

TEST_CASE("186 G02: address-only change, new output written and old buffer untouched", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }
    ev_header("G02 address-only update");

    half const sentinel = __float2half(-1234.0f);

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture first(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, __float2half(0.f));
    MatmulAddFixture second(
        b, m, n, k, 0.5f, __float2half(2.f), __float2half(1.f), __float2half(3.f), workspace_bytes, sentinel);

    cudaGraph_t graph_direct = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_direct, 0));
    REQUIRE(populate_graph(graph, handle, first.variant_pack(), first.workspace_ptr(), graph_direct).is_good());
    cudaGraphExec_t exec_direct = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_direct, graph_direct, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphLaunch(exec_direct, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(first.output_all_equal(first.expected()));
    ev("[186][G02] first output correct expected=" + fmt(__half2float(first.expected()), 4));

    REQUIRE(second.output_all_equal(sentinel));
    REQUIRE(update_graph(graph, handle, second.variant_pack(), second.workspace_ptr(), graph_direct).is_good());
    cudaGraphExecUpdateResultInfo update_info{};
    cudaError_t exec_update = cudaGraphExecUpdate(exec_direct, graph_direct, &update_info);
    ev("[186][G02] cudaGraphExecUpdate cuda_error=" + std::to_string(static_cast<int>(exec_update)) +
       " update_result=" + std::to_string(static_cast<int>(update_info.result)));
    if (exec_update != cudaSuccess) {
        CUDA_CHECK(cudaGraphExecDestroy(exec_direct));
        CUDA_CHECK(cudaGraphInstantiate(&exec_direct, graph_direct, nullptr, nullptr, 0));
        ev("[186][G02] re-instantiated after a failed exec update");
    }
    CUDA_CHECK(cudaGraphLaunch(exec_direct, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    REQUIRE(second.output_all_equal(second.expected()));
    ev("[186][G02] NEW output written expected=" + fmt(__half2float(second.expected()), 4) +
       " (buffer held the sentinel before the update)");
    REQUIRE(first.output_all_equal(first.expected()));
    ev("[186][G02] OLD output buffer still holds its previous value -> not clobbered");

    CUDA_CHECK(cudaGraphExecDestroy(exec_direct));
    CUDA_CHECK(cudaGraphDestroy(graph_direct));
#endif
}

TEST_CASE("186 G03: workspace address rebinding with an unchanged plan", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }
    ev_header("G03 workspace address rebinding");

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture fixture(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, __float2half(0.f));
    Surface<int8_t> workspace_two(static_cast<size_t>(std::max<int64_t>(workspace_bytes + 4096, 4096)));

    ev("[186][G03] workspace_size_bytes=" + std::to_string(workspace_bytes) + " workspace_a=0x" +
       std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(fixture.workspace_ptr()))) +
       " workspace_b=0x" +
       std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(workspace_two.devPtr))));

    cudaGraph_t graph_direct = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_direct, 0));
    REQUIRE(populate_graph(graph, handle, fixture.variant_pack(), fixture.workspace_ptr(), graph_direct).is_good());
    cudaGraphExec_t exec_direct = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_direct, graph_direct, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec_direct, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(fixture.output_all_equal(fixture.expected()));

    fixture.fill_output(__float2half(0.f));
    REQUIRE(update_graph(graph, handle, fixture.variant_pack(), workspace_two.devPtr, graph_direct).is_good());
    cudaGraphExecUpdateResultInfo update_info{};
    cudaError_t exec_update = cudaGraphExecUpdate(exec_direct, graph_direct, &update_info);
    ev("[186][G03] cudaGraphExecUpdate cuda_error=" + std::to_string(static_cast<int>(exec_update)));
    if (exec_update != cudaSuccess) {
        CUDA_CHECK(cudaGraphExecDestroy(exec_direct));
        CUDA_CHECK(cudaGraphInstantiate(&exec_direct, graph_direct, nullptr, nullptr, 0));
    }
    CUDA_CHECK(cudaGraphLaunch(exec_direct, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(fixture.output_all_equal(fixture.expected()));
    ev("[186][G03] PASS rebinding the workspace to a different allocation kept the result correct");

    CUDA_CHECK(cudaGraphExecDestroy(exec_direct));
    CUDA_CHECK(cudaGraphDestroy(graph_direct));
#endif
}

TEST_CASE("186 G04: clone then update with the original execution plan", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }
    ev_header("G04 clone then update");

    half const sentinel = __float2half(-1234.0f);

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture first(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, __float2half(0.f));
    MatmulAddFixture second(
        b, m, n, k, 0.5f, __float2half(2.f), __float2half(1.f), __float2half(3.f), workspace_bytes, sentinel);

    // ---- wrapped path clone --------------------------------------------------
    cudaGraph_t graph_wrapped = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_wrapped, 0));
    REQUIRE(graph->populate_cuda_graph(handle, first.variant_pack(), first.workspace_ptr(), graph_wrapped).is_good());
    cudaGraphExec_t exec_wrapped = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_wrapped, graph_wrapped, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec_wrapped, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(first.output_all_equal(first.expected()));
    ev("[186][G04] original wrapped graph executed with the first variant pack");

    cudaGraph_t graph_clone = nullptr;
    CUDA_CHECK(cudaGraphClone(&graph_clone, graph_wrapped));
    auto clone_topology = analyze_graph(graph_clone, "G04 clone of the wrapped graph");
    auto orig_topology  = analyze_graph(graph_wrapped, "G04 original wrapped graph");
    ev("[186][G04] wrapped clone top_level_nodes=" + std::to_string(clone_topology.top_level_nodes) +
       " original top_level_nodes=" + std::to_string(orig_topology.top_level_nodes) +
       " clone recursive_nodes=" + std::to_string(clone_topology.total_nodes) +
       " original recursive_nodes=" + std::to_string(orig_topology.total_nodes));
    REQUIRE(clone_topology.total_nodes == orig_topology.total_nodes);

    REQUIRE(graph->update_cuda_graph(handle, second.variant_pack(), second.workspace_ptr(), graph_clone).is_good());
    cudaGraphExec_t exec_clone = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_clone, graph_clone, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec_clone, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(second.output_all_equal(second.expected()));
    REQUIRE(first.output_all_equal(first.expected()));
    ev("[186][G04] PASS wrapped clone updated with the original plan produced correct output");

    // ---- direct path clone ---------------------------------------------------
    MatmulAddFixture third(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, sentinel);
    cudaGraph_t graph_direct = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_direct, 0));
    REQUIRE(populate_graph(graph, handle, third.variant_pack(), third.workspace_ptr(), graph_direct).is_good());
    cudaGraph_t direct_clone = nullptr;
    CUDA_CHECK(cudaGraphClone(&direct_clone, graph_direct));
    REQUIRE(update_graph(graph, handle, second.variant_pack(), second.workspace_ptr(), direct_clone).is_good());
    cudaGraphExec_t exec_direct_clone = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_direct_clone, direct_clone, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec_direct_clone, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(second.output_all_equal(second.expected()));
    ev("[186][G04] PASS direct-path clone updated with the original plan produced correct output");

    CUDA_CHECK(cudaGraphExecDestroy(exec_direct_clone));
    CUDA_CHECK(cudaGraphDestroy(direct_clone));
    CUDA_CHECK(cudaGraphDestroy(graph_direct));
    CUDA_CHECK(cudaGraphExecDestroy(exec_clone));
    CUDA_CHECK(cudaGraphDestroy(graph_clone));
    CUDA_CHECK(cudaGraphExecDestroy(exec_wrapped));
    CUDA_CHECK(cudaGraphDestroy(graph_wrapped));
#endif
}

TEST_CASE("186 G05: caller parent child exec update propagates to the instantiated parent", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }
    ev_header("G05 caller parent child exec update");

    half const sentinel = __float2half(-1234.0f);

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture first(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, __float2half(0.f));
    MatmulAddFixture second(
        b, m, n, k, 0.5f, __float2half(2.f), __float2half(1.f), __float2half(3.f), workspace_bytes, sentinel);

    cudaGraph_t graph_p1 = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_p1, 0));
    REQUIRE(graph->populate_cuda_graph(handle, first.variant_pack(), first.workspace_ptr(), graph_p1).is_good());

    cudaGraph_t parent_graph          = nullptr;
    cudaGraphNode_t parent_child_node = nullptr;
    CUDA_CHECK(cudaGraphCreate(&parent_graph, 0));
    CUDA_CHECK(cudaGraphAddChildGraphNode(&parent_child_node, parent_graph, nullptr, 0, graph_p1));
    cudaGraphExec_t exec_parent = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_parent, parent_graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec_parent, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(first.output_all_equal(first.expected()));
    ev("[186][G05] parent graph executed with the first variant pack");

    cudaGraph_t child_inside_parent = nullptr;
    CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(parent_child_node, &child_inside_parent));
    REQUIRE(
        graph->update_cuda_graph(handle, second.variant_pack(), second.workspace_ptr(), child_inside_parent).is_good());
    CUDA_CHECK(cudaGraphExecChildGraphNodeSetParams(exec_parent, parent_child_node, child_inside_parent));
    CUDA_CHECK(cudaGraphLaunch(exec_parent, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(second.output_all_equal(second.expected()));
    REQUIRE(first.output_all_equal(first.expected()));
    ev("[186][G05] PASS cudaGraphExecChildGraphNodeSetParams propagated the child update to the parent exec; new "
       "output correct and old buffer untouched");

    CUDA_CHECK(cudaGraphExecDestroy(exec_parent));
    CUDA_CHECK(cudaGraphDestroy(parent_graph));
    CUDA_CHECK(cudaGraphDestroy(graph_p1));
#endif
}

TEST_CASE("186 G10: non-default stream launch and event attribution", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }
    ev_header("G10 non-default stream");

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture fixture(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, __float2half(0.f));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ev("[186][G10] stream_handle=0x" +
       std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream))) + " is_non_default=true");

    cudaGraph_t graph_direct = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_direct, 0));
    REQUIRE(populate_graph(graph, handle, fixture.variant_pack(), fixture.workspace_ptr(), graph_direct).is_good());
    cudaGraphExec_t exec_direct = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_direct, graph_direct, nullptr, nullptr, 0));

    cudaEvent_t done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    CUDA_CHECK(cudaGraphLaunch(exec_direct, stream));
    CUDA_CHECK(cudaEventRecord(done, stream));
    cudaError_t query_before = cudaEventQuery(done);
    CUDA_CHECK(cudaEventSynchronize(done));
    REQUIRE(fixture.output_all_equal(fixture.expected()));
    ev("[186][G10] PASS output correct after syncing an event recorded on the non-default stream (query_immediately=" +
       std::string(query_before == cudaSuccess ? "ready" : "not_ready") + ")");

    fixture.fill_output(__float2half(0.f));
    REQUIRE(graph->execute(handle, fixture.variant_pack(), fixture.workspace_ptr()).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(fixture.output_all_equal(fixture.expected()));
    ev("[186][G10] PASS frontend execute on the default stream still agrees");

    CUDA_CHECK(cudaEventDestroy(done));
    CUDA_CHECK(cudaGraphExecDestroy(exec_direct));
    CUDA_CHECK(cudaGraphDestroy(graph_direct));
    CUDA_CHECK(cudaStreamDestroy(stream));
#endif
}

// ===========================================================================
// P4 / G06: frontend auxiliary node
// ===========================================================================

TEST_CASE("186 P4 G06: frontend auxiliary memcpy node ordering and behaviour", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    ev_header("P4 / G06 frontend auxiliary node (SDPA forward with alibi mask)");

    int64_t const b = 4, h = 4, s_q = 256, s_kv = 256, d = 64;

    auto graph  = make_sdpa_fwd_alibi_graph(b, h, s_q, s_kv, d, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    print_plan_identity(graph, "P4 SDPA forward + alibi");
    ev("[186][P4] prepare_native_plan built=" + std::string(status.built ? "true" : "false") +
       " native_api=" + std::string(status.native_api ? "true" : "false") + " reason=" + status.reason);
    if (!status.built) {
        SKIP("SDPA forward graph with alibi mask could not be built: " + status.reason);
        return;
    }

    int64_t const workspace_bytes = workspace_size_of(graph);
    SdpaFwdFixture fixture(b, h, s_q, s_kv, d, workspace_bytes);

    ev("[186][ELIGIBILITY] sdpa_fwd_alibi total_workspace_bytes=" + std::to_string(workspace_bytes) +
       " direct_population_eligible=" + std::string(direct_population_eligible(graph) ? "true" : "false"));
    REQUIRE(!direct_population_eligible(graph));

    // The direct entry points must refuse an ineligible plan instead of silently
    // dropping the frontend auxiliary node. On a baseline build there is no
    // direct entry point at all, so there is nothing to refuse.
    if (direct_population_supported()) {
        cudaGraph_t refused = nullptr;
        CUDA_CHECK(cudaGraphCreate(&refused, 0));
        auto refused_status = populate_graph(graph, handle, fixture.variant_pack(), fixture.workspace_ptr(), refused);
        ev("[186][P4] direct populate on an ineligible plan code=" +
           std::to_string(static_cast<int>(refused_status.get_code())) + " message=" + refused_status.get_message());
        REQUIRE(refused_status.is_bad());
        REQUIRE(refused_status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(count_nodes(refused) == 0);
        CUDA_CHECK(cudaGraphDestroy(refused));
    } else {
        ev("[186][P4] direct populate refusal not applicable: this tree has no direct-population entry point");
    }

    if (!status.native_api) {
        ev("[186][P4] SKIP-REASON plan built but carries no SUPPORTS_CUDA_GRAPH_NATIVE_API note for the alibi "
           "variant, so the native contract cannot be exercised");
        SKIP("SDPA forward + alibi plan has no native CUDA graph note on this device: " + status.reason);
        return;
    }

    cudaGraph_t graph_p4 = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_p4, 0));
    auto populate_status =
        graph->populate_cuda_graph(handle, fixture.variant_pack(), fixture.workspace_ptr(), graph_p4);
    ev("[186][P4] populate_cuda_graph status=" + std::string(populate_status.is_good() ? "OK" : "ERROR") +
       " message=" + populate_status.get_message());
    REQUIRE(populate_status.is_good());

    auto topology_p4 = analyze_graph(graph_p4, "P4 frontend graph with an auxiliary memcpy node");
    print_topology(topology_p4, true);

    auto memcpy_nodes = top_level_nodes_of_type(graph_p4, cudaGraphNodeTypeMemcpy);
    auto graph_nodes  = top_level_nodes_of_type(graph_p4, cudaGraphNodeTypeGraph);
    auto memset_nodes = top_level_nodes_of_type(graph_p4, cudaGraphNodeTypeMemset);
    ev("[186][G06] frontend_top_level_memcpy_nodes=" + std::to_string(memcpy_nodes.size()) +
       " frontend_top_level_memset_nodes=" + std::to_string(memset_nodes.size()) +
       " frontend_top_level_child_graph_nodes=" + std::to_string(graph_nodes.size()));
    REQUIRE(!memcpy_nodes.empty());
    REQUIRE(!graph_nodes.empty());

    // The auxiliary memcpy must run before the backend graph: the backend child
    // graph node has to depend on it.
    bool ordering_ok = false;
    for (auto memcpy_node : memcpy_nodes) {
        size_t dependent_count = 0;
        REQUIRE(cudnn_frontend::detail::cuda_graph_node_get_dependent_nodes(memcpy_node, nullptr, &dependent_count) ==
                cudaSuccess);
        std::vector<cudaGraphNode_t> dependent_nodes(dependent_count);
        if (dependent_count > 0) {
            REQUIRE(cudnn_frontend::detail::cuda_graph_node_get_dependent_nodes(
                        memcpy_node, dependent_nodes.data(), &dependent_count) == cudaSuccess);
            for (auto candidate : dependent_nodes) {
                for (auto graph_node : graph_nodes) {
                    if (candidate == graph_node) {
                        ordering_ok = true;
                    }
                }
            }
        }
    }
    ev("[186][G06] aux memcpy is a dependency of the backend child graph node: " +
       std::string(ordering_ok ? "true" : "false"));
    REQUIRE(ordering_ok);

    // Read the auxiliary memcpy parameters so the destination content can be
    // checked after a replay. cudaGraphMemcpyNodeGetParams hands back the host
    // source pointer, which the frontend keeps alive in its own cache: the
    // object lifetime contract is unchanged and the host buffer must outlive the
    // graph, exactly as before.
    cudaMemcpy3DParms memcpy_params{};
    REQUIRE(cudaGraphMemcpyNodeGetParams(memcpy_nodes[0], &memcpy_params) == cudaSuccess);
    size_t const copy_bytes = memcpy_params.extent.width;
    ev("[186][G06] aux memcpy bytes=" + std::to_string(copy_bytes) + " kind_is_host_to_device=" +
       std::string(memcpy_params.kind == cudaMemcpyHostToDevice ? "true" : "false") + " dst_inside_workspace=" +
       std::string((static_cast<char *>(memcpy_params.dstPtr.ptr) >= static_cast<char *>(fixture.workspace_ptr())) &&
                           (static_cast<char *>(memcpy_params.dstPtr.ptr) <
                            static_cast<char *>(fixture.workspace_ptr()) + workspace_bytes)
                       ? "true"
                       : "false"));

    cudaGraphExec_t exec_p4 = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_p4, graph_p4, nullptr, nullptr, 0));
    fixture.reset_output();
    REQUIRE(fixture.output_is_all_zero());
    CUDA_CHECK(cudaGraphLaunch(exec_p4, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> copied_back(copy_bytes);
    CUDA_CHECK(cudaMemcpy(copied_back.data(), memcpy_params.dstPtr.ptr, copy_bytes, cudaMemcpyDeviceToHost));
    bool workspace_matches_source =
        (copy_bytes > 0) && (std::memcmp(copied_back.data(), memcpy_params.srcPtr.ptr, copy_bytes) == 0);
    ev("[186][G06] aux memcpy destination content equals its host source after replay: " +
       std::string(workspace_matches_source ? "true" : "false"));
    REQUIRE(workspace_matches_source);

    auto output_from_graph = fixture.read_output();
    bool outputs_written   = false;
    for (auto element : output_from_graph) {
        if (__half2float(element) != 0.0f) {
            outputs_written = true;
        }
    }
    ev("[186][G06] SDPA output written by the graph replay: " + std::string(outputs_written ? "true" : "false"));
    REQUIRE(outputs_written);

    // P0 on the same plan must produce the same numbers as the wrapped graph.
    // The plan is run twice first so that run-to-run determinism is established
    // before any P0-versus-graph difference is interpreted.
    auto max_abs_diff = [](std::vector<half> const &lhs, std::vector<half> const &rhs) {
        float worst = 0.0f;
        if (lhs.size() != rhs.size()) {
            return -1.0f;
        }
        for (size_t i = 0; i < lhs.size(); ++i) {
            float const delta = std::fabs(__half2float(lhs[i]) - __half2float(rhs[i]));
            if (delta > worst) {
                worst = delta;
            }
        }
        return worst;
    };

    SdpaFwdFixture reference(b, h, s_q, s_kv, d, workspace_bytes);
    reference.reset_output();
    REQUIRE(graph->execute(handle, reference.variant_pack(), reference.workspace_ptr()).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto output_from_execute = reference.read_output();

    SdpaFwdFixture reference_two(b, h, s_q, s_kv, d, workspace_bytes);
    reference_two.reset_output();
    REQUIRE(graph->execute(handle, reference_two.variant_pack(), reference_two.workspace_ptr()).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto output_from_execute_two = reference_two.read_output();

    float const p0_self_diff = max_abs_diff(output_from_execute, output_from_execute_two);
    float const p0_vs_graph  = max_abs_diff(output_from_execute, output_from_graph);
    ev("[186][G06] P0 run-to-run max_abs_diff=" + fmt(static_cast<double>(p0_self_diff), 9) +
       " P0-vs-graph-replay max_abs_diff=" + fmt(static_cast<double>(p0_vs_graph), 9) + " reference_value=" +
       fmt(static_cast<double>(__half2float(output_from_execute.empty() ? __float2half(0.f) : output_from_execute[0])),
           6));
    REQUIRE(p0_self_diff == 0.0f);
    REQUIRE(p0_vs_graph == p0_self_diff);
    ev("[186][G06] P0 plain execute of the same plan matches the graph replay bitwise: true");

    CUDA_CHECK(cudaGraphExecDestroy(exec_p4));
    CUDA_CHECK(cudaGraphDestroy(graph_p4));
#endif
}

// ===========================================================================
// G07: failure paths
// ===========================================================================

TEST_CASE("186 G07: failure paths keep ownership and do not disturb the caller graph", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }
    ev_header("G07 failure paths");

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture fixture(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, __float2half(0.f));
    auto variant_pack = fixture.variant_pack();

    // (a) populate refuses a non-empty caller graph and leaves it usable.
    cudaGraph_t non_empty      = nullptr;
    cudaGraphNode_t empty_node = nullptr;
    CUDA_CHECK(cudaGraphCreate(&non_empty, 0));
    CUDA_CHECK(cudaGraphAddEmptyNode(&empty_node, non_empty, nullptr, 0));
    auto populate_non_empty = graph->populate_cuda_graph(handle, variant_pack, fixture.workspace_ptr(), non_empty);
    ev("[186][G07a] populate into a non-empty graph code=" +
       std::to_string(static_cast<int>(populate_non_empty.get_code())) +
       " message=" + populate_non_empty.get_message());
    REQUIRE(populate_non_empty.is_bad());
    REQUIRE(populate_non_empty.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(count_nodes(non_empty) == 1);

    auto direct_non_empty = populate_graph(graph, handle, variant_pack, fixture.workspace_ptr(), non_empty);
    ev("[186][G07a] direct populate into a non-empty graph code=" +
       std::to_string(static_cast<int>(direct_non_empty.get_code())) + " message=" + direct_non_empty.get_message());
    REQUIRE(direct_non_empty.is_bad());
    REQUIRE(count_nodes(non_empty) == 1);

    // The caller's graph was neither destroyed nor damaged: it still instantiates
    // and launches.
    cudaGraphExec_t non_empty_exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&non_empty_exec, non_empty, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(non_empty_exec, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGraphExecDestroy(non_empty_exec));
    CUDA_CHECK(cudaGraphDestroy(non_empty));
    ev("[186][G07a] PASS caller graph still instantiated, launched and destroyed cleanly");

    // (b) update refuses a foreign graph (one root node that is not a frontend
    //     child graph node) instead of misinterpreting it.
    cudaGraph_t foreign_graph    = nullptr;
    cudaGraphNode_t foreign_node = nullptr;
    CUDA_CHECK(cudaGraphCreate(&foreign_graph, 0));
    CUDA_CHECK(cudaGraphAddEmptyNode(&foreign_node, foreign_graph, nullptr, 0));
    auto foreign_update = graph->update_cuda_graph(handle, variant_pack, fixture.workspace_ptr(), foreign_graph);
    ev("[186][G07b] update against a foreign graph code=" +
       std::to_string(static_cast<int>(foreign_update.get_code())) + " message=" + foreign_update.get_message());
    REQUIRE(foreign_update.is_bad());
    REQUIRE(count_nodes(foreign_graph) == 1);
    CUDA_CHECK(cudaGraphDestroy(foreign_graph));
    ev("[186][G07b] PASS foreign graph rejected and left untouched");

    // (c) a variant pack missing a required uid is rejected.
    auto bad_variant_pack = variant_pack;
    bad_variant_pack.erase(kUidA);
    cudaGraph_t bad_graph = nullptr;
    CUDA_CHECK(cudaGraphCreate(&bad_graph, 0));
    auto bad_status = graph->populate_cuda_graph(handle, bad_variant_pack, fixture.workspace_ptr(), bad_graph);
    ev("[186][G07c] populate with a missing uid code=" + std::to_string(static_cast<int>(bad_status.get_code())) +
       " message=" + bad_status.get_message());
    REQUIRE(bad_status.is_bad());
    REQUIRE(bad_status.get_code() == fe::error_code_t::INVALID_VARIANT_PACK);
    REQUIRE(count_nodes(bad_graph) == 0);

    auto bad_direct_status = populate_graph(graph, handle, bad_variant_pack, fixture.workspace_ptr(), bad_graph);
    ev("[186][G07c] direct populate with a missing uid code=" +
       std::to_string(static_cast<int>(bad_direct_status.get_code())) + " message=" + bad_direct_status.get_message());
    REQUIRE(bad_direct_status.is_bad());
    REQUIRE(bad_direct_status.get_code() == fe::error_code_t::INVALID_VARIANT_PACK);
    CUDA_CHECK(cudaGraphDestroy(bad_graph));
    ev("[186][G07c] PASS missing uid rejected by both populate entry points");

    // (d) repeated failures must not grow host or device memory.
    size_t const rss_before  = host_rss_bytes();
    auto const memory_before = device_memory();
    int const failure_iters  = 200;
    for (int i = 0; i < failure_iters; ++i) {
        cudaGraph_t scratch = nullptr;
        REQUIRE(cudaGraphCreate(&scratch, 0) == cudaSuccess);
        auto first  = graph->populate_cuda_graph(handle, bad_variant_pack, fixture.workspace_ptr(), scratch);
        auto second = populate_graph(graph, handle, bad_variant_pack, fixture.workspace_ptr(), scratch);
        REQUIRE(first.is_bad());
        REQUIRE(second.is_bad());
        REQUIRE(cudaGraphDestroy(scratch) == cudaSuccess);
    }
    size_t const rss_after  = host_rss_bytes();
    auto const memory_after = device_memory();
    ev("[186][G07d] failures=" + std::to_string(failure_iters) + " host_rss_delta_bytes=" +
       std::to_string(static_cast<long long>(rss_after) - static_cast<long long>(rss_before)) +
       " device_free_delta_bytes=" +
       std::to_string(static_cast<long long>(memory_after.first) - static_cast<long long>(memory_before.first)));
    REQUIRE(rss_after <= rss_before + 8u * 1024u * 1024u);

    // The frontend graph is still fully usable after all those failures.
    cudaGraph_t still_works = nullptr;
    CUDA_CHECK(cudaGraphCreate(&still_works, 0));
    REQUIRE(graph->populate_cuda_graph(handle, variant_pack, fixture.workspace_ptr(), still_works).is_good());
    cudaGraphExec_t still_works_exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&still_works_exec, still_works, nullptr, nullptr, 0));
    fixture.fill_output(__float2half(0.f));
    CUDA_CHECK(cudaGraphLaunch(still_works_exec, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(fixture.output_all_equal(fixture.expected()));
    CUDA_CHECK(cudaGraphExecDestroy(still_works_exec));
    CUDA_CHECK(cudaGraphDestroy(still_works));
    ev("[186][G07] PASS all failure paths, graph still usable afterwards");
#endif
}

// ===========================================================================
// G08: growing lifecycle loop
// ===========================================================================

TEST_CASE("186 G08: repeated create/update/instantiate/destroy with growing loop count", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 8, m = 32, n = 16, k = 8;
    auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }
    ev_header("G08 growing lifecycle loop");

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture first(
        b, m, n, k, 0.5f, __float2half(1.f), __float2half(1.f), __float2half(2.f), workspace_bytes, __float2half(0.f));
    MatmulAddFixture second(b,
                            m,
                            n,
                            k,
                            0.5f,
                            __float2half(2.f),
                            __float2half(1.f),
                            __float2half(3.f),
                            workspace_bytes,
                            __float2half(-1234.f));

    std::vector<int> loop_counts = {10, 50, 100, 200};
    for (auto loops : loop_counts) {
        // Warm the allocator / driver caches before taking the baseline.
        for (int i = 0; i < 5; ++i) {
            cudaGraph_t warm = nullptr;
            REQUIRE(cudaGraphCreate(&warm, 0) == cudaSuccess);
            REQUIRE(populate_graph(graph, handle, first.variant_pack(), first.workspace_ptr(), warm).is_good());
            cudaGraphExec_t warm_exec = nullptr;
            REQUIRE(cudaGraphInstantiate(&warm_exec, warm, nullptr, nullptr, 0) == cudaSuccess);
            REQUIRE(cudaGraphExecDestroy(warm_exec) == cudaSuccess);
            REQUIRE(cudaGraphDestroy(warm) == cudaSuccess);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        size_t const rss_before  = host_rss_bytes();
        auto const memory_before = device_memory();
        auto const start         = std::chrono::steady_clock::now();

        size_t checksum_failures = 0;
        for (int i = 0; i < loops; ++i) {
            cudaGraph_t round_graph = nullptr;
            REQUIRE(cudaGraphCreate(&round_graph, 0) == cudaSuccess);
            REQUIRE(populate_graph(graph, handle, first.variant_pack(), first.workspace_ptr(), round_graph).is_good());

            cudaGraphExec_t round_exec = nullptr;
            REQUIRE(cudaGraphInstantiate(&round_exec, round_graph, nullptr, nullptr, 0) == cudaSuccess);
            REQUIRE(cudaGraphLaunch(round_exec, 0) == cudaSuccess);

            REQUIRE(update_graph(graph, handle, second.variant_pack(), second.workspace_ptr(), round_graph).is_good());
            cudaGraphExecUpdateResultInfo update_info{};
            if (cudaGraphExecUpdate(round_exec, round_graph, &update_info) != cudaSuccess) {
                REQUIRE(cudaGraphExecDestroy(round_exec) == cudaSuccess);
                REQUIRE(cudaGraphInstantiate(&round_exec, round_graph, nullptr, nullptr, 0) == cudaSuccess);
            }
            REQUIRE(cudaGraphLaunch(round_exec, 0) == cudaSuccess);
            REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

            if (!second.output_all_equal(second.expected())) {
                ++checksum_failures;
            }

            REQUIRE(cudaGraphExecDestroy(round_exec) == cudaSuccess);
            REQUIRE(cudaGraphDestroy(round_graph) == cudaSuccess);
        }

        auto const stop         = std::chrono::steady_clock::now();
        size_t const rss_after  = host_rss_bytes();
        auto const memory_after = device_memory();
        double const total_ms   = std::chrono::duration<double, std::milli>(stop - start).count();

        ev("[186][G08] loops=" + std::to_string(loops) + " checksum_failures=" + std::to_string(checksum_failures) +
           " total_ms=" + fmt(total_ms) + " per_loop_us=" + fmt(total_ms * 1000.0 / static_cast<double>(loops)) +
           " host_rss_delta_bytes=" +
           std::to_string(static_cast<long long>(rss_after) - static_cast<long long>(rss_before)) +
           " device_free_delta_bytes=" +
           std::to_string(static_cast<long long>(memory_after.first) - static_cast<long long>(memory_before.first)));
        REQUIRE(checksum_failures == 0);
        // A leak that scales with the loop count would show up as device memory
        // that never comes back. Allow a small driver-level slack only.
        REQUIRE(static_cast<long long>(memory_after.first) >=
                static_cast<long long>(memory_before.first) - 32LL * 1024LL * 1024LL);
    }
    ev("[186][G08] PASS, see the per-loop-count deltas above (host and device)");
#endif
}

// ===========================================================================
// G09: two device isolation
// ===========================================================================

TEST_CASE("186 G09: two L20 device isolation", "[cudagraph_186][cudagraph_186_multidevice]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    log_environment("G09 two device isolation");
    ev("[186][G09] visible_device_count=" + std::to_string(device_count));
    if (device_count < 2) {
        SKIP("needs two visible CUDA devices; only " + std::to_string(device_count) + " visible");
        return;
    }

    int64_t const b = 8, m = 32, n = 16, k = 8;

    struct DeviceState {
        int index = 0;
        std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter> handle;
        cudaStream_t stream = nullptr;
        std::shared_ptr<fe::graph::Graph> graph;
        cudaGraph_t cuda_graph = nullptr;
        cudaGraphExec_t exec   = nullptr;
        std::vector<std::unique_ptr<MatmulAddFixture> > fixtures;
        half expected = __float2half(0.f);
    };

    std::vector<DeviceState> states(2);
    for (int device = 0; device < 2; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        states[device].index  = device;
        states[device].handle = create_cudnn_handle();

        auto graph  = make_matmul_add_graph(b, m, n, k, 0.5f);
        auto status = prepare_native_plan(graph, *states[device].handle);
        if (!status.built || !status.native_api) {
            SKIP("device " + std::to_string(device) + ": plan is not native CUDA graph capable: " + status.reason);
            CUDA_CHECK(cudaSetDevice(0));
            return;
        }
        states[device].graph = graph;

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        states[device].stream = stream;

        int64_t const workspace_bytes = workspace_size_of(graph);
        auto fixture                  = std::make_unique<MatmulAddFixture>(b,
                                                          m,
                                                          n,
                                                          k,
                                                          0.5f,
                                                          __float2half(1.f),
                                                          __float2half(1.f),
                                                          __float2half(2.f),
                                                          workspace_bytes,
                                                          __float2half(0.f));

        cudaGraph_t cuda_graph = nullptr;
        CUDA_CHECK(cudaGraphCreate(&cuda_graph, 0));
        REQUIRE(
            populate_graph(graph, *states[device].handle, fixture->variant_pack(), fixture->workspace_ptr(), cuda_graph)
                .is_good());
        states[device].cuda_graph = cuda_graph;

        cudaGraphExec_t exec = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&exec, cuda_graph, nullptr, nullptr, 0));
        states[device].exec = exec;

        cudaPointerAttributes attributes{};
        REQUIRE(cudaPointerGetAttributes(&attributes, fixture->output_ptr()) == cudaSuccess);
        ev("[186][G09] device=" + std::to_string(device) +
           " output_buffer_device=" + std::to_string(attributes.device) + " handle_ptr=0x" +
           std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(*states[device].handle))) +
           " stream_ptr=0x" + std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream))) +
           " graph_ptr=0x" + std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(cuda_graph))));
        REQUIRE(attributes.device == device);

        REQUIRE(cudaGraphLaunch(exec, stream) == cudaSuccess);
        // Keep the fixture alive for the verification below.
        states[device].expected = fixture->expected();
        states[device].fixtures.push_back(std::move(fixture));
    }

    // Launch on both devices before synchronising either one, then verify each
    // device only after synchronising its own stream.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamSynchronize(states[0].stream));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamSynchronize(states[1].stream));

    for (int device = 0; device < 2; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        auto &fixture = states[device].fixtures[0];
        REQUIRE(fixture->output_all_equal(states[device].expected));
        ev("[186][G09] device=" + std::to_string(device) + " output correct after its own stream was synchronised");
    }

    // Re-run device 1 and confirm device 0's buffer is untouched.
    CUDA_CHECK(cudaSetDevice(1));
    states[1].fixtures[0]->fill_output(__float2half(-1234.f));
    REQUIRE(states[1].fixtures[0]->output_all_equal(__float2half(-1234.f)));
    CUDA_CHECK(cudaGraphLaunch(states[1].exec, states[1].stream));
    CUDA_CHECK(cudaStreamSynchronize(states[1].stream));

    CUDA_CHECK(cudaSetDevice(0));
    REQUIRE(states[0].fixtures[0]->output_all_equal(states[0].expected));
    ev("[186][G09] PASS device 0 buffer unchanged while device 1 replayed; no cross-device reuse");

    for (int device = 0; device < 2; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaGraphExecDestroy(states[device].exec));
        CUDA_CHECK(cudaGraphDestroy(states[device].cuda_graph));
        CUDA_CHECK(cudaStreamDestroy(states[device].stream));
        states[device].fixtures.clear();
        states[device].graph.reset();
        states[device].handle.reset();
    }
    CUDA_CHECK(cudaSetDevice(0));
#endif
}

// ===========================================================================
// Timing: setup / update / replay, kept strictly separate
// ===========================================================================

TEST_CASE("186 timing: populate, instantiate, update, exec update, launch, replay", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    ev_header("timing (matmul+add 8x1024x1024x1024)");

    int64_t const b = 8, m = 1024, n = 1024, k = 1024;
    float const scale   = 0.5f;
    half const a_fill   = __float2half(1.0f);
    half const b_fill   = __float2half(1.0f);
    half const bias_a   = __float2half(2.0f);
    half const bias_b   = __float2half(3.0f);
    half const a_fill_b = __float2half(2.0f);

    auto graph  = make_matmul_add_graph(b, m, n, k, scale);
    auto status = prepare_native_plan(graph, handle);
    print_plan_identity(graph, "timing matmul+add");
    ev("[186][TIMING] setup native_api=" + std::string(status.native_api ? "true" : "false") +
       " direct_population_supported=" + std::string(direct_population_supported() ? "true" : "false") +
       " reason=" + status.reason);
    if (!status.built || !status.native_api) {
        SKIP("plan is not native CUDA graph capable: " + status.reason);
        return;
    }

    int64_t const workspace_bytes = workspace_size_of(graph);
    MatmulAddFixture variant_a(b, m, n, k, scale, a_fill, b_fill, bias_a, workspace_bytes, __float2half(0.f));
    MatmulAddFixture variant_b(b, m, n, k, scale, a_fill_b, b_fill, bias_b, workspace_bytes, __float2half(-1234.f));

    auto pack_a = variant_a.variant_pack();
    auto pack_b = variant_b.variant_pack();
    void *ws_a  = variant_a.workspace_ptr();
    void *ws_b  = variant_b.workspace_ptr();

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Graph::execute runs on the handle's stream, so bind the handle to the same
    // non-default stream the CUDA graph launches use, otherwise the event-based
    // P0 measurement would time an idle stream.
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    size_t const rss_start  = host_rss_bytes();
    auto const memory_start = device_memory();
    ev("[186][TIMING] shape=" + std::to_string(b) + "x" + std::to_string(m) + "x" + std::to_string(n) + "x" +
       std::to_string(k) + " dtype=half workspace_bytes=" + std::to_string(workspace_bytes) + " host_rss_start_bytes=" +
       std::to_string(rss_start) + " device_free_start_bytes=" + std::to_string(memory_start.first));

    constexpr int kPopulateReps    = 200;
    constexpr int kInstantiateReps = 100;
    constexpr int kUpdateReps      = 200;
    constexpr int kExecUpdateReps  = 100;
    constexpr int kLaunchReps      = 200;
    constexpr int kReplayReps      = 50;
    constexpr int kReplayOuter     = 20;
    constexpr int kReplayInner     = 50;
    constexpr int kExecuteReps     = 50;

    // ---- floor: an empty graph create + destroy cycle, measured repeatedly so
    // that the populate numbers below can be compared against a floor taken at
    // the same point in time.
    auto measure_empty_cycle = [&](std::string const &label) {
        auto cycle = time_host(kPopulateReps, 20, []() {
            cudaGraph_t scratch = nullptr;
            REQUIRE(cudaGraphCreate(&scratch, 0) == cudaSuccess);
            REQUIRE(cudaGraphDestroy(scratch) == cudaSuccess);
        });
        report("empty_graph_cycle", label, cycle, "cudaGraphCreate+cudaGraphDestroy");
        return cycle.median();
    };
    measure_empty_cycle("floor_before_populate");

    // ---- host populate ------------------------------------------------------
    double populate_p1_median_us = 0.0;
    double populate_p3_median_us = 0.0;
    {
        int failures = 0;
        auto wrapped = time_host(kPopulateReps, 20, [&]() {
            cudaGraph_t scratch = nullptr;
            if (cudaGraphCreate(&scratch, 0) != cudaSuccess) {
                ++failures;
                return;
            }
            auto st = graph->populate_cuda_graph(handle, pack_a, ws_a, scratch);
            if (st.is_bad()) {
                ++failures;
            }
            cudaGraphDestroy(scratch);
        });
        report("host_populate_cycle", "P1", wrapped, "cudaGraphCreate+populate_cuda_graph+cudaGraphDestroy");

        auto direct = time_host(kPopulateReps, 20, [&]() {
            cudaGraph_t scratch = nullptr;
            if (cudaGraphCreate(&scratch, 0) != cudaSuccess) {
                ++failures;
                return;
            }
            auto st = populate_graph(graph, handle, pack_a, ws_a, scratch);
            if (st.is_bad()) {
                ++failures;
            }
            cudaGraphDestroy(scratch);
        });
        report("host_populate_cycle",
               direct_path_label(),
               direct,
               "cudaGraphCreate+populate_cuda_graph_direct+cudaGraphDestroy");
        REQUIRE(failures == 0);
        populate_p1_median_us = wrapped.median();
        populate_p3_median_us = direct.median();
    }
    double const floor_after_us = measure_empty_cycle("floor_after_populate");
    // Both populate cycles below contain the same cudaGraphCreate + cudaGraphDestroy
    // pair, so subtracting the empty-graph floor measured in the same process at
    // nearly the same time isolates the populate call itself.
    ev("[186][TIMING] normalized phase=host_populate_minus_empty_graph_floor path=P1 delta_us=" +
       fmt(populate_p1_median_us - floor_after_us) + " floor_us=" + fmt(floor_after_us) +
       " raw_cycle_us=" + fmt(populate_p1_median_us));
    ev("[186][TIMING] normalized phase=host_populate_minus_empty_graph_floor path=" + direct_path_label() +
       " delta_us=" + fmt(populate_p3_median_us - floor_after_us) + " floor_us=" + fmt(floor_after_us) +
       " raw_cycle_us=" + fmt(populate_p3_median_us));

    // ---- persistent graphs for P1 / P2 / P3 ---------------------------------
    cudaGraph_t graph_p1 = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_p1, 0));
    REQUIRE(graph->populate_cuda_graph(handle, pack_a, ws_a, graph_p1).is_good());
    cudaGraphExec_t exec_p1 = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_p1, graph_p1, nullptr, nullptr, 0));
    auto p1_graph_nodes = top_level_nodes_of_type(graph_p1, cudaGraphNodeTypeGraph);
    REQUIRE(p1_graph_nodes.size() == 1);
    cudaGraph_t p1_child = nullptr;
    CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(p1_graph_nodes[0], &p1_child));

    cudaGraph_t parent_graph          = nullptr;
    cudaGraphNode_t parent_child_node = nullptr;
    CUDA_CHECK(cudaGraphCreate(&parent_graph, 0));
    CUDA_CHECK(cudaGraphAddChildGraphNode(&parent_child_node, parent_graph, nullptr, 0, graph_p1));
    cudaGraphExec_t exec_p2 = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_p2, parent_graph, nullptr, nullptr, 0));
    cudaGraph_t parent_child_graph = nullptr;
    CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(parent_child_node, &parent_child_graph));

    cudaGraph_t graph_p3 = nullptr;
    CUDA_CHECK(cudaGraphCreate(&graph_p3, 0));
    REQUIRE(populate_graph(graph, handle, pack_a, ws_a, graph_p3).is_good());
    cudaGraphExec_t exec_p3 = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec_p3, graph_p3, nullptr, nullptr, 0));

    ev("[186][TIMING] topology p1_top_level_nodes=" + std::to_string(count_nodes(graph_p1)) + " p3_top_level_nodes=" +
       std::to_string(count_nodes(graph_p3)) + " p2_top_level_nodes=" + std::to_string(count_nodes(parent_graph)));

    // ---- host instantiate ---------------------------------------------------
    {
        auto instantiate = [&](cudaGraph_t target, std::string const &path) {
            Distribution distribution;
            for (int i = 0; i < 5; ++i) {
                cudaGraphExec_t scratch = nullptr;
                REQUIRE(cudaGraphInstantiate(&scratch, target, nullptr, nullptr, 0) == cudaSuccess);
                REQUIRE(cudaGraphExecDestroy(scratch) == cudaSuccess);
            }
            for (int i = 0; i < kInstantiateReps; ++i) {
                cudaGraphExec_t scratch = nullptr;
                auto start              = std::chrono::steady_clock::now();
                REQUIRE(cudaGraphInstantiate(&scratch, target, nullptr, nullptr, 0) == cudaSuccess);
                auto stop = std::chrono::steady_clock::now();
                distribution.add(std::chrono::duration<double, std::micro>(stop - start).count());
                REQUIRE(cudaGraphExecDestroy(scratch) == cudaSuccess);
            }
            report("host_instantiate", path, distribution, "cudaGraphInstantiate");
        };
        instantiate(graph_p1, std::string("P1"));
        instantiate(graph_p3, direct_path_label());
        instantiate(parent_graph, std::string("P2"));
    }

    // ---- host update --------------------------------------------------------
    {
        int failures   = 0;
        auto p1_update = time_host(kUpdateReps, 20, [&]() {
            auto st = graph->update_cuda_graph(handle, pack_b, ws_b, graph_p1);
            if (st.is_bad()) {
                ++failures;
            }
        });
        report("host_update", "P1", p1_update, "Graph::update_cuda_graph on the frontend graph");

        auto p2_update = time_host(kUpdateReps, 20, [&]() {
            auto st = graph->update_cuda_graph(handle, pack_b, ws_b, parent_child_graph);
            if (st.is_bad()) {
                ++failures;
            }
        });
        report("host_update", "P2", p2_update, "Graph::update_cuda_graph on the embedded child graph");

        auto p3_update = time_host(kUpdateReps, 20, [&]() {
            auto st = update_graph(graph, handle, pack_b, ws_b, graph_p3);
            if (st.is_bad()) {
                ++failures;
            }
        });
        report("host_update", direct_path_label(), p3_update, "Graph::update_cuda_graph_direct on the backend graph");
        REQUIRE(failures == 0);
    }

    // ---- exec update --------------------------------------------------------
    {
        // P1: the documented mechanism for an instantiated graph that contains a
        // child graph node.
        int failures        = 0;
        auto p1_exec_update = time_host(kExecUpdateReps, 10, [&]() {
            if (cudaGraphExecChildGraphNodeSetParams(exec_p1, p1_graph_nodes[0], p1_child) != cudaSuccess) {
                ++failures;
            }
        });
        report("exec_update", "P1", p1_exec_update, "cudaGraphExecChildGraphNodeSetParams");

        auto p2_exec_update = time_host(kExecUpdateReps, 10, [&]() {
            if (cudaGraphExecChildGraphNodeSetParams(exec_p2, parent_child_node, parent_child_graph) != cudaSuccess) {
                ++failures;
            }
        });
        report("exec_update", "P2", p2_exec_update, "cudaGraphExecChildGraphNodeSetParams on the caller parent");

        cudaGraphExecUpdateResultInfo direct_update_info{};
        cudaError_t direct_probe = cudaGraphExecUpdate(exec_p3, graph_p3, &direct_update_info);
        ev("[186][TIMING] exec_update P3 probe cuda_error=" + std::to_string(static_cast<int>(direct_probe)) +
           " update_result=" + std::to_string(static_cast<int>(direct_update_info.result)));
        if (direct_probe == cudaSuccess) {
            auto p3_exec_update = time_host(kExecUpdateReps, 10, [&]() {
                cudaGraphExecUpdateResultInfo info{};
                if (cudaGraphExecUpdate(exec_p3, graph_p3, &info) != cudaSuccess) {
                    ++failures;
                }
            });
            report("exec_update", direct_path_label(), p3_exec_update, "cudaGraphExecUpdate on the backend graph");
        } else {
            ev("[186][TIMING] phase=exec_update path=P3 mechanism=cudaGraphExecUpdate status=UNMEASURED reason="
               "cudaGraphExecUpdate returned " +
               std::to_string(static_cast<int>(direct_probe)));
            CUDA_CHECK(cudaGraphExecDestroy(exec_p3));
            CUDA_CHECK(cudaGraphInstantiate(&exec_p3, graph_p3, nullptr, nullptr, 0));
        }

        // P1 through cudaGraphExecUpdate, for reference: report whatever it does
        // rather than assuming it works.
        cudaGraphExecUpdateResultInfo p1_update_info{};
        cudaError_t p1_probe = cudaGraphExecUpdate(exec_p1, graph_p1, &p1_update_info);
        ev("[186][TIMING] exec_update P1 via cudaGraphExecUpdate probe cuda_error=" +
           std::to_string(static_cast<int>(p1_probe)) +
           " update_result=" + std::to_string(static_cast<int>(p1_update_info.result)));

        // Re-instantiate as the fallback mechanism, for all three paths.
        auto p1_reinst = time_host(kExecUpdateReps, 10, [&]() {
            cudaGraphExec_t scratch = nullptr;
            if (cudaGraphInstantiate(&scratch, graph_p1, nullptr, nullptr, 0) != cudaSuccess) {
                ++failures;
                return;
            }
            cudaGraphExecDestroy(scratch);
        });
        report("exec_update_via_reinstantiate", "P1", p1_reinst, "cudaGraphInstantiate+cudaGraphExecDestroy");

        auto p3_reinst = time_host(kExecUpdateReps, 10, [&]() {
            cudaGraphExec_t scratch = nullptr;
            if (cudaGraphInstantiate(&scratch, graph_p3, nullptr, nullptr, 0) != cudaSuccess) {
                ++failures;
                return;
            }
            cudaGraphExecDestroy(scratch);
        });
        report("exec_update_via_reinstantiate",
               direct_path_label(),
               p3_reinst,
               "cudaGraphInstantiate+cudaGraphExecDestroy");
        REQUIRE(failures == 0);
    }

    // ---- correctness after all the updates ----------------------------------
    CUDA_CHECK(cudaGraphLaunch(exec_p1, stream));
    CUDA_CHECK(cudaGraphLaunch(exec_p2, stream));
    CUDA_CHECK(cudaGraphLaunch(exec_p3, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    REQUIRE(variant_b.output_all_equal(variant_b.expected()));
    ev("[186][TIMING] PASS P1/P2/P3 all still produce the updated result");

    // ---- CPU launch submission ---------------------------------------------
    // Measured in an ABBA order so that a slowly drifting machine cannot be
    // mistaken for a difference between the wrapped and the direct graph.
    {
        Distribution p1_blocks;
        Distribution p2_blocks;
        Distribution p3_blocks;
        double p1_best_block_us = 1e30;
        double p2_best_block_us = 1e30;
        double p3_best_block_us = 1e30;
        auto submit = [&](cudaGraphExec_t target, std::string const &path, Distribution &sink, double &best_block) {
            auto block = time_host(kLaunchReps, 20, [&]() { REQUIRE(cudaGraphLaunch(target, stream) == cudaSuccess); });
            report("cpu_launch_submission", path, block, "cudaGraphLaunch without synchronisation");
            if (block.median() < best_block) {
                best_block = block.median();
            }
            for (auto sample : block.samples_us) {
                sink.add(sample);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        };
        submit(exec_p1, "P1_blockA", p1_blocks, p1_best_block_us);
        submit(exec_p3, direct_path_label() + "_blockA", p3_blocks, p3_best_block_us);
        submit(exec_p2, "P2_blockA", p2_blocks, p2_best_block_us);
        submit(exec_p2, "P2_blockB", p2_blocks, p2_best_block_us);
        submit(exec_p3, direct_path_label() + "_blockB", p3_blocks, p3_best_block_us);
        submit(exec_p1, "P1_blockB", p1_blocks, p1_best_block_us);
        report("cpu_launch_submission_pooled", "P1", p1_blocks, "cudaGraphLaunch without synchronisation");
        report("cpu_launch_submission_pooled", "P2", p2_blocks, "cudaGraphLaunch without synchronisation");
        report(
            "cpu_launch_submission_pooled", direct_path_label(), p3_blocks, "cudaGraphLaunch without synchronisation");
        // The quietest of the two blocks is the least contaminated by other
        // tenants on this shared host, so it is reported separately.
        ev("[186][TIMING] launch_submission_quietest_block path=P1 median_us=" + fmt(p1_best_block_us));
        ev("[186][TIMING] launch_submission_quietest_block path=P2 median_us=" + fmt(p2_best_block_us));
        ev("[186][TIMING] launch_submission_quietest_block path=" + direct_path_label() +
           " median_us=" + fmt(p3_best_block_us));
    }

    // ---- GPU replay ---------------------------------------------------------
    {
        auto p1_single = time_device_event(exec_p1, stream, kReplayReps, 10);
        report("gpu_replay_single", "P1", p1_single, "events around one cudaGraphLaunch");
        auto p2_single = time_device_event(exec_p2, stream, kReplayReps, 10);
        report("gpu_replay_single", "P2", p2_single, "events around one cudaGraphLaunch");
        auto p3_single = time_device_event(exec_p3, stream, kReplayReps, 10);
        report("gpu_replay_single", direct_path_label(), p3_single, "events around one cudaGraphLaunch");

        auto p1_batched = time_device_batched(exec_p1, stream, kReplayOuter, kReplayInner, 10);
        report("gpu_replay_batched",
               "P1",
               p1_batched,
               "events around " + std::to_string(kReplayInner) + " launches, divided");
        auto p2_batched = time_device_batched(exec_p2, stream, kReplayOuter, kReplayInner, 10);
        report("gpu_replay_batched",
               "P2",
               p2_batched,
               "events around " + std::to_string(kReplayInner) + " launches, divided");
        auto p3_batched = time_device_batched(exec_p3, stream, kReplayOuter, kReplayInner, 10);
        report("gpu_replay_batched",
               direct_path_label(),
               p3_batched,
               "events around " + std::to_string(kReplayInner) + " launches, divided");
    }

    // ---- P0 plain execution -------------------------------------------------
    {
        auto p0_submit = time_host(kExecuteReps, 10, [&]() {
            auto st = graph->execute(handle, pack_b, ws_b);
            REQUIRE(st.is_good());
        });
        report("cpu_launch_submission", "P0", p0_submit, "Graph::execute submission on the handle stream");
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto p0_gpu = time_device_execute(graph, handle, pack_b, ws_b, stream, kExecuteReps, 10);
        report("gpu_replay_single", "P0", p0_gpu, "events around one Graph::execute on the handle stream");
    }

    auto const memory_end = device_memory();
    ev("[186][TIMING] memory host_rss_start_bytes=" + std::to_string(rss_start) + " host_rss_end_bytes=" +
       std::to_string(host_rss_bytes()) + " device_free_start_bytes=" + std::to_string(memory_start.first) +
       " device_free_end_bytes=" + std::to_string(memory_end.first) +
       " device_total_bytes=" + std::to_string(memory_end.second) + " device_used_by_process_bytes=" +
       std::to_string(static_cast<long long>(memory_end.second) - static_cast<long long>(memory_end.first)));

    CUDA_CHECK(cudaGraphExecDestroy(exec_p3));
    CUDA_CHECK(cudaGraphDestroy(graph_p3));
    CUDA_CHECK(cudaGraphExecDestroy(exec_p2));
    CUDA_CHECK(cudaGraphDestroy(parent_graph));
    CUDA_CHECK(cudaGraphExecDestroy(exec_p1));
    CUDA_CHECK(cudaGraphDestroy(graph_p1));
    CUDA_CHECK(cudaStreamDestroy(stream));
#endif
}

// ===========================================================================
// Eligibility rule: frontend node-producing workspace operations, not
// workspace_size == 0
// ===========================================================================

TEST_CASE("186 eligibility comes from node-producing frontend workspace operations", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    ev_header("eligibility rule");

    // Case 1: matmul + add, pass-by-value bias tensor, no workspace
    // modification at all.
    {
        auto graph  = make_matmul_add_graph(8, 32, 16, 8, 0.5f);
        auto status = prepare_native_plan(graph, handle);
        if (!status.built) {
            ev("[186][ELIGIBILITY] case=matmul_add built=false reason=" + status.reason);
        } else {
            ev("[186][ELIGIBILITY] case=matmul_add total_workspace_bytes=" + std::to_string(workspace_size_of(graph)) +
               " frontend_node_producing_ops=0 eligible=" +
               std::string(direct_population_eligible(graph) ? "true" : "false"));
        }
    }

    // Case 2: SDPA backward without alibi. The frontend caches several
    // workspace modifications here (dQ_accum / dK / dV / softmax_sum), but on
    // cuDNN >= 9.6.0 they only rewrite variant pack pointers. The frontend
    // workspace is therefore non-zero while no frontend CUDA graph node is
    // produced, which is exactly why eligibility cannot be
    // "workspace_size == 0".
    {
        int64_t const b = 2, h = 2, s_q = 128, s_kv = 128, d = 64;
        auto graph  = create_sdpa_backward_graph(b,
                                                h,
                                                h,
                                                h,
                                                s_q,
                                                s_kv,
                                                d,
                                                d,
                                                1.0f,
                                                /*generate_stats=*/true,
                                                /*causal_mask=*/false,
                                                /*alibi_mask=*/false,
                                                /*padding_mask=*/false,
                                                /*has_attn_bias=*/false,
                                                /*is_deterministic=*/false);
        auto status = prepare_native_plan(graph, handle);
        if (!status.built) {
            ev("[186][ELIGIBILITY] case=sdpa_bwd built=false reason=" + status.reason);
        } else {
            int64_t const workspace_bytes = workspace_size_of(graph);
            SdpaBwdProbe probe(b, h, s_q, s_kv, d, workspace_bytes, handle, graph);
            ev("[186][ELIGIBILITY] case=sdpa_bwd total_workspace_bytes=" + std::to_string(workspace_bytes) +
               " eligible=" + std::string(direct_population_eligible(graph) ? "true" : "false") +
               " frontend_top_level_nodes=" + std::to_string(probe.top_level_nodes) + " frontend_memcpy_nodes=" +
               std::to_string(probe.memcpy_nodes) + " frontend_memset_nodes=" + std::to_string(probe.memset_nodes) +
               " frontend_child_graph_nodes=" + std::to_string(probe.child_graph_nodes) +
               " populate_status=" + (probe.populated ? std::string("OK") : probe.populate_message));
            if (probe.populated) {
                REQUIRE(probe.memcpy_nodes == 0);
                REQUIRE(probe.memset_nodes == 0);
                REQUIRE(probe.child_graph_nodes == 1);
                REQUIRE(direct_population_eligible(graph) == direct_population_supported());
            }
        }
    }

    // Case 3: SDPA forward with alibi. One node-producing workspace
    // modification (a memcpy of the alibi slopes) -> ineligible, and the
    // frontend graph carries that memcpy node.
    {
        int64_t const b = 4, h = 4, s_q = 256, s_kv = 256, d = 64;
        auto graph  = make_sdpa_fwd_alibi_graph(b, h, s_q, s_kv, d, 0.5f);
        auto status = prepare_native_plan(graph, handle);
        if (!status.built) {
            ev("[186][ELIGIBILITY] case=sdpa_fwd_alibi built=false reason=" + status.reason);
        } else {
            int64_t const workspace_bytes = workspace_size_of(graph);
            SdpaFwdFixture fixture(b, h, s_q, s_kv, d, workspace_bytes);
            cudaGraph_t populated = nullptr;
            CUDA_CHECK(cudaGraphCreate(&populated, 0));
            auto populate_status =
                graph->populate_cuda_graph(handle, fixture.variant_pack(), fixture.workspace_ptr(), populated);
            auto topology = analyze_graph(populated, "eligibility case 3");
            ev("[186][ELIGIBILITY] case=sdpa_fwd_alibi total_workspace_bytes=" + std::to_string(workspace_bytes) +
               " eligible=" + std::string(direct_population_eligible(graph) ? "true" : "false") +
               " frontend_top_level_nodes=" + std::to_string(topology.top_level_nodes) + " frontend_memcpy_nodes=" +
               std::to_string(topology.memcpys) + " frontend_memset_nodes=" + std::to_string(topology.memsets) +
               " frontend_child_graph_nodes=" + std::to_string(topology.child_graphs) +
               " populate_status=" + (populate_status.is_good() ? std::string("OK") : populate_status.get_message()));
            REQUIRE(populate_status.is_good());
            REQUIRE(!direct_population_eligible(graph));
            REQUIRE(topology.memcpys == 1);
            REQUIRE(topology.memsets == 0);
            CUDA_CHECK(cudaGraphDestroy(populated));
        }
    }
#endif
}

// ===========================================================================
// G11: the legacy Slice alias reaches every CUDA graph entry point
// ===========================================================================

TEST_CASE("186 G11: legacy Slice aliasing is materialised for every CUDA graph path", "[cudagraph_186]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    if (fe::detail::get_backend_version() >= 92200 && fe::detail::get_compiled_version() >= 92200) {
        SKIP("this frontend/backend pair handles Slice natively, so there is no replacement to materialise");
        return;
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 2, s = 64, d = 64, n = 8;
    auto graph  = make_slice_matmul_graph(b, s, d, n);
    auto status = prepare_native_plan(graph, handle);
    if (!status.built) {
        SKIP("slice plan could not be built: " + status.reason);
        return;
    }

    ev_header("G11 legacy Slice aliasing: the destination pointer is derived from the source");
    print_plan_identity(graph, "G11 slice+matmul");

    int64_t const workspace_bytes = workspace_size_of(graph);
    int64_t const z_elements      = b * n;
    half const sentinel           = __float2half(-1234.0f);
    // Filled with exact halves (all-ones / all-twos against an all-ones W), so
    // the expected output is the row sum itself and needs no tolerance.
    Surface<half> x(static_cast<size_t>(b * s * d), __float2half(1.f));
    Surface<half> x2(static_cast<size_t>(b * s * d), __float2half(2.f));
    Surface<half> w(static_cast<size_t>(b * d * n), __float2half(1.f));
    Surface<half> z(static_cast<size_t>(z_elements), sentinel);
    Surface<half> z2(static_cast<size_t>(z_elements), sentinel);
    Surface<int8_t> workspace(static_cast<size_t>(std::max<int64_t>(workspace_bytes, 1)));

    // slices = {{0, b}, {1, 2}, {0, d}} on strides {s*d, d, 1}: the source pointer
    // moves by exactly one d-row.
    int64_t const offset_bytes = d * static_cast<int64_t>(sizeof(half));

    auto buffer_all_equal = [&](Surface<half> const &buffer, half expected) {
        std::vector<half> host(buffer.size);
        CUDA_CHECK(cudaMemcpy(host.data(), buffer.devPtr, sizeof(half) * buffer.size, cudaMemcpyDeviceToHost));
        for (auto element : host) {
            if (element != expected) {
                return false;
            }
        }
        return true;
    };

    // What the caller is expected to bind: the slice source, the matmul weight
    // and the graph output.  The slice destination is derived by the frontend.
    std::unordered_map<int64_t, void *> first_bind = {
        {kUidSliceX, x.devPtr}, {kUidSliceW, w.devPtr}, {kUidSliceZ, z.devPtr}};
    // The SAME graph re-bound to FRESH source and output buffers: the alias has
    // to be re-derived from the new source pointer, not remembered from the
    // populate that created it.
    std::unordered_map<int64_t, void *> second_bind = {
        {kUidSliceX, x2.devPtr}, {kUidSliceW, w.devPtr}, {kUidSliceZ, z2.devPtr}};
    // The same call with the derived pointer spelled out by hand.
    std::unordered_map<int64_t, void *> explicit_alias = {
        {kUidSliceX, x.devPtr},
        {kUidSliceW, w.devPtr},
        {kUidSliceY, reinterpret_cast<char *>(x.devPtr) + offset_bytes},
        {kUidSliceZ, z.devPtr}};

    auto replay_and_check = [&](cudaGraph_t target, std::string const &label, Surface<half> &buffer, float expected) {
        cudaGraphExec_t exec = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&exec, target, nullptr, nullptr, 0));
        fillImage(buffer.devPtr, buffer.size, sentinel);
        CUDA_CHECK(cudaGraphLaunch(exec, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(buffer_all_equal(buffer, __float2half(expected)));
        ev("[186][G11] PASS " + label + " replayed the alias and produced " + fmt(expected, 1));
        CUDA_CHECK(cudaGraphExecDestroy(exec));
    };

    // 1. Wrapped populate, then update to FRESH buffers: both bind the
    //    replacement from the source they were given.
    cudaGraph_t wrapped = nullptr;
    CUDA_CHECK(cudaGraphCreate(&wrapped, 0));
    auto wrapped_populate = graph->populate_cuda_graph(handle, first_bind, workspace.devPtr, wrapped);
    ev("[186][G11] wrapped populate with a source-only slice binding: " +
       (wrapped_populate.is_good() ? std::string("OK") : wrapped_populate.get_message()));
    REQUIRE(wrapped_populate.is_good());
    replay_and_check(wrapped, "wrapped populate", z, 64.0f);

    auto wrapped_update = graph->update_cuda_graph(handle, second_bind, workspace.devPtr, wrapped);
    ev("[186][G11] wrapped update onto fresh source/output buffers: " +
       (wrapped_update.is_good() ? std::string("OK") : wrapped_update.get_message()));
    REQUIRE(wrapped_update.is_good());
    replay_and_check(wrapped, "wrapped update (fresh buffers)", z2, 128.0f);
    REQUIRE(buffer_all_equal(z, __float2half(64.0f)));
    ev("[186][G11] the first output buffer still holds its own result -> the alias was not carried over");

    // 2. Spelling the derived pointer out by hand stays accepted.
    cudaGraph_t explicit_graph = nullptr;
    CUDA_CHECK(cudaGraphCreate(&explicit_graph, 0));
    auto explicit_populate = graph->populate_cuda_graph(handle, explicit_alias, workspace.devPtr, explicit_graph);
    ev("[186][G11] wrapped populate with the alias spelled out: " +
       (explicit_populate.is_good() ? std::string("OK") : explicit_populate.get_message()));
    REQUIRE(explicit_populate.is_good());
    replay_and_check(explicit_graph, "explicit alias populate", z, 64.0f);

    // 3. The direct entry points take the same binding, including the rebind.
    if (direct_population_eligible(graph)) {
        cudaGraph_t direct = nullptr;
        CUDA_CHECK(cudaGraphCreate(&direct, 0));
        auto direct_populate = populate_graph(graph, handle, first_bind, workspace.devPtr, direct);
        ev("[186][G11] " + direct_path_label() + " populate with a source-only slice binding: " +
           (direct_populate.is_good() ? std::string("OK") : direct_populate.get_message()));
        REQUIRE(direct_populate.is_good());
        replay_and_check(direct, direct_path_label() + " populate", z, 64.0f);

        auto direct_update = update_graph(graph, handle, second_bind, workspace.devPtr, direct);
        ev("[186][G11] " + direct_path_label() + " update onto fresh source/output buffers: " +
           (direct_update.is_good() ? std::string("OK") : direct_update.get_message()));
        REQUIRE(direct_update.is_good());
        replay_and_check(direct, direct_path_label() + " update (fresh buffers)", z2, 128.0f);
        REQUIRE(buffer_all_equal(z, __float2half(64.0f)));
        CUDA_CHECK(cudaGraphDestroy(direct));
    } else {
        ev("[186][G11] direct population is not eligible for this graph: " + status.reason);
    }

    // 4. A variant pack that names neither end of the alias fails closed
    //    instead of binding a pointer that was never supplied.
    cudaGraph_t output_only_graph = nullptr;
    CUDA_CHECK(cudaGraphCreate(&output_only_graph, 0));
    std::unordered_map<int64_t, void *> output_only = {{kUidSliceW, w.devPtr}, {kUidSliceZ, z.devPtr}};
    auto unbound = graph->populate_cuda_graph(handle, output_only, workspace.devPtr, output_only_graph);
    ev("[186][G11] populate without the slice source: " +
       (unbound.is_good() ? std::string("OK") : unbound.get_message()));
    REQUIRE(unbound.is_bad());
    CUDA_CHECK(cudaGraphDestroy(output_only_graph));

    CUDA_CHECK(cudaGraphDestroy(wrapped));
    CUDA_CHECK(cudaGraphDestroy(explicit_graph));
#endif
}
