/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

// The AOT surface: export a set of built graphs to one container, import them
// back by name, or publish them into the process-global table.
//
// This header is NOT included by cudnn_frontend.h. Including it is what opts a
// translation unit into the tvm-ffi dependency; a consumer that never exports
// or imports an artifact needs none of it.
//
//     cudnn_frontend::KernelLibrary lib;
//     CHECK(cudnn_frontend::import_from_disk("mykernels.cudnn", &lib));
//     cudnn_frontend::graph::Graph graph_ln;
//     CHECK(lib.get("ln_fwd_bf16", &graph_ln));
//     CHECK(graph_ln.execute(handle, ptrs.data(), (int)ptrs.size(), workspace));

#include "../cudnn_frontend.h"
#include "experimental/cutedsl_ffi_engine.h"

#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cudnn_frontend {

// Re-exported so callers say cudnn_frontend::CuteDslPayload.
using experimental::CuteDslArgKind;
using experimental::CuteDslArgSpec;
using experimental::CuteDslPayload;

// Bumped when the container layout changes in a way an older reader cannot
// cope with. The per-graph blobs carry their own json_version independently.
static constexpr int KERNEL_LIBRARY_CONTAINER_VERSION = 1;

/**
 * @brief A set of AOT kernels, addressed by name.
 *
 * Lookup is by name and never by position, so adding a kernel to a container
 * cannot change what an existing caller resolves.
 */
class KernelLibrary {
   public:
    /**
     * @brief Fetch the graph published under @p name.
     *
     * @param name Name given to set_name() when the graph was exported.
     * @param out  Receives a graph that is ready to execute. A snapshot: it
     *             keeps a reference to the compiled kernel it was fetched with.
     * @return error_t OK, or an error naming the keys that do exist.
     */
    error_t
    get(std::string const &name, graph::Graph *out) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(out == nullptr, error_code_t::INVALID_VALUE, "get() needs an out graph.");
        auto it = graphs_.find(name);
        if (it == graphs_.end()) {
            std::string known;
            for (auto const &[k, v] : graphs_) {
                (void)v;
                known += (known.empty() ? "" : ", ") + k;
            }
            return {
                error_code_t::INVALID_VALUE,
                "No kernel named '" + name + "' in this library. It holds: " + (known.empty() ? "(nothing)" : known)};
        }
        *out = *it->second;
        return {error_code_t::OK, ""};
    }

    std::vector<std::string>
    keys() const {
        std::vector<std::string> names;
        names.reserve(graphs_.size());
        for (auto const &[k, v] : graphs_) {
            (void)v;
            names.push_back(k);
        }
        return names;
    }

    size_t
    size() const {
        return graphs_.size();
    }

    // Populated by import_from_disk(); public so the Python bindings can hand
    // the same shared graphs out without a second copy.
    std::map<std::string, std::shared_ptr<graph::Graph>> graphs_;
};

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB

namespace detail {

inline error_t
build_container(std::vector<std::shared_ptr<graph::Graph>> const &graphs, json &container) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        graphs.empty(), error_code_t::INVALID_VALUE, "export_to_disk() needs at least one graph.");

    container                      = json::object();
    container["container_version"] = KERNEL_LIBRARY_CONTAINER_VERSION;
    container["fe_version"]        = std::to_string(CUDNN_FRONTEND_MAJOR_VERSION) + "." +
                              std::to_string(CUDNN_FRONTEND_MINOR_VERSION) + "." +
                              std::to_string(CUDNN_FRONTEND_PATCH_VERSION);
    container["cudnn_backend_version"] = static_cast<int64_t>(detail::get_backend_version());
    container["graphs"]                = json::object();

    for (auto const &g : graphs) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(g == nullptr, error_code_t::INVALID_VALUE, "export_to_disk() got a null graph.");
        std::string const name = g->get_name();
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            name.empty(),
            error_code_t::INVALID_VALUE,
            "Every graph must be named with set_name() before export: the name is the lookup key.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            container["graphs"].contains(name),
            error_code_t::INVALID_VALUE,
            "Two graphs in this call are both named '" + name + "'; names must be unique within one container.");

        json graph_json;
        CHECK_CUDNN_FRONTEND_ERROR(g->serialize_plan(graph_json));
        container["graphs"][name] = std::move(graph_json);
    }
    return {error_code_t::OK, ""};
}

inline error_t
load_container(json const &container, cudnnHandle_t handle, KernelLibrary &lib) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(!container.contains("container_version"),
                                   error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                   "This file is not a cuDNN kernel library (no container_version).");
    int const version = container["container_version"].get<int>();
    RETURN_CUDNN_FRONTEND_ERROR_IF(version > KERNEL_LIBRARY_CONTAINER_VERSION,
                                   error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                   "This container is version " + std::to_string(version) +
                                       ", newer than this build understands (" +
                                       std::to_string(KERNEL_LIBRARY_CONTAINER_VERSION) + ").");
    RETURN_CUDNN_FRONTEND_ERROR_IF(!container.contains("graphs") || !container["graphs"].is_object(),
                                   error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                   "Container has no graphs section.");

    for (auto it = container["graphs"].begin(); it != container["graphs"].end(); ++it) {
        auto g = std::make_shared<graph::Graph>();
        // Every mismatch a graph can have — json version, engine kind,
        // architecture, missing runtime dependency, corrupt module — is raised
        // here, before anything is executable.
        auto status = g->deserialize(handle, it.value(), false, false);
        if (status.is_bad()) {
            return {status.get_code(), "Loading kernel '" + it.key() + "': " + status.get_message()};
        }
        lib.graphs_[it.key()] = std::move(g);
    }
    return {error_code_t::OK, ""};
}

}  // namespace detail

/**
 * @brief Pack a set of built graphs into one container file.
 *
 * The whole set goes in and one container comes out; adding a kernel later
 * means calling this again with the full set.
 */
inline error_t
export_to_disk(std::vector<std::shared_ptr<graph::Graph>> const &graphs, std::string const &path) {
    json container;
    CHECK_CUDNN_FRONTEND_ERROR(detail::build_container(graphs, container));

    std::vector<uint8_t> const data = json::to_ubjson(container);

    // Written to a sibling temporary and renamed, so a reader never sees a
    // half-written container under the real name.
    std::string const tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !f.good(), error_code_t::INVALID_VALUE, "Could not open " + tmp_path + " for writing.");
        f.write(reinterpret_cast<char const *>(data.data()), static_cast<std::streamsize>(data.size()));
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !f.good(), error_code_t::INVALID_VALUE, "Short write producing " + tmp_path + ".");
    }
    RETURN_CUDNN_FRONTEND_ERROR_IF(std::rename(tmp_path.c_str(), path.c_str()) != 0,
                                   error_code_t::INVALID_VALUE,
                                   "Could not publish the container as " + path + ".");
    return {error_code_t::OK, ""};
}

/**
 * @brief Read a container back. Every graph in it is executable on return.
 *
 * @param path   Container written by export_to_disk().
 * @param lib    Receives the name -> graph mapping.
 * @param handle Only used by graphs whose plan is a cuDNN backend plan; a
 *               container of CuTeDSL kernels loads without one.
 */
inline error_t
import_from_disk(std::string const &path, KernelLibrary *lib, cudnnHandle_t handle = nullptr) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(lib == nullptr, error_code_t::INVALID_VALUE, "import_from_disk() needs a library.");

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    RETURN_CUDNN_FRONTEND_ERROR_IF(!f.good(), error_code_t::INVALID_VALUE, "Could not open " + path + " for reading.");
    std::streamsize const size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    RETURN_CUDNN_FRONTEND_ERROR_IF(!f.read(reinterpret_cast<char *>(data.data()), size),
                                   error_code_t::INVALID_VALUE,
                                   "Short read loading " + path + ".");

    json container;
    try {
        container = json::from_ubjson(data);
    } catch (std::exception const &e) {
        return {error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                "Could not parse " + path + " as a cuDNN kernel library: " + e.what()};
    }
    return detail::load_container(container, handle, *lib);
}

#endif  // CUDNN_FRONTEND_SKIP_JSON_LIB

// ---------------------------------------------------------------------------
// Flow 3: publish into the process, no file.
//
// A kernel compiled mid-run — by an autotuner, or a dynamic specialisation — is
// immediately reachable from the C++ executor under a name. The registry owns a
// reference, so the compiled object cannot be collected out from under a live
// handle. On the wire the tvm-ffi name is "cudnn.<name>"; that prefix is applied
// and stripped here and never appears in user code.
// ---------------------------------------------------------------------------

namespace detail {

struct GlobalRegistry {
    std::mutex mu;
    std::map<std::string, std::shared_ptr<graph::Graph>> graphs;
};

// The registry must be ONE instance per process, not one per shared object.
// Flow 3's whole point is that a kernel registered from Python is reachable
// from a C++ extension in the same process -- and that extension is a separate
// .so. cuDNN FE's own python module is built with -fvisibility=hidden, which
// would give each .so a private copy of this function-local static and leave
// the extension looking at an empty registry. Marking it default-visibility
// makes the dynamic linker bind every DSO to the first definition loaded.
#if defined(_WIN32)
// Windows has no equivalent interposition: each DLL would still get its own
// copy, so flow 3 across DLL boundaries needs the registry to live in one
// owning DLL with dllexport/dllimport. Not needed for this proof of concept.
#define CUDNN_FRONTEND_SHARED_ACROSS_DSO
#else
#define CUDNN_FRONTEND_SHARED_ACROSS_DSO __attribute__((visibility("default")))
#endif

CUDNN_FRONTEND_SHARED_ACROSS_DSO inline GlobalRegistry &
global_registry() {
    static GlobalRegistry registry;
    return registry;
}

}  // namespace detail

// The one place the wire prefix is spelled.
inline std::string
global_symbol_for(std::string const &name) {
    return "cudnn." + name;
}

/**
 * @brief Publish a built graph under its set_name() name.
 *
 * @param graph    The graph. The registry keeps a reference to it.
 * @param override Replace an existing registration instead of failing. Callers
 *                 holding a handle from get_global() must re-fetch afterwards.
 */
inline error_t
register_global(std::shared_ptr<graph::Graph> const &graph, bool override = false) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(graph == nullptr, error_code_t::INVALID_VALUE, "register_global() needs a graph.");
    std::string const name = graph->get_name();
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        name.empty(),
        error_code_t::INVALID_VALUE,
        "register_global() needs a name: call set_name() first, it is the lookup key in both AOT flows.");

    auto &registry = detail::global_registry();
    std::lock_guard<std::mutex> lk(registry.mu);
    RETURN_CUDNN_FRONTEND_ERROR_IF(!override && registry.graphs.count(name) != 0,
                                   error_code_t::INVALID_VALUE,
                                   "'" + name + "' is already registered. Pass override=true to replace it.");
    registry.graphs[name] = graph;
    return {error_code_t::OK, ""};
}

/**
 * @brief Fetch a graph published by register_global().
 *
 * The result is a snapshot: after an override=true re-registration, an existing
 * handle keeps running the kernel it was fetched with, and a caller that wants
 * the new one must call get_global() again.
 */
inline error_t
get_global(std::string const &name, graph::Graph *out) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(out == nullptr, error_code_t::INVALID_VALUE, "get_global() needs an out graph.");
    auto &registry = detail::global_registry();
    std::lock_guard<std::mutex> lk(registry.mu);
    auto it = registry.graphs.find(name);
    if (it == registry.graphs.end()) {
        std::string known;
        for (auto const &[k, v] : registry.graphs) {
            (void)v;
            known += (known.empty() ? "" : ", ") + k;
        }
        return {error_code_t::INVALID_VALUE,
                "Nothing is registered under '" + name + "'. Registered: " + (known.empty() ? "(nothing)" : known)};
    }
    *out = *it->second;
    return {error_code_t::OK, ""};
}

/// @brief Drop a registration. Snapshots already handed out keep working.
inline error_t
unregister_global(std::string const &name) {
    auto &registry = detail::global_registry();
    std::lock_guard<std::mutex> lk(registry.mu);
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        registry.graphs.erase(name) == 0, error_code_t::INVALID_VALUE, "Nothing is registered under '" + name + "'.");
    return {error_code_t::OK, ""};
}

inline std::vector<std::string>
registered_global_names() {
    auto &registry = detail::global_registry();
    std::lock_guard<std::mutex> lk(registry.mu);
    std::vector<std::string> names;
    names.reserve(registry.graphs.size());
    for (auto const &[k, v] : registry.graphs) {
        (void)v;
        names.push_back(k);
    }
    return names;
}

}  // namespace cudnn_frontend
