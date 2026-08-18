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

// Payload description and abstract engine for AOT-exported CuTeDSL-family
// kernels. Deliberately free of any tvm-ffi dependency: graph_interface.h and
// plans.h include this header, so cuDNN FE core must stay buildable without
// the tvm-ffi headers on the include path. The concrete engine that actually
// loads a module and calls into it lives in cutedsl_ffi_engine.h, which is
// pulled in only by kernel_library.h (the opt-in AOT surface).

#include "../graph_helpers.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cudnn_frontend {
namespace experimental {

// One positional argument of the exported kernel's C entry point, in call
// order. The graph blob knows the sorted-uid order of the variant pack; it does
// not know the kernel's signature, which is what this describes.
enum class CuteDslArgKind {
    TENSOR,      // a variant-pack tensor, addressed by uid
    WORKSPACE,   // a slice of the engine workspace at workspace_offset
    SCALAR_I64,  // a compile-time-frozen integer baked into the artifact
    SCALAR_F64,  // a compile-time-frozen double baked into the artifact
    ARRAY_I64,   // a tuple of integers the kernel takes as ONE argument
    ENV_STREAM,  // the kernel takes the tvm-ffi environment stream (no argument)
    STREAM,      // the execution stream, passed positionally as an opaque handle
    NONE  // an optional port the kernel was compiled without; still a positional slot
};

struct CuteDslArgSpec {
    CuteDslArgKind kind = CuteDslArgKind::TENSOR;

    int64_t uid              = -1;  // TENSOR
    int64_t workspace_offset = 0;   // WORKSPACE

    // Shape / stride / dtype as the KERNEL sees the buffer. Carried explicitly
    // rather than read back off the graph's tensor list so that a plan-only
    // blob (serialize_structure=false) round-trips, and so a mismatch against
    // the graph is detectable at load instead of at launch.
    DataType_t data_type = DataType_t::NOT_SET;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;

    int64_t scalar_i64 = 0;
    double scalar_f64  = 0.0;
    std::vector<int64_t> values;  // ARRAY_I64

    // Index into VariantPackTemplate::all_uids, resolved by bind_slots().
    // Mirrors how Execution_plan_list::set_oss_slot_indices() gives the -2/-3
    // engines O(1) access to the finished pointer array.
    int slot = -1;
};

enum class CuteDslStepKind {
    CALL,        // invoke an exported kernel through its tvm-ffi entry point
    MEMSET_ZERO  // zero a workspace region, stream-ordered, before the next step
};

// One entry in the launch sequence.
//
// A single exported symbol is enough for a kernel that is one launch, which is
// what the elementwise-add vehicle is. Real families are not: the FROST GDN
// forward partitions its work on the device, builds its TMA descriptors, and
// only then runs the chunked kernel -- three kernels plus a zeroed counter,
// with the intermediates living in the caller's workspace. The host code that
// sequences them is Python, and Python is exactly what an AOT artifact must not
// need, so the sequence itself becomes part of the payload.
//
// Every step launches on the same stream in order, so device-side ordering is
// the stream's and no host synchronization appears anywhere in execute().
struct CuteDslStep {
    CuteDslStepKind kind = CuteDslStepKind::CALL;

    // CALL: resolved inside the payload's module, or -- for flow 3, where
    // nothing was ever serialised -- in the tvm-ffi global table.
    std::string function_name;
    std::string global_symbol;
    std::vector<CuteDslArgSpec> args;

    // MEMSET_ZERO
    int64_t workspace_offset = 0;
    int64_t nbytes           = 0;
};

// Everything needed to reconstitute and call one exported kernel. Written to
// the container under the "cutedsl_data" key, next to "cudnn_backend_data".
struct CuteDslPayload {
    std::string abi = "tvm-ffi";
    int abi_version = 1;

    // The linked shared object. The cubin is embedded inside it by the CuTeDSL
    // exporter, so there is no separate device-code blob and FE never calls
    // cuModuleLoad: the exported host shim does that on first call.
    std::vector<uint8_t> module_bytes;
    // Integrity, not authenticity: catches a truncated or garbled container
    // before the bytes are handed to dlopen. FNV-1a/64 rather than a digest so
    // FE does not grow a crypto implementation; both sides compute it the same
    // way and the value doubles as the content-addressed cache-file name.
    uint64_t module_hash = 0;

    // Checked at load, loudly, so a wrong-arch artifact is an error and not an
    // illegal memory access.
    std::string sm_arch;

    // dlopen'd before the module itself, so a deployment missing
    // libcute_dsl_runtime.so gets a named error rather than an unresolved
    // symbol abort out of the dynamic loader.
    std::vector<std::string> runtime_deps;

    int64_t engine_workspace_size = 0;

    // What execute() runs, in order. A one-launch kernel is a one-element list.
    std::vector<CuteDslStep> steps;
};

// The per-engine half of the plugin interface. One implementation per ABI; the
// only one today is tvm-ffi (CuteDslFfiEngine).
class ICuteDslEngine {
   public:
    virtual ~ICuteDslEngine() = default;

    virtual int64_t
    get_workspace_size() const = 0;

    // Resolve every TENSOR arg's uid to its slot in the finished pointer array.
    // Called once, from prepare_variant_pack_template().
    virtual error_t
    bind_slots(std::function<int(int64_t)> const &slot_for) = 0;

    // Hot path. const, and must not mutate the engine: any number of threads
    // may execute one graph concurrently.
    virtual error_t
    execute(void *const *ptrs, void *engine_workspace, cudaStream_t stream) const = 0;
};

// Constructing an engine needs the tvm-ffi headers, which core FE does not
// include. cutedsl_ffi_engine.h installs itself here on include; a graph that
// deserializes a cutedsl payload without that header available fails with a
// clear message instead of dereferencing null.
using CuteDslEngineFactory = error_t (*)(CuteDslPayload const &, std::shared_ptr<ICuteDslEngine> &);

inline CuteDslEngineFactory &
cutedsl_engine_factory() {
    static CuteDslEngineFactory factory = nullptr;
    return factory;
}

inline error_t
make_cutedsl_engine(CuteDslPayload const &payload, std::shared_ptr<ICuteDslEngine> &engine) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(cutedsl_engine_factory() == nullptr,
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "This graph carries a CuTeDSL (" + payload.abi +
                                       ") payload, but AOT support is not available in this build. Include "
                                       "cudnn_frontend/kernel_library.h and build against the tvm-ffi headers.");
    return cutedsl_engine_factory()(payload, engine);
}

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB

NLOHMANN_JSON_SERIALIZE_ENUM(CuteDslArgKind,
                             {
                                 {CuteDslArgKind::TENSOR, "TENSOR"},
                                 {CuteDslArgKind::WORKSPACE, "WORKSPACE"},
                                 {CuteDslArgKind::SCALAR_I64, "SCALAR_I64"},
                                 {CuteDslArgKind::SCALAR_F64, "SCALAR_F64"},
                                 {CuteDslArgKind::ARRAY_I64, "ARRAY_I64"},
                                 {CuteDslArgKind::ENV_STREAM, "ENV_STREAM"},
                                 {CuteDslArgKind::STREAM, "STREAM"},
                                 {CuteDslArgKind::NONE, "NONE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(CuteDslStepKind,
                             {
                                 {CuteDslStepKind::CALL, "CALL"},
                                 {CuteDslStepKind::MEMSET_ZERO, "MEMSET_ZERO"},
                             })

inline void
to_json(json &j, CuteDslArgSpec const &a) {
    j         = json::object();
    j["kind"] = a.kind;
    switch (a.kind) {
        case CuteDslArgKind::TENSOR:
            j["uid"] = a.uid;
            break;
        case CuteDslArgKind::WORKSPACE:
            j["workspace_offset"] = a.workspace_offset;
            break;
        case CuteDslArgKind::SCALAR_I64:
            j["value"] = a.scalar_i64;
            break;
        case CuteDslArgKind::ARRAY_I64:
            j["values"] = a.values;
            break;
        case CuteDslArgKind::SCALAR_F64:
            j["value"] = a.scalar_f64;
            break;
        case CuteDslArgKind::ENV_STREAM:
        case CuteDslArgKind::STREAM:
        case CuteDslArgKind::NONE:
            return;
    }
    j["data_type"] = a.data_type;
    j["shape"]     = a.shape;
    j["stride"]    = a.stride;
}

inline void
from_json(json const &j, CuteDslArgSpec &a) {
    a.kind = j.at("kind").get<CuteDslArgKind>();
    if (a.kind == CuteDslArgKind::ENV_STREAM || a.kind == CuteDslArgKind::STREAM ||
        a.kind == CuteDslArgKind::NONE) {
        return;
    }
    if (a.kind == CuteDslArgKind::ARRAY_I64) {
        if (j.contains("values")) a.values = j.at("values").get<std::vector<int64_t>>();
        return;  // carries no shape/stride: it is not a buffer
    }
    if (j.contains("uid")) a.uid = j.at("uid").get<int64_t>();
    if (j.contains("workspace_offset")) a.workspace_offset = j.at("workspace_offset").get<int64_t>();
    if (a.kind == CuteDslArgKind::SCALAR_I64 && j.contains("value")) a.scalar_i64 = j.at("value").get<int64_t>();
    if (a.kind == CuteDslArgKind::SCALAR_F64 && j.contains("value")) a.scalar_f64 = j.at("value").get<double>();
    if (j.contains("data_type")) a.data_type = j.at("data_type").get<DataType_t>();
    if (j.contains("shape")) a.shape = j.at("shape").get<std::vector<int64_t>>();
    if (j.contains("stride")) a.stride = j.at("stride").get<std::vector<int64_t>>();
}

inline void
to_json(json &j, CuteDslStep const &s) {
    j         = json::object();
    j["kind"] = s.kind;
    if (s.kind == CuteDslStepKind::MEMSET_ZERO) {
        j["workspace_offset"] = s.workspace_offset;
        j["nbytes"]           = s.nbytes;
        return;
    }
    j["function_name"] = s.function_name;
    j["global_symbol"] = s.global_symbol;
    j["args"]          = s.args;
}

inline void
from_json(json const &j, CuteDslStep &s) {
    s.kind = j.value("kind", CuteDslStepKind::CALL);
    if (s.kind == CuteDslStepKind::MEMSET_ZERO) {
        s.workspace_offset = j.value("workspace_offset", static_cast<int64_t>(0));
        s.nbytes           = j.value("nbytes", static_cast<int64_t>(0));
        return;
    }
    s.function_name = j.value("function_name", std::string{});
    s.global_symbol = j.value("global_symbol", std::string{});
    if (j.contains("args")) s.args = j.at("args").get<std::vector<CuteDslArgSpec>>();
}

inline void
to_json(json &j, CuteDslPayload const &p) {
    j                          = json::object();
    j["abi"]                   = p.abi;
    j["abi_version"]           = p.abi_version;
    j["module"]                = json::object();
    j["module"]["format"]      = "elf-so";
    j["module"]["hash_algo"]   = "fnv1a64";
    j["module"]["hash"]        = p.module_hash;
    j["module"]["size"]        = static_cast<int64_t>(p.module_bytes.size());
    j["module"]["bytes"]       = json::binary(p.module_bytes);
    j["sm_arch"]               = p.sm_arch;
    j["runtime_deps"]          = p.runtime_deps;
    j["engine_workspace_size"] = p.engine_workspace_size;
    j["steps"]                 = p.steps;
}

inline void
from_json(json const &j, CuteDslPayload &p) {
    p.abi         = j.value("abi", std::string("tvm-ffi"));
    p.abi_version = j.value("abi_version", 1);
    if (j.contains("module")) {
        auto const &m = j.at("module");
        if (m.contains("hash")) p.module_hash = m.at("hash").get<uint64_t>();
        if (m.contains("bytes")) {
            auto const &b = m.at("bytes");
            if (b.is_binary()) {
                auto const &bin = b.get_binary();
                p.module_bytes.assign(bin.begin(), bin.end());
            } else {
                p.module_bytes = b.get<std::vector<uint8_t>>();
            }
        }
    }
    p.sm_arch = j.value("sm_arch", std::string{});
    if (j.contains("runtime_deps")) p.runtime_deps = j.at("runtime_deps").get<std::vector<std::string>>();
    p.engine_workspace_size = j.value("engine_workspace_size", static_cast<int64_t>(0));
    if (j.contains("steps")) p.steps = j.at("steps").get<std::vector<CuteDslStep>>();
}

#endif

}  // namespace experimental
}  // namespace cudnn_frontend
