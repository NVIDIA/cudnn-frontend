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

// The tvm-ffi implementation of ICuteDslEngine. This is the only FE header
// that includes the tvm-ffi headers; it is reached through kernel_library.h,
// never through cudnn_frontend.h, so a consumer that does not use AOT keeps
// building with no tvm-ffi on the include path.

#include "cutedsl_engine_interface.h"

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/module.h>

// <tvm/ffi/extra/c_env_api.h> is not included on purpose. It declares an
// allocator hook whose type comes from a newer DLPack than the one cudnn-frontend
// vendors, and whichever dlpack.h is first on the include path wins for the whole
// translation unit. Only the stream hook is needed here, and it is a stable C ABI
// entry point, so it is declared directly rather than dragging in a second DLPack.
extern "C" {
TVM_FFI_DLL int
TVMFFIEnvSetStream(int32_t device_type, int32_t device_id, void *stream, void **opt_out_original_stream);
}

#include "../../cudnn_frontend_shim.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cudnn_frontend {
namespace experimental {

namespace cutedsl_detail {

inline uint64_t
fnv1a64(void const *data, size_t n) {
    auto const *p            = static_cast<unsigned char const *>(data);
    uint64_t hash            = 1469598103934665603ull;
    constexpr uint64_t prime = 1099511628211ull;
    for (size_t i = 0; i < n; i++) {
        hash ^= static_cast<uint64_t>(p[i]);
        hash *= prime;
    }
    return hash;
}

inline std::string
cache_dir() {
    if (char const *env = std::getenv("CUDNN_FRONTEND_AOT_CACHE_DIR")) {
        return std::string(env);
    }
    char const *tmp = std::getenv("TMPDIR");
    return std::string(tmp ? tmp : "/tmp") + "/cudnn_fe_aot_cache";
}

// Materialise the module bytes as a file so the dynamic loader can map them.
// Content-addressed: the same artifact loaded twice, or by two processes,
// resolves to one file, and a partially written file is never observed under
// its final name because the write goes to a pid-unique temporary first.
inline error_t
write_module_file(CuteDslPayload const &payload, std::string &out_path) {
    std::string const dir = cache_dir();
    // mkdir -p, one level: the parent is /tmp or a user-supplied directory.
    ::mkdir(dir.c_str(), 0700);

    char name[64];
    std::snprintf(name, sizeof(name), "/cudnn_fe_%016llx.so", static_cast<unsigned long long>(payload.module_hash));
    out_path = dir + name;

    {
        std::ifstream probe(out_path, std::ios::binary | std::ios::ate);
        if (probe.good() && static_cast<size_t>(probe.tellg()) == payload.module_bytes.size()) {
            return {error_code_t::OK, ""};
        }
    }

    char tmp_name[96];
    std::snprintf(tmp_name,
                  sizeof(tmp_name),
                  "/cudnn_fe_%016llx.so.%d.tmp",
                  static_cast<unsigned long long>(payload.module_hash),
                  static_cast<int>(::getpid()));
    std::string const tmp_path = dir + tmp_name;

    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!f.good(),
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "Could not open " + tmp_path +
                                           " to materialise the AOT module: " + std::strerror(errno) +
                                           ". Set CUDNN_FRONTEND_AOT_CACHE_DIR to a writable directory.");
        f.write(reinterpret_cast<char const *>(payload.module_bytes.data()),
                static_cast<std::streamsize>(payload.module_bytes.size()));
        RETURN_CUDNN_FRONTEND_ERROR_IF(!f.good(),
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "Short write materialising the AOT module to " + tmp_path);
    }

    RETURN_CUDNN_FRONTEND_ERROR_IF(::rename(tmp_path.c_str(), out_path.c_str()) != 0,
                                   error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                   "Could not publish the AOT module as " + out_path + ": " + std::strerror(errno));
    return {error_code_t::OK, ""};
}

// A missing libcute_dsl_runtime.so otherwise surfaces as an unresolved-symbol
// failure from inside dlopen of the artifact, which names the artifact and not
// the thing that is actually absent. Load the declared dependencies first so
// the error says what to install.
inline error_t
preload_runtime_deps(CuteDslPayload const &payload) {
    for (auto const &dep : payload.runtime_deps) {
        ::dlerror();
        if (::dlopen(dep.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) {
            char const *why = ::dlerror();
            return {error_code_t::GRAPH_NOT_SUPPORTED,
                    "This artifact needs the runtime dependency '" + dep +
                        "', which could not be loaded: " + std::string(why ? why : "unknown error") +
                        ". Install it, or put its directory on LD_LIBRARY_PATH."};
        }
    }
    return {error_code_t::OK, ""};
}

// Two graphs exported from one build usually share nothing, but re-importing
// the same container, or importing one that holds several graphs against the
// same module, should dlopen once.
inline error_t
load_module_cached(CuteDslPayload const &payload, std::optional<tvm::ffi::Module> &out) {
    static std::mutex mu;
    static std::map<uint64_t, tvm::ffi::Module> cache;

    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(payload.module_hash);
    if (it != cache.end()) {
        out = it->second;
        return {error_code_t::OK, ""};
    }

    CHECK_CUDNN_FRONTEND_ERROR(preload_runtime_deps(payload));

    std::string path;
    CHECK_CUDNN_FRONTEND_ERROR(write_module_file(payload, path));

    // tvm-ffi reports failure by throwing; FE never lets an exception escape.
    try {
        tvm::ffi::Module module = tvm::ffi::Module::LoadFromFile(path);
        cache.emplace(payload.module_hash, module);
        out = std::move(module);
    } catch (std::exception const &e) {
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                "Failed to load the AOT module written to " + path + ": " + e.what()};
    }
    return {error_code_t::OK, ""};
}

inline error_t
to_dl_data_type(DataType_t dt, DLDataType &out) {
    out.lanes = 1;
    switch (dt) {
        case DataType_t::FLOAT:
            out.code = kDLFloat, out.bits = 32;
            break;
        case DataType_t::DOUBLE:
            out.code = kDLFloat, out.bits = 64;
            break;
        case DataType_t::HALF:
            out.code = kDLFloat, out.bits = 16;
            break;
        case DataType_t::BFLOAT16:
            out.code = kDLBfloat, out.bits = 16;
            break;
        case DataType_t::INT8:
            out.code = kDLInt, out.bits = 8;
            break;
        case DataType_t::INT32:
            out.code = kDLInt, out.bits = 32;
            break;
        case DataType_t::INT64:
            out.code = kDLInt, out.bits = 64;
            break;
        case DataType_t::UINT8:
            out.code = kDLUInt, out.bits = 8;
            break;
        case DataType_t::BOOLEAN:
        case DataType_t::BYTE_BOOLEAN:
            out.code = kDLBool, out.bits = 8;
            break;
        case DataType_t::FP8_E4M3:
            out.code = kDLFloat8_e4m3fn, out.bits = 8;
            break;
        case DataType_t::FP8_E5M2:
            out.code = kDLFloat8_e5m2, out.bits = 8;
            break;
        default:
            return {error_code_t::GRAPH_NOT_SUPPORTED,
                    "No DLPack encoding for cuDNN data type " + std::to_string(static_cast<int>(dt)) +
                        " in a CuTeDSL argument"};
    }
    return {error_code_t::OK, ""};
}

}  // namespace cutedsl_detail

// Calls an AOT-exported CuTeDSL kernel through its tvm-ffi entry point.
//
// Everything that can be prepared once is prepared at load: the module is
// dlopen'd, every step's function is resolved, and one DLTensor per tensor
// argument is built with its shape/stride storage owned by this object. The hot
// path copies that array to the stack, patches the data pointers out of FE's
// finished pointer array, and calls. Nothing on the engine is mutated, so
// concurrent execute() of one graph is safe.
//
// A payload with a launch sequence (see CuteDslStep) runs its steps in order on
// one stream. All of them come out of the same module, so a sequence costs the
// same single dlopen as a one-launch kernel.
class CuteDslFfiEngine : public ICuteDslEngine {
   public:
    static error_t
    create(CuteDslPayload const &payload, std::shared_ptr<ICuteDslEngine> &out) {
        auto engine = std::shared_ptr<CuteDslFfiEngine>(new CuteDslFfiEngine());
        CHECK_CUDNN_FRONTEND_ERROR(engine->initialize(payload));
        out = std::move(engine);
        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const override {
        return payload_.engine_workspace_size;
    }

    error_t
    bind_slots(std::function<int(int64_t)> const &slot_for) override {
        for (auto &step : steps_) {
            for (size_t i = 0; i < step.args.size(); i++) {
                if (step.args[i].kind != CuteDslArgKind::TENSOR) continue;
                int const slot = slot_for(step.args[i].uid);
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    slot < 0,
                    error_code_t::INVALID_VARIANT_PACK,
                    "CuTeDSL step '" + step.label + "' argument " + std::to_string(i) + " refers to uid " +
                        std::to_string(step.args[i].uid) +
                        ", which is not part of this graph's variant pack. The artifact does not match the graph.");
                step.args[i].slot = slot;
            }
        }
        slots_bound_ = true;
        return {error_code_t::OK, ""};
    }

    error_t
    execute(void *const *ptrs, void *engine_workspace, cudaStream_t stream) const override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!slots_bound_,
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "CuTeDSL argument slots are not bound; prepare_variant_pack_template() must "
                                       "run before execute().");

        // Steps that take the stream as an explicit argument do not need this,
        // but a payload may mix both conventions, and it is one thread-local
        // store per execute() rather than per step. Thread-local, so setting it
        // does not disturb another thread executing another graph; restored so
        // we do not leak our stream into whatever the caller does next.
        void *original   = nullptr;
        int const set_rc = TVMFFIEnvSetStream(kDLCUDA, device_ordinal_, static_cast<void *>(stream), &original);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            set_rc != 0, error_code_t::GRAPH_EXECUTION_FAILED, "TVMFFIEnvSetStream failed for the CuTeDSL engine");

        error_t status = {error_code_t::OK, ""};
        for (auto const &step : steps_) {
            status = run_step(step, ptrs, engine_workspace, stream);
            if (status.is_bad()) break;
        }

        TVMFFIEnvSetStream(kDLCUDA, device_ordinal_, original, nullptr);
        return status;
    }

   private:
    CuteDslFfiEngine() = default;

    // One step of the launch sequence, with everything resolvable resolved.
    struct BoundStep {
        CuteDslStepKind kind = CuteDslStepKind::CALL;
        std::string label;  // for diagnostics: the symbol, or "memset"

        std::optional<tvm::ffi::Function> function;
        std::vector<CuteDslArgSpec> args;
        std::vector<DLTensor> prototypes;
        std::vector<std::vector<int64_t>> shape_storage;
        std::vector<std::vector<int64_t>> stride_storage;
        // ARRAY_I64 arguments, built ONCE here rather than per call: the values
        // are frozen into the artifact, and allocating a container on every
        // launch would put host cost back into the path this exists to keep cheap.
        std::vector<tvm::ffi::Array<int64_t>> arrays;

        int64_t workspace_offset = 0;
        int64_t nbytes           = 0;
    };

    error_t
    run_step(BoundStep const &step, void *const *ptrs, void *engine_workspace, cudaStream_t stream) const {
        if (step.kind == CuteDslStepKind::MEMSET_ZERO) {
            // Stream-ordered: the kernel that reads this counter is the next
            // step on the same stream, so no host synchronization is implied.
            _CUDNN_CHECK_CUDA_ERROR(detail::cuda_mem_set_async(
                static_cast<char *>(engine_workspace) + step.workspace_offset, 0, step.nbytes, stream));
            return {error_code_t::OK, ""};
        }

        int const n             = static_cast<int>(step.args.size());
        constexpr int STACK_MAX = 32;
        DLTensor stack_tensors[STACK_MAX];
        tvm::ffi::AnyView stack_views[STACK_MAX];
        std::vector<DLTensor> heap_tensors;
        std::vector<tvm::ffi::AnyView> heap_views;

        DLTensor *tensors        = stack_tensors;
        tvm::ffi::AnyView *views = stack_views;
        if (n > STACK_MAX) {
            heap_tensors.resize(n);
            heap_views.resize(n);
            tensors = heap_tensors.data();
            views   = heap_views.data();
        }

        int n_view = 0;
        for (int i = 0; i < n; i++) {
            auto const &a = step.args[i];
            switch (a.kind) {
                case CuteDslArgKind::ENV_STREAM:
                    continue;  // carried by the environment stream, not an argument
                case CuteDslArgKind::STREAM:
                    views[n_view++] = static_cast<void *>(stream);
                    continue;
                case CuteDslArgKind::NONE:
                    // An optional port the kernel was compiled without. CuTeDSL
                    // keeps it as a positional parameter typed None, so the slot
                    // has to be filled -- with nullptr, not skipped.
                    views[n_view++] = nullptr;
                    continue;
                case CuteDslArgKind::SCALAR_I64:
                    views[n_view++] = a.scalar_i64;
                    continue;
                case CuteDslArgKind::SCALAR_F64:
                    views[n_view++] = a.scalar_f64;
                    continue;
                case CuteDslArgKind::ARRAY_I64:
                    // built at bind time; a tuple parameter the kernel takes whole
                    views[n_view++] = step.arrays[i];
                    continue;
                case CuteDslArgKind::TENSOR:
                    tensors[n_view]      = step.prototypes[i];
                    tensors[n_view].data = ptrs[a.slot];
                    views[n_view]        = &tensors[n_view];
                    n_view++;
                    continue;
                case CuteDslArgKind::WORKSPACE:
                    tensors[n_view]      = step.prototypes[i];
                    tensors[n_view].data = static_cast<char *>(engine_workspace) + a.workspace_offset;
                    views[n_view]        = &tensors[n_view];
                    n_view++;
                    continue;
            }
        }

        try {
            tvm::ffi::Any result;
            step.function->CallPacked(views, n_view, &result);
        } catch (std::exception const &e) {
            return {error_code_t::GRAPH_EXECUTION_FAILED, "CuTeDSL kernel '" + step.label + "' failed: " + e.what()};
        }
        return {error_code_t::OK, ""};
    }

    error_t
    initialize(CuteDslPayload const &payload) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(payload.abi != "tvm-ffi",
                                       error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                       "Unknown CuTeDSL ABI '" + payload.abi + "'; this build understands 'tvm-ffi'.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(payload.abi_version != 1,
                                       error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                       "CuTeDSL tvm-ffi ABI version " + std::to_string(payload.abi_version) +
                                           " is newer than this build understands (1).");
        auto const &sequence = payload.steps;
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            sequence.empty(), error_code_t::UNSUPPORTED_GRAPH_FORMAT, "CuTeDSL payload describes no launch sequence.");

        // A step resolved from the global table needs nothing on disk; any step
        // resolved from the module means the module must be there.
        bool needs_module = false;
        for (auto const &step : sequence) {
            if (step.kind != CuteDslStepKind::CALL) continue;
            RETURN_CUDNN_FRONTEND_ERROR_IF(step.function_name.empty() && step.global_symbol.empty(),
                                           error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                           "A CuTeDSL step names no entry point.");
            if (step.global_symbol.empty()) needs_module = true;
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(needs_module && payload.module_bytes.empty(),
                                       error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                       "CuTeDSL payload carries neither module bytes nor a global symbol.");

        CHECK_CUDNN_FRONTEND_ERROR(check_target(payload));

        payload_ = payload;

        if (needs_module) {
            uint64_t const actual = cutedsl_detail::fnv1a64(payload.module_bytes.data(), payload.module_bytes.size());
            RETURN_CUDNN_FRONTEND_ERROR_IF(actual != payload.module_hash,
                                           error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                           "The AOT module in this artifact is corrupt: content hash is " +
                                               std::to_string(actual) + ", the container says " +
                                               std::to_string(payload.module_hash) + ".");
            CHECK_CUDNN_FRONTEND_ERROR(cutedsl_detail::load_module_cached(payload_, module_));
        }

        int device = 0;
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_get_device(&device));
        device_ordinal_ = device;

        steps_.reserve(sequence.size());
        for (auto const &step : sequence) {
            BoundStep bound;
            bound.kind = step.kind;
            if (step.kind == CuteDslStepKind::MEMSET_ZERO) {
                bound.label            = "memset";
                bound.workspace_offset = step.workspace_offset;
                bound.nbytes           = step.nbytes;
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    bound.nbytes < 0 || bound.workspace_offset < 0 ||
                        bound.workspace_offset + bound.nbytes > payload.engine_workspace_size,
                    error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                    "A CuTeDSL MEMSET_ZERO step falls outside the declared engine workspace.");
                steps_.push_back(std::move(bound));
                continue;
            }

            bound.args  = step.args;
            bound.label = step.global_symbol.empty() ? step.function_name : step.global_symbol;
            CHECK_CUDNN_FRONTEND_ERROR(resolve_function(step, bound));
            CHECK_CUDNN_FRONTEND_ERROR(build_prototypes(bound));
            steps_.push_back(std::move(bound));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    resolve_function(CuteDslStep const &step, BoundStep &bound) const {
        tvm::ffi::Optional<tvm::ffi::Function> fn;
        if (!step.global_symbol.empty()) {
            // Flow 3: the kernel was compiled in this process and published
            // under a name. Nothing was written out, so there is nothing to
            // hash or dlopen.
            try {
                fn = tvm::ffi::Function::GetGlobal(step.global_symbol);
            } catch (std::exception const &e) {
                return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                        "Looking up '" + step.global_symbol + "' in the global function table failed: " + e.what()};
            }
            RETURN_CUDNN_FRONTEND_ERROR_IF(!fn.has_value(),
                                           error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                           "Nothing is registered under '" + step.global_symbol + "'.");
        } else {
            try {
                fn = (*module_)->GetFunction(step.function_name);
            } catch (std::exception const &e) {
                return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                        "Looking up '" + step.function_name + "' in the AOT module failed: " + e.what()};
            }
            RETURN_CUDNN_FRONTEND_ERROR_IF(!fn.has_value(),
                                           error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                           "The AOT module does not export '" + step.function_name + "'.");
        }
        bound.function = fn.value();
        return {error_code_t::OK, ""};
    }

    // An artifact built for another architecture must be a clean error at load.
    // Executing it would be an illegal memory access at best.
    error_t
    check_target(CuteDslPayload const &payload) const {
        if (payload.sm_arch.empty()) {
            return {error_code_t::OK, ""};
        }
        int device = 0;
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_get_device(&device));
        cudaDeviceProp prop{};
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_get_device_properties(&prop, device));

        std::string const running = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
        // Artifacts are tagged sm_100a / sm_100f / sm_100; compare the numeric
        // part and let the family suffix through.
        std::string tagged = payload.sm_arch;
        while (!tagged.empty() && (tagged.back() == 'a' || tagged.back() == 'f')) {
            tagged.pop_back();
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            tagged != running,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "This artifact was built for " + payload.sm_arch + " but this device is " + running + ".");
        return {error_code_t::OK, ""};
    }

    error_t
    build_prototypes(BoundStep &bound) const {
        size_t const n = bound.args.size();
        bound.prototypes.assign(n, DLTensor{});
        bound.shape_storage.resize(n);
        bound.stride_storage.resize(n);
        bound.arrays.assign(n, tvm::ffi::Array<int64_t>{});

        int n_view = 0;
        for (size_t i = 0; i < n; i++) {
            auto const &a = bound.args[i];
            if (a.kind == CuteDslArgKind::ENV_STREAM) continue;
            n_view++;
            if (a.kind == CuteDslArgKind::ARRAY_I64) {
                // built once, here: the values are frozen into the artifact
                bound.arrays[i] = tvm::ffi::Array<int64_t>(a.values.begin(), a.values.end());
                continue;
            }
            if (a.kind != CuteDslArgKind::TENSOR && a.kind != CuteDslArgKind::WORKSPACE) continue;

            RETURN_CUDNN_FRONTEND_ERROR_IF(a.shape.size() != a.stride.size(),
                                           error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                           "CuTeDSL step '" + bound.label + "' argument " + std::to_string(i) +
                                               " has " + std::to_string(a.shape.size()) + " dims but " +
                                               std::to_string(a.stride.size()) + " strides.");

            bound.shape_storage[i]  = a.shape;
            bound.stride_storage[i] = a.stride;

            DLTensor &t = bound.prototypes[i];
            t.data      = nullptr;
            t.device    = DLDevice{kDLCUDA, device_ordinal_};
            t.ndim      = static_cast<int32_t>(a.shape.size());
            CHECK_CUDNN_FRONTEND_ERROR(cutedsl_detail::to_dl_data_type(a.data_type, t.dtype));
            t.shape       = bound.shape_storage[i].data();
            t.strides     = bound.stride_storage[i].data();
            t.byte_offset = 0;
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(n_view > 2047,
                                       error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                       "A tvm-ffi signature is limited to 2047 arguments; step '" + bound.label +
                                           "' has " + std::to_string(n_view) + ".");
        return {error_code_t::OK, ""};
    }

    CuteDslPayload payload_;

    // ffi::Module is a non-nullable object ref, so it is held in an optional
    // rather than default-constructed. Every step comes out of this one module,
    // unless the payload resolves them from the global table instead.
    std::optional<tvm::ffi::Module> module_;

    std::vector<BoundStep> steps_;

    int device_ordinal_ = 0;
    bool slots_bound_   = false;
};

namespace cutedsl_detail {

inline error_t
ffi_engine_factory(CuteDslPayload const &payload, std::shared_ptr<ICuteDslEngine> &out) {
    return CuteDslFfiEngine::create(payload, out);
}

// Installing on include is what makes the AOT surface opt-in: core FE holds a
// null factory and says so; including kernel_library.h fills it in.
struct FactoryInstaller {
    FactoryInstaller() { cutedsl_engine_factory() = &ffi_engine_factory; }
};

inline FactoryInstaller const &
install_factory() {
    static FactoryInstaller const installer;
    return installer;
}

namespace {
[[maybe_unused]] auto const &cutedsl_factory_installed = install_factory();
}

}  // namespace cutedsl_detail

}  // namespace experimental
}  // namespace cudnn_frontend
