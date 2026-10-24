/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Header-only C++ API over libcudnn_frontend_aot_sm120.so (ahead-of-time compiled sm_120a attention forward).
 * The library is dlopened; nothing here needs cuDNN, Python or a runtime compiler. See
 * docs/fe-oss-apis/aot_sm120_attention.md.
 *
 *   namespace aot = cudnn_frontend::experimental::sm120_aot;
 *   aot::SdpaDesc desc = aot::make_desc();            // head_dim, dtypes, mask, sinks, gqa_ratio
 *   aot::SdpaPlan plan;
 *   if (plan.build(desc) != CUDNN_FE_AOT_STATUS_SUCCESS) { ... plan.last_error() ... }   // once per shape class
 *   aot::SdpaArgs args = aot::make_args();            // extents, byte strides, pointers
 *   plan.execute(args, workspace, workspace_size, stream);                               // launches only
 *
 * Library lookup: the path given to Library::load(), else $CUDNN_FRONTEND_AOT_SM120_LIB, else
 * "libcudnn_frontend_aot_sm120.so" on the dynamic loader's search path.
 */

#pragma once

#include "sm120_aot_attention_abi.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>

namespace cudnn_frontend {
namespace experimental {
namespace sm120_aot {

using Status    = cudnn_fe_aot_status_t;
using SdpaDesc  = cudnn_fe_aot_sdpa_desc_t;
using SdpaArgs  = cudnn_fe_aot_sdpa_args_t;
using SdpaKnobs = cudnn_fe_aot_sdpa_knobs_t;

inline SdpaDesc
make_desc() {
    SdpaDesc d{};
    d.struct_size = sizeof(d);
    d.q_dtype = d.kv_dtype = d.o_dtype = CUDNN_FE_AOT_DTYPE_F16;
    d.mask                             = CUDNN_FE_AOT_MASK_NONE;
    d.window_left                      = -1;
    return d;
}

inline SdpaArgs
make_args() {
    SdpaArgs a{};
    a.struct_size = sizeof(a);
    return a;
}

inline SdpaKnobs
make_knobs() {
    SdpaKnobs k{};
    k.struct_size = sizeof(k);
    return k;
}

/* The dlopened library and its entry points. Thread-safe to load once and share. */
class Library {
   public:
    using AbiVersionFn   = uint32_t (*)();
    using BuildInfoFn    = const char* (*)();
    using PlanCreateFn   = Status (*)(const SdpaDesc*, const SdpaKnobs*, int, cudnn_fe_aot_sdpa_plan_t**);
    using PlanDestroyFn  = void (*)(cudnn_fe_aot_sdpa_plan_t*);
    using WorkspaceFn    = Status (*)(const cudnn_fe_aot_sdpa_plan_t*, const SdpaArgs*, size_t*);
    using ExecuteFn      = Status (*)(const cudnn_fe_aot_sdpa_plan_t*, const SdpaArgs*, void*, size_t, cudaStream_t);
    using QueryFn        = Status (*)(const cudnn_fe_aot_sdpa_plan_t*, const SdpaArgs*, SdpaKnobs*, const char**);
    using NumKnobSetsFn  = int32_t (*)(const SdpaDesc*);
    using GetKnobSetFn   = Status (*)(const SdpaDesc*, int32_t, SdpaKnobs*);
    using StatusStringFn = const char* (*)(Status);
    using LastErrorFn    = const char* (*)();

    Library()               = default;
    Library(const Library&) = delete;
    Library&
    operator=(const Library&) = delete;
    ~Library() = default;  // the handle is deliberately not closed: plans and launched kernels may outlive this object

    /* Loads once; later calls return the first result. */
    Status
    load(const char* path = nullptr) {
        std::call_once(once_, [&] { status_ = do_load(path); });
        return status_;
    }

    bool
    loaded() const {
        return status_ == CUDNN_FE_AOT_STATUS_SUCCESS && handle_ != nullptr;
    }
    const std::string&
    error() const {
        return error_;
    }
    const char*
    build_info() const {
        return loaded() ? build_info_() : "";
    }

    /* The process-wide instance SdpaPlan uses by default, loaded on first use. */
    static Library&
    instance() {
        static Library lib;
        return lib;
    }

    PlanCreateFn plan_create     = nullptr;
    PlanDestroyFn plan_destroy   = nullptr;
    WorkspaceFn workspace_size   = nullptr;
    ExecuteFn execute            = nullptr;
    QueryFn query                = nullptr;
    NumKnobSetsFn num_knob_sets  = nullptr;
    GetKnobSetFn get_knob_set    = nullptr;
    StatusStringFn status_string = nullptr;
    LastErrorFn last_error       = nullptr;

   private:
#ifdef _WIN32
    using Handle = HMODULE;
    static Handle
    open(const char* p) {
        return LoadLibraryA(p);
    }
    static void*
    sym(Handle h, const char* n) {
        return reinterpret_cast<void*>(GetProcAddress(h, n));
    }
    static std::string
    dl_error() {
        return "Windows error " + std::to_string(GetLastError());
    }
    static constexpr const char* kDefaultName = "cudnn_frontend_aot_sm120.dll";
#else
    using Handle = void*;
    static Handle
    open(const char* p) {
        return dlopen(p, RTLD_NOW | RTLD_LOCAL);
    }
    static void*
    sym(Handle h, const char* n) {
        return dlsym(h, n);
    }
    static std::string
    dl_error() {
        const char* e = dlerror();
        return e ? e : "unknown error";
    }
    static constexpr const char* kDefaultName = "libcudnn_frontend_aot_sm120.so";
#endif

    template <class Fn>
    bool
    resolve(Fn* out, const char* name) {
        *out = reinterpret_cast<Fn>(sym(handle_, name));
        if (!*out) error_ = std::string("symbol ") + name + " not found: " + dl_error();
        return *out != nullptr;
    }

    Status
    do_load(const char* path) {
        if (!path || !*path) {
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
            const char* env = std::getenv("CUDNN_FRONTEND_AOT_SM120_LIB");
#ifdef _WIN32
#pragma warning(pop)
#endif
            path = (env && *env) ? env : kDefaultName;
        }
        handle_ = open(path);
        if (!handle_) {
            error_ = std::string("unable to load ") + path + ": " + dl_error();
            return CUDNN_FE_AOT_STATUS_LIBRARY_NOT_LOADED;
        }
        AbiVersionFn abi_version = nullptr;
        if (!resolve(&abi_version, "cudnn_fe_aot_sm120_abi_version") ||
            !resolve(&build_info_, "cudnn_fe_aot_sm120_build_info") ||
            !resolve(&plan_create, "cudnn_fe_aot_sm120_sdpa_plan_create") ||
            !resolve(&plan_destroy, "cudnn_fe_aot_sm120_sdpa_plan_destroy") ||
            !resolve(&workspace_size, "cudnn_fe_aot_sm120_sdpa_workspace_size") ||
            !resolve(&execute, "cudnn_fe_aot_sm120_sdpa_execute") ||
            !resolve(&query, "cudnn_fe_aot_sm120_sdpa_query") ||
            !resolve(&num_knob_sets, "cudnn_fe_aot_sm120_sdpa_num_knob_sets") ||
            !resolve(&get_knob_set, "cudnn_fe_aot_sm120_sdpa_get_knob_set") ||
            !resolve(&status_string, "cudnn_fe_aot_sm120_status_string") ||
            !resolve(&last_error, "cudnn_fe_aot_sm120_last_error")) {
            handle_ = nullptr;
            return CUDNN_FE_AOT_STATUS_LIBRARY_NOT_LOADED;
        }
        if (abi_version() < CUDNN_FE_AOT_SM120_ABI_VERSION) {
            error_ = std::string(path) + " has ABI version " + std::to_string(abi_version()) + "; this header needs " +
                     std::to_string(CUDNN_FE_AOT_SM120_ABI_VERSION);
            handle_ = nullptr;
            return CUDNN_FE_AOT_STATUS_ABI_MISMATCH;
        }
        return CUDNN_FE_AOT_STATUS_SUCCESS;
    }

    std::once_flag once_;
    Status status_          = CUDNN_FE_AOT_STATUS_LIBRARY_NOT_LOADED;
    Handle handle_          = nullptr;
    BuildInfoFn build_info_ = nullptr;
    std::string error_;
};

/* One shape class. Building it loads every kernel it can launch; execute() only launches. Immutable once built, so
   one plan may be executed concurrently from several threads and streams (each call with its own workspace). */
class SdpaPlan {
   public:
    SdpaPlan()                = default;
    SdpaPlan(const SdpaPlan&) = delete;
    SdpaPlan&
    operator=(const SdpaPlan&) = delete;
    SdpaPlan(SdpaPlan&& o) noexcept
        : lib_(o.lib_), plan_(std::exchange(o.plan_, nullptr)), error_(std::move(o.error_)) {}
    SdpaPlan&
    operator=(SdpaPlan&& o) noexcept {
        if (this != &o) {
            reset();
            lib_   = o.lib_;
            plan_  = std::exchange(o.plan_, nullptr);
            error_ = std::move(o.error_);
        }
        return *this;
    }
    ~SdpaPlan() { reset(); }

    /* `knobs` pins one prebuilt knob set (see knob_sets); NULL uses the built-in selection table. device -1 = current.
     */
    Status
    build(const SdpaDesc& desc, const SdpaKnobs* knobs = nullptr, int device = -1, Library* lib = nullptr) {
        reset();
        lib_ = lib ? lib : &Library::instance();
        if (Status s = lib_->load()) {
            error_ = lib_->error();
            return s;
        }
        const Status s = lib_->plan_create(&desc, knobs, device, &plan_);
        if (s) error_ = lib_->last_error();
        return s;
    }

    bool
    valid() const {
        return plan_ != nullptr;
    }

    Status
    workspace_size(const SdpaArgs& args, size_t* bytes) const {
        return plan_ ? lib_->workspace_size(plan_, &args, bytes) : CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT;
    }

    /* Launches on `stream`; no allocation, synchronization or code loading. On failure, last_error() says why. */
    Status
    execute(const SdpaArgs& args, void* workspace, size_t workspace_size, cudaStream_t stream) const {
        return plan_ ? lib_->execute(plan_, &args, workspace, workspace_size, stream)
                     : CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT;
    }

    /* The knob set and kernel execute() would launch for these extents. */
    Status
    query(const SdpaArgs& args, SdpaKnobs* knobs, const char** kernel_name) const {
        return plan_ ? lib_->query(plan_, &args, knobs, kernel_name) : CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT;
    }

    /* The library's message for this thread's last failed call, or the build()/loader message. */
    std::string
    last_error() const {
        if (lib_ && lib_->loaded() && *lib_->last_error()) return lib_->last_error();
        return error_;
    }

   private:
    void
    reset() {
        if (plan_ && lib_) lib_->plan_destroy(plan_);
        plan_ = nullptr;
    }

    Library* lib_                   = nullptr;
    cudnn_fe_aot_sdpa_plan_t* plan_ = nullptr;
    std::string error_;
};

/* The prebuilt knob sets serving a shape class, to tune over (build one SdpaPlan per set). */
inline Status
knob_sets(const SdpaDesc& desc, int32_t* count, SdpaKnobs* out, int32_t capacity, Library* lib = nullptr) {
    Library& l = lib ? *lib : Library::instance();
    if (Status s = l.load()) return s;
    *count = l.num_knob_sets(&desc);
    for (int32_t i = 0; out && i < *count && i < capacity; ++i)
        if (Status s = l.get_knob_set(&desc, i, &out[i])) return s;
    return CUDNN_FE_AOT_STATUS_SUCCESS;
}

}  // namespace sm120_aot
}  // namespace experimental
}  // namespace cudnn_frontend
