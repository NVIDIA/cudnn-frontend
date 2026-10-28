/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Compiled kernels for cudnn_frontend::ops.
//
// The kernels are CuTeDSL, so Python compiles them; C++ never runs Python and
// never compiles. KernelLibrary::registry() holds the kernels Python
// registered in this process, each under its configuration (a JSON object) as
// the tvm-ffi global functions "cudnn.ops.kernel.<config>" and
// "cudnn.ops.meta.<config>" (the binding of the kernel's arguments, JSON). An
// op's create() looks its kernel up here.
//
// Calls go through the tvm-ffi C ABI, mirrored below, so no tvm-ffi header is
// needed; libtvm_ffi.so is whatever the Python process loaded.
//
// Linux only; elsewhere a lookup reports GRAPH_NOT_SUPPORTED.

#include "tensor_view.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace cudnn_frontend {
namespace ops {

class KernelLibrary;

namespace detail {

// ---------------------------------------------------------------------------
// The tvm-ffi C ABI (tvm/ffi/c_api.h, apache-tvm-ffi 0.1) as far as calling a
// compiled kernel needs it, and the DLPack tensor tensors are passed as.
// ---------------------------------------------------------------------------
namespace ffi {

enum TypeIndex : int32_t {
    kNone        = 0,
    kInt         = 1,
    kBool        = 2,
    kFloat       = 3,
    kOpaquePtr   = 4,
    kDLTensorPtr = 7,
    kSmallStr    = 11,
    kObject      = 64,  // first heap-object type index; everything >= is ref counted
    kStr         = 65,
    kError       = 67,
};

struct Object {
    uint64_t combined_ref_count;
    int32_t type_index;
    uint32_t padding;
    void (*deleter)(void *self, int flags);
};

struct Any {
    int32_t type_index;
    uint32_t small_len;  // a small string's length; zero otherwise
    union {
        int64_t v_int64;
        double v_float64;
        void const *v_ptr;
        Object *v_obj;
        char v_bytes[8];
    };
};

struct ByteArray {
    char const *data;
    size_t size;
};

struct ErrorCell {  // the fields after the Object header
    ByteArray kind;
    ByteArray message;
};

// DLPack's DLTensor (dlpack/dlpack.h).
struct DLDevice {
    int32_t device_type;  // kDLCUDA = 2
    int32_t device_id;
};
struct DLDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};
struct DLTensor {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
};

using GetGlobal      = int (*)(ByteArray const *name, void **out);
using FunctionCall   = int (*)(void *func, Any const *args, int32_t num_args, Any *result);
using DecRef         = int (*)(void *obj);
using MoveFromRaised = void (*)(void **result);

static_assert(sizeof(Object) == 24, "tvm-ffi Object header layout");
static_assert(sizeof(Any) == 16, "tvm-ffi Any layout");
static_assert(sizeof(DLTensor) == 48, "DLPack DLTensor layout");

}  // namespace ffi

inline ffi::Any
any_none() {
    return ffi::Any{};
}

inline ffi::Any
any_int(int64_t v) {
    ffi::Any a{};
    a.type_index = ffi::kInt;
    a.v_int64    = v;
    return a;
}

inline ffi::Any
any_bool(bool v) {
    ffi::Any a{};
    a.type_index = ffi::kBool;
    a.v_int64    = v;
    return a;
}

inline ffi::Any
any_float(double v) {
    ffi::Any a{};
    a.type_index = ffi::kFloat;
    a.v_float64  = v;
    return a;
}

inline ffi::Any
any_pointer(void const *p) {
    ffi::Any a{};
    a.type_index = ffi::kOpaquePtr;
    a.v_ptr      = p;
    return a;
}

inline ffi::Any
any_tensor(ffi::DLTensor const *t) {
    ffi::Any a{};
    a.type_index = ffi::kDLTensorPtr;
    a.v_ptr      = t;
    return a;
}

// The DLPack type of a cudnn data type; false for one without a DLPack equivalent.
inline bool
dl_dtype(DataType_t t, ffi::DLDataType &out) {
    switch (t) {
        case DataType_t::HALF:
            out = {2, 16, 1};
            return true;
        case DataType_t::BFLOAT16:
            out = {4, 16, 1};
            return true;
        case DataType_t::FLOAT:
            out = {2, 32, 1};
            return true;
        case DataType_t::INT8:
            out = {0, 8, 1};
            return true;
        case DataType_t::UINT8:
            out = {1, 8, 1};
            return true;
        case DataType_t::INT32:
            out = {0, 32, 1};
            return true;
        case DataType_t::INT64:
            out = {0, 64, 1};
            return true;
        case DataType_t::FP8_E4M3:
            out = {10, 8, 1};
            return true;
        case DataType_t::FP8_E5M2:
            out = {12, 8, 1};
            return true;
        case DataType_t::FP8_E8M0:
            out = {14, 8, 1};
            return true;
        default:
            return false;
    }
}

#if defined(__linux__)

// The tvm-ffi entry points this file calls, from the libtvm_ffi.so the process loaded.
struct Runtime {
    ffi::GetGlobal get_global            = nullptr;
    ffi::FunctionCall call               = nullptr;
    ffi::DecRef dec_ref                  = nullptr;
    ffi::MoveFromRaised move_from_raised = nullptr;
};

inline Runtime &
runtime() {
    static Runtime rt;
    return rt;
}

inline error_t
bind_runtime() {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    auto &rt = runtime();
    if (rt.call != nullptr) {
        return {error_code_t::OK, ""};
    }
    void *lib = ::dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
    RETURN_CUDNN_FRONTEND_ERROR_IF(lib == nullptr,
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "libtvm_ffi.so is not loaded: compile and register the kernels from Python first");
    Runtime r;
    r.get_global       = reinterpret_cast<ffi::GetGlobal>(::dlsym(lib, "TVMFFIFunctionGetGlobal"));
    r.call             = reinterpret_cast<ffi::FunctionCall>(::dlsym(lib, "TVMFFIFunctionCall"));
    r.dec_ref          = reinterpret_cast<ffi::DecRef>(::dlsym(lib, "TVMFFIObjectDecRef"));
    r.move_from_raised = reinterpret_cast<ffi::MoveFromRaised>(::dlsym(lib, "TVMFFIErrorMoveFromRaised"));
    RETURN_CUDNN_FRONTEND_ERROR_IF(!r.get_global || !r.call || !r.dec_ref || !r.move_from_raised,
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "libtvm_ffi.so does not export the tvm-ffi C API this build expects");
    rt = r;
    return {error_code_t::OK, ""};
}

// The pending tvm-ffi error as text, consumed.
inline std::string
take_error() {
    auto const &rt = runtime();
    void *err      = nullptr;
    rt.move_from_raised(&err);
    if (err == nullptr) {
        return "unknown tvm-ffi error";
    }
    std::string msg = "tvm-ffi error";
    if (static_cast<ffi::Object const *>(err)->type_index == ffi::kError) {
        auto const *cell =
            reinterpret_cast<ffi::ErrorCell const *>(static_cast<char const *>(err) + sizeof(ffi::Object));
        msg =
            std::string(cell->kind.data, cell->kind.size) + ": " + std::string(cell->message.data, cell->message.size);
    }
    rt.dec_ref(err);
    return msg;
}

// A registered global function's string result, or "" when there is none.
inline std::string
global_string(std::string const &name) {
    auto const &rt   = runtime();
    void *fn         = nullptr;
    ffi::ByteArray n = {name.data(), name.size()};
    std::string text;
    ffi::Any result{};
    if (rt.get_global(&n, &fn) != 0 || fn == nullptr) {
        return text;
    }
    if (rt.call(fn, nullptr, 0, &result) == 0) {
        if (result.type_index == ffi::kStr) {
            auto const *s = reinterpret_cast<ffi::ByteArray const *>(reinterpret_cast<char const *>(result.v_obj) +
                                                                     sizeof(ffi::Object));
            text.assign(s->data, s->size);
            rt.dec_ref(result.v_obj);
        } else if (result.type_index == ffi::kSmallStr) {
            text.assign(result.v_bytes, result.small_len);
        }
    }
    rt.dec_ref(fn);
    return text;
}

#endif  // __linux__

// One compiled kernel: a function Python registered in this process.
class Kernel {
   public:
    Kernel()               = default;
    Kernel(Kernel const &) = delete;
    Kernel &
    operator=(Kernel const &) = delete;
    ~Kernel() {
#if defined(__linux__)
        if (function_ != nullptr) {
            runtime().dec_ref(function_);
        }
#endif
    }

    json meta;  // how the kernel's arguments bind, as Python registered it

    // Calls the kernel; arguments are borrowed for the call.
    error_t
    call(ffi::Any const *args, int num_args) const {
#if defined(__linux__)
        ffi::Any result{};
        RETURN_CUDNN_FRONTEND_ERROR_IF(runtime().call(function_, args, num_args, &result) != 0,
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "cudnn_frontend::ops kernel failed: " + take_error());
        if (result.type_index >= ffi::kObject && result.v_obj != nullptr) {
            runtime().dec_ref(result.v_obj);
        }
        return {error_code_t::OK, ""};
#else
        (void)args;
        (void)num_args;
        return {error_code_t::GRAPH_NOT_SUPPORTED, "cudnn_frontend::ops kernels run on Linux only"};
#endif
    }

   private:
    friend class cudnn_frontend::ops::KernelLibrary;
    void *function_ = nullptr;  // owned reference
};

// The current device's ordinal, compute capability (major * 10 + minor) and SM count.
inline error_t
current_device(int &device, int &cc, int &num_sm) {
    int major = 0, minor = 0;
    using cudnn_frontend::detail::cuda_device_get_attribute;
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        cudnn_frontend::detail::cuda_get_device(&device) != cudaSuccess ||
            cuda_device_get_attribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cuda_device_get_attribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
            cuda_device_get_attribute(&num_sm, cudaDevAttrMultiProcessorCount, device) != cudaSuccess,
        error_code_t::INVALID_CUDA_DEVICE,
        "cannot query the current CUDA device");
    cc = major * 10 + minor;
    return {error_code_t::OK, ""};
}

}  // namespace detail

class KernelLibrary {
   public:
    // The kernels Python registered in this process.
    static KernelLibrary
    registry() {
        return KernelLibrary();
    }

    // The kernel for a configuration; an error naming it when there is none.
    error_t
    find(json const &config, std::shared_ptr<detail::Kernel const> &out) const {
        std::string const key = config.dump();
#if defined(__linux__)
        CHECK_CUDNN_FRONTEND_ERROR(detail::bind_runtime());
        auto const &rt           = detail::runtime();
        std::string const name   = "cudnn.ops.kernel." + key;
        void *fn                 = nullptr;
        detail::ffi::ByteArray n = {name.data(), name.size()};
        if (rt.get_global(&n, &fn) == 0 && fn != nullptr) {
            auto kernel       = std::make_shared<detail::Kernel>();
            kernel->function_ = fn;
            kernel->meta      = json::parse(detail::global_string("cudnn.ops.meta." + key), nullptr, false);
            RETURN_CUDNN_FRONTEND_ERROR_IF(kernel->meta.is_discarded() || !kernel->meta.is_object(),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "the registry has kernel " + key + " but no metadata for it");
            out = std::move(kernel);
            return {error_code_t::OK, ""};
        }
#endif
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "no kernel is registered for " + key + "; the Python op registers it on its second call"};
    }
};

}  // namespace ops
}  // namespace cudnn_frontend
