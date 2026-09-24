/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Executes a plan that the Python graph API compiled ahead of time.
//
// graph.serialize() of a graph whose selected plan is a CuTeDSL engine writes
// the compiled kernels (shared objects exported by the DSL) and the exact
// launch sequence the plan runs, with every argument that is not a buffer
// address already bound. Graph::deserialize() hands that payload here; execute
// patches this call's addresses into the frozen arguments and calls the
// kernels' tvm-ffi entry points. Nothing here needs Python, a JIT or the
// tvm-ffi headers: the tvm-ffi C ABI is mirrored below, and the two runtime
// libraries the exported kernels link against are resolved when the first such
// graph is deserialized:
//
//     libtvm_ffi.so           (apache-tvm-ffi)
//     libcute_dsl_runtime.so  (nvidia-cutlass-dsl, cu12/ or cu13/ lib directory)
//
// by SONAME, so they must be loaded already (the Python package preloads them)
// or be on the loader's search path (LD_LIBRARY_PATH, rpath).
//
// Linux only. Elsewhere, deserializing such a graph reports GRAPH_NOT_SUPPORTED.

#include "../graph_helpers.h"
#include "../../cudnn_frontend_shim.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace cudnn_frontend {
namespace experimental {
namespace aot {

// Bumped when the payload changes in a way an older reader cannot execute.
static constexpr int64_t AOT_FORMAT_VERSION = 1;

// ---------------------------------------------------------------------------
// The tvm-ffi C ABI, as far as calling an exported function needs it. Mirrors
// tvm/ffi/c_api.h (apache-tvm-ffi 0.1); that header documents the ABI as
// stable, and the asserts pin the layouts this relies on.
// ---------------------------------------------------------------------------
namespace ffi {

enum TypeIndex : int32_t {
    kNone   = 0,
    kInt    = 1,
    kBool   = 2,
    kFloat  = 3,
    kError  = 67,
    kArray  = 71,
    kObject = 64,  // first heap-object type index; everything >= is ref counted
};

struct Object {
    uint64_t combined_ref_count;
    int32_t type_index;
    uint32_t padding;
    void (*deleter)(void *self, int flags);
};

struct Any {
    int32_t type_index;
    uint32_t zero_padding;
    union {
        int64_t v_int64;
        double v_float64;
        void *v_ptr;
        Object *v_obj;
    };
};

struct ByteArray {
    char const *data;
    size_t size;
};

// The leading fields of TVMFFIErrorCell, which follows the Object header.
struct ErrorCell {
    ByteArray kind;
    ByteArray message;
};

using SafeCall       = int (*)(void *self, Any const *args, int32_t num_args, Any *result);
using GetGlobal      = int (*)(ByteArray const *name, void **out);
using FunctionCall   = int (*)(void *func, Any *args, int32_t num_args, Any *result);
using DecRef         = int (*)(void *obj);
using MoveFromRaised = void (*)(void **result);

static_assert(sizeof(Object) == 24, "tvm-ffi Object header layout");
static_assert(sizeof(Any) == 16, "tvm-ffi Any layout");

}  // namespace ffi

// ---------------------------------------------------------------------------
// Payload
// ---------------------------------------------------------------------------

// One positional argument of an exported kernel entry point. Only buffer
// addresses and the stream vary per call; everything else was bound when the
// plan was exported.
struct Arg {
    enum class Kind { INT, FLOAT, BOOL, NONE, STREAM, ARRAY, TENSOR, WORKSPACE };
    Kind kind      = Kind::NONE;
    int64_t i      = 0;           // INT, BOOL; TENSOR: uid
    double f       = 0.0;         // FLOAT
    int64_t offset = 0;           // TENSOR / WORKSPACE: byte offset from the base address
    std::vector<int64_t> values;  // ARRAY
    int slot = -1;                // TENSOR: index into the graph's pointer array (bound at load)
};

struct Step {
    enum class Kind { CALL, FILL32, FILL32_2D };
    Kind kind      = Kind::CALL;
    int64_t module = -1;    // CALL
    std::string symbol;     // CALL
    std::vector<Arg> args;  // CALL; FILL*: args[0] is the destination
    int64_t count  = 0;     // FILL32: words
    int64_t pitch  = 0;     // FILL32_2D: bytes between rows
    int64_t width  = 0;     // FILL32_2D: words per row
    int64_t height = 0;     // FILL32_2D: rows
    uint32_t word  = 0;     // FILL*
};

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB

inline error_t
parse_arg(json const &j, Arg &a) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        !j.is_object() || j.size() < 1, error_code_t::UNSUPPORTED_GRAPH_FORMAT, "AOT argument must be an object");
    if (j.contains("int")) {
        a.kind = Arg::Kind::INT;
        a.i    = j["int"].get<int64_t>();
    } else if (j.contains("float")) {
        a.kind = Arg::Kind::FLOAT;
        if (j["float"].is_string()) {  // JSON has no spelling for the non-finite values
            auto const v = j["float"].get<std::string>();
            a.f          = v == "nan"    ? std::numeric_limits<double>::quiet_NaN()
                           : v == "-inf" ? -std::numeric_limits<double>::infinity()
                                         : std::numeric_limits<double>::infinity();
        } else {
            a.f = j["float"].get<double>();
        }
    } else if (j.contains("bool")) {
        a.kind = Arg::Kind::BOOL;
        a.i    = j["bool"].get<bool>() ? 1 : 0;
    } else if (j.contains("none")) {
        a.kind = Arg::Kind::NONE;
    } else if (j.contains("stream")) {
        a.kind = Arg::Kind::STREAM;
    } else if (j.contains("array")) {
        a.kind   = Arg::Kind::ARRAY;
        a.values = j["array"].get<std::vector<int64_t>>();
    } else if (j.contains("tensor")) {
        a.kind   = Arg::Kind::TENSOR;
        a.i      = j["tensor"].get<int64_t>();
        a.offset = j.value("offset", static_cast<int64_t>(0));
    } else if (j.contains("workspace")) {
        a.kind   = Arg::Kind::WORKSPACE;
        a.offset = j["workspace"].get<int64_t>();
    } else {
        return {error_code_t::UNSUPPORTED_GRAPH_FORMAT, "Unknown AOT argument kind: " + j.dump()};
    }
    return {error_code_t::OK, ""};
}

inline error_t
parse_step(json const &j, Step &s) {
    auto const op = j.at("op").get<std::string>();
    if (op == "call") {
        s.kind   = Step::Kind::CALL;
        s.module = j.at("module").get<int64_t>();
        s.symbol = j.at("symbol").get<std::string>();
        for (auto const &a : j.at("args")) {
            Arg arg;
            CHECK_CUDNN_FRONTEND_ERROR(parse_arg(a, arg));
            s.args.push_back(std::move(arg));
        }
        return {error_code_t::OK, ""};
    }
    RETURN_CUDNN_FRONTEND_ERROR_IF(op != "fill32" && op != "fill32_2d",
                                   error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                   "Unknown AOT step '" + op + "'; this build understands call, fill32, fill32_2d.");
    Arg dst;
    CHECK_CUDNN_FRONTEND_ERROR(parse_arg(j.at("dst"), dst));
    RETURN_CUDNN_FRONTEND_ERROR_IF(dst.kind != Arg::Kind::TENSOR && dst.kind != Arg::Kind::WORKSPACE,
                                   error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                   "An AOT fill writes a tensor or the workspace.");
    s.args.push_back(std::move(dst));
    s.word = j.at("word").get<uint32_t>();
    if (op == "fill32") {
        s.kind  = Step::Kind::FILL32;
        s.count = j.at("count").get<int64_t>();
    } else {
        s.kind   = Step::Kind::FILL32_2D;
        s.pitch  = j.at("pitch").get<int64_t>();
        s.width  = j.at("width").get<int64_t>();
        s.height = j.at("height").get<int64_t>();
    }
    return {error_code_t::OK, ""};
}

#endif  // CUDNN_FRONTEND_SKIP_JSON_LIB

#if defined(__linux__)

namespace detail {

// The two runtime libraries every exported kernel links against, and the few
// tvm-ffi entry points this file calls itself. Resolved once per process.
struct Runtime {
    void *tvm_ffi                        = nullptr;
    void *cute_dsl_runtime               = nullptr;
    ffi::GetGlobal get_global            = nullptr;
    ffi::FunctionCall call               = nullptr;
    ffi::DecRef dec_ref                  = nullptr;
    ffi::MoveFromRaised move_from_raised = nullptr;
    void *array_ctor                     = nullptr;  // the "ffi.Array" global function
    std::string error;                               // non-empty when the runtime is unusable
};

inline Runtime
load_runtime() {
    Runtime r;
    // RTLD_GLOBAL: the exported kernels resolve their undefined symbols
    // against these, and a library already loaded under the same SONAME
    // (the Python package preloads both) is reused rather than loaded twice.
    r.tvm_ffi = ::dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    if (r.tvm_ffi == nullptr) {
        char const *e = ::dlerror();
        r.error       = std::string("cannot load libtvm_ffi.so (apache-tvm-ffi): ") + (e ? e : "unknown error");
        return r;
    }
    r.cute_dsl_runtime = ::dlopen("libcute_dsl_runtime.so", RTLD_NOW | RTLD_GLOBAL);
    if (r.cute_dsl_runtime == nullptr) {
        char const *e = ::dlerror();
        r.error = std::string("cannot load libcute_dsl_runtime.so (nvidia-cutlass-dsl): ") + (e ? e : "unknown error");
        return r;
    }
    r.get_global       = reinterpret_cast<ffi::GetGlobal>(::dlsym(r.tvm_ffi, "TVMFFIFunctionGetGlobal"));
    r.call             = reinterpret_cast<ffi::FunctionCall>(::dlsym(r.tvm_ffi, "TVMFFIFunctionCall"));
    r.dec_ref          = reinterpret_cast<ffi::DecRef>(::dlsym(r.tvm_ffi, "TVMFFIObjectDecRef"));
    r.move_from_raised = reinterpret_cast<ffi::MoveFromRaised>(::dlsym(r.tvm_ffi, "TVMFFIErrorMoveFromRaised"));
    if (!r.get_global || !r.call || !r.dec_ref || !r.move_from_raised) {
        r.error = "libtvm_ffi.so does not export the tvm-ffi C API this build expects";
        return r;
    }
    char const name[]  = "ffi.Array";
    ffi::ByteArray key = {name, sizeof(name) - 1};
    if (r.get_global(&key, &r.array_ctor) != 0 || r.array_ctor == nullptr) {
        r.error = "libtvm_ffi.so has no global function ffi.Array";
    }
    return r;
}

inline Runtime &
runtime_storage() {
    static Runtime rt;
    return rt;
}

// Loads the runtime once; a failure is reported and retried on the next call,
// so a process that loads the libraries after a failed attempt recovers.
inline std::string
ensure_runtime() {
    static std::mutex mu;
    static bool ready = false;
    std::lock_guard<std::mutex> lock(mu);
    if (ready) {
        return "";
    }
    Runtime r = load_runtime();
    if (!r.error.empty()) {
        return r.error;
    }
    runtime_storage() = r;
    ready             = true;
    return "";
}

// Valid once ensure_runtime() has succeeded, which every AotEngine did.
inline Runtime const &
runtime() {
    return runtime_storage();
}

// The pending tvm-ffi error, as text, consumed.
inline std::string
take_error() {
    auto const &rt = runtime();
    void *err      = nullptr;
    rt.move_from_raised(&err);
    if (err == nullptr) {
        return "unknown tvm-ffi error";
    }
    std::string msg = "tvm-ffi error";
    auto const *obj = static_cast<ffi::Object const *>(err);
    if (obj->type_index == ffi::kError) {
        auto const *cell =
            reinterpret_cast<ffi::ErrorCell const *>(static_cast<char const *>(err) + sizeof(ffi::Object));
        msg =
            std::string(cell->kind.data, cell->kind.size) + ": " + std::string(cell->message.data, cell->message.size);
    }
    rt.dec_ref(err);
    return msg;
}

// A loaded exported shared object. Loaded once per distinct content and kept
// for the life of the process: a CUDA graph captured from an execute keeps
// launching its kernels after every Graph that loaded them is gone, and a
// /proc/self/fd path that was dlclose'd could otherwise be handed back by the
// loader for a different module that reused the descriptor number.
struct Module {
    void *handle = nullptr;
    int fd       = -1;  // held open for the same reason
    std::vector<uint8_t> bytes;
};

inline uint64_t
fnv1a64(uint8_t const *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

// dlopen needs a path. An anonymous in-memory file gives it one without
// writing the kernel to a shared directory; a private temporary file (O_EXCL,
// mode 0600, a name unique in this process), unlinked right after the load, is
// the fallback where memfd is unavailable.
inline error_t
dlopen_bytes(std::vector<uint8_t> const &bytes, Module &m) {
    int fd = -1;
    std::string path;
    std::string tmp_path;
#if defined(SYS_memfd_create)
    fd = static_cast<int>(::syscall(SYS_memfd_create, "cudnn_frontend_aot", 1u /* MFD_CLOEXEC */));
    if (fd >= 0) {
        path = "/proc/self/fd/" + std::to_string(fd);
    }
#endif
    if (fd < 0) {
        static std::atomic<uint64_t> counter{0};
        char const *dir = std::getenv("TMPDIR");
        tmp_path        = std::string(dir ? dir : "/tmp") + "/cudnn_frontend_aot_" + std::to_string(::getpid()) + "_" +
                   std::to_string(counter.fetch_add(1)) + "_XXXXXX";
        std::vector<char> name(tmp_path.begin(), tmp_path.end());
        name.push_back('\0');
        fd = ::mkstemp(name.data());
        RETURN_CUDNN_FRONTEND_ERROR_IF(fd < 0,
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "Cannot create a file to load an AOT kernel from (memfd and mkstemp failed).");
        tmp_path = name.data();
        path     = tmp_path;
    }
    size_t done = 0;
    while (done < bytes.size()) {
        ssize_t const n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n <= 0) {
            break;
        }
        done += static_cast<size_t>(n);
    }
    std::string load_error = "short write";
    if (done == bytes.size()) {
        m.handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (m.handle == nullptr) {
            char const *e = ::dlerror();
            load_error    = e ? e : "unknown error";
        }
    }
    if (tmp_path.empty() && m.handle != nullptr) {
        m.fd = fd;
    } else {
        ::close(fd);
    }
    if (!tmp_path.empty()) {
        ::unlink(tmp_path.c_str());
    }
    RETURN_CUDNN_FRONTEND_ERROR_IF(m.handle == nullptr,
                                   error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                   "Cannot load an AOT kernel module: " + load_error);
    return {error_code_t::OK, ""};
}

inline error_t
load_module(std::vector<uint8_t> const &bytes, std::shared_ptr<Module> &out) {
    static std::mutex mu;
    static std::unordered_map<uint64_t, std::vector<std::shared_ptr<Module>>> loaded;
    uint64_t const key = fnv1a64(bytes.data(), bytes.size());

    std::lock_guard<std::mutex> lock(mu);
    auto &bucket = loaded[key];
    for (auto const &m : bucket) {
        if (m->bytes == bytes) {
            out = m;
            return {error_code_t::OK, ""};
        }
    }
    auto m   = std::make_shared<Module>();
    m->bytes = bytes;
    CHECK_CUDNN_FRONTEND_ERROR(dlopen_bytes(bytes, *m));
    bucket.push_back(m);
    out = std::move(m);
    return {error_code_t::OK, ""};
}

// An exported entry point. The CuTeDSL runtime initializes each function on
// its first call under one process-wide spinlock that concurrent first calls
// can deadlock on (nvidia-cutlass-dsl 4.7), so first calls are serialized
// here; once a function has run, calls go straight through.
struct Function {
    ffi::SafeCall call = nullptr;
    std::shared_ptr<Module> module;
    mutable std::atomic<bool> warm{false};
};

inline std::mutex &
first_call_mutex() {
    static std::mutex mu;
    return mu;
}

template <typename T>
inline T
driver_entry(char const *name) {
    return reinterpret_cast<T>(cudnn_frontend::detail::get_driver_entry_point(name));
}

}  // namespace detail

// The engine behind a deserialized AOT plan. Immutable once bound, so one
// graph may be executed from any number of threads.
class AotEngine {
   public:
    ~AotEngine() {
        if (!arrays_.empty()) {
            auto const &rt = detail::runtime();
            for (void *a : arrays_) {
                rt.dec_ref(a);
            }
        }
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    static error_t
    create(json const &payload, std::shared_ptr<AotEngine> &out) {
        auto engine = std::shared_ptr<AotEngine>(new AotEngine());
        CHECK_CUDNN_FRONTEND_ERROR(engine->initialize(payload));
        out = std::move(engine);
        return {error_code_t::OK, ""};
    }
#endif

    int64_t
    get_workspace_size() const {
        return workspace_size_;
    }

    // Every TENSOR argument's uid resolved to its slot in the graph's sorted
    // pointer array. Called once, after the variant-pack template exists.
    template <typename SlotOf>
    error_t
    bind_slots(SlotOf const &slot_of) {
        for (auto &step : steps_) {
            for (auto &a : step.args) {
                if (a.kind != Arg::Kind::TENSOR) {
                    continue;
                }
                a.slot = slot_of(a.i);
                RETURN_CUDNN_FRONTEND_ERROR_IF(a.slot < 0,
                                               error_code_t::INVALID_VARIANT_PACK,
                                               "The AOT plan addresses tensor uid " + std::to_string(a.i) +
                                                   ", which is not in this graph's variant pack.");
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    execute(void *const *ptrs, void *workspace, cudaStream_t stream) const {
        for (size_t s = 0; s < steps_.size(); s++) {
            auto const &step = steps_[s];
            if (step.kind == Step::Kind::CALL) {
                CHECK_CUDNN_FRONTEND_ERROR(call(s, ptrs, workspace, stream));
            } else {
                CHECK_CUDNN_FRONTEND_ERROR(fill(step, ptrs, workspace, stream));
            }
        }
        return {error_code_t::OK, ""};
    }

   private:
    AotEngine() = default;

    static void *
    address(Arg const &a, void *const *ptrs, void *workspace) {
        char *base = static_cast<char *>(a.kind == Arg::Kind::TENSOR ? ptrs[a.slot] : workspace);
        return base + a.offset;
    }

    error_t
    call(size_t s, void *const *ptrs, void *workspace, cudaStream_t stream) const {
        auto const &step  = steps_[s];
        auto const &fn    = *functions_[s];
        auto const &proto = protos_[s];
        int const n       = static_cast<int>(proto.size());

        constexpr int STACK_MAX = 48;
        ffi::Any stack_args[STACK_MAX];
        std::vector<ffi::Any> heap_args;
        ffi::Any *args = stack_args;
        if (n > STACK_MAX) {
            heap_args.resize(n);
            args = heap_args.data();
        }
        std::memcpy(static_cast<void *>(args), proto.data(), sizeof(ffi::Any) * n);
        for (int i = 0; i < n; i++) {
            auto const &a = step.args[i];
            if (a.kind == Arg::Kind::TENSOR || a.kind == Arg::Kind::WORKSPACE) {
                args[i].v_int64 = static_cast<int64_t>(reinterpret_cast<intptr_t>(address(a, ptrs, workspace)));
            } else if (a.kind == Arg::Kind::STREAM) {
                args[i].v_int64 = static_cast<int64_t>(reinterpret_cast<intptr_t>(stream));
            }
        }

        ffi::Any result{};
        int rc = 0;
        if (fn.warm.load(std::memory_order_acquire)) {
            rc = fn.call(nullptr, args, n, &result);
        } else {
            std::lock_guard<std::mutex> lock(detail::first_call_mutex());
            rc = fn.call(nullptr, args, n, &result);
            if (rc == 0) {
                fn.warm.store(true, std::memory_order_release);
            }
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(rc != 0,
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "AOT kernel '" + step.symbol + "' failed: " + detail::take_error());
        if (result.type_index >= ffi::kObject && result.v_obj != nullptr) {
            detail::runtime().dec_ref(result.v_obj);
        }
        return {error_code_t::OK, ""};
    }

    static error_t
    fill(Step const &step, void *const *ptrs, void *workspace, cudaStream_t stream) {
        using MemsetD32         = CUresult(CUDAAPI *)(CUdeviceptr, unsigned int, size_t, CUstream);
        using MemsetD2D32       = CUresult(CUDAAPI *)(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream);
        static auto const d32   = detail::driver_entry<MemsetD32>("cuMemsetD32Async");
        static auto const d2d32 = detail::driver_entry<MemsetD2D32>("cuMemsetD2D32Async");
        auto const dst          = reinterpret_cast<CUdeviceptr>(address(step.args[0], ptrs, workspace));
        auto const cu_stream    = reinterpret_cast<CUstream>(stream);
        CUresult rc             = CUDA_ERROR_NOT_SUPPORTED;
        if (step.kind == Step::Kind::FILL32 && d32 != nullptr) {
            rc = d32(dst, step.word, static_cast<size_t>(step.count), cu_stream);
        } else if (step.kind == Step::Kind::FILL32_2D && d2d32 != nullptr) {
            rc = d2d32(dst,
                       static_cast<size_t>(step.pitch),
                       step.word,
                       static_cast<size_t>(step.width),
                       static_cast<size_t>(step.height),
                       cu_stream);
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(rc != CUDA_SUCCESS,
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "AOT fill failed with CUresult " + std::to_string(static_cast<int>(rc)));
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    // The artifact runs only on the GPU it was compiled for: the cubins are
    // arch-specific SASS (no PTX), and the FROST kernels bake the SM count into
    // their persistent schedules. Checked here, before any kernel initializes,
    // because a failed initialization poisons the CuTeDSL runtime's lock.
    static error_t
    check_device(json const &target) {
        using DeviceGetAttribute   = CUresult(CUDAAPI *)(int *, CUdevice_attribute, CUdevice);
        static auto const get_attr = detail::driver_entry<DeviceGetAttribute>("cuDeviceGetAttribute");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            get_attr == nullptr, error_code_t::GRAPH_NOT_SUPPORTED, "Cannot query the device for an AOT plan.");
        int device = 0;
        RETURN_CUDNN_FRONTEND_ERROR_IF(cudnn_frontend::detail::cuda_get_device(&device) != cudaSuccess,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "No current CUDA device to load an AOT plan on.");
        int major = 0, minor = 0, sms = 0;
        get_attr(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        get_attr(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        get_attr(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        auto const cc  = target.at("compute_capability").get<std::vector<int>>();
        int const want = target.at("sm_count").get<int>();
        RETURN_CUDNN_FRONTEND_ERROR_IF(cc.size() != 2 || cc[0] != major || cc[1] != minor || want != sms,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "This AOT plan was compiled for compute capability " + std::to_string(cc.at(0)) +
                                           "." + std::to_string(cc.at(1)) + " with " + std::to_string(want) +
                                           " SMs; device " + std::to_string(device) + " is " + std::to_string(major) +
                                           "." + std::to_string(minor) + " with " + std::to_string(sms) + " SMs.");
        return {error_code_t::OK, ""};
    }

    error_t
    initialize(json const &payload) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !payload.is_object(), error_code_t::UNSUPPORTED_GRAPH_FORMAT, "AOT payload must be an object.");
        int64_t const format = payload.value("format", static_cast<int64_t>(0));
        RETURN_CUDNN_FRONTEND_ERROR_IF(format != AOT_FORMAT_VERSION,
                                       error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                       "AOT payload format " + std::to_string(format) + "; this build reads format " +
                                           std::to_string(AOT_FORMAT_VERSION) + ".");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            payload.value("abi", std::string()) != "tvm-ffi",
            error_code_t::UNSUPPORTED_GRAPH_FORMAT,
            "AOT payload ABI '" + payload.value("abi", std::string()) + "'; this build calls tvm-ffi entry points.");
        CHECK_CUDNN_FRONTEND_ERROR(check_device(payload.at("target")));

        std::string const runtime_error = detail::ensure_runtime();
        RETURN_CUDNN_FRONTEND_ERROR_IF(!runtime_error.empty(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "An AOT plan needs the CuTeDSL runtime libraries: " + runtime_error +
                                           ". Put their directories on LD_LIBRARY_PATH, or load them first.");

        workspace_size_ = payload.at("workspace_size").get<int64_t>();

        std::vector<std::shared_ptr<detail::Module>> modules;
        for (auto const &m : payload.at("modules")) {
            // UBJSON has no binary type: a module comes back as an array of bytes.
            RETURN_CUDNN_FRONTEND_ERROR_IF(!m.is_binary() && !m.is_array(),
                                           error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                           "AOT module must be a byte array.");
            std::vector<uint8_t> const bytes = m.is_binary() ? static_cast<std::vector<uint8_t> const &>(m.get_binary())
                                                             : m.get<std::vector<uint8_t>>();
            std::shared_ptr<detail::Module> loaded;
            CHECK_CUDNN_FRONTEND_ERROR(detail::load_module(bytes, loaded));
            modules.push_back(std::move(loaded));
        }

        for (auto const &js : payload.at("steps")) {
            Step step;
            CHECK_CUDNN_FRONTEND_ERROR(parse_step(js, step));
            std::shared_ptr<detail::Function> fn;
            std::vector<ffi::Any> proto;
            if (step.kind == Step::Kind::CALL) {
                RETURN_CUDNN_FRONTEND_ERROR_IF(step.module < 0 || step.module >= static_cast<int64_t>(modules.size()),
                                               error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                               "AOT step names module " + std::to_string(step.module) + " of " +
                                                   std::to_string(modules.size()) + ".");
                auto const &module      = modules[step.module];
                std::string const entry = "__tvm_ffi_" + step.symbol;
                fn                      = std::make_shared<detail::Function>();
                fn->module              = module;
                fn->call                = reinterpret_cast<ffi::SafeCall>(::dlsym(module->handle, entry.c_str()));
                RETURN_CUDNN_FRONTEND_ERROR_IF(fn->call == nullptr,
                                               error_code_t::UNSUPPORTED_GRAPH_FORMAT,
                                               "AOT module has no entry point " + entry + ".");
                for (auto const &a : step.args) {
                    ffi::Any v{};
                    v.type_index = ffi::kInt;  // addresses and the stream are passed as integers
                    switch (a.kind) {
                        case Arg::Kind::INT:
                            v.v_int64 = a.i;
                            break;
                        case Arg::Kind::BOOL:
                            v.type_index = ffi::kBool;
                            v.v_int64    = a.i;
                            break;
                        case Arg::Kind::FLOAT:
                            v.type_index = ffi::kFloat;
                            v.v_float64  = a.f;
                            break;
                        case Arg::Kind::NONE:
                            v.type_index = ffi::kNone;
                            break;
                        case Arg::Kind::ARRAY:
                            CHECK_CUDNN_FRONTEND_ERROR(make_array(a.values, v));
                            break;
                        case Arg::Kind::STREAM:
                        case Arg::Kind::TENSOR:
                        case Arg::Kind::WORKSPACE:
                            break;  // patched per call
                    }
                    proto.push_back(v);
                }
            }
            steps_.push_back(std::move(step));
            functions_.push_back(std::move(fn));
            protos_.push_back(std::move(proto));
        }
        return {error_code_t::OK, ""};
    }
#endif  // CUDNN_FRONTEND_SKIP_JSON_LIB

    // A tuple argument, built once: its values were bound at export.
    error_t
    make_array(std::vector<int64_t> const &values, ffi::Any &out) {
        auto const &rt = detail::runtime();
        std::vector<ffi::Any> items(values.size());
        for (size_t i = 0; i < values.size(); i++) {
            items[i]            = ffi::Any{};
            items[i].type_index = ffi::kInt;
            items[i].v_int64    = values[i];
        }
        ffi::Any result{};
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            rt.call(rt.array_ctor, items.data(), static_cast<int32_t>(items.size()), &result) != 0,
            error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
            "Cannot build an AOT tuple argument: " + detail::take_error());
        RETURN_CUDNN_FRONTEND_ERROR_IF(result.type_index != ffi::kArray,
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "ffi.Array returned type index " + std::to_string(result.type_index) + ".");
        arrays_.push_back(result.v_obj);
        out = result;
        return {error_code_t::OK, ""};
    }

    int64_t workspace_size_ = 0;
    std::vector<Step> steps_;
    std::vector<std::shared_ptr<detail::Function>> functions_;  // per step; null for fills
    std::vector<std::vector<ffi::Any>> protos_;                 // per step: constant args filled
    std::vector<void *> arrays_;                                // owned tuple arguments
};

#else  // !__linux__

class AotEngine {
   public:
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    static error_t
    create(json const &, std::shared_ptr<AotEngine> &) {
        return {error_code_t::GRAPH_NOT_SUPPORTED, "AOT plans can be loaded on Linux only."};
    }
#endif
    int64_t
    get_workspace_size() const {
        return 0;
    }
    template <typename SlotOf>
    error_t
    bind_slots(SlotOf const &) {
        return {error_code_t::GRAPH_NOT_SUPPORTED, "AOT plans can be loaded on Linux only."};
    }
    error_t
    execute(void *const *, void *, cudaStream_t) const {
        return {error_code_t::GRAPH_NOT_SUPPORTED, "AOT plans can be loaded on Linux only."};
    }
};

#endif  // __linux__

}  // namespace aot
}  // namespace experimental
}  // namespace cudnn_frontend
