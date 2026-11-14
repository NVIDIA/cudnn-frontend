/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Launch plans (python/cudnn/launch_plan.py): the fills and kernel launches one
// planned Python call made, replayed from C++. Python records the call and binds
// every kernel argument to a slot (a buffer the caller passes per call: an
// argument tensor, a returned tensor, scratch), the stream, or a constant; a
// later call passes only the slots' addresses and the stream. Nothing is bound
// by an address of the recorded call.
//
// Kernels are tvm-ffi functions, called through the tvm-ffi C ABI mirrored below
// (libtvm_ffi.so is whatever the Python process loaded, and has to report the
// version the mirror was written for); fills go through the CUDA driver. Python
// hands the functions over through a tvm-ffi call (the global function
// "cudnn.launch_plan.hold"), so a plan owns a reference to each. Linux only.

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "cudnn_frontend.h"

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace py = nanobind;

namespace cudnn_frontend {
namespace python_bindings {
namespace {

// The tvm-ffi C ABI (tvm/ffi/c_api.h, apache-tvm-ffi 0.1) as far as calling and holding a function needs it, and
// DLPack's DLTensor.
namespace ffi {

constexpr uint32_t VERSION_MAJOR = 0, VERSION_MINOR = 1;  // the ABI mirrored here

enum TypeIndex : int32_t {
    kNone        = 0,
    kInt         = 1,
    kBool        = 2,
    kFloat       = 3,
    kOpaquePtr   = 4,
    kDLTensorPtr = 7,
    kObject      = 64,  // first heap-object type index; everything >= is ref counted
    kError       = 67,
    kFunction    = 68,
};

struct Version {
    uint32_t major_number, minor_number, patch_number;
};

struct Object {
    uint64_t combined_ref_count;
    int32_t type_index;
    uint32_t padding;
    void (*deleter)(void *self, int flags);
};

struct Any {
    int32_t type_index;
    uint32_t small_len;
    union {
        int64_t v_int64;
        double v_float64;
        void const *v_ptr;
        Object *v_obj;
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

static_assert(sizeof(Object) == 24, "tvm-ffi Object header layout");
static_assert(sizeof(Any) == 16, "tvm-ffi Any layout");
static_assert(sizeof(DLTensor) == 48, "DLPack DLTensor layout");

}  // namespace ffi

// The tvm-ffi and CUDA driver entry points a plan calls, resolved once.
struct Runtime {
    int (*call)(void *func, ffi::Any const *args, int32_t num_args, ffi::Any *result) = nullptr;
    int (*inc_ref)(void *obj)                                                         = nullptr;
    int (*dec_ref)(void *obj)                                                         = nullptr;
    void (*move_from_raised)(void **result)                                           = nullptr;
    void (*set_raised)(char const *kind, char const *message)                         = nullptr;
    int (*memset8)(uint64_t dst, unsigned char value, size_t count, void *stream)     = nullptr;
    int (*memset16)(uint64_t dst, unsigned short value, size_t count, void *stream)   = nullptr;
    int (*memset32)(uint64_t dst, unsigned int value, size_t count, void *stream)     = nullptr;
};

Runtime const &
runtime();

// tvm-ffi functions Python handed over for one plan, each with a reference of ours.
struct Functions {
    std::vector<void *> handles;
    Functions() = default;
    Functions(Functions &&other) noexcept : handles(std::move(other.handles)) { other.handles.clear(); }
    Functions(Functions const &) = delete;
    Functions &
    operator=(Functions const &) = delete;
    Functions &
    operator=(Functions &&other) noexcept {
        std::swap(handles, other.handles);
        return *this;
    }
    ~Functions() {
        for (void *handle : handles) {
            runtime().dec_ref(handle);
        }
    }
};

std::mutex held_mutex;
std::unordered_map<int64_t, Functions> held;  // token -> the functions of a plan about to be made
int64_t next_token = 1;

// "cudnn.launch_plan.hold"(function, ...) -> token: takes a reference to each function for the plan made with the
// token.
[[maybe_unused]] int
hold(void *, ffi::Any const *args, int32_t num_args, ffi::Any *result) {
    auto const &rt = runtime();
    for (int32_t i = 0; i < num_args; i++) {
        if (args[i].type_index != ffi::kFunction || args[i].v_obj == nullptr) {
            rt.set_raised("TypeError", "cudnn.launch_plan.hold takes tvm-ffi functions");
            return -1;
        }
    }
    try {
        Functions functions;
        functions.handles.reserve(static_cast<size_t>(num_args));
        for (int32_t i = 0; i < num_args; i++) {
            rt.inc_ref(args[i].v_obj);
            functions.handles.push_back(args[i].v_obj);
        }
        std::lock_guard<std::mutex> lock(held_mutex);
        held.emplace(next_token, std::move(functions));
        result->type_index = ffi::kInt;
        result->v_int64    = next_token++;
    } catch (std::exception const &e) {
        rt.set_raised("RuntimeError", e.what());
        return -1;
    }
    return 0;
}

Runtime const &
runtime() {
    static Runtime const rt = [] {
        Runtime r;
#if defined(__linux__)
        void *lib = ::dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
        if (lib == nullptr) {
            throw std::runtime_error("cudnn launch plan: libtvm_ffi.so is not loaded");
        }
        auto const symbol = [lib](char const *name) {
            void *address = ::dlsym(lib, name);
            if (address == nullptr) {
                throw std::runtime_error(std::string("cudnn launch plan: libtvm_ffi.so does not export ") + name);
            }
            return address;
        };
        ffi::Version version{};
        reinterpret_cast<void (*)(ffi::Version *)>(symbol("TVMFFIGetVersion"))(&version);
        if (version.major_number != ffi::VERSION_MAJOR || version.minor_number != ffi::VERSION_MINOR) {
            throw std::runtime_error("cudnn launch plan: written for the tvm-ffi 0.1 C ABI; libtvm_ffi.so is " +
                                     std::to_string(version.major_number) + "." + std::to_string(version.minor_number) +
                                     "." + std::to_string(version.patch_number));
        }
        r.call             = reinterpret_cast<decltype(r.call)>(symbol("TVMFFIFunctionCall"));
        r.inc_ref          = reinterpret_cast<decltype(r.inc_ref)>(symbol("TVMFFIObjectIncRef"));
        r.dec_ref          = reinterpret_cast<decltype(r.dec_ref)>(symbol("TVMFFIObjectDecRef"));
        r.move_from_raised = reinterpret_cast<decltype(r.move_from_raised)>(symbol("TVMFFIErrorMoveFromRaised"));
        r.set_raised       = reinterpret_cast<decltype(r.set_raised)>(symbol("TVMFFIErrorSetRaisedFromCStr"));
        auto const create  = reinterpret_cast<int (*)(void *, decltype(&hold), void (*)(void *), void **)>(
            symbol("TVMFFIFunctionCreate"));
        auto const set_global =
            reinterpret_cast<int (*)(ffi::ByteArray const *, void *, int)>(symbol("TVMFFIFunctionSetGlobal"));
        char const name[]        = "cudnn.launch_plan.hold";
        ffi::ByteArray const key = {name, sizeof(name) - 1};
        void *function           = nullptr;
        if (create(nullptr, &hold, [](void *) {}, &function) != 0 || set_global(&key, function, 1) != 0) {
            throw std::runtime_error("cudnn launch plan: could not register cudnn.launch_plan.hold with tvm-ffi");
        }
        r.dec_ref(function);
        using cudnn_frontend::detail::CudaLibrary;
        using cudnn_frontend::detail::get_cuda_symbol;
        r.memset8  = reinterpret_cast<decltype(r.memset8)>(get_cuda_symbol(CudaLibrary::CUDA, "cuMemsetD8Async"));
        r.memset16 = reinterpret_cast<decltype(r.memset16)>(get_cuda_symbol(CudaLibrary::CUDA, "cuMemsetD16Async"));
        r.memset32 = reinterpret_cast<decltype(r.memset32)>(get_cuda_symbol(CudaLibrary::CUDA, "cuMemsetD32Async"));
#else
        throw std::runtime_error("cudnn launch plans run on Linux only");
#endif
        return r;
    }();
    return rt;
}

// The pending tvm-ffi error as text, consumed.
std::string
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

ffi::Any
any(int32_t type_index, int64_t v) {
    ffi::Any a{};
    a.type_index = type_index;
    a.v_int64    = v;
    return a;
}

class LaunchPlan {
   public:
    static constexpr size_t MAX_ARGS = 128;
    static constexpr int NULL_SLOT   = -1;  // a slot index meaning the null address

    // The plan as launch_plan.py writes it:
    //   {"device": d, "align": [bytes per slot], "extent": [bytes per argument slot],
    //    "overlaps": [[i, j, byte offset of i from j], ...],
    //    "steps": [step, ...]}, a step being
    //   {"memset": {"slot": s, "offset": o, "width": 1 | 2 | 4, "value": v, "count": n}}  or
    //   {"function": i, "args": [arg, ...]}, an arg being
    //     null, {"int": v}, {"float": v}, {"bool": v}                  constants
    //     {"stream": true}                                             the stream
    //     {"slot": s, "offset": o, "as": "tensor" | "int" | "pointer",
    //      "dim": [...], "stride": [...], "dtype": [code, bits, lanes]} an address in a slot; slot -1 is the null
    //      address
    // ``i`` indexes the functions "cudnn.launch_plan.hold" took for ``token``.
    LaunchPlan(std::string const &text, int64_t token) {
        runtime();
        {
            std::lock_guard<std::mutex> lock(held_mutex);
            auto const found = held.find(token);
            if (found == held.end()) {
                throw std::invalid_argument("cudnn launch plan: no functions held for this plan");
            }
            functions_ = std::move(found->second);
            held.erase(found);
        }
        json const plan = json::parse(text);
        device_         = plan.at("device").get<int>();
        for (auto const &a : plan.at("align")) {
            align_.push_back(std::max<int64_t>(1, a.get<int64_t>()));
        }
        for (auto const &e : plan.at("extent")) {
            extent_.push_back(e.get<uint64_t>());
        }
        overlaps_.assign(extent_.size() * extent_.size(), false);
        delta_.assign(extent_.size() * extent_.size(), 0);
        for (auto const &pair : plan.at("overlaps")) {
            size_t const i = pair.at(0).get<size_t>(), j = pair.at(1).get<size_t>();
            if (i >= extent_.size() || j >= extent_.size()) {
                throw std::invalid_argument("cudnn launch plan: no argument slot " + pair.dump());
            }
            overlaps_[i * extent_.size() + j] = true;
            delta_[i * extent_.size() + j]    = pair.at(2).get<int64_t>();
        }
        for (auto const &s : plan.at("steps")) {
            Step step;
            if (s.contains("memset")) {
                auto const &m = s.at("memset");
                step.memset   = true;
                step.target   = place(m);
                step.width    = m.at("width").get<int>();
                step.value    = m.at("value").get<uint32_t>();
                step.count    = m.at("count").get<uint64_t>();
                steps_.push_back(std::move(step));
                continue;
            }
            step.function       = functions_.handles.at(s.at("function").get<size_t>());
            auto const &records = s.at("args");
            if (records.size() > MAX_ARGS) {
                throw std::invalid_argument("cudnn launch plan: a kernel takes more than 128 arguments");
            }
            step.proto.assign(records.size(), ffi::Any{});
            for (size_t i = 0; i < records.size(); i++) {
                auto const &a = records[i];
                if (a.is_null()) continue;
                if (a.contains("int")) {
                    step.proto[i] = any(ffi::kInt, a.at("int").get<int64_t>());
                } else if (a.contains("bool")) {
                    step.proto[i] = any(ffi::kBool, a.at("bool").get<bool>());
                } else if (a.contains("float")) {
                    step.proto[i]           = any(ffi::kFloat, 0);
                    step.proto[i].v_float64 = a.at("float").get<double>();
                } else if (a.contains("stream")) {
                    step.binds.push_back(Bind{As::STREAM, static_cast<int>(i)});
                } else {
                    Bind b          = place(a);
                    b.arg           = static_cast<int>(i);
                    auto const kind = a.at("as").get<std::string>();
                    b.as            = kind == "int" ? As::INT : kind == "pointer" ? As::POINTER : As::TENSOR;
                    if (b.as == As::TENSOR) {
                        auto const &dim = a.at("dim"), &stride = a.at("stride"), &dtype = a.at("dtype");
                        if (dim.size() > 8 || stride.size() != dim.size()) {
                            throw std::invalid_argument("cudnn launch plan: malformed tensor argument " + a.dump());
                        }
                        b.rank = static_cast<int>(dim.size());
                        for (int k = 0; k < b.rank; k++) {
                            b.dim[k]    = dim[k].get<int64_t>();
                            b.stride[k] = stride[k].get<int64_t>();
                        }
                        b.dtype = {dtype[0].get<uint8_t>(), dtype[1].get<uint8_t>(), dtype[2].get<uint16_t>()};
                    }
                    step.binds.push_back(b);
                }
            }
            steps_.push_back(std::move(step));
        }
    }

    // The slots' addresses, in the plan's order; false (nothing ran) when one is less aligned than the recorded call's
    // or the argument tensors overlap each other differently (overlapping ones at another byte offset from each other).
    bool
    execute(py::list addresses, intptr_t stream_handle) const {
        if (py::len(addresses) != align_.size()) {
            throw std::invalid_argument("cudnn launch plan: one address per slot");
        }
        uintptr_t slots[MAX_ARGS];
        if (align_.size() > MAX_ARGS) {
            throw std::invalid_argument("cudnn launch plan: more than 128 slots");
        }
        for (size_t i = 0; i < align_.size(); i++) {
            slots[i] = static_cast<uintptr_t>(py::cast<intptr_t>(addresses[i]));
            if (slots[i] % static_cast<uintptr_t>(align_[i]) != 0) {
                return false;
            }
        }
        size_t const n = extent_.size();
        for (size_t i = 1; i < n; i++) {
            for (size_t j = 0; j < i; j++) {
                bool const overlap =
                    extent_[i] && extent_[j] && slots[i] < slots[j] + extent_[j] && slots[j] < slots[i] + extent_[i];
                if (overlap != overlaps_[i * n + j] ||
                    (overlap && static_cast<int64_t>(slots[i] - slots[j]) != delta_[i * n + j])) {
                    return false;
                }
            }
        }
        auto const &rt = runtime();
        void *stream   = reinterpret_cast<void *>(stream_handle);
        ffi::Any args[MAX_ARGS];
        ffi::DLTensor tensors[MAX_ARGS];
        for (auto const &step : steps_) {
            if (step.memset) {
                uint64_t const dst = address(step.target, slots);
                int const status   = step.width == 4 ? rt.memset32(dst, step.value, step.count, stream)
                                     : step.width == 2
                                         ? rt.memset16(dst, static_cast<unsigned short>(step.value), step.count, stream)
                                         : rt.memset8(dst, static_cast<unsigned char>(step.value), step.count, stream);
                if (status != 0) {
                    throw std::runtime_error("cudnn launch plan: a fill failed with CUDA error " +
                                             std::to_string(status));
                }
                continue;
            }
            std::copy(step.proto.begin(), step.proto.end(), args);
            int t = 0;
            for (auto const &b : step.binds) {
                if (b.as == As::STREAM) {
                    args[b.arg] = any(ffi::kOpaquePtr, stream_handle);
                    continue;
                }
                uint64_t const p = address(b, slots);
                if (b.as == As::TENSOR) {
                    auto &d           = tensors[t++];
                    d                 = {};
                    d.data            = reinterpret_cast<void *>(p);
                    d.device          = {2, device_};
                    d.ndim            = b.rank;
                    d.dtype           = b.dtype;
                    d.shape           = const_cast<int64_t *>(b.dim);
                    d.strides         = const_cast<int64_t *>(b.stride);
                    args[b.arg]       = any(ffi::kDLTensorPtr, 0);
                    args[b.arg].v_ptr = &d;
                } else {
                    args[b.arg] = any(b.as == As::INT ? ffi::kInt : ffi::kOpaquePtr, static_cast<int64_t>(p));
                }
            }
            ffi::Any result{};
            if (rt.call(step.function, args, static_cast<int32_t>(step.proto.size()), &result) != 0) {
                throw std::runtime_error("cudnn launch plan: a kernel failed: " + take_error());
            }
            if (result.type_index >= ffi::kObject && result.v_obj != nullptr) {
                rt.dec_ref(result.v_obj);
            }
        }
        return true;
    }

   private:
    enum class As { TENSOR, INT, POINTER, STREAM };

    struct Bind {
        As as          = As::TENSOR;
        int arg        = 0;
        int slot       = NULL_SLOT;
        int64_t offset = 0;  // bytes from the slot's address
        int rank       = 0;
        ffi::DLDataType dtype{};
        int64_t dim[8]{}, stride[8]{};
    };

    struct Step {
        bool memset = false;
        Bind target;  // memset: where
        int width      = 1;
        uint32_t value = 0;
        uint64_t count = 0;
        void *function = nullptr;  // one of functions_
        std::vector<ffi::Any> proto;
        std::vector<Bind> binds;
    };

    Bind
    place(json const &a) const {
        Bind b;
        b.slot   = a.at("slot").get<int>();
        b.offset = a.value("offset", int64_t{0});
        if (b.slot == NULL_SLOT ? b.offset != 0 : (b.slot < 0 || static_cast<size_t>(b.slot) >= align_.size())) {
            throw std::invalid_argument("cudnn launch plan: no slot " + a.dump());
        }
        return b;
    }

    static uint64_t
    address(Bind const &b, uintptr_t const *slots) {
        return b.slot == NULL_SLOT ? 0 : static_cast<uint64_t>(slots[b.slot] + b.offset);
    }

    Functions functions_;
    int device_ = 0;
    std::vector<int64_t> align_;
    std::vector<uint64_t> extent_;  // bytes each argument tensor spans
    std::vector<bool> overlaps_;    // [i * n + j], j < i: argument tensors i and j overlapped in the recorded call
    std::vector<int64_t> delta_;    // [i * n + j]: then, the byte offset of i from j
    std::vector<Step> steps_;
};

}  // namespace

void
init_launch_plan(py::module_ &m) {
    py::class_<LaunchPlan>(m, "_LaunchPlan")
        .def(py::init<std::string const &, int64_t>())
        .def("execute", &LaunchPlan::execute);
    // Resolves tvm-ffi and the CUDA driver and registers "cudnn.launch_plan.hold"; raises where plans cannot run.
    m.def("_launch_plan_runtime", [] { runtime(); });
}

}  // namespace python_bindings
}  // namespace cudnn_frontend
