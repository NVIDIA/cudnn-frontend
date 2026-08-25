// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The variant pack, held as DLTensors rather than python objects.
//
// `__dlpack_c_exchange_api__` is a vtable on the buffer's TYPE whose
// dltensor_from_py_object_no_sync fills a caller-provided DLTensor in place --
// no capsule, no allocation. This file consumes it to read the caller's
// operands and implements it so the operands it hands a kernel are read the same
// way, which is why nothing is given up by refusing to pass the caller's
// object through.
//
// A producer without the vtable is not an error: read_operand returns false and
// python fills that operand from its own reader, so a mixed pack costs the sum of
// its parts.
#include "variant_pack.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dlpack/dlpack.h"

namespace py = pybind11;

namespace cudnn_frontend {
namespace python_bindings {

namespace {

// The vtable is a property of the type, and the DLPack docs tell consumers to
// cache it per type. A handful of buffer types occur in one process.
constexpr int kTypeCacheSlots = 8;

struct TypeCache {
    PyTypeObject *types[kTypeCacheSlots]     = {};
    DLPackExchangeAPI *apis[kTypeCacheSlots] = {};
    int count                                = 0;
};

TypeCache &
type_cache() {
    static TypeCache cache;
    return cache;
}

// The newest table in the producer's chain whose layout matches the one this
// was compiled against, or null.
//
// The protocol requires this walk: a table is only safe to call when its major
// version is ours, and a producer that has moved on keeps older tables reachable
// through prev_api for exactly this reason. Skipping the check would mean
// calling function pointers at offsets that a future major version is free to
// move -- a crash years later, in code that had been correct all along.
DLPackExchangeAPI *
compatible_api(DLPackExchangeAPI *api) {
    for (int hops = 0; api != nullptr && hops < 8; hops++) {
        if (api->header.version.major == DLPACK_MAJOR_VERSION) return api;
        api = reinterpret_cast<DLPackExchangeAPI *>(api->header.prev_api);
    }
    return nullptr;
}

// The producer's exchange vtable, or null when its type does not implement the
// protocol at a version we speak. A missing attribute is the common case for
// older frameworks, not an error, so the python exception it raises is
// swallowed and the caller falls back to reading the buffer from python.
DLPackExchangeAPI *
exchange_api_for(PyObject *obj) {
    PyTypeObject *type = Py_TYPE(obj);
    TypeCache &cache   = type_cache();
    for (int i = 0; i < cache.count; i++) {
        if (cache.types[i] == type) return cache.apis[i];
    }
    PyObject *capsule      = PyObject_GetAttrString(reinterpret_cast<PyObject *>(type), "__dlpack_c_exchange_api__");
    DLPackExchangeAPI *api = nullptr;
    if (capsule == nullptr) {
        PyErr_Clear();
    } else {
        api = compatible_api(static_cast<DLPackExchangeAPI *>(PyCapsule_GetPointer(capsule, "dlpack_exchange_api")));
        Py_DECREF(capsule);
        if (api == nullptr) PyErr_Clear();
    }
    // Only a hit is cached. A type can acquire the vtable AFTER we first look:
    // on torch builds without it natively, tvm-ffi installs one when it is
    // imported, and a graph normalized before that import would otherwise be
    // pinned to the python fallback for the life of the process.
    //
    // Keyed on the type's ADDRESS, so the entry must own a reference: a heap
    // type that got collected could be replaced by a different type allocated
    // at the same address, and this would hand out its vtable. The cache never
    // evicts, so this pins at most kTypeCacheSlots types.
    if (api != nullptr && cache.count < kTypeCacheSlots) {
        Py_INCREF(type);
        cache.types[cache.count] = type;
        cache.apis[cache.count]  = api;
        cache.count++;
    }
    return api;
}

// The name each DLPack (code, bits) travels under in the kernels' vocabulary,
// which is torch's spelling minus the "torch." prefix.
std::string
dtype_name(DLDataType dtype) {
    const int code = dtype.code;
    const int bits = dtype.bits;
    if (code == kDLFloat) {
        if (bits == 16) return "float16";
        if (bits == 32) return "float32";
        if (bits == 64) return "float64";
    } else if (code == kDLBfloat && bits == 16) {
        return "bfloat16";
    } else if (code == kDLInt) {
        if (bits == 8) return "int8";
        if (bits == 32) return "int32";
        if (bits == 64) return "int64";
    } else if (code == kDLUInt && bits == 8) {
        return "uint8";
    } else if (code == kDLBool) {
        return "bool";
    } else if (code == kDLFloat8_e4m3fn) {
        return "float8_e4m3fn";
    } else if (code == kDLFloat8_e5m2) {
        return "float8_e5m2";
    } else if (code == kDLFloat8_e8m0fnu) {
        return "float8_e8m0fnu";
    }
    return "code" + std::to_string(code) + "_" + std::to_string(bits);
}

bool
is_dense(const DLTensor &t) {
    if (t.strides == nullptr) return true;  // compact by definition
    int64_t expect = 1;
    for (int d = t.ndim - 1; d >= 0; d--) {
        if (t.shape[d] != 1 && t.strides[d] != expect) return false;
        expect *= t.shape[d];
    }
    return true;
}

// One operand. The shape and stride live here rather than behind the DLTensor's
// pointers so a operand stays valid once the producer's own DLTensor is gone.
struct Operand {
    void *data       = nullptr;
    int32_t ndim     = 0;
    DLDataType dtype = {0, 0, 1};
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;  // empty means compact row-major
    bool filled = false;
};

}  // namespace

// A pack's operand, exposed to a kernel. It implements the same exchange protocol
// it was read through, so a consumer that has the fast path for a torch tensor
// has it for this too.
class OperandBuffer {
   public:
    OperandBuffer(Operand operand, int32_t device_id) : operand_(std::move(operand)) {
        tensor_.data        = operand_.data;
        tensor_.device      = DLDevice{kDLCUDA, device_id};
        tensor_.ndim        = operand_.ndim;
        tensor_.dtype       = operand_.dtype;
        tensor_.shape       = operand_.shape.empty() ? nullptr : operand_.shape.data();
        tensor_.strides     = operand_.stride.empty() ? nullptr : operand_.stride.data();
        tensor_.byte_offset = 0;
    }

    // tensor_ points into operand_'s vectors, so a copy would leave the new
    // object's DLTensor describing the old one's storage. Operands are always
    // heap-allocated and handed out by pointer, so nothing needs to copy one.
    OperandBuffer(const OperandBuffer &) = delete;
    OperandBuffer &
    operator=(const OperandBuffer &) = delete;
    OperandBuffer(OperandBuffer &&)  = delete;
    OperandBuffer &
    operator=(OperandBuffer &&) = delete;

    const DLTensor &
    tensor() const {
        return tensor_;
    }

    int64_t
    data_ptr() const {
        return reinterpret_cast<int64_t>(tensor_.data);
    }

    std::vector<int64_t>
    shape() const {
        return operand_.shape;
    }

    std::vector<int64_t>
    stride() const {
        if (!operand_.stride.empty()) return operand_.stride;
        std::vector<int64_t> dense(operand_.ndim, 1);
        for (int d = operand_.ndim - 2; d >= 0; d--) dense[d] = dense[d + 1] * operand_.shape[d + 1];
        return dense;
    }

    // One axis of it, the way a framework tensor is asked (stride(-1)).
    int64_t
    stride_at(int64_t dim) const {
        int64_t axis = dim < 0 ? dim + operand_.ndim : dim;
        if (axis < 0 || axis >= operand_.ndim)
            throw py::index_error("stride(): dimension " + std::to_string(dim) + " is out of range for a " +
                                  std::to_string(operand_.ndim) + "-D operand");
        return stride()[axis];
    }

    // The bare dtype NAME, which is what a kernel means when it asks a buffer
    // for its dtype: they all reach it through str(x.dtype).split(".")[-1], so
    // a torch tensor's "torch.bfloat16" and this "bfloat16" answer the same.
    std::string
    dtype() const {
        return dtype_name(operand_.dtype);
    }

    int64_t
    element_size() const {
        return operand_.dtype.bits / 8;
    }

    int64_t
    numel() const {
        int64_t n = 1;
        for (int64_t extent : operand_.shape) n *= extent;
        return n;
    }

    int64_t
    nbytes() const {
        return numel() * element_size();
    }

    int64_t
    length() const {
        return operand_.shape.empty() ? 0 : operand_.shape[0];
    }

    py::tuple
    dlpack_device() const {
        return py::make_tuple(static_cast<int>(kDLCUDA), tensor_.device.device_id);
    }

    // A differently shaped view of the same memory, with one -1 wildcard. Only
    // meaningful for a dense operand, which is why a strided one is refused rather
    // than silently reinterpreted.
    OperandBuffer *
    reshape(std::vector<int64_t> shape) const {
        if (!operand_.stride.empty() && !is_dense(tensor_))
            throw py::value_error("cannot reshape a non-contiguous variant-pack operand");
        int64_t numel = 1;
        for (int64_t extent : operand_.shape) numel *= extent;
        int64_t fixed = 1;
        int wildcard  = -1;
        for (size_t d = 0; d < shape.size(); d++) {
            if (shape[d] == -1) {
                if (wildcard >= 0) throw py::value_error("reshape accepts at most one -1");
                wildcard = static_cast<int>(d);
            } else {
                fixed *= shape[d];
            }
        }
        if (wildcard >= 0) {
            if (fixed == 0 || numel % fixed != 0)
                throw py::value_error("cannot reshape " + std::to_string(numel) + " elements to the requested shape");
            shape[wildcard] = numel / fixed;
        } else if (fixed != numel) {
            throw py::value_error("cannot reshape " + std::to_string(numel) + " elements to " + std::to_string(fixed));
        }
        Operand out = operand_;
        out.shape   = std::move(shape);
        out.ndim    = static_cast<int32_t>(out.shape.size());
        out.stride.clear();  // dense by construction, as the reshape required
        return new OperandBuffer(out, tensor_.device.device_id);
    }

    // The same memory with its axes relabelled; exact for a strided operand too.
    OperandBuffer *
    permute(const std::vector<int64_t> &axes) const {
        if (axes.size() != static_cast<size_t>(operand_.ndim))
            throw py::value_error("permute needs one axis per dimension: this operand is " +
                                  std::to_string(operand_.ndim) + "-D");
        std::vector<bool> seen(axes.size(), false);
        Operand out = operand_;
        out.stride.assign(operand_.ndim, 1);
        if (operand_.stride.empty()) {
            for (int d = operand_.ndim - 2; d >= 0; d--) out.stride[d] = out.stride[d + 1] * operand_.shape[d + 1];
        } else {
            out.stride = operand_.stride;
        }
        const std::vector<int64_t> from_stride = out.stride;
        for (size_t d = 0; d < axes.size(); d++) {
            int64_t axis = axes[d] < 0 ? axes[d] + operand_.ndim : axes[d];
            if (axis < 0 || axis >= operand_.ndim || seen[axis])
                throw py::value_error("permute axes must be a permutation of the operand's dimensions");
            seen[axis]    = true;
            out.shape[d]  = operand_.shape[axis];
            out.stride[d] = from_stride[axis];
        }
        return new OperandBuffer(std::move(out), tensor_.device.device_id);
    }

    // Row-major contiguous by construction, so this is the identity a caller
    // written against a framework tensor expects to be able to call.
    py::object
    contiguous(py::object self) const {
        return self;
    }

    // The capsule form of the same tensor, for a consumer that does not read
    // the exchange vtable -- cute's from_dlpack at compile time is the one that
    // matters here. It costs an allocation where the vtable costs none, which
    // is why it is not what the per-launch path uses.
    //
    // Ownership transfers with the capsule, per DLPack: the struct carries its
    // own copy of the shape and stride and a deleter that frees them, so it
    // outlives this operand rather than aliasing storage the operand owns.
    //
    // Unversioned only: max_version is ignored and the capsule is always
    // "dltensor". The consumer this exists for is cute's compile-time
    // from_dlpack; tvm-ffi reads a operand through the exchange vtable and never
    // gets here.
    py::capsule
    dlpack(py::object /*stream*/, py::object /*max_version*/) const {
        struct Owned {
            DLManagedTensor managed;
            std::vector<int64_t> shape;
            std::vector<int64_t> stride;
        };
        auto *owned                      = new Owned{{}, operand_.shape, operand_.stride};
        owned->managed.dl_tensor         = tensor_;
        owned->managed.dl_tensor.shape   = owned->shape.empty() ? nullptr : owned->shape.data();
        owned->managed.dl_tensor.strides = owned->stride.empty() ? nullptr : owned->stride.data();
        owned->managed.manager_ctx       = owned;
        owned->managed.deleter = [](DLManagedTensor *self) { delete static_cast<Owned *>(self->manager_ctx); };
        return py::capsule(&owned->managed, "dltensor", [](PyObject *capsule) {
            // only reached when nobody consumed it: a consumer renames the
            // capsule to "used_dltensor" and takes the deleter over
            if (PyCapsule_IsValid(capsule, "dltensor")) {
                auto *managed = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
                if (managed != nullptr && managed->deleter != nullptr) managed->deleter(managed);
            }
        });
    }

   private:
    Operand operand_;  // owns the shape/stride storage the DLTensor points into
    DLTensor tensor_{};
};

namespace {

int
buffer_dltensor_from_py_object(void *py_object, DLTensor *out) {
    auto *operand = py::cast<OperandBuffer *>(py::handle(static_cast<PyObject *>(py_object)));
    *out          = operand->tensor();
    return 0;
}

int
buffer_managed_from_py_object(void *py_object, DLManagedTensorVersioned **out) {
    auto *operand = py::cast<OperandBuffer *>(py::handle(static_cast<PyObject *>(py_object)));
    // A managed tensor is the form a consumer is allowed to OUTLIVE the
    // producer with, so it cannot point at the operand's vectors: the shape and
    // stride are copied and owned here, and the deleter frees them.
    struct Managed {
        DLManagedTensorVersioned versioned;
        std::vector<int64_t> shape;
        std::vector<int64_t> stride;
    };
    auto *owned = new Managed{
        {}, operand->shape(), operand->tensor().strides == nullptr ? std::vector<int64_t>() : operand->stride()};
    auto &tensor = owned->versioned.dl_tensor;
    tensor       = operand->tensor();
    tensor.shape = owned->shape.empty() ? nullptr : owned->shape.data();
    // a operand with no stride array is dense, and DLPack spells that as null
    tensor.strides                 = owned->stride.empty() ? nullptr : owned->stride.data();
    owned->versioned.version.major = DLPACK_MAJOR_VERSION;
    owned->versioned.version.minor = DLPACK_MINOR_VERSION;
    owned->versioned.manager_ctx   = owned;
    owned->versioned.deleter = [](DLManagedTensorVersioned *self) { delete static_cast<Managed *>(self->manager_ctx); };
    *out                     = &owned->versioned;
    return 0;
}

int
buffer_allocator(DLTensor *,
                 DLManagedTensorVersioned **,
                 void *error_ctx,
                 void (*set_error)(void *, const char *, const char *)) {
    set_error(error_ctx, "NotImplementedError", "a variant-pack operand views the caller's memory; it never allocates");
    return -1;
}

int
buffer_to_py_object(DLManagedTensorVersioned *, void **) {
    PyErr_SetString(PyExc_NotImplementedError, "a variant-pack operand is not an importer");
    return -1;
}

// The graph launches on the stream its handle carries, which execute() passes
// explicitly. Reporting no producer stream is what tells a consumer to use the
// one it was given rather than going looking for ours.
int
buffer_current_work_stream(DLDeviceType, int32_t, void **out_stream) {
    *out_stream = nullptr;
    return 0;
}

DLPackExchangeAPI &
buffer_exchange_api() {
    static DLPackExchangeAPI api = [] {
        DLPackExchangeAPI table{};
        table.header.version.major                  = DLPACK_MAJOR_VERSION;
        table.header.version.minor                  = DLPACK_MINOR_VERSION;
        table.header.prev_api                       = nullptr;
        table.managed_tensor_allocator              = buffer_allocator;
        table.managed_tensor_from_py_object_no_sync = buffer_managed_from_py_object;
        table.managed_tensor_to_py_object_no_sync   = buffer_to_py_object;
        table.dltensor_from_py_object_no_sync       = buffer_dltensor_from_py_object;
        table.current_work_stream                   = buffer_current_work_stream;
        return table;
    }();
    return api;
}

}  // namespace

class VariantPackNative {
   public:
    explicit VariantPackNative(size_t n) : operands_(n), pointers_(n, nullptr) {}

    // Fill one operand from the caller's buffer. False means its type does not
    // implement the exchange protocol and python must describe it instead.
    bool
    read_operand(size_t index, py::handle buffer) {
        Operand &operand       = operands_.at(index);
        DLPackExchangeAPI *api = exchange_api_for(buffer.ptr());
        if (api == nullptr || api->dltensor_from_py_object_no_sync == nullptr) return false;
        DLTensor t{};
        if (api->dltensor_from_py_object_no_sync(buffer.ptr(), &t) != 0) throw py::error_already_set();
        operand.data  = static_cast<char *>(t.data) + t.byte_offset;
        operand.ndim  = t.ndim;
        operand.dtype = t.dtype;
        operand.shape.assign(t.shape, t.shape + t.ndim);
        if (t.strides != nullptr) {
            operand.stride.assign(t.strides, t.strides + t.ndim);
        } else {
            operand.stride.clear();
        }
        operand.filled   = true;
        pointers_[index] = operand.data;
        return true;
    }

    // Every operand in one call. Returns the indices whose producer has no vtable,
    // for python to describe and report back through set_operand -- crossing the
    // binding once per pack rather than once per operand is 2.53 us against
    // 1.0 for eight. A None entry is a operand the caller did not fill.
    // A uid the map does not carry is left unfilled rather than refused: whether
    // that is the caller's mistake or an optional port depends on the graph,
    // which python knows and this does not.
    std::vector<size_t>
    read_from(const py::dict &uid_to_data, const std::vector<int64_t> &uids) {
        std::vector<size_t> unread;
        const size_t n = uids.size();
        for (size_t i = 0; i < n && i < operands_.size(); i++) {
            PyObject *buffer = PyDict_GetItem(uid_to_data.ptr(), py::int_(uids[i]).ptr());
            if (buffer == nullptr || buffer == Py_None) {
                skip_operand(i);
            } else if (!read_operand(i, py::handle(buffer))) {
                unread.push_back(i);
            }
        }
        return unread;
    }

    // The first operand no one filled, or -1.
    int64_t
    first_unfilled() const {
        for (size_t i = 0; i < operands_.size(); i++) {
            if (!operands_[i].filled) return static_cast<int64_t>(i);
        }
        return -1;
    }

    // The fallback: python read the buffer its own way and reports the result.
    void
    set_operand(size_t index,
                int64_t ptr,
                std::vector<int64_t> shape,
                std::vector<int64_t> stride,
                int dtype_code,
                int dtype_bits) {
        Operand &operand = operands_.at(index);
        operand.data     = reinterpret_cast<void *>(ptr);
        operand.ndim     = static_cast<int32_t>(shape.size());
        operand.dtype    = DLDataType{static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits), 1};
        operand.shape    = std::move(shape);
        operand.stride   = std::move(stride);
        operand.filled   = true;
        pointers_[index] = operand.data;
    }

    // Re-describe a operand at the shape this execute runs, keeping its buffer.
    // Applying override_shapes here rather than in an engine is what keeps the
    // two paths answering the same question: an engine that reads the pack
    // honours the override without knowing the concept exists.
    void
    override_operand(size_t index, std::vector<int64_t> shape, std::vector<int64_t> stride) {
        Operand &operand = operands_.at(index);
        if (!operand.filled) {
            throw py::value_error("variant-pack operand " + std::to_string(index) + " has no buffer to re-describe");
        }
        // ndim comes from the shape and the stride array is read ndim deep, and
        // this is the one place the two arrive from different lists.
        if (shape.size() != stride.size()) {
            throw py::value_error("override shape and stride must have the same rank; got " +
                                  std::to_string(shape.size()) + " and " + std::to_string(stride.size()) +
                                  " for operand " + std::to_string(index));
        }
        operand.ndim   = static_cast<int32_t>(shape.size());
        operand.shape  = std::move(shape);
        operand.stride = std::move(stride);
    }

    void
    skip_operand(size_t index) {
        operands_.at(index).filled = false;
        pointers_[index]           = nullptr;
    }

    bool
    all_contiguous(std::string &offender) const {
        for (size_t i = 0; i < operands_.size(); i++) {
            const Operand &operand = operands_[i];
            if (!operand.filled || operand.stride.empty()) continue;
            int64_t expect = 1;
            for (int d = operand.ndim - 1; d >= 0; d--) {
                if (operand.shape[d] != 1 && operand.stride[d] != expect) {
                    offender = std::to_string(i);
                    return false;
                }
                expect *= operand.shape[d];
            }
        }
        return true;
    }

    bool
    all_dense_layout(std::string &offender) const {
        for (size_t i = 0; i < operands_.size(); i++) {
            const Operand &operand = operands_[i];
            if (!operand.filled || operand.stride.empty()) continue;
            for (int d = operand.ndim - 1; d >= 0; d--) {
                if (operand.shape[d] == 1) continue;
                if (operand.stride[d] != 1) {
                    offender = std::to_string(i);
                    return false;
                }
                break;
            }
        }
        return true;
    }

    bool
    operand_contiguous(size_t index) const {
        const Operand &operand = operands_.at(index);
        if (!operand.filled || operand.stride.empty()) return true;
        int64_t expect = 1;
        for (int d = operand.ndim - 1; d >= 0; d--) {
            if (operand.shape[d] != 1 && operand.stride[d] != expect) return false;
            expect *= operand.shape[d];
        }
        return true;
    }

    bool
    is_filled(size_t index) const {
        return operands_.at(index).filled;
    }

    int64_t
    pointer(size_t index) const {
        return reinterpret_cast<int64_t>(pointers_.at(index));
    }

    std::vector<int64_t>
    shape(size_t index) const {
        return operands_.at(index).shape;
    }

    std::vector<int64_t>
    stride(size_t index) const {
        const Operand &operand = operands_.at(index);
        if (!operand.stride.empty()) return operand.stride;
        std::vector<int64_t> dense(operand.ndim, 1);
        for (int d = operand.ndim - 2; d >= 0; d--) dense[d] = dense[d + 1] * operand.shape[d + 1];
        return dense;
    }

    py::tuple
    dtype(size_t index) const {
        const Operand &operand = operands_.at(index);
        return py::make_tuple(operand.dtype.code, operand.dtype.bits);
    }

    // The address the backend's variant pack reads: a contiguous void*[] in
    // operand order, so it goes to _execute_with_raw_ptrs with no copy.
    int64_t
    pointer_array(void) const {
        return reinterpret_cast<int64_t>(pointers_.data());
    }

    size_t
    size(void) const {
        return operands_.size();
    }

    // Every requested operand in one crossing, for an engine binding a whole node.
    std::vector<OperandBuffer *>
    operands(const std::vector<size_t> &indices, int32_t device_id) const {
        std::vector<OperandBuffer *> out;
        out.reserve(indices.size());
        for (size_t index : indices) out.push_back(operand(index, device_id));
        return out;
    }

    OperandBuffer *
    operand(size_t index, int32_t device_id) const {
        const Operand &operand = operands_.at(index);
        if (!operand.filled)
            throw py::value_error("variant-pack operand " + std::to_string(index) + " was not filled by the caller");
        return new OperandBuffer(operand, device_id);
    }

   private:
    std::vector<Operand> operands_;
    std::vector<void *> pointers_;
};

// A operand over memory that is not a caller operand: the regions a plan carves
// out of the workspace. Same type, so a kernel is handed one kind of buffer
// whether it came from the caller or from the workspace, and both are read
// through the exchange vtable rather than a per-call capsule.
OperandBuffer *
make_operand_buffer(int64_t ptr, std::vector<int64_t> shape, int dtype_code, int dtype_bits, int32_t device_id) {
    Operand operand;
    operand.data   = reinterpret_cast<void *>(ptr);
    operand.ndim   = static_cast<int32_t>(shape.size());
    operand.dtype  = DLDataType{static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits), 1};
    operand.shape  = std::move(shape);
    operand.filled = true;  // stride left empty: a carve is dense by construction
    return new OperandBuffer(std::move(operand), device_id);
}

// (pointer, bytes) for a buffer that publishes the vtable, else None. The
// workspace has no uid and no operand, but an engine still bounds-checks its
// carves against it.
py::object
read_buffer_extent(py::handle buffer) {
    DLPackExchangeAPI *api = exchange_api_for(buffer.ptr());
    if (api == nullptr || api->dltensor_from_py_object_no_sync == nullptr) return py::none();
    DLTensor t{};
    if (api->dltensor_from_py_object_no_sync(buffer.ptr(), &t) != 0) throw py::error_already_set();
    // A byte count is only a byte RANGE when the buffer is dense; a carve
    // bounds-checked against a strided one would write outside the allocation.
    if (!is_dense(t)) return py::none();
    int64_t numel = 1;
    for (int d = 0; d < t.ndim; d++) numel *= t.shape[d];
    const int64_t itemsize = (static_cast<int64_t>(t.dtype.bits) * t.dtype.lanes + 7) / 8;
    return py::make_tuple(reinterpret_cast<int64_t>(static_cast<char *>(t.data) + t.byte_offset), numel * itemsize);
}

// A workspace carve, planned once: the regions are fixed when the engine
// builds and only the base pointer arrives per execute.
class WorkspaceCarve {
   public:
    WorkspaceCarve(std::string owner, const std::vector<py::tuple> &regions) : owner_(std::move(owner)) {
        protos_.reserve(regions.size());
        offsets_.reserve(regions.size());
        ends_.reserve(regions.size());
        for (const py::tuple &region : regions) {
            if (region.size() != 4) {
                throw py::value_error("a carve region is (offset, dtype_code, dtype_bits, shape)");
            }
            int64_t offset = region[0].cast<int64_t>();
            Operand proto;
            proto.dtype   = DLDataType{region[1].cast<uint8_t>(), region[2].cast<uint8_t>(), 1};
            proto.shape   = region[3].cast<std::vector<int64_t>>();
            proto.ndim    = static_cast<int32_t>(proto.shape.size());
            proto.filled  = true;  // stride left empty: a carve is dense by construction
            int64_t numel = 1;
            for (int64_t extent : proto.shape) numel *= extent;
            offsets_.push_back(offset);
            ends_.push_back(offset + numel * ((proto.dtype.bits + 7) / 8));
            protos_.push_back(std::move(proto));
        }
    }

    std::vector<OperandBuffer *>
    carve(int64_t base, int64_t nbytes, int32_t device_id) const {
        std::vector<OperandBuffer *> out;
        out.reserve(protos_.size());
        for (size_t i = 0; i < protos_.size(); i++) {
            if (nbytes != 0 && ends_[i] > nbytes) {  // 0 = size unknown (bare address)
                throw py::value_error(owner_ + ": workspace overrun -- region [" + std::to_string(offsets_[i]) + ", " +
                                      std::to_string(ends_[i]) + ") exceeds the " + std::to_string(nbytes) +
                                      "-byte buffer (sizing bug)");
            }
            Operand operand = protos_[i];
            operand.data    = reinterpret_cast<void *>(base + offsets_[i]);
            out.push_back(new OperandBuffer(std::move(operand), device_id));
        }
        return out;
    }

    size_t
    size() const {
        return protos_.size();
    }

   private:
    std::string owner_;
    std::vector<Operand> protos_;
    std::vector<int64_t> offsets_;
    std::vector<int64_t> ends_;
};

void
init_variant_pack(py::module_ &m) {
    auto operand_class =
        py::class_<OperandBuffer>(m, "OperandBuffer", R"(
One operand of a variant pack, as a DLPack producer.

Implements ``__dlpack_c_exchange_api__``, so a consumer reads it through the
same C function table it uses for a framework tensor rather than through a
capsule built in python.
)")
            .def("data_ptr", &OperandBuffer::data_ptr)
            .def_property_readonly("shape", &OperandBuffer::shape)
            // ndim, because `__len__` is the first EXTENT and a caller
            // reaching for a rank through getattr(.., "ndim", default)
            // silently gets the default instead.
            .def_property_readonly("ndim", [](const OperandBuffer &self) { return self.shape().size(); })
            .def_property_readonly("dtype", &OperandBuffer::dtype)
            .def_property_readonly("nbytes", &OperandBuffer::nbytes)
            .def(
                "stride",
                [](const OperandBuffer &self, py::object dim) -> py::object {
                    if (dim.is_none()) return py::cast(self.stride());
                    return py::cast(self.stride_at(dim.cast<int64_t>()));
                },
                py::arg("dim") = py::none())
            .def("element_size", &OperandBuffer::element_size)
            .def("numel", &OperandBuffer::numel)
            .def("__len__", &OperandBuffer::length)
            .def("reshape",
                 [](const OperandBuffer &self, py::args dims) {
                     std::vector<int64_t> shape;
                     if (dims.size() == 1 && py::isinstance<py::sequence>(dims[0]) &&
                         !py::isinstance<py::int_>(dims[0])) {
                         shape = dims[0].cast<std::vector<int64_t>>();
                     } else {
                         for (auto d : dims) shape.push_back(d.cast<int64_t>());
                     }
                     return self.reshape(std::move(shape));
                 })
            .def("permute",
                 [](const OperandBuffer &self, py::args axes) {
                     std::vector<int64_t> order;
                     if (axes.size() == 1 && py::isinstance<py::sequence>(axes[0]) &&
                         !py::isinstance<py::int_>(axes[0])) {
                         order = axes[0].cast<std::vector<int64_t>>();
                     } else {
                         for (auto a : axes) order.push_back(a.cast<int64_t>());
                     }
                     return self.permute(order);
                 })
            .def("contiguous", [](py::object self) { return self; })
            .def("__dlpack_device__", &OperandBuffer::dlpack_device)
            .def("__dlpack__",
                 &OperandBuffer::dlpack,
                 py::kw_only(),
                 py::arg("stream")      = py::none(),
                 py::arg("max_version") = py::none());

    // The protocol looks the attribute up on the TYPE, and a pybind11 class is
    // a heap type, so it takes a plain setattr.
    PyObject *capsule = PyCapsule_New(&buffer_exchange_api(), "dlpack_exchange_api", nullptr);
    if (capsule == nullptr) throw py::error_already_set();
    if (PyObject_SetAttrString(operand_class.ptr(), "__dlpack_c_exchange_api__", capsule) < 0) {
        Py_DECREF(capsule);
        throw py::error_already_set();
    }
    Py_DECREF(capsule);

    m.def("make_operand_buffer",
          &make_operand_buffer,
          py::arg("ptr"),
          py::arg("shape"),
          py::arg("dtype_code"),
          py::arg("dtype_bits"),
          py::arg("device_id"),
          "A DLPack producer over memory the caller did not supply -- a workspace carve.");

    m.def("read_buffer_extent",
          &read_buffer_extent,
          py::arg("buffer"),
          "(pointer, bytes) through the exchange vtable, or None when the type does not publish one.");

    py::class_<WorkspaceCarve>(m, "WorkspaceCarve", R"(
A workspace carve compiled once, at build.

Regions are ``(offset, dtype_code, dtype_bits, shape)``. Only the base pointer
arrives per execute, so ``carve`` hands back every region in one crossing
instead of one per region.
)")
        .def(py::init<std::string, std::vector<py::tuple>>(), py::arg("owner"), py::arg("regions"))
        .def("carve", &WorkspaceCarve::carve, py::arg("base"), py::arg("nbytes"), py::arg("device_id"))
        .def("__len__", &WorkspaceCarve::size);

    operand_class.attr("view") = operand_class.attr("reshape");

    py::class_<VariantPackNative>(m, "VariantPackNative", R"(
The caller's operands, held as DLTensors.

``read_operand`` returns False for a buffer whose type does not implement
``__dlpack_c_exchange_api__``; the caller describes that one itself and reports
it through ``set_operand``, so a pack mixing producers costs exactly the sum of
its parts.
)")
        .def(py::init<size_t>())
        .def("read_operand", &VariantPackNative::read_operand)
        .def("read_from", &VariantPackNative::read_from)
        .def("first_unfilled", &VariantPackNative::first_unfilled)
        .def("set_operand", &VariantPackNative::set_operand)
        .def("override_operand", &VariantPackNative::override_operand)
        .def("skip_operand", &VariantPackNative::skip_operand)
        .def("operand_contiguous", &VariantPackNative::operand_contiguous)
        .def("is_filled", &VariantPackNative::is_filled)
        .def("pointer", &VariantPackNative::pointer)
        .def("shape", &VariantPackNative::shape)
        .def("stride", &VariantPackNative::stride)
        .def("dtype", &VariantPackNative::dtype)
        .def("operand", &VariantPackNative::operand)
        .def("operands", &VariantPackNative::operands)
        .def_property_readonly("address", &VariantPackNative::pointer_array)
        .def("__len__", &VariantPackNative::size)
        .def("all_contiguous",
             [](const VariantPackNative &self) {
                 std::string offender;
                 bool ok = self.all_contiguous(offender);
                 return py::make_tuple(ok, offender);
             })
        .def("all_dense_layout", [](const VariantPackNative &self) {
            std::string offender;
            bool ok = self.all_dense_layout(offender);
            return py::make_tuple(ok, offender);
        });
}

}  // namespace python_bindings
}  // namespace cudnn_frontend
