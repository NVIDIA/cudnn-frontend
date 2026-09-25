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

#include <algorithm>
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
    } else if (code == kDLFloat4_e2m1fn && bits == 4 && dtype.lanes == 2) {
        return "float4_e2m1fn_x2";  // two elements per slot, as torch spells the storage dtype
    }
    std::string name = "code" + std::to_string(code) + "_" + std::to_string(bits);
    if (dtype.lanes != 1) {
        name += "_x" + std::to_string(dtype.lanes);
    }
    return name;
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
    // What the PRODUCER said about its buffer, kept apart from the effective (graph-described /
    // overridden) geometry above: the element span it guarantees addressable (-1: unknown, a bare
    // address) and its DLPack device (-1: unknown). An engine deriving a capacity reads these.
    int64_t observed_bytes       = -1;  // producer's guaranteed span, in BYTES (its own element width)
    int32_t observed_device_type = -1;
    int32_t observed_device_id   = -1;
};

// Emit effective strides into either a native vector or a Python tuple without
// materializing an intermediate container. Empty producer strides mean compact.
template <typename Store>
void
write_effective_strides(const Operand &operand, Store &&store) {
    if (!operand.stride.empty()) {
        for (size_t d = 0; d < operand.stride.size(); ++d) store(d, operand.stride[d]);
        return;
    }
    int64_t running = 1;
    for (int d = operand.ndim - 1; d >= 0; --d) {
        store(d, running);
        if (d > 0) running *= operand.shape[d];
    }
}

// Slots from the base to one past the last addressed slot.
int64_t
span_of(const std::vector<int64_t> &shape, const std::vector<int64_t> &stride) {
    int64_t span = 1;
    for (size_t d = 0; d < shape.size(); d++) span += (shape[d] - 1) * stride[d];
    return span;
}

int64_t
numel_of(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (int64_t extent : shape) n *= extent;
    return n;
}

std::vector<int64_t>
dense_stride_of(const std::vector<int64_t> &shape) {
    std::vector<int64_t> dense(shape.size(), 1);
    for (int d = static_cast<int>(shape.size()) - 2; d >= 0; d--) dense[d] = dense[d + 1] * shape[d + 1];
    return dense;
}

// A cuDNN (element) geometry as the STORAGE-slot geometry a buffer reports
// (graph_types.storage_geometry): fp4 packs two elements per slot along the
// unit-stride axis, so that extent halves and every other stride halves with it.
// False when the packed extent is odd (no slot geometry spells it).
bool
storage_geometry_of(std::vector<int64_t> &shape, std::vector<int64_t> &stride, bool fp4) {
    if (stride.empty()) stride = dense_stride_of(shape);
    if (!fp4) return true;
    for (size_t c = 0; c < shape.size(); c++) {
        if (stride[c] != 1 || shape[c] <= 1 || shape[c] % 2) continue;
        bool ok = true;
        for (size_t j = 0; j < stride.size(); j++)
            if (j != c && stride[j] != 1 && stride[j] % 2) ok = false;
        if (!ok) continue;
        shape[c] /= 2;
        for (size_t j = 0; j < stride.size(); j++)
            if (j != c && stride[j] != 1) stride[j] /= 2;
        return true;
    }
    return false;
}

// (shape, stride) re-expressed in the axis order `reference` uses (_pygraph._in_axis_order_of):
// both orders rank their axes the same way by stride, which gives the permutation.
void
in_axis_order_of(std::vector<int64_t> &shape, std::vector<int64_t> &stride, const std::vector<int64_t> &reference) {
    const size_t n = shape.size();
    if (n != stride.size() || n != reference.size()) return;
    std::vector<size_t> by_stride(n), ref(n);
    for (size_t i = 0; i < n; i++) by_stride[i] = ref[i] = i;
    std::stable_sort(by_stride.begin(), by_stride.end(), [&](size_t a, size_t b) { return stride[a] > stride[b]; });
    std::stable_sort(ref.begin(), ref.end(), [&](size_t a, size_t b) { return reference[a] > reference[b]; });
    std::vector<size_t> perm(n);
    for (size_t rank = 0; rank < n; rank++) perm[ref[rank]] = by_stride[rank];
    std::vector<int64_t> s2(n), t2(n);
    for (size_t i = 0; i < n; i++) {
        s2[i] = shape[perm[i]];
        t2[i] = stride[perm[i]];
    }
    shape  = std::move(s2);
    stride = std::move(t2);
}

}  // namespace

// The STORAGE-slot geometry every variant-pack slot was declared with. A
// declaration does not change between executes, so python builds this once per
// graph and the pack compares a caller's buffer against it in one crossing
// (VariantPackNative::describe_from) rather than two per operand per call.
class DeclaredLayout {
   public:
    struct Slot {
        std::vector<int64_t> shape;
        std::vector<int64_t> stride;
        int64_t slot_bytes   = 0;  // 0: width unknown, never re-described from
        int64_t span         = 0;
        DLDataType dtype     = {0, 0, 1};  // bits == 0: no DLPack spelling, the buffer's own dtype stands
        bool present         = false;
        bool preserve_extent = false;
    };

    explicit DeclaredLayout(size_t n) : slots_(n) {}

    void
    set(size_t index,
        std::vector<int64_t> shape,
        std::vector<int64_t> stride,
        int64_t slot_bytes,
        int dtype_code       = 0,
        int dtype_bits       = 0,
        int dtype_lanes      = 1,
        bool preserve_extent = false) {
        if (shape.size() != stride.size()) {
            throw py::value_error("declared shape and stride must have the same rank; got " +
                                  std::to_string(shape.size()) + " and " + std::to_string(stride.size()) +
                                  " for slot " + std::to_string(index));
        }
        Slot &slot      = slots_.at(index);
        slot.span       = span_of(shape, stride);
        slot.shape      = std::move(shape);
        slot.stride     = std::move(stride);
        slot.slot_bytes = slot_bytes;
        slot.dtype      = DLDataType{
            static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits), static_cast<uint16_t>(dtype_lanes)};
        slot.present         = true;
        slot.preserve_extent = preserve_extent;
    }

    // A slot whose declaration has a dtype but no storage geometry (no dims, or an
    // fp4 extent no slot geometry spells): overrides still speak its dtype.
    void
    set_dtype(size_t index, int dtype_code, int dtype_bits, int dtype_lanes) {
        Slot &slot = slots_.at(index);
        slot.dtype = DLDataType{
            static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits), static_cast<uint16_t>(dtype_lanes)};
    }

    const std::vector<Slot> &
    slots() const {
        return slots_;
    }

    size_t
    size() const {
        return slots_.size();
    }

   private:
    std::vector<Slot> slots_;
};

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

    // Bytes per storage slot: an fp4 slot is 4 bits x 2 lanes = one byte.
    int64_t
    element_size() const {
        return (static_cast<int64_t>(operand_.dtype.bits) * operand_.dtype.lanes + 7) / 8;
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
        operand.filled = true;
        {
            // A zero-element producer spans nothing: the affine formula assumes a nonempty index domain.
            const int64_t elems    = numel_of(operand.shape) == 0 ? 0
                                     : operand.stride.empty()     ? numel_of(operand.shape)
                                                                  : span_of(operand.shape, operand.stride);
            const int64_t bytes    = (static_cast<int64_t>(t.dtype.bits) * t.dtype.lanes + 7) / 8;
            operand.observed_bytes = elems * bytes;
        }
        operand.observed_device_type = static_cast<int32_t>(t.device.device_type);
        operand.observed_device_id   = t.device.device_id;
        pointers_[index]             = operand.data;
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
                int dtype_bits,
                int dtype_lanes          = 1,
                int64_t observed_bytes   = -1,
                int observed_device_type = -1,
                int observed_device_id   = -1) {
        Operand &operand             = operands_.at(index);
        operand.observed_bytes       = observed_bytes;
        operand.observed_device_type = observed_device_type;
        operand.observed_device_id   = observed_device_id;
        operand.data                 = reinterpret_cast<void *>(ptr);
        operand.ndim                 = static_cast<int32_t>(shape.size());
        operand.dtype                = DLDataType{
            static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits), static_cast<uint16_t>(dtype_lanes)};
        operand.shape    = std::move(shape);
        operand.stride   = std::move(stride);
        operand.filled   = true;
        pointers_[index] = operand.data;
    }

    // Re-describe a operand at the shape this execute runs, keeping its buffer.
    // Applying override_shapes here rather than in an engine is what keeps the
    // two paths answering the same question: an engine that reads the pack
    // honours the override without knowing the concept exists. The override
    // describes the DECLARED tensor at another size, so the slot takes the
    // declared dtype too when the buffer's slots are as wide (dtype_bits > 0
    // names it; FlashInfer binds fp4 and fp8 blocks as uint8).
    void
    override_operand(size_t index,
                     std::vector<int64_t> shape,
                     std::vector<int64_t> stride,
                     int dtype_code  = 0,
                     int dtype_bits  = 0,
                     int dtype_lanes = 1) {
        Operand &operand = operands_.at(index);
        if (!operand.filled) {
            throw py::value_error("variant-pack operand " + std::to_string(index) + " has no buffer to re-describe");
        }
        if (dtype_bits > 0) {
            const int64_t own_bytes  = (static_cast<int64_t>(operand.dtype.bits) * operand.dtype.lanes + 7) / 8;
            const int64_t want_bytes = (static_cast<int64_t>(dtype_bits) * dtype_lanes + 7) / 8;
            if (own_bytes == want_bytes) {
                operand.dtype = DLDataType{static_cast<uint8_t>(dtype_code),
                                           static_cast<uint8_t>(dtype_bits),
                                           static_cast<uint16_t>(dtype_lanes)};
            }
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

    // Every execute-time override in one crossing (_pygraph._normalize): per slot the
    // override's element geometry becomes storage-slot geometry (fp4 packing), is
    // re-expressed in the buffer's own axis order, and is applied like override_operand
    // with the declared dtype.
    void
    override_many(const DeclaredLayout &declared,
                  const std::vector<size_t> &indices,
                  std::vector<std::vector<int64_t>> shapes,
                  std::vector<std::vector<int64_t>> strides) {
        if (shapes.size() != indices.size() || strides.size() != indices.size()) {
            throw py::value_error("override_many: indices, shapes and strides must have the same length");
        }
        const auto &slots = declared.slots();
        for (size_t j = 0; j < indices.size(); j++) {
            const size_t i = indices[j];
            if (i >= operands_.size())
                throw py::index_error("override_many: slot " + std::to_string(i) + " out of range");
            const DLDataType dtype      = i < slots.size() ? slots[i].dtype : DLDataType{0, 0, 1};
            const bool fp4              = dtype.bits == 4 && dtype.lanes == 2;
            std::vector<int64_t> shape  = std::move(shapes[j]);
            std::vector<int64_t> stride = std::move(strides[j]);
            if (shape.size() != stride.size() && !stride.empty()) {
                throw py::value_error("override shape and stride must have the same rank; got " +
                                      std::to_string(shape.size()) + " and " + std::to_string(stride.size()) +
                                      " for operand " + std::to_string(i));
            }
            if (!storage_geometry_of(shape, stride, fp4)) {
                std::string geom;
                for (size_t d = 0; d < shape.size(); d++) geom += (d ? ", " : "") + std::to_string(shape[d]);
                throw py::value_error("override_shapes for operand " + std::to_string(i) +
                                      ": an fp4 tensor packs two elements per storage slot, so its unit-stride extent "
                                      "must be even; got (" +
                                      geom + ")");
            }
            Operand &operand = operands_[i];
            if (!operand.filled) {
                throw py::value_error("variant-pack operand " + std::to_string(i) + " has no buffer to re-describe");
            }
            // the buffer's EFFECTIVE strides: an empty vector is valid DLPack for compact row-major storage
            const std::vector<int64_t> reference =
                operand.stride.empty() ? dense_stride_of(operand.shape) : operand.stride;
            in_axis_order_of(shape, stride, reference);
            override_operand(i, std::move(shape), std::move(stride), dtype.code, dtype.bits, dtype.lanes);
        }
    }

    void
    skip_operand(size_t index) {
        operands_.at(index).filled = false;
        pointers_[index]           = nullptr;
    }

    // The declaration is the contract (see _pygraph._normalize). For a filled
    // operand that has one: OTHER extents that are one dense run of slots and
    // cover the declared bytes are re-described AS the declaration, dtype
    // included (a byte blob covering a bf16 output becomes that output; what a
    // bare address gets). The declared extents keep the caller's strides, and
    // take the declared dtype when the slots are as wide (FlashInfer binds fp4
    // and fp8 blocks as uint8; the backend reads a pointer and never knew). A
    // strided view of other extents, or a buffer too small, keeps its own
    // description entirely. `skip` names the slots already described from the
    // graph (a bare address). Returns the indices whose extents were re-described.
    std::vector<size_t>
    describe_from(const DeclaredLayout &declared, const std::vector<size_t> &skip) {
        std::vector<size_t> described;
        const auto &slots = declared.slots();
        for (size_t i = 0; i < operands_.size() && i < slots.size(); i++) {
            const DeclaredLayout::Slot &want = slots[i];
            Operand &operand                 = operands_[i];
            if (!want.present || want.slot_bytes <= 0 || !operand.filled) continue;
            if (std::find(skip.begin(), skip.end(), i) != skip.end()) continue;
            // BYTES, not slots: a uint8 view spanning as many slots as a bf16
            // declaration covers half its bytes. A width the producer did not
            // spell (bits == 0) never qualifies.
            const int64_t own_bytes = (static_cast<int64_t>(operand.dtype.bits) * operand.dtype.lanes + 7) / 8;
            if (own_bytes <= 0) continue;
            if (want.preserve_extent) {
                if (own_bytes == want.slot_bytes && want.dtype.bits > 0) operand.dtype = want.dtype;
                if (want.shape.size() == 3 && operand.shape.size() != 3) {
                    const int64_t elements = numel_of(operand.shape);
                    const int64_t batches  = want.shape[0];
                    if (operand_contiguous(i) && batches > 0 && elements % batches == 0) {
                        const int64_t per_batch = elements / batches;
                        operand.ndim            = 3;
                        operand.shape           = {batches, per_batch, 1};
                        operand.stride          = {per_batch, 1, 1};
                    }
                }
                continue;
            }
            if (operand.shape == want.shape) {
                if (own_bytes == want.slot_bytes && want.dtype.bits > 0) operand.dtype = want.dtype;
                continue;
            }
            const int64_t own_span =
                operand.stride.empty() ? numel_of(operand.shape) : span_of(operand.shape, operand.stride);
            if (own_span != numel_of(operand.shape)) continue;
            if (own_span * own_bytes < want.span * want.slot_bytes) continue;
            operand.ndim   = static_cast<int32_t>(want.shape.size());
            operand.shape  = want.shape;
            operand.stride = want.stride;
            if (want.dtype.bits > 0) operand.dtype = want.dtype;
            described.push_back(i);
        }
        return described;
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

    // The producer's guaranteed span in BYTES (-1 unknown) and DLPack (device_type, device_id) (-1, -1 unknown).
    int64_t
    observed_bytes(size_t index) const {
        return operands_.at(index).observed_bytes;
    }

    // Build the consumer's immutable records in one native crossing, without
    // intermediate Python lists or a second observation/validation path.
    // Effective dtype/geometry and observed producer span/device remain separate.
    py::list
    facts_as(const std::vector<size_t> &indices, const py::object &constructor, const py::dict &dtype_names) const {
        py::list out(indices.size());
        const py::str unknown_dtype("");
        for (size_t i = 0; i < indices.size(); ++i) {
            const size_t index     = indices[i];
            const Operand &operand = operands_.at(index);
            py::tuple shape(operand.shape.size());
            for (size_t d = 0; d < operand.shape.size(); ++d) shape[d] = py::int_(operand.shape[d]);

            py::tuple strides(operand.stride.empty() ? operand.shape.size() : operand.stride.size());
            write_effective_strides(operand, [&strides](size_t d, int64_t value) { strides[d] = py::int_(value); });
            const auto dtype_key =
                py::make_tuple(static_cast<int>(operand.dtype.code), static_cast<int>(operand.dtype.bits));
            PyObject *dtype = PyDict_GetItem(dtype_names.ptr(), dtype_key.ptr());
            // Match the facts consumer: producer bytes in the EFFECTIVE element width.
            const int64_t width = std::max<int64_t>(1, (static_cast<int64_t>(operand.dtype.bits) + 7) / 8);
            const int64_t span  = operand.observed_bytes < 0 ? -1 : operand.observed_bytes / width;
            out[i] =
                constructor(reinterpret_cast<int64_t>(pointers_.at(index)),
                            dtype == nullptr ? py::object(unknown_dtype) : py::reinterpret_borrow<py::object>(dtype),
                            py::make_tuple(operand.observed_device_type, operand.observed_device_id),
                            span,
                            shape,
                            strides);
        }
        return out;
    }

    std::pair<int32_t, int32_t>
    observed_device(size_t index) const {
        const Operand &operand = operands_.at(index);
        return {operand.observed_device_type, operand.observed_device_id};
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
        write_effective_strides(operand, [&dense](size_t d, int64_t value) { dense[d] = value; });
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

    py::class_<DeclaredLayout>(m, "DeclaredLayout", R"(
The storage-slot geometry each variant-pack slot was declared with, built once
per graph; ``VariantPackNative.describe_from`` compares a whole pack against it.
)")
        .def(py::init<size_t>())
        .def("set",
             &DeclaredLayout::set,
             py::arg("index"),
             py::arg("shape"),
             py::arg("stride"),
             py::arg("slot_bytes"),
             py::arg("dtype_code")      = 0,
             py::arg("dtype_bits")      = 0,
             py::arg("dtype_lanes")     = 1,
             py::arg("preserve_extent") = false)
        .def("set_dtype",
             &DeclaredLayout::set_dtype,
             py::arg("index"),
             py::arg("dtype_code"),
             py::arg("dtype_bits"),
             py::arg("dtype_lanes") = 1)
        .def("__len__", &DeclaredLayout::size);

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
        .def("set_operand",
             &VariantPackNative::set_operand,
             py::arg("index"),
             py::arg("ptr"),
             py::arg("shape"),
             py::arg("stride"),
             py::arg("dtype_code"),
             py::arg("dtype_bits"),
             py::arg("dtype_lanes")          = 1,
             py::arg("observed_bytes")       = -1,
             py::arg("observed_device_type") = -1,
             py::arg("observed_device_id")   = -1)
        .def("override_operand",
             &VariantPackNative::override_operand,
             py::arg("index"),
             py::arg("shape"),
             py::arg("stride"),
             py::arg("dtype_code")  = 0,
             py::arg("dtype_bits")  = 0,
             py::arg("dtype_lanes") = 1)
        .def("override_many",
             &VariantPackNative::override_many,
             py::arg("declared"),
             py::arg("indices"),
             py::arg("shapes"),
             py::arg("strides"))
        .def("describe_from", &VariantPackNative::describe_from)
        .def("skip_operand", &VariantPackNative::skip_operand)
        .def("operand_contiguous", &VariantPackNative::operand_contiguous)
        .def("is_filled", &VariantPackNative::is_filled)
        .def("pointer", &VariantPackNative::pointer)
        .def("observed_bytes", &VariantPackNative::observed_bytes)
        .def("_facts_as", &VariantPackNative::facts_as)
        .def("observed_device", &VariantPackNative::observed_device)
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
