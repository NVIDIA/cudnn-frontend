// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The variant pack, held as DLTensors rather than python objects.
//
// execute() reads the caller's operands once and everything below works from
// the result. Doing that in python cost 1.5 us per operand -- 0.6 to ask the
// buffer for its facts one method call at a time, 0.9 to build the Tensor that
// carries them -- and each operand then had to be turned back into a DLPack
// producer for the kernel, which python cannot do quickly: tvm-ffi reads a
// torch tensor through a C function table and any python producer through a
// freshly built capsule, 0.32 us against 1.91.
//
// Both halves are the same protocol. `__dlpack_c_exchange_api__` is a vtable
// on the TYPE whose dltensor_from_py_object_no_sync fills a caller-provided
// DLTensor in place, with no capsule and no allocation. This file consumes it
// to read the caller's buffers and implements it so the slots it hands out are
// read the same way. Measured on SM100, eight operands: reading 0.64 us
// against 10.4, contiguity 0.13 against 4.2, and a slot converts in 0.27 --
// slightly cheaper than the torch tensor it replaces, so nothing is given up
// by refusing to pass the caller's object through.
//
// A producer without the vtable is not an error: `read_slot` returns false and
// python fills that slot from its own reader. The degradation is per operand,
// so a pack mixing torch with something else is exactly as fast as its parts.

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
    if (cache.count < kTypeCacheSlots) {
        // Keyed on the type's ADDRESS, so the entry must own a reference: a
        // heap type that got collected could be replaced by a different type
        // allocated at the same address, and this would hand out its vtable.
        // The cache never evicts, so this pins at most kTypeCacheSlots types.
        Py_INCREF(type);
        cache.types[cache.count] = type;
        cache.apis[cache.count]  = api;  // a null answer is worth caching too
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
// pointers so a slot stays valid once the producer's own DLTensor is gone.
struct Slot {
    void *data       = nullptr;
    int32_t ndim     = 0;
    DLDataType dtype = {0, 0, 1};
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;  // empty means compact row-major
    bool filled = false;
};

}  // namespace

// A pack's slot, exposed to a kernel. It implements the same exchange protocol
// it was read through, so a consumer that has the fast path for a torch tensor
// has it for this too.
class VariantPackSlot {
   public:
    VariantPackSlot(Slot slot, int32_t device_id) : slot_(std::move(slot)) {
        tensor_.data        = slot_.data;
        tensor_.device      = DLDevice{kDLCUDA, device_id};
        tensor_.ndim        = slot_.ndim;
        tensor_.dtype       = slot_.dtype;
        tensor_.shape       = slot_.shape.empty() ? nullptr : slot_.shape.data();
        tensor_.strides     = slot_.stride.empty() ? nullptr : slot_.stride.data();
        tensor_.byte_offset = 0;
    }

    // tensor_ points into slot_'s vectors, so a copy would leave the new
    // object's DLTensor describing the old one's storage. Slots are always
    // heap-allocated and handed out by pointer, so nothing needs to copy one.
    VariantPackSlot(const VariantPackSlot &) = delete;
    VariantPackSlot &
    operator=(const VariantPackSlot &)  = delete;
    VariantPackSlot(VariantPackSlot &&) = delete;
    VariantPackSlot &
    operator=(VariantPackSlot &&) = delete;

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
        return slot_.shape;
    }

    std::vector<int64_t>
    stride() const {
        if (!slot_.stride.empty()) return slot_.stride;
        std::vector<int64_t> dense(slot_.ndim, 1);
        for (int d = slot_.ndim - 2; d >= 0; d--) dense[d] = dense[d + 1] * slot_.shape[d + 1];
        return dense;
    }

    // The bare dtype NAME, which is what a kernel means when it asks a buffer
    // for its dtype: they all reach it through str(x.dtype).split(".")[-1], so
    // a torch tensor's "torch.bfloat16" and this "bfloat16" answer the same.
    std::string
    dtype() const {
        return dtype_name(slot_.dtype);
    }

    int64_t
    element_size() const {
        return slot_.dtype.bits / 8;
    }

    int64_t
    numel() const {
        int64_t n = 1;
        for (int64_t extent : slot_.shape) n *= extent;
        return n;
    }

    int64_t
    nbytes() const {
        return numel() * element_size();
    }

    int64_t
    length() const {
        return slot_.shape.empty() ? 0 : slot_.shape[0];
    }

    py::tuple
    dlpack_device() const {
        return py::make_tuple(static_cast<int>(kDLCUDA), tensor_.device.device_id);
    }

    // A differently shaped view of the same memory, with one -1 wildcard. Only
    // meaningful for a dense slot, which is why a strided one is refused rather
    // than silently reinterpreted.
    VariantPackSlot *
    reshape(std::vector<int64_t> shape) const {
        if (!slot_.stride.empty() && !is_dense(tensor_))
            throw py::value_error("cannot reshape a non-contiguous variant-pack slot");
        int64_t numel = 1;
        for (int64_t extent : slot_.shape) numel *= extent;
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
        Slot out  = slot_;
        out.shape = std::move(shape);
        out.ndim  = static_cast<int32_t>(out.shape.size());
        out.stride.clear();  // dense by construction, as the reshape required
        return new VariantPackSlot(out, tensor_.device.device_id);
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
    // outlives this slot rather than aliasing storage the slot owns.
    py::capsule
    dlpack(py::object /*stream*/, py::object /*max_version*/) const {
        struct Owned {
            DLManagedTensor managed;
            std::vector<int64_t> shape;
            std::vector<int64_t> stride;
        };
        auto *owned                      = new Owned{{}, slot_.shape, slot_.stride};
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
    Slot slot_;  // owns the shape/stride storage the DLTensor points into
    DLTensor tensor_{};
};

namespace {

int
slot_dltensor_from_py_object(void *py_object, DLTensor *out) {
    auto *slot = py::cast<VariantPackSlot *>(py::handle(static_cast<PyObject *>(py_object)));
    *out       = slot->tensor();
    return 0;
}

int
slot_managed_from_py_object(void *py_object, DLManagedTensorVersioned **out) {
    auto *slot    = py::cast<VariantPackSlot *>(py::handle(static_cast<PyObject *>(py_object)));
    auto *managed = static_cast<DLManagedTensorVersioned *>(std::calloc(1, sizeof(DLManagedTensorVersioned)));
    if (managed == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    managed->version.major = DLPACK_MAJOR_VERSION;
    managed->version.minor = DLPACK_MINOR_VERSION;
    managed->dl_tensor     = slot->tensor();
    managed->manager_ctx   = nullptr;
    managed->deleter       = [](DLManagedTensorVersioned *self) { std::free(self); };
    *out                   = managed;
    return 0;
}

int
slot_allocator(DLTensor *,
               DLManagedTensorVersioned **,
               void *error_ctx,
               void (*set_error)(void *, const char *, const char *)) {
    set_error(error_ctx, "NotImplementedError", "a variant-pack slot views the caller's memory; it never allocates");
    return -1;
}

int
slot_to_py_object(DLManagedTensorVersioned *, void **) {
    PyErr_SetString(PyExc_NotImplementedError, "a variant-pack slot is not an importer");
    return -1;
}

// The graph launches on the stream its handle carries, which execute() passes
// explicitly. Reporting no producer stream is what tells a consumer to use the
// one it was given rather than going looking for ours.
int
slot_current_work_stream(DLDeviceType, int32_t, void **out_stream) {
    *out_stream = nullptr;
    return 0;
}

DLPackExchangeAPI &
slot_exchange_api() {
    static DLPackExchangeAPI api = [] {
        DLPackExchangeAPI table{};
        table.header.version.major                  = DLPACK_MAJOR_VERSION;
        table.header.version.minor                  = DLPACK_MINOR_VERSION;
        table.header.prev_api                       = nullptr;
        table.managed_tensor_allocator              = slot_allocator;
        table.managed_tensor_from_py_object_no_sync = slot_managed_from_py_object;
        table.managed_tensor_to_py_object_no_sync   = slot_to_py_object;
        table.dltensor_from_py_object_no_sync       = slot_dltensor_from_py_object;
        table.current_work_stream                   = slot_current_work_stream;
        return table;
    }();
    return api;
}

}  // namespace

class VariantPackNative {
   public:
    explicit VariantPackNative(size_t n) : slots_(n), pointers_(n, nullptr) {}

    // Fill one slot from the caller's buffer. False means its type does not
    // implement the exchange protocol and python must describe it instead.
    bool
    read_slot(size_t index, py::handle buffer) {
        Slot &slot             = slots_.at(index);
        DLPackExchangeAPI *api = exchange_api_for(buffer.ptr());
        if (api == nullptr || api->dltensor_from_py_object_no_sync == nullptr) return false;
        DLTensor t{};
        if (api->dltensor_from_py_object_no_sync(buffer.ptr(), &t) != 0) throw py::error_already_set();
        slot.data  = static_cast<char *>(t.data) + t.byte_offset;
        slot.ndim  = t.ndim;
        slot.dtype = t.dtype;
        slot.shape.assign(t.shape, t.shape + t.ndim);
        if (t.strides != nullptr) {
            slot.stride.assign(t.strides, t.strides + t.ndim);
        } else {
            slot.stride.clear();
        }
        slot.filled      = true;
        pointers_[index] = slot.data;
        return true;
    }

    // Every slot in one call. Returns the indices whose producer has no vtable,
    // for python to describe and report back through set_slot -- crossing the
    // binding once per pack rather than once per operand is 2.53 us against
    // 1.0 for eight. A None entry is a slot the caller did not fill.
    std::vector<size_t>
    read_all(py::sequence buffers) {
        std::vector<size_t> unread;
        const size_t n = py::len(buffers);
        for (size_t i = 0; i < n && i < slots_.size(); i++) {
            py::handle buffer = buffers[i];
            if (buffer.is_none()) {
                skip_slot(i);
            } else if (!read_slot(i, buffer)) {
                unread.push_back(i);
            }
        }
        return unread;
    }

    // The fallback: python read the buffer its own way and reports the result.
    void
    set_slot(size_t index,
             int64_t ptr,
             std::vector<int64_t> shape,
             std::vector<int64_t> stride,
             int dtype_code,
             int dtype_bits) {
        Slot &slot       = slots_.at(index);
        slot.data        = reinterpret_cast<void *>(ptr);
        slot.ndim        = static_cast<int32_t>(shape.size());
        slot.dtype       = DLDataType{static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits), 1};
        slot.shape       = std::move(shape);
        slot.stride      = std::move(stride);
        slot.filled      = true;
        pointers_[index] = slot.data;
    }

    // A slot the caller did not fill: an optional port it did not request.
    void
    skip_slot(size_t index) {
        slots_.at(index).filled = false;
        pointers_[index]        = nullptr;
    }

    bool
    all_contiguous(std::string &offender) const {
        for (size_t i = 0; i < slots_.size(); i++) {
            const Slot &slot = slots_[i];
            if (!slot.filled || slot.stride.empty()) continue;
            int64_t expect = 1;
            for (int d = slot.ndim - 1; d >= 0; d--) {
                if (slot.shape[d] != 1 && slot.stride[d] != expect) {
                    offender = std::to_string(i);
                    return false;
                }
                expect *= slot.shape[d];
            }
        }
        return true;
    }

    bool
    slot_contiguous(size_t index) const {
        const Slot &slot = slots_.at(index);
        if (!slot.filled || slot.stride.empty()) return true;
        int64_t expect = 1;
        for (int d = slot.ndim - 1; d >= 0; d--) {
            if (slot.shape[d] != 1 && slot.stride[d] != expect) return false;
            expect *= slot.shape[d];
        }
        return true;
    }

    bool
    is_filled(size_t index) const {
        return slots_.at(index).filled;
    }

    int64_t
    pointer(size_t index) const {
        return reinterpret_cast<int64_t>(pointers_.at(index));
    }

    std::vector<int64_t>
    shape(size_t index) const {
        return slots_.at(index).shape;
    }

    std::vector<int64_t>
    stride(size_t index) const {
        const Slot &slot = slots_.at(index);
        if (!slot.stride.empty()) return slot.stride;
        std::vector<int64_t> dense(slot.ndim, 1);
        for (int d = slot.ndim - 2; d >= 0; d--) dense[d] = dense[d + 1] * slot.shape[d + 1];
        return dense;
    }

    py::tuple
    dtype(size_t index) const {
        const Slot &slot = slots_.at(index);
        return py::make_tuple(slot.dtype.code, slot.dtype.bits);
    }

    // The address the backend's variant pack reads: a contiguous void*[] in
    // slot order, so it goes to _execute_with_raw_ptrs with no copy.
    int64_t
    pointer_array(void) const {
        return reinterpret_cast<int64_t>(pointers_.data());
    }

    size_t
    size(void) const {
        return slots_.size();
    }

    // Every requested slot in one crossing, for an engine binding a whole node.
    std::vector<VariantPackSlot *>
    views(const std::vector<size_t> &indices, int32_t device_id) const {
        std::vector<VariantPackSlot *> out;
        out.reserve(indices.size());
        for (size_t index : indices) out.push_back(view(index, device_id));
        return out;
    }

    VariantPackSlot *
    view(size_t index, int32_t device_id) const {
        const Slot &slot = slots_.at(index);
        if (!slot.filled)
            throw py::value_error("variant-pack slot " + std::to_string(index) + " was not filled by the caller");
        return new VariantPackSlot(slot, device_id);
    }

   private:
    std::vector<Slot> slots_;
    std::vector<void *> pointers_;
};

// A slot over memory that is not a caller operand: the regions a plan carves
// out of the workspace. Same type, so a kernel is handed one kind of buffer
// whether it came from the caller or from the workspace, and both are read
// through the exchange vtable rather than a per-call capsule.
VariantPackSlot *
make_slot(int64_t ptr, std::vector<int64_t> shape, int dtype_code, int dtype_bits, int32_t device_id) {
    Slot slot;
    slot.data   = reinterpret_cast<void *>(ptr);
    slot.ndim   = static_cast<int32_t>(shape.size());
    slot.dtype  = DLDataType{static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits), 1};
    slot.shape  = std::move(shape);
    slot.filled = true;  // stride left empty: a carve is dense by construction
    return new VariantPackSlot(std::move(slot), device_id);
}

// A workspace carve, planned once. Which regions a plan cuts, at what offsets,
// with what dtypes and shapes, is fixed when the engine builds; only the
// caller's base pointer arrives per execute. Describing them here rather than
// at each view() turns one crossing per region into one crossing per execute.
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
            Slot proto;
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

    std::vector<VariantPackSlot *>
    carve(int64_t base, int64_t nbytes, int32_t device_id) const {
        std::vector<VariantPackSlot *> out;
        out.reserve(protos_.size());
        for (size_t i = 0; i < protos_.size(); i++) {
            if (ends_[i] > nbytes) {
                throw py::value_error(owner_ + ": workspace overrun -- region [" + std::to_string(offsets_[i]) + ", " +
                                      std::to_string(ends_[i]) + ") exceeds the " + std::to_string(nbytes) +
                                      "-byte buffer (sizing bug)");
            }
            Slot slot = protos_[i];
            slot.data = reinterpret_cast<void *>(base + offsets_[i]);
            out.push_back(new VariantPackSlot(std::move(slot), device_id));
        }
        return out;
    }

    size_t
    size() const {
        return protos_.size();
    }

   private:
    std::string owner_;
    std::vector<Slot> protos_;
    std::vector<int64_t> offsets_;
    std::vector<int64_t> ends_;
};

void
init_variant_pack(py::module_ &m) {
    auto slot_class = py::class_<VariantPackSlot>(m, "VariantPackSlot", R"(
One operand of a variant pack, as a DLPack producer.

Implements ``__dlpack_c_exchange_api__``, so a consumer reads it through the
same C function table it uses for a framework tensor rather than through a
capsule built in python.
)")
                          .def("data_ptr", &VariantPackSlot::data_ptr)
                          .def_property_readonly("shape", &VariantPackSlot::shape)
                          .def_property_readonly("dtype", &VariantPackSlot::dtype)
                          .def_property_readonly("nbytes", &VariantPackSlot::nbytes)
                          .def("stride", &VariantPackSlot::stride)
                          .def("element_size", &VariantPackSlot::element_size)
                          .def("numel", &VariantPackSlot::numel)
                          .def("__len__", &VariantPackSlot::length)
                          .def("reshape",
                               [](const VariantPackSlot &self, py::args dims) {
                                   std::vector<int64_t> shape;
                                   if (dims.size() == 1 && py::isinstance<py::sequence>(dims[0]) &&
                                       !py::isinstance<py::int_>(dims[0])) {
                                       shape = dims[0].cast<std::vector<int64_t>>();
                                   } else {
                                       for (auto d : dims) shape.push_back(d.cast<int64_t>());
                                   }
                                   return self.reshape(std::move(shape));
                               })
                          .def("contiguous", [](py::object self) { return self; })
                          .def("__dlpack_device__", &VariantPackSlot::dlpack_device)
                          .def("__dlpack__",
                               &VariantPackSlot::dlpack,
                               py::kw_only(),
                               py::arg("stream")      = py::none(),
                               py::arg("max_version") = py::none());

    // The protocol looks the attribute up on the TYPE, and a pybind11 class is
    // a heap type, so it takes a plain setattr.
    PyObject *capsule = PyCapsule_New(&slot_exchange_api(), "dlpack_exchange_api", nullptr);
    if (capsule == nullptr) throw py::error_already_set();
    if (PyObject_SetAttrString(slot_class.ptr(), "__dlpack_c_exchange_api__", capsule) < 0) {
        Py_DECREF(capsule);
        throw py::error_already_set();
    }
    Py_DECREF(capsule);

    m.def("make_slot",
          &make_slot,
          py::arg("ptr"),
          py::arg("shape"),
          py::arg("dtype_code"),
          py::arg("dtype_bits"),
          py::arg("device_id"),
          "A DLPack producer over memory the caller did not supply -- a workspace carve.");

    py::class_<WorkspaceCarve>(m, "WorkspaceCarve", R"(
A workspace carve compiled once, at build.

Regions are ``(offset, dtype_code, dtype_bits, shape)``. Only the base pointer
arrives per execute, so ``carve`` hands back every region in one crossing
instead of one per region.
)")
        .def(py::init<std::string, std::vector<py::tuple>>(), py::arg("owner"), py::arg("regions"))
        .def("carve", &WorkspaceCarve::carve, py::arg("base"), py::arg("nbytes"), py::arg("device_id"))
        .def("__len__", &WorkspaceCarve::size);

    slot_class.attr("view") = slot_class.attr("reshape");

    py::class_<VariantPackNative>(m, "VariantPackNative", R"(
The caller's operands, held as DLTensors.

``read_slot`` returns False for a buffer whose type does not implement
``__dlpack_c_exchange_api__``; the caller describes that one itself and reports
it through ``set_slot``, so a pack mixing producers costs exactly the sum of
its parts.
)")
        .def(py::init<size_t>())
        .def("read_slot", &VariantPackNative::read_slot)
        .def("read_all", &VariantPackNative::read_all)
        .def("set_slot", &VariantPackNative::set_slot)
        .def("skip_slot", &VariantPackNative::skip_slot)
        .def("slot_contiguous", &VariantPackNative::slot_contiguous)
        .def("is_filled", &VariantPackNative::is_filled)
        .def("pointer", &VariantPackNative::pointer)
        .def("shape", &VariantPackNative::shape)
        .def("stride", &VariantPackNative::stride)
        .def("dtype", &VariantPackNative::dtype)
        .def("view", &VariantPackNative::view)
        .def("views", &VariantPackNative::views)
        .def_property_readonly("address", &VariantPackNative::pointer_array)
        .def("__len__", &VariantPackNative::size)
        .def("all_contiguous", [](const VariantPackNative &self) {
            std::string offender;
            bool ok = self.all_contiguous(offender);
            return py::make_tuple(ok, offender);
        });
}

}  // namespace python_bindings
}  // namespace cudnn_frontend
