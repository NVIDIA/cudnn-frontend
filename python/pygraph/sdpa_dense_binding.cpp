// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Bounded SM100 half decode binding. Pure geometry uses the Python admission
// predicates on a cache miss; every invocation checks fresh storage observations.
#include "variant_pack.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace cudnn_frontend {
namespace python_bindings {
namespace {
[[noreturn]] void
invalid(const std::string &message) {
    throw py::value_error("cudnn.sdpa: " + message);
}
int64_t
multiply(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
        invalid("operand geometry must be nonnegative and fit in int64");
    return a * b;
}
int64_t
add(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b)
        invalid("operand geometry must be nonnegative and fit in int64");
    return a + b;
}
enum Role : size_t { Q, K, V, O, LSE, Sinks, KVLens, QLens, KTable, VTable, Gate, NumRoles };
constexpr std::array<const char *, NumRoles> names =
    {"q", "k", "v", "o", "lse_tensor", "sinks", "seq_kv_lens", "seq_q_lens", "block_table", "block_table_v", "gate"};
enum Slot : size_t {
    QPtr,
    KPtr,
    VPtr,
    OPtr,
    QStrides,
    KStrides,
    VStrides,
    OStrides,
    LSEPtr,
    LSEStrides,
    SinksPtr,
    MetaPtr,
    ODescPtr,
    QLensPtr,
    KTablePtr,
    VTablePtr,
    KTableStrides,
    VTableStrides,
    NPages,
    ProblemSize,
    Stream,
    NumSlots
};
constexpr std::array<const char *, NumSlots> slot_names = {
    "q_ptr",         "k_ptr",           "v_ptr",           "o_ptr",
    "q_strides",     "k_strides",       "v_strides",       "o_strides",
    "lse_ptr",       "lse_strides",     "sinks_ptr",       "meta_ptr",
    "o_desc_ptr",    "seq_q_lens_addr", "block_table_ptr", "block_table_v_ptr",
    "table_strides", "table_v_strides", "n_pages",         "problem_size",
    "stream"};
struct BoundGeometry {
    int64_t extent0 = 0, extent1 = 0, need = 0;
    py::tuple bound;
};
// One immutable geometry result per role, replaced after successful validation.
// Runtime owners, addresses, device observations and byte spans never enter it.
struct Geometry : BoundGeometry {
    std::vector<int64_t> shape, strides;
    int64_t batch = -1, sequence = -1;
    bool valid = false;
};
class SdpaDenseBinder {
   public:
    explicit SdpaDenseBinder(const py::object &spec)
        : fn_(spec.attr("fn")), owner_(spec.attr("owner")), template_(py::tuple(spec.attr("template"))) {
        auto integer = [&](const char *name) { return spec.attr(name).cast<int64_t>(); };
        auto flag    = [&](const char *name) { return spec.attr(name).cast<bool>(); };
        if (integer("split") != 1 || flag("ragged") || flag("has_sink") || !spec.attr("quant").is_none() ||
            !spec.attr("gate_expect").is_none() || integer("d_qk") != 128 || integer("d_v") != 128 ||
            integer("s_q_max") != 1)
            invalid("native dense binding requires unsplit half D128 decode without ragged Q, sinks or gate");
        b_            = integer("b");
        qh_           = integer("qh");
        kh_           = integer("kh");
        sk_           = integer("s_k_max");
        device_       = integer("device_index");
        page_size_    = integer("page_size");
        tile_n_       = integer("tile_n");
        window_right_ = integer("window_right");
        paged_        = flag("paged");
        hnd_          = flag("paged_hnd");
        has_lse_      = flag("has_lse");
        seq_kv_       = flag("seq_kv_present");
        seq_q_        = flag("seq_q_present");
        shape_fixed_  = flag("shape_fixed");
        lpt_fixed_    = flag("lpt_grid_fixed");
        tail_native_  = flag("kv_tail_native");
        causal_       = flag("causal");
        bottom_right_ = flag("causal_bottom_right");
        if (b_ <= 0 || qh_ <= 0 || kh_ <= 0 || sk_ <= 0 || device_ < 0 || tile_n_ <= 0 || (paged_ && page_size_ <= 0))
            invalid("invalid native dense plan geometry");
        auto expect = spec.attr("expect").cast<py::dict>();
        for (size_t i = Q; i <= O; ++i) {
            const auto dtype = expect[names[i]].cast<std::string>();
            if (dtype != "float16" && dtype != "bfloat16") invalid("native dense binding requires half operands");
            dtype_code_[i] = dtype == "float16" ? kDLFloat : kDLBfloat;
        }
        auto order = spec.attr("order").cast<std::vector<std::string>>();
        if (order.size() != template_.size()) invalid("native dense host template has the wrong size");
        for (size_t slot = 0; slot < NumSlots; ++slot) {
            auto found = std::find(order.begin(), order.end(), slot_names[slot]);
            if (found == order.end() && slot != VTableStrides)
                invalid(std::string("native dense host has no argument ") + slot_names[slot]);
            index_[slot] = static_cast<size_t>(found - order.begin());
        }
        auto prep     = py::module_::import("cudnn.sdpa.fwd.prepared");
        dense_layout_ = prep.attr("_dense_role_layout");
        pool_layout_  = prep.attr("_paged_pool_layout");
        table_layout_ = prep.attr("_paged_table_layout");
        lse_layout_   = prep.attr("_dense_lse_layout");
    }
    py::tuple
    bind(const py::handle &pack, const std::vector<int64_t> &indices, py::object stream) {
        if (indices.size() != NumRoles) invalid("native dense binding requires eleven role indices");
        const auto facts = read_native_operand_views(pack, indices);
        py::tuple frame(template_.size());
        for (size_t i = 0; i < template_.size(); ++i) frame[i] = template_[i];
        for (size_t i = Q; i <= O; ++i) {
            operand(facts[i], i, dtype_code_[i], 16, 16);
            put(frame, static_cast<Slot>(QPtr + i), py::int_(facts[i].pointer));
        }
        auto q = geometry(facts[Q], Q), o = geometry(facts[O], O);
        const int64_t b = q.extent0, sq = q.extent1;
        if (o.extent0 != b || o.extent1 != sq) invalid("o must match q batch and sequence extents");
        put(frame, QStrides, q.bound);
        put(frame, OStrides, o.bound);
        int64_t sk;
        auto k = geometry(facts[K], K), v = geometry(facts[V], V);
        put(frame, KStrides, k.bound);
        put(frame, VStrides, v.bound);
        if (paged_) {
            auto kt = table(facts[KTable], KTable, b), vt = table(facts[VTable], VTable, b);
            if (kt.extent1 != vt.extent1) invalid("K and V page tables must share max_pages");
            if (k.extent0 != v.extent0) invalid("K and V pools must hold the same number of pages");
            if (index_[VTableStrides] == template_.size() && !kt.bound.equal(vt.bound))
                invalid("this prepared host requires matching K/V table strides");
            sk = multiply(kt.extent1, page_size_);
            put(frame, KTablePtr, py::int_(facts[KTable].pointer));
            put(frame, VTablePtr, py::int_(facts[VTable].pointer));
            put(frame, KTableStrides, kt.bound);
            if (index_[VTableStrides] != template_.size()) put(frame, VTableStrides, vt.bound);
            put(frame, NPages, py::int_(k.extent0));
        } else {
            if (k.extent0 != b || v.extent0 != b || k.extent1 != v.extent1)
                invalid("k / v must match q batch and share sequence extent");
            sk = k.extent1;
        }
        if (shape_fixed_ && sk != sk_) invalid("this artifact requires the declared sequence extents");
        if (lpt_fixed_ && b != b_) invalid("this artifact requires the declared batch extent");
        if (!(tail_native_ || paged_ || sk % tile_n_ == 0 || seq_kv_ ||
              (causal_ && ((bottom_right_ && window_right_ == 0) || (!bottom_right_ && window_right_ <= sk - sq)))))
            invalid("S_kv must be a tile multiple unless device KV lengths or the causal mask cover its tail");
        put(frame, ProblemSize, py::make_tuple(b, qh_, kh_, sq, sk, 0));
        if (has_lse_) {
            operand(facts[LSE], LSE, kDLFloat, 32, 4);
            auto lse = layout(facts[LSE], LSE, b, sq);
            check_span(facts[LSE], lse.need, LSE);
            put(frame, LSEPtr, py::int_(facts[LSE].pointer));
            put(frame, LSEStrides, lse.bound);
        } else {
            if (facts[LSE].filled) invalid("this specialization was compiled without a Stats output");
            put(frame, LSEPtr, py::none());
            put(frame, LSEStrides, py::make_tuple(0, 0, 0));
        }
        if (facts[Sinks].filled) invalid("this specialization was compiled without a sink");
        if (facts[Gate].filled) invalid("this specialization was compiled without an epilogue gate");
        put(frame, SinksPtr, py::int_(0));
        put(frame, ODescPtr, py::int_(0));
        put(frame, MetaPtr, py::int_(seq_kv_ ? lengths(facts[KVLens], KVLens, b) : 0));
        if (seq_q_) put(frame, QLensPtr, py::int_(lengths(facts[QLens], QLens, b)));
        put(frame, Stream, std::move(stream));
        return frame;
    }
    void
    execute(const py::handle &pack, const std::vector<int64_t> &indices, py::object stream) {
        auto frame  = bind(pack, indices, std::move(stream));
        auto result = py::reinterpret_steal<py::object>(PyObject_CallObject(fn_.ptr(), frame.ptr()));
        if (!result) throw py::error_already_set();
    }
    py::object
    execute_ordered(py::handle schema,
                    py::handle buffers,
                    py::handle tensor_uids,
                    const py::dict &auto_bindings,
                    py::handle workspace,
                    py::handle override_uids,
                    py::handle override_shapes,
                    py::handle override_strides,
                    const std::vector<int64_t> &indices,
                    py::object stream) {
        auto read = read_ordered_binding(
            schema, buffers, tensor_uids, auto_bindings, workspace, override_uids, override_shapes, override_strides);
        // Supplied workspace is observed even when this bounded plan needs no
        // scratch. Python completes unsupported producer protocols before the
        // same binder runs; malformed buffers never trigger another executor.
        if (!read[1].cast<py::list>().empty() || read[2].is_none()) return read;
        execute(read[0], indices, std::move(stream));
        return py::none();
    }

   private:
    void
    operand(const NativeOperandView &f, size_t role, int code, int bits, int alignment) const {
        const std::string name = names[role];
        if (!f.filled) invalid(name + " is required");
        if (f.dtype.code != code || f.dtype.bits != bits || f.dtype.lanes != 1)
            invalid(name + ": runtime buffer dtype does not match its declaration");
        if (f.device_type != -1 && (f.device_type != kDLCUDA || f.device_id != device_))
            invalid(name + " must be on this plan's CUDA device");
        if (f.pointer % alignment) invalid(name + ": runtime buffer address is misaligned");
    }
    static void
    check_span(const NativeOperandView &f, int64_t need, size_t role) {
        const int64_t width = (static_cast<int64_t>(f.dtype.bits) + 7) / 8;
        if (f.observed_bytes >= 0 && f.observed_bytes / width < need)
            invalid(std::string(names[role]) + ": observed storage is too small for its effective geometry");
    }
    BoundGeometry
    layout(const NativeOperandView &f, size_t role, int64_t b = -1, int64_t sq = -1) {
        auto &cached = geometry_[role];
        if (cached.valid && cached.batch == b && cached.sequence == sq && !f.stride.empty() &&
            std::equal(cached.shape.begin(), cached.shape.end(), f.shape.begin(), f.shape.end()) &&
            std::equal(cached.strides.begin(), cached.strides.end(), f.stride.begin(), f.stride.end()))
            return cached;
        std::vector<int64_t> shape(f.shape.begin(), f.shape.end()), strides;
        if (f.stride.empty()) {
            strides.reserve(shape.size());
            int64_t value = 1;
            for (size_t i = shape.size(); i-- > 0;) {
                strides.push_back(value);
                value = multiply(value, shape[i]);
            }
            std::reverse(strides.begin(), strides.end());
        } else {
            strides.assign(f.stride.begin(), f.stride.end());
        }
        if (strides.size() != shape.size()) invalid("operand shape and stride must have the same rank");
        if (cached.valid && cached.shape == shape && cached.strides == strides && cached.batch == b &&
            cached.sequence == sq)
            return cached;
        Geometry result;
        result.shape    = std::move(shape);
        result.strides  = std::move(strides);
        result.batch    = b;
        result.sequence = sq;
        auto sh = py::cast(result.shape), st = py::cast(result.strides);
        // Python's lru_cache requires hashable arguments. These are allocated only
        // for a new effective layout, never for the warm stable geometry.
        auto shape_tuple = py::tuple(sh), stride_tuple = py::tuple(st);
        if (role == LSE) {
            auto value   = lse_layout_(shape_tuple, stride_tuple, b, sq, qh_).cast<py::tuple>();
            result.bound = value[0].cast<py::tuple>();
            result.need  = value[1].cast<int64_t>();
        } else if (role == KTable || role == VTable) {
            auto value     = table_layout_(shape_tuple, stride_tuple).cast<py::tuple>();
            auto extents   = value[0].cast<std::array<int64_t, 2>>();
            result.extent0 = extents[0];
            result.extent1 = extents[1];
            result.bound   = value[1].cast<py::tuple>();
            if (result.extent0 < b) invalid("page table batch extent is smaller than q batch");
            auto ts     = result.bound.cast<std::array<int64_t, 2>>();
            result.need = add(add(multiply(b - 1, ts[0]), multiply(result.extent1 - 1, ts[1])), 1);
        } else if (paged_ && (role == K || role == V)) {
            auto value     = pool_layout_(shape_tuple, stride_tuple, 2, hnd_, kh_, page_size_, 128).cast<py::tuple>();
            result.bound   = value[0].cast<py::tuple>();
            result.need    = value[1].cast<int64_t>();
            result.extent0 = result.shape[0];
        } else {
            auto value = dense_layout_(shape_tuple,
                                       stride_tuple,
                                       role == Q || role == O ? qh_ : kh_,
                                       128,
                                       role == Q || role == O ? 1 : sk_,
                                       b_,
                                       2,
                                       true,
                                       names[role])
                             .cast<py::tuple>();
            result.bound   = value[0].cast<py::tuple>();
            result.extent0 = value[1].cast<int64_t>();
            result.extent1 = value[2].cast<int64_t>();
            result.need    = value[3].cast<int64_t>();
        }
        result.valid = true;
        cached       = result;
        return result;
    }
    BoundGeometry
    geometry(const NativeOperandView &f, size_t role) {
        auto result = layout(f, role);
        check_span(f, result.need, role);
        return result;
    }
    BoundGeometry
    table(const NativeOperandView &f, size_t role, int64_t b) {
        operand(f, role, kDLInt, 32, 4);
        if (!f.pointer) invalid(std::string(names[role]) + " must have a non-null address");
        auto result = layout(f, role, b);
        check_span(f, result.need, role);
        return result;
    }
    int64_t
    lengths(const NativeOperandView &f, size_t role, int64_t b) const {
        operand(f, role, kDLInt, 32, 4);
        if (!f.stride.empty() && f.stride.size() != f.shape.size())
            invalid("operand shape and stride must have the same rank");
        int64_t n = 1;
        for (size_t i = f.shape.size(); i-- > 0;) {
            if (!f.stride.empty() && f.shape[i] != 1 && f.stride[i] != n)
                invalid(std::string(names[role]) + " must be contiguous");
            n = multiply(n, f.shape[i]);
        }
        if (n < b) invalid(std::string(names[role]) + " must hold at least batch elements");
        check_span(f, b, role);
        return f.pointer;
    }
    void
    put(py::tuple &frame, Slot slot, py::object value) const {
        frame[index_[slot]] = std::move(value);
    }
    py::object fn_, owner_, dense_layout_, pool_layout_, table_layout_, lse_layout_;
    py::tuple template_;
    std::array<size_t, NumSlots> index_;
    std::array<int, 4> dtype_code_;
    std::array<Geometry, NumRoles> geometry_;
    int64_t b_, qh_, kh_, sk_, device_, page_size_, tile_n_, window_right_;
    bool paged_, hnd_, has_lse_, seq_kv_, seq_q_, shape_fixed_, lpt_fixed_, tail_native_, causal_, bottom_right_;
};
}  // namespace
void
init_sdpa_dense_binding(py::module_ &m) {
    py::class_<SdpaDenseBinder>(m, "_SdpaDenseBinder")
        .def(py::init<const py::object &>(), py::arg("spec"))
        .def("bind", &SdpaDenseBinder::bind, py::arg("pack"), py::arg("indices"), py::arg("stream"))
        .def("execute", &SdpaDenseBinder::execute, py::arg("pack"), py::arg("indices"), py::arg("stream"))
        .def("execute_ordered",
             &SdpaDenseBinder::execute_ordered,
             py::arg("schema"),
             py::arg("buffers"),
             py::arg("tensor_uids"),
             py::arg("auto_bindings"),
             py::arg("workspace"),
             py::arg("override_uids"),
             py::arg("override_shapes"),
             py::arg("override_strides"),
             py::arg("indices"),
             py::arg("stream"));
}
}  // namespace python_bindings
}  // namespace cudnn_frontend
