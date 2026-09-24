// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Binding for the existing explicit-pointer f16 THD host. This changes no kernel
// ABI: a fresh positional frame goes to the same retained tvm-ffi Function. The
// native pack's observed storage/device and effective geometry are kept separate.
#include "variant_pack.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <unordered_map>

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
    if (a < 0 || b < 0 || (b != 0 && a > std::numeric_limits<int64_t>::max() / b)) {
        invalid("operand geometry must be nonnegative and fit in int64");
    }
    return a * b;
}

int64_t
add(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) {
        invalid("operand geometry must be nonnegative and fit in int64");
    }
    return a + b;
}

int64_t
numel(const NativeOperandView &f) {
    int64_t n = 1;
    for (int64_t extent : f.shape) n = multiply(n, extent);
    return n;
}

int64_t
stride(const NativeOperandView &f, size_t dim) {
    if (f.stride.size() != 0) {
        if (f.stride.size() != f.shape.size()) invalid("operand shape and stride must have the same rank");
        return f.stride[dim];
    }
    int64_t result = 1;
    for (size_t i = dim + 1; i < f.shape.size(); ++i) result = multiply(result, f.shape[i]);
    return result;
}

bool
contiguous(const NativeOperandView &f) {
    if (f.stride.empty()) return true;
    int64_t expected = 1;
    for (size_t i = f.shape.size(); i-- > 0;) {
        if (f.shape[i] != 1 && stride(f, i) != expected) return false;
        expected = multiply(expected, f.shape[i]);
    }
    return true;
}

int64_t
span(const NativeOperandView &f) {
    // Same conversion as VariantPackNative::_facts_as: the producer guarantees
    // BYTES, while capacity is measured in the effective declared element width.
    const int64_t width = std::max<int64_t>(1, (static_cast<int64_t>(f.dtype.bits) + 7) / 8);
    return f.observed_bytes < 0 ? -1 : f.observed_bytes / width;
}

bool
dtype_is(const NativeOperandView &f, int code, int bits) {
    return f.dtype.code == code && f.dtype.bits == bits && f.dtype.lanes == 1;
}

// Fixed role order shared by graph and standalone observation. -1 in an index
// table means an absent optional role; unfilled required roles fail below.
enum Role : size_t { Q, K, V, O, QLens, KVLens, LSE, Sinks, NumRoles };
constexpr std::array<const char *, NumRoles> names = {"q", "k", "v", "o", "q_lens", "kv_lens", "lse_tensor", "sinks"};

struct Geometry {
    int64_t token, head, element, row;
};

class SdpaThdBinder {
   public:
    explicit SdpaThdBinder(const py::object &spec)
        : fn_(spec.attr("fn")), owner_(spec.attr("owner")), template_(py::tuple(spec.attr("template"))) {
        if (spec.attr("paged").cast<bool>() || spec.attr("has_sink").cast<bool>() ||
            spec.attr("lse_padded").cast<bool>()) {
            invalid("native THD binding requires nonpaged f16 without sinks or padded Stats");
        }
        b_               = integer(spec, "b");
        qh_              = integer(spec, "qh");
        kh_              = integer(spec, "kh");
        device_          = integer(spec, "device_index");
        lens_form_       = integer(spec, "lens_form");
        off_o_desc_      = integer(spec, "off_o_desc");
        total_q_         = optional_integer(spec, "total_q");
        total_kv_        = optional_integer(spec, "total_kv");
        has_lse_         = spec.attr("has_lse").cast<bool>();
        lse_head_major_  = spec.attr("lse_head_major").cast<bool>();
        lse_head_stride_ = integer(spec, "lse_head_stride");
        if (b_ <= 0 || qh_ <= 0 || kh_ <= 0 || device_ < 0 || lens_form_ < 0 || lens_form_ > 3 || off_o_desc_ < 0 ||
            lse_head_stride_ < 0)
            invalid("invalid native THD plan geometry");
        auto expect = spec.attr("expect").cast<py::dict>();
        auto decl   = spec.attr("decl").cast<py::dict>();
        for (size_t i = Q; i <= O; ++i) {
            const auto dtype = expect[names[i]].cast<std::string>();
            if (dtype != "float16" && dtype != "bfloat16")
                invalid("native THD binding requires float16 or bfloat16 operands");
            dtype_code_[i]       = dtype == "float16" ? kDLFloat : kDLBfloat;
            declarations_[i]     = decl[names[i]].cast<std::array<int64_t, 6>>();
            const auto &geometry = declarations_[i];
            if (geometry[0] <= 0 || geometry[1] <= 0 || geometry[2] <= 0 || geometry[3] <= 0 || geometry[4] != 1)
                invalid("invalid native THD operand declaration");
        }
        auto order = spec.attr("order").cast<std::vector<std::string>>();
        if (order.size() != template_.size()) invalid("native THD host argument template has the wrong size");
        for (size_t i = 0; i < order.size(); ++i) index_[order[i]] = i;
        for (const auto *name : {"q_ptr",
                                 "k_ptr",
                                 "v_ptr",
                                 "o_ptr",
                                 "q_strides",
                                 "k_strides",
                                 "v_strides",
                                 "o_strides",
                                 "thd_q_lens_ptr",
                                 "thd_kv_lens_ptr",
                                 "lse_ptr",
                                 "lse_ext",
                                 "problem_size",
                                 "sinks_ptr",
                                 "meta_ptr",
                                 "o_desc_ptr",
                                 "stream",
                                 "scale_softmax_log2"}) {
            if (!index_.count(name)) invalid(std::string("native THD host has no argument ") + name);
        }
    }

    py::object
    bind(const py::handle &pack,
         const std::vector<int64_t> &indices,
         int64_t workspace,
         py::object stream,
         py::object scale) const {
        if (indices.size() != NumRoles) invalid("native THD binding requires eight role indices");
        const auto facts = read_native_operand_views(pack, indices);
        std::array<Geometry, 4> geometry;
        for (size_t i = Q; i <= O; ++i) {
            const auto &f = required(facts, i);
            if (!dtype_is(f, dtype_code_[i], 16))
                invalid(std::string(names[i]) + ": runtime buffer dtype does not match its declaration");
            on_device(f, names[i]);
            if (f.pointer % 16 != 0)
                invalid(std::string(names[i]) +
                        ": runtime buffer base address must be 16-byte aligned (TMA global-address rule)");
        }
        const auto &q_lens  = required(facts, QLens);
        const auto &kv_lens = required(facts, KVLens);
        const int64_t b     = numel(q_lens) - ((lens_form_ & 1) ? 1 : 0);
        if (b <= 0 || b > b_)
            invalid("seq_q_lens describes " + std::to_string(b) + " sequences; this plan is prepared for 1.." +
                    std::to_string(b_));
        const int64_t nk = add(b, (lens_form_ & 2) ? 1 : 0);
        if (numel(kv_lens) != nk)
            invalid("seq_kv_lens must describe the same " + std::to_string(b) + " sequences as seq_q_lens");
        for (size_t i = Q; i <= O; ++i) geometry[i] = resolve(facts[i], i, b);
        check_lens(q_lens, "q_lens", add(b, (lens_form_ & 1) ? 1 : 0));
        check_lens(kv_lens, "kv_lens", nk);

        const auto &lse      = facts[LSE];
        int64_t lse_capacity = -1;
        if (has_lse_) {
            if (!lse.filled) invalid("lse_tensor is required by this compiled specialization");
            on_device(lse, "lse_tensor");
            if (!dtype_is(lse, kDLFloat, 32)) invalid("lse_tensor must be float32");
            if (lse.pointer % 4 != 0) invalid("lse_tensor must be 4-byte aligned");
            if (lse_head_major_ && lse_head_stride_) {
                if (span(lse) >= 0 && span(lse) < multiply(qh_, lse_head_stride_))
                    invalid("head-major lse_tensor observed storage must hold H_q*head_stride elements");
                if (numel(lse) < multiply(qh_, lse_head_stride_))
                    invalid("head-major lse_tensor must hold H_q*head_stride elements");
            } else {
                if (span(lse) < 0)
                    invalid("lse_tensor: a ragged Stats operand needs a sized buffer, not a bare address");
                lse_capacity = span(lse) / qh_;
            }
            check_stats_layout(lse);
        } else if (lse.filled) {
            invalid("this specialization was compiled without a Stats output; construct the API without sample_lse");
        }
        int64_t tq = std::min(capacity(facts[Q], geometry[Q], "q"), capacity(facts[O], geometry[O], "o"));
        if (total_q_ >= 0) tq = std::min(tq, total_q_);
        if (lse_capacity >= 0) tq = std::min(tq, lse_capacity);
        if (tq == 0) return py::none();  // same empty-Q semantics as the Python binder: no launch or writes
        int64_t tkv = std::min(capacity(facts[K], geometry[K], "k"), capacity(facts[V], geometry[V], "v"));
        if (total_kv_ >= 0) tkv = std::min(tkv, total_kv_);
        if (facts[Sinks].filled)
            invalid("this specialization was compiled without a sink; construct the API with has_sink");
        if (workspace % 16 != 0) invalid("the workspace must be 16-byte aligned");

        // Copy references to immutable constants, then replace invocation-local
        // slots. No frame or runtime pointer is ever written into the plan.
        py::tuple frame(template_.size());
        for (size_t i = 0; i < template_.size(); ++i) frame[i] = template_[i];
        for (size_t i = Q; i <= O; ++i) {
            put(frame, std::string(names[i]) + "_ptr", py::int_(facts[i].pointer));
            put(frame,
                std::string(names[i]) + "_strides",
                py::make_tuple(geometry[i].token, geometry[i].token, geometry[i].head));
        }
        put(frame, "thd_q_lens_ptr", py::int_(q_lens.pointer));
        put(frame, "thd_kv_lens_ptr", py::int_(kv_lens.pointer));
        if (has_lse_) put(frame, "lse_ptr", py::int_(lse.pointer));
        if (tkv == 0) {
            // Descriptor-only K/V dummy rows alias Q/O. Zero device KV lengths
            // make setup/kernel skip all reads, exactly as in the existing ABI.
            tkv = 1;
            put(frame, "k_ptr", py::int_(facts[Q].pointer));
            put(frame, "v_ptr", py::int_(facts[O].pointer));
            const int64_t dk = declarations_[K][1], dv = declarations_[V][1];
            put(frame, "k_strides", py::make_tuple(multiply(kh_, dk), multiply(kh_, dk), dk));
            put(frame, "v_strides", py::make_tuple(multiply(kh_, dv), multiply(kh_, dv), dv));
        }
        if (has_lse_ && lse_head_major_ && lse_head_stride_ == 0) put(frame, "lse_ext", py::int_(tq));
        put(frame, "problem_size", py::make_tuple(b, qh_, kh_, tq, tkv, 0));
        put(frame, "sinks_ptr", py::int_(0));
        put(frame, "meta_ptr", py::int_(workspace));
        put(frame, "o_desc_ptr", py::int_(add(workspace, off_o_desc_)));
        put(frame, "stream", std::move(stream));
        if (!scale.is_none()) put(frame, "scale_softmax_log2", std::move(scale));
        return frame;
    }

    bool
    execute(const py::handle &pack,
            const std::vector<int64_t> &indices,
            int64_t workspace,
            py::object stream,
            py::object scale) const {
        py::object frame = bind(pack, indices, workspace, std::move(stream), std::move(scale));
        if (frame.is_none()) return false;
        // Retain the official Python tvm-ffi entry: it owns error conversion and
        // the stable tuple/stream ABI. No private TVM object layouts or new build
        // dependency. Observation and validation stay entirely native above.
        py::object result = py::reinterpret_steal<py::object>(PyObject_CallObject(fn_.ptr(), frame.ptr()));
        if (!result) throw py::error_already_set();
        return true;
    }

   private:
    static int64_t
    integer(const py::object &spec, const char *name) {
        return spec.attr(name).cast<int64_t>();
    }
    static int64_t
    optional_integer(const py::object &spec, const char *name) {
        auto value = spec.attr(name);
        if (value.is_none()) return -1;
        const int64_t result = value.cast<int64_t>();
        if (result < 0) invalid(std::string(name) + " must be nonnegative or None");
        return result;
    }
    static const NativeOperandView &
    required(const std::vector<NativeOperandView> &facts, size_t role) {
        if (!facts[role].filled) invalid(std::string(names[role]) + " is required");
        return facts[role];
    }
    void
    on_device(const NativeOperandView &f, const char *name) const {
        if (f.device_type != -1 && (f.device_type != kDLCUDA || f.device_id != device_)) {
            invalid(std::string(name) + " must be on CUDA device " + std::to_string(device_) + " (this plan's)");
        }
    }
    void
    check_lens(const NativeOperandView &f, const char *name, int64_t n) const {
        on_device(f, name);
        if (!dtype_is(f, kDLInt, 32)) invalid(std::string(name) + " must be int32");
        if (numel(f) != n) invalid(std::string(name) + " must have " + std::to_string(n) + " elements");
        if (!contiguous(f)) invalid(std::string(name) + " must be contiguous (read as a flat operand)");
        if (f.pointer % 4 != 0) invalid(std::string(name) + " must be 4-byte aligned");
        if (span(f) >= 0 && span(f) < n)
            invalid(std::string(name) + " observed storage is too small for its effective length");
    }
    Geometry
    resolve(const NativeOperandView &f, size_t role, int64_t b) const {
        const auto &decl = declarations_[role];
        const int64_t h = decl[0], d = decl[1];
        int64_t ts, hs, es;
        if (numel(f) == 0) {
            ts = decl[2];
            hs = decl[3];
            es = decl[4];
        } else if (f.shape.size() == 4) {
            if (f.shape[0] != b || f.shape[1] != h || f.shape[3] != d)
                invalid(std::string(names[role]) + ": effective shape must be (B, H, S, D) for this plan");
            ts = stride(f, 2);
            hs = stride(f, 1);
            es = stride(f, 3);
        } else if (f.shape.size() == 3) {
            if (f.shape[1] != h || f.shape[2] != d)
                invalid(std::string(names[role]) + ": a packed THD buffer is (T, H, D) for this plan");
            ts = stride(f, 0);
            hs = stride(f, 1);
            es = stride(f, 2);
        } else {
            invalid(std::string(names[role]) + ": a THD operand is (T, H, D) or the graph's (B, H, S, D)");
        }
        if (es != 1) invalid(std::string(names[role]) + ": the head dim must be contiguous (elem stride 1)");
        if (hs < d || hs % 8 != 0)
            invalid(std::string(names[role]) + ": head stride must cover the head dim and be a 16-byte multiple");
        const int64_t row = add(multiply(h - 1, hs), d);
        if (ts < row || ts % 8 != 0)
            invalid(std::string(names[role]) + ": token stride must cover the heads and be a 16-byte multiple");
        return {ts, hs, es, row};
    }
    static int64_t
    capacity(const NativeOperandView &f, const Geometry &g, const char *name) {
        const int64_t n = span(f);
        if (n < 0) invalid(std::string(name) + " was passed as a bare address; a ragged operand needs a sized buffer");
        return n < g.row ? 0 : (n - g.row) / g.token + 1;
    }
    void
    check_stats_layout(const NativeOperandView &f) const {
        if (numel(f) == 0 || (f.shape.size() <= 2 && contiguous(f))) return;
        int64_t hs, ts;
        if (f.shape.size() == 4 || (f.shape.size() == 3 && lse_head_major_)) {
            hs = stride(f, 1);
            ts = stride(f, 2);
        } else if (f.shape.size() == 3) {
            hs = stride(f, 1);
            ts = stride(f, 0);
        } else if (f.shape.size() == 2) {
            hs = stride(f, lse_head_major_ ? 0 : 1);
            ts = stride(f, lse_head_major_ ? 1 : 0);
        } else {
            invalid("lse_tensor: unsupported Stats geometry");
        }
        if (lse_head_major_) {
            if (ts != 1) invalid("head-major lse_tensor must have the token axis contiguous");
            if (lse_head_stride_ && hs != lse_head_stride_)
                invalid("head-major lse_tensor head stride must be the declared head stride");
        } else if (hs != 1 || ts != qh_) {
            invalid("token-major lse_tensor must be packed (T, H): head stride 1, token stride H_q");
        }
    }
    void
    put(py::tuple &frame, const std::string &name, py::object value) const {
        frame[index_.at(name)] = std::move(value);
    }

    py::object fn_, owner_;
    py::tuple template_;
    std::unordered_map<std::string, size_t> index_;
    std::array<std::array<int64_t, 6>, 4> declarations_;
    std::array<int, 4> dtype_code_;
    int64_t b_, qh_, kh_, device_, lens_form_, off_o_desc_, total_q_, total_kv_, lse_head_stride_;
    bool has_lse_, lse_head_major_;
};

}  // namespace

void
init_sdpa_thd_binding(py::module_ &m) {
    py::class_<SdpaThdBinder>(m, "_SdpaThdBinder")
        .def(py::init<const py::object &>(), py::arg("spec"))
        .def("bind",
             &SdpaThdBinder::bind,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("workspace"),
             py::arg("stream"),
             py::arg("scale") = py::none())
        .def("execute",
             &SdpaThdBinder::execute,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("workspace"),
             py::arg("stream"),
             py::arg("scale") = py::none());
}

}  // namespace python_bindings
}  // namespace cudnn_frontend
