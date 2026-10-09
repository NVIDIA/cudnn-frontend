// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Binding for existing explicit-pointer half and per-tensor FP8 THD hosts. This changes no kernel
// ABI: a fresh positional frame goes to the same retained tvm-ffi Function. The
// native pack's observed storage/device and effective geometry are kept separate.
#include "variant_pack.h"
#include "sdpa_mxfp8_binding.h"
#include <memory>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <vector>

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
enum Role : size_t {
    Q,
    K,
    V,
    O,
    QLens,
    KVLens,
    LSE,
    Sinks,
    KTable,
    VTable,
    DescaleQ,
    DescaleK,
    DescaleV,
    ScaleO,
    AmaxO,
    SfQ,
    SfK,
    SfV,
    NumRoles
};
constexpr size_t PerTensorNumRoles                 = SfQ;
constexpr size_t HalfNumRoles                      = DescaleQ;
constexpr std::array<const char *, NumRoles> names = {"q",
                                                      "k",
                                                      "v",
                                                      "o",
                                                      "q_lens",
                                                      "kv_lens",
                                                      "lse_tensor",
                                                      "sinks",
                                                      "paged_attention_k_table",
                                                      "paged_attention_v_table",
                                                      "descale_q",
                                                      "descale_k",
                                                      "descale_v",
                                                      "scale_o",
                                                      "amax_o",
                                                      "sf_q",
                                                      "sf_k",
                                                      "sf_v"};

struct Geometry {
    int64_t token, head, element, row;
};

// Host signatures may place these arguments at different positions. Resolve
// their immutable names once; each invocation addresses the frame by slot.
enum HostSlot : size_t {
    QPtr,
    KPtr,
    VPtr,
    OPtr,
    QStrides,
    KStrides,
    VStrides,
    OStrides,
    QLensPtr,
    KVLensPtr,
    LSEPtr,
    LSEExtent,
    ProblemSize,
    SinksPtr,
    MetaPtr,
    ODescPtr,
    Stream,
    ScaleSoftmaxLog2,
    ThdUnits,
    KTablePtr,
    VTablePtr,
    TableStrides,
    NumPages,
    OPartialPtr,
    LSEPartialPtr,
    PartialOStrides,
    NumHostSlots
};
constexpr std::array<const char *, NumHostSlots> host_slot_names = {"q_ptr",
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
                                                                    "scale_softmax_log2",
                                                                    "n_thd_units",
                                                                    "block_table_ptr",
                                                                    "block_table_v_ptr",
                                                                    "table_strides",
                                                                    "n_pages",
                                                                    "o_partial_ptr",
                                                                    "lse_partial_ptr",
                                                                    "partial_o_strides"};
constexpr std::array<HostSlot, 4> pointer_slots                  = {QPtr, KPtr, VPtr, OPtr};
constexpr std::array<HostSlot, 4> stride_slots                   = {QStrides, KStrides, VStrides, OStrides};

struct BoundLaunch {
    py::object frame;
    int64_t identity = 0, amax = 0, padded_lse = 0;
};

class SdpaThdBinder {
   public:
    explicit SdpaThdBinder(const py::object &spec)
        : fn_(spec.attr("fn")), owner_(spec.attr("owner")), template_(py::tuple(spec.attr("template"))) {
        lse_padded_      = spec.attr("lse_padded").cast<bool>();
        py::object quant = py::none();
        if (py::hasattr(spec, "quant")) quant = spec.attr("quant");
        quantized_ = !quant.is_none();
        if (quantized_) {
            if ((py::len(quant.attr("sf_sizes")) != 0 && py::len(quant.attr("sf_sizes")) != 3) ||
                !quant.attr("block_output").is_none() || spec.attr("paged").cast<bool>() ||
                (py::hasattr(spec, "split_workspace") && !spec.attr("split_workspace").is_none()))
                invalid("native THD FP8 binding requires nonpaged, unsplit per-tensor scales and scalar output");
            if (py::len(quant.attr("sf_sizes"))) mx_scales_ = std::make_unique<SdpaMxScaleBinding>(spec);
            quant_offset_ = quant.attr("scratch_offset").cast<int64_t>();
            add(quant_offset_, 8);
            has_amax_    = quant.attr("has_amax").cast<bool>();
            auto buffers = py::module_::import("cudnn.frost.buffers");
            fill_word_   = buffers.attr("fill_word_async");
            zero_word_   = buffers.attr("memset_zero_async");
        }
        paged_     = spec.attr("paged").cast<bool>();
        paged_hnd_ = paged_ && spec.attr("paged_hnd").cast<bool>();
        page_size_ = paged_ ? integer(spec, "page_size") : 0;
        if (paged_ && page_size_ <= 0) invalid("page_size must be positive for a paged plan");
        fixed_batch_         = py::hasattr(spec, "fixed_batch") && spec.attr("fixed_batch").cast<bool>();
        workspace_alignment_ = py::hasattr(spec, "workspace_alignment") ? integer(spec, "workspace_alignment") : 16;
        if (workspace_alignment_ < 16 || (workspace_alignment_ & (workspace_alignment_ - 1)))
            invalid("native THD workspace alignment must be a power of two of at least 16 bytes");
        b_               = integer(spec, "b");
        qh_              = integer(spec, "qh");
        kh_              = integer(spec, "kh");
        device_          = integer(spec, "device_index");
        lens_form_       = integer(spec, "lens_form");
        off_o_desc_      = integer(spec, "off_o_desc");
        cga_tile_m_      = integer(spec, "cga_tile_m");
        total_q_         = optional_integer(spec, "total_q");
        total_kv_        = optional_integer(spec, "total_kv");
        has_lse_         = spec.attr("has_lse").cast<bool>();
        has_sink_        = spec.attr("has_sink").cast<bool>();
        lse_head_major_  = spec.attr("lse_head_major").cast<bool>();
        lse_head_stride_ = integer(spec, "lse_head_stride");
        lse_stride_override_ =
            py::hasattr(spec, "lse_stride_override") && spec.attr("lse_stride_override").cast<bool>();
        if (b_ <= 0 || qh_ <= 0 || kh_ <= 0 || device_ < 0 || lens_form_ < 0 || lens_form_ > 3 || off_o_desc_ < 0 ||
            lse_head_stride_ < 0 || cga_tile_m_ <= 0)
            invalid("invalid native THD plan geometry");
        if (has_lse_ && lse_padded_) {
            sq_max_       = integer(spec, "s_q_max");
            lse_strides_  = spec.attr("lse_stride").cast<std::array<int64_t, 3>>();
            lse_elements_ = multiply(multiply(b_, qh_), sq_max_);
            lse_span_     = lse_elements_ ? 1 : 0;
            const std::array<int64_t, 3> shape{b_, qh_, sq_max_};
            for (size_t i = 0; i < shape.size(); ++i) {
                if (lse_strides_[i] < 0) invalid("padded Stats strides must be nonnegative");
                if (lse_elements_) lse_span_ = add(lse_span_, multiply(shape[i] - 1, lse_strides_[i]));
            }
            multiply(lse_span_, 4);
            lse_fill_plan_ = spec.attr("lse_fill_plan");
            if (lse_fill_plan_.is_none()) invalid("padded Stats strides must not overlap");
            seed_stats_ = py::module_::import("cudnn.frost.buffers").attr("apply_fill_plan");
            neg_inf_    = spec.attr("neg_inf");
        }
        if (py::hasattr(spec, "split_workspace") && !spec.attr("split_workspace").is_none()) {
            const auto split       = spec.attr("split_workspace").cast<std::array<int64_t, 4>>();
            splits_                = split[0];
            split_capacity_        = split[1];
            off_partial_o_         = split[2];
            off_partial_lse_       = split[3];
            const int64_t split_dq = integer(spec, "d_qk"), split_dv = integer(spec, "d_v");
            const bool d64_split = paged_ && cga_tile_m_ == 128 && split_dq == 64 && split_dv == 64;
            const bool d128_split =
                cga_tile_m_ == 128 && split_dv == 128 && (split_dq == 128 || (!paged_ && split_dq == 192));
            const bool d256_split = cga_tile_m_ == 256 && split_dq == 256 && split_dv == 256;
            if ((has_sink_ && !(paged_ && d128_split && split_dq == 128)) || splits_ <= 1 || split_capacity_ <= 0 ||
                split_capacity_ > INT32_MAX || off_partial_o_ < 0 || off_partial_lse_ < 0 || off_partial_o_ % 16 ||
                off_partial_lse_ % 16 || !(d64_split || d128_split || d256_split))
                invalid("invalid prepared packed split geometry");
            const int64_t partial_rows = multiply(multiply(splits_, split_capacity_), qh_);
            if (off_partial_o_ < add(off_o_desc_, multiply(add(b_, 3), 128)) ||
                off_partial_lse_ < add(off_partial_o_, multiply(partial_rows, multiply(split_dv, 4))) ||
                integer(spec, "scratch_bytes") < add(off_partial_lse_, multiply(partial_rows, 4)))
                invalid("packed split workspace regions overlap or exceed the reservation");
        }
        auto expect = spec.attr("expect").cast<py::dict>();
        auto decl   = spec.attr("decl").cast<py::dict>();
        for (size_t i = Q; i <= O; ++i) {
            const auto dtype = expect[names[i]].cast<std::string>();
            const bool fp8   = dtype == "float8_e4m3fn" || dtype == "float8_e5m2";
            if ((quantized_ && i != O) ? !fp8 : (dtype != "float16" && dtype != "bfloat16" && !(quantized_ && fp8)))
                invalid("native THD binding has an unsupported operand dtype");
            dtype_code_[i]       = fp8 ? (dtype == "float8_e4m3fn" ? kDLFloat8_e4m3fn : kDLFloat8_e5m2)
                                       : (dtype == "float16" ? kDLFloat : kDLBfloat);
            dtype_bits_[i]       = fp8 ? 8 : 16;
            declarations_[i]     = decl[names[i]].cast<std::array<int64_t, 6>>();
            const auto &geometry = declarations_[i];
            if (geometry[0] <= 0 || geometry[1] <= 0 || geometry[2] <= 0 || geometry[3] <= 0 || geometry[4] != 1)
                invalid("invalid native THD operand declaration");
        }
        auto order = spec.attr("order").cast<std::vector<std::string>>();
        if (order.size() != template_.size()) invalid("native THD host argument template has the wrong size");
        for (size_t slot = 0; slot < NumHostSlots; ++slot) {
            // Nonpaged hosts (including SM120) need not expose paged ABI slots.
            if (!paged_ && slot >= KTablePtr && slot <= NumPages) continue;
            if (splits_ == 1 && slot >= OPartialPtr) continue;
            auto found = std::find(order.begin(), order.end(), host_slot_names[slot]);
            if (slot == ScaleSoftmaxLog2 && found == order.end())
                found = std::find(order.begin(), order.end(), "scale_softmax");
            if (found == order.end()) invalid(std::string("native THD host has no argument ") + host_slot_names[slot]);
            index_[slot] = static_cast<size_t>(found - order.begin());
        }
        if (quantized_) {
            for (size_t role = mx_scales_ ? AmaxO : DescaleQ; role <= AmaxO; ++role) {
                const auto name  = std::string(names[role]) + "_ptr";
                const auto found = std::find(order.begin(), order.end(), name);
                if (found == order.end()) invalid("native THD FP8 host has no argument " + name);
                quant_indices_[role - DescaleQ] = static_cast<size_t>(found - order.begin());
            }
        }
        units_ = template_[index_[ThdUnits]].cast<int64_t>();
        if (units_ <= 0) invalid("native THD launch bound must be positive");
    }

    py::object
    bind(const py::handle &pack,
         const std::vector<int64_t> &indices,
         int64_t workspace,
         py::object stream,
         py::object scale) const {
        return bind_launch(pack, indices, workspace, std::move(stream), std::move(scale)).frame;
    }

    py::tuple
    bind_quantized(const py::handle &pack,
                   const std::vector<int64_t> &indices,
                   int64_t workspace,
                   py::object stream) const {
        if (!quantized_) invalid("bind_quantized requires a per-tensor FP8 plan");
        auto bound = bind_launch(pack, indices, workspace, std::move(stream), py::none());
        return py::make_tuple(bound.frame, bound.identity, bound.amax);
    }

    BoundLaunch
    bind_launch(const py::handle &pack,
                const std::vector<int64_t> &indices,
                int64_t workspace,
                py::object stream,
                py::object scale) const {
        if (indices.size() != (mx_scales_ ? NumRoles : (quantized_ ? PerTensorNumRoles : HalfNumRoles)))
            invalid("native THD binding has the wrong number of role indices");
        const auto facts = read_native_operand_views(pack, indices);
        BoundLaunch bound{bind_facts(facts, workspace, std::move(stream), std::move(scale))};
        if (quantized_) bind_quantized_scalars(facts, bound, workspace);
        if (has_lse_ && lse_padded_) bound.padded_lse = facts[LSE].pointer;
        return bound;
    }

    py::object
    bind_facts(const std::vector<NativeOperandView> &facts,
               int64_t workspace,
               py::object stream,
               py::object scale) const {
        std::array<Geometry, 4> geometry;
        for (size_t i = Q; i <= O; ++i) {
            const auto &f = required(facts, i);
            if (!dtype_is(f, dtype_code_[i], dtype_bits_[i]))
                invalid(std::string(names[i]) + ": runtime buffer dtype does not match its declaration");
            on_device(f, names[i]);
            if (f.pointer % 16 != 0)
                invalid(std::string(names[i]) +
                        ": runtime buffer base address must be 16-byte aligned (TMA global-address rule)");
        }
        const auto &q_lens  = required(facts, QLens);
        const auto &kv_lens = required(facts, KVLens);
        const int64_t b     = numel(q_lens) - ((lens_form_ & 1) ? 1 : 0);
        if (fixed_batch_ && b != b_)
            invalid("this artifact requires exactly " + std::to_string(b_) + " sequences; got " + std::to_string(b));
        if (b <= 0 || b > b_)
            invalid("seq_q_lens describes " + std::to_string(b) + " sequences; this plan is prepared for 1.." +
                    std::to_string(b_));
        if (has_lse_ && lse_padded_ && b != b_)
            invalid("a per-batch padded Stats buffer is declared for " + std::to_string(b_) + " sequences; running " +
                    std::to_string(b) + " is not supported");
        const int64_t nk = add(b, (lens_form_ & 2) ? 1 : 0);
        if (numel(kv_lens) != nk)
            invalid("seq_kv_lens must describe the same " + std::to_string(b) + " sequences as seq_q_lens");
        for (size_t i = Q; i <= O; ++i) {
            if (!paged_ || i == Q || i == O) geometry[i] = resolve(facts[i], i, b);
        }
        check_lens(q_lens, "q_lens", add(b, (lens_form_ & 1) ? 1 : 0));
        check_lens(kv_lens, "kv_lens", nk);

        const auto &lse         = facts[LSE];
        int64_t lse_capacity    = -1;
        int64_t lse_head_stride = lse_head_stride_;
        if (has_lse_) {
            if (!lse.filled) invalid("lse_tensor is required by this compiled specialization");
            on_device(lse, "lse_tensor");
            if (!dtype_is(lse, kDLFloat, 32)) invalid("lse_tensor must be float32");
            if (lse.pointer % 4 != 0) invalid("lse_tensor must be 4-byte aligned");
            lse_head_stride = check_stats_layout(lse);
            if (lse_padded_) {
                if (numel(lse) != lse_elements_)
                    invalid("padded lse_tensor must have B*H_q*S_q_max = " + std::to_string(lse_elements_) +
                            " elements");
                if (span(lse) >= 0 && span(lse) < lse_span_)
                    invalid("padded lse_tensor observed storage must cover the declared strides");
                if (lse_elements_ && !lse.pointer) invalid("padded lse_tensor requires a non-null address");
                add(lse.pointer, multiply(lse_span_, 4));
            } else if (lse_head_major_ && lse_head_stride) {
                if (span(lse) >= 0 && span(lse) < multiply(qh_, lse_head_stride))
                    invalid("head-major lse_tensor observed storage must hold H_q*head_stride elements");
                if (span(lse) < 0 && numel(lse) < multiply(qh_, lse_head_stride))
                    invalid("head-major lse_tensor must hold H_q*head_stride elements");
            } else {
                if (span(lse) < 0)
                    invalid("lse_tensor: a ragged Stats operand needs a sized buffer, not a bare address");
                lse_capacity = span(lse) / qh_;
            }
        } else if (lse.filled) {
            invalid("this specialization was compiled without a Stats output; construct the API without sample_lse");
        }
        int64_t tq = std::min(capacity(facts[Q], geometry[Q], "q"), capacity(facts[O], geometry[O], "o"));
        if (total_q_ >= 0) tq = std::min(tq, total_q_);
        if (lse_capacity >= 0) tq = std::min(tq, lse_capacity);
        // Spare backing storage does not enlarge the declared live-Q bound.
        if (splits_ > 1) tq = std::min(tq, split_capacity_);
        // Head padding occupies storage, not logical tokens. Keep the full
        // observed-span check above and check logical rows against bounded Q.
        if (has_lse_ && !lse_padded_ && lse_head_major_ && lse_head_stride &&
            numel(lse) < multiply(qh_, std::min(tq, lse_head_stride)))
            invalid("head-major lse_tensor logical shape must cover bounded packed Q");
        // Empty Q still initializes padded Stats or quantized Amax/scalars.
        // Finish binding validation before authorizing those writes.
        if (tq == 0 && !(has_lse_ && lse_padded_) && !quantized_) return py::none();
        int64_t tkv = 0;
        if (!paged_) {
            tkv = std::min(capacity(facts[K], geometry[K], "k"), capacity(facts[V], geometry[V], "v"));
            if (total_kv_ >= 0) tkv = std::min(tkv, total_kv_);
        }
        if (has_sink_) {
            const auto &sink = required(facts, Sinks);
            on_device(sink, "sinks");
            if (!dtype_is(sink, kDLFloat, 32) || numel(sink) != qh_ || !contiguous(sink))
                invalid("sinks must be contiguous float32 with exactly H_q elements");
            if (sink.pointer % 4) invalid("sinks must be 4-byte aligned");
            if (span(sink) >= 0 && span(sink) < qh_) invalid("sinks observed storage is too small");
        } else if (facts[Sinks].filled) {
            invalid("this specialization was compiled without a sink; construct the API with has_sink");
        }
        if (!workspace) invalid("prepared THD requires a non-null workspace");
        if (workspace % workspace_alignment_ != 0)
            invalid("the workspace must be " + std::to_string(workspace_alignment_) + "-byte aligned");

        // Copy references to immutable constants, then replace invocation-local
        // slots. No frame or runtime pointer is ever written into the plan.
        py::tuple frame(template_.size());
        for (size_t i = 0; i < template_.size(); ++i) frame[i] = template_[i];
        for (size_t i = Q; i <= O; ++i) {
            put(frame, pointer_slots[i], py::int_(facts[i].pointer));
            if (!paged_ || i == Q || i == O)
                put(frame, stride_slots[i], py::make_tuple(geometry[i].token, geometry[i].token, geometry[i].head));
        }
        put(frame, QLensPtr, py::int_(q_lens.pointer));
        put(frame, KVLensPtr, py::int_(kv_lens.pointer));
        if (has_lse_) put(frame, LSEPtr, py::int_(lse.pointer));
        if (paged_) tkv = bind_paged(frame, facts, b);
        if (tq == 0) return py::none();
        if (!paged_ && tkv == 0) {
            // Descriptor-only K/V dummy rows alias Q/O. Zero device KV lengths
            // make setup/kernel skip all reads, exactly as in the existing ABI.
            tkv = 1;
            put(frame, KPtr, py::int_(facts[Q].pointer));
            put(frame, VPtr, py::int_(facts[O].pointer));
            const int64_t dk = declarations_[K][1], dv = declarations_[V][1];
            put(frame, KStrides, py::make_tuple(multiply(kh_, dk), multiply(kh_, dk), dk));
            put(frame, VStrides, py::make_tuple(multiply(kh_, dv), multiply(kh_, dv), dv));
        }
        if (has_lse_ && !lse_padded_ && lse_head_major_)
            put(frame, LSEExtent, py::int_(lse_head_stride ? lse_head_stride : tq));
        put(frame, ProblemSize, py::make_tuple(b, qh_, kh_, tq, tkv, 0));
        // Sum of per-sequence ceil divisions <= ceil(total capacity / tile) + B - 1.
        // Rebind a safe grid without reading device lengths or mutating the plan.
        // min also preserves a persistent kernel's resident-cluster launch cap.
        const int64_t units = multiply(multiply(add((tq - 1) / cga_tile_m_, b), qh_), splits_);
        put(frame, ThdUnits, py::int_(std::min(units_, units)));
        put(frame, SinksPtr, py::int_(has_sink_ ? facts[Sinks].pointer : 0));
        put(frame, MetaPtr, py::int_(workspace));
        put(frame, ODescPtr, py::int_(add(workspace, off_o_desc_)));
        if (splits_ > 1) {
            put(frame, OPartialPtr, py::int_(add(workspace, off_partial_o_)));
            put(frame, LSEPartialPtr, py::int_(add(workspace, off_partial_lse_)));
            const int64_t dv  = declarations_[V][1];
            const int64_t row = multiply(qh_, dv);
            put(frame, PartialOStrides, py::make_tuple(multiply(tq, row), row, dv));
        }
        put(frame, Stream, std::move(stream));
        if (!scale.is_none()) put(frame, ScaleSoftmaxLog2, std::move(scale));
        return frame;
    }

    bool
    execute(const py::handle &pack,
            const std::vector<int64_t> &indices,
            int64_t workspace,
            py::object stream,
            py::object scale) const {
        auto bound = bind_launch(pack, indices, workspace, stream, std::move(scale));
        // Binding (including all quantized scalars) finishes before any write.
        // The seed is an existing declared operation, including empty Q.
        if (bound.padded_lse)
            seed_stats_(bound.padded_lse, lse_fill_plan_, neg_inf_, stream.is_none() ? py::int_(0) : py::int_(stream));
        if (bound.identity) fill_word_(bound.identity, 1, 0x3f800000, py::int_(stream));
        py::object &frame = bound.frame;
        if (frame.is_none()) {
            // Preserve the existing quantized empty-Q contract: there is no
            // attention host to reset Amax_O, so clear its current output word.
            if (quantized_) zero_word_(bound.amax, 4, py::int_(stream));
            return false;
        }
        // Retain the official Python tvm-ffi entry: it owns error conversion and
        // the stable tuple/stream ABI. No private TVM object layouts or new build
        // dependency. Observation and validation stay entirely native above.
        py::object result = py::reinterpret_steal<py::object>(PyObject_CallObject(fn_.ptr(), frame.ptr()));
        if (!result) throw py::error_already_set();
        return true;
    }

   private:
    void
    bind_quantized_scalars(const std::vector<NativeOperandView> &facts, BoundLaunch &bound, int64_t workspace) const {
        if (!workspace || workspace % 16) invalid("prepared FP8 requires an aligned caller workspace");
        const auto scratch = add(workspace, quant_offset_), identity = add(scratch, 4), end = add(identity, 4);
        std::array<int64_t, 5> pointers{};
        for (size_t role = mx_scales_ ? AmaxO : DescaleQ; role <= AmaxO; ++role) {
            const auto &f = facts[role];
            int64_t ptr;
            if (!f.filled) {
                ptr = role == AmaxO ? scratch : identity;
                if (role != AmaxO) bound.identity = identity;
            } else {
                on_device(f, names[role]);
                if (!dtype_is(f, kDLFloat, 32) || numel(f) != 1 || !contiguous(f) || !f.pointer || f.pointer % 4 ||
                    (span(f) >= 0 && span(f) < 1))
                    invalid(std::string(names[role]) +
                            " must be one aligned float32 device element with sufficient storage when observed");
                if (role == AmaxO && !has_amax_) invalid("this specialization does not produce amax_o");
                ptr = f.pointer;
            }
            pointers[role - DescaleQ] = ptr;
        }
        bound.amax          = pointers[AmaxO - DescaleQ];
        const auto amax_end = add(bound.amax, 4);
        if (facts[AmaxO].filled && workspace < amax_end && bound.amax < end)
            invalid("prepared FP8 workspace overlaps amax_o");
        for (size_t role = Q; role < facts.size(); ++role) {
            const auto &f = facts[role];
            if (!f.filled || role == AmaxO) continue;
            int64_t bytes = f.observed_bytes;
            if (bytes < 0) {
                int64_t extent = 1;
                bool empty     = false;
                for (size_t dim = 0; dim < f.shape.size(); ++dim) {
                    if (f.shape[dim] < 0 || stride(f, dim) < 0) invalid("operand geometry must be nonnegative");
                    empty |= f.shape[dim] == 0;
                    if (f.shape[dim]) extent = add(extent, multiply(f.shape[dim] - 1, stride(f, dim)));
                }
                bytes = empty ? 0 : multiply(extent, (f.dtype.bits + 7) / 8);
            }
            if (!bytes) continue;
            const auto operand_end = add(f.pointer, bytes);
            if (workspace < operand_end && f.pointer < end)
                invalid("prepared FP8 workspace overlaps " + std::string(names[role]));
            if (bound.amax < operand_end && f.pointer < amax_end)
                invalid("amax_o overlaps " + std::string(names[role]));
        }
        if (mx_scales_) {
            for (size_t role = DescaleQ; role <= ScaleO; ++role)
                if (facts[role].filled) invalid("MXFP8 scalar-output plans do not consume per-tensor scales");
            if (bound.frame.is_none()) {
                mx_scales_->bind(facts, SfQ, nullptr, true, false, b_, 0, 0, 0);
            } else {
                auto frame = py::reinterpret_steal<py::tuple>(bound.frame.release());
                mx_scales_->bind(facts, SfQ, &frame, true, false, b_, 0, 0, 0);
                bound.frame = std::move(frame);
            }
        }
        if (!bound.frame.is_none()) {
            // Move the uniquely owned tuple: PyTuple_SetItem rejects a second owning reference.
            auto frame = py::reinterpret_steal<py::tuple>(bound.frame.release());
            for (size_t role = mx_scales_ ? AmaxO : DescaleQ; role <= AmaxO; ++role)
                frame[quant_indices_[role - DescaleQ]] = py::int_(pointers[role - DescaleQ]);
            bound.frame = std::move(frame);
        }
    }

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
        if (hs < d || hs % (128 / dtype_bits_[role]) != 0)
            invalid(std::string(names[role]) + ": head stride must cover the head dim and be a 16-byte multiple");
        const int64_t row = add(multiply(h - 1, hs), d);
        if (ts < row || ts % (128 / dtype_bits_[role]) != 0)
            invalid(std::string(names[role]) + ": token stride must cover the heads and be a 16-byte multiple");
        return {ts, hs, es, row};
    }
    static int64_t
    capacity(const NativeOperandView &f, const Geometry &g, const char *name) {
        const int64_t n = span(f);
        if (n < 0) invalid(std::string(name) + " was passed as a bare address; a ragged operand needs a sized buffer");
        return n < g.row ? 0 : (n - g.row) / g.token + 1;
    }
    int64_t
    check_stats_layout(const NativeOperandView &f) const {
        if (lse_padded_) {
            if ((f.shape.size() == 3 || f.shape.size() == 4) && stride(f, 0) == lse_strides_[0] &&
                stride(f, 1) == lse_strides_[1] && stride(f, 2) == lse_strides_[2])
                return lse_head_stride_;
            if (!contiguous(f)) invalid("padded lse_tensor strides must be the declared strides or contiguous storage");
            return lse_head_stride_;
        }
        if (numel(f) == 0 || (f.shape.size() <= 2 && contiguous(f))) return lse_head_stride_;
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
            if (hs <= 0) invalid("head-major lse_tensor head stride must be positive");
            if (lse_stride_override_) return hs;
            if (lse_head_stride_ && hs != lse_head_stride_)
                invalid("head-major lse_tensor head stride must be the declared head stride");
        } else if (hs != 1 || ts != qh_) {
            invalid("token-major lse_tensor must be packed (T, H): head stride 1, token stride H_q");
        }
        return lse_head_stride_;
    }
    struct TableGeometry {
        int64_t batch, pages, batch_stride, page_stride;
    };

    TableGeometry
    table_geometry(const NativeOperandView &f, size_t role, int64_t b) const {
        const auto name = names[role];
        on_device(f, name);
        if (!dtype_is(f, kDLInt, 32)) invalid(std::string(name) + " must be int32");
        if (!f.pointer || f.pointer % 4) invalid(std::string(name) + " must have a non-null, 4-byte-aligned address");
        const bool rank4 = f.shape.size() == 4;
        if ((!rank4 && f.shape.size() != 2) || (rank4 && (f.shape[1] != 1 || f.shape[3] != 1)))
            invalid(std::string(name) + " must be (B, max_pages) or (B, 1, max_pages, 1)");
        TableGeometry g{f.shape[0], f.shape[rank4 ? 2 : 1], stride(f, 0), stride(f, rank4 ? 2 : 1)};
        if (g.batch < b || g.pages <= 0 || g.batch_stride < 0 || g.page_stride < 0)
            invalid(std::string(name) + " needs covering dimensions and nonnegative strides");
        const auto need = add(add(multiply(b - 1, g.batch_stride), multiply(g.pages - 1, g.page_stride)), 1);
        if (span(f) >= 0 && span(f) < need) invalid(std::string(name) + " observed storage is too small");
        return g;
    }

    std::array<int64_t, 3>
    pool_geometry(const NativeOperandView &f, size_t role) const {
        const auto name = std::string(names[role]);
        const auto d    = declarations_[role][1];
        if (f.shape.size() != 4 || f.shape[0] <= 0 || f.shape[1] != kh_ || f.shape[2] != page_size_ || f.shape[3] != d)
            invalid(name + ": a page pool must be (n_pages, H_kv, page_size, D)");
        if (stride(f, 3) != 1) invalid(name + ": the page pool's head dim must be contiguous");
        if ((stride(f, 1) > stride(f, 2)) != paged_hnd_)
            invalid(name + ": page pool layout differs from the compiled HND/NHD specialization");
        // Same storage-order canonicalization as dense_bind_strides: unused
        // singleton strides are not observable. Every stepped axis is covering
        // and TMA-aligned; observed storage remains a separate per-call bound.
        const int64_t h = paged_hnd_ ? page_size_ : kh_, seq = paged_hnd_ ? kh_ : page_size_;
        int64_t bs = stride(f, 0), hs = stride(f, paged_hnd_ ? 2 : 1), ss = stride(f, paged_hnd_ ? 1 : 2);
        if (h == 1) hs = d;
        if (seq == 1) ss = multiply(h, hs);
        if (f.shape[0] == 1) bs = multiply(seq, ss);
        if (hs < d || ss < multiply(h, hs) || bs < multiply(seq, ss) || hs % 8 || ss % 8 || (f.shape[0] > 1 && bs % 8))
            invalid(name + ": page-pool strides must be covering and 16-byte aligned");
        const int64_t token = paged_hnd_ ? hs : ss, head = paged_hnd_ ? ss : hs;
        const auto need =
            add(add(add(multiply(f.shape[0] - 1, bs), multiply(page_size_ - 1, token)), multiply(kh_ - 1, head)), d);
        if (span(f) >= 0 && span(f) < need) invalid(name + ": page pool observed storage is too small");
        return {bs, token, head};
    }

    int64_t
    bind_paged(py::tuple &frame, const std::vector<NativeOperandView> &facts, int64_t b) const {
        const auto &kt = required(facts, KTable), &vt = required(facts, VTable);
        const auto kg = table_geometry(kt, KTable, b), vg = table_geometry(vt, VTable, b);
        if (kg.pages != vg.pages || kg.batch_stride != vg.batch_stride || kg.page_stride != vg.page_stride)
            invalid("paged K/V tables must share max_pages and strides");
        const auto ks = pool_geometry(facts[K], K), vs = pool_geometry(facts[V], V);
        if (facts[K].shape[0] != facts[V].shape[0]) invalid("K and V pools must hold the same number of pages");
        put(frame, KStrides, py::make_tuple(ks[0], ks[1], ks[2]));
        put(frame, VStrides, py::make_tuple(vs[0], vs[1], vs[2]));
        put(frame, KTablePtr, py::int_(kt.pointer));
        put(frame, VTablePtr, py::int_(vt.pointer));
        put(frame, TableStrides, py::make_tuple(kg.batch_stride, kg.page_stride));
        put(frame, NumPages, py::int_(facts[K].shape[0]));
        return multiply(kg.pages, page_size_);
    }

    void
    put(py::tuple &frame, HostSlot slot, py::object value) const {
        frame[index_[slot]] = std::move(value);
    }

    py::object fn_, owner_, fill_word_, zero_word_;
    py::object lse_fill_plan_, seed_stats_, neg_inf_;
    py::tuple template_;
    std::array<size_t, NumHostSlots> index_;
    std::array<std::array<int64_t, 6>, 4> declarations_;
    std::array<int, 4> dtype_code_, dtype_bits_;
    std::array<size_t, 5> quant_indices_;
    std::unique_ptr<SdpaMxScaleBinding> mx_scales_;
    int64_t quant_offset_ = 0;
    bool quantized_ = false, has_amax_ = false;
    int64_t b_, qh_, kh_, device_, lens_form_, off_o_desc_, total_q_, total_kv_, lse_head_stride_;
    int64_t cga_tile_m_, units_, page_size_, workspace_alignment_;
    int64_t splits_ = 1, split_capacity_ = 0, off_partial_o_ = 0, off_partial_lse_ = 0;
    int64_t sq_max_ = 0, lse_elements_ = 0, lse_span_ = 0;
    std::array<int64_t, 3> lse_strides_{};
    bool lse_padded_ = false;
    bool has_lse_, has_sink_, lse_head_major_, lse_stride_override_, paged_, paged_hnd_, fixed_batch_;
};

}  // namespace

void
init_sdpa_thd_binding(py::module_ &m) {
    py::class_<SdpaThdBinder>(m, "_SdpaThdBinder")
        .def(py::init<const py::object &>(), py::arg("spec"))
        .def_property_readonly_static("supports_stats_stride_override", [](py::object) { return true; })
        .def_property_readonly_static("supports_paged_packed_split", [](py::object) { return true; })
        .def_property_readonly_static("supports_paged_split_sink", [](py::object) { return true; })
        .def_property_readonly_static("supports_paged_d64_packed_split", [](py::object) { return true; })
        .def_property_readonly_static("supports_paged_d256_packed_split", [](py::object) { return true; })
        .def_property_readonly_static("supports_nonpaged_d256_packed_split", [](py::object) { return true; })
        .def_property_readonly_static("supports_nonpaged_packed_split", [](py::object) { return true; })
        .def_property_readonly_static("supports_nonpaged_d128_packed_split", [](py::object) { return true; })
        .def("bind",
             &SdpaThdBinder::bind,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("workspace"),
             py::arg("stream"),
             py::arg("scale") = py::none())
        .def("bind_quantized",
             &SdpaThdBinder::bind_quantized,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("workspace"),
             py::arg("stream"))
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
