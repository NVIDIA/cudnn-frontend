// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Half dense attention binding for SM90/SM100/SM103/SM107 and SM120/SM121. Pure geometry uses the Python admission
// predicates on a cache miss; every invocation checks fresh storage observations.
#include "variant_pack.h"
#include "sdpa_mxfp8_binding.h"
#include <memory>

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
enum Role : size_t {
    Q,
    K,
    V,
    O,
    LSE,
    Sinks,
    KVLens,
    QLens,
    KTable,
    VTable,
    Gate,
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
                                                      "lse_tensor",
                                                      "sinks",
                                                      "seq_kv_lens",
                                                      "seq_q_lens",
                                                      "block_table",
                                                      "block_table_v",
                                                      "gate",
                                                      "descale_q",
                                                      "descale_k",
                                                      "descale_v",
                                                      "scale_o",
                                                      "amax_o",
                                                      "sf_q",
                                                      "sf_k",
                                                      "sf_v"};
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
    PartialOPtr,
    Scale,
    Stream,
    NumSlots
};
constexpr std::array<const char *, NumSlots> slot_names = {"q_ptr",           "k_ptr",
                                                           "v_ptr",           "o_ptr",
                                                           "q_strides",       "k_strides",
                                                           "v_strides",       "o_strides",
                                                           "lse_ptr",         "lse_strides",
                                                           "sinks_ptr",       "meta_ptr",
                                                           "o_desc_ptr",      "seq_q_lens_addr",
                                                           "block_table_ptr", "block_table_v_ptr",
                                                           "table_strides",   "table_v_strides",
                                                           "n_pages",         "problem_size",
                                                           "o_partial_ptr",   "scale_softmax_log2",
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
struct BoundLaunch {
    py::tuple first, second;
    int64_t identity = 0;
};
class SdpaDenseBinder {
   public:
    explicit SdpaDenseBinder(const py::object &spec)
        : fn_(spec.attr("fn")), owner_(spec.attr("owner")), template_(py::tuple(spec.attr("template"))) {
        auto integer     = [&](const char *name) { return spec.attr(name).cast<int64_t>(); };
        auto flag        = [&](const char *name) { return spec.attr(name).cast<bool>(); };
        const auto quant = spec.attr("quant");
        quantized_       = !quant.is_none();
        if (quantized_) {
            if ((py::len(quant.attr("sf_sizes")) != 0 && py::len(quant.attr("sf_sizes")) != 3) ||
                !quant.attr("block_output").is_none())
                invalid("native dense FP8 binding requires per-tensor scales and a scalar output dtype");
            if (py::len(quant.attr("sf_sizes"))) mx_scales_ = std::make_unique<SdpaMxScaleBinding>(spec);
            quant_offset_ = quant.attr("scratch_offset").cast<int64_t>();
            add(quant_offset_, 8);
            has_amax_  = quant.attr("has_amax").cast<bool>();
            fill_word_ = py::module_::import("cudnn.frost.buffers").attr("fill_word_async");
        }
        if (integer("split") < 1 || flag("ragged") || (integer("split") > 1 && flag("has_sink")) ||
            !spec.attr("gate_expect").is_none())
            invalid("native dense binding requires half attention without ragged Q, split sinks or gate");
        const auto dq = integer("d_qk"), dv = integer("d_v");
        if (dq <= 0 || dv <= 0 || dq > 512 || dv > 512 || dq % 8 || dv % 8)
            invalid("native dense binding requires a supported half attention head dimension pair");
        fp32_partial_ = flag("fp32_partial");
        split_        = integer("split");
        b_            = integer("b");
        qh_           = integer("qh");
        kh_           = integer("kh");
        d_qk_         = dq;
        d_v_          = dv;
        sq_           = integer("s_q_max");
        sk_           = integer("s_k_max");
        device_       = integer("device_index");
        page_size_    = integer("page_size");
        tile_n_       = integer("tile_n");
        window_right_ = integer("window_right");
        paged_        = flag("paged");
        hnd_          = flag("paged_hnd");
        has_lse_      = flag("has_lse");
        has_sink_     = flag("has_sink");
        seq_kv_       = flag("seq_kv_present");
        seq_q_        = flag("seq_q_present");
        dense_flex_   = py::hasattr(spec, "dense_flex") && flag("dense_flex");
        shape_fixed_  = flag("shape_fixed");
        lpt_fixed_    = flag("lpt_grid_fixed");
        tail_native_  = flag("kv_tail_native");
        causal_       = flag("causal");
        bottom_right_ = flag("causal_bottom_right");
        if (b_ <= 0 || qh_ <= 0 || kh_ <= 0 || sq_ <= 0 || sk_ <= 0 || device_ < 0 || tile_n_ <= 0 ||
            (paged_ && page_size_ <= 0))
            invalid("invalid native dense plan geometry");
        auto expect  = spec.attr("expect").cast<py::dict>();
        auto combine = spec.attr("combine");
        if (split_ > 1) {
            if (combine.is_none()) invalid("native split binding requires a combine artifact");
            const auto partial_dtype = expect["o"].cast<std::string>();
            const auto output_dtype  = combine.attr("output_dtype").cast<std::string>();
            if (fp32_partial_ ? partial_dtype != "float32"
                              : ((!quantized_ && partial_dtype != output_dtype) ||
                                 (partial_dtype != "float16" && partial_dtype != "bfloat16")))
                invalid("native split binding requires matching half or FP32 partials");
            const int64_t partial_bytes = fp32_partial_ ? 4 : 2;
            combine_fn_                 = combine.attr("fn");
            combine_owner_              = combine.attr("owner");
            has_lse_                    = combine.attr("has_stats").cast<bool>();
            lse_offset_                 = combine.attr("lse_offset").cast<int64_t>();
            auto partial_o = combine.attr("o"), partial_lse = combine.attr("lse");
            auto os              = partial_o.attr("strides").cast<std::array<int64_t, 4>>();
            partial_o_strides_   = py::make_tuple(os[0], os[2], os[1]);
            partial_lse_strides_ = partial_lse.attr("strides").cast<py::tuple>();
            const auto rows      = multiply(multiply(split_, b_), multiply(qh_, sq_));
            if (lse_offset_ < multiply(multiply(rows, d_v_), partial_bytes) || lse_offset_ % 16)
                invalid("invalid native split workspace layout");
            workspace_bytes_ = add(lse_offset_, multiply(rows, 4));
        }
        for (size_t i = Q; i <= O; ++i) {
            const auto dtype = i == O && split_ > 1 ? combine.attr("output_dtype").cast<std::string>()
                                                    : expect[names[i]].cast<std::string>();
            const bool fp8   = dtype == "float8_e4m3fn" || dtype == "float8_e5m2";
            if ((quantized_ && i != O) ? !fp8 : (dtype != "float16" && dtype != "bfloat16" && !(quantized_ && fp8)))
                invalid("native dense binding has an unsupported operand dtype");
            dtype_code_[i] = fp8 ? (dtype == "float8_e4m3fn" ? kDLFloat8_e4m3fn : kDLFloat8_e5m2)
                                 : (dtype == "float16" ? kDLFloat : kDLBfloat);
            dtype_bits_[i] = fp8 ? 8 : 16;
        }
        auto order = spec.attr("order").cast<std::vector<std::string>>();
        if (order.size() != template_.size()) invalid("native dense host template has the wrong size");
        for (size_t slot = 0; slot < NumSlots; ++slot) {
            auto found = std::find(order.begin(), order.end(), slot_names[slot]);
            if (slot == Scale && found == order.end()) found = std::find(order.begin(), order.end(), "scale_softmax");
            // SM107 D256 and SM120 have no paged slots. SM120 half partials
            // use o_ptr; only FP32 partials require the separate output slot.
            const bool paged_slot = slot >= KTablePtr && slot <= NPages;
            if (found == order.end() && slot != VTableStrides && !(slot == PartialOPtr && !fp32_partial_) &&
                !(paged_slot && !paged_))
                invalid(std::string("native dense host has no argument ") + slot_names[slot]);
            index_[slot] = static_cast<size_t>(found - order.begin());
        }
        if (quantized_) {
            for (size_t role = mx_scales_ ? AmaxO : DescaleQ; role <= AmaxO; ++role) {
                const auto name = std::string(names[role]) + "_ptr";
                auto found      = std::find(order.begin(), order.end(), name);
                if (found == order.end()) invalid("native dense FP8 host has no argument " + name);
                quant_indices_[role - DescaleQ] = static_cast<size_t>(found - order.begin());
            }
            if (split_ > 1 && quant_offset_ < workspace_bytes_)
                invalid("native dense FP8 split requires partial slabs before scalar scratch");
        }
        auto prep     = py::module_::import("cudnn.sdpa.fwd.prepared");
        dense_layout_ = prep.attr("_dense_role_layout");
        pool_layout_  = prep.attr("_paged_pool_layout");
        table_layout_ = prep.attr("_paged_table_layout");
        lse_layout_   = prep.attr("_dense_lse_layout");
    }
    py::tuple
    bind(const py::handle &pack, const std::vector<int64_t> &indices, py::object stream) {
        if (quantized_) invalid("FP8 binding needs a workspace and bind_quantized");
        if (split_ != 1) invalid("split decode needs a workspace and bind_split");
        return bind_launch(pack, indices, std::move(stream), 0).first;
    }
    std::pair<py::tuple, py::tuple>
    bind_split(const py::handle &pack, const std::vector<int64_t> &indices, int64_t workspace, py::object stream) {
        if (split_ == 1) invalid("bind_split requires a split decode plan");
        auto bound = bind_launch(pack, indices, std::move(stream), workspace);
        return {bound.first, bound.second};
    }
    py::tuple
    bind_quantized(const py::handle &pack, const std::vector<int64_t> &indices, int64_t workspace, py::object stream) {
        if (!quantized_) invalid("bind_quantized requires a per-tensor FP8 plan");
        auto bound = bind_launch(pack, indices, std::move(stream), workspace);
        return py::make_tuple(bound.first, bound.second, bound.identity);
    }
    BoundLaunch
    bind_launch(const py::handle &pack, const std::vector<int64_t> &indices, py::object stream, int64_t workspace) {
        if (indices.size() != (mx_scales_ ? NumRoles : (quantized_ ? PerTensorNumRoles : HalfNumRoles)))
            invalid("native dense binding has the wrong number of role indices");
        const auto facts = read_native_operand_views(pack, indices);
        py::tuple frame(template_.size());
        for (size_t i = 0; i < template_.size(); ++i) frame[i] = template_[i];
        for (size_t i = Q; i <= O; ++i) {
            operand(facts[i], i, dtype_code_[i], dtype_bits_[i], i == O && split_ > 1 ? dtype_bits_[i] / 8 : 16);
            put(frame, static_cast<Slot>(QPtr + i), py::int_(facts[i].pointer));
        }
        auto q = geometry(facts[Q], Q), o = geometry(facts[O], O);
        const int64_t b = q.extent0, sq = q.extent1;
        if (split_ > 1 && (b != b_ || sq != sq_)) invalid("a split launch runs the declared batch and query extents");
        if (o.extent0 != b || o.extent1 != sq)
            invalid(split_ > 1 ? "split output must match the declared batch and query extents"
                               : "o must match q batch and sequence extents");
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
        if (shape_fixed_ && (sq != sq_ || sk != sk_))
            invalid("this artifact was lowered for exactly S_q=" + std::to_string(sq_) +
                    ", S_kv=" + std::to_string(sk_) +
                    " (a square-mask / schedule canonicalization read the declared extents); "
                    "it does not serve (" +
                    std::to_string(sq) + ", " + std::to_string(sk) + ")");
        if (lpt_fixed_ && (b != b_ || sq != sq_))
            invalid("this artifact's grouped LPT schedule was compiled for exactly batch=" + std::to_string(b_) +
                    ", S_q=" + std::to_string(sq_) + "; it does not serve batch=" + std::to_string(b) +
                    ", S_q=" + std::to_string(sq));
        if (!(tail_native_ || paged_ || sk % tile_n_ == 0 || seq_kv_ ||
              (causal_ && ((bottom_right_ && window_right_ == 0) || (!bottom_right_ && window_right_ <= sk - sq)))))
            invalid("S_kv (" + std::to_string(sk) + ") must be a multiple of " + std::to_string(tile_n_) +
                    " for this artifact unless per-batch KV lengths are present or the causal mask covers the KV tail "
                    "(the compiled specialization does not mask a partial last tile)");
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
        if (has_sink_) {
            operand(facts[Sinks], Sinks, kDLFloat, 32, 4);
            if (contiguous_elements(facts[Sinks], Sinks) != qh_) invalid("sinks must have exactly H_q elements");
            check_span(facts[Sinks], qh_, Sinks);
            put(frame, SinksPtr, py::int_(facts[Sinks].pointer));
        } else {
            if (facts[Sinks].filled) invalid("this specialization was compiled without a sink");
            put(frame, SinksPtr, py::int_(0));
        }
        if (facts[Gate].filled) invalid("this specialization was compiled without an epilogue gate");
        put(frame, ODescPtr, py::int_(0));
        put(frame, MetaPtr, py::int_(seq_kv_ ? lengths(facts[KVLens], KVLens, b) : 0));
        if (seq_q_) put(frame, QLensPtr, py::int_(lengths(facts[QLens], QLens, b)));
        put(frame, Stream, stream);
        py::tuple combine_frame;
        if (split_ > 1) {
            if (!workspace || workspace % 16) invalid("split workspace must be non-null and 16-byte aligned");
            // The graph/standalone adapter validates observed workspace bytes.
            // This binder also checks all offset arithmetic before either launch.
            add(workspace, workspace_bytes_);
            const auto partial_lse = add(workspace, lse_offset_);
            auto os                = o.bound.cast<std::array<int64_t, 3>>();
            combine_frame          = py::make_tuple(workspace,
                                           partial_lse,
                                           facts[O].pointer,
                                           frame[index_[LSEPtr]],
                                           py::make_tuple(b_, qh_, sq_, d_v_),
                                           split_,
                                           py::make_tuple(os[0], os[1], os[2], 1),
                                           frame[index_[LSEStrides]],
                                           stream);
            put(frame, OPtr, py::int_(workspace));
            if (fp32_partial_) put(frame, PartialOPtr, py::int_(workspace));
            put(frame, OStrides, partial_o_strides_);
            put(frame, LSEPtr, py::int_(partial_lse));
            put(frame, LSEStrides, partial_lse_strides_);
        }
        const auto identity = quantized_ ? bind_quantized_scalars(facts, frame, combine_frame, workspace, stream) : 0;
        return {frame, combine_frame, identity};
    }
    void
    execute(const py::handle &pack,
            const std::vector<int64_t> &indices,
            py::object stream,
            py::object scale  = py::none(),
            int64_t workspace = 0) {
        auto bound  = bind_launch(pack, indices, stream, workspace);
        auto &frame = bound.first;
        if (!scale.is_none()) put(frame, Scale, std::move(scale));
        if (bound.identity) fill_word_(bound.identity, 1, 0x3f800000, py::int_(stream));
        auto result = py::reinterpret_steal<py::object>(PyObject_CallObject(fn_.ptr(), frame.ptr()));
        if (!result) throw py::error_already_set();
        if (split_ > 1) {
            auto combined =
                py::reinterpret_steal<py::object>(PyObject_CallObject(combine_fn_.ptr(), bound.second.ptr()));
            if (!combined) throw py::error_already_set();
        }
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
        if (quantized_ || split_ > 1 || !read[1].cast<py::list>().empty() || read[2].is_none()) return read;
        execute(read[0], indices, std::move(stream));
        return py::none();
    }

   private:
    int64_t
    bind_quantized_scalars(const std::vector<NativeOperandView> &facts,
                           py::tuple &frame,
                           py::tuple &combine,
                           int64_t workspace,
                           const py::object &stream) const {
        if (!workspace || workspace % 16) invalid("prepared FP8 requires an aligned caller workspace");
        const auto scratch = add(workspace, quant_offset_), identity = add(scratch, 4), end = add(identity, 4);
        int64_t initialize_identity = 0;
        std::array<int64_t, 5> pointers{};
        for (size_t role = mx_scales_ ? AmaxO : DescaleQ; role <= AmaxO; ++role) {
            const auto &f = facts[role];
            int64_t ptr;
            if (!f.filled) {
                ptr = role == AmaxO ? scratch : identity;
                if (role != AmaxO) initialize_identity = identity;
            } else {
                operand(f, role, kDLFloat, 32, 4);
                if (!f.pointer || contiguous_elements(f, role) != 1)
                    invalid(std::string(names[role]) + " must be one aligned float32 device element");
                check_span(f, 1, role);
                if (role == AmaxO && !has_amax_) invalid("this specialization does not produce amax_o");
                ptr = f.pointer;
            }
            pointers[role - DescaleQ]              = ptr;
            frame[quant_indices_[role - DescaleQ]] = py::int_(ptr);
        }
        const auto amax = pointers[AmaxO - DescaleQ], amax_end = add(amax, 4);
        if (facts[AmaxO].filled && workspace < amax_end && amax < end)
            invalid("prepared FP8 workspace overlaps amax_o");
        for (size_t role = Q; role < facts.size(); ++role) {
            const auto &f = facts[role];
            if (!f.filled || role == AmaxO) continue;
            int64_t bytes = f.observed_bytes;
            if (bytes < 0) {
                if (!f.stride.empty() && f.stride.size() != f.shape.size())
                    invalid("operand shape and stride must have the same rank");
                int64_t extent = 1, compact = 1;
                bool empty = false;
                for (size_t i = f.shape.size(); i-- > 0;) {
                    if (f.shape[i] < 0 || (!f.stride.empty() && f.stride[i] < 0))
                        invalid("operand geometry must be nonnegative");
                    empty |= f.shape[i] == 0;
                    if (f.shape[i])
                        extent = add(extent, multiply(f.shape[i] - 1, f.stride.empty() ? compact : f.stride[i]));
                    compact = multiply(compact, f.shape[i]);
                }
                bytes = empty ? 0 : multiply(extent, (f.dtype.bits + 7) / 8);
            }
            if (!bytes) continue;
            const auto operand_end = add(f.pointer, bytes);
            if (workspace < operand_end && f.pointer < end)
                invalid("prepared FP8 workspace overlaps " + std::string(names[role]));
            if (amax < operand_end && f.pointer < amax_end) invalid("amax_o overlaps " + std::string(names[role]));
        }
        if (mx_scales_) {
            for (size_t role = DescaleQ; role <= ScaleO; ++role)
                if (facts[role].filled) invalid("MXFP8 scalar-output plans do not consume per-tensor scales");
            mx_scales_->bind(facts, SfQ, &frame, false, paged_, b_, sq_, sk_, page_size_);
        }
        if (split_ > 1) {
            py::tuple expanded(combine.size() + 2);
            for (size_t i = 0; i + 1 < combine.size(); ++i) expanded[i] = combine[i];
            expanded[combine.size() - 1] = has_amax_ ? py::cast(amax) : py::none();
            expanded[combine.size()]     = mx_scales_ ? py::none() : py::cast(pointers[ScaleO - DescaleQ]);
            expanded[combine.size() + 1] = stream;
            combine                      = std::move(expanded);
            if (!mx_scales_ && dtype_bits_[O] == 8) frame[quant_indices_[ScaleO - DescaleQ]] = py::none();
        }
        return initialize_identity;
    }
    void
    operand(const NativeOperandView &f, size_t role, int code, int bits, int alignment) const {
        const std::string name = names[role];
        if (!f.filled) invalid(name + " is required");
        if (f.dtype.code != code || f.dtype.bits != bits || f.dtype.lanes != 1)
            invalid(role == Sinks ? "sinks must be float32"
                                  : name + ": runtime buffer dtype does not match its declaration");
        if (f.device_type != -1 && (f.device_type != kDLCUDA || f.device_id != device_))
            invalid(name + " must be on this plan's CUDA device");
        if (quantized_ && (role == KTable || role == VTable) && (!f.pointer || f.pointer % alignment))
            invalid(std::string(role == KTable ? "paged_attention_k_table" : "paged_attention_v_table") +
                    " must have a non-null, 4-byte-aligned address");
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
            auto value =
                pool_layout_(
                    shape_tuple, stride_tuple, dtype_bits_[role] / 8, hnd_, kh_, page_size_, role == K ? d_qk_ : d_v_)
                    .cast<py::tuple>();
            result.bound   = value[0].cast<py::tuple>();
            result.need    = value[1].cast<int64_t>();
            result.extent0 = result.shape[0];
        } else {
            auto value = dense_layout_(shape_tuple,
                                       stride_tuple,
                                       role == Q || role == O ? qh_ : kh_,
                                       role == Q || role == K ? d_qk_ : d_v_,
                                       role == Q || role == O ? sq_ : sk_,
                                       b_,
                                       dtype_bits_[role] / 8,
                                       role != O || split_ == 1,
                                       names[role],
                                       dense_flex_)
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
        if (quantized_ && paged_ && (role == K || role == V) && f.observed_bytes >= 0 &&
            f.observed_bytes / (dtype_bits_[role] / 8) < result.need)
            invalid(std::string(names[role]) + ": page pool spans " +
                    std::to_string(f.observed_bytes / (dtype_bits_[role] / 8)) +
                    " elements; its effective geometry needs " + std::to_string(result.need));
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
    static int64_t
    contiguous_elements(const NativeOperandView &f, size_t role) {
        if (!f.stride.empty() && f.stride.size() != f.shape.size())
            invalid("operand shape and stride must have the same rank");
        int64_t n = 1;
        for (size_t i = f.shape.size(); i-- > 0;) {
            if (!f.stride.empty() && f.shape[i] != 1 && f.stride[i] != n)
                invalid(std::string(names[role]) + " must be contiguous");
            n = multiply(n, f.shape[i]);
        }
        return n;
    }
    int64_t
    lengths(const NativeOperandView &f, size_t role, int64_t b) const {
        operand(f, role, kDLInt, 32, 4);
        if (contiguous_elements(f, role) < b) invalid(std::string(names[role]) + " must hold at least batch elements");
        check_span(f, b, role);
        return f.pointer;
    }
    void
    put(py::tuple &frame, Slot slot, py::object value) const {
        frame[index_[slot]] = std::move(value);
    }
    py::object fn_, owner_, dense_layout_, pool_layout_, table_layout_, lse_layout_;
    py::object combine_fn_, combine_owner_, fill_word_;
    bool quantized_ = false, has_amax_ = false;
    int64_t quant_offset_ = 0;
    std::array<size_t, 5> quant_indices_;
    std::unique_ptr<SdpaMxScaleBinding> mx_scales_;
    py::tuple partial_o_strides_, partial_lse_strides_;
    py::tuple template_;
    std::array<size_t, NumSlots> index_;
    std::array<int, 4> dtype_code_, dtype_bits_;
    std::array<Geometry, NumRoles> geometry_;
    int64_t b_, qh_, kh_, d_qk_, d_v_, sq_, sk_, device_, page_size_, tile_n_, window_right_;
    int64_t split_, lse_offset_ = 0, workspace_bytes_ = 0;
    bool paged_, hnd_, has_lse_, has_sink_, seq_kv_, seq_q_, shape_fixed_, lpt_fixed_, tail_native_, causal_,
        bottom_right_, fp32_partial_, dense_flex_;
};
}  // namespace
void
init_sdpa_dense_binding(py::module_ &m) {
    py::class_<SdpaDenseBinder>(m, "_SdpaDenseBinder")
        .def(py::init<const py::object &>(), py::arg("spec"))
        .def("bind", &SdpaDenseBinder::bind, py::arg("pack"), py::arg("indices"), py::arg("stream"))
        .def("bind_split",
             &SdpaDenseBinder::bind_split,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("workspace"),
             py::arg("stream"))
        .def("bind_quantized",
             &SdpaDenseBinder::bind_quantized,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("workspace"),
             py::arg("stream"))
        .def("execute",
             &SdpaDenseBinder::execute,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("stream"),
             py::arg("scale_softmax_log2") = py::none(),
             py::arg("workspace")          = 0)
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
