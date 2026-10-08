// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The standalone packed SM80 ABI shares one native binder across all flavors.
#include "sdpa_fixed_binding.h"
#include <array>

namespace py = pybind11;
namespace cudnn_frontend {
namespace python_bindings {
namespace {
class SdpaSm80ThdBinder : private FixedSdpaOperands {
    py::object fn_, owner_;
    int64_t h_, h_kv_, d_qk_, d_v_, n_seq_;
    bool has_sink_;

    Operand
    operand(const std::string &name, const std::string &dtype, int64_t alignment) const {
        Operand op;
        op.name      = name;
        op.dtype     = dtype;
        op.enabled   = true;
        op.alignment = alignment;
        if (dtype == "bfloat16") {
            op.code = kDLBfloat;
            op.bits = 16;
        } else if (dtype == "float16" || dtype == "float32") {
            op.code = kDLFloat;
            op.bits = dtype == "float16" ? 16 : 32;
        } else if (dtype == "int32" || dtype == "int64") {
            op.code = kDLInt;
            op.bits = dtype == "int32" ? 32 : 64;
        } else {
            invalid("unsupported packed operand dtype " + dtype);
        }
        return op;
    }

    std::vector<int64_t>
    strides(const NativeOperandView &f) const {
        if (!f.stride.empty()) {
            if (f.stride.size() != f.shape.size()) invalid("shape and stride ranks differ");
            return {f.stride.begin(), f.stride.end()};
        }
        std::vector<int64_t> result(f.shape.size(), 1);
        for (size_t i = result.size(); i > 1; --i) result[i - 2] = multiply(result[i - 1], f.shape[i - 1]);
        return result;
    }

    void
    storage(const Operand &op, const NativeOperandView &f, const std::vector<int64_t> &stride) const {
        if (!f.filled) invalid(op.name + " is required");
        if (f.device_type != kDLCUDA || f.device_id != device_)
            invalid(op.name + " must be on the compiled CUDA device");
        if (f.dtype.code != op.code || f.dtype.bits != op.bits || f.dtype.lanes != 1)
            invalid(op.name + " dtype differs from the compiled host");
        bool empty   = false;
        int64_t span = 1;
        for (size_t i = 0; i < f.shape.size(); ++i) {
            if (f.shape[i] < 0 || stride[i] < 0) invalid(op.name + " requires nonnegative extents and strides");
            empty = empty || f.shape[i] == 0;
        }
        if (!empty) {
            for (size_t i = 0; i < f.shape.size(); ++i) {
                const auto term = multiply(f.shape[i] - 1, stride[i]);
                if (span > std::numeric_limits<int64_t>::max() - term)
                    invalid(op.name + " storage span must fit in int64");
                span += term;
            }
        }
        const auto bytes = multiply(empty ? 0 : span, op.bits / 8);
        if (f.pointer < 0 || (bytes && !f.pointer) || f.pointer % op.alignment)
            invalid(op.name + " has an invalid or misaligned address");
        if (f.pointer) end(f.pointer, bytes);
        if (f.observed_bytes >= 0 && f.observed_bytes < bytes)
            invalid(op.name + " backing storage is too small for its strides");
    }

   public:
    explicit SdpaSm80ThdBinder(const py::object &spec)
        : FixedSdpaOperands("sdpa_fwd_sm80_thd", spec.attr("device_index").cast<int64_t>()),
          fn_(spec.attr("fn")),
          owner_(spec.attr("artifact")),
          n_seq_(spec.attr("n_seq").cast<int64_t>()),
          has_sink_(spec.attr("has_sink").cast<bool>()) {
        const auto heads = spec.attr("heads").cast<std::array<int64_t, 2>>();
        const auto dims  = spec.attr("dimensions").cast<std::array<int64_t, 2>>();
        h_               = heads[0];
        h_kv_            = heads[1];
        d_qk_            = dims[0];
        d_v_             = dims[1];
        if (h_ <= 0 || h_kv_ <= 0 || h_ % h_kv_ || d_qk_ <= 0 || d_v_ <= 0 || n_seq_ <= 0 ||
            n_seq_ == std::numeric_limits<int64_t>::max())
            invalid("invalid immutable packed geometry");
        const auto dtype = spec.attr("dtype").cast<std::string>();
        if (dtype != "float16" && dtype != "bfloat16") invalid("packed input requires half dtype");
        for (const auto &name : {"q", "k", "v", "o"}) operands_.push_back(operand(name, dtype, 16));
        operands_.push_back(operand("stats", "float32", 4));
        const auto prefixes = spec.attr("prefix_dtypes").cast<std::array<std::string, 2>>();
        for (size_t i = 0; i < 2; ++i) {
            if (prefixes[i] != "int32" && prefixes[i] != "int64") invalid("prefix dtype requires int32 or int64");
            operands_.push_back(operand(i == 0 ? "cu_q" : "cu_k", prefixes[i], prefixes[i] == "int32" ? 4 : 8));
        }
        const auto sink = spec.attr("sink_dtype").cast<std::string>();
        if (sink != "float16" && sink != "bfloat16" && sink != "float32") invalid("invalid sink dtype");
        operands_.push_back(operand("sink", sink, sink == "float32" ? 4 : 2));
    }

    py::tuple
    bind(py::handle pack, int64_t max_sq, double scale, int64_t right_bound, int64_t stream) const {
        if (max_sq <= 0 || right_bound < 0 || right_bound > INT32_MAX) invalid("invalid query bound or right window");
        if (scale == 0.0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "float division by zero");
            throw py::error_already_set();
        }
        const auto facts = read_native_operand_views(pack, {0, 1, 2, 3, 4, 5, 6, 7});
        std::array<std::vector<int64_t>, 8> stride;
        for (size_t i = 0; i < 8; ++i) {
            if (i == 7 && !has_sink_) {
                if (facts[i].filled) invalid("sink was not compiled into this specialization");
                continue;
            }
            const auto &f   = facts[i];
            const auto rank = i < 4 ? 4 : i == 4 ? 3 : 1;
            if (!f.filled || f.shape.size() != static_cast<size_t>(rank))
                invalid(operands_[i].name + " has an invalid packed rank");
            stride[i] = strides(f);
            storage(operands_[i], f, stride[i]);
            if (i < 4) {
                const auto heads = i == 0 || i == 3 ? h_ : h_kv_;
                const auto dim   = i < 2 ? d_qk_ : d_v_;
                if (f.shape[0] != 1 || f.shape[2] != heads || f.shape[3] != dim || stride[i][3] != 1)
                    invalid(operands_[i].name + " must match the compiled [1,T,H,D] geometry");
                for (size_t axis : {1, 2})
                    if (f.shape[axis] > 1 && stride[i][axis] % 8)
                        invalid(operands_[i].name + " requires 16-byte aligned rows and heads");
            } else if (i >= 5 && f.shape[0] != (i == 7 ? h_ : n_seq_ + 1)) {
                invalid(operands_[i].name + " has an incorrect element count");
            }
        }
        const auto tq = facts[0].shape[1], tkv = facts[1].shape[1];
        if (facts[2].shape[1] != tkv || facts[3].shape[1] != tq) invalid("packed token capacities must agree");
        if (facts[4].shape[0] != 1 || facts[4].shape[1] != h_ || facts[4].shape[2] != tq)
            invalid("Stats must be [1,Hq,Tq]");
        // The wrapper owns compact output allocations; their layouts are fixed
        // by this ABI even when an input is strided or has an empty capacity.
        const auto row = multiply(h_, d_v_);
        if ((tq > 1 && stride[3][1] != row) || (h_ > 1 && stride[3][2] != d_v_) ||
            (tq > 0 && h_ > 1 && stride[4][1] != tq) || (tq > 1 && stride[4][2] != 1))
            invalid("packed O and Stats must have their compact output layouts");
        py::tuple frame(22);
        for (size_t i = 0; i < 8; ++i)
            frame[i] = i == 7 && !has_sink_ ? py::none() : py::object(py::int_(facts[i].pointer));
        frame[8]  = py::int_(tq);
        frame[9]  = py::int_(tkv);
        frame[10] = py::int_(max_sq);
        for (size_t i = 0; i < 3; ++i) {
            frame[11 + i * 2] = py::int_(stride[i][1]);
            frame[12 + i * 2] = py::int_(stride[i][2]);
        }
        frame[17] = py::float_(scale * 1.4426950408889634);
        frame[18] = py::float_(1.0 / scale);
        frame[19] = py::int_(right_bound);
        frame[20] = py::make_tuple(stride[5][0], stride[6][0], has_sink_ ? stride[7][0] : 1);
        frame[21] = py::int_(stream);
        return frame;
    }

    void
    execute(py::handle pack, int64_t max_sq, double scale, int64_t right_bound, int64_t stream) const {
        const auto frame = bind(pack, max_sq, scale, right_bound, stream);
        auto result      = py::reinterpret_steal<py::object>(PyObject_CallObject(fn_.ptr(), frame.ptr()));
        if (!result) throw py::error_already_set();
    }
};
}  // namespace

void
init_sdpa_sm80_thd_binding(py::module_ &m) {
    py::class_<SdpaSm80ThdBinder>(m, "_SdpaSm80ThdBinder")
        .def(py::init<const py::object &>())
        .def("bind", &SdpaSm80ThdBinder::bind)
        .def("execute", &SdpaSm80ThdBinder::execute);
}
}  // namespace python_bindings
}  // namespace cudnn_frontend
