// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Fixed half-backward graph contracts. Storage observations and argument frames
// belong to each call; only declared metadata and compiled owners are retained.
#include "variant_pack.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace cudnn_frontend {
namespace python_bindings {
namespace {
using Geometry = std::vector<std::pair<int64_t, int64_t>>;

class SdpaBwdBinder {
    struct Operand {
        bool enabled  = false;
        int64_t bytes = 0, alignment = 1;
        std::string name, dtype;
        uint8_t code = 0, bits = 0;
        Geometry geometry;
    };
    py::object fn_, owner_;
    std::vector<Operand> operands_;
    std::string name_;
    int64_t workspace_bytes_, device_;
    double scale_;
    bool scale_log2_, length_form_;

    [[noreturn]] void
    invalid(const std::string &message) const {
        throw py::value_error(name_ + ": " + message);
    }

    int64_t
    multiply(int64_t a, int64_t b) const {
        if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
            invalid("operand span must be nonnegative and fit in int64");
        return a * b;
    }

    int64_t
    end(int64_t pointer, int64_t bytes) const {
        if (pointer <= 0 || bytes < 0 || pointer > std::numeric_limits<int64_t>::max() - bytes)
            invalid("storage address range must fit in int64");
        return pointer + bytes;
    }

    Geometry
    geometry(std::span<const int64_t> shape, std::span<const int64_t> strides) const {
        if (!strides.empty() && strides.size() != shape.size()) invalid("shape and stride ranks differ");
        Geometry result;
        int64_t compact = 1;
        for (size_t j = shape.size(); j > 0; --j) {
            const auto i      = j - 1;
            const auto stride = strides.empty() ? compact : strides[i];
            if (shape[i] != 1) result.emplace_back(shape[i], stride);
            if (strides.empty()) compact = multiply(compact, shape[i]);
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

   public:
    SdpaBwdBinder(const py::object &spec, const py::tuple &declared)
        : fn_(spec.attr("fn")),
          owner_(spec.attr("artifact")),
          name_(spec.attr("name").cast<std::string>()),
          workspace_bytes_(spec.attr("workspace_bytes").cast<int64_t>()),
          device_(spec.attr("device_index").cast<int64_t>()),
          scale_(spec.attr("scale").cast<double>()),
          scale_log2_(spec.attr("scale_log2").cast<bool>()),
          length_form_(spec.attr("length_form").cast<bool>()) {
        if (workspace_bytes_ < 0 || device_ < 0) invalid("invalid fixed backward workspace or device");
        const auto roles = spec.attr("roles").cast<std::vector<std::string>>();
        const auto ops   = spec.attr("operands").cast<py::tuple>();
        if (roles.size() < ops.size() || declared.size() < ops.size()) invalid("invalid backward operand declarations");
        operands_.reserve(ops.size());
        for (size_t i = 0; i < ops.size(); ++i) {
            Operand op;
            op.name          = roles[i];
            const auto value = ops[i];
            if (!value.is_none()) {
                op.enabled = true;
                if (value.attr("opaque_bytes").cast<bool>()) invalid("opaque scale factors require the Python binder");
                op.dtype = value.attr("dtype").cast<std::string>();
                if (op.dtype == "bfloat16") {
                    op.code = kDLBfloat;
                    op.bits = 16;
                } else if (op.dtype == "float16" || op.dtype == "float32") {
                    op.code = kDLFloat;
                    op.bits = op.dtype == "float16" ? 16 : 32;
                } else if (op.dtype == "int32") {
                    op.code = kDLInt;
                    op.bits = 32;
                } else {
                    invalid("native half backward does not support " + op.dtype);
                }
                const auto itemsize = value.attr("itemsize").cast<int64_t>();
                if (itemsize != op.bits / 8) invalid("operand dtype and element width disagree");
                op.bytes     = multiply(value.attr("span").cast<int64_t>(), itemsize);
                op.alignment = value.attr("alignment").cast<int64_t>();
                if (op.alignment <= 0) invalid("invalid operand alignment");
            }
            if (!declared[i].is_none()) {
                auto pair    = declared[i].cast<py::tuple>();
                auto shape   = pair[0].cast<std::vector<int64_t>>();
                auto strides = pair[1].cast<std::vector<int64_t>>();
                op.geometry  = geometry(shape, strides);
            }
            operands_.push_back(std::move(op));
        }
    }

    py::tuple
    bind(const py::handle &pack,
         const std::vector<int64_t> &indices,
         int64_t workspace,
         int64_t stream,
         const std::vector<int64_t> &overridden) const {
        if (workspace <= 0 || workspace % 16) invalid("needs an aligned caller workspace");
        const auto workspace_end = end(workspace, workspace_bytes_);
        if (indices.size() != operands_.size()) invalid("incorrect native backward role indices");
        const auto facts = read_native_operand_views(pack, indices);
        py::tuple frame(operands_.size() + 3 + (scale_log2_ ? 1 : 0) + (length_form_ ? 1 : 0));
        for (size_t i = 0; i < operands_.size(); ++i) {
            const auto &op = operands_[i];
            const auto &f  = facts[i];
            if (!op.enabled) {
                if (f.filled) invalid(op.name + " was not compiled into this specialization");
                frame[i] = py::none();
                continue;
            }
            if (!f.filled) invalid(op.name + " is required by this specialization");
            const auto label = op.name == "seq_q" || op.name == "seq_kv" ? op.name + "_lens" : op.name;
            if (!((f.device_type == kDLCUDA && f.device_id == device_) || (f.device_type == -1 && f.device_id == -1)))
                invalid(label + " must be on CUDA device " + std::to_string(device_));
            if (f.dtype.bits && (f.dtype.code != op.code || f.dtype.bits != op.bits || f.dtype.lanes != 1))
                invalid(op.name + " must be " + op.dtype);
            if (f.pointer <= 0 || f.pointer % op.alignment)
                invalid(op.name + " base address must be " + std::to_string(op.alignment) + "-byte aligned");
            if (f.observed_bytes >= 0 && f.observed_bytes < op.bytes)
                invalid(op.name + " backing storage is too small for the declared strides");
            if (!f.shape.empty() && std::find(overridden.begin(), overridden.end(), indices[i]) != overridden.end() &&
                geometry(f.shape, f.stride) != op.geometry)
                invalid(op.name + " runtime geometry must match this fixed backward plan");
            // Graph operands are raw storage under their declarations, including
            // strided producer views. Standalone carrier rules do not apply here.
            if (workspace < end(f.pointer, op.bytes) && f.pointer < workspace_end)
                invalid("caller workspace overlaps " + op.name);
            frame[i] = py::int_(f.pointer);
        }
        size_t slot   = operands_.size();
        frame[slot++] = py::int_(workspace);
        if (scale_log2_) frame[slot++] = py::float_(scale_ * 1.4426950408889634);
        frame[slot++] = py::float_(scale_);
        if (length_form_) frame[slot++] = py::int_(0);  // Graph declarations always bind B lengths.
        frame[slot] = py::int_(stream);
        return frame;
    }

    void
    execute(const py::handle &pack,
            const std::vector<int64_t> &indices,
            int64_t workspace,
            int64_t stream,
            const std::vector<int64_t> &overridden) const {
        const auto frame = bind(pack, indices, workspace, stream, overridden);
        auto result      = py::reinterpret_steal<py::object>(PyObject_CallObject(fn_.ptr(), frame.ptr()));
        if (!result) throw py::error_already_set();
    }
};
}  // namespace

void
init_sdpa_bwd_binding(py::module_ &m) {
    py::class_<SdpaBwdBinder>(m, "_SdpaBwdBinder")
        .def(py::init<const py::object &, const py::tuple &>())
        .def("bind", &SdpaBwdBinder::bind)
        .def("execute", &SdpaBwdBinder::execute);
}
}  // namespace python_bindings
}  // namespace cudnn_frontend
