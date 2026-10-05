// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Fixed half-backward graph contracts. Storage observations and argument frames
// belong to each call; only declared metadata and compiled owners are retained.
#include "sdpa_fixed_binding.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace cudnn_frontend {
namespace python_bindings {
namespace {
class SdpaBwdBinder : private FixedSdpaOperands {
    py::object fn_, owner_;
    int64_t workspace_bytes_;
    double scale_;
    bool scale_log2_, length_form_;

   public:
    SdpaBwdBinder(const py::object &spec, const py::tuple &declared)
        : FixedSdpaOperands(spec.attr("name").cast<std::string>(), spec.attr("device_index").cast<int64_t>()),
          fn_(spec.attr("fn")),
          owner_(spec.attr("artifact")),
          workspace_bytes_(spec.attr("workspace_bytes").cast<int64_t>()),
          scale_(spec.attr("scale").cast<double>()),
          scale_log2_(spec.attr("scale_log2").cast<bool>()),
          length_form_(spec.attr("length_form").cast<bool>()) {
        if (workspace_bytes_ < 0 || device_ < 0) invalid("invalid fixed backward workspace or device");
        const auto roles = spec.attr("roles").cast<std::vector<std::string>>();
        const auto ops   = spec.attr("operands").cast<py::tuple>();
        if (roles.size() < ops.size() || declared.size() < ops.size()) invalid("invalid backward operand declarations");
        operands_.reserve(ops.size());
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto value = ops[i];
            if (!value.is_none()) {
                if (value.attr("opaque_bytes").cast<bool>()) invalid("opaque scale factors require the Python binder");
            }
            add_operand(roles[i], value, declared[i]);
            if (!value.is_none() && value.attr("itemsize").cast<int64_t>() != operands_.back().bits / 8)
                invalid("operand dtype and element width disagree");
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
            const auto &op   = operands_[i];
            const auto &f    = facts[i];
            const auto label = op.name == "seq_q" || op.name == "seq_kv" ? op.name + "_lens" : op.name;
            if (!validate(op, f, label)) {
                frame[i] = py::none();
                continue;
            }
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
