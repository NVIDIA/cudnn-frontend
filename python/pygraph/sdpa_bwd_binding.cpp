// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Fixed half-backward graph and standalone contracts. Storage observations and argument frames
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
    struct Carrier {
        int64_t elements = 0;
        std::vector<int64_t> allowed_numels;
        bool contiguous = false;
        int length_bit  = -1;
    };
    std::vector<Carrier> carriers_;
    py::object fn_, owner_;
    int64_t workspace_bytes_;
    double scale_;
    bool scale_log2_, length_form_;

    int64_t
    numel(const NativeOperandView &f) const {
        int64_t result = 1;
        for (auto n : f.shape) result = multiply(result, n);
        return result;
    }

    bool
    contiguous(const NativeOperandView &f) const {
        if (f.stride.empty()) return true;
        if (f.stride.size() != f.shape.size()) invalid("shape and stride ranks differ");
        int64_t step = 1;
        for (size_t i = f.shape.size(); i > 0; --i) {
            if (f.shape[i - 1] != 1 && f.stride[i - 1] != step) return false;
            step = multiply(step, f.shape[i - 1]);
        }
        return true;
    }

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
        carriers_.reserve(ops.size());
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto value = ops[i];
            Carrier carrier;
            if (!value.is_none()) {
                if (value.attr("opaque_bytes").cast<bool>()) invalid("opaque scale factors require the Python binder");
                auto shape       = value.attr("shape").cast<std::vector<int64_t>>();
                carrier.elements = 1;
                for (auto n : shape) carrier.elements = multiply(carrier.elements, n);
                carrier.allowed_numels = value.attr("allowed_numels").cast<std::vector<int64_t>>();
                carrier.contiguous     = roles[i] == "seq_q" || roles[i] == "seq_kv" || roles[i] == "sink" ||
                                     roles[i] == "dsink" || roles[i] == "bias" || roles[i] == "dbias";
                carrier.length_bit = roles[i] == "seq_q" ? 0 : roles[i] == "seq_kv" ? 1 : -1;
            }
            add_operand(roles[i], value, declared[i]);
            if (!value.is_none() && value.attr("itemsize").cast<int64_t>() != operands_.back().bits / 8)
                invalid("operand dtype and element width disagree");
            carriers_.push_back(std::move(carrier));
        }
    }

    py::tuple
    bind(const py::handle &pack,
         const std::vector<int64_t> &indices,
         int64_t workspace,
         int64_t stream,
         const std::vector<int64_t> &overridden,
         const py::object &scale    = py::none(),
         bool raw_storage           = true,
         const py::object &declared = py::none()) const {
        if (workspace <= 0 || workspace % 16) invalid("needs an aligned caller workspace");
        const auto workspace_end = end(workspace, workspace_bytes_);
        if (indices.size() != operands_.size()) invalid("incorrect native backward role indices");
        const auto facts            = read_native_operand_views(pack, indices);
        const auto runtime_geometry = declared.is_none() ? py::tuple() : declared.cast<py::tuple>();
        if (!declared.is_none() && runtime_geometry.size() != operands_.size())
            invalid("incorrect standalone geometry count");
        int length_form = 0;
        py::tuple frame(operands_.size() + 3 + (scale_log2_ ? 1 : 0) + (length_form_ ? 1 : 0));
        for (size_t i = 0; i < operands_.size(); ++i) {
            const auto &op      = operands_[i];
            const auto &f       = facts[i];
            const auto &carrier = carriers_[i];
            const auto label    = op.name == "seq_q" || op.name == "seq_kv" ? op.name + "_lens" : op.name;
            if (!validate(op, f, label)) {
                frame[i] = py::none();
                continue;
            }
            int64_t elements = 0;
            if (!raw_storage && !f.shape.empty() && (carrier.contiguous || !carrier.allowed_numels.empty())) {
                elements = numel(f);
                if (carrier.contiguous &&
                    (!contiguous(f) ||
                     (carrier.allowed_numels.empty()
                          ? elements != carrier.elements
                          : std::find(carrier.allowed_numels.begin(), carrier.allowed_numels.end(), elements) ==
                                carrier.allowed_numels.end())))
                    invalid(label + " must be contiguous with " + std::to_string(carrier.elements) + " elements");
                if (length_form_ && carrier.length_bit >= 0 && carrier.elements < std::numeric_limits<int64_t>::max() &&
                    elements == carrier.elements + 1)
                    length_form |= 1 << carrier.length_bit;
            }
            if (!f.shape.empty()) {
                if (!declared.is_none() && !runtime_geometry[i].is_none()) {
                    const auto value = runtime_geometry[i].cast<py::tuple>();
                    auto shape       = value[0].cast<std::vector<int64_t>>();
                    auto stride      = value[1].cast<std::vector<int64_t>>();
                    if (geometry(f.shape, f.stride) != geometry(shape, stride))
                        invalid(op.name + " runtime geometry must match this fixed backward plan");
                } else if (std::find(overridden.begin(), overridden.end(), indices[i]) != overridden.end() &&
                           geometry(f.shape, f.stride) != op.geometry) {
                    invalid(op.name + " runtime geometry must match this fixed backward plan");
                }
            }
            // Graph operands are raw storage under their declarations. A standalone
            // B+1 prefix includes one more live element than the graph's B lengths.
            const auto bytes = !raw_storage && !f.shape.empty() && !carrier.allowed_numels.empty()
                                   ? multiply(elements, op.bits / 8)
                                   : op.bytes;
            if (f.observed_bytes >= 0 && f.observed_bytes < bytes)
                invalid(op.name + " backing storage is too small for the bound length form");
            if (workspace < end(f.pointer, bytes) && f.pointer < workspace_end)
                invalid("caller workspace overlaps " + op.name);
            frame[i] = py::int_(f.pointer);
        }
        size_t slot            = operands_.size();
        const auto scale_value = scale.is_none() ? scale_ : scale.cast<double>();
        frame[slot++]          = py::int_(workspace);
        if (scale_log2_) frame[slot++] = py::float_(scale_value * 1.4426950408889634);
        frame[slot++] = py::float_(scale_value);
        if (length_form_)
            frame[slot++] =
                py::int_(length_form);  // Graph declarations bind B lengths; standalone also accepts B+1 prefixes.
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
        .def("bind",
             &SdpaBwdBinder::bind,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("workspace"),
             py::arg("stream"),
             py::arg("overridden"),
             py::arg("scale")       = py::none(),
             py::arg("raw_storage") = true,
             py::arg("geometry")    = py::none())
        .def("execute", &SdpaBwdBinder::execute);
}
}  // namespace python_bindings
}  // namespace cudnn_frontend
