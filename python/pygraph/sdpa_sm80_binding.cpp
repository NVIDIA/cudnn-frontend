// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SM80 fixed forward contracts share native storage checks with half backward.
#include <cmath>

#include "sdpa_fixed_binding.h"

namespace py = nanobind;
namespace cudnn_frontend {
namespace python_bindings {
namespace {
class SdpaSm80FwdBinder : private FixedSdpaOperands {
    struct Carrier {
        int64_t numel   = 0;
        bool contiguous = false;
        Geometry bias_plane;
    };
    py::object fn_, owner_;
    std::vector<Carrier> carriers_;
    double scale_;

    int64_t
    numel(std::span<const int64_t> shape) const {
        int64_t result = 1;
        for (auto n : shape) result = multiply(result, n);
        return result;
    }
    bool
    contiguous(const NativeOperandView &f) const {
        if (f.stride.empty()) return true;
        int64_t step = 1;
        for (size_t i = f.shape.size(); i > 0; --i) {
            if (f.shape[i - 1] != 1 && f.stride[i - 1] != step) return false;
            step = multiply(step, f.shape[i - 1]);
        }
        return true;
    }

   public:
    explicit SdpaSm80FwdBinder(const py::object &spec)
        : FixedSdpaOperands("sdpa_fwd_sm80", py::cast<int64_t>(spec.attr("device_index"))),
          fn_(spec.attr("fn")),
          owner_(spec.attr("artifact")),
          scale_(py::cast<double>(spec.attr("scale"))) {
        const std::vector<std::string> roles{"q", "k", "v", "o", "stats", "seq_kv", "seq_q", "sink", "bias", "rope"};
        const auto ops = py::cast<py::tuple>(spec.attr("operands"));
        if (ops.size() != 9 && ops.size() != roles.size()) invalid("invalid native forward operand declarations");
        for (size_t i = 0; i < ops.size(); ++i) {
            Carrier carrier;
            py::object declared = py::none();
            if (!ops[i].is_none()) {
                auto shape         = py::cast<std::vector<int64_t>>(ops[i].attr("shape"));
                auto strides       = py::cast<std::vector<int64_t>>(ops[i].attr("strides"));
                declared           = py::make_tuple(shape, strides);
                carrier.numel      = numel(shape);
                carrier.contiguous = py::cast<bool>(ops[i].attr("contiguous"));
                if (roles[i] == "bias") {
                    if (shape.size() != 4 || strides.size() != 4) invalid("invalid fixed bias declaration");
                    carrier.bias_plane = geometry(std::span(shape).subspan(1), std::span(strides).subspan(1));
                }
            }
            add_operand(roles[i], ops[i], declared);
            carriers_.push_back(std::move(carrier));
        }
    }

    py::tuple
    bind(const py::handle &pack,
         const std::vector<int64_t> &indices,
         int64_t stream,
         const py::object &scale,
         const std::vector<int64_t> &overridden,
         bool raw_storage) const {
        if (indices.size() != operands_.size()) invalid("incorrect native forward role indices");
        const auto facts = read_native_operand_views(pack, indices);
        py::list frame;
        for (size_t i = 0; i < operands_.size(); ++i) {
            const auto &op      = operands_[i];
            const auto &f       = facts[i];
            const auto &carrier = carriers_[i];
            if (!validate(op, f, op.name)) {
                frame.append(py::none());
                continue;
            }
            if (!f.shape.empty()) {
                if (!raw_storage && (op.name == "seq_q" || op.name == "seq_kv" || op.name == "sink")) {
                    if (!contiguous(f) || numel(f.shape) != carrier.numel)
                        invalid(op.name + " must be contiguous with " + std::to_string(carrier.numel) + " elements");
                } else if (!raw_storage && op.name == "stats") {
                    if (numel(f.shape) != carrier.numel || (carrier.contiguous && !contiguous(f)))
                        invalid("Stats must match the declared element count and storage layout");
                } else if (!raw_storage && op.name == "bias") {
                    if (f.shape.size() != 4 || f.shape[0] < 1 ||
                        geometry(f.shape.subspan(1), f.stride.empty() ? f.stride : f.stride.subspan(1)) !=
                            carrier.bias_plane)
                        invalid("bias must have a contiguous [H,SQ,SKV] first plane");
                } else if ((!raw_storage ||
                            std::find(overridden.begin(), overridden.end(), indices[i]) != overridden.end()) &&
                           geometry(f.shape, f.stride) != op.geometry) {
                    invalid(op.name + " runtime geometry must match this fixed forward plan");
                }
            }
            frame.append(py::int_(f.pointer));
        }
        const double current_scale = scale.is_none() ? scale_ : py::cast<double>(scale);
        // The kernel folds the scale's sign into the scores at compile time (score_sign) and runs at |scale|, or 1 for
        // 0.
        if ((current_scale > 0) != (scale_ > 0) || (current_scale < 0) != (scale_ < 0))
            throw py::value_error("sdpa_fwd_sm80: attn_scale sign must match the compiled plan's");
        const double kernel_scale = current_scale == 0.0 ? 1.0 : std::fabs(current_scale);
        frame.append(py::float_(kernel_scale * 1.4426950408889634));
        frame.append(py::float_(1.0 / kernel_scale));
        frame.append(py::int_(stream));
        return py::tuple(frame);
    }

    void
    execute(const py::handle &pack,
            const std::vector<int64_t> &indices,
            int64_t stream,
            const py::object &scale,
            const std::vector<int64_t> &overridden,
            bool raw_storage) const {
        const auto frame = bind(pack, indices, stream, scale, overridden, raw_storage);
        auto result      = py::steal<py::object>(PyObject_CallObject(fn_.ptr(), frame.ptr()));
        if (!result) throw py::python_error();
    }
};
}  // namespace

void
init_sdpa_sm80_binding(py::module_ &m) {
    py::class_<SdpaSm80FwdBinder>(m, "_SdpaSm80FwdBinder")
        .def(py::init<const py::object &>())
        .def("bind",
             &SdpaSm80FwdBinder::bind,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("stream"),
             py::arg("scale").none(),
             py::arg("overridden"),
             py::arg("raw_storage"))
        .def("execute",
             &SdpaSm80FwdBinder::execute,
             py::arg("pack"),
             py::arg("indices"),
             py::arg("stream"),
             py::arg("scale").none(),
             py::arg("overridden"),
             py::arg("raw_storage"));
}
}  // namespace python_bindings
}  // namespace cudnn_frontend
