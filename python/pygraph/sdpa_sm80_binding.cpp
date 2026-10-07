// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SM80 fixed forward contracts share native storage checks with half backward.
#include "sdpa_fixed_binding.h"

namespace py = pybind11;
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
        : FixedSdpaOperands("sdpa_fwd_sm80", spec.attr("device_index").cast<int64_t>()),
          fn_(spec.attr("fn")),
          owner_(spec.attr("artifact")),
          scale_(spec.attr("scale").cast<double>()) {
        const std::vector<std::string> roles{"q", "k", "v", "o", "stats", "seq_kv", "seq_q", "sink", "bias", "rope"};
        const auto ops = spec.attr("operands").cast<py::tuple>();
        if (ops.size() != 9 && ops.size() != roles.size()) invalid("invalid native forward operand declarations");
        for (size_t i = 0; i < ops.size(); ++i) {
            Carrier carrier;
            py::object declared = py::none();
            if (!ops[i].is_none()) {
                auto shape         = ops[i].attr("shape").cast<std::vector<int64_t>>();
                auto strides       = ops[i].attr("strides").cast<std::vector<int64_t>>();
                declared           = py::make_tuple(shape, strides);
                carrier.numel      = numel(shape);
                carrier.contiguous = ops[i].attr("contiguous").cast<bool>();
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
        py::tuple frame(operands_.size() + 3);
        for (size_t i = 0; i < operands_.size(); ++i) {
            const auto &op      = operands_[i];
            const auto &f       = facts[i];
            const auto &carrier = carriers_[i];
            if (!validate(op, f, op.name)) {
                frame[i] = py::none();
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
            frame[i] = py::int_(f.pointer);
        }
        double current_scale = scale.is_none() ? scale_ : scale.cast<double>();
        if (current_scale == 0.0) throw py::value_error("attn_scale = 0 is not supported on this kernel (#1435)");
        frame[operands_.size()]     = py::float_(current_scale * 1.4426950408889634);
        frame[operands_.size() + 1] = py::float_(1.0 / current_scale);
        frame[operands_.size() + 2] = py::int_(stream);
        return frame;
    }

    void
    execute(const py::handle &pack,
            const std::vector<int64_t> &indices,
            int64_t stream,
            const py::object &scale,
            const std::vector<int64_t> &overridden,
            bool raw_storage) const {
        const auto frame = bind(pack, indices, stream, scale, overridden, raw_storage);
        auto result      = py::reinterpret_steal<py::object>(PyObject_CallObject(fn_.ptr(), frame.ptr()));
        if (!result) throw py::error_already_set();
    }
};
}  // namespace

void
init_sdpa_sm80_binding(py::module_ &m) {
    py::class_<SdpaSm80FwdBinder>(m, "_SdpaSm80FwdBinder")
        .def(py::init<const py::object &>())
        .def("bind", &SdpaSm80FwdBinder::bind)
        .def("execute", &SdpaSm80FwdBinder::execute);
}
}  // namespace python_bindings
}  // namespace cudnn_frontend
