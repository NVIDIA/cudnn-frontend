// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Python bindings of cudnn_frontend::ops (include/cudnn_frontend/ops/): the
// Python ops run the same C++ op C++ callers do, with kernels from the
// in-process registry. Tensors arrive as (address, cudnn.data_type, shape,
// strides) tuples, and a repeated call passes only addresses.

#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "cudnn_frontend.h"
#include "cudnn_frontend/ops/grouped_gemm.h"

namespace py = nanobind;

namespace cudnn_frontend {
namespace python_bindings {

void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const &error_msg);

namespace {

using namespace cudnn_frontend::ops;

void
check(error_t const &e) {
    throw_if(e.is_bad(), e.code, e.err_msg);
}

TensorView
view(py::handle desc) {
    auto t = py::cast<py::tuple>(desc);
    TensorView v;
    v.data     = reinterpret_cast<void *>(py::cast<intptr_t>(t[0]));
    v.dtype    = py::cast<DataType_t>(t[1]);
    auto shape = py::cast<py::sequence>(t[2]), stride = py::cast<py::sequence>(t[3]);
    v.rank = static_cast<int>(py::len(shape));
    if (v.rank > 8 || static_cast<int>(py::len(stride)) != v.rank) {
        throw std::invalid_argument("cudnn.ops: a tensor has rank at most 8 and one stride per dimension");
    }
    for (int i = 0; i < v.rank; i++) {
        v.dim[i]    = py::cast<int64_t>(shape[i]);
        v.stride[i] = py::cast<int64_t>(stride[i]);
    }
    return v;
}

// (name, tensor tuple) pairs, in the order execute() takes their addresses.
std::vector<GroupedGemmOperand>
grouped_gemm_operands(py::sequence operands) {
    std::vector<GroupedGemmOperand> out;
    for (py::handle item : operands) {
        auto pair = py::cast<py::tuple>(item);
        out.push_back({py::cast<std::string>(pair[0]), view(pair[1])});
    }
    return out;
}

}  // namespace

void
init_ops_submodule(py::module_ &m) {
    py::class_<GroupedGemm>(m, "_GroupedGemm")
        .def("__init__",
             [](GroupedGemm *self, std::string kind, py::sequence operands, std::string options) {
                 GroupedGemm plan;
                 check(GroupedGemm::create(
                     kind, grouped_gemm_operands(operands), json::parse(options), KernelLibrary::registry(), plan));
                 new (self) GroupedGemm(std::move(plan));
             })
        .def_prop_ro("workspace_size", &GroupedGemm::workspace_size)
        // The operands' addresses in create()'s order; False when one is less aligned than create()'s (nothing ran).
        .def("execute", [](GroupedGemm const &self, py::list addresses, intptr_t workspace, intptr_t stream) {
            constexpr size_t MAX_OPERANDS = 64;
            void *data[MAX_OPERANDS];
            size_t const n = std::min(py::len(addresses), MAX_OPERANDS);
            for (size_t i = 0; i < n; i++) {
                data[i] = reinterpret_cast<void *>(py::cast<intptr_t>(addresses[i]));
            }
            auto const e =
                self.execute(data, n, reinterpret_cast<void *>(workspace), reinterpret_cast<cudaStream_t>(stream));
            if (e.code == error_code_t::INVALID_VARIANT_PACK) {
                return false;
            }
            check(e);
            return true;
        });
    m.def("_grouped_gemm_config", [](std::string kind, py::sequence operands, std::string options) {
        json config;
        check(GroupedGemm::config(kind, grouped_gemm_operands(operands), json::parse(options), config));
        return config.dump();
    });
}

}  // namespace python_bindings
}  // namespace cudnn_frontend
