/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cudnn_frontend.h"
#include "pygraph.h"

namespace py = nanobind;

namespace cudnn_frontend::python_bindings {

std::vector<PyGraph::Tensor_t>
PyGraph::custom_reduction(std::vector<Tensor_t> inputs,
                          py::callable reducer,
                          std::string const& reducer_semantic_key,
                          std::vector<int64_t> reduction_dims,
                          std::vector<cudnn_frontend::graph::Custom_reduction_identity> identity_values,
                          std::optional<std::vector<cudnn_frontend::DataType_t>> compute_data_types,
                          std::optional<cudnn_frontend::DataType_t> compute_data_type,
                          std::string const& name) {
    if (compute_data_types.has_value() && compute_data_type.has_value()) {
        throw py::value_error("custom_reduction takes compute_data_types or compute_data_type, not both");
    }

    auto owner           = callback_root_owner();
    auto state           = std::make_shared<CustomCallbackState>();
    state->reducer       = std::move(reducer);
    state->owner_weakref = py::weakref(owner);
    auto& states         = py::cast<PyGraph&>(owner).custom_callback_states;
    states.erase(std::remove_if(states.begin(), states.end(), [](auto const& weak) { return weak.expired(); }),
                 states.end());
    states.emplace_back(state);

    auto reducer_wrapper = [state = std::move(state)](Graph_t child_graph,
                                                      std::vector<Tensor_t> lhs,
                                                      std::vector<Tensor_t> rhs) mutable -> std::vector<Tensor_t> {
        py::gil_scoped_acquire gil;
        if (!state->reducer || !state->owner_weakref) {
            throw py::value_error("Custom reduction callback owner was released");
        }
        auto owner = state->owner_weakref();
        if (owner.is_none()) {
            throw py::value_error("Custom reduction callback owner was released");
        }

        // The reducer's logical construction scope is this callback. Use the
        // shared_ptr caster so an accidentally retained Python
        // reference remains memory-safe instead of pointing at stack storage.
        auto child_proxy = std::make_shared<PyGraph>(std::move(child_graph), std::move(owner));
        auto result      = state->reducer(child_proxy, std::move(lhs), std::move(rhs));
        return py::cast<std::vector<Tensor_t>>(result);
    };

    auto attributes = cudnn_frontend::graph::Custom_reduction_attributes()
                          .set_name(name)
                          .set_reducer(std::move(reducer_wrapper), reducer_semantic_key)
                          .set_reduction_dims(std::move(reduction_dims))
                          .set_identity_values(std::move(identity_values));
    if (compute_data_types.has_value()) {
        attributes.set_compute_data_types(std::move(compute_data_types.value()));
    }
    if (compute_data_type.has_value()) {
        attributes.set_compute_data_type(compute_data_type.value());
    }

    return graph->custom_reduction(std::move(inputs), std::move(attributes));
}

void
init_pygraph_custom_reduction_submodule(PyGraphClass& m) {
    m.def("custom_reduction",
          &PyGraph::custom_reduction,
          py::arg("inputs"),
          py::arg("reducer"),
          py::arg("reducer_semantic_key"),
          py::arg("reduction_dims"),
          py::arg("identity_values"),
          py::kw_only(),
          py::arg("compute_data_types") = std::nullopt,
          py::arg("compute_data_type")  = std::nullopt,
          py::arg("name")               = "",
          R"pbdoc(
                Define a variadic custom reduction.

                The reducer is invoked once while the operation graph is built.
                It receives a shared-owned, callback-scoped backend_graph plus
                lhs and rhs tensor lists, and must return one
                tensor per input. Retaining the proxy is memory-safe, but do not
                edit the child graph after the callback returns: the frontend
                does not check such an edit, and the edit can reach the reducer
                body that cuDNN gets.
                The semantic key must change whenever reducer behavior changes.
                compute_data_types sets one compute type per stream;
                compute_data_type sets the same type for every stream. Give at
                most one of them. With neither, each stream computes in its own
                type.
            )pbdoc");
}

}  // namespace cudnn_frontend::python_bindings
