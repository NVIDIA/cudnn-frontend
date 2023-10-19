#include "pybind11/pybind11.h"

#include "cudnn_frontend.h"
#include "pyplans.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend::python_bindings {

void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const& error_msg);

void
PyPlans::filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const& notes) {
    plans.filter_out_numeric_notes(notes);
    return;
}

void
PyPlans::filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const& notes) {
    plans.filter_out_behavior_notes(notes);
    return;
}

void
PyPlans::filter_out_workspace_greater_than(int64_t const workspace) {
    plans.filter_out_workspace_greater_than(workspace);
    return;
}

void
PyPlans::build_all_plans() {
    auto status = plans.build_all_plans(handle);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyPlans::check_support() {
    auto status = plans.check_support(handle);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

int64_t
PyPlans::get_max_workspace_size() {
    return plans.get_max_workspace_size();
}

void
init_pyplans_submodule(py::module_& m) {
    py::class_<PyPlans> pyplans_(m, "pyplans");
    pyplans_.def("filter_out_numeric_notes", &PyPlans::filter_out_numeric_notes)
        .def("filter_out_behavior_notes", &PyPlans::filter_out_behavior_notes)
        .def("filter_out_workspace_greater_than", &PyPlans::filter_out_workspace_greater_than)
        .def("build_all_plans", &PyPlans::build_all_plans)
        .def("check_support", &PyPlans::check_support)
        .def("get_max_workspace_size", &PyPlans::get_max_workspace_size);
}

}  // namespace cudnn_frontend::python_bindings