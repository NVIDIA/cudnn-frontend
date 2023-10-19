#include "pybind11/pybind11.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend::python_bindings {

class PyPlans {
   public:
    cudnn_frontend::graph::Plans plans;
    cudnnHandle_t handle;

    void
    filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const& notes);

    void
    filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const& notes);

    void
    filter_out_workspace_greater_than(int64_t const workspace);

    void
    build_all_plans();

    void
    check_support();

    int64_t
    get_max_workspace_size();
};

}  // namespace cudnn_frontend::python_bindings