#include <utility>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend::python_bindings {

// This class is only meant direct pythonic API calls to c++ Graph class.
class PyGraph {
   public:
    template <cudnn_frontend::PointwiseMode_t MODE>
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    pointwise_ternary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& b,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& c,
                      cudnn_frontend::DataType_t const& compute_data_type,
                      std::string const& name);

    template <cudnn_frontend::PointwiseMode_t MODE>
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    pointwise_binary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& b,
                     cudnn_frontend::DataType_t const& compute_data_type,
                     std::string const& name);

    template <cudnn_frontend::PointwiseMode_t MODE>
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    pointwise_unary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name);

    // This Graph class is the sole structure which implicitly makes PyGraph own all tensors, nodes, and cudnn
    // descriptors.
    cudnn_frontend::graph::Graph graph;
    cudnnHandle_t handle;
    bool is_handle_owner;

    PyGraph(std::string const&,
            cudnn_frontend::DataType_t io_data_type,
            cudnn_frontend::DataType_t intermediate_data_type,
            cudnn_frontend::DataType_t compute_data_type,
            void* handle_ = nullptr)
        : graph(), handle((cudnnHandle_t)handle_), is_handle_owner(false) {
        graph.set_compute_data_type(compute_data_type)
            .set_intermediate_data_type(intermediate_data_type)
            .set_io_data_type(io_data_type);

        if (handle_ == nullptr) {
            cudnnCreate(&handle);
            is_handle_owner = true;
        }
    }

    ~PyGraph() {
        if (is_handle_owner) {
            cudnnDestroy(handle);
        }
    }

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    tensor(std::vector<int64_t> const& dim,
           std::vector<int64_t> const& stride,
           cudnn_frontend::DataType_t const& data_type,
           bool const& is_virtual,
           bool const& is_pass_by_value,
           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& ragged_offset,
           std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    tensor_like(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& pyobj, std::string const&);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    tensor_like(py::object const& pyobj);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    batchnorm(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& in_running_mean,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& in_running_var,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& momentum,
              std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& peer_stats,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    layernorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    batchnorm_inference(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& mean,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& inv_variance,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                        cudnn_frontend::DataType_t const& compute_data_type,
                        std::string const& name);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    layernorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& mean,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                       cudnn_frontend::DataType_t const& compute_data_type,
                       std::string const& name);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    batchnorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& mean,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                       std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& peer_stats,
                       cudnn_frontend::DataType_t const& compute_data_type,
                       std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_fprop(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
               std::vector<int64_t> const& pre_padding,
               std::vector<int64_t> const& post_padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_dgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& filter,
               std::vector<int64_t> const& pre_padding,
               std::vector<int64_t> const& post_padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_wgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::vector<int64_t> const& pre_padding,
               std::vector<int64_t> const& post_padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    matmul(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& A,
           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& B,
           cudnn_frontend::DataType_t const& compute_data_type,
           double const padding,
           std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    relu(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
         float const negative_slope,
         cudnn_frontend::DataType_t const& compute_data_type,
         std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    gen_index(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
              int64_t const axis,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    relu_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                  float const negative_slope,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    leaky_relu_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                        float const negative_slope,
                        cudnn_frontend::DataType_t const& compute_data_type,
                        std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    leaky_relu(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
               float const negative_slope,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name);

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2UL>
    genstats(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
             cudnn_frontend::DataType_t const& compute_data_type,
             std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    reduction(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
              cudnn_frontend::ReductionMode_t const mode,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    rmsnorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
            cudnn_frontend::DataType_t const& compute_data_type,
            std::string const& name);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    rmsnorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                     bool const has_dbias,
                     cudnn_frontend::DataType_t const& compute_data_type,
                     std::string const& name);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    instancenorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
                 cudnn_frontend::DataType_t const& compute_data_type,
                 std::string const& name);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    instancenorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                          std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                          std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                          std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& mean,
                          std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                          cudnn_frontend::DataType_t const& compute_data_type,
                          std::string const& name);

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2>
    sdpa(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
         bool const is_inference,
         py::object const& attn_scale,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
         bool const use_alibi_mask,
         bool const use_padding_mask,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
         bool const use_causal_mask,
         py::object const& dropout,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
         cudnn_frontend::DataType_t const& compute_data_type,
         std::string const& name);

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 3>
    sdpa_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& o,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& dO,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& stats,
                  py::object const& attn_scale,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& dBias,
                  bool const use_alibi_mask,
                  bool const use_padding_mask,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
                  bool const use_causal_mask,
                  py::object const& dropout,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name);

    void
    validate();

    size_t
    key();

    void
    build_operation_graph();

    void
    create_execution_plans(std::vector<cudnn_frontend::HeurMode_t> const&);

    void
    build_plans(BuildPlanPolicy_t const);

    void
    check_support();

    void
    build(std::vector<cudnn_frontend::HeurMode_t> const&);

    int64_t
    get_workspace_size();

    void
    execute(std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, py::object> var_pack,
            py::object workspace);

    void
    execute(std::unordered_map<int64_t, py::object> var_pack, py::object workspace);

    void
    deselect_numeric_notes(std::vector<NumericalNote_t> const& notes) {
        graph.deselect_numeric_notes(notes);
        return;
    }

    void
    deselect_behavior_notes(std::vector<BehaviorNote_t> const& notes) {
        graph.deselect_behavior_notes(notes);
        return;
    }

    void
    deselect_workspace_greater_than(int64_t const workspace) {
        graph.deselect_workspace_greater_than(workspace);
        return;
    }
};

}  // namespace cudnn_frontend::python_bindings