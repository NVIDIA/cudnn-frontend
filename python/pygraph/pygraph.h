#include <utility>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend::python_bindings {

// This class is only meant direct pythonic API calls to c++ Graph class.
class PyGraph {
   public:
    using Tensor_t   = std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>;
    using Graph_t    = std::shared_ptr<cudnn_frontend::graph::Graph>;
    using PyCallback = std::function<Tensor_t(PyGraph&, Tensor_t)>;

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
    Graph_t graph;
    cudnnHandle_t handle = nullptr;
    bool is_handle_owner = false;

    std::optional<PyCallback> callback_fn;

    PyGraph(Graph_t graph_) : graph(graph_) {};

    PyGraph(std::string const&,
            cudnn_frontend::DataType_t io_data_type,
            cudnn_frontend::DataType_t intermediate_data_type,
            cudnn_frontend::DataType_t compute_data_type,
            std::optional<std::intptr_t> handle_,
            py::object sm_count,
            py::object sm_version,
            std::shared_ptr<KernelCache> kernel_cache,
            std::shared_ptr<cudnn_frontend::DeviceProperties> device_properties)
        : graph(std::make_shared<cudnn_frontend::graph::Graph>()) {
        graph->set_compute_data_type(compute_data_type)
            .set_intermediate_data_type(intermediate_data_type)
            .set_io_data_type(io_data_type);

        // If device_properties is set, use it (consider it is an AoT compilation test).
        if (device_properties != nullptr) {
            graph->set_device_properties(device_properties);
        } else if (handle_.has_value()) {
            handle = static_cast<cudnnHandle_t>((void*)(handle_.value()));
        } else {
            detail::create_handle(&handle);
            is_handle_owner = true;
        }

        if (sm_count.is(py::none()) == false) {
            graph->set_sm_count(sm_count.cast<int32_t>());
        }

        if (sm_version.is(py::none()) == false) {
            graph->set_sm_version(sm_version.cast<int32_t>());
        }

        if (kernel_cache) {
            graph->set_kernel_cache(kernel_cache);
            graph->set_dynamic_shape_enabled(true);
        }
    }

    ~PyGraph() {
        if (is_handle_owner) {
            detail::destroy_handle(handle);
        }
    }

    std::function<Tensor_t(Graph_t, Tensor_t)> wrapper_function = [this](Graph_t graph, Tensor_t q_kt) {
        auto py_graph = std::make_shared<PyGraph>(graph);

        if (callback_fn.has_value()) {
            q_kt = this->callback_fn.value()(*py_graph, q_kt);
        }

        return q_kt;
    };

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    tensor(std::vector<int64_t> const& dim,
           std::vector<int64_t> const& stride,
           cudnn_frontend::DataType_t const& data_type,
           bool const& is_virtual,
           bool const& is_pass_by_value,
           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& ragged_offset,
           cudnn_frontend::TensorReordering_t const reordering_type,
           std::string const& name,
           int64_t const& uid);

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
    adalayernorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                 std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
                 cudnn_frontend::DataType_t const& compute_data_type,
                 std::string const& name);

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    adalayernorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
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
    slice(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
          std::vector<py::slice> const& slices,
          cudnn_frontend::DataType_t const& compute_data_type,
          std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_fprop(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
               std::vector<int64_t> const& pre_padding,
               std::vector<int64_t> const& post_padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const& conv_mode,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_dgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& filter,
               std::vector<int64_t> const& pre_padding,
               std::vector<int64_t> const& post_padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const& conv_mode,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_wgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::vector<int64_t> const& pre_padding,
               std::vector<int64_t> const& post_padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const& conv_mode,
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
         std::optional<float> const& negative_slope,
         std::optional<float> const& lower_clip,
         std::optional<float> const& upper_clip,
         cudnn_frontend::DataType_t const& compute_data_type,
         std::string const& name);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    block_scale_dequantize(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale,
                           std::vector<int32_t> const& block_size,
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

                  std::optional<float> const& negative_slope,
                  std::optional<float> const& lower_clip,
                  std::optional<float> const& upper_clip,
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

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    reshape(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input, std::string const& name);

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

    // return [o, stats]
    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2>
    sdpa(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
         py::object const& is_inference,
         py::object const& attn_scale,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
         bool const use_alibi_mask,
         bool const use_padding_mask,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
         bool const use_causal_mask,
         bool const use_causal_mask_bottom_right,
         py::object const& sliding_window_length,
         cudnn_frontend::DiagonalAlignment_t const& diagonal_alignment,
         py::object const& left_bound,
         py::object const& right_bound,
         py::object const& dropout,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& paged_attention_k_table,
         std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& paged_attention_v_table,
         py::object const& paged_attention_max_seq_len_kv,
         cudnn_frontend::DataType_t const& compute_data_type,
         std::string const& name,
         std::optional<PyCallback> fn,
         py::object const& generate_stats,
         cudnn_frontend::AttentionImplementation_t const& implementation);

    // return [dQ, dK, dV]
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
                  py::object const& max_total_seq_len_q,
                  py::object const& max_total_seq_len_kv,
                  bool const use_causal_mask,
                  bool const use_causal_mask_bottom_right,
                  py::object const& sliding_window_length,
                  cudnn_frontend::DiagonalAlignment_t const& diagonal_alignment,
                  py::object const& left_bound,
                  py::object const& right_bound,
                  py::object const& dropout,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
                  bool const use_deterministic_algorithm,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name);

    // return [o, stats, amax_s, amax_o]
    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 4>
    sdpa_fp8(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_q,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_k,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_v,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_s,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_s,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_o,
             py::object const& is_inference,
             py::object const& attn_scale,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
             bool const use_alibi_mask,
             bool const use_padding_mask,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
             bool const use_causal_mask,
             bool const use_causal_mask_bottom_right,
             py::object const& sliding_window,
             cudnn_frontend::DiagonalAlignment_t const& diagonal_alignment,
             py::object const& left_bound,
             py::object const& right_bound,
             py::object const& dropout,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& paged_attention_k_table,
             std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& paged_attention_v_table,
             py::object const& paged_attention_max_seq_len_kv,
             cudnn_frontend::DataType_t const& compute_data_type,
             std::string const& name,
             std::optional<PyCallback> fn,
             py::object const& generate_stats);

    // return [dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP]
    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 7>
    sdpa_fp8_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& o,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& dO,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& stats,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_q,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_k,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_v,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_o,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_dO,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_s,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale_dP,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_s,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_dQ,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_dK,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_dV,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_dP,
                      py::object const& attn_scale,
                      bool const use_padding_mask,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
                      bool const use_causal_mask,
                      bool const use_causal_mask_bottom_right,
                      py::object const& dropout,
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
    create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const& knobs);

    int64_t
    get_engine_count();

    std::vector<Knob>
    get_knobs_for_engine(int64_t const engine_id);

    void
    build_plans(BuildPlanPolicy_t const);

    void
    build_plan_at_index(int64_t const index);

    void
    check_support();

    void
    build(std::vector<cudnn_frontend::HeurMode_t> const&);

    void
    build();

    int64_t
    get_workspace_size();

    void
    populate_cuda_graph(std::intptr_t handle,
                        std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, int64_t> var_pack,
                        std::intptr_t workspace,
                        std::intptr_t cuda_graph);

    void
    update_cuda_graph(std::intptr_t handle,
                      std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, int64_t> var_pack,
                      std::intptr_t workspace,
                      std::intptr_t cuda_graph);

    void
    execute(std::unordered_map<int64_t, int64_t> var_pack, int64_t workspace, std::optional<std::intptr_t>);

    void
    execute_plan_at_index(std::unordered_map<int64_t, int64_t> var_pack,
                          int64_t workspace,
                          int64_t index,
                          std::optional<std::intptr_t>);

    std::vector<BehaviorNote_t>
    get_behavior_notes();

    std::vector<BehaviorNote_t>
    get_behavior_notes_for_plan_at_index(int64_t const index);

    void
    select_numeric_notes(std::vector<NumericalNote_t> const& notes) {
        graph->select_numeric_notes(notes);
        return;
    }

    void
    select_behavior_notes(std::vector<BehaviorNote_t> const& notes) {
        graph->select_behavior_notes(notes);
        return;
    }

    void
    deselect_engines(std::vector<std::string> const& engine_names) {
        graph->deselect_engines(engine_names);
        return;
    }

    void
    deselect_numeric_notes(std::vector<NumericalNote_t> const& notes) {
        graph->deselect_numeric_notes(notes);
        return;
    }

    void
    deselect_behavior_notes(std::vector<BehaviorNote_t> const& notes) {
        graph->deselect_behavior_notes(notes);
        return;
    }

    void
    deselect_workspace_greater_than(int64_t const workspace) {
        graph->deselect_workspace_greater_than(workspace);
        return;
    }

    std::vector<uint8_t>
    serialize() const;

    void
    deserialize(std::optional<std::intptr_t> handle_, py::object const& pyobj);

    void
    deserialize(py::object const& pyobj);

    int64_t
    get_execution_plan_count() const {
        return graph->get_execution_plan_count();
    }

    int64_t
    get_workspace_size_plan_at_index(int64_t index);

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    query_tensor_attributes_of_uid(int64_t const uid) const;

    std::string
    get_plan_name_at_index(int64_t index);

   private:
    // Internal SDPA implementation - delegates to sdpa() or sdpa_fp8() based on mma_core_mode
    // return SDPA_outputs struct: {O, Stats, RNG_DUMP, Amax_S, Amax_O}
    cudnn_frontend::graph::SDPA_attributes::SDPA_outputs
    sdpa_internal(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
                  py::object const& attn_scale,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                  bool const use_alibi_mask,
                  bool const use_padding_mask,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
                  cudnn_frontend::DiagonalAlignment_t const& diagonal_alignment,
                  py::object const& left_bound,
                  py::object const& right_bound,
                  py::object const& dropout,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& rng_dump,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& paged_attention_k_table,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& paged_attention_v_table,
                  py::object const& paged_attention_max_seq_len_kv,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name,
                  std::optional<PyCallback> fn,
                  py::object const& generate_stats,
                  cudnn_frontend::DataType_t const& mma_core_mode = cudnn_frontend::DataType_t::HALF,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> descale_q = nullptr,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> descale_k = nullptr,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> descale_v = nullptr,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> descale_s = nullptr,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> scale_s   = nullptr,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> scale_o   = nullptr,
                  cudnn_frontend::AttentionImplementation_t const& implementation = AttentionImplementation_t::AUTO);
};

}  // namespace cudnn_frontend::python_bindings