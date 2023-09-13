#include <utility>
#include <unordered_map>

#include "dlpack/dlpack.h"

// Part of the Array API specification.
#define CUDNN_FRONTEND_DLPACK_CAPSULE_NAME "dltensor"
#define CUDNN_FRONTEND_DLPACK_USED_CAPSULE_NAME "used_dltensor"

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend {

namespace python_bindings {

// Raise C++ exceptions corresponding to C++ FE error codes.
// Pybinds will automatically convert C++ exceptions to pythpn exceptions.
void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const& error_msg) {
    if (cond == false) return;

    switch (error_code) {
        case cudnn_frontend::error_code_t::OK:
            return;
        case cudnn_frontend::error_code_t::ATTRIBUTE_NOT_SET:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::SHAPE_DEDUCTION_FAILED:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::INVALID_TENSOR_NAME:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::INVALID_VARIANT_PACK:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_EXECUTION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::HEURISTIC_QUERY_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::INVALID_CUDA_DEVICE:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::UNSUPPORTED_GRAPH_FORMAT:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_NOT_SUPPORTED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::HANDLE_ERROR:
            throw std::runtime_error(error_msg);
    }
}

char*
extract_data_pointer(py::object obj) {
    throw_if(!py::hasattr(obj, "__dlpack__"),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Object does not have the __dlpack__() method");

    py::capsule capsule = obj.attr("__dlpack__")();
    throw_if(capsule.is_none(),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Failed to retrieve the DLPack capsule.");

    DLManagedTensor* managed =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), CUDNN_FRONTEND_DLPACK_CAPSULE_NAME));
    throw_if(managed == nullptr, cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Invalid DLPack capsule.");

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    throw_if(
        device_type != kDLCPU && device_type != kDLCUDAHost && device_type != kDLCUDA && device_type != kDLCUDAManaged,
        cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
        "Invalid device type.");

    return (char*)managed->dl_tensor.data + managed->dl_tensor.byte_offset;
}

// This class is only meant direct pythonic API calls to c++ Graph class.
class PyGraph {
   public:
    template <cudnn_frontend::PointwiseMode_t MODE>
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    pointwise_ternary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& b,
                      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& c,
                      cudnn_frontend::DataType_t const& compute_data_type,
                      std::string const& name) {
        auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                              .set_mode(MODE)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);
        return graph.pointwise(a, b, c, attributes);
    }

    template <cudnn_frontend::PointwiseMode_t MODE>
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    pointwise_binary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& b,
                     cudnn_frontend::DataType_t const& compute_data_type,
                     std::string const& name) {
        auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                              .set_mode(MODE)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);
        return graph.pointwise(a, b, attributes);
    }

    template <cudnn_frontend::PointwiseMode_t MODE>
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    pointwise_unary(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& a,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
        auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                              .set_mode(MODE)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);
        return graph.pointwise(a, attributes);
    }

    // This Graph class is the sole structure which implicitly makes PyGraph own all tensors, nodes, and cudnn
    // descriptors.
    cudnn_frontend::graph::Graph graph;
    cudnnHandle_t handle;
    bool is_handle_owner;
    bool is_built;

    PyGraph(std::string const&,
            cudnn_frontend::DataType_t io_data_type,
            cudnn_frontend::DataType_t intermediate_data_type,
            cudnn_frontend::DataType_t compute_data_type,
            void * handle_ = nullptr)
        : graph(), handle((cudnnHandle_t)handle_), 
        is_handle_owner(false), is_built(false) {
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

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    tensor(std::vector<int64_t> const& dim,
           std::vector<int64_t> const& stride,
           cudnn_frontend::DataType_t const& data_type,
           bool const& is_virtual,
           bool const& is_pass_by_value,
           std::string const& name) {
        auto props = cudnn_frontend::graph::Tensor_attributes()
                         .set_data_type(data_type)
                         .set_is_virtual(is_virtual)
                         .set_is_pass_by_value(is_pass_by_value)
                         .set_dim(dim)
                         .set_stride(stride)
                         .set_name(name);

        return graph.tensor(props);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes all tensor properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    batchnorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& in_running_mean,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& in_running_var,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& momentum,
              std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& peer_stats,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name) {
        auto attributes = cudnn_frontend::graph::Batchnorm_attributes()
                              .set_forward_phase(forward_phase)
                              .set_compute_data_type(compute_data_type)
                              .set_epsilon(epsilon)
                              .set_previous_running_stats(in_running_mean, in_running_var, momentum)
                              .set_peer_stats(peer_stats)
                              .set_name(name);

        auto [Y, mean, inv_var, next_running_mean, next_running_var] = graph.batchnorm(x, scale, bias, attributes);
        return {Y, mean, inv_var, next_running_mean, next_running_var};
    }

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    layernorm(cudnn_frontend::NormFwdPhase_t const forward_phase,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
              std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name) {
        auto attributes = cudnn_frontend::graph::Layernorm_attributes()
                              .set_forward_phase(forward_phase)
                              .set_compute_data_type(compute_data_type)
                              .set_epsilon(epsilon)
                              .set_name(name);

        auto [Y, mean, inv_var] = graph.layernorm(x, scale, bias, attributes);
        return {Y, mean, inv_var};
    }

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    batchnorm_inference(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& x,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& mean,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& inv_variance,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                        cudnn_frontend::DataType_t const& compute_data_type,
                        std::string const& name) {
        auto attributes = cudnn_frontend::graph::Batchnorm_inference_attributes()
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);

        return graph.batchnorm_inference(x, mean, inv_variance, scale, bias, attributes);
    }

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    layernorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& mean,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                       cudnn_frontend::DataType_t const& compute_data_type,
                       std::string const& name) {
        auto attributes = cudnn_frontend::graph::Layernorm_backward_attributes()
                              .set_saved_mean_and_inv_variance(mean, inv_variance)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);

        auto [DX, DScale, DBias] = graph.layernorm_backward(dy, x, scale, attributes);
        return {DX, DScale, DBias};
    }

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    batchnorm_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& dy,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& x,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& scale,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& mean,
                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& inv_variance,
                       std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& peer_stats,
                       cudnn_frontend::DataType_t const& compute_data_type,
                       std::string const& name) {
        auto attributes = cudnn_frontend::graph::Batchnorm_backward_attributes()
                              .set_saved_mean_and_inv_variance(mean, inv_variance)
                              .set_peer_stats(peer_stats)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);

        auto [DX, DScale, DBias] = graph.batchnorm_backward(dy, x, scale, attributes);
        return {DX, DScale, DBias};
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and weight properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_fprop(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
        auto attributes = cudnn_frontend::graph::Conv_fprop_attributes()
                              .set_padding(padding)
                              .set_stride(stride)
                              .set_dilation(dilation)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);

        auto Y = graph.conv_fprop(image, weight, attributes);
        return Y;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and loss properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_dgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& filter,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
        auto attributes = cudnn_frontend::graph::Conv_dgrad_attributes()
                              .set_padding(padding)
                              .set_stride(stride)
                              .set_dilation(dilation)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);
        auto DX = graph.conv_dgrad(loss, filter, attributes);
        return DX;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and loss properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv_wgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
        auto attributes = cudnn_frontend::graph::Conv_wgrad_attributes()
                              .set_padding(padding)
                              .set_stride(stride)
                              .set_dilation(dilation)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);
        auto DW = graph.conv_wgrad(loss, image, attributes);
        return DW;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and weight properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    matmul(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& A,
           std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& B,
           cudnn_frontend::DataType_t const& compute_data_type,
           std::string const& name) {
        auto attributes = cudnn_frontend::graph::Matmul_attributes().set_compute_data_type(compute_data_type).set_name(name);

        auto C = graph.matmul(A, B, attributes);
        return C;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    relu(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
         float const negative_slope,
         cudnn_frontend::DataType_t const& compute_data_type,
         std::string const& name) {
        auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                              .set_compute_data_type(compute_data_type)
                              .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                              .set_relu_lower_clip_slope(negative_slope)
                              .set_name(name);

        auto OUT_0 = graph.pointwise(input, attributes);
        return OUT_0;
    }

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    gen_index(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
              int64_t const axis,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name) {
        auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                              .set_compute_data_type(compute_data_type)
                              .set_mode(cudnn_frontend::PointwiseMode_t::GEN_INDEX)
                              .set_axis(axis)
                              .set_name(name);

        auto OUT_0 = graph.pointwise(input, attributes);
        return OUT_0;
    }

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    relu_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                  float const negative_slope,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name) {
        auto attributes = cudnn_frontend::graph::Pointwise_attributes()
                              .set_compute_data_type(compute_data_type)
                              .set_mode(cudnn_frontend::PointwiseMode_t::RELU_BWD)
                              .set_relu_lower_clip_slope(negative_slope)
                              .set_name(name);

        auto OUT_0 = graph.pointwise(loss, input, attributes);
        return OUT_0;
    }

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    leaky_relu_backward(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                        float const negative_slope,
                        cudnn_frontend::DataType_t const& compute_data_type,
                        std::string const& name) {
        return relu_backward(loss, input, negative_slope, compute_data_type, name);
    }

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    leaky_relu(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
               float const negative_slope,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
        return relu(input, negative_slope, compute_data_type, name);
    }

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2UL>
    genstats(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
             cudnn_frontend::DataType_t const& compute_data_type,
             std::string const& name) {
        auto attributes =
            cudnn_frontend::graph::Genstats_attributes().set_compute_data_type(compute_data_type).set_name(name);

        auto [SUM, SQ_SUM] = graph.genstats(input, attributes);
        return {SUM, SQ_SUM};
    }

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    reduction(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
              cudnn_frontend::ReductionMode_t const mode,
              cudnn_frontend::DataType_t const& compute_data_type,
              std::string const& name) {
        auto attributes = cudnn_frontend::graph::Reduction_attributes()
                              .set_mode(mode)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);

        auto OUT_0 = graph.reduction(input, attributes);
        return OUT_0;
    }

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2>
    scaled_dot_product_flash_attention(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q,
                                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k,
                                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v,
                                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q,
                                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_kv,
                                       bool const is_inference,
                                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& attn_scale,
                                       std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias,
                                       bool const use_padding_mask,
                                       bool const use_alibi_mask,
                                       bool const use_causal_mask,
                                       py::object const& dropout,
                                       cudnn_frontend::DataType_t const& compute_data_type,
                                       std::string const& name) {
        auto attributes = cudnn_frontend::graph::Scaled_dot_product_flash_attention_attributes()
                              .set_is_inference(is_inference)
                              .set_seq_len_q(seq_len_q)
                              .set_seq_len_kv(seq_len_kv)
                              .set_attn_scale(attn_scale)
                              .set_bias(bias)
                              .set_padding_mask(use_padding_mask)
                              .set_alibi_mask(use_alibi_mask)
                              .set_causal_mask(use_causal_mask)
                              .set_compute_data_type(compute_data_type)
                              .set_name(name);

        if (!dropout.is_none()) {
            py::tuple dropout_tuple = dropout.cast<py::tuple>();
            if ((!dropout_tuple) || (dropout_tuple.size() != 3 && dropout_tuple.size() != 2)) {
                throw std::runtime_error(
                    "dropout must be a tuple of (float probability, a seed tensor, and an offset tensor) or (mask "
                    "tensor, scale tensor)");
            }
            if (py::isinstance<py::float_>(dropout_tuple[0])) {
                auto const probability = dropout_tuple[0].cast<float>();
                auto const seed = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if (!seed) {
                    throw std::runtime_error("dropout seed must be a cudnn_tensor.");
                }

                auto const offset = dropout_tuple[2].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if (!offset) {
                    throw std::runtime_error("dropout offset must be a cudnn_tensor.");
                }

                attributes.set_dropout(probability, seed, offset);
            } else {
                auto const mask = dropout_tuple[0].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if (!mask) {
                    throw std::runtime_error("dropout mask must be a cudnn_tensor.");
                }

                auto const scale = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if (!scale) {
                    throw std::runtime_error("dropout scale must be a cudnn_tensor.");
                }

                attributes.set_dropout(mask, scale);
            }
        }

        auto [O, Stats] = graph.scaled_dot_product_flash_attention(q, k, v, attributes);
        return {O, Stats};
    }

    void
    check_support() {
        build();
    }

    void
    build() {
        if (is_built) {
            return;
        }

        is_built = true;

        auto status = graph.validate();
        throw_if(status.is_bad(), status.get_code(), status.get_message());

        status = graph.build_operation_graph(handle);
        throw_if(status.is_bad(), status.get_code(), status.get_message());

        auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A);

        status = plans.check_support(handle);
        if (status.is_bad()) {
            auto fallback_plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_FALLBACK);
            status              = fallback_plans.check_support(handle);
            throw_if(status.is_bad(), status.get_code(), status.get_message());
            status = graph.set_execution_plans(fallback_plans);
        } else {
            status = graph.set_execution_plans(plans);
        }
        return;
    }

    int64_t
    get_workspace_size() {
        return graph.get_workspace_size();
    }

    void
    execute(std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, py::object> var_pack,
            py::object workspace) {
        std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> var_pack_;
        for (auto const& [tensor, pyobject] : var_pack) {
            var_pack_.emplace(tensor, extract_data_pointer(pyobject));
        }

        void* workspace_ptr = extract_data_pointer(workspace);

        // TODO: Probably concatenate in a macro?
        auto status = graph.execute(handle, var_pack_, workspace_ptr);
        throw_if(status.is_bad(), status.get_code(), status.get_message());

        return;
    }
};

std::vector<int64_t>
default_vector(void) {
    return {};
}

void
init_pygraph_submodule(py::module_& m) {
    py::class_<PyGraph> pygraph_(m, "pygraph");
    pygraph_
        .def(py::init<std::string const&,
                      cudnn_frontend::DataType_t,
                      cudnn_frontend::DataType_t,
                      cudnn_frontend::DataType_t,
                      void *>(),
             py::arg_v("name", "test_graph"),
             py::arg_v("io_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("intermediate_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("handle", nullptr))
        .def("tensor",
             &PyGraph::tensor,
             py::arg{"dim"},
             py::arg{"stride"},
             py::arg_v("data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"is_virtual", false},
             py::arg_v{"is_pass_by_value", false},
             py::arg_v("name", ""),
             R"pbdoc(
                Create a tensor.

                Args:
                    dim (List[int]): The dimensions of the tensor.
                    stride (List[int]): The strides of the tensor.
                    data_type (cudnn.data_type): The data type of the tensor. Default is cudnn.data_type.NOT_SET.
                    is_virtual (bool): Flag indicating if the tensor is virtual. Default is False.
                    is_pass_by_value (bool): Flag indicating if the tensor is passed by value. Default is False.
                    name (Optional[str]): The name of the tensor.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def("batchnorm",
             &PyGraph::batchnorm,
             py::arg("norm_forward_phase"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg("in_running_mean"),
             py::arg("in_running_var"),
             py::arg("epsilon"),
             py::arg("momentum"),
             py::arg_v("peer_stats", std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>()),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("layernorm",
             &PyGraph::layernorm,
             py::arg("norm_forward_phase"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg("epsilon"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("batchnorm_inference",
             &PyGraph::batchnorm_inference,
             py::arg("input"),
             py::arg("mean"),
             py::arg("inv_variance"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("batchnorm_backward",
             &PyGraph::batchnorm_backward,
             py::arg("grad"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("mean"),
             py::arg("inv_variance"),
             py::arg_v("peer_stats", std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>()),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("layernorm_backward",
             &PyGraph::layernorm_backward,
             py::arg("grad"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("mean"),
             py::arg("inv_variance"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("genstats",
             &PyGraph::genstats,
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""))
        .def("conv_fprop",
             &PyGraph::conv_fprop,
             py::arg("image"),
             py::arg("weight"),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()},
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Perform convolution operation with the given inputs.

                Args:
                    image (cudnn_tensor): The image tensor.
                    weight (cudnn_tensor): The weight tensor.
                    padding (Optional[List[int]]): The padding values for the operation. Default is an empty list.
                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def("conv_wgrad",
             &PyGraph::conv_wgrad,
             py::arg("image"),
             py::arg("loss"),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()},
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Compute weight gradients using the given inputs and loss.

                Args:
                    image (cudnn_tensor): The image tensor.
                    loss (cudnn_tensor): The loss tensor.
                    padding (Optional[List[int]]): The padding values for the operation. Default is an empty list.
                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def("conv_dgrad",
             &PyGraph::conv_dgrad,
             py::arg("loss"),
             py::arg("filter"),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()},
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Compute filter gradients using the given inputs and loss.

                Args:
                    loss (cudnn_tensor): The loss tensor.
                    filter (cudnn_tensor): The filter tensor.
                    padding (Optional[List[int]]): The padding values for the operation. Default is an empty list.
                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[pycudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def("matmul",
             &PyGraph::matmul,
             py::arg("A"),
             py::arg("B"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Perform matrix multiplication of two tensors A and B.

                Args:
                    A (cudnn_tensor): The first tensor.
                    B (cudnn_tensor): The second matrix tensor.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The result of the matrix multiplication.
            )pbdoc")
        .def("reduction",
             &PyGraph::reduction,
             py::arg("input"),
             py::arg("mode"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Reduce an input tensor along certain dimensions. These dimensions to reduce on are inferred from output tensor shape.

                Args:
                    input (cudnn_tensor): The input tensor.
                    mode (cudnn.reduction_mode): The mode to use to reduce along a dimension.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The result of reduction operation.
            )pbdoc")
        .def("scaled_dot_product_flash_attention",
             &PyGraph::scaled_dot_product_flash_attention,
             py::arg("q"),
             py::arg("k"),
             py::arg("v"),
             py::arg_v("seq_len_q", nullptr),
             py::arg_v("seq_len_kv", nullptr),
             py::arg("is_inference"),
             py::arg_v("attn_scale", nullptr),
             py::arg_v("bias", nullptr),
             py::arg_v("use_padding_mask", false),
             py::arg_v("use_alibi_mask", false),
             py::arg_v("use_causal_mask", false),
             py::arg_v("dropout", py::none()),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""),
             R"pbdoc(
                Perform scaled dot-product flash attention.

                Args:
                    q (cudnn_tensor): The query data.
                    k (cudnn_tensor): The key data.
                    v (cudnn_tensor): The value data.
                    seq_len_q (Optional[cudnn_tensor]): The sequence length of the query.
                    seq_len_kv (Optional[cudnn_tensor]): The sequence length of the key.
                    is_inference (bool): Whether it is an inference step or training step.
                    attn_scale (Optional[cudnn_tensor]): The scale factor for attention. Default is None.
                    bias (Optional[cudnn_tensor]): The bias data for attention. Default is None.
                    use_padding_mask (Optional[bool]): Whether to use padding mask. Default is False.
                    use_alibi_mask (Optional[bool]): Whether to use alibi mask. Default is False.
                    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
                    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): The name of the operation.

                Returns:
                    cudnn_tensor: The result of scaled dot-product flash attention.
                    Optional[cudnn_tensor]: The softmax statistics in case the operation is in a training step.
            )pbdoc")
        .def("build", &PyGraph::build)
        .def("check_support", &PyGraph::check_support)
        .def("get_workspace_size", &PyGraph::get_workspace_size)
        .def("execute", &PyGraph::execute)
        .def("__repr__", [](PyGraph const& pygraph) {
            std::stringstream ss;
            json j = pygraph.graph;
            ss << j.dump(4);
            return ss.str();
        });

    // Pointwise ops
    pygraph_.def("add",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::ADD>,
                 py::arg("a"),
                 py::arg("b"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Adds two cudnn tensors.

            Args:
                a (cudnn_tensor): The first tensor.
                b (cudnn_tensor): The second tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of addition operation.
        )pbdoc");
    pygraph_.def("bias",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::ADD>,
                 py::arg("input"),
                 py::arg("bias"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Add bias to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                bias (cudnn_tensor): The bias tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of adding bias to the input.
        )pbdoc");
    pygraph_.def("mul",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MUL>,
                 py::arg("a"),
                 py::arg("b"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Computes elementwise multiplication of two cudnn tensors.

        Args:
            a (cudnn_tensor): The first tensor.
            b (cudnn_tensor): The second tensor.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of the elementwise multiplication operation.
            )pbdoc");
    pygraph_.def("scale",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MUL>,
                 py::arg("input"),
                 py::arg("scale"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Scale the input.

            Args:
                input (cudnn_tensor): The input tensor.
                scale (cudnn_tensor): The scale tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the scaling operation.
        )pbdoc");

    pygraph_.def("sqrt",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SQRT>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Square root of the input tensor is computed

        Args:
            input (cudnn_tensor): The input tensor.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: pointwise square root of the input tensor is computed
        )pbdoc");

    pygraph_.def("max",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MAX>,
                 py::arg("input0"),
                 py::arg("input1"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Max of the input tensors is computed

        Args:
            input (cudnn_tensor): The input tensor 0.
            input (cudnn_tensor): The input tensor 1.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: a pointwise maximum is taken between two tensors.
        )pbdoc");
    pygraph_.def("min",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MIN>,
                 py::arg("input0"),
                 py::arg("input1"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Max of the input tensors is computed

        Args:
            input (cudnn_tensor): The input tensor 0.
            input (cudnn_tensor): The input tensor 1.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: a pointwise minimum is taken between two tensors.
        )pbdoc");

    pygraph_.def("gen_index",
                 &PyGraph::gen_index,
                 py::arg("input"),
                 py::arg_v("axis", 0),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Generates pointwise index value of the input tensor is generated along a given axis.

        Args:
            input (cudnn_tensor): The input tensor.
            negative_slope (Optional[float]): The slope of the activation for negative inputs.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result tensor containing the indices
        )pbdoc");

    // forward activations
    pygraph_.def("relu",
                 &PyGraph::relu,
                 py::arg("input"),
                 py::arg_v("negative_slope", 0.0),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Apply the Rectified Linear Unit (ReLU) activation function to the input.

        Args:
            input (cudnn_tensor): The input tensor.
            negative_slope (Optional[float]): The slope of the activation for negative inputs.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of the ReLU activation.
        )pbdoc");
    pygraph_.def("leaky_relu",
                 &PyGraph::leaky_relu,
                 py::arg("input"),
                 py::arg("negative_slope"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Apply the Leaky Rectified Linear Unit (Leaky ReLU) activation function to the input.

        Args:
            input (cudnn_tensor): The input tensor.
            negative_slope (float): The slope of the activation for negative inputs.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of the Leaky ReLU activation.
        )pbdoc");
    pygraph_.def("tanh",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::TANH_FWD>,
                 py::arg("input0"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        tanh activation of the input tensors is computed

        Args:
            input (cudnn_tensor): The input tensor.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: Result of tanh activation
        )pbdoc");
    pygraph_.def("elu",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::ELU_FWD>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Exponential Linear Unit (ELU) activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the ELU activation.
        )pbdoc");
    pygraph_.def("gelu",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::GELU_FWD>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Gaussian Error Linear Unit (GELU) activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the GELU activation.
        )pbdoc");
    pygraph_.def("sigmoid",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SIGMOID_FWD>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the sigmoid activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the sigmoid activation.
        )pbdoc");
    pygraph_.def("swish",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SWISH_FWD>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Swish activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the Swish activation.
        )pbdoc");
    pygraph_.def("softplus",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SOFTPLUS_FWD>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Softplus activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the Softplus activation.
        )pbdoc");
    pygraph_.def("gelu_approx_tanh",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::GELU_APPROX_TANH_FWD>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Approximate GELU activation function to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the Approximate GELU activation.
        )pbdoc");
    // End of forward activations

    // Backward activations
    pygraph_.def("relu_backward",
                 &PyGraph::relu_backward,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg_v("negative_slope", 0.0),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on Rectified Linear Unit (ReLU) activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                negative_slope (Optional[float]): The slope of the activation for negative inputs.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of ReLU activation.
        )pbdoc");
    pygraph_.def("leaky_relu_backward",
                 &PyGraph::leaky_relu_backward,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg("negative_slope"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on Leaky Rectified Linear Unit (Leaky ReLU) activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                negative_slope (float): The slope of the activation for negative inputs.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of Leaky ReLU activation.
        )pbdoc");
    pygraph_.def("tanh_backward",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::TANH_BWD>,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on tanh activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of tanh activation.
        )pbdoc");
    pygraph_.def("sigmoid_backward",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::SIGMOID_BWD>,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on sigmoid activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of sigmoid activation.
        )pbdoc");
    pygraph_.def("elu_backward",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::ELU_BWD>,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on elu activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of elu activation.
        )pbdoc");
    pygraph_.def("gelu_backward",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::GELU_BWD>,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on gelu activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of gelu activation.
        )pbdoc");
    pygraph_.def("softplus_backward",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::SOFTPLUS_BWD>,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on softplus activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of softplus activation.
        )pbdoc");
    pygraph_.def("swish_backward",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::SWISH_BWD>,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on swish activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of swish activation.
        )pbdoc");
    pygraph_.def("gelu_approx_tanh_backward",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::GELU_APPROX_TANH_BWD>,
                 py::arg("loss"),
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply backpropagation on approximate gelu activation function.

            Args:
                loss (cudnn_tensor): The loss tensor.
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of backpropagation of approximate gelu activation.
        )pbdoc");
    // End of backward activation functions
    pygraph_.def("erf",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::ERF>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute erf of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of erf of input.
        )pbdoc");
    pygraph_.def("identity",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::IDENTITY>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Copy input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The copy of input.
        )pbdoc");

    pygraph_.def("exp",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::EXP>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute exponential of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of exponential of input.
        )pbdoc");
    pygraph_.def("log",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::LOG>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute natural logarithm of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of natural logarithm of input.
        )pbdoc");
    pygraph_.def("neg",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::NEG>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute numerical negative of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of numerical sign negation of input.
        )pbdoc");
    pygraph_.def("mod",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::MOD>,
                 py::arg("input0"),
                 py::arg("input1"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            In this mode, a pointwise floating-point remainder of the first tensor's division by the second tensor is computed.

            Args:
                input0 (cudnn_tensor): The input tensor.
                input1 (cudnn_tensor): The divisor tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of pointwise floating-point remainder of the input0 tensor's division by the input1 tensor
        )pbdoc");
    pygraph_.def("pow",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::POW>,
                 py::arg("input0"),
                 py::arg("input1"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            In this mode, a pointwise value from the first tensor to the power of the second tensor is computed.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of first tensor to the power of the second tensor.
        )pbdoc");
    pygraph_.def("abs",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::ABS>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Absolute value of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of absolute value of input.
        )pbdoc");
    pygraph_.def("ceil",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::CEIL>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            A pointwise ceiling of the input tensor is computed.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of ceil of input.
        )pbdoc");
    pygraph_.def("floor",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::FLOOR>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute floor of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of floor of input.
        )pbdoc");
    pygraph_.def("rsqrt",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::RSQRT>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute reciprocal square root of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of reciprocal square root of input.
        )pbdoc");
    pygraph_.def("reciprocal",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::RECIPROCAL>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute reciprocal input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of reciprocal of input.
        )pbdoc");
    pygraph_.def("sin",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::SIN>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute Sine of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of sine of input.
        )pbdoc");
    pygraph_.def("cos",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::COS>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute Cosine of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of cosine of input.
        )pbdoc");
    pygraph_.def("tan",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::TAN>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Compute Tangent of input tensor.

            Args:
                input (cudnn_tensor): The input tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of tangent of input.
        )pbdoc");
    pygraph_.def("logical_not",
                 &PyGraph::pointwise_unary<cudnn_frontend::PointwiseMode_t::LOGICAL_NOT>,
                 py::arg("input"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Compute logical_not of input tensor.

        Args:
            input (cudnn_tensor): The input tensor.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of logical_not of input.
    )pbdoc");
    pygraph_.def("logical_and",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::LOGICAL_AND>,
                 py::arg("a"),
                 py::arg("b"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Computes logical and of two tensors.

        Args:
            a (cudnn_tensor): The tensor to subtract from.
            b (cudnn_tensor): The tensor to subtract with.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of logical and between two tensors.
    )pbdoc");
    pygraph_.def("logical_or",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::LOGICAL_OR>,
                 py::arg("a"),
                 py::arg("b"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Computes logical or of two tensors.

        Args:
            a (cudnn_tensor): The tensor to subtract from.
            b (cudnn_tensor): The tensor to subtract with.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of logical or between two tensors.
    )pbdoc");

    pygraph_.def("sub",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::SUB>,
                 py::arg("a"),
                 py::arg("b"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
        Computes subtraction of two tensors.

        Args:
            a (cudnn_tensor): The tensor to subtract from.
            b (cudnn_tensor): The tensor to subtract with.
            compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
            name (Optional[str]): A name for the operation to be performed.

        Returns:
            cudnn_tensor: The result of subtration.
    )pbdoc");
    pygraph_.def("div",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::DIV>,
                 py::arg("a"),
                 py::arg("b"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Computes Division of two tensors.

            Args:
                a (cudnn_tensor): The tensor to subtract from.
                b (cudnn_tensor): The tensor to subtract with.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of Division.
        )pbdoc");
    pygraph_.def("add_square",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::ADD_SQUARE>,
                 py::arg("a"),
                 py::arg("b"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            a pointwise addition between the first tensor and the square of the second tensor is computed.

            Args:
                a (cudnn_tensor): The tensor to subtract from.
                b (cudnn_tensor): The tensor to subtract with.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of a pointwise addition between the first tensor and the square of the second tensor is computed.
        )pbdoc");

    pygraph_.def("cmp_eq",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_EQ>,
                 py::arg("input"),
                 py::arg("comparison"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Compare Equal to Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    pygraph_.def("cmp_neq",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_NEQ>,
                 py::arg("input"),
                 py::arg("comparison"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Compare Not equal to Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    pygraph_.def("cmp_gt",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_GT>,
                 py::arg("input"),
                 py::arg("comparison"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Compare Greater Than Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    pygraph_.def("cmp_ge",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_GE>,
                 py::arg("input"),
                 py::arg("comparison"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Compare Greater Than or Equal Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    pygraph_.def("cmp_lt",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_LT>,
                 py::arg("input"),
                 py::arg("comparison"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Compare Lesser Than Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    pygraph_.def("cmp_le",
                 &PyGraph::pointwise_binary<cudnn_frontend::PointwiseMode_t::CMP_LE>,
                 py::arg("input"),
                 py::arg("comparison"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Apply the Compare Lesser Than or Equal Comparison to the input.

            Args:
                input (cudnn_tensor): The input tensor.
                comparison (cudnn_tensor): The comparison tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
    pygraph_.def("binary_select",
                 &PyGraph::pointwise_ternary<cudnn_frontend::PointwiseMode_t::BINARY_SELECT>,
                 py::arg("input0"),
                 py::arg("input1"),
                 py::arg("mask"),
                 py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
                 py::arg_v("name", ""),
                 R"pbdoc(
            Selects between input0 or input1 based on the mask

            Args:
                input0 (cudnn_tensor): The input tensor0.
                input1 (cudnn_tensor): The input tensor1.
                mask (cudnn_tensor): The mask tensor.
                compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                name (Optional[str]): A name for the operation to be performed.

            Returns:
                cudnn_tensor: The result of the comparison.
        )pbdoc");
}

}

}