#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include "context.h"

#include "../cudnn_frontend_utils.h"

namespace cudnn_frontend {

namespace graph {

using managed_backend_descriptor_t = std::vector<ManagedOpaqueDescriptor>;

// simple structure to hold all properties of a tensor.
// Each property has a getter setter.
class Tensor_attributes {
   public:
    using uid_t = int64_t;

    // There are two usecases of pass by value tensors:
    // 1. Fused scalar constants
    // 2. Scalar passed during execution
    // In approach 1, users provide a value to embed into the graph.
    // In approach 2, users set is_pass_by_value boolean and then pass a pointer to scalar value with execute() API.
    // A closed set of types that are allowed to be passed by value.
    using pass_by_values_t = std::variant<int64_t, int32_t, half, float, nv_bfloat16>;

    error_t
    validate() const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            dim.empty(), error_code_t::ATTRIBUTE_NOT_SET, "Tensor '" + name + "' dims not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            stride.empty(), error_code_t::ATTRIBUTE_NOT_SET, "Tensor '" + name + "' strides not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(dim.size() != stride.size(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Tensor '" + name + "' does not equal dimensionality in dim and stride.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            is_virtual && is_pass_by_value,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Tensor '" + name + "' can't be both virutal and pass_by_value at the same time.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            pass_by_value.has_value() & (!is_pass_by_value),
            error_code_t::ATTRIBUTE_NOT_SET,
            "Tensor '" + name + "' can't be a fused scalar and not a pass_by_value tensor at the same time.");

        return {error_code_t::OK, ""};
    }

   private:
    template <typename>
    friend class Attributes;

    std::string name;
    DataType_t data_type        = DataType_t::NOT_SET;
    std::vector<int64_t> dim    = {};
    std::vector<int64_t> stride = {};
    bool is_virtual             = false;

    std::optional<pass_by_values_t> pass_by_value = std::nullopt;
    bool is_pass_by_value                         = false;

    TensorReordering_t reordering_type = TensorReordering_t::NONE;
    uid_t uid                          = 0;
    bool uid_assigned                  = false;

    std::shared_ptr<Tensor_attributes> ragged_offset;

    auto
    fill_from_context(detail::Context const& context) -> Tensor_attributes& {
        if (get_data_type() == DataType_t::NOT_SET) {
            if (get_is_virtual()) {
                set_data_type(context.get_intermediate_data_type());
            } else {
                set_data_type(context.get_io_data_type());
            }
        }
        return *this;
    }

   public:
    // Serialization functions
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    friend void
    to_json(nlohmann::json& j, const Tensor_attributes& ta);
    friend void
    from_json(const nlohmann::json& j, Tensor_attributes& ta);
#endif

    Tensor_attributes() = default;

    Tensor_attributes(float const& scalar) {
        pass_by_value    = scalar;
        is_pass_by_value = true;
        dim = stride = {1};
        data_type    = DataType_t::FLOAT;
    }

    Tensor_attributes(half const& scalar) {
        pass_by_value    = scalar;
        is_pass_by_value = true;
        dim = stride = {1};
        data_type    = DataType_t::HALF;
    }

    Tensor_attributes(nv_bfloat16 const& scalar) {
        pass_by_value    = scalar;
        is_pass_by_value = true;
        dim = stride = {1};
        data_type    = DataType_t::BFLOAT16;
    }

    Tensor_attributes(int32_t const& scalar) {
        pass_by_value    = scalar;
        is_pass_by_value = true;
        dim = stride = {1};
        data_type    = DataType_t::INT32;
    }

    Tensor_attributes(int64_t const& scalar) {
        pass_by_value    = scalar;
        is_pass_by_value = true;
        dim = stride = {1};
        data_type    = DataType_t::INT64;
    }

    std::string
    get_name() const {
        return name;
    }

    auto
    set_name(std::string const& value) -> Tensor_attributes& {
        name = value;
        return *this;
    }

    DataType_t
    get_data_type() const {
        return data_type;
    }

    auto
    set_data_type(DataType_t const value) -> Tensor_attributes& {
        data_type = value;
        return *this;
    }

    std::vector<int64_t>
    get_dim() const {
        return dim;
    }

    auto
    set_dim(std::vector<int64_t> const& value) -> Tensor_attributes& {
        dim = value;
        return *this;
    }

    int64_t
    get_volume() const {
        int64_t volume = 1ul;
        for (int64_t d : dim) {
            volume *= d;
        }
        return volume;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    auto
    set_stride(std::vector<int64_t> const& value) -> Tensor_attributes& {
        stride = value;
        return *this;
    }

    bool
    get_is_virtual() const {
        return is_virtual;
    }

    std::shared_ptr<Tensor_attributes>
    get_ragged_offset() {
        return ragged_offset;
    }

    auto
    set_is_virtual(bool const value) -> Tensor_attributes& {
        is_virtual = value;
        return *this;
    }

    auto
    set_output(bool const value) -> Tensor_attributes& {
        return set_is_virtual(!value);
    }

    std::optional<pass_by_values_t>
    get_pass_by_value() const {
        return pass_by_value;
    }

    bool
    get_is_pass_by_value() const {
        return is_pass_by_value;
    }

    auto
    set_is_pass_by_value(bool const value) -> Tensor_attributes& {
        is_pass_by_value = value;
        return *this;
    }

    TensorReordering_t
    get_reordering_type() const {
        return reordering_type;
    }

    auto
    set_reordering_type(TensorReordering_t const value) -> Tensor_attributes& {
        reordering_type = value;
        return *this;
    }

    uid_t
    get_uid() const {
        return uid;
    }

    uid_t
    has_uid() const {
        return uid_assigned;
    }

    auto
    clear_uid(void) -> Tensor_attributes& {
        uid          = 0;
        uid_assigned = false;
        return *this;
    }

    auto
    set_uid(uid_t value) -> Tensor_attributes& {
        uid          = value;
        uid_assigned = true;
        return *this;
    }

    auto
    set_ragged_offset(std::shared_ptr<Tensor_attributes> const& value) -> Tensor_attributes& {
        ragged_offset = value;
        return *this;
    }
};

class Batchnorm_attributes;
class Batchnorm_backward_attributes;

template <typename DerivedT>
class Attributes {
    DerivedT&
    self() {
        return *static_cast<DerivedT*>(this);
    }
    DerivedT const&
    self() const {
        return *static_cast<DerivedT const*>(this);
    }

   protected:
    std::vector<int64_t>
    get_non_virtual_uids() const {
        std::vector<int64_t> non_virtual_uids;
        auto derived = static_cast<DerivedT const*>(this);
        for (auto& [name, tensor] : derived->inputs) {
            (void)name;
            if (tensor && tensor->get_is_virtual() == false) {
                non_virtual_uids.push_back(tensor->get_uid());
                if (auto ragged_offset = tensor->get_ragged_offset()) {
                    non_virtual_uids.push_back(ragged_offset->get_uid());
                }
            }
        }
        for (auto& [name, tensor] : derived->outputs) {
            (void)name;
            if (tensor && tensor->get_is_virtual() == false) {
                non_virtual_uids.push_back(tensor->get_uid());
                if (auto ragged_offset = tensor->get_ragged_offset()) {
                    non_virtual_uids.push_back(ragged_offset->get_uid());
                }
            }
        }

        // Handle special case of BN where peer_stats is also an input
        if constexpr (std::is_same_v<DerivedT, Batchnorm_attributes> ||
                      std::is_same_v<DerivedT, Batchnorm_backward_attributes>) {
            for (auto& tensor : derived->peer_stats) {
                if (tensor && tensor->get_is_virtual() == false) {
                    non_virtual_uids.push_back(tensor->get_uid());
                    if (auto ragged_offset = tensor->get_ragged_offset()) {
                        non_virtual_uids.push_back(ragged_offset->get_uid());
                    }
                }
            }
        }

        return non_virtual_uids;
    }

   public:
    error_t
    fill_pass_by_value(std::unordered_map<Tensor_attributes::uid_t, Tensor_attributes::pass_by_values_t>&
                           tensor_to_pass_by_value) const {
        auto derived = static_cast<DerivedT const*>(this);
        for (auto& [name, tensor] : derived->inputs) {
            (void)name;
            if (tensor && tensor->get_pass_by_value().has_value()) {
                tensor_to_pass_by_value.emplace(tensor->get_uid(), tensor->get_pass_by_value().value());
            }
        }

        return {error_code_t::OK, ""};
    }

    void
    fill_from_context(detail::Context const& context) {
        auto derived = static_cast<DerivedT const*>(this);
        for (auto& [name, tensor] : derived->inputs) {
            (void)name;
            if (tensor) {
                tensor->fill_from_context(context);
            }
        }
        for (auto& [name, tensor] : derived->outputs) {
            (void)name;
            if (tensor) {
                tensor->fill_from_context(context);
            }
        }
        // Handle special case of BN where peer_stats is also an input
        if constexpr (std::is_same_v<DerivedT, Batchnorm_attributes> ||
                      std::is_same_v<DerivedT, Batchnorm_backward_attributes>) {
            for (auto& tensor : derived->peer_stats) {
                if (tensor) {
                    tensor->fill_from_context(context);
                }
            }
        }

        if (compute_data_type == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }

        // Handle shape and stride inferencing for fused scalars.
        // Pick number of dimensions from anyone of non-fused-scalar input/output tensors
        // In case, all tensors are fused scalars, just keep them 1D.
        int64_t number_of_dims = 1;
        for (auto [name, tensor] : derived->inputs) {
            (void)name;
            if (tensor && (tensor->get_pass_by_value().has_value() == false)) {
                number_of_dims = tensor->get_dim().size();
                break;
            }
        }

        // If number of dims is still 1, try to see if user set output dims.
        if (number_of_dims == 1) {
            for (auto [name, tensor] : derived->outputs) {
                (void)name;
                if (tensor && (tensor->get_pass_by_value().has_value() == false)) {
                    number_of_dims = tensor->get_dim().size();
                    break;
                }
            }
        }

        for (auto [name, tensor] : derived->inputs) {
            (void)name;
            if (tensor && tensor->get_pass_by_value().has_value()) {
                tensor->set_dim(std::vector<int64_t>(number_of_dims, 1));
                tensor->set_stride(std::vector<int64_t>(number_of_dims, 1));
            }
        }
    }

    std::string name;
    DataType_t compute_data_type = DataType_t::NOT_SET;

    DerivedT&
    set_name(std::string const& value) {
        name = value;
        return self();
    }

    DerivedT&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return self();
    }
};

class BN_finalize_attributes : public Attributes<BN_finalize_attributes> {
    friend class Attributes<BN_finalize_attributes>;
    friend class BatchNormFinalizeNode;
    friend class Graph;

   public:
    enum class input_names {
        SUM,
        SQ_SUM,
        SCALE,
        BIAS,
        EPSILON,
        ACCUM_COUNT,
        PREV_RUNNING_MEAN,
        PREV_RUNNING_VAR,
        MOMENTUM
    };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR };

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(BN_finalize_attributes, name, compute_data_type, inputs, outputs)
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    BN_finalize_attributes&
    set_previous_running_stats(std::shared_ptr<Tensor_attributes>& mean,
                               std::shared_ptr<Tensor_attributes>& variance,
                               std::shared_ptr<Tensor_attributes>& momentum) {
        inputs[BN_finalize_attributes::input_names::PREV_RUNNING_MEAN] = mean;
        inputs[BN_finalize_attributes::input_names::PREV_RUNNING_VAR]  = variance;
        inputs[BN_finalize_attributes::input_names::MOMENTUM]          = momentum;
        return *this;
    }
};

class Genstats_attributes : public Attributes<Genstats_attributes> {
    friend class Attributes<Genstats_attributes>;
    friend class GenstatsNode;
    friend class Graph;

   public:
    enum class input_names { X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { SUM, SQ_SUM };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Genstats_attributes, name, compute_data_type, inputs, outputs)
};

class Conv_fprop_attributes : public Attributes<Conv_fprop_attributes> {
    friend class Attributes<Conv_fprop_attributes>;
    friend class ConvolutionNode;
    friend class Graph;

    std::vector<int64_t> pre_padding;
    std::vector<int64_t> post_padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

    ConvolutionMode_t math_mode = ConvolutionMode_t::CROSS_CORRELATION;

   public:
    enum class input_names { X, W };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_fprop_attributes,
                                   name,
                                   compute_data_type,
                                   inputs,
                                   outputs,
                                   pre_padding,
                                   post_padding,
                                   stride,
                                   dilation,
                                   math_mode)

    ConvolutionMode_t
    get_convolution_mode() const {
        return math_mode;
    }

    std::vector<int64_t>
    get_pre_padding() const {
        return pre_padding;
    }

    std::vector<int64_t>
    get_post_padding() const {
        return post_padding;
    }

    Conv_fprop_attributes&
    set_padding(std::vector<int64_t> value) {
        pre_padding  = value;
        post_padding = value;
        return *this;
    }

    Conv_fprop_attributes&
    set_pre_padding(std::vector<int64_t> value) {
        pre_padding = value;
        return *this;
    }

    Conv_fprop_attributes&
    set_post_padding(std::vector<int64_t> value) {
        post_padding = value;
        return *this;
    }

    Conv_fprop_attributes&
    set_convolution_mode(ConvolutionMode_t mode_) {
        math_mode = mode_;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    Conv_fprop_attributes&
    set_stride(std::vector<int64_t> value) {
        stride = value;
        return *this;
    }

    std::vector<int64_t>
    get_dilation() const {
        return dilation;
    }

    Conv_fprop_attributes&
    set_dilation(std::vector<int64_t> value) {
        dilation = value;
        return *this;
    }
};

class Batchnorm_backward_attributes : public Attributes<Batchnorm_backward_attributes> {
    friend class Attributes<Batchnorm_backward_attributes>;
    friend class DBNNode;
    friend class Graph;

   public:
    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    // Only special case where one of the inputs is a vector.
    std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;
    enum class output_names { DX, DSCALE, DBIAS };
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_backward_attributes, name, compute_data_type, inputs, peer_stats, outputs)
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    Batchnorm_backward_attributes&
    set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                    std::shared_ptr<Tensor_attributes> inv_variance) {
        inputs[Batchnorm_backward_attributes::input_names::MEAN]         = mean;
        inputs[Batchnorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;
        return *this;
    }

    Batchnorm_backward_attributes&
    set_peer_stats(std::vector<std::shared_ptr<Tensor_attributes>> const& input_peer_stats) {
        peer_stats = input_peer_stats;
        return *this;
    }
};

class DBN_weight_attributes : public Attributes<DBN_weight_attributes> {
    friend class Attributes<DBN_weight_attributes>;
    friend class DBNWeightNode;
    friend class Graph;

   public:
    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DSCALE, DBIAS, EQ_BIAS, EQ_SCALE_DY, EQ_SCALE_X };
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DBN_weight_attributes, name, compute_data_type, inputs, outputs)
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
};

class Conv_dgrad_attributes : public Attributes<Conv_dgrad_attributes> {
    friend class Attributes<Conv_dgrad_attributes>;
    friend class DgradNode;
    friend class Graph;

    std::vector<int64_t> pre_padding;
    std::vector<int64_t> post_padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

    ConvolutionMode_t math_mode = ConvolutionMode_t::CROSS_CORRELATION;

   public:
    enum class input_names { DY, W };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DX };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_dgrad_attributes,
                                   name,
                                   compute_data_type,
                                   inputs,
                                   outputs,
                                   pre_padding,
                                   post_padding,
                                   stride,
                                   dilation,
                                   math_mode)

    ConvolutionMode_t
    get_convolution_mode() const {
        return math_mode;
    }

    std::vector<int64_t>
    get_pre_padding() const {
        return pre_padding;
    }

    std::vector<int64_t>
    get_post_padding() const {
        return post_padding;
    }

    Conv_dgrad_attributes&
    set_padding(std::vector<int64_t> value) {
        pre_padding  = value;
        post_padding = value;
        return *this;
    }

    Conv_dgrad_attributes&
    set_pre_padding(std::vector<int64_t> value) {
        pre_padding = value;
        return *this;
    }

    Conv_dgrad_attributes&
    set_post_padding(std::vector<int64_t> value) {
        post_padding = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    Conv_dgrad_attributes&
    set_convolution_mode(ConvolutionMode_t mode_) {
        math_mode = mode_;
        ;
        return *this;
    }

    Conv_dgrad_attributes&
    set_stride(std::vector<int64_t> value) {
        stride = value;
        return *this;
    }

    std::vector<int64_t>
    get_dilation() const {
        return dilation;
    }

    Conv_dgrad_attributes&
    set_dilation(std::vector<int64_t> value) {
        dilation = value;
        return *this;
    }
};

class Matmul_attributes : public Attributes<Matmul_attributes> {
    friend class Attributes<Matmul_attributes>;
    friend class MatmulNode;
    friend class INode;

    double padding_value = 0.0;

   public:
    enum class input_names { A, B, M_override, N_override, K_override };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { C };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Matmul_attributes, name, compute_data_type, inputs, outputs, padding_value)

    Matmul_attributes&
    set_m_override(std::shared_ptr<Tensor_attributes> const& value) {
        inputs[input_names::M_override] = value;
        return *this;
    }

    Matmul_attributes&
    set_n_override(std::shared_ptr<Tensor_attributes> const& value) {
        inputs[input_names::N_override] = value;
        return *this;
    }

    Matmul_attributes&
    set_k_override(std::shared_ptr<Tensor_attributes> const& value) {
        inputs[input_names::K_override] = value;
        return *this;
    }

    Matmul_attributes&
    set_padding(double const padding_val) {
        padding_value = padding_val;
        return *this;
    }
};

class Matmul_fp8_attributes : public Attributes<Matmul_fp8_attributes> {
    friend class Attributes<Matmul_fp8_attributes>;
    friend class MatmulFP8Node;
    friend class INode;

    double padding_value = 0.0;

   public:
    enum class input_names { Descale_A, Descale_B, A, B, M_override, N_override, K_override, Scale_C };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { C, Amax_C };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Matmul_fp8_attributes, name, compute_data_type, inputs, outputs)

    Matmul_fp8_attributes&
    set_m_override(std::shared_ptr<Tensor_attributes> const& value) {
        inputs[input_names::M_override] = value;
        return *this;
    }

    Matmul_fp8_attributes&
    set_n_override(std::shared_ptr<Tensor_attributes> const& value) {
        inputs[input_names::N_override] = value;
        return *this;
    }

    Matmul_fp8_attributes&
    set_k_override(std::shared_ptr<Tensor_attributes> const& value) {
        inputs[input_names::K_override] = value;
        return *this;
    }

    Matmul_fp8_attributes&
    set_padding(double const padding_val) {
        padding_value = padding_val;
        return *this;
    }
};

class Pointwise_attributes : public Attributes<Pointwise_attributes> {
    friend class Attributes<Pointwise_attributes>;
    friend class PointwiseNode;
    friend class SoftmaxNode;
    friend class INode;

    PointwiseMode_t mode = PointwiseMode_t::NOT_SET;

    std::optional<int64_t> axis;

    std::optional<float> relu_lower_clip;
    std::optional<float> relu_upper_clip;
    std::optional<float> relu_lower_clip_slope;

   public:
    enum class input_names { IN_0, IN_1, IN_2 };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { OUT_0 };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Pointwise_attributes,
                                   name,
                                   compute_data_type,
                                   inputs,
                                   outputs,
                                   mode,
                                   axis,
                                   relu_lower_clip,
                                   relu_upper_clip,
                                   relu_lower_clip_slope)

    Pointwise_attributes&
    set_mode(PointwiseMode_t const value) {
        mode = value;
        return *this;
    }

    std::optional<int64_t>
    get_axis() const {
        return axis;
    }

    Pointwise_attributes&
    set_axis(int64_t const axis) {
        this->axis = axis;
        return *this;
    }

    Pointwise_attributes&
    set_relu_lower_clip_slope(float const negative_slope) {
        this->relu_lower_clip_slope = negative_slope;
        return *this;
    }

    Pointwise_attributes&
    set_relu_lower_clip(float const value) {
        this->relu_lower_clip = value;
        return *this;
    }

    Pointwise_attributes&
    set_relu_upper_clip(float const value) {
        this->relu_upper_clip = value;
        return *this;
    }
};

class Instancenorm_backward_attributes : public Attributes<Instancenorm_backward_attributes> {
    friend class Attributes<Instancenorm_backward_attributes>;
    friend class DINNode;
    friend class Graph;

   public:
    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DX, DSCALE, DBIAS };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Instancenorm_backward_attributes, name, compute_data_type, inputs, outputs)

    Instancenorm_backward_attributes&
    set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                    std::shared_ptr<Tensor_attributes> inv_variance) {
        inputs[Instancenorm_backward_attributes::input_names::MEAN]         = mean;
        inputs[Instancenorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;
        return *this;
    }
};

class Layernorm_backward_attributes : public Attributes<Layernorm_backward_attributes> {
    friend class Attributes<Layernorm_backward_attributes>;
    friend class DLNNode;
    friend class Graph;

   public:
    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE, EPSILON };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DX, DSCALE, DBIAS };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Layernorm_backward_attributes, name, compute_data_type, inputs, outputs)

    Layernorm_backward_attributes&
    set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                    std::shared_ptr<Tensor_attributes> inv_variance) {
        inputs[Layernorm_backward_attributes::input_names::MEAN]         = mean;
        inputs[Layernorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;
        return *this;
    }
};

class Layernorm_attributes : public Attributes<Layernorm_attributes> {
    friend class Attributes<Layernorm_attributes>;
    friend class LayerNormNode;
    friend class Graph;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

   public:
    enum class input_names { X, SCALE, BIAS, EPSILON };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y, MEAN, INV_VARIANCE };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Layernorm_attributes, name, compute_data_type, inputs, outputs, forward_phase)

    Layernorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Layernorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs[Layernorm_attributes::input_names::EPSILON] = value;
        return *this;
    }
};

class Instancenorm_attributes : public Attributes<Instancenorm_attributes> {
    friend class Attributes<Instancenorm_attributes>;
    friend class InstanceNormNode;
    friend class Graph;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

   public:
    enum class input_names { X, SCALE, BIAS, EPSILON };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y, MEAN, INV_VARIANCE };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Instancenorm_attributes, name, compute_data_type, inputs, outputs, forward_phase)

    Instancenorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Instancenorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs[Instancenorm_attributes::input_names::EPSILON] = value;
        return *this;
    }
};

class Batchnorm_attributes : public Attributes<Batchnorm_attributes> {
    friend class Attributes<Batchnorm_attributes>;
    friend class BatchNormNode;
    friend class Graph;

   public:
    enum class input_names { X, SCALE, BIAS, PREV_RUNNING_MEAN, PREV_RUNNING_VAR, EPSILON, MOMENTUM };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    // Only special case where one of the inputs is a vector.
    std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;
    enum class output_names { Y, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_attributes, name, compute_data_type, inputs, peer_stats, outputs)

    Batchnorm_attributes&
    set_previous_running_stats(std::shared_ptr<Tensor_attributes>& mean,
                               std::shared_ptr<Tensor_attributes>& variance,
                               std::shared_ptr<Tensor_attributes>& momentum) {
        inputs[input_names::PREV_RUNNING_MEAN] = mean;
        inputs[input_names::PREV_RUNNING_VAR]  = variance;
        inputs[input_names::MOMENTUM]          = momentum;
        return *this;
    }

    Batchnorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs[input_names::EPSILON] = value;
        return *this;
    }

    Batchnorm_attributes&
    set_peer_stats(std::vector<std::shared_ptr<Tensor_attributes>> const& input_peer_stats) {
        peer_stats = input_peer_stats;
        return *this;
    }
};

class Batchnorm_inference_attributes : public Attributes<Batchnorm_inference_attributes> {
    friend class Attributes<Batchnorm_inference_attributes>;
    friend class BatchnormInferenceNode;
    friend class Graph;

   public:
    enum class input_names { X, MEAN, INV_VARIANCE, SCALE, BIAS };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_inference_attributes, name, compute_data_type, inputs, outputs)
};

class Reduction_attributes : public Attributes<Reduction_attributes> {
    friend class Attributes<Reduction_attributes>;
    friend class ReductionNode;
    friend class INode;

    std::optional<ReductionMode_t> mode;

   public:
    enum class input_names { X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Reduction_attributes, name, compute_data_type, inputs, outputs, mode)

    std::optional<ReductionMode_t>
    get_mode() const {
        return mode;
    }

    Reduction_attributes&
    set_mode(ReductionMode_t value) {
        mode = value;
        return *this;
    }
};

class Rng_attributes : public Attributes<Rng_attributes> {
    friend class Attributes<Rng_attributes>;
    friend class RngNode;
    friend class INode;

    RngDistribution_t distribution = RngDistribution_t::NOT_SET;
    std::vector<int64_t> dim       = {};
    std::vector<int64_t> stride    = {};
    std::optional<int64_t> seed;
    std::optional<double> bernoulli_probability;

   public:
    enum class input_names { Seed, Offset };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rng_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   distribution,
                                   dim,
                                   stride,
                                   seed,
                                   bernoulli_probability)

    std::vector<int64_t>
    get_dim() const {
        return dim;
    }

    auto
    set_dim(std::vector<int64_t> const& value) -> Rng_attributes& {
        dim = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    auto
    set_stride(std::vector<int64_t> const& value) -> Rng_attributes& {
        stride = value;
        return *this;
    }

    RngDistribution_t
    get_distribution() const {
        return distribution;
    }

    Rng_attributes&
    set_distribution(RngDistribution_t value) {
        distribution = value;
        return *this;
    }

    std::optional<int64_t>
    get_seed() const {
        return seed;
    }

    Rng_attributes&
    set_seed(std::optional<int64_t> value) {
        seed = value;
        return *this;
    }

    std::optional<double>
    get_bernoulli_probability() const {
        return bernoulli_probability;
    }

    Rng_attributes&
    set_bernoulli_probability(std::optional<double> value) {
        bernoulli_probability = value;
        return *this;
    }
};

class Resample_attributes : public Attributes<Resample_attributes> {
    friend class Attributes<Resample_attributes>;
    friend class ResampleNode;
    friend class INode;

    std::optional<bool> is_inference;
    ResampleMode_t resample_mode;
    PaddingMode_t padding_mode;
    std::vector<cudnnFraction_t> pre_padding;
    std::vector<cudnnFraction_t> post_padding;
    std::vector<cudnnFraction_t> stride;
    std::vector<cudnnFraction_t> window;

   public:
    enum class input_names { X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { Y, Index };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Resample_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   resample_mode,
                                   padding_mode,
                                   pre_padding,
                                   post_padding,
                                   stride,
                                   window)

    auto
    set_resampling_mode(ResampleMode_t const& value) -> Resample_attributes& {
        resample_mode = value;
        return *this;
    }

    auto
    set_padding_mode(PaddingMode_t const& value) -> Resample_attributes& {
        padding_mode = value;
        return *this;
    }

    auto
    set_window(std::vector<int64_t> const& value) -> Resample_attributes& {
        window.resize(value.size());
        for (auto i = 0u; i < value.size(); i++) {
            window[i].numerator   = value[i];
            window[i].denominator = 1;
        }
        return *this;
    }

    auto
    set_window(std::vector<cudnnFraction_t> const& value) -> Resample_attributes& {
        window = value;
        return *this;
    }

    auto
    set_stride(std::vector<int64_t> const& value) -> Resample_attributes& {
        stride.resize(value.size());
        for (auto i = 0u; i < value.size(); i++) {
            stride[i].numerator   = value[i];
            stride[i].denominator = 1;
        }
        return *this;
    }

    auto
    set_stride(std::vector<cudnnFraction_t> const& value) -> Resample_attributes& {
        stride = value;
        return *this;
    }

    auto
    set_pre_padding(std::vector<int64_t> const& value) -> Resample_attributes& {
        pre_padding.resize(value.size());
        for (auto i = 0u; i < value.size(); i++) {
            pre_padding[i].numerator   = value[i];
            pre_padding[i].denominator = 1;
        }
        return *this;
    }

    auto
    set_pre_padding(std::vector<cudnnFraction_t> const& value) -> Resample_attributes& {
        pre_padding = value;
        return *this;
    }

    auto
    set_post_padding(std::vector<int64_t> const& value) -> Resample_attributes& {
        post_padding.resize(value.size());
        for (auto i = 0u; i < value.size(); i++) {
            post_padding[i].numerator   = value[i];
            post_padding[i].denominator = 1;
        }
        return *this;
    }

    auto
    set_post_padding(std::vector<cudnnFraction_t> const& value) -> Resample_attributes& {
        post_padding = value;
        return *this;
    }

    auto
    set_is_inference(bool const value) -> Resample_attributes& {
        is_inference = value;
        return *this;
    }
};

class Reshape_attributes : public Attributes<Reshape_attributes> {
    friend class Attributes<Reshape_attributes>;
    friend class ReshapeNode;
    friend class INode;

    std::vector<int64_t> dim    = {};
    std::vector<int64_t> stride = {};

   public:
    enum class input_names { X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Reshape_attributes, name, compute_data_type, inputs, outputs, dim, stride)

    std::vector<int64_t>
    get_dim() const {
        return dim;
    }

    auto
    set_dim(std::vector<int64_t> const& value) -> Reshape_attributes& {
        dim = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    auto
    set_stride(std::vector<int64_t> const& value) -> Reshape_attributes& {
        stride = value;
        return *this;
    }
};

class Rmsnorm_attributes : public Attributes<Rmsnorm_attributes> {
    friend class Attributes<Rmsnorm_attributes>;
    friend class RMSNormNode;
    friend class Graph;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

   public:
    enum class input_names { X, SCALE, BIAS, EPSILON };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y, INV_VARIANCE };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rmsnorm_attributes, name, compute_data_type, inputs, outputs, forward_phase)

    Rmsnorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Rmsnorm_attributes&
    set_bias(std::shared_ptr<Tensor_attributes>& value) {
        inputs[Rmsnorm_attributes::input_names::BIAS] = value;
        return *this;
    }

    Rmsnorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs[Rmsnorm_attributes::input_names::EPSILON] = value;
        return *this;
    }
};

class Rmsnorm_backward_attributes : public Attributes<Rmsnorm_backward_attributes> {
    friend class Attributes<Rmsnorm_backward_attributes>;
    friend class DRMSNormNode;
    friend class Graph;

    std::optional<bool> use_dbias;

   public:
    enum class input_names { DY, X, SCALE, INV_VARIANCE };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DX, DSCALE, DBIAS };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rmsnorm_backward_attributes, name, compute_data_type, inputs, outputs)

    Rmsnorm_backward_attributes&
    has_dbias(bool value) {
        use_dbias = value;
        return *this;
    }
};

// class Scaled_dot_product_attention_attributes : public Operation {
//    public:
//     struct Inputs {
//         std::shared_ptr<Tensor_attributes> Q;
//         std::shared_ptr<Tensor_attributes> K;
//         std::shared_ptr<Tensor_attributes> Attn_scale;
//         std::shared_ptr<Tensor_attributes> Bias;  // Optional bias after bmm1
//         std::shared_ptr<Tensor_attributes> V;
//         std::shared_ptr<Tensor_attributes> SEQ_LEN_Q;
//         std::shared_ptr<Tensor_attributes> SEQ_LEN_KV;
//         std::shared_ptr<Tensor_attributes> Mask;
//         std::shared_ptr<Tensor_attributes> Dropout_mask;
//         std::shared_ptr<Tensor_attributes> Dropout_scale;
//     } inputs;

//     struct Outputs {
//         std::shared_ptr<Tensor_attributes> O;
//         std::shared_ptr<Tensor_attributes>
//             S;  // softmax output dumped when is_inference false. Users first need to check whether its nullptr.
//     } outputs;

//     std::optional<bool> is_inference;
//     bool padding_mask = false;
//     bool causal_mask  = false;
//     std::optional<float> dropout_probability;
//     int64_t seed;
//     float dropout_scale = 1.f;

//    public:
//     Scaled_dot_product_attention_attributes() : Operation(Tag::Scaled_dot_product_attention), is_inference(false) {}

//     Scaled_dot_product_attention_attributes&
//     set_is_inference(bool const value) {
//         is_inference = value;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
//         inputs.SEQ_LEN_Q = value;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
//         inputs.SEQ_LEN_KV = value;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_padding_mask(bool const value) {
//         padding_mask = value;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_causal_mask(bool const value) {
//         causal_mask = value;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
//         inputs.Attn_scale = value;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_bias(std::shared_ptr<Tensor_attributes> bias) {
//         inputs.Bias = bias;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_dropout(float const probability, int64_t const seed_) {
//         dropout_probability = probability;
//         seed                = seed_;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
//         inputs.Dropout_mask  = mask;
//         inputs.Dropout_scale = scale;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_compute_data_type(DataType_t const value) {
//         compute_data_type = value;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     set_name(std::string const& value) {
//         name = value;
//         return *this;
//     }

//     Scaled_dot_product_attention_attributes&
//     fill_from_context(detail::Context const& context) {
//         // Fill node's tensors
//         inputs.Q->fill_from_context(context);
//         inputs.K->fill_from_context(context);
//         inputs.V->fill_from_context(context);
//         inputs.SEQ_LEN_Q->fill_from_context(context);
//         inputs.SEQ_LEN_KV->fill_from_context(context);
//         outputs.O->fill_from_context(context);

//         // Fill this node
//         if (get_compute_data_type() == DataType_t::NOT_SET) {
//             set_compute_data_type(context.get_compute_data_type());
//         }
//         return *this;
//     }
// };

class SDPA_attributes : public Attributes<SDPA_attributes> {
    friend class Attributes<SDPA_attributes>;
    friend class SDPANode;
    friend class Graph;

    std::optional<bool> is_inference;
    bool alibi_mask               = false;
    bool padding_mask             = false;
    bool causal_mask              = false;
    bool causal_mask_bottom_right = false;
    std::optional<int> sliding_window_length;
    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;
    std::optional<int> max_seq_len_kv;

   public:
    enum class input_names {
        Q,
        K,
        V,
        Attn_scale,
        Bias,
        SEQ_LEN_Q,
        SEQ_LEN_KV,
        Seed,
        Offset,
        Dropout_mask,
        Dropout_scale,
        Page_table_K,
        Page_table_V
    };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { O, Stats, RNG_DUMP };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDPA_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   is_inference,
                                   alibi_mask,
                                   padding_mask,
                                   causal_mask,
                                   causal_mask_bottom_right,
                                   dropout_probability,
                                   attn_scale_value,
                                   sliding_window_length)

    SDPA_attributes&
    set_is_inference(bool const value) {
        is_inference = value;
        return *this;
    }

    SDPA_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::Attn_scale] = std::move(value);
        return *this;
    }

    SDPA_attributes&
    set_attn_scale(float const value) {
        attn_scale_value = value;
        return *this;
    }

    SDPA_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::Bias] = std::move(value);
        return *this;
    }

    SDPA_attributes&
    set_alibi_mask(bool const value) {
        alibi_mask = value;
        return *this;
    }

    SDPA_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    SDPA_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::SEQ_LEN_Q] = std::move(value);
        return *this;
    }

    SDPA_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::SEQ_LEN_KV] = std::move(value);
        return *this;
    }

    SDPA_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    SDPA_attributes&
    set_causal_mask_bottom_right(bool const value) {
        causal_mask_bottom_right = value;
        return *this;
    }

    SDPA_attributes&
    set_sliding_window_length(int const value) {
        sliding_window_length = value;
        return *this;
    }

    SDPA_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability                          = probability;
        inputs[SDPA_attributes::input_names::Seed]   = std::move(seed);
        inputs[SDPA_attributes::input_names::Offset] = std::move(offset);
        return *this;
    }

    SDPA_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
        inputs[SDPA_attributes::input_names::Dropout_mask]  = std::move(mask);
        inputs[SDPA_attributes::input_names::Dropout_scale] = std::move(scale);
        return *this;
    }

    // For debugging purposes only.
    SDPA_attributes&
    set_rng_dump(std::shared_ptr<Tensor_attributes> value) {
        outputs[SDPA_attributes::output_names::RNG_DUMP] = std::move(value);
        return *this;
    }

    SDPA_attributes&
    set_paged_attention_k_table(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::Page_table_K] = std::move(value);
        return *this;
    }

    SDPA_attributes&
    set_paged_attention_v_table(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::Page_table_V] = std::move(value);
        return *this;
    }

    SDPA_attributes&
    set_paged_attention_max_seq_len_kv(int const value) {
        max_seq_len_kv = value;
        return *this;
    }
};

class SDPA_fp8_attributes : public Attributes<SDPA_fp8_attributes> {
    friend class Attributes<SDPA_fp8_attributes>;
    friend class SDPAFP8Node;
    friend class Graph;

    std::optional<bool> is_inference;
    bool padding_mask = false;
    bool causal_mask  = false;
    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;

   public:
    enum class input_names {
        Q,
        K,
        V,
        Attn_scale,
        Bias,
        SEQ_LEN_Q,
        SEQ_LEN_KV,
        Seed,
        Offset,
        Dropout_mask,
        Dropout_scale,

        Descale_Q,
        Descale_K,
        Descale_V,
        Descale_S,
        Scale_S,
        Scale_O,
    };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { O, Stats, Amax_S, Amax_O };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDPA_fp8_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   is_inference,
                                   padding_mask,
                                   causal_mask,
                                   dropout_probability,
                                   attn_scale_value)

    SDPA_fp8_attributes&
    set_is_inference(bool const value) {
        is_inference = value;
        return *this;
    }

    SDPA_fp8_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_fp8_attributes::input_names::Attn_scale] = value;
        return *this;
    }

    SDPA_fp8_attributes&
    set_attn_scale(float const value) {
        attn_scale_value = value;
        return *this;
    }

    SDPA_fp8_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_fp8_attributes::input_names::Bias] = value;
        return *this;
    }

    SDPA_fp8_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    SDPA_fp8_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_fp8_attributes::input_names::SEQ_LEN_Q] = value;
        return *this;
    }

    SDPA_fp8_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_fp8_attributes::input_names::SEQ_LEN_KV] = value;
        return *this;
    }

    SDPA_fp8_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    SDPA_fp8_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability                              = probability;
        inputs[SDPA_fp8_attributes::input_names::Seed]   = seed;
        inputs[SDPA_fp8_attributes::input_names::Offset] = offset;
        return *this;
    }

    SDPA_fp8_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
        inputs[SDPA_fp8_attributes::input_names::Dropout_mask]  = mask;
        inputs[SDPA_fp8_attributes::input_names::Dropout_scale] = scale;
        return *this;
    }
};

class SDPA_backward_attributes : public Attributes<SDPA_backward_attributes> {
    friend class Attributes<SDPA_backward_attributes>;
    friend class SDPABackwardNode;
    friend class Graph;

    bool alibi_mask               = false;
    bool padding_mask             = false;
    bool causal_mask              = false;
    bool causal_mask_bottom_right = false;
    std::optional<int> sliding_window_length;

    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;

    std::optional<int64_t> max_total_seq_len_q;
    std::optional<int64_t> max_total_seq_len_kv;

    bool is_deterministic_algorithm = false;

   public:
    enum class input_names {
        Q,
        K,
        V,
        O,
        dO,
        Stats,
        Attn_scale,
        Bias,
        SEQ_LEN_Q,
        SEQ_LEN_KV,
        Seed,
        Offset,
        Dropout_mask,
        Dropout_scale,
        Dropout_scale_inv,
    };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { dQ, dK, dV, dBias, RNG_DUMP };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDPA_backward_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   alibi_mask,
                                   padding_mask,
                                   causal_mask,
                                   causal_mask_bottom_right,
                                   dropout_probability,
                                   attn_scale_value,
                                   sliding_window_length,
                                   is_deterministic_algorithm)

    SDPA_backward_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_backward_attributes::input_names::Attn_scale] = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_attn_scale(float const value) {
        attn_scale_value = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_backward_attributes::input_names::Bias] = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_dbias(std::shared_ptr<Tensor_attributes> value) {
        outputs[SDPA_backward_attributes::output_names::dBias] = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_alibi_mask(bool const value) {
        alibi_mask = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_backward_attributes::input_names::SEQ_LEN_Q] = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_backward_attributes::input_names::SEQ_LEN_KV] = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_max_total_seq_len_q(int64_t const value) {
        max_total_seq_len_q = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_max_total_seq_len_kv(int64_t const value) {
        max_total_seq_len_kv = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_causal_mask_bottom_right(bool const value) {
        causal_mask_bottom_right = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_sliding_window_length(int const value) {
        sliding_window_length = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability                                   = probability;
        inputs[SDPA_backward_attributes::input_names::Seed]   = seed;
        inputs[SDPA_backward_attributes::input_names::Offset] = offset;
        return *this;
    }

    SDPA_backward_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask,
                std::shared_ptr<Tensor_attributes> scale,
                std::shared_ptr<Tensor_attributes> scale_inv) {
        inputs[SDPA_backward_attributes::input_names::Dropout_mask]      = mask;
        inputs[SDPA_backward_attributes::input_names::Dropout_scale]     = scale;
        inputs[SDPA_backward_attributes::input_names::Dropout_scale_inv] = scale_inv;
        return *this;
    }

    // For debugging purposes only.
    SDPA_backward_attributes&
    set_rng_dump(std::shared_ptr<Tensor_attributes> value) {
        outputs[SDPA_backward_attributes::output_names::RNG_DUMP] = value;
        return *this;
    }

    SDPA_backward_attributes&
    set_deterministic_algorithm(bool const value) {
        is_deterministic_algorithm = value;
        return *this;
    }
};

class SDPA_fp8_backward_attributes : public Attributes<SDPA_fp8_backward_attributes> {
    friend class Attributes<SDPA_fp8_backward_attributes>;
    friend class SDPAFP8BackwardNode;
    friend class Graph;

    bool padding_mask = false;
    bool causal_mask  = false;

    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;

   public:
    enum class input_names {
        Q,
        K,
        V,
        O,
        dO,
        Stats,
        Attn_scale,
        Bias,
        SEQ_LEN_Q,
        SEQ_LEN_KV,
        Seed,
        Offset,
        Dropout_mask,
        Dropout_scale,
        Dropout_scale_inv,

        Descale_Q,
        Descale_K,
        Descale_V,
        Descale_O,
        Descale_dO,
        Descale_S,
        Descale_dP,
        Scale_dQ,
        Scale_dK,
        Scale_dV,
        Scale_S,
        Scale_dP,
    };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV, Amax_dP };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDPA_fp8_backward_attributes,
                                   name,
                                   compute_data_type,
                                   inputs,
                                   outputs,
                                   padding_mask,
                                   causal_mask,
                                   dropout_probability,
                                   attn_scale_value)

    SDPA_fp8_backward_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_fp8_backward_attributes::input_names::Attn_scale] = value;
        return *this;
    }

    SDPA_fp8_backward_attributes&
    set_attn_scale(float const value) {
        attn_scale_value = value;
        return *this;
    }

    SDPA_fp8_backward_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_fp8_backward_attributes::input_names::Bias] = value;
        return *this;
    }

    SDPA_fp8_backward_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    SDPA_fp8_backward_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_fp8_backward_attributes::input_names::SEQ_LEN_Q] = value;
        return *this;
    }

    SDPA_fp8_backward_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_fp8_backward_attributes::input_names::SEQ_LEN_KV] = value;
        return *this;
    }

    SDPA_fp8_backward_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    SDPA_fp8_backward_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability                                       = probability;
        inputs[SDPA_fp8_backward_attributes::input_names::Seed]   = seed;
        inputs[SDPA_fp8_backward_attributes::input_names::Offset] = offset;
        return *this;
    }

    SDPA_fp8_backward_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask,
                std::shared_ptr<Tensor_attributes> scale,
                std::shared_ptr<Tensor_attributes> scale_inv) {
        inputs[SDPA_fp8_backward_attributes::input_names::Dropout_mask]      = mask;
        inputs[SDPA_fp8_backward_attributes::input_names::Dropout_scale]     = scale;
        inputs[SDPA_fp8_backward_attributes::input_names::Dropout_scale_inv] = scale_inv;
        return *this;
    }
};

using Scaled_dot_product_flash_attention_attributes [[deprecated]]          = SDPA_attributes;
using Scaled_dot_product_flash_attention_backward_attributes [[deprecated]] = SDPA_backward_attributes;

class Softmax_attributes : public Attributes<Softmax_attributes> {
    friend class Attributes<Softmax_attributes>;
    friend class SoftmaxNode;
    friend class INode;

    std::optional<bool> use_stats;
    std::optional<bool> use_M_Zinv;

   public:
    enum class input_names { P };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { S, Stats, M, Zinv };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Softmax_attributes, name, compute_data_type, inputs, outputs, use_stats, use_M_Zinv)

    Softmax_attributes&
    has_stats(bool const value) {
        use_stats = value;
        return *this;
    }

    Softmax_attributes&
    has_M_Zinv(bool const value) {
        use_M_Zinv = value;
        return *this;
    }
};

class Conv_wgrad_attributes : public Attributes<Conv_wgrad_attributes> {
    friend class Attributes<Conv_wgrad_attributes>;
    friend class WgradNode;
    friend class Graph;

    std::vector<int64_t> pre_padding;
    std::vector<int64_t> post_padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    ConvolutionMode_t math_mode = ConvolutionMode_t::CROSS_CORRELATION;

   public:
    enum class input_names { DY, X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { DW };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_wgrad_attributes,
                                   name,
                                   compute_data_type,
                                   inputs,
                                   outputs,
                                   pre_padding,
                                   post_padding,
                                   stride,
                                   dilation,
                                   math_mode)

    ConvolutionMode_t
    get_convolution_mode() const {
        return math_mode;
    }

    std::vector<int64_t>
    get_pre_padding() const {
        return pre_padding;
    }

    std::vector<int64_t>
    get_post_padding() const {
        return post_padding;
    }

    Conv_wgrad_attributes&
    set_convolution_mode(ConvolutionMode_t mode_) {
        math_mode = mode_;
        ;
        return *this;
    }

    Conv_wgrad_attributes&
    set_padding(std::vector<int64_t> value) {
        pre_padding  = value;
        post_padding = value;
        return *this;
    }

    Conv_wgrad_attributes&
    set_pre_padding(std::vector<int64_t> value) {
        pre_padding = value;
        return *this;
    }

    Conv_wgrad_attributes&
    set_post_padding(std::vector<int64_t> value) {
        post_padding = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    Conv_wgrad_attributes&
    set_stride(std::vector<int64_t> value) {
        stride = value;
        return *this;
    }

    std::vector<int64_t>
    get_dilation() const {
        return dilation;
    }

    Conv_wgrad_attributes&
    set_dilation(std::vector<int64_t> value) {
        dilation = value;
        return *this;
    }
};

class Slice_attributes : public Attributes<Slice_attributes> {
    friend class Attributes<Slice_attributes>;
    friend class SliceNode;
    friend class INode;

    std::vector<std::pair<int64_t, int64_t>> slices;

   public:
    enum class input_names { X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Slice_attributes, name, compute_data_type, inputs, outputs, slices)

    Slice_attributes&
    set_slices(std::vector<std::pair<int64_t, int64_t>> const value) {
        slices = value;
        return *this;
    }

    int64_t
    get_offset() const {
        auto& input             = inputs.at(input_names::X);
        auto const input_stride = input->get_stride();

        int64_t offset = 0;

        // Get number of elements to skip
        for (size_t i = 0; i < slices.size(); ++i) {
            offset += slices[i].first * input_stride[i];
        }

        // multiply by element size to get offset in bytes
        offset *= detail::get_data_type_size(input->get_data_type());
        return offset;
    }
};

class PagedCacheLoad_attributes : public Attributes<PagedCacheLoad_attributes> {
    friend class Attributes<PagedCacheLoad_attributes>;
    friend class PagedCacheLoadNode;
    friend class INode;

   public:
    enum class input_names { container, seqLen, pageTable };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { yOut };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(PagedCacheLoad_attributes, name, compute_data_type, inputs, outputs)
};

}  // namespace graph

}  // namespace cudnn_frontend