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
// simple structure to hold all properties of a tensor.
// Each property has a getter setter.
class Tensor_attributes {
    template <typename>
    friend class Attributes;

    std::string name;
    DataType_t data_type               = DataType_t::NOT_SET;
    std::vector<int64_t> dim           = {};
    std::vector<int64_t> stride        = {};
    bool is_virtual                    = false;
    bool is_pass_by_value              = false;
    TensorReordering_t reordering_type = TensorReordering_t::NONE;
    int64_t uid                        = 0;
    bool uid_assigned                  = false;

    std::shared_ptr<Tensor_attributes> ragged_offset;

    error_t
    validate() const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            dim.empty(), error_code_t::ATTRIBUTE_NOT_SET, "Tensor '" + name + "' dims not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            stride.empty(), error_code_t::ATTRIBUTE_NOT_SET, "Tensor '" + name + "' strides not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(dim.size() != stride.size(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Tensor '" + name + "' does not equal dimensinoality in dim and stride.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            is_virtual && is_pass_by_value,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Tensor '" + name + "' can't be both virutal and pass_by_value at the same time.");

        return {error_code_t::OK, ""};
    }

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
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Tensor_attributes,
                                   name,
                                   data_type,
                                   dim,
                                   stride,
                                   is_virtual,
                                   is_pass_by_value,
                                   reordering_type,
                                   uid,
                                   uid_assigned)

    Tensor_attributes() = default;

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

    int64_t
    get_uid() const {
        return uid;
    }

    int64_t
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
    set_uid(int64_t value) -> Tensor_attributes& {
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
    }

   public:
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

    error_t
    validate_inputs() const {
        auto derived = static_cast<DerivedT const*>(this);
        for (auto const& [enum_name, tensor] : derived->inputs) {
            (void)enum_name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(tensor->validate());
            }
        }

        // Handle special case of BN where peer_stats is also an input
        if constexpr (std::is_same_v<DerivedT, Batchnorm_attributes> ||
                      std::is_same_v<DerivedT, Batchnorm_backward_attributes>) {
            for (auto const& tensor : derived->peer_stats) {
                if (tensor) {
                    CHECK_CUDNN_FRONTEND_ERROR(tensor->validate());
                }
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    validate_outputs() const {
        auto derived = static_cast<DerivedT const*>(this);
        for (auto const& [enum_name, tensor] : derived->outputs) {
            (void)enum_name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(tensor->validate());
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    get_prefilled_uids(std::unordered_set<int64_t>& pre_assigned_uids) const {
        auto derived = static_cast<DerivedT const*>(this);

        for (auto& [name, tensor] : derived->inputs) {
            (void)name;
            if (tensor && tensor->has_uid()) {
                pre_assigned_uids.insert(tensor->get_uid());
                if (auto ragged_offset = tensor->get_ragged_offset()) {
                    pre_assigned_uids.insert(ragged_offset->get_uid());
                }
            }
        }
        for (auto& [name, tensor] : derived->outputs) {
            (void)name;
            if (tensor && tensor->has_uid()) {
                pre_assigned_uids.insert(tensor->get_uid());
                if (auto ragged_offset = tensor->get_ragged_offset()) {
                    pre_assigned_uids.insert(ragged_offset->get_uid());
                }
            }
        }

        // Handle special case of BN where peer_stats is also an input
        if constexpr (std::is_same_v<DerivedT, Batchnorm_attributes> ||
                      std::is_same_v<DerivedT, Batchnorm_backward_attributes>) {
            for (auto& tensor : derived->peer_stats) {
                if (tensor && tensor->has_uid()) {
                    pre_assigned_uids.insert(tensor->get_uid());
                    if (auto ragged_offset = tensor->get_ragged_offset()) {
                        pre_assigned_uids.insert(ragged_offset->get_uid());
                    }
                }
            }
        }

        return {error_code_t::OK, ""};
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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR };

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(BN_finalize_attributes, name, inputs, outputs)
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { SUM, SQ_SUM };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Genstats_attributes, name, inputs, outputs)
};

class Conv_fprop_attributes : public Attributes<Conv_fprop_attributes> {
    friend class Attributes<Conv_fprop_attributes>;
    friend class ConvolutionNode;
    friend class Graph;

    std::vector<int64_t> pre_padding;
    std::vector<int64_t> post_padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

   public:
    enum class input_names { X, W };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_fprop_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   pre_padding,
                                   post_padding,
                                   stride,
                                   dilation)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    // Only special case where one of the inputs is a vector.
    std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;
    enum class output_names { DX, DSCALE, DBIAS };
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_backward_attributes, name, inputs, peer_stats, outputs)
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DSCALE, DBIAS, EQ_BIAS, EQ_SCALE_DY, EQ_SCALE_X };
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DBN_weight_attributes, name, inputs, outputs)
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
};

class Conv_dgrad_attributes : public Attributes<Conv_dgrad_attributes> {
    friend class Attributes<Conv_dgrad_attributes>;
    friend class DgradNode;
    friend class Graph;

    std::vector<int64_t> pre_padding;
    std::vector<int64_t> post_padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

   public:
    enum class input_names { DY, W };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DX };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_dgrad_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   pre_padding,
                                   post_padding,
                                   stride,
                                   dilation)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { C };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Matmul_attributes, name, inputs, outputs)

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

class Pointwise_attributes : public Attributes<Pointwise_attributes> {
    friend class Attributes<Pointwise_attributes>;
    friend class PointwiseNode;
    friend class SoftmaxNode;
    friend class INode;

    PointwiseMode_t mode = PointwiseMode_t::NOT_SET;
    std::optional<int64_t> axis;

    std::optional<float> relu_lower_clip_slope;

   public:
    enum class input_names { IN_0, IN_1, IN_2 };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { OUT_0 };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Pointwise_attributes, name, inputs, outputs, mode, axis)

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
};

class Instancenorm_backward_attributes : public Attributes<Instancenorm_backward_attributes> {
    friend class Attributes<Instancenorm_backward_attributes>;
    friend class DINNode;
    friend class Graph;

   public:
    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DX, DSCALE, DBIAS };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Instancenorm_backward_attributes, name, inputs, outputs)

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
    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DX, DSCALE, DBIAS };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Layernorm_backward_attributes, name, inputs, outputs)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y, MEAN, INV_VARIANCE };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Layernorm_attributes, name, inputs, outputs, forward_phase)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y, MEAN, INV_VARIANCE };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Instancenorm_attributes, name, inputs, outputs, forward_phase)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    // Only special case where one of the inputs is a vector.
    std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;
    enum class output_names { Y, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_attributes, name, inputs, peer_stats, outputs)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_inference_attributes, name, inputs, outputs)
};

class Reduction_attributes : public Attributes<Reduction_attributes> {
    friend class Attributes<Reduction_attributes>;
    friend class ReductionNode;
    friend class INode;

    std::optional<ReductionMode_t> mode;

   public:
    enum class input_names { X };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Reduction_attributes, name, inputs, outputs, mode)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
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

class Reshape_attributes : public Attributes<Reshape_attributes> {
    friend class Attributes<Reshape_attributes>;
    friend class ReshapeNode;
    friend class INode;

    std::vector<int64_t> dim    = {};
    std::vector<int64_t> stride = {};

   public:
    enum class input_names { X };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Reshape_attributes, name, inputs, outputs, dim, stride)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { Y, INV_VARIANCE };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rmsnorm_attributes, name, inputs, outputs, forward_phase)

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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { DX, DSCALE, DBIAS };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rmsnorm_backward_attributes, name, inputs, outputs)

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
    bool alibi_mask   = false;
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
        Dropout_scale
    };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { O, Stats, RNG_DUMP };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDPA_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   is_inference,
                                   alibi_mask,
                                   padding_mask,
                                   causal_mask,
                                   dropout_probability,
                                   attn_scale_value)

    SDPA_attributes&
    set_is_inference(bool const value) {
        is_inference = value;
        return *this;
    }

    SDPA_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::Attn_scale] = value;
        return *this;
    }

    SDPA_attributes&
    set_attn_scale(float const value) {
        attn_scale_value = value;
        return *this;
    }

    SDPA_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::Bias] = value;
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
        inputs[SDPA_attributes::input_names::SEQ_LEN_Q] = value;
        return *this;
    }

    SDPA_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_attributes::input_names::SEQ_LEN_KV] = value;
        return *this;
    }

    SDPA_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    SDPA_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability                          = probability;
        inputs[SDPA_attributes::input_names::Seed]   = seed;
        inputs[SDPA_attributes::input_names::Offset] = offset;
        return *this;
    }

    SDPA_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
        inputs[SDPA_attributes::input_names::Dropout_mask]  = mask;
        inputs[SDPA_attributes::input_names::Dropout_scale] = scale;
        return *this;
    }

    // For debugging purposes only.
    SDPA_attributes&
    set_rng_dump(std::shared_ptr<Tensor_attributes> value) {
        outputs[SDPA_attributes::output_names::RNG_DUMP] = value;
        return *this;
    }
};

class SDPA_backward_attributes : public Attributes<SDPA_backward_attributes> {
    friend class Attributes<SDPA_backward_attributes>;
    friend class SDPABackwardNode;
    friend class Graph;

    bool alibi_mask   = false;
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
        Dropout_scale_inv
    };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { dQ, dK, dV, dBias, RNG_DUMP };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDPA_backward_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   alibi_mask,
                                   padding_mask,
                                   causal_mask,
                                   dropout_probability,
                                   attn_scale_value)

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
    set_causal_mask(bool const value) {
        causal_mask = value;
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
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    enum class output_names { S, Stats, M, Zinv };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Softmax_attributes, name, inputs, outputs, use_stats, use_M_Zinv)

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

class SDPA_FP8_attributes : public Attributes<SDPA_FP8_attributes> {
    friend class Attributes<SDPA_FP8_attributes>;
    friend class SDPA_FP8_Node;
    friend class Graph;

    enum class input_names {
        Q,
        K,
        V,
        SEQ_LEN_Q,
        SEQ_LEN_KV,
        Attn_scale,
        Bias,
        Seed,
        Offset,
        Dropout_mask,
        Dropout_scale,
        descale_Q,
        descale_K,
        descale_V,
        scale_S,
        scale_O,
        ragged_offset_QKV,
        ragged_offset_O
    };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { O, Stats, M, Zinv, AMax_S, AMax_O };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    std::optional<bool> is_inference;
    bool padding_mask = false;
    bool causal_mask  = false;
    std::optional<float> dropout_probability;

   public:
    SDPA_FP8_attributes&
    set_is_inference(bool const value) {
        is_inference = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_FP8_attributes::input_names::Attn_scale] = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_FP8_attributes::input_names::Bias] = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_FP8_attributes::input_names::SEQ_LEN_Q] = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_FP8_attributes::input_names::SEQ_LEN_KV] = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_ragged_offset_qkv(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_FP8_attributes::input_names::ragged_offset_QKV] = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_ragged_offset_o(std::shared_ptr<Tensor_attributes> value) {
        inputs[SDPA_FP8_attributes::input_names::ragged_offset_O] = value;
        return *this;
    }

    SDPA_FP8_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability                              = probability;
        inputs[SDPA_FP8_attributes::input_names::Seed]   = seed;
        inputs[SDPA_FP8_attributes::input_names::Offset] = offset;
        return *this;
    }

    SDPA_FP8_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
        inputs[SDPA_FP8_attributes::input_names::Dropout_mask]  = mask;
        inputs[SDPA_FP8_attributes::input_names::Dropout_scale] = scale;
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

   public:
    enum class input_names { DY, X };
    std::map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { DW };
    std::map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_wgrad_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   pre_padding,
                                   post_padding,
                                   stride,
                                   dilation)

    std::vector<int64_t>
    get_pre_padding() const {
        return pre_padding;
    }

    std::vector<int64_t>
    get_post_padding() const {
        return post_padding;
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

}  // namespace graph

}  // namespace cudnn_frontend