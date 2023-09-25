#pragma once

#include <iostream>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include "cudnn_frontend_graph_helpers.h"

namespace cudnn_frontend {

namespace graph {
// simple structure to hold all properties of a tensor.
// Each property has a getter setter.
class Tensor_attributes {
   protected:
    std::string name;
    DataType_t data_type               = DataType_t::NOT_SET;
    std::vector<int64_t> dim           = {};
    std::vector<int64_t> stride        = {};
    bool is_virtual                    = false;
    bool is_pass_by_value              = false;
    TensorReordering_t reordering_type = TensorReordering_t::NONE;
    using uid_t                        = int64_t;
    uid_t uid;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Tensor_attributes,
                                   name,
                                   data_type,
                                   dim,
                                   stride,
                                   is_virtual,
                                   is_pass_by_value,
                                   reordering_type,
                                   uid)

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

    uid_t
    get_uid() const {
        return uid;
    }

    auto
    set_uid(uid_t value) -> Tensor_attributes& {
        uid = value;
        return *this;
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
};

class Operation {
   public:
    enum class Tag {
        BN,
        BN_inference,
        BN_finalize,
        Conv_fprop,
        Conv_dgrad,
        Conv_wgrad,
        DBN,
        DLN,
        DBN_weight,
        Genstats,
        LN,
        Matmul,
        Pointwise,
        Reduction,
        Rng,
        Reshape,
        Scaled_dot_product_attention,
        Scaled_dot_product_flash_attention,
        Scaled_dot_product_flash_attention_backward,
        Softmax,
    };
    Tag tag;

    std::string name;
    DataType_t compute_data_type = DataType_t::NOT_SET;

    Operation(Tag t) : tag(t) {}

    std::string const
    get_name() const {
        return name;
    }

    DataType_t
    get_compute_data_type() const {
        return compute_data_type;
    }

    virtual ~Operation() = default;
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    Operation::Tag,
    {
        {Operation::Tag::BN, "BN"},
        {Operation::Tag::BN_inference, "BN_inference"},
        {Operation::Tag::BN_finalize, "BN_finalize"},
        {Operation::Tag::Conv_fprop, "Conv_fprop"},
        {Operation::Tag::Conv_dgrad, "Conv_dgrad"},
        {Operation::Tag::Conv_wgrad, "Conv_wgrad"},
        {Operation::Tag::DBN, "DBN"},
        {Operation::Tag::DBN_weight, "DBN_weight"},
        {Operation::Tag::Genstats, "Genstats"},
        {Operation::Tag::Matmul, "Matmul"},
        {Operation::Tag::Pointwise, "Pointwise"},
        {Operation::Tag::Reduction, "Reduction"},
        {Operation::Tag::Rng, "Rng"},
        {Operation::Tag::Reshape, "Reshape"},
        {Operation::Tag::Scaled_dot_product_attention, "Scaled_dot_product_attention"},
        {Operation::Tag::Scaled_dot_product_flash_attention, "Scaled_dot_product_flash_attention"},
        {Operation::Tag::Scaled_dot_product_flash_attention_backward, "Scaled_dot_product_flash_attention_backward"},
        {Operation::Tag::Softmax, "Softmax"},
    })

class BN_finalize_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> SUM;
        std::shared_ptr<Tensor_attributes> SQ_SUM;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> BIAS;
        std::shared_ptr<Tensor_attributes> EPSILON;
        std::shared_ptr<Tensor_attributes> ACCUM_COUNT;
        std::shared_ptr<Tensor_attributes> PREV_RUNNING_MEAN;
        std::shared_ptr<Tensor_attributes> PREV_RUNNING_VAR;
        std::shared_ptr<Tensor_attributes> MOMENTUM;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> EQ_SCALE;
        std::shared_ptr<Tensor_attributes> EQ_BIAS;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
        std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN;
        std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs,
                                   SUM,
                                   SQ_SUM,
                                   SCALE,
                                   BIAS,
                                   EPSILON,
                                   ACCUM_COUNT,
                                   PREV_RUNNING_MEAN,
                                   PREV_RUNNING_VAR,
                                   MOMENTUM)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(BN_finalize_attributes, name, tag, inputs, outputs)

    BN_finalize_attributes() : Operation(Tag::BN_finalize) {}

    BN_finalize_attributes&
    set_previous_running_stats(std::shared_ptr<Tensor_attributes>& mean,
                               std::shared_ptr<Tensor_attributes>& variance,
                               std::shared_ptr<Tensor_attributes>& momentum) {
        inputs.PREV_RUNNING_MEAN = mean;
        inputs.PREV_RUNNING_VAR  = variance;
        inputs.MOMENTUM          = momentum;
        return *this;
    }

    BN_finalize_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    BN_finalize_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor_attributes>(std::string const&)> output_tensor) {
        outputs.EQ_SCALE          = output_tensor(name + "_EQ_SCALE_output");
        outputs.EQ_BIAS           = output_tensor(name + "_EQ_BIAS_output");
        outputs.MEAN              = output_tensor(name + "_MEAN_output");
        outputs.INV_VARIANCE      = output_tensor(name + "_INV_VARIANCE_output");
        outputs.NEXT_RUNNING_MEAN = output_tensor(name + "_NEXT_RUNNING_MEAN_output");
        outputs.NEXT_RUNNING_VAR  = output_tensor(name + "_NEXT_RUNNING_VAR_output");
    }

    auto
    fill_from_context(detail::Context const& context) -> BN_finalize_attributes& {
        // Fill node's tensors
        inputs.SUM->fill_from_context(context);
        inputs.SQ_SUM->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.BIAS->fill_from_context(context);
        inputs.PREV_RUNNING_MEAN->fill_from_context(context);
        inputs.PREV_RUNNING_VAR->fill_from_context(context);
        inputs.EPSILON->fill_from_context(context);
        inputs.MOMENTUM->fill_from_context(context);
        inputs.ACCUM_COUNT->fill_from_context(context);

        outputs.EQ_SCALE->fill_from_context(context);
        outputs.EQ_BIAS->fill_from_context(context);
        outputs.MEAN->fill_from_context(context);
        outputs.INV_VARIANCE->fill_from_context(context);
        outputs.NEXT_RUNNING_MEAN->fill_from_context(context);
        outputs.NEXT_RUNNING_VAR->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Genstats_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> SUM;
        std::shared_ptr<Tensor_attributes> SQ_SUM;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, SUM, SQ_SUM)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Genstats_attributes, name, tag, inputs, outputs)

    Genstats_attributes() : Operation(Tag::Genstats) {}

    Genstats_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Genstats_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Genstats_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        outputs.SUM->fill_from_context(context);
        outputs.SQ_SUM->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Conv_fprop_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> W;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
    } outputs;

    std::vector<int64_t> padding  = {};
    std::vector<int64_t> stride   = {};
    std::vector<int64_t> dilation = {};

    bool is_padding_set  = false;
    bool is_dilation_set = false;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X, W)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_fprop_attributes, name, tag, inputs, outputs, padding, stride, dilation)

    Conv_fprop_attributes() : Operation(Tag::Conv_fprop) {}

    std::vector<int64_t>
    get_padding() const {
        return padding;
    }

    Conv_fprop_attributes&
    set_padding(std::vector<int64_t> value) {
        padding        = value;
        is_padding_set = true;
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
        dilation        = value;
        is_dilation_set = true;
        return *this;
    }

    Conv_fprop_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Conv_fprop_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Conv_fprop_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.W->fill_from_context(context);
        outputs.Y->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Batchnorm_backward_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> DY;
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
        std::shared_ptr<Tensor_attributes> EPSILON;
        std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> DX;
        std::shared_ptr<Tensor_attributes> DSCALE;
        std::shared_ptr<Tensor_attributes> DBIAS;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, DY, X, SCALE, MEAN, INV_VARIANCE, EPSILON)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, DX, DSCALE, DBIAS)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_backward_attributes, name, tag, inputs, outputs)

    Batchnorm_backward_attributes() : Operation(Tag::DBN) {}

    Batchnorm_backward_attributes&
    set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                    std::shared_ptr<Tensor_attributes> inv_variance) {
        inputs.MEAN         = mean;
        inputs.INV_VARIANCE = inv_variance;
        return *this;
    }

    Batchnorm_backward_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes> epsilon) {
        inputs.EPSILON = epsilon;
        return *this;
    }

    Batchnorm_backward_attributes&
    set_peer_stats(std::vector<std::shared_ptr<Tensor_attributes>> const& peer_stats) {
        inputs.peer_stats = peer_stats;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor_attributes>(std::string const&)> output_tensor) {
        outputs.DX     = output_tensor(name + "_DX_output");
        outputs.DSCALE = output_tensor(name + "_DSCALE_output");
        outputs.DBIAS  = output_tensor(name + "_DBIAS_output");
    }

    Batchnorm_backward_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Batchnorm_backward_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Batchnorm_backward_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.DY->fill_from_context(context);
        inputs.MEAN->fill_from_context(context);
        inputs.INV_VARIANCE->fill_from_context(context);

        if (inputs.EPSILON) inputs.EPSILON->fill_from_context(context);

        outputs.DX->fill_from_context(context);
        outputs.DSCALE->fill_from_context(context);
        outputs.DBIAS->fill_from_context(context);

        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class DBN_weight_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> DY;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> DSCALE;
        std::shared_ptr<Tensor_attributes> DBIAS;
        std::shared_ptr<Tensor_attributes> EQ_SCALE_DY;
        std::shared_ptr<Tensor_attributes> EQ_SCALE_X;
        std::shared_ptr<Tensor_attributes> EQ_BIAS;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X, MEAN, INV_VARIANCE, SCALE, DY)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, DSCALE, DBIAS, EQ_SCALE_DY, EQ_SCALE_X, EQ_BIAS)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DBN_weight_attributes, name, tag, inputs, outputs)

    DBN_weight_attributes() : Operation(Tag::DBN_weight) {}

    DBN_weight_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    DBN_weight_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor_attributes>(std::string const&)> output_tensor) {
        outputs.DSCALE      = output_tensor(name + "_dscale_output");
        outputs.DBIAS       = output_tensor(name + "_dbias_output");
        outputs.EQ_SCALE_DY = output_tensor(name + "_eq_scale_dy_output");
        outputs.EQ_SCALE_X  = output_tensor(name + "_eq_scale_x_output");
        outputs.EQ_BIAS     = output_tensor(name + "_eq_bias_output");
    }

    auto
    fill_from_context(detail::Context const& context) -> DBN_weight_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.MEAN->fill_from_context(context);
        inputs.INV_VARIANCE->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.DY->fill_from_context(context);
        outputs.DSCALE->fill_from_context(context);
        outputs.DBIAS->fill_from_context(context);
        outputs.EQ_SCALE_DY->fill_from_context(context);
        outputs.EQ_SCALE_X->fill_from_context(context);
        outputs.EQ_BIAS->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Conv_dgrad_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> DY;
        std::shared_ptr<Tensor_attributes> W;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> DX;
    } outputs;

    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, DY, W)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, DX)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_dgrad_attributes, name, tag, inputs, outputs, padding, stride, dilation)

    Conv_dgrad_attributes() : Operation(Tag::Conv_dgrad) {}

    std::vector<int64_t>
    get_padding() const {
        return padding;
    }

    Conv_dgrad_attributes&
    set_padding(std::vector<int64_t> value) {
        padding = value;
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

    Conv_dgrad_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Conv_dgrad_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Conv_dgrad_attributes& {
        // Fill node's tensors
        inputs.DY->fill_from_context(context);
        inputs.W->fill_from_context(context);
        outputs.DX->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Matmul_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> A;
        std::shared_ptr<Tensor_attributes> B;
        std::shared_ptr<Tensor_attributes> M_override;
        std::shared_ptr<Tensor_attributes> N_override;
        std::shared_ptr<Tensor_attributes> K_override;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> C;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, A, B, M_override, N_override, K_override)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, C)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Matmul_attributes, name, tag, inputs, outputs)

    Matmul_attributes() : Operation(Tag::Matmul) {}

    Matmul_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Matmul_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Matmul_attributes& {
        // Fill node's tensors
        inputs.A->fill_from_context(context);
        inputs.B->fill_from_context(context);
        outputs.C->fill_from_context(context);

        if (inputs.M_override) inputs.M_override->fill_from_context(context);
        if (inputs.N_override) inputs.N_override->fill_from_context(context);
        if (inputs.K_override) inputs.K_override->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Pointwise_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> IN_0;
        std::shared_ptr<Tensor_attributes> IN_1;
        std::shared_ptr<Tensor_attributes> IN_2;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> OUT_0;
    } outputs;

    PointwiseMode_t mode = PointwiseMode_t::NOT_SET;
    std::optional<int64_t> axis;
    std::optional<float> relu_lower_clip_slope;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, IN_0, IN_1, IN_2)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, OUT_0)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Pointwise_attributes, name, tag, inputs, outputs, mode, axis)

    Pointwise_attributes() : Operation(Tag::Pointwise) {}

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
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Pointwise_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Pointwise_attributes& {
        // Fill node's tensors
        inputs.IN_0->fill_from_context(context);
        if (inputs.IN_1) inputs.IN_1->fill_from_context(context);
        if (inputs.IN_2) inputs.IN_2->fill_from_context(context);
        outputs.OUT_0->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Layernorm_backward_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> DY;
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
        std::shared_ptr<Tensor_attributes> EPSILON;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> DX;
        std::shared_ptr<Tensor_attributes> DSCALE;
        std::shared_ptr<Tensor_attributes> DBIAS;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, DY, X, SCALE, MEAN, INV_VARIANCE, EPSILON)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, DX, DSCALE, DBIAS)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Layernorm_backward_attributes, name, tag, inputs, outputs)

    Layernorm_backward_attributes() : Operation(Tag::DLN) {}

    Layernorm_backward_attributes&
    set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                    std::shared_ptr<Tensor_attributes> inv_variance) {
        inputs.MEAN         = mean;
        inputs.INV_VARIANCE = inv_variance;
        return *this;
    }

    Layernorm_backward_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes> epsilon) {
        inputs.EPSILON = epsilon;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor_attributes>(std::string const&)> output_tensor) {
        outputs.DX     = output_tensor(name + "_DX_output");
        outputs.DSCALE = output_tensor(name + "_DSCALE_output");
        outputs.DBIAS  = output_tensor(name + "_DBIAS_output");
    }

    Layernorm_backward_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Layernorm_backward_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    Layernorm_backward_attributes&
    fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.DY->fill_from_context(context);

        if (inputs.MEAN) {
            inputs.MEAN->fill_from_context(context);
        }
        if (inputs.INV_VARIANCE) {
            inputs.INV_VARIANCE->fill_from_context(context);
        }
        if (inputs.EPSILON) {
            inputs.EPSILON->fill_from_context(context);
        }

        outputs.DX->fill_from_context(context);
        outputs.DSCALE->fill_from_context(context);
        outputs.DBIAS->fill_from_context(context);

        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Layernorm_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> BIAS;
        std::shared_ptr<Tensor_attributes> EPSILON;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
    } outputs;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X, SCALE, BIAS, EPSILON)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y, MEAN, INV_VARIANCE)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Layernorm_attributes, name, tag, inputs, outputs, forward_phase)

    Layernorm_attributes() : Operation(Tag::LN) {}

    Layernorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Layernorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs.EPSILON = value;
        return *this;
    }

    Layernorm_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Layernorm_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor_attributes>(std::string const&)> output_tensor) {
        outputs.Y = output_tensor(name + "_Y_output");
        if (forward_phase == NormFwdPhase_t::TRAINING) {
            outputs.MEAN         = output_tensor(name + "_MEAN_output");
            outputs.INV_VARIANCE = output_tensor(name + "_INV_VARIANCE_output");
        }
    }

    auto
    fill_from_context(detail::Context const& context) -> Layernorm_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.BIAS->fill_from_context(context);
        inputs.EPSILON->fill_from_context(context);

        outputs.Y->fill_from_context(context);
        if (forward_phase == NormFwdPhase_t::TRAINING) {
            outputs.MEAN->fill_from_context(context);
            outputs.INV_VARIANCE->fill_from_context(context);
        }

        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Batchnorm_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> BIAS;
        std::shared_ptr<Tensor_attributes> PREV_RUNNING_MEAN;
        std::shared_ptr<Tensor_attributes> PREV_RUNNING_VAR;
        std::shared_ptr<Tensor_attributes> EPSILON;
        std::shared_ptr<Tensor_attributes> MOMENTUM;
        std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
        std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN;
        std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR;
    } outputs;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X, SCALE, BIAS, PREV_RUNNING_MEAN, PREV_RUNNING_VAR, EPSILON, MOMENTUM)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_attributes, name, tag, inputs, outputs, forward_phase)

    Batchnorm_attributes() : Operation(Tag::BN) {}

    Batchnorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Batchnorm_attributes&
    set_previous_running_stats(std::shared_ptr<Tensor_attributes>& mean,
                               std::shared_ptr<Tensor_attributes>& variance,
                               std::shared_ptr<Tensor_attributes>& momentum) {
        inputs.PREV_RUNNING_MEAN = mean;
        inputs.PREV_RUNNING_VAR  = variance;
        inputs.MOMENTUM          = momentum;
        return *this;
    }

    Batchnorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs.EPSILON = value;
        return *this;
    }

    Batchnorm_attributes&
    set_peer_stats(std::vector<std::shared_ptr<Tensor_attributes>> const& peer_stats) {
        inputs.peer_stats = peer_stats;
        return *this;
    }

    Batchnorm_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Batchnorm_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor_attributes>(std::string const&)> output_tensor) {
        outputs.Y                 = output_tensor(name + "_Y_output");
        outputs.MEAN              = output_tensor(name + "_MEAN_output");
        outputs.INV_VARIANCE      = output_tensor(name + "_INV_VARIANCE_output");
        outputs.NEXT_RUNNING_MEAN = output_tensor(name + "_NEXT_RUNNING_MEAN_output");
        outputs.NEXT_RUNNING_VAR  = output_tensor(name + "_NEXT_RUNNING_VAR_output");
    }

    auto
    fill_from_context(detail::Context const& context) -> Batchnorm_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.BIAS->fill_from_context(context);
        inputs.PREV_RUNNING_MEAN->fill_from_context(context);
        inputs.PREV_RUNNING_VAR->fill_from_context(context);
        inputs.EPSILON->fill_from_context(context);
        inputs.MOMENTUM->fill_from_context(context);

        outputs.Y->fill_from_context(context);
        outputs.MEAN->fill_from_context(context);
        outputs.INV_VARIANCE->fill_from_context(context);
        outputs.NEXT_RUNNING_MEAN->fill_from_context(context);
        outputs.NEXT_RUNNING_VAR->fill_from_context(context);

        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Batchnorm_inference_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> BIAS;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X, MEAN, INV_VARIANCE, SCALE, BIAS)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_inference_attributes, name, tag, inputs, outputs)

    Batchnorm_inference_attributes() : Operation(Tag::BN_inference) {}

    Batchnorm_inference_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Batchnorm_inference_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Batchnorm_inference_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.BIAS->fill_from_context(context);
        inputs.MEAN->fill_from_context(context);
        inputs.INV_VARIANCE->fill_from_context(context);

        outputs.Y->fill_from_context(context);

        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Reduction_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
    } outputs;

    std::optional<ReductionMode_t> mode;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Reduction_attributes, name, tag, inputs, outputs, mode)

    Reduction_attributes() : Operation(Tag::Reduction) {}

    std::optional<ReductionMode_t>
    get_mode() const {
        return mode;
    }

    Reduction_attributes&
    set_mode(ReductionMode_t value) {
        mode = value;
        return *this;
    }

    Reduction_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Reduction_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Reduction_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        outputs.Y->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Rng_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> Seed;
        std::shared_ptr<Tensor_attributes> Offset;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
    } outputs;

    RngDistribution_t distribution = RngDistribution_t::NOT_SET;
    std::vector<int64_t> dim       = {};
    std::vector<int64_t> stride    = {};
    std::optional<int64_t> seed;
    std::optional<double> bernoulli_probability;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, Seed, Offset)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rng_attributes,
                                   name,
                                   tag,
                                   inputs,
                                   outputs,
                                   distribution,
                                   dim,
                                   stride,
                                   seed,
                                   bernoulli_probability)

    Rng_attributes() : Operation(Tag::Rng) {}

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

    Rng_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Rng_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Rng_attributes& {
        // Fill node's tensors
        if (inputs.Seed) inputs.Seed->fill_from_context(context);
        if (inputs.Offset) inputs.Offset->fill_from_context(context);
        outputs.Y->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Reshape_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
    } outputs;

    std::vector<int64_t> dim    = {};
    std::vector<int64_t> stride = {};

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Reshape_attributes, name, tag, inputs, outputs, dim, stride)

    Reshape_attributes() : Operation(Tag::Reshape) {}

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

    Reshape_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Reshape_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Reshape_attributes& {
        inputs.X->fill_from_context(context);
        outputs.Y->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Scaled_dot_product_attention_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> Q;
        std::shared_ptr<Tensor_attributes> K;
        std::shared_ptr<Tensor_attributes> Attn_scale;
        std::shared_ptr<Tensor_attributes> Bias;  // Optional bias after bmm1
        std::shared_ptr<Tensor_attributes> V;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_Q;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_KV;
        std::shared_ptr<Tensor_attributes> Mask;
        std::shared_ptr<Tensor_attributes> Dropout_mask;
        std::shared_ptr<Tensor_attributes> Dropout_scale;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> O;
        std::shared_ptr<Tensor_attributes>
            S;  // softmax output dumped when is_inference false. Users first need to check whether its nullptr.
    } outputs;

    std::optional<bool> is_inference;
    bool padding_mask = false;
    bool causal_mask  = false;
    std::optional<float> dropout_probability;
    int64_t seed;
    float dropout_scale = 1.f;

   public:
    Scaled_dot_product_attention_attributes() : Operation(Tag::Scaled_dot_product_attention), is_inference(false) {}

    Scaled_dot_product_attention_attributes&
    set_is_inference(bool const value) {
        is_inference = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_Q = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_KV = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs.Attn_scale = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> bias) {
        inputs.Bias = bias;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_dropout(float const probability, int64_t const seed_) {
        dropout_probability = probability;
        seed                = seed_;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
        inputs.Dropout_mask  = mask;
        inputs.Dropout_scale = scale;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.Q->fill_from_context(context);
        inputs.K->fill_from_context(context);
        inputs.V->fill_from_context(context);
        inputs.SEQ_LEN_Q->fill_from_context(context);
        inputs.SEQ_LEN_KV->fill_from_context(context);
        outputs.O->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Scaled_dot_product_flash_attention_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> Q;
        std::shared_ptr<Tensor_attributes> K;
        std::shared_ptr<Tensor_attributes> V;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_Q;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_KV;
        std::shared_ptr<Tensor_attributes> Attn_scale;
        std::shared_ptr<Tensor_attributes> Bias;
        std::shared_ptr<Tensor_attributes> Seed;
        std::shared_ptr<Tensor_attributes> Offset;
        std::shared_ptr<Tensor_attributes> Dropout_mask;
        std::shared_ptr<Tensor_attributes> Dropout_scale;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> O;
        std::shared_ptr<Tensor_attributes> Stats;  // softmax stats dumped when in forward training mode. Users first
                                                   // need to check whether its nullptr.
    } outputs;

    std::optional<bool> is_inference;
    bool padding_mask = false;
    bool alibi_mask   = false;
    bool causal_mask  = false;
    std::optional<float> dropout_probability;

    Scaled_dot_product_flash_attention_attributes() : Operation(Tag::Scaled_dot_product_flash_attention) {}

    Scaled_dot_product_flash_attention_attributes&
    set_is_inference(bool const value) {
        is_inference = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_alibi_mask(bool const value) {
        alibi_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs.Attn_scale = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs.Bias = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_Q = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_KV = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability = probability;
        inputs.Seed         = seed;
        inputs.Offset       = offset;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
        inputs.Dropout_mask  = mask;
        inputs.Dropout_scale = scale;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.Q->fill_from_context(context);
        inputs.K->fill_from_context(context);
        inputs.V->fill_from_context(context);
        outputs.O->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Scaled_dot_product_flash_attention_backward_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> Q;
        std::shared_ptr<Tensor_attributes> K;
        std::shared_ptr<Tensor_attributes> V;
        std::shared_ptr<Tensor_attributes> O;
        std::shared_ptr<Tensor_attributes> dO;
        std::shared_ptr<Tensor_attributes> Stats;
        std::shared_ptr<Tensor_attributes> Attn_scale;
        std::shared_ptr<Tensor_attributes> Bias;
        std::shared_ptr<Tensor_attributes> Seed;
        std::shared_ptr<Tensor_attributes> Offset;
        std::shared_ptr<Tensor_attributes> Dropout_mask;
        std::shared_ptr<Tensor_attributes> Dropout_scale;
        std::shared_ptr<Tensor_attributes> Dropout_scale_inv;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> dQ;
        std::shared_ptr<Tensor_attributes> dK;
        std::shared_ptr<Tensor_attributes> dV;
    } outputs;

    bool causal_mask = false;
    std::optional<float> dropout_probability;

   public:
    Scaled_dot_product_flash_attention_backward_attributes()
        : Operation(Tag::Scaled_dot_product_flash_attention_backward) {}

    Scaled_dot_product_flash_attention_backward_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs.Attn_scale = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs.Bias = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability = probability;
        inputs.Seed         = seed;
        inputs.Offset       = offset;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask,
                std::shared_ptr<Tensor_attributes> scale,
                std::shared_ptr<Tensor_attributes> scale_inv) {
        inputs.Dropout_mask      = mask;
        inputs.Dropout_scale     = scale;
        inputs.Dropout_scale_inv = scale_inv;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.Q->fill_from_context(context);
        inputs.K->fill_from_context(context);
        inputs.V->fill_from_context(context);
        inputs.O->fill_from_context(context);
        inputs.dO->fill_from_context(context);
        inputs.Stats->fill_from_context(context);
        outputs.dQ->fill_from_context(context);
        outputs.dK->fill_from_context(context);
        outputs.dV->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Softmax_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> P;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes>
            S;  // softmax output dumped when in forward training mode. Users first need to check whether its nullptr.
        std::shared_ptr<Tensor_attributes> Stats;  // softmax stats dumped when in forward training mode. Users first
                                                   // need to check whether its nullptr.
    } outputs;

    std::optional<bool> use_stats;

    Softmax_attributes() : Operation(Tag::Softmax) {}

    Softmax_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Softmax_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Softmax_attributes&
    fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.P->fill_from_context(context);
        outputs.S->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Conv_wgrad_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> DY;
        std::shared_ptr<Tensor_attributes> X;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> DW;
    } outputs;

    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, DY, X)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, DW)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_wgrad_attributes, name, tag, inputs, outputs, padding, stride, dilation)

    Conv_wgrad_attributes() : Operation(Tag::Conv_wgrad) {}

    std::vector<int64_t>
    get_padding() const {
        return padding;
    }

    Conv_wgrad_attributes&
    set_padding(std::vector<int64_t> value) {
        padding = value;
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

    Conv_wgrad_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Conv_wgrad_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Conv_wgrad_attributes& {
        // Fill node's tensors
        inputs.DY->fill_from_context(context);
        inputs.X->fill_from_context(context);
        outputs.DW->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

}  // namespace graph

}  // namespace cudnn_frontend