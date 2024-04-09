#pragma once

#include "../graph_properties.h"
#include "../graph_helpers.h"

namespace cudnn_frontend::graph {

NLOHMANN_JSON_SERIALIZE_ENUM(BN_finalize_attributes::input_names,
                             {
                                 {BN_finalize_attributes::input_names::SUM, "SUM"},
                                 {BN_finalize_attributes::input_names::SQ_SUM, "SQ_SUM"},
                                 {BN_finalize_attributes::input_names::SCALE, "SCALE"},
                                 {BN_finalize_attributes::input_names::BIAS, "BIAS"},
                                 {BN_finalize_attributes::input_names::EPSILON, "EPSILON"},
                                 {BN_finalize_attributes::input_names::ACCUM_COUNT, "ACCUM_COUNT"},
                                 {BN_finalize_attributes::input_names::PREV_RUNNING_MEAN, "PREV_RUNNING_MEAN"},
                                 {BN_finalize_attributes::input_names::PREV_RUNNING_VAR, "PREV_RUNNING_VAR"},
                                 {BN_finalize_attributes::input_names::MOMENTUM, "MOMENTUM"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(BN_finalize_attributes::output_names,
                             {
                                 {BN_finalize_attributes::output_names::EQ_SCALE, "EQ_SCALE"},
                                 {BN_finalize_attributes::output_names::EQ_BIAS, "EQ_BIAS"},
                                 {BN_finalize_attributes::output_names::MEAN, "MEAN"},
                                 {BN_finalize_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                                 {BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN, "NEXT_RUNNING_MEAN"},
                                 {BN_finalize_attributes::output_names::NEXT_RUNNING_VAR, "NEXT_RUNNING_VAR"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_attributes::input_names,
                             {
                                 {Batchnorm_attributes::input_names::X, "X"},
                                 {Batchnorm_attributes::input_names::SCALE, "SCALE"},
                                 {Batchnorm_attributes::input_names::BIAS, "BIAS"},
                                 {Batchnorm_attributes::input_names::EPSILON, "EPSILON"},
                                 {Batchnorm_attributes::input_names::PREV_RUNNING_MEAN, "PREV_RUNNING_MEAN"},
                                 {Batchnorm_attributes::input_names::PREV_RUNNING_VAR, "PREV_RUNNING_VAR"},
                                 {Batchnorm_attributes::input_names::MOMENTUM, "MOMENTUM"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_attributes::output_names,
                             {
                                 {Batchnorm_attributes::output_names::Y, "Y"},
                                 {Batchnorm_attributes::output_names::MEAN, "MEAN"},
                                 {Batchnorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                                 {Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN, "NEXT_RUNNING_MEAN"},
                                 {Batchnorm_attributes::output_names::NEXT_RUNNING_VAR, "NEXT_RUNNING_VAR"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_backward_attributes::input_names,
                             {
                                 {Batchnorm_backward_attributes::input_names::DY, "DY"},
                                 {Batchnorm_backward_attributes::input_names::X, "X"},
                                 {Batchnorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {Batchnorm_backward_attributes::input_names::MEAN, "MEAN"},
                                 {Batchnorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_backward_attributes::output_names,
                             {
                                 {Batchnorm_backward_attributes::output_names::DX, "DX"},
                                 {Batchnorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {Batchnorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_inference_attributes::input_names,
                             {
                                 {Batchnorm_inference_attributes::input_names::X, "X"},
                                 {Batchnorm_inference_attributes::input_names::SCALE, "SCALE"},
                                 {Batchnorm_inference_attributes::input_names::BIAS, "BIAS"},
                                 {Batchnorm_inference_attributes::input_names::MEAN, "MEAN"},
                                 {Batchnorm_inference_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_inference_attributes::output_names,
                             {{Batchnorm_inference_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_dgrad_attributes::input_names,
                             {
                                 {Conv_dgrad_attributes::input_names::W, "W"},
                                 {Conv_dgrad_attributes::input_names::DY, "DY"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_dgrad_attributes::output_names,
                             {
                                 {Conv_dgrad_attributes::output_names::DX, "DX"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_fprop_attributes::input_names,
                             {
                                 {Conv_fprop_attributes::input_names::X, "X"},
                                 {Conv_fprop_attributes::input_names::W, "W"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_fprop_attributes::output_names,
                             {
                                 {Conv_fprop_attributes::output_names::Y, "Y"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_wgrad_attributes::input_names,
                             {
                                 {Conv_wgrad_attributes::input_names::DY, "DY"},
                                 {Conv_wgrad_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_wgrad_attributes::output_names,
                             {
                                 {Conv_wgrad_attributes::output_names::DW, "DW"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(DBN_weight_attributes::input_names,
                             {
                                 {DBN_weight_attributes::input_names::DY, "DY"},
                                 {DBN_weight_attributes::input_names::X, "X"},
                                 {DBN_weight_attributes::input_names::SCALE, "SCALE"},
                                 {DBN_weight_attributes::input_names::MEAN, "MEAN"},
                                 {DBN_weight_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(DBN_weight_attributes::output_names,
                             {
                                 {DBN_weight_attributes::output_names::DSCALE, "DSCALE"},
                                 {DBN_weight_attributes::output_names::DBIAS, "DBIAS"},
                                 {DBN_weight_attributes::output_names::EQ_BIAS, "EQ_BIAS"},
                                 {DBN_weight_attributes::output_names::EQ_SCALE_DY, "EQ_SCALE_DY"},
                                 {DBN_weight_attributes::output_names::EQ_SCALE_X, "EQ_SCALE_X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Genstats_attributes::input_names,
                             {
                                 {Genstats_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Genstats_attributes::output_names,
                             {
                                 {Genstats_attributes::output_names::SUM, "SUM"},
                                 {Genstats_attributes::output_names::SQ_SUM, "SQ_SUM"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Instancenorm_attributes::input_names,
                             {
                                 {Instancenorm_attributes::input_names::X, "X"},
                                 {Instancenorm_attributes::input_names::SCALE, "SCALE"},
                                 {Instancenorm_attributes::input_names::BIAS, "BIAS"},
                                 {Instancenorm_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Instancenorm_attributes::output_names,
                             {
                                 {Instancenorm_attributes::output_names::Y, "Y"},
                                 {Instancenorm_attributes::output_names::MEAN, "MEAN"},
                                 {Instancenorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Instancenorm_backward_attributes::input_names,
                             {
                                 {Instancenorm_backward_attributes::input_names::DY, "DY"},
                                 {Instancenorm_backward_attributes::input_names::X, "X"},
                                 {Instancenorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {Instancenorm_backward_attributes::input_names::MEAN, "MEAN"},
                                 {Instancenorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Instancenorm_backward_attributes::output_names,
                             {
                                 {Instancenorm_backward_attributes::output_names::DX, "DX"},
                                 {Instancenorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {Instancenorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Layernorm_attributes::input_names,
                             {
                                 {Layernorm_attributes::input_names::X, "X"},
                                 {Layernorm_attributes::input_names::SCALE, "SCALE"},
                                 {Layernorm_attributes::input_names::BIAS, "BIAS"},
                                 {Layernorm_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Layernorm_attributes::output_names,
                             {
                                 {Layernorm_attributes::output_names::Y, "Y"},
                                 {Layernorm_attributes::output_names::MEAN, "MEAN"},
                                 {Layernorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Layernorm_backward_attributes::input_names,
                             {
                                 {Layernorm_backward_attributes::input_names::DY, "DY"},
                                 {Layernorm_backward_attributes::input_names::X, "X"},
                                 {Layernorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {Layernorm_backward_attributes::input_names::MEAN, "MEAN"},
                                 {Layernorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Layernorm_backward_attributes::output_names,
                             {
                                 {Layernorm_backward_attributes::output_names::DX, "DX"},
                                 {Layernorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {Layernorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Matmul_attributes::input_names,
                             {
                                 {Matmul_attributes::input_names::A, "A"},
                                 {Matmul_attributes::input_names::B, "B"},
                                 {Matmul_attributes::input_names::M_override, "M_override"},
                                 {Matmul_attributes::input_names::N_override, "N_override"},
                                 {Matmul_attributes::input_names::K_override, "K_override"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Matmul_attributes::output_names,
                             {
                                 {Matmul_attributes::output_names::C, "C"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Matmul_fp8_attributes::input_names,
                             {
                                 {Matmul_fp8_attributes::input_names::A, "A"},
                                 {Matmul_fp8_attributes::input_names::B, "B"},
                                 {Matmul_fp8_attributes::input_names::Descale_A, "Descale_A"},
                                 {Matmul_fp8_attributes::input_names::Descale_B, "Descale_B"},
                                 {Matmul_fp8_attributes::input_names::Scale_C, "Scale_C"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Matmul_fp8_attributes::output_names,
                             {
                                 {Matmul_fp8_attributes::output_names::C, "C"},
                                 {Matmul_fp8_attributes::output_names::Amax_C, "Amax_C"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Pointwise_attributes::input_names,
                             {
                                 {Pointwise_attributes::input_names::IN_0, "IN_0"},
                                 {Pointwise_attributes::input_names::IN_1, "IN_1"},
                                 {Pointwise_attributes::input_names::IN_2, "IN_2"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Pointwise_attributes::output_names,
                             {
                                 {Pointwise_attributes::output_names::OUT_0, "OUT_0"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Reduction_attributes::input_names,
                             {
                                 {Reduction_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Reduction_attributes::output_names, {{Reduction_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Reshape_attributes::input_names,
                             {
                                 {Reshape_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Reshape_attributes::output_names, {{Reshape_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Rmsnorm_attributes::input_names,
                             {
                                 {Rmsnorm_attributes::input_names::X, "X"},
                                 {Rmsnorm_attributes::input_names::SCALE, "SCALE"},
                                 {Rmsnorm_attributes::input_names::BIAS, "BIAS"},
                                 {Rmsnorm_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rmsnorm_attributes::output_names,
                             {
                                 {Rmsnorm_attributes::output_names::Y, "Y"},
                                 {Rmsnorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rmsnorm_backward_attributes::input_names,
                             {
                                 {Rmsnorm_backward_attributes::input_names::DY, "DY"},
                                 {Rmsnorm_backward_attributes::input_names::X, "X"},
                                 {Rmsnorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {Rmsnorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rmsnorm_backward_attributes::output_names,
                             {
                                 {Rmsnorm_backward_attributes::output_names::DX, "DX"},
                                 {Rmsnorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {Rmsnorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rng_attributes::input_names,
                             {
                                 {Rng_attributes::input_names::Seed, "Seed"},
                                 {Rng_attributes::input_names::Offset, "Offset"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rng_attributes::output_names, {{Rng_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_attributes::input_names,
                             {
                                 {SDPA_attributes::input_names::Q, "Q"},
                                 {SDPA_attributes::input_names::K, "K"},
                                 {SDPA_attributes::input_names::V, "V"},
                                 {SDPA_attributes::input_names::Attn_scale, "Attn_scale"},
                                 {SDPA_attributes::input_names::Bias, "Bias"},
                                 {SDPA_attributes::input_names::SEQ_LEN_Q, "SEQ_LEN_Q"},
                                 {SDPA_attributes::input_names::SEQ_LEN_KV, "SEQ_LEN_KV"},
                                 {SDPA_attributes::input_names::Seed, "Seed"},
                                 {SDPA_attributes::input_names::Offset, "Offset"},
                                 {SDPA_attributes::input_names::Dropout_mask, "Dropout_mask"},
                                 {SDPA_attributes::input_names::Dropout_scale, "Dropout_scale"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_attributes::output_names,
                             {{SDPA_attributes::output_names::O, "O"},
                              {SDPA_attributes::output_names::Stats, "Stats"},
                              {SDPA_attributes::output_names::RNG_DUMP, "RNG_DUMP"}})

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_fp8_attributes::input_names,
                             {
                                 {SDPA_fp8_attributes::input_names::Q, "Q"},
                                 {SDPA_fp8_attributes::input_names::K, "K"},
                                 {SDPA_fp8_attributes::input_names::V, "V"},
                                 {SDPA_fp8_attributes::input_names::Attn_scale, "Attn_scale"},
                                 {SDPA_fp8_attributes::input_names::Descale_Q, "Descale_Q"},
                                 {SDPA_fp8_attributes::input_names::Descale_K, "Descale_K"},
                                 {SDPA_fp8_attributes::input_names::Descale_V, "Descale_V"},
                                 {SDPA_fp8_attributes::input_names::Descale_S, "Descale_S"},
                                 {SDPA_fp8_attributes::input_names::Scale_S, "Scale_S"},
                                 {SDPA_fp8_attributes::input_names::Scale_O, "Scale_O"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_fp8_attributes::output_names,
                             {
                                 {SDPA_fp8_attributes::output_names::O, "O"},
                                 {SDPA_fp8_attributes::output_names::Stats, "Stats"},
                                 {SDPA_fp8_attributes::output_names::Amax_O, "Amax_O"},
                                 {SDPA_fp8_attributes::output_names::Amax_S, "Amax_S"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_backward_attributes::input_names,
                             {
                                 {SDPA_backward_attributes::input_names::Q, "Q"},
                                 {SDPA_backward_attributes::input_names::K, "K"},
                                 {SDPA_backward_attributes::input_names::V, "V"},
                                 {SDPA_backward_attributes::input_names::O, "O"},
                                 {SDPA_backward_attributes::input_names::dO, "dO"},
                                 {SDPA_backward_attributes::input_names::Stats, "Stats"},
                                 {SDPA_backward_attributes::input_names::Attn_scale, "Attn_scale"},
                                 {SDPA_backward_attributes::input_names::Bias, "Bias"},
                                 {SDPA_backward_attributes::input_names::SEQ_LEN_Q, "SEQ_LEN_Q"},
                                 {SDPA_backward_attributes::input_names::SEQ_LEN_KV, "SEQ_LEN_KV"},
                                 {SDPA_backward_attributes::input_names::Seed, "Seed"},
                                 {SDPA_backward_attributes::input_names::Offset, "Offset"},
                                 {SDPA_backward_attributes::input_names::Dropout_mask, "Dropout_mask"},
                                 {SDPA_backward_attributes::input_names::Dropout_scale, "Dropout_scale"},
                                 {SDPA_backward_attributes::input_names::Dropout_scale_inv, "Dropout_scale_inv"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_backward_attributes::output_names,
                             {
                                 {SDPA_backward_attributes::output_names::dQ, "dQ"},
                                 {SDPA_backward_attributes::output_names::dK, "dK"},
                                 {SDPA_backward_attributes::output_names::dV, "dV"},
                                 {SDPA_backward_attributes::output_names::dBias, "dBias"},
                                 {SDPA_backward_attributes::output_names::RNG_DUMP, "RNG_DUMP"},
                             })

inline void
to_json(nlohmann::json& j, const Tensor_attributes& ta) {
    j = nlohmann::json{{"name", ta.name},
                       {"data_type", ta.data_type},
                       {"dim", ta.dim},
                       {"stride", ta.stride},
                       {"is_virtual", ta.is_virtual},
                       {"pass_by_value", ta.pass_by_value},
                       {"is_pass_by_value", ta.is_pass_by_value},
                       {"reordering_type", ta.reordering_type},
                       {"uid", ta.uid},
                       {"uid_assigned", ta.uid_assigned}};
}

inline void
from_json(const nlohmann::json& j, Tensor_attributes& ta) {
    ta.name             = j.at("name").get<std::string>();
    ta.data_type        = j.at("data_type").get<DataType_t>();
    ta.dim              = j.at("dim").get<std::vector<int64_t>>();
    ta.stride           = j.at("stride").get<std::vector<int64_t>>();
    ta.is_virtual       = j.at("is_virtual").get<bool>();
    ta.is_pass_by_value = j.at("is_pass_by_value").get<bool>();
    ta.reordering_type  = j.at("reordering_type").get<TensorReordering_t>();
    ta.uid              = j.at("uid").get<Tensor_attributes::uid_t>();
    ta.uid_assigned     = j.at("uid_assigned").get<bool>();

    if (ta.is_pass_by_value && !j["pass_by_value"].is_null()) {
        ta.pass_by_value = j.at("pass_by_value");
    }
}

}  // namespace cudnn_frontend::graph