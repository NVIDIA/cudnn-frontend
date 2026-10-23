// Construct advanced Graph operations that do not have Python counterparts.

#include <cudnn_frontend.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

namespace fe = cudnn_frontend;

std::shared_ptr<fe::graph::Tensor_attributes>
tensor(fe::graph::Graph& graph, char const* name, fe::DataType_t type = fe::DataType_t::FLOAT) {
    return graph.tensor(fe::graph::Tensor_attributes()
                            .set_name(name)
                            .set_dim({1, 1, 4, 4})
                            .set_stride({16, 16, 4, 1})
                            .set_data_type(type));
}

int
main() {
    // --8<-- [start:bn-finalize]
    fe::graph::Graph bn_graph;
    auto sum               = tensor(bn_graph, "sum");
    auto sq_sum            = tensor(bn_graph, "sq_sum");
    auto scale             = tensor(bn_graph, "scale");
    auto bias              = tensor(bn_graph, "bias");
    auto previous_mean     = tensor(bn_graph, "previous_mean");
    auto previous_variance = tensor(bn_graph, "previous_variance");
    auto epsilon           = bn_graph.tensor(0.001f, fe::graph::ScalarType::COMPILE_TIME_CONST);
    auto momentum          = bn_graph.tensor(0.1f, fe::graph::ScalarType::COMPILE_TIME_CONST);
    auto count      = bn_graph.tensor(int64_t{16}, fe::graph::ScalarType::COMPILE_TIME_CONST);
    auto bn_options = fe::graph::BN_finalize_attributes().set_previous_running_stats(
        previous_mean, previous_variance, momentum);
    auto bn_outputs = bn_graph.bn_finalize(sum, sq_sum, scale, bias, epsilon, count, bn_options);
    // --8<-- [end:bn-finalize]

    // --8<-- [start:dbn-weight]
    fe::graph::Graph dbn_graph;
    auto dy           = tensor(dbn_graph, "dY");
    auto x            = tensor(dbn_graph, "X");
    auto mean         = tensor(dbn_graph, "mean");
    auto inv_variance = tensor(dbn_graph, "inv_variance");
    auto dbn_scale    = tensor(dbn_graph, "scale");
    auto dbn_outputs  = dbn_graph.dbn_weight(
        dy, x, mean, inv_variance, dbn_scale, fe::graph::DBN_weight_attributes());
    // --8<-- [end:dbn-weight]

    // --8<-- [start:diagonal-mask]
    fe::graph::Graph mask_graph;
    auto scores            = tensor(mask_graph, "scores");
    auto negative_infinity = mask_graph.tensor(-std::numeric_limits<float>::infinity(),
                                               fe::graph::ScalarType::COMPILE_TIME_CONST);
    auto masked            = mask_graph.diagonal_band_mask(
        scores,
        negative_infinity,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        fe::graph::DiagonalBandMask_attributes().set_comparison_mode(fe::PointwiseMode_t::CMP_GE));
    // --8<-- [end:diagonal-mask]

    // --8<-- [start:resample]
    fe::graph::Graph resample_graph;
    auto image  = tensor(resample_graph, "image", fe::DataType_t::HALF);
    auto pooled = resample_graph.resample(image,
                                          fe::graph::Resample_attributes()
                                              .set_generate_index(true)
                                              .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
                                              .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
                                              .set_window({2, 2})
                                              .set_stride({2, 2})
                                              .set_pre_padding({0, 0})
                                              .set_post_padding({0, 0}));
    // --8<-- [end:resample]

    // --8<-- [start:rng]
    fe::graph::Graph rng_graph;
    auto seed   = tensor(rng_graph, "seed", fe::DataType_t::INT32);
    auto offset = tensor(rng_graph, "offset", fe::DataType_t::INT32);
    auto random = rng_graph.rng(seed,
                                offset,
                                fe::graph::Rng_attributes()
                                    .set_distribution(fe::RngDistribution_t::BERNOULLI)
                                    .set_bernoulli_probability(0.9)
                                    .set_dim({1, 1, 4, 4})
                                    .set_stride({16, 16, 4, 1}));
    // --8<-- [end:rng]

    (void)bn_outputs;
    (void)dbn_outputs;
    (void)masked;
    (void)pooled;
    (void)random;
    std::cout << "advanced operation graphs constructed successfully\n";
    return 0;
}
