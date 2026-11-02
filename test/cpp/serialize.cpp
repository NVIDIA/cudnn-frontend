/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include <sstream>

TEST_CASE("Tensor integer JSON fields reject numeric coercion before conversion",
          "[serialize][boundary][integer_schema]") {
    namespace fe      = cudnn_frontend;
    json const tensor = fe::graph::Tensor_attributes(1.0F).set_uid(-17);
    std::vector<json> const invalid{1.5,
                                    1.0e100,
                                    -1.0e100,
                                    std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::quiet_NaN(),
                                    uint64_t{1} << 63,
                                    true,
                                    "1",
                                    nullptr};
    for (auto field : {"uid", "dim", "stride", "ragged_offset_uid", "ragged_offset_multiplier", "alignment"}) {
        for (auto const &value : invalid) {
            auto info = tensor;
            if (std::string(field) == "dim" || std::string(field) == "stride") {
                info[field] = json::array({1, value});
            } else {
                info[field] = value;
            }
            INFO(field << ": " << value.dump());
            REQUIRE_THROWS_AS(info.get<fe::graph::Tensor_attributes>(), std::exception);
            json graph{{"json_version", "2.0"}, {"gid", 1}, {"nodes", json::array()}, {"tensors", json::array({info})}};
            fe::graph::Graph structural;
            fe::error_t status;
            REQUIRE_NOTHROW(status = structural.deserialize(graph));
            REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
            fe::graph::Graph plan;
            REQUIRE_NOTHROW(status = plan.deserialize(static_cast<cudnnHandle_t>(nullptr), graph, false, false));
            REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        }
    }
    for (auto value : {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}) {
        auto valid   = tensor;
        valid["uid"] = value;
        REQUIRE(valid.get<fe::graph::Tensor_attributes>().get_uid() == value);
    }
    auto reused = fe::graph::Tensor_attributes(2.0F).set_ragged_offset(
        std::make_shared<fe::graph::Tensor_attributes>(fe::graph::Tensor_attributes().set_uid(99)));
    reused.set_ragged_offset_multiplier(4);
    from_json(tensor, reused);
    REQUIRE(reused.get_ragged_offset() == nullptr);
    REQUIRE_FALSE(reused.has_ragged_offset_multiplier());
}

TEST_CASE(
    "Structural graph integer fields and inspection-only reducers have "
    "safe boundaries",
    "[serialize][boundary][integer_schema]") {
    namespace fe = cudnn_frontend;
    json const base{{"json_version", "2.0"}, {"gid", 1}, {"nodes", json::array()}, {"tensors", json::array()}};
    for (auto const &value : std::vector<json>{1.0e100, -1.0e100, 1.5, true, "1", nullptr}) {
        for (auto field : {"gid", "sm_count"}) {
            auto j = base;
            if (std::string(field) == "gid") {
                j[field] = value;
            } else {
                j["context"][field] = value;
            }
            fe::graph::Graph graph;
            fe::error_t status;
            REQUIRE_NOTHROW(status = graph.deserialize(j));
            // Null SM count is the historical optional-field spelling.
            if (std::string(field) == "sm_count" && value.is_null()) {
                REQUIRE(status.is_good());
            } else {
                REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
            }
        }
    }
    auto valid                   = base;
    valid["gid"]                 = std::numeric_limits<uint64_t>::max();
    valid["context"]["sm_count"] = -1;
    fe::graph::Graph graph;
    REQUIRE(graph.deserialize(valid).is_good());
    auto refused                   = base;
    refused["context"]["sm_count"] = 1.0e100;
    refused["nodes"] = json::array({json{{"tag", "MATMUL"}, {"inputs", 1.0e100}}, json{{"tag", "CUSTOM_REDUCTION"}}});
    fe::error_t status;
    REQUIRE_NOTHROW(status = graph.deserialize(refused));
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(status.get_message().find("inspection-only") != std::string::npos);
    for (auto const &tag :
         std::vector<json>{"CUSTOM_REDUCTIONsuffix", std::string("CUSTOM_REDUCTION\0suffix", 23), nullptr, 1}) {
        auto malformed     = base;
        malformed["nodes"] = json::array({json{{"tag", tag}}});
        REQUIRE_NOTHROW(status = graph.deserialize(malformed));
        REQUIRE(status.get_code() == fe::error_code_t::UNSUPPORTED_GRAPH_FORMAT);
    }
}

TEST_CASE("Runtime scalar JSON preserves its typed wire representation", "[serialize][plan_metadata][scalar]") {
    namespace fe = cudnn_frontend;
    using Scalar = fe::graph::Tensor_attributes::pass_by_values_t;
    std::vector<Scalar> const values{int64_t{std::numeric_limits<int64_t>::min()},
                                     int32_t{std::numeric_limits<int32_t>::max()},
                                     __float2half(-0.0F),
                                     -0.0F,
                                     std::numeric_limits<double>::infinity(),
                                     __float2bfloat16(1.5F)};
    for (auto const &value : values) {
        auto restored = json::from_ubjson(json::to_ubjson(json(value))).get<Scalar>();
        REQUIRE(restored.index() == value.index());
        std::visit(
            [&](auto const &expected) {
                using T = std::decay_t<decltype(expected)>;
                REQUIRE(std::memcmp(&expected, &std::get<T>(restored), sizeof(T)) == 0);
            },
            value);
    }
    auto legacy_double = json{{"index", 4}, {"value", 2}}.get<Scalar>();
    REQUIRE(std::get<double>(legacy_double) == 2.0);
}

TEST_CASE("Runtime scalar JSON rejects malformed tags and values", "[serialize][plan_metadata][scalar]") {
    using Scalar = cudnn_frontend::graph::Tensor_attributes::pass_by_values_t;
    std::vector<json> const malformed{nullptr,
                                      json::array(),
                                      json::object(),
                                      json{{"index", 0}},
                                      json{{"value", 0}},
                                      json{{"index", -1}, {"value", 0}},
                                      json{{"index", 6}, {"value", 0}},
                                      json{{"index", 1.5}, {"value", 0}},
                                      json{{"index", true}, {"value", 0}},
                                      json{{"index", 0}, {"value", 1.5}},
                                      json{{"index", 0}, {"value", true}},
                                      json{{"index", 0}, {"value", uint64_t{1} << 63}},
                                      json{{"index", 1}, {"value", int64_t{std::numeric_limits<int32_t>::max()} + 1}},
                                      json{{"index", 1}, {"value", int64_t{std::numeric_limits<int32_t>::min()} - 1}},
                                      json{{"index", 4}, {"value", "2.0"}},
                                      json{{"index", 4}, {"value", nullptr}}};
    for (auto const &value : malformed) {
        INFO(value.dump());
        REQUIRE_THROWS_AS(value.get<Scalar>(), std::exception);
    }
    for (auto index : {2, 3, 5}) {
        for (auto const &word : std::vector<std::string>{"",
                                                         "0",
                                                         "100000000",
                                                         "-0000000",
                                                         "+0000000",
                                                         " 0000000",
                                                         "0000000 ",
                                                         "0x000000",
                                                         "0000000Z",
                                                         std::string("0000000\0", 8),
                                                         "00000000junk"}) {
            INFO(index << ": " << word);
            auto value = json{{"index", index}, {"value", word}};
            REQUIRE_THROWS_AS(value.get<Scalar>(), std::exception);
        }
    }
}

TEST_CASE("Plan metadata validates sibling UID and workspace representations", "[serialize][plan_metadata][host]") {
    namespace md = cudnn_frontend::graph::plan_serialization_detail;
    json const base{{"cudnn_backend_data", "opaque"},
                    {"variant_pack_uids", json::array({1, -2})},
                    {"fe_workspace_size", 16},
                    {"variant_pack_replacements", json::array()},
                    {"behavior_notes", json::array({json::array()})}};
    REQUIRE(md::parse(base).workspace_size == 16);
    for (auto field : {"pass_by_values", "workspace_modifications", "aligned_operands"}) {
        json entry;
        if (std::string(field) == "pass_by_values") {
            entry = json{{"index", 0}, {"value", 0}};
        } else if (std::string(field) == "workspace_modifications") {
            entry = json::array({2, 0, json::array()});
        } else {
            entry = json::array({16, 4});
        }
        for (auto const &key : std::vector<std::string>{"",
                                                        "01",
                                                        "-0",
                                                        "+1",
                                                        " 1",
                                                        "1 ",
                                                        "1tail",
                                                        "9223372036854775808",
                                                        "-9223372036854775809",
                                                        std::string("1\0tail", 6)}) {
            auto j   = base;
            j[field] = json::object({{key, entry}});
            INFO(std::string(field) << ": " << key);
            REQUIRE_THROWS_AS(md::parse(j), std::exception);
        }
        for (auto const &key : {"-9223372036854775808", "9223372036854775807", "-2", "0"}) {
            auto j   = base;
            j[field] = json::object({{key, entry}});
            REQUIRE_NOTHROW(md::parse(j));
        }
    }
    std::vector<json> const bad_work{json::array({0, -1, json::array()}),
                                     json::array({0, 17, json::array()}),
                                     json::array({0, 16, json::array({"00000000"})}),
                                     json::array({3, 0, json::array()}),
                                     json::array({1, 0, json::array()}),
                                     json::array({1, 0, json::array({"41880000"})}),
                                     json::array({1, 0, json::array({"3FC00000"})}),
                                     json::array({1, 0, json::array({"BF800000"})}),
                                     json::array({1, 0, json::array({"7F800000"})}),
                                     json::array({1, 0, json::array({"7FC00000"})}),
                                     json::array({1, 0, json::array({"5F000000"})}),
                                     json::array({1, 0, json::array({"00000000", "00000000"})}),
                                     json::array({2, 0, json::array({"00000000"})}),
                                     json::array({0, 0.5, json::array()})};
    for (auto const &entry : bad_work) {
        auto j                       = base;
        j["workspace_modifications"] = {{"3", entry}};
        INFO(entry.dump());
        REQUIRE_THROWS_AS(md::parse(j), std::exception);
    }
    auto j                       = base;
    j["workspace_modifications"] = {{"3", json::array({0, 8, json::array({"7F800000", "7FC00000"})})},
                                    {"4", json::array({1, 0, json::array({"41800000"})})},
                                    {"5", json::array({2, 16, json::array()})}};
    REQUIRE(md::parse(j).workspace.size() == 3);
    j["aligned_operands"] = {{"1", json::array({16, 32})}};
    REQUIRE_NOTHROW(md::parse(j));
    for (auto value : {json(nullptr), json(true), json(1.5), json(uint64_t{1} << 63)}) {
        auto bad                    = base;
        bad["variant_pack_uids"][1] = value;
        REQUIRE_THROWS_AS(md::parse(bad), std::exception);
    }
    auto duplicate                 = base;
    duplicate["variant_pack_uids"] = {1, 1};
    REQUIRE_THROWS_AS(md::parse(duplicate), std::exception);
}

TEST_CASE(
    "Plan metadata preserves virtual predicates and bounds owned aliases "
    "and dumps",
    "[serialize][plan_metadata][host]") {
    namespace fe = cudnn_frontend;
    namespace md = fe::graph::plan_serialization_detail;
    json base{{"cudnn_backend_data", "opaque"},
              {"variant_pack_uids", json::array({1})},
              {"fe_workspace_size", 4},
              {"variant_pack_replacements", json::array()},
              {"behavior_notes", json::array({json::array()})},
              {"gid", 1},
              {"nodes", json::array()},
              {"tensors", json::array()}};
    auto predicate = fe::graph::Tensor_attributes()
                         .set_uid(2)
                         .set_dim({2})
                         .set_stride({1})
                         .set_data_type(fe::DataType_t::BOOLEAN)
                         .set_is_virtual(true);
    base["tensors"].push_back(json(predicate));
    REQUIRE_NOTHROW(md::parse(base));
    auto scalar = fe::graph::Tensor_attributes(2.0F).set_uid(3);
    base["tensors"].push_back(json(scalar));
    auto reconstructed = md::parse(base);
    REQUIRE(reconstructed.scalars.count(3) == 1);
    auto missing              = base;
    missing["pass_by_values"] = json::object();
    REQUIRE_THROWS_AS(md::parse(missing), std::exception);
    base["pass_by_values"] = {{"3", scalar.get_pass_by_value().value()}};
    REQUIRE_NOTHROW(md::parse(base));
    for (auto field : {"uid", "ragged_offset_uid", "ragged_offset_multiplier", "alignment"}) {
        auto bad                 = base;
        bad["tensors"][0][field] = 1.5;
        REQUIRE_THROWS_AS(md::parse(bad), std::exception);
    }
    auto bad_reordering                             = base;
    bad_reordering["tensors"][0]["reordering_type"] = "UNKNOWN";
    REQUIRE_THROWS_AS(md::parse(bad_reordering), std::exception);
    auto conflicting                        = base;
    conflicting["tensors"][1]["is_virtual"] = true;
    REQUIRE_THROWS_AS(md::parse(conflicting), std::exception);
    auto metadata = md::parse(base);
    std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>> chain{{3, {4, 0}}, {4, {5, 0}}};
    REQUIRE_NOTHROW(md::validate_owned_spans(metadata, chain));
    chain[1].second.second = 1;
    REQUIRE_THROWS_AS(md::validate_owned_spans(metadata, chain), std::exception);
    auto workspace                       = base;
    workspace["workspace_modifications"] = {{"6", json::array({2, 0, json::array()})}};
    auto work_metadata                   = md::parse(workspace);
    chain                                = {{6, {7, 2}}, {7, {8, 2}}};
    REQUIRE_NOTHROW(md::validate_owned_spans(work_metadata, chain));
    chain[1].second.second = 3;
    REQUIRE_THROWS_AS(md::validate_owned_spans(work_metadata, chain), std::exception);
    auto dumped =
        fe::graph::Tensor_attributes().set_uid(6).set_dim({1024}).set_stride({0}).set_data_type(fe::DataType_t::FLOAT);
    workspace["tensors"].push_back(json(dumped));
    workspace["tensors_to_dump"] = json::array({json::array({6, int('x')})});
    auto dump_metadata           = md::parse(workspace);
    REQUIRE_THROWS_AS(md::validate_owned_spans(dump_metadata, {}), std::exception);
    workspace["tensors"].back()["dim"] = {1};
    REQUIRE_NOTHROW(md::validate_owned_spans(md::parse(workspace), {}));
    workspace.erase("tensors");
    workspace.erase("nodes");
    workspace.erase("gid");
    REQUIRE(md::parse(workspace).dumps.empty());
}

TEST_CASE("Plan metadata separates scalar reads empty shapes aliases and dumps",
          "[serialize][plan_metadata][pbv_contract]") {
    namespace fe = cudnn_frontend;
    namespace md = fe::graph::plan_serialization_detail;
    json base{{"cudnn_backend_data", "opaque"},
              {"variant_pack_uids", json::array({1})},
              {"fe_workspace_size", 16},
              {"variant_pack_replacements", json::array()},
              {"behavior_notes", json::array({json::array()})},
              {"gid", 1},
              {"nodes", json::array()},
              {"tensors", json::array()}};
    for (auto extent : {int64_t{0}, int64_t{2}}) {
        DYNAMIC_SECTION("PBV extent " << extent) {
            auto scalar  = fe::graph::Tensor_attributes(2.0F).set_uid(1).set_dim({extent}).set_stride({1});
            auto payload = base;
            payload["tensors"].push_back(json(scalar));
            auto metadata = md::parse(payload);
            REQUIRE(metadata.scalars.count(1) == 1);
            REQUIRE(md::tensor_bytes(*metadata.tensors.at(1)) == extent * 4);
            REQUIRE(md::dump_bytes(*metadata.tensors.at(1)) == extent * 4);
            REQUIRE_NOTHROW(md::validate_owned_spans(metadata, {}));
            auto roundtrip = md::parse(json::from_ubjson(json::to_ubjson(payload)));
            REQUIRE(roundtrip.tensors.at(1)->get_dim() == std::vector<int64_t>{extent});
            payload["tensors_to_dump"] = json::array({json::array({1, int('x')})});
            if (extent == 0) {
                REQUIRE_NOTHROW(md::validate_owned_spans(md::parse(payload), {}));
            } else {
                REQUIRE_THROWS_AS(md::validate_owned_spans(md::parse(payload), {}), std::exception);
            }
            payload["tensors_to_dump"]         = json::array();
            payload["tensors"][0]["data_type"] = "DOUBLE";
            REQUIRE_THROWS_AS(md::parse(payload), std::exception);
        }
    }
    auto zero = fe::graph::Tensor_attributes()
                    .set_uid(1)
                    .set_dim({0, 2})
                    .set_stride({2, 1})
                    .set_data_type(fe::DataType_t::FLOAT)
                    .set_is_virtual(true);
    base["tensors"].push_back(json(zero));
    REQUIRE_NOTHROW(md::parse(base));
    REQUIRE(md::tensor_bytes(zero) == 0);
    REQUIRE(md::dump_bytes(zero) == 0);
    auto scalar      = fe::graph::Tensor_attributes(2.0F).set_uid(1);
    base["tensors"]  = json::array({json(scalar)});
    auto empty_alias = fe::graph::Tensor_attributes()
                           .set_uid(2)
                           .set_dim({0})
                           .set_stride({1})
                           .set_data_type(fe::DataType_t::FLOAT)
                           .set_is_pass_by_value(true);
    auto end_alias = empty_alias;
    end_alias.set_uid(3).set_dim({1});
    base["tensors"].push_back(json(empty_alias));
    base["tensors"].push_back(json(end_alias));
    std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>> chain{{1, {2, 0}}, {2, {3, 0}}};
    REQUIRE_NOTHROW(md::validate_owned_spans(md::parse(base), chain));
    // A PBV vector consumes one scalar, even when its declared tensor span is
    // larger than the scalar owner. This distinguishes read width from span.
    auto vector_alias = empty_alias;
    vector_alias.set_dim({2});
    auto vector_payload       = base;
    vector_payload["tensors"] = json::array({json(scalar), json(vector_alias)});
    std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>> vector_chain{{1, {2, 0}}};
    REQUIRE_NOTHROW(md::validate_owned_spans(md::parse(vector_payload), vector_chain));
    base["tensors"][2]["data_type"] = "DOUBLE";
    REQUIRE_THROWS_AS(md::validate_owned_spans(md::parse(base), chain), std::exception);
    // A raw PBV vector still owns a scalar argument read even when its shape is
    // empty.
    base["tensors"] = json::array({json(empty_alias.set_uid(1)), json(end_alias.set_uid(2))});
    chain           = {{1, {2, 0}}};
    REQUIRE_NOTHROW(md::validate_owned_spans(md::parse(base), chain));
    auto workspace                       = base;
    workspace["workspace_modifications"] = {{"1", json::array({2, 0, json::array()})}};
    REQUIRE_THROWS_AS(md::parse(workspace), std::exception);
    workspace["tensors"][0]["is_pass_by_value"] = false;
    REQUIRE_THROWS_AS(md::validate_owned_spans(md::parse(workspace), chain), std::exception);
    workspace["tensors"][1]["is_pass_by_value"] = false;
    workspace["tensors"][0]["dim"]              = {4};
    REQUIRE_NOTHROW(md::validate_owned_spans(md::parse(workspace), chain));
}

TEST_CASE("Reduction plan warmup omits fake dumps and preserves execution dumps",
          "[serialize][custom_reduction][warmup_dumps]") {
    namespace fe = cudnn_frontend;
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        SKIP("Requires cuDNN9.29");
    }
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto input   = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_uid(1)
                                  .set_dim({1, 4, 1, 1})
                                  .set_stride({4, 1, 4, 4})
                                  .set_name("warmup_input"));
    auto scalar  = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_uid(2)
                                   .set_dim({1, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_is_pass_by_value(true)
                                   .set_name("warmup_raw_scalar"));
    auto shifted = graph.pointwise(input, scalar, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    auto output =
        graph.custom_reduction({shifted},
                               fe::graph::reduction_modifiers::sum(
                                   fe::graph::Custom_reduction_attributes().set_reduction_dims({1}).set_identity_values(
                                       {fe::graph::Custom_reduction_identity(0.0F)})))[0];
    output->set_output(true).set_uid(3);
    struct Resources {
        cudnnHandle_t handle = nullptr;
        void *input = nullptr, *output = nullptr, *workspace = nullptr;
        ~Resources() {
            cudaFree(workspace);
            cudaFree(output);
            cudaFree(input);
            if (handle) cudnnDestroy(handle);
        }
    } resources;
    REQUIRE(cudnnCreate(&resources.handle) == CUDNN_STATUS_SUCCESS);
    auto built = graph.build(resources.handle, {fe::HeurMode_t::A});
    INFO(built.get_message());
    REQUIRE(built.is_good());
    std::vector<uint8_t> blob;
    REQUIRE(graph.serialize(blob).is_good());
    auto payload               = json::from_ubjson(blob);
    payload["tensors_to_dump"] = json::array({json::array({scalar->get_uid(), int('x')})});
    blob                       = json::to_ubjson(payload);
    std::ostringstream captured;
    // Test-only logger capture. Product warmup must not toggle shared logger
    // state.
    (void)fe::getLogger();
    struct LoggerCapture {
        std::ostream &stream;
        std::streambuf *previous;
        bool enabled = fe::isLoggingEnabled();
        bool dumps   = fe::isLoggingTensorDumpEnabled();
        explicit LoggerCapture(std::ostringstream &output)
            : stream(fe::getStream()), previous(stream.rdbuf(output.rdbuf())) {
            fe::isLoggingEnabled()           = true;
            fe::isLoggingTensorDumpEnabled() = true;
        }
        ~LoggerCapture() {
            stream.rdbuf(previous);
            fe::isLoggingEnabled()           = enabled;
            fe::isLoggingTensorDumpEnabled() = dumps;
        }
    } logging(captured);
    fe::graph::Graph restored;
    REQUIRE(restored.deserialize(resources.handle, blob, false, true).is_good());
    REQUIRE(captured.str().find("Tensor Dump uid:") == std::string::npos);
    REQUIRE(fe::isLoggingTensorDumpEnabled());
    float values[] = {1.0F, 2.0F, 3.0F, 4.0F};
    float bias     = 2.0F;
    REQUIRE(cudaMalloc(&resources.input, sizeof(values)) == cudaSuccess);
    REQUIRE(cudaMalloc(&resources.output, sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(resources.input, values, sizeof(values), cudaMemcpyHostToDevice) == cudaSuccess);
    int64_t workspace_size = 0;
    REQUIRE(restored.get_workspace_size(workspace_size).is_good());
    if (workspace_size) REQUIRE(cudaMalloc(&resources.workspace, workspace_size) == cudaSuccess);
    std::unordered_map<int64_t, void *> pointers{{1, resources.input}, {2, &bias}, {3, resources.output}};
    REQUIRE(restored.execute(resources.handle, pointers, resources.workspace).is_good());
    REQUIRE(captured.str().find("Tensor Dump uid: 2 Name: warmup_raw_scalar") != std::string::npos);
    float actual = 0.0F;
    REQUIRE(cudaMemcpy(&actual, resources.output, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(actual == 18.0F);
}

TEST_CASE("Tensor attributes", "[tensor][serialize]") {
    namespace fe = cudnn_frontend;

    auto tensor_attributes = fe::graph::Tensor_attributes()
                                 .set_name("image")
                                 .set_dim({4, 32, 16, 16})
                                 .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                                 .set_is_virtual(true)
                                 .set_is_pass_by_value(true)
                                 .set_uid(12312)
                                 .set_reordering_type(fe::TensorReordering_t::F16x16)
                                 .set_data_type(fe::DataType_t::HALF);

    json j                              = tensor_attributes;
    auto tensor_attributes_deserialized = j;

    REQUIRE(tensor_attributes_deserialized == tensor_attributes);
}

TEST_CASE("Tensor compile-time constants preserve exact typed values", "[tensor][serialize][compile_time_constant]") {
    namespace fe = cudnn_frontend;

    auto check_round_trip = [](auto const &value, char const *expected_type, char const *expected_raw) {
        using value_t = std::decay_t<decltype(value)>;

        auto tensor_attributes = fe::graph::Tensor_attributes(value, fe::graph::ScalarType::COMPILE_TIME_CONST)
                                     .set_name("compile_time_constant")
                                     .set_uid(12312);

        json j = tensor_attributes;
        REQUIRE(j.at("has_compile_time_constant").get<bool>());
        REQUIRE(j.at("compile_time_constant_value").at("type").get<std::string>() == expected_type);
        REQUIRE(j.at("compile_time_constant_value").at("raw").get<std::string>() == expected_raw);

        auto deserialized = j.get<fe::graph::Tensor_attributes>();
        REQUIRE(deserialized.get_has_compile_time_constant());
        REQUIRE(deserialized.get_is_pass_by_value());
        REQUIRE_FALSE(deserialized.get_pass_by_value().has_value());
        auto restored_constant = deserialized.get_compile_time_constant();
        REQUIRE(restored_constant.has_value());

        auto const &restored = std::get<value_t>(restored_constant.value());
        REQUIRE(std::memcmp(&restored, &value, sizeof(value_t)) == 0);
    };

    int64_t const int64_value = std::numeric_limits<int64_t>::min() + 17;
    check_round_trip(int64_value, "INT64", "8000000000000011");

    int32_t const int32_value = std::numeric_limits<int32_t>::min() + 17;
    check_round_trip(int32_value, "INT32", "80000011");

    uint16_t const half_bits = 0x7E01;
    __half_raw half_raw{};
    half_raw.x = half_bits;
    half half_value{};
    half_value = half_raw;
    check_round_trip(half_value, "HALF", "7E01");

    uint32_t const float_bits = 0x80000000;
    float float_value;
    std::memcpy(&float_value, &float_bits, sizeof(float_value));
    check_round_trip(float_value, "FLOAT", "80000000");

    uint64_t const double_bits = 0x7FF8000000000001ULL;
    double double_value;
    std::memcpy(&double_value, &double_bits, sizeof(double_value));
    check_round_trip(double_value, "DOUBLE", "7FF8000000000001");

    uint16_t const bfloat16_bits = 0x7FC1;
    __nv_bfloat16_raw bfloat16_raw{};
    bfloat16_raw.x = bfloat16_bits;
    nv_bfloat16 bfloat16_value{};
    bfloat16_value = bfloat16_raw;
    check_round_trip(bfloat16_value, "BFLOAT16", "7FC1");
}

TEST_CASE("Tensor compile-time constant JSON remains backward compatible", "[tensor][serialize]") {
    namespace fe = cudnn_frontend;

    auto tensor_attributes = fe::graph::Tensor_attributes()
                                 .set_name("legacy_tensor")
                                 .set_dim({1})
                                 .set_stride({1})
                                 .set_uid(12312)
                                 .set_data_type(fe::DataType_t::FLOAT);

    json j = tensor_attributes;
    REQUIRE_FALSE(j.contains("has_compile_time_constant"));
    REQUIRE_FALSE(j.contains("compile_time_constant_value"));

    auto deserialized = j.get<fe::graph::Tensor_attributes>();
    REQUIRE_FALSE(deserialized.get_has_compile_time_constant());
    REQUIRE_FALSE(deserialized.get_compile_time_constant().has_value());

    auto reused_runtime = fe::graph::Tensor_attributes(2.0f);
    from_json(j, reused_runtime);
    REQUIRE_FALSE(reused_runtime.get_is_pass_by_value());
    REQUIRE_FALSE(reused_runtime.get_pass_by_value().has_value());

    auto reused_constant = fe::graph::Tensor_attributes(2.0f, fe::graph::ScalarType::COMPILE_TIME_CONST);
    from_json(j, reused_constant);
    REQUIRE_FALSE(reused_constant.get_has_compile_time_constant());
    REQUIRE_FALSE(reused_constant.get_compile_time_constant().has_value());
}

TEST_CASE("Tensor compile-time constant JSON rejects malformed state", "[tensor][serialize][compile_time_constant]") {
    namespace fe = cudnn_frontend;

    json valid = fe::graph::Tensor_attributes(1.0f, fe::graph::ScalarType::COMPILE_TIME_CONST);

    SECTION("wrong raw width") {
        valid["compile_time_constant_value"]["raw"] = "000000";
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }

    SECTION("non-hexadecimal raw value") {
        valid["compile_time_constant_value"]["raw"] = "3F80000Z";
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }

    SECTION("unknown type tag") {
        valid["compile_time_constant_value"]["type"] = "UNKNOWN";
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }

    SECTION("compile-time constant is not pass by value") {
        valid["is_pass_by_value"] = false;
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }

    SECTION("compile-time and runtime values are both present") {
        valid["pass_by_value"] = {{"index", 3}, {"value", "3F800000"}};
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }

    SECTION("value is present while compile-time flag is false") {
        valid["has_compile_time_constant"] = false;
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }

    SECTION("constant flag is not a boolean") {
        valid["has_compile_time_constant"] = 1;
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }

    SECTION("constant flag is true but the value is absent") {
        valid.erase("compile_time_constant_value");
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }

    SECTION("constant value is not an object with string fields") {
        valid["compile_time_constant_value"] = {{"type", "FLOAT"}};
        REQUIRE_THROWS_AS(valid.get<fe::graph::Tensor_attributes>(), std::invalid_argument);
    }
}

TEST_CASE(
    "Tensor compile-time constant with a mismatched data type serializes "
    "and validates",
    "[graph][serialize][compile_time_constant][key]") {
    namespace fe = cudnn_frontend;

    // The develop public setters can make this state: a FLOAT constant on an INT32 tensor.
    auto mismatched = fe::graph::Tensor_attributes(1.0f, fe::graph::ScalarType::COMPILE_TIME_CONST)
                          .set_data_type(fe::DataType_t::INT32);
    json tensor_json;
    REQUIRE_NOTHROW(tensor_json = mismatched);
    REQUIRE(tensor_json.at("compile_time_constant_value").at("type").get<std::string>() == "FLOAT");
    fe::graph::Tensor_attributes restored_tensor;
    REQUIRE_NOTHROW(restored_tensor = tensor_json.get<fe::graph::Tensor_attributes>());
    REQUIRE(restored_tensor.get_data_type() == fe::DataType_t::INT32);
    REQUIRE(std::get<float>(restored_tensor.get_compile_time_constant().value()) == 1.0f);

    fe::graph::Graph graph;
    auto x = graph.tensor(fe::graph::Tensor_attributes().set_name("x").set_dim({1, 4}).set_stride({4, 1}).set_data_type(
        fe::DataType_t::INT32));
    auto constant = graph.tensor(1.0f, fe::graph::ScalarType::COMPILE_TIME_CONST);
    constant->set_name("constant").set_data_type(fe::DataType_t::INT32).set_dim({1, 1}).set_stride({1, 1});
    auto y = graph.pointwise(x,
                             constant,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::INT32));
    y->set_name("y").set_output(true).set_data_type(fe::DataType_t::INT32);

    REQUIRE_NOTHROW(graph.key());
    json graph_json;
    REQUIRE_NOTHROW(graph_json = graph);
    std::vector<uint8_t> data;
    fe::error_t serialize_status;
    REQUIRE_NOTHROW(serialize_status = graph.serialize(data));
    REQUIRE(serialize_status.get_code() == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED);

    REQUIRE(graph.validate().is_good());
    fe::graph::Graph restored_graph;
    REQUIRE(restored_graph.deserialize(graph_json).is_good());
}

TEST_CASE(
    "Graph deserialization returns an error for a malformed compile-time "
    "constant",
    "[graph][serialize][compile_time_constant]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    auto x = graph.tensor(fe::graph::Tensor_attributes().set_name("x").set_dim({1, 4}).set_stride({4, 1}).set_data_type(
        fe::DataType_t::FLOAT));
    auto constant = graph.tensor(2.0f, fe::graph::ScalarType::COMPILE_TIME_CONST);
    constant->set_name("constant").set_dim({1, 1}).set_stride({1, 1});
    auto y = graph.pointwise(x,
                             constant,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    y->set_name("y").set_output(true).set_data_type(fe::DataType_t::FLOAT);

    json serialized = graph;
    bool corrupted{false};
    for (auto &tensor : serialized.at("tensors")) {
        if (tensor.value("name", "") == "constant") {
            tensor["compile_time_constant_value"]["raw"] = "3F80000Z";
            corrupted                                    = true;
        }
    }
    REQUIRE(corrupted);

    SECTION("graph structure") {
        fe::graph::Graph restored;
        fe::error_t status;
        REQUIRE_NOTHROW(status = restored.deserialize(serialized));
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("non-hexadecimal") != std::string::npos);
    }

    SECTION("execution plan with graph structure") {
        fe::graph::Graph restored;
        fe::error_t status;
        REQUIRE_NOTHROW(status = restored.deserialize(static_cast<cudnnHandle_t>(nullptr), serialized));
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("non-hexadecimal") != std::string::npos);
    }
}

TEST_CASE("Graph serialization preserves signed-zero compile-time constants",
          "[graph][serialize][compile_time_constant][key]") {
    namespace fe = cudnn_frontend;

    auto populate_graph = [](fe::graph::Graph &graph, float constant_value) {
        auto x =
            graph.tensor(fe::graph::Tensor_attributes().set_name("x").set_dim({1, 4}).set_stride({4, 1}).set_data_type(
                fe::DataType_t::FLOAT));

        auto constant = graph.tensor(constant_value, fe::graph::ScalarType::COMPILE_TIME_CONST);
        constant->set_name("constant").set_dim({1, 1}).set_stride({1, 1});

        auto y = graph.pointwise(x,
                                 constant,
                                 fe::graph::Pointwise_attributes()
                                     .set_name("add")
                                     .set_mode(fe::PointwiseMode_t::ADD)
                                     .set_compute_data_type(fe::DataType_t::FLOAT));
        y->set_name("y").set_output(true).set_data_type(fe::DataType_t::FLOAT);
    };

    float const positive_zero         = 0.0f;
    uint32_t const negative_zero_bits = 0x80000000;
    float negative_zero;
    std::memcpy(&negative_zero, &negative_zero_bits, sizeof(negative_zero));

    fe::graph::Graph positive_graph;
    populate_graph(positive_graph, positive_zero);

    fe::graph::Graph negative_graph;
    populate_graph(negative_graph, negative_zero);

    REQUIRE(positive_graph.key() != negative_graph.key());

    json serialized = negative_graph;
    bool found_constant{false};
    for (auto const &tensor : serialized.at("tensors")) {
        if (tensor.value("name", "") == "constant") {
            found_constant = true;
            REQUIRE(tensor.at("has_compile_time_constant").get<bool>());
            REQUIRE(tensor.at("compile_time_constant_value").at("type").get<std::string>() == "FLOAT");
            REQUIRE(tensor.at("compile_time_constant_value").at("raw").get<std::string>() == "80000000");
        }
    }
    REQUIRE(found_constant);

    fe::graph::Graph restored;
    REQUIRE(restored.deserialize(serialized).is_good());

    json restored_json = restored;
    REQUIRE(restored_json == serialized);
    REQUIRE(restored.key() == negative_graph.key());
}

TEST_CASE("Tensor attributes alignment", "[tensor][serialize]") {
    namespace fe = cudnn_frontend;

    auto make_tensor = []() -> fe::graph::Tensor_attributes {
        return fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({4, 32, 16, 16})
            .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
            .set_uid(12312)
            .set_data_type(fe::DataType_t::HALF);
    };

    SECTION("a non-default alignment survives the round trip") {
        auto tensor_attributes = make_tensor().set_alignment(4);

        json j = tensor_attributes;
        REQUIRE(j.contains("alignment"));
        REQUIRE(j["alignment"].get<int64_t>() == 4);

        auto deserialized = j.get<fe::graph::Tensor_attributes>();
        REQUIRE(deserialized.get_alignment() == 4);
    }

    SECTION("the default alignment is not emitted") {
        auto tensor_attributes = make_tensor();
        REQUIRE(tensor_attributes.get_alignment() == fe::graph::Tensor_attributes::default_alignment);

        json j = tensor_attributes;
        REQUIRE_FALSE(j.contains("alignment"));

        auto deserialized = j.get<fe::graph::Tensor_attributes>();
        REQUIRE(deserialized.get_alignment() == fe::graph::Tensor_attributes::default_alignment);
    }

    SECTION("a payload with no alignment key deserializes to the default") {
        json j = make_tensor().set_alignment(4);
        REQUIRE(j.erase("alignment") == 1);

        fe::graph::Tensor_attributes deserialized;
        REQUIRE_NOTHROW(deserialized = j.get<fe::graph::Tensor_attributes>());
        REQUIRE(deserialized.get_alignment() == fe::graph::Tensor_attributes::default_alignment);
    }
}

TEST_CASE("conv graph serialization preserves alignment", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;

    auto x = graph.tensor(fe::graph::Tensor_attributes());
    x->set_name("image")
        .set_dim({4, 32, 16, 16})
        .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
        .set_data_type(fe::DataType_t::HALF)
        .set_alignment(4);

    auto w = graph.tensor(fe::graph::Tensor_attributes());
    w->set_name("weight")
        .set_dim({64, 32, 3, 3})
        .set_stride({32 * 3 * 3, 1, 32 * 3, 32})
        .set_data_type(fe::DataType_t::HALF);

    auto conv_fprop_attributes = fe::graph::Conv_fprop_attributes()
                                     .set_name("conv_fprop")
                                     .set_padding({1, 1})
                                     .set_stride({1, 1})
                                     .set_dilation({1, 1})
                                     .set_compute_data_type(fe::DataType_t::FLOAT);

    auto y = graph.conv_fprop(x, w, conv_fprop_attributes);
    y->set_name("output").set_output(true).set_data_type(fe::DataType_t::HALF);

    auto alignment_of = [](json const &serialized, std::string const &name) -> int64_t {
        for (auto const &tensor : serialized["tensors"]) {
            if (tensor.contains("name") && tensor["name"].get<std::string>() == name) {
                return tensor.value("alignment", fe::graph::Tensor_attributes::default_alignment);
            }
        }
        return -1;
    };

    json j = graph;

    REQUIRE(alignment_of(j, "image") == 4);
    REQUIRE(alignment_of(j, "weight") == fe::graph::Tensor_attributes::default_alignment);

    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(j).is_good());

    json j2 = graph_deserialized;

    REQUIRE(alignment_of(j2, "image") == 4);
    REQUIRE(j == j2);
}

TEST_CASE("Context serialization", "[context][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;

    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_sm_count(24);

    json j = graph;

    std::cout << j << std::endl;

    fe::graph::Graph graph_deserialized;

    REQUIRE(graph_deserialized.deserialize(j).is_good());

    json j2 = graph_deserialized;

    REQUIRE(j == j2);

    auto status = fe::graph::Graph().deserialize(j, true);
    REQUIRE(status.is_bad());
    REQUIRE(status.get_message().find("enforce_precompiled") != std::string::npos);

    REQUIRE(graph.validate().is_good());
}

TEST_CASE("Conv fprop attributes", "[conv_fprop][serialize]") {
    namespace fe = cudnn_frontend;

    auto x = std::make_shared<fe::graph::Tensor_attributes>();
    x->set_name("image")
        .set_dim({4, 32, 16, 16})
        .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
        .set_is_virtual(true)
        .set_is_pass_by_value(true)
        .set_uid(12312)
        .set_reordering_type(fe::TensorReordering_t::F16x16)
        .set_data_type(fe::DataType_t::HALF);

    auto conv_fprop_attributes = fe::graph::Conv_fprop_attributes()
                                     .set_name("conv_fprop")
                                     .set_padding({1, 1})
                                     .set_stride({1, 1})
                                     .set_dilation({1, 1})
                                     .set_compute_data_type(fe::DataType_t::FLOAT);

    json j                                  = conv_fprop_attributes;
    auto conv_fprop_attributes_deserialized = j;

    REQUIRE(conv_fprop_attributes_deserialized == conv_fprop_attributes);
}

TEST_CASE("Graph key", "[graph][key]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(
        fe::graph::Tensor_attributes().set_name("image").set_dim({4, 16, 64}).set_stride({16 * 64, 1, 16}));
    auto Y = graph.tensor(
        fe::graph::Tensor_attributes().set_name("filter").set_dim({4, 64, 32}).set_stride({32 * 64, 1, 64}));

    fe::graph::Matmul_attributes matmul;
    auto Z = graph.matmul(X, Y, matmul);

    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto S             = graph.tensor(
        fe::graph::Tensor_attributes().set_name("scale").set_dim({4, 16, 32}).set_stride({16 * 32, 32, 1}));
    auto scale_output = graph.pointwise(Z, S, scale_options);

    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto B =
        graph.tensor(fe::graph::Tensor_attributes().set_name("bias").set_dim({4, 16, 32}).set_stride({16 * 32, 32, 1}));
    auto bias_output = graph.pointwise(scale_output, B, bias_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto O            = graph.pointwise(bias_output, relu_options);
    O->set_output(true);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());
    auto key = graph.key();

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(key == graph.key());

    REQUIRE(graph.check_support().is_good());
    REQUIRE(key == graph.key());

    REQUIRE(graph.build_plans().is_good());
    REQUIRE(key == graph.key());

    cudnnDestroy(handle);
}

TEST_CASE("Graph key dynamic shape", "[graph][key][dynamic_shape]") {
    namespace fe = cudnn_frontend;
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    // clang-format off
    struct {
        int64_t b,    m,    n,    k;
    } shapes[] = {
        {       4,   16,   32,   64},
        {       8,   32,   64,  128},
    };
    // clang-format on

    constexpr int shapes_count = sizeof(shapes) / sizeof(shapes[0]);
    size_t key                 = 0;  // Save key between runs to verify that dim and stride
                                     // information is deleted

    for (int idx_shape = 0; idx_shape < shapes_count; idx_shape++) {
        auto b = shapes[idx_shape].b;
        auto m = shapes[idx_shape].m;
        auto n = shapes[idx_shape].n;
        auto k = shapes[idx_shape].k;

        fe::graph::Graph graph;
        graph.set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_dynamic_shape_enabled(true);

        auto X =
            graph.tensor(fe::graph::Tensor_attributes().set_name("image").set_dim({b, m, k}).set_stride({m * k, 1, m}));
        auto Y = graph.tensor(
            fe::graph::Tensor_attributes().set_name("filter").set_dim({b, k, n}).set_stride({n * k, 1, k}));

        fe::graph::Matmul_attributes matmul;
        auto Z = graph.matmul(X, Y, matmul);

        auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
        auto S =
            graph.tensor(fe::graph::Tensor_attributes().set_name("scale").set_dim({b, m, n}).set_stride({m * n, n, 1}));
        auto scale_output = graph.pointwise(Z, S, scale_options);

        auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
        auto B =
            graph.tensor(fe::graph::Tensor_attributes().set_name("bias").set_dim({b, m, n}).set_stride({m * n, n, 1}));
        auto bias_output = graph.pointwise(scale_output, B, bias_options);

        auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
        auto O            = graph.pointwise(bias_output, relu_options);
        O->set_output(true);

        cudnnHandle_t handle;
        cudnnCreate(&handle);

        auto status = graph.validate();
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Dynamic shapes not supported pre 9.4");
        }

        REQUIRE(graph.build_operation_graph(handle).is_good());

        if (!key) {
            key = graph.key();
        }

        REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
        REQUIRE(key == graph.key());

        REQUIRE(graph.check_support().is_good());
        REQUIRE(key == graph.key());

        REQUIRE(graph.build_plans().is_good());
        REQUIRE(key == graph.key());

        cudnnDestroy(handle);
    }
}

TEST_CASE("Matmul fp8 fusion", "[graph][serialize]") {
    namespace fe = cudnn_frontend;
    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::FP8_E4M3);
    auto A = graph.tensor(A_attributes);

    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, 1, k})
                            .set_data_type(fe::DataType_t::FP8_E4M3);
    auto B = graph.tensor(B_attributes);

    auto A_descale_attributes = fe::graph::Tensor_attributes()
                                    .set_name("descale0")
                                    .set_dim({1, 1, 1})
                                    .set_stride({1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT);
    auto B_descale_attributes = fe::graph::Tensor_attributes()
                                    .set_name("descale1")
                                    .set_dim({1, 1, 1})
                                    .set_stride({1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT);

    auto A_descale = graph.tensor(A_descale_attributes);
    auto B_descale = graph.tensor(B_descale_attributes);

    auto matmul_attributes =
        // fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_data_type(fe::DataType_t::FLOAT);

    // Add scale_A operation
    auto pw_0_attributes = fe::graph::Pointwise_attributes()
                               //    .set_name("pw0_Mul")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto C_after_pw_0 = graph.pointwise(C, A_descale, pw_0_attributes);
    C_after_pw_0->set_data_type(fe::DataType_t::FLOAT);

    // Add descale_B operation
    auto pw_1_attributes = fe::graph::Pointwise_attributes()
                               //    .set_name("pw1_Mul")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto C_after_pw_1 = graph.pointwise(C_after_pw_0, B_descale, pw_1_attributes);
    C_after_pw_1->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    json j = graph;

    std::cout << j << std::endl;

    fe::graph::Graph graph_deserialized;

    REQUIRE(graph_deserialized.deserialize(j).is_good());

    json j2 = graph_deserialized;

    REQUIRE(j == j2);

    REQUIRE(graph.validate().is_good());

    std::cout << "Validating deserialized graph" << std::endl;

    cudnnHandle_t handle;  // Handle to use during deserialize and execute

    cudnnCreate(&handle);

    REQUIRE(graph_deserialized.validate().is_good());

    REQUIRE(graph_deserialized.build_operation_graph(handle).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("conv graph serialization", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;

    auto x = graph.tensor(fe::graph::Tensor_attributes());
    x->set_name("image")
        .set_dim({4, 32, 16, 16})
        .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto w = graph.tensor(fe::graph::Tensor_attributes());
    w->set_name("weight")
        .set_dim({64, 32, 3, 3})
        .set_stride({32 * 3 * 3, 1, 32 * 3, 32})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto conv_fprop_attributes = fe::graph::Conv_fprop_attributes()
                                     .set_name("conv_fprop")
                                     .set_padding({1, 1})
                                     .set_stride({1, 1})
                                     .set_dilation({1, 1})
                                     .set_compute_data_type(fe::DataType_t::FLOAT);

    auto y = graph.conv_fprop(x, w, conv_fprop_attributes);

    auto b = graph.tensor(fe::graph::Tensor_attributes());
    b->set_name("bias")
        .set_dim({1, 64, 1, 1})
        .set_stride({64, 1, 64, 64})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto pointwise_attributes = fe::graph::Pointwise_attributes()
                                    .set_name("bias")
                                    .set_mode(fe::PointwiseMode_t::ADD)
                                    .set_compute_data_type(fe::DataType_t::FLOAT);

    auto o = graph.pointwise(y, b, pointwise_attributes);

    auto reduction_attributes = fe::graph::Reduction_attributes()
                                    .set_name("reduction")
                                    .set_mode(fe::ReductionMode_t::ADD)
                                    .set_compute_data_type(fe::DataType_t::FLOAT);
    auto r = graph.reduction(o, reduction_attributes);

    r->set_output(true).set_data_type(fe::DataType_t::HALF);

    json j = graph;

    fe::graph::Graph graph_deserialized;

    REQUIRE(graph_deserialized.deserialize(j).is_good());

    json j2 = graph_deserialized;

    REQUIRE(j == j2);

    REQUIRE(graph_deserialized.validate().is_good());
}

TEST_CASE("sdpa graph serialization", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t b    = 3;     // batch size
    int64_t h    = 4;     // head dim
    int64_t s_q  = 1024;  // q tensor is padded to this seq length
    int64_t s_kv = 1024;  // k and v tensor is padded to this seq length
    int64_t d    = 128;   // hidden dim

    auto Q = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("Q")
                              .set_dim({b, h, s_q, d})
                              .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
    auto K = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("K")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
    auto V = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("V")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));

    auto attn_scale = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("attn_scale")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_is_pass_by_value(true)
                                       .set_data_type(fe::DataType_t::FLOAT));

    auto seed   = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("Seed")
                                 .set_dim({1, 1, 1, 1})
                                 .set_stride({1, 1, 1, 1})
                                 .set_data_type(fe::DataType_t::INT32));
    auto offset = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("Offset")
                                   .set_dim({1, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(fe::DataType_t::INT32));

    auto bias = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({b, 1, s_q, s_kv})
                                 .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));

    auto seq_q  = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("seq_q")
                                  .set_dim({b, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(fe::DataType_t::INT32));
    auto seq_kv = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("seq_kv")
                                   .set_dim({b, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(fe::DataType_t::INT32));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_generate_stats(true)
                            .set_attn_scale(attn_scale)
                            .set_alibi_mask(true)
                            .set_diagonal_band_right_bound(0)
                            .set_dropout(0.1f, seed, offset)
                            .set_bias(bias)
                            .set_padding_mask(true)
                            .set_seq_len_q(seq_q)
                            .set_seq_len_kv(seq_kv);

    auto [O, stats] = graph.sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * d, d, b * h * d, 1});
    stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    json j = graph;

    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(j).is_good());
    json j2 = graph_deserialized;

    REQUIRE(j == j2);
}

TEST_CASE("sdpa backward graph serialization", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    int64_t b    = 3;     // batch size
    int64_t h    = 4;     // head dim
    int64_t s_q  = 1024;  // q tensor is padded to this seq length
    int64_t s_kv = 1024;  // k and v tensor is padded to this seq length
    int64_t d    = 128;   // hidden dim

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    std::shared_ptr<fe::graph::Tensor_attributes> bias, dropout_seed, dropout_offset;

    auto q = graph.tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto k = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("K")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({h * s_kv * d, s_kv * d, d, 1}));
    auto v = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("V")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({h * s_kv * d, s_kv * d, d, 1}));
    auto o = graph.tensor(
        fe::graph::Tensor_attributes().set_name("O").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto dO = graph.tensor(
        fe::graph::Tensor_attributes().set_name("dO").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto stats = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("stats")
                                  .set_dim({b, h, s_q, 1})
                                  .set_stride({h * s_q, s_q, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));

    auto attn_scale = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("attn_scale")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_is_pass_by_value(true)
                                       .set_data_type(fe::DataType_t::FLOAT));

    bias = graph.tensor(fe::graph::Tensor_attributes()
                            .set_name("bias")
                            .set_dim({b, 1, s_q, s_kv})
                            .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));

    dropout_seed   = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("Seed")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
    dropout_offset = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Offset")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));

    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                     .set_name("flash_attention_backward")
                                     .set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
                                     .set_diagonal_band_right_bound(0)
                                     .set_attn_scale(attn_scale)
                                     .set_bias(bias)
                                     .set_dropout(0.1f, dropout_seed, dropout_offset);

    auto [dQ, dK, dV] = graph.sdpa_backward(q, k, v, o, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1});
    dK->set_output(true).set_dim({b, h, s_kv, d}).set_stride({h * s_kv * d, s_kv * d, d, 1});
    dV->set_output(true).set_dim({b, h, s_kv, d}).set_stride({h * s_kv * d, s_kv * d, d, 1});

    json j = graph;
    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(j).is_good());
    json j2 = graph_deserialized;

    REQUIRE(j == j2);

    REQUIRE(graph_deserialized.validate().is_good());
}

// Round-trips a plan through binary serialize/deserialize and confirms the variant
// pack template is populated on the deserialized graph. The check verifies that
// deserialize(handle, ...) eagerly prepares the template (instead of waiting for
// first execute).
TEST_CASE("Plan deserialize prepares variant pack template", "[graph][serialize][deserialize]") {
    namespace fe = cudnn_frontend;

    constexpr int64_t a_uid = 1, b_uid = 2, c_uid = 3;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph.tensor(
        fe::graph::Tensor_attributes().set_name("A").set_dim({4, 16, 64}).set_stride({16 * 64, 64, 1}).set_uid(a_uid));
    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("B").set_dim({4, 64, 32}).set_stride({64 * 32, 32, 1}).set_uid(b_uid));

    auto C = graph.matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true).set_uid(c_uid);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    std::vector<uint8_t> serialized_data;
    REQUIRE(graph.serialize(serialized_data).is_good());

    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(handle, serialized_data, true).is_good());

    // Variant pack template should already be populated; no execute needed.
    auto const user_uids = graph_deserialized.get_variant_pack_uids_sorted();
    REQUIRE(user_uids == std::vector<int64_t>{a_uid, b_uid, c_uid});

    cudnnDestroy(handle);
}

// Exercises the run_warmup=false fast path of deserialize(handle, ...). Skipping
// the throwaway warmup capture must not change the resulting plan: the variant
// pack template is built by prepare_variant_pack_template(), which is independent
// of warmup, so the deserialized graph is still fully usable.
//
// Note the additive argument order: deserialize(handle, data, enforce_precompiled, run_warmup).
// Both bools are spelled explicitly below so the positional meaning is unambiguous.
TEST_CASE("Plan deserialize with run_warmup=false still prepares template", "[graph][serialize][deserialize]") {
    namespace fe = cudnn_frontend;

    constexpr int64_t a_uid = 1, b_uid = 2, c_uid = 3;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph.tensor(
        fe::graph::Tensor_attributes().set_name("A").set_dim({4, 16, 64}).set_stride({16 * 64, 64, 1}).set_uid(a_uid));
    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("B").set_dim({4, 64, 32}).set_stride({64 * 32, 32, 1}).set_uid(b_uid));

    auto C = graph.matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true).set_uid(c_uid);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    // serialize the graph
    std::vector<uint8_t> serialized_data;
    REQUIRE(graph.serialize(serialized_data).is_good());

    // expected uids
    std::vector<int64_t> const expected_uids{a_uid, b_uid, c_uid};

    // test the blob overload, run_warmup=false.
    // Check for success and variant pack uids are the same as the expected uids
    SECTION("blob overload, run_warmup=false") {
        fe::graph::Graph graph_deserialized;
        REQUIRE(graph_deserialized
                    .deserialize(handle,
                                 serialized_data,
                                 /*enforce_precompiled=*/false,
                                 /*run_warmup=*/false)
                    .is_good());
        // check the variant pack uids are the same as the expected uids
        REQUIRE(graph_deserialized.get_variant_pack_uids_sorted() == expected_uids);
        // tensor metadata must still resolve with warmup skipped (deserialized_tensor_properties)
        fe::graph::Tensor_attributes queried;
        REQUIRE(graph_deserialized.query_tensor_attributes_of_uid(a_uid, queried).is_good());
    }

    // test the json overload, run_warmup=false.
    // same assertion via the pre-parsed-json overload, covering the path that avoids a second from_ubjson.
    SECTION("json overload, run_warmup=false") {
        json const j = json::from_ubjson(serialized_data);
        fe::graph::Graph graph_deserialized;
        auto const status =
            graph_deserialized.deserialize(handle, j, /*enforce_precompiled=*/false, /*run_warmup=*/false);
        REQUIRE(status.is_good());
        REQUIRE(graph_deserialized.get_variant_pack_uids_sorted() == expected_uids);
    }

    // test the run_warmup=false matches the default warmup=true template
    // Confirms warmup is purely a priming step with no effect on the executable plan.
    SECTION("run_warmup=false matches the default warmup=true template") {
        fe::graph::Graph warmed;
        REQUIRE(warmed
                    .deserialize(handle,
                                 serialized_data,
                                 /*enforce_precompiled=*/false,
                                 /*run_warmup=*/true)
                    .is_good());

        fe::graph::Graph skipped;
        auto const skipped_status = skipped.deserialize(handle,
                                                        serialized_data,
                                                        /*enforce_precompiled=*/false,
                                                        /*run_warmup=*/false);
        REQUIRE(skipped_status.is_good());

        // The plan-side state the fast execute path relies on is identical either way.
        REQUIRE(warmed.get_variant_pack_uids_sorted() == skipped.get_variant_pack_uids_sorted());
        REQUIRE(skipped.get_variant_pack_uids_sorted() == expected_uids);
    }

    cudnnDestroy(handle);
}

// serialize(data, serialize_structure=false) omits the graph structure
// (nodes/tensors) while keeping the plan reloadable via deserialize(handle, ...).
// The default (true) still emits the structure, so existing callers are unaffected.
TEST_CASE("serialize_structure flag controls structural payload", "[graph][serialize][deserialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph.tensor(
        fe::graph::Tensor_attributes().set_name("A").set_dim({4, 16, 64}).set_stride({16 * 64, 64, 1}).set_uid(1));
    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("B").set_dim({4, 64, 32}).set_stride({64 * 32, 32, 1}).set_uid(2));
    auto C = graph.matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true).set_uid(3);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    std::vector<uint8_t> with_structure, without_structure;
    REQUIRE(graph.serialize(with_structure).is_good());  // default: serialize_structure=true
    REQUIRE(graph.serialize(without_structure, /*serialize_structure=*/false).is_good());

    json const j_with    = json::from_ubjson(with_structure);
    json const j_without = json::from_ubjson(without_structure);

    // Default emits the structure; opting out drops it and shrinks the blob.
    REQUIRE(j_with.contains("nodes"));
    REQUIRE_FALSE(j_without.contains("nodes"));
    REQUIRE(without_structure.size() < with_structure.size());

    // Both remain reloadable through the plan path with identical variant packs.
    auto const expected_uids = graph.get_variant_pack_uids_sorted();
    for (auto const &blob : {with_structure, without_structure}) {
        fe::graph::Graph reloaded;
        REQUIRE(reloaded
                    .deserialize(handle,
                                 blob,
                                 /*enforce_precompiled=*/false,
                                 /*run_warmup=*/false)
                    .is_good());
        REQUIRE(reloaded.get_variant_pack_uids_sorted() == expected_uids);
    }

    cudnnDestroy(handle);
}

// A graph loaded via deserialize(handle, ...) might have no node subtree,
// so serialize() must source its pass-by-value and workspace modifications from the
// cached members instead of walking the (now empty) subtree.
TEST_CASE("Plan re-serialize preserves pass-by-value and workspace modifications", "[graph][serialize][deserialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    constexpr int64_t N = 4;
    auto X              = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({N, N, N})
                              .set_stride({N * N, N, 1})
                              .set_data_type(fe::DataType_t::HALF)
                              .set_uid(1));
    auto scalar         = graph.tensor(5.0f);
    scalar->set_name("scalar").set_uid(2);
    auto Y = graph.pointwise(X,
                             scalar,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    Y->set_output(true).set_data_type(fe::DataType_t::HALF).set_uid(3);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    std::vector<uint8_t> blob;
    REQUIRE(graph.serialize(blob, /*serialize_structure=*/false).is_good());

    // The fresh graph must actually carry pass-by-value data, else the test is moot.
    json const j_fresh = json::from_ubjson(blob);
    REQUIRE(j_fresh.contains("pass_by_values"));
    REQUIRE_FALSE(j_fresh["pass_by_values"].empty());

    auto const expected_uids = graph.get_variant_pack_uids_sorted();

    fe::graph::Graph reloaded;
    REQUIRE(reloaded
                .deserialize(handle,
                             blob,
                             /*enforce_precompiled=*/false,
                             /*run_warmup=*/false)
                .is_good());
    REQUIRE(reloaded.get_variant_pack_uids_sorted() == expected_uids);

    // Re-serialize the plan-only graph; the cached maps must survive intact.
    std::vector<uint8_t> next;
    REQUIRE(reloaded.serialize(next, /*serialize_structure=*/false).is_good());

    json const j_next = json::from_ubjson(next);
    REQUIRE(j_next["pass_by_values"] == j_fresh["pass_by_values"]);
    REQUIRE(j_next["workspace_modifications"] == j_fresh["workspace_modifications"]);

    cudnnDestroy(handle);
}

// Develop restores a plan whose compile-time constant has a type other than its tensor type. The plan reader must
// accept the state that validate() accepts.
TEST_CASE("Plan round trip keeps a compile-time constant with a mismatched type",
          "[graph][serialize][deserialize][compile_time_constant]") {
    namespace fe = cudnn_frontend;

    auto const check_round_trip = [](fe::DataType_t tensor_type, bool int32_constant) {
        constexpr int64_t N = 1, C = 8, H = 16, W = 16;
        constexpr int64_t count = N * C * H * W;
        constexpr size_t bytes  = static_cast<size_t>(count) * 4;

        fe::graph::Graph graph;
        graph.set_io_data_type(tensor_type).set_compute_data_type(tensor_type).set_intermediate_data_type(tensor_type);
        auto X        = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("X")
                                  .set_dim({N, C, H, W})
                                  .set_stride({C * H * W, 1, C * W, C})
                                  .set_data_type(tensor_type));
        auto constant = int32_constant ? graph.tensor(int32_t{1}, fe::graph::ScalarType::COMPILE_TIME_CONST)
                                       : graph.tensor(1.0f, fe::graph::ScalarType::COMPILE_TIME_CONST);
        constant->set_data_type(tensor_type);
        auto Y = graph.pointwise(
            X,
            constant,
            fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD).set_compute_data_type(tensor_type));
        Y->set_output(true).set_data_type(tensor_type);

        cudnnHandle_t handle;
        REQUIRE(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);
        REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

        // Both tensor types are 4 bytes; the input holds the integers 0 .. count-1 in the tensor type.
        std::vector<uint8_t> input(bytes);
        for (int64_t i = 0; i < count; ++i) {
            if (tensor_type == fe::DataType_t::FLOAT) {
                auto const value = static_cast<float>(i);
                std::memcpy(input.data() + static_cast<size_t>(i) * 4, &value, 4);
            } else {
                auto const value = static_cast<int32_t>(i);
                std::memcpy(input.data() + static_cast<size_t>(i) * 4, &value, 4);
            }
        }
        void *x_dev = nullptr, *y_dev = nullptr, *restored_y_dev = nullptr, *workspace = nullptr;
        REQUIRE(cudaMalloc(&x_dev, bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&y_dev, bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&restored_y_dev, bytes) == cudaSuccess);
        REQUIRE(cudaMemcpy(x_dev, input.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
        // Different fill bytes: equal outputs then show that both executions wrote every element.
        REQUIRE(cudaMemset(y_dev, 0xAB, bytes) == cudaSuccess);
        REQUIRE(cudaMemset(restored_y_dev, 0xCD, bytes) == cudaSuccess);
        int64_t workspace_size = 0;
        REQUIRE(graph.get_workspace_size(workspace_size).is_good());
        if (workspace_size > 0) {
            REQUIRE(cudaMalloc(&workspace, static_cast<size_t>(workspace_size)) == cudaSuccess);
        }

        std::unordered_map<int64_t, void *> variant_pack = {{X->get_uid(), x_dev}, {Y->get_uid(), y_dev}};
        REQUIRE(graph.execute(handle, variant_pack, workspace).is_good());

        std::vector<uint8_t> data;
        REQUIRE(graph.serialize(data).is_good());
        fe::graph::Graph restored;
        auto status = restored.deserialize(handle, data, /*enforce_precompiled=*/true, /*run_warmup=*/false);
        INFO(status.get_message());
        REQUIRE(status.is_good());
        std::unordered_map<int64_t, void *> restored_variant_pack = {{X->get_uid(), x_dev},
                                                                     {Y->get_uid(), restored_y_dev}};
        REQUIRE(restored.execute(handle, restored_variant_pack, workspace).is_good());
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        std::vector<uint8_t> output(bytes), restored_output(bytes);
        REQUIRE(cudaMemcpy(output.data(), y_dev, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(restored_output.data(), restored_y_dev, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(restored_output == output);

        cudaFree(workspace);
        cudaFree(restored_y_dev);
        cudaFree(y_dev);
        cudaFree(x_dev);
        cudnnDestroy(handle);
    };

    SECTION("INT32 constant on a FLOAT tensor") { check_round_trip(fe::DataType_t::FLOAT, true); }
    SECTION("FLOAT constant on an INT32 tensor") { check_round_trip(fe::DataType_t::INT32, false); }
}

// ============================================================
// Helpers shared by handle-less deserialize tests below.
// ============================================================

namespace {
namespace fe = cudnn_frontend;

// Build a small NHWC FP32 conv graph and return (graph, X-uid, W-uid, Y-uid).
// N=1, C=4, H=8, W=8, K=4, R=1, S=1, padding=0, stride=1, dilation=1.
// This shape is simple but exercises the full stack.
struct ConvGraph {
    static constexpr int64_t N = 1, C = 4, H = 8, W = 8, K = 4, R = 1, S = 1;
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> X, W_t, Y;

    ConvGraph() {
        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        // NHWC layout
        X = graph->tensor(
            fe::graph::Tensor_attributes().set_name("X").set_dim({N, C, H, W}).set_stride({C * H * W, 1, C * W, C}));
        W_t = graph->tensor(
            fe::graph::Tensor_attributes().set_name("W").set_dim({K, C, R, S}).set_stride({C * R * S, 1, C * S, C}));
        auto opts = fe::graph::Conv_fprop_attributes().set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});
        Y         = graph->conv_fprop(X, W_t, opts);
        Y->set_output(true);
    }
};

// Build a conv+relu fusion graph for the RTC test.
struct FusionGraph {
    static constexpr int64_t N = 16, C = 128, H = 64, W = 64, K = 256, R = 3, S = 3;
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> X, W_t, Y;

    FusionGraph() {
        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        // NHWC layout
        X = graph->tensor(
            fe::graph::Tensor_attributes().set_name("X").set_dim({N, C, H, W}).set_stride({C * H * W, 1, C * W, C}));
        W_t = graph->tensor(
            fe::graph::Tensor_attributes().set_name("W").set_dim({K, C, R, S}).set_stride({C * R * S, 1, C * S, C}));
        auto conv_opts = fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
        auto conv      = graph->conv_fprop(X, W_t, conv_opts);
        Y = graph->pointwise(conv, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));
        Y->set_output(true).set_data_type(fe::DataType_t::HALF);
    }
};

// Return the deviceVer encoded in a serialized DeviceProperties byte buffer.
// The wire format is a JSON string, e.g. {"deviceVer":890,...}.
// Returns -1 if the key is not found.
static int
parse_device_ver(std::vector<uint8_t> const &buf) {
    std::string s(reinterpret_cast<const char *>(buf.data()), buf.size());
    auto pos = s.find("\"deviceVer\":");
    if (pos == std::string::npos) return -1;
    pos += std::string("\"deviceVer\":").size();
    return std::stoi(s.substr(pos));
}

// Produce a DeviceProperties whose deviceVer has been changed to a different
// SM than the current device. Returns nullptr if the edit fails.
static std::shared_ptr<fe::DeviceProperties>
make_wrong_arch_devprop() {
    auto dp_real = std::make_shared<fe::DeviceProperties>();
    if (dp_real->set_device_id(0).build().is_bad()) return nullptr;

    std::vector<uint8_t> buf;
    if (dp_real->serialize(buf).is_bad()) return nullptr;

    int real_ver = parse_device_ver(buf);
    if (real_ver < 0) return nullptr;

    // Pick a different SM: add 100 (one major step up) if possible, else subtract 100.
    int wrong_ver = (real_ver < 900) ? (real_ver + 100) : (real_ver - 100);

    std::string s(reinterpret_cast<const char *>(buf.data()), buf.size());
    std::string needle = "\"deviceVer\":" + std::to_string(real_ver);
    std::string repl   = "\"deviceVer\":" + std::to_string(wrong_ver);
    auto pos           = s.find(needle);
    if (pos == std::string::npos) return nullptr;
    s.replace(pos, needle.size(), repl);

    std::vector<uint8_t> bad(s.begin(), s.end());
    auto dp_wrong = std::make_shared<fe::DeviceProperties>();
    if (dp_wrong->deserialize(bad).is_bad()) return nullptr;
    return dp_wrong;
}

}  // namespace

// ============================================================
// Compile-only overload resolution (no GPU, no cuDNN handle).
// Verifies that std::vector<uint8_t> selects the new handle-less
// deserialize overload and not the structural-graph json overload.
// ============================================================

TEST_CASE("Handle-less deserialize overload resolution", "[serialize][graph]") {
    namespace fe = cudnn_frontend;
    // This test is purely compile-time: if the wrong overload is selected the
    // assertion message distinguishes plan vs. structural deserialize.
    // We call it on an empty graph that will fail at runtime (no devprop),
    // but the call must COMPILE to the blob overload, not convert to json.
    fe::graph::Graph g;
    std::vector<uint8_t> blob{0, 1, 2};  // intentionally invalid
    auto err = g.deserialize(blob);
    // No devprop set → ATTRIBUTE_NOT_SET, not a json parse error.
    REQUIRE(err.is_bad());
    REQUIRE(err.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
}

// ============================================================
// Negative: no device properties set → clean ATTRIBUTE_NOT_SET error.
// ============================================================

TEST_CASE("Handle-less deserialize with no devprop returns ATTRIBUTE_NOT_SET", "[serialize][graph]") {
#if (CUDNN_VERSION < 90800)
    SKIP("Handle-less plan deserialize requires cuDNN >= 9.8 headers");
#else
    namespace fe = cudnn_frontend;

    fe::graph::Graph g;
    std::vector<uint8_t> dummy(64, 0);
    auto status = g.deserialize(dummy);
    REQUIRE(status.is_bad());
    REQUIRE(status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
    REQUIRE(status.get_message().find("set_device_properties") != std::string::npos);
#endif
}

// ============================================================
// Main runtime tests — gated at 9.11 (matching the deviceless sample).
// ============================================================

TEST_CASE("Handle-less plan deserialize", "[serialize][graph]") {
#if (CUDNN_VERSION < 90800)
    SKIP("Handle-less plan deserialize requires cuDNN >= 9.8 headers");
#else
    if (cudnn_frontend::detail::get_backend_version() < 91100) {
        SKIP("Handle-less plan deserialize runtime tests require cuDNN >= 9.11");
    }

    // ── build a device properties descriptor ──────────────────────────────
    auto dp = std::make_shared<cudnn_frontend::DeviceProperties>();
    REQUIRE(dp->set_device_id(0).build().is_good());

    namespace fe = cudnn_frontend;

    SECTION("wrong-arch devprop causes arch mismatch on deserialize") {
        // This is the primary discriminator: if devprop were ignored,
        // deserialize would succeed regardless of which SM is claimed.
        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
        std::vector<uint8_t> blob;
        REQUIRE(cg.graph->serialize(blob).is_good());

        auto dp_wrong = make_wrong_arch_devprop();
        REQUIRE(dp_wrong != nullptr);

        fe::graph::Graph deser;
        deser.set_device_properties(dp_wrong);
        auto status = deser.deserialize(blob);
        REQUIRE(status.is_bad());
        // Backend returns NOT_SUPPORTED for arch mismatch during plan finalize.
        auto const code  = status.get_code();
        bool is_plan_err = (code == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
        bool is_be_err   = (code == fe::error_code_t::CUDNN_BACKEND_API_FAILED);
        REQUIRE((is_plan_err || is_be_err));
    }

    SECTION("RTC fusion plan carries RUNTIME_COMPILATION behavior note") {
        FusionGraph fg;
        fg.graph->set_device_properties(dp);
        REQUIRE(fg.graph->build_operation_graph().is_good());
        REQUIRE(fg.graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
        fg.graph->select_behavior_notes({fe::BehaviorNote_t::RUNTIME_COMPILATION});
        if (fg.graph->check_support().is_bad()) {
            SKIP(
                "No RUNTIME_COMPILATION engine available for this graph on this "
                "GPU/cuDNN");
        }
        REQUIRE(fg.graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

        std::vector<fe::BehaviorNote_t> notes;
        REQUIRE(fg.graph->get_behavior_notes(notes).is_good());
        bool has_rtc = std::find(notes.begin(), notes.end(), fe::BehaviorNote_t::RUNTIME_COMPILATION) != notes.end();
        REQUIRE(has_rtc);

        std::vector<uint8_t> blob;
        REQUIRE(fg.graph->serialize(blob).is_good());

        fe::graph::Graph deser;
        deser.set_device_properties(dp);
        REQUIRE(deser.deserialize(blob).is_good());
    }

    SECTION("plan-only blob (serialize_structure=false) + handle-less deserialize") {
        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());

        std::vector<uint8_t> blob;
        REQUIRE(cg.graph->serialize(blob, /*serialize_structure=*/false).is_good());

        fe::graph::Graph deser;
        deser.set_device_properties(dp);
        auto status = deser.deserialize(blob);
        REQUIRE(status.is_good());

        int64_t ws = 0;
        REQUIRE(deser.get_workspace_size(ws).is_good());
    }

    SECTION("enforce_precompiled=true rejects blob with no plan data") {
        // enforce_precompiled checks that the blob contains serialized plan bytes
        // (cudnn_backend_data field). Manufacture a blob without that field.
        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());

        std::vector<uint8_t> full_blob;
        REQUIRE(cg.graph->serialize(full_blob).is_good());

        auto j = json::from_ubjson(full_blob);
        j.erase("cudnn_backend_data");
        auto no_plan_blob = json::to_ubjson(j);

        fe::graph::Graph deser;
        deser.set_device_properties(dp);
        auto status = deser.deserialize(no_plan_blob, /*enforce_precompiled=*/true);
        REQUIRE(status.is_bad());
        auto const code  = status.get_code();
        bool is_plan_err = (code == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
        REQUIRE(is_plan_err);
    }

    SECTION("handle overload still uses handle even when devprop is set on graph") {
        // Decision #7: the existing deserialize(handle, blob) overloads forward an
        // explicit device_prop=nullptr so the graph's device_properties is NOT consulted.
        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
        std::vector<uint8_t> blob;
        REQUIRE(cg.graph->serialize(blob).is_good());

        auto dp_wrong = make_wrong_arch_devprop();
        REQUIRE(dp_wrong != nullptr);

        cudnnHandle_t handle;
        REQUIRE(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);

        // Graph has incompatible devprop, but we deserialize via handle overload.
        // Should succeed because handle overload ignores graph's device_properties.
        fe::graph::Graph deser;
        deser.set_device_properties(dp_wrong);  // deliberately incompatible
        auto status = deser.deserialize(handle, blob, false, false);
        REQUIRE(status.is_good());  // handle path used, not devprop

        cudnnDestroy(handle);
    }

    SECTION(
        "concurrency: N threads share one devprop, each deserializes its own "
        "blob") {
        // Concurrent rehydration is the primary motivation for this feature.
        // One shared read-only DeviceProperties descriptor is used by all threads;
        // each thread deserializes into its own Graph.
        constexpr int kThreads = 4;

        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
        std::vector<uint8_t> blob;
        REQUIRE(cg.graph->serialize(blob).is_good());

        std::vector<int> results(kThreads, 0);
        std::vector<std::thread> threads;
        threads.reserve(kThreads);

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&dp, &blob, &results, t]() {
                fe::graph::Graph g;
                g.set_device_properties(dp);  // shared descriptor
                results[t] = g.deserialize(blob).is_good() ? 1 : 0;
            });
        }
        for (auto &th : threads) th.join();

        for (int t = 0; t < kThreads; ++t) {
            REQUIRE(results[t] == 1);
        }
    }
#endif  // CUDNN_VERSION >= 90800
}

// serialize() must return a clear error (not an OOB read or silent UB) when
// no execution plan has been selected.  get_name_at_index OOB must error, not
// crash.
TEST_CASE("Graph::serialize rejects unbuilt candidate; get_name_at_index rejects OOB",
          "[serialize][serialize_validate]") {
    if (cudnnGetVersion() < 90400) {
        SKIP("requires cuDNN >= 9.4");
    }

    namespace fe = cudnn_frontend;
    ConvGraph cg;

    cudnnHandle_t handle;
    cudnnCreate(&handle);
    REQUIRE(cg.graph->build_operation_graph(handle).is_good());
    REQUIRE(cg.graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(cg.graph->check_support().is_good());

    // candidate == -1 (build_plans not yet called): serialize must fail with a
    // diagnostic mentioning the candidate value, not silently write a partial blob.
    {
        std::vector<uint8_t> data;
        auto err = cg.graph->serialize(data);
        REQUIRE(err.is_bad());
        REQUIRE(err.get_message().find("candidate") != std::string::npos);
        REQUIRE(data.empty());
    }

    REQUIRE(cg.graph->build_plans().is_good());

    // After a successful build, serialize must succeed.
    {
        std::vector<uint8_t> data;
        REQUIRE(cg.graph->serialize(data).is_good());
        REQUIRE(!data.empty());
    }

    // get_name_at_index with out-of-range index must return an error, not crash.
    {
        std::string name;
        auto const count = cg.graph->get_execution_plan_count();
        REQUIRE(cg.graph->get_plan_name_at_index(-1, name).is_bad());
        REQUIRE(cg.graph->get_plan_name_at_index(count, name).is_bad());
    }

    cudnnDestroy(handle);
}
