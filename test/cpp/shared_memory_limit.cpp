/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

namespace {

struct ForwardingFlow {
    char const *name;
    cudnnBackendAttributeName_t attribute;
};

struct ForwardingCase {
    char const *name;
    int64_t requested_limit;
    size_t backend_version;
    bool expect_call;
    int32_t expected_limit;
};

}  // namespace

TEST_CASE("Shared-memory limits are forwarded consistently", "[graph][shared_memory_limit]") {
    // Forced-engine creation and knob queries both use create_engine(), and therefore
    // share the engine-descriptor path below. Heuristic queries use the other path.
    // Use the numeric values so this forwarding helper test also compiles with pre-9.27 headers.
    constexpr auto heuristic_attribute = static_cast<cudnnBackendAttributeName_t>(205);
    constexpr auto engine_attribute    = static_cast<cudnnBackendAttributeName_t>(1309);
#if CUDNN_VERSION >= 92700
    static_assert(heuristic_attribute == CUDNN_ATTR_ENGINEHEUR_SHARED_MEMORY_LIMIT);
    static_assert(engine_attribute == CUDNN_ATTR_ENGINE_SHARED_MEMORY_LIMIT);
#endif
    auto const flows = std::array<ForwardingFlow, 2>{
        ForwardingFlow{"heuristic query", heuristic_attribute},
        ForwardingFlow{"forced-engine and knob query", engine_attribute},
    };

    constexpr int64_t positive_limit = 98 * 1024;
    auto const cases                 = std::array<ForwardingCase, 8>{
        ForwardingCase{"unset on old runtime", -1, 92600, false, 0},
        ForwardingCase{"zero on old runtime", 0, 92600, false, 0},
        ForwardingCase{"positive on old runtime", positive_limit, 92600, false, 0},
        ForwardingCase{"oversized on old runtime", std::numeric_limits<int64_t>::max(), 92600, false, 0},
        ForwardingCase{"unset on new runtime", -1, 92700, false, 0},
        ForwardingCase{"zero on new runtime", 0, 92700, false, 0},
        ForwardingCase{"positive on new runtime", positive_limit, 92700, true, static_cast<int32_t>(positive_limit)},
        ForwardingCase{"oversized on new runtime",
                       std::numeric_limits<int64_t>::max(),
                       92700,
                       true,
                       std::numeric_limits<int32_t>::max()},
    };

    for (auto const &flow : flows) {
        for (auto const &test_case : cases) {
            CAPTURE(flow.name, test_case.name);

            int call_count                                 = 0;
            cudnnBackendAttributeName_t recorded_attribute = static_cast<cudnnBackendAttributeName_t>(0);
            cudnnBackendAttributeType_t recorded_type      = static_cast<cudnnBackendAttributeType_t>(0);
            int64_t recorded_count                         = 0;
            int32_t recorded_limit                         = 0;

            auto recorder = [&](cudnnBackendDescriptor_t,
                                cudnnBackendAttributeName_t attribute,
                                cudnnBackendAttributeType_t type,
                                int64_t count,
                                void const *value) {
                call_count++;
                recorded_attribute = attribute;
                recorded_type      = type;
                recorded_count     = count;
                recorded_limit     = *static_cast<int32_t const *>(value);
                return CUDNN_STATUS_SUCCESS;
            };

            auto const status = cudnn_frontend::detail::set_shared_memory_limit_if_supported(
                nullptr, flow.attribute, test_case.requested_limit, test_case.backend_version, recorder);

            REQUIRE(status == CUDNN_STATUS_SUCCESS);
            REQUIRE(call_count == (test_case.expect_call ? 1 : 0));
            if (test_case.expect_call) {
                REQUIRE(recorded_attribute == flow.attribute);
                REQUIRE(recorded_type == CUDNN_TYPE_INT32);
                REQUIRE(recorded_count == 1);
                REQUIRE(recorded_limit == test_case.expected_limit);
            }
        }
    }
}

TEST_CASE("Execution plan list distinguishes unset and explicit shared-memory limits", "[graph][shared_memory_limit]") {
    cudnn_frontend::graph::Execution_plan_list plans;

    REQUIRE(plans.get_max_shared_mem_allowed() == -1);
    plans.set_max_shared_mem_allowed(0);
    REQUIRE(plans.get_max_shared_mem_allowed() == 0);
    plans.set_max_shared_mem_allowed(98 * 1024);
    REQUIRE(plans.get_max_shared_mem_allowed() == 98 * 1024);
}
