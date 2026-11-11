/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn.h>
#include <cudnn_frontend_utils.h>

namespace fe = cudnn_frontend;

TEST_CASE("Full-precision numerical note serialization", "[numerical_notes][serialize]") {
    json encoded = fe::NumericalNote_t::FULL_PRECISION_MATH;
    REQUIRE(encoded == "FULL_PRECISION_MATH");
    REQUIRE(json::parse(encoded.dump()).get<fe::NumericalNote_t>() == fe::NumericalNote_t::FULL_PRECISION_MATH);
}

TEST_CASE("Full-precision numerical note backend conversion", "[numerical_notes]") {
    cudnnBackendNumericalNote_t backend_note = CUDNN_NUMERICAL_NOTE_TENSOR_CORE;
    auto status = fe::detail::convert_to_cudnn_type(fe::NumericalNote_t::FULL_PRECISION_MATH, backend_note);
#if defined(CUDNN_HAS_FULL_PRECISION_MATH_NUMERICAL_NOTE)
    REQUIRE(status == CUDNN_STATUS_SUCCESS);
    REQUIRE(backend_note == CUDNN_NUMERICAL_NOTE_FULL_PRECISION_MATH);
    REQUIRE(fe::detail::convert_from_cudnn_type(backend_note) == fe::NumericalNote_t::FULL_PRECISION_MATH);
    REQUIRE(fe::to_string(backend_note) == "CUDNN_NUMERICAL_NOTE_FULL_PRECISION_MATH");
#else
    REQUIRE(status == CUDNN_STATUS_INVALID_VALUE);
    REQUIRE(backend_note == CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
#endif
}
