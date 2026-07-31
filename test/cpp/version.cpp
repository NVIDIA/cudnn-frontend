/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

TEST_CASE("version checks", "[version]") {
    namespace fe = cudnn_frontend;

    REQUIRE(fe::detail::convert_version_to_str(8907) == "8.9.7");
    REQUIRE(fe::detail::convert_version_to_str(90000) == "9.0.0");
    REQUIRE(fe::detail::convert_version_to_str(90100) == "9.1.0");
    REQUIRE(fe::detail::convert_version_to_str(123456) == "12.34.56");
}