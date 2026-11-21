# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Find Catch2 (may already be available from parent project, or fetch it)
find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.3.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Find cuDNN
include(${PROJECT_SOURCE_DIR}/cmake/cuDNN.cmake)

add_executable(
    custom_scale_op

    my_scale_op.cpp
    test_list.cpp
    helpers.cpp
)

add_executable(
    custom_scale_op_ut

    test/ut_tests.cpp
    helpers.cpp
)

if(MSVC)
    target_compile_options(custom_scale_op PRIVATE /W4 /WX /wd4100 /wd4458 /wd4505 /wd4101 /wd4189 /bigobj)
    target_compile_options(custom_scale_op_ut PRIVATE /W4 /WX /wd4100 /wd4458 /wd4505 /wd4101 /wd4189 /bigobj)
else()
    target_compile_options(custom_scale_op PRIVATE -Wall -Wextra -Werror -Wno-unused-function)
    target_compile_options(custom_scale_op_ut PRIVATE -Wall -Wextra -Werror -Wno-unused-function)
endif()

target_link_libraries(
    custom_scale_op PRIVATE
    Threads::Threads
    Catch2::Catch2WithMain
    cudnn_frontend
    _cudnn_frontend_pch
    CUDNN::cudnn
    CUDA::cublasLt
    CUDA::cudart
    CUDA::nvrtc
)

target_link_libraries(
    custom_scale_op_ut PRIVATE
    Threads::Threads
    Catch2::Catch2WithMain
    cudnn_frontend
    _cudnn_frontend_pch
    CUDNN::cudnn
    CUDA::cublasLt
    CUDA::cudart
    CUDA::nvrtc
)

set_target_properties(custom_scale_op PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

set_target_properties(custom_scale_op_ut PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
