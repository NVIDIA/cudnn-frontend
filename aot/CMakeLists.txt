# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ahead-of-time compiled kernels (opt-in: -DCUDNN_FRONTEND_BUILD_AOT_ATTENTION=ON). See docs/operations/attention/aot_attention.md.
#
# libcudnn_frontend_aot_attention.so is produced by aot/attention/build.py, which needs a Python with this checkout importable
# (e.g. `pip install -e .`, which brings the CuTeDSL wheel). Point CMake at it with -DPython3_EXECUTABLE=<python>.
# No GPU is needed to build. The test needs a 12.x GPU (or 8.x for the quantized-KV part only); the benchmark a 12.x GPU.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CUDNN_FRONTEND_AOT_ATTENTION_JOBS "" CACHE STRING "Parallel export jobs for aot/attention/build.py (empty: its default)")
set(AOT_ATTENTION_OUT ${CMAKE_CURRENT_BINARY_DIR}/attention)
set(AOT_ATTENTION_LIB ${AOT_ATTENTION_OUT}/lib/libcudnn_frontend_aot_attention.so)
set(AOT_ATTENTION_JOBS_ARG)
if(NOT CUDNN_FRONTEND_AOT_ATTENTION_JOBS STREQUAL "")
    set(AOT_ATTENTION_JOBS_ARG "--jobs=${CUDNN_FRONTEND_AOT_ATTENTION_JOBS}")
endif()

file(GLOB AOT_ATTENTION_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/attention/*.py ${CMAKE_CURRENT_SOURCE_DIR}/attention/src/*)
file(GLOB_RECURSE AOT_ATTENTION_TEMPLATES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/python/cudnn/*.py)

add_custom_command(
    OUTPUT ${AOT_ATTENTION_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${PROJECT_SOURCE_DIR}/python" ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/attention/build.py --out ${AOT_ATTENTION_OUT} --cuda ${CUDAToolkit_LIBRARY_ROOT}
            --cxx ${CMAKE_CXX_COMPILER} --reuse ${AOT_ATTENTION_JOBS_ARG}
    DEPENDS ${AOT_ATTENTION_SOURCES} ${AOT_ATTENTION_TEMPLATES} ${PROJECT_SOURCE_DIR}/include/cudnn_frontend/experimental/aot_attention_abi.h
    COMMENT "Exporting the SM120 FROST kernels ahead of time and linking libcudnn_frontend_aot_attention.so"
    USES_TERMINAL
    VERBATIM
)
add_custom_target(cudnn_frontend_aot_attention ALL DEPENDS ${AOT_ATTENTION_LIB})

# The test's reference kernels are CUDA; the root's warning flags are host-compiler flags, so route them per language.
enable_language(CUDA)
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>"
)

add_executable(test_aot_attention attention/test/test_aot_attention.cu)
set_target_properties(test_aot_attention PROPERTIES CUDA_ARCHITECTURES "80;86;89;120" CUDA_STANDARD 17)
target_include_directories(test_aot_attention PRIVATE ${PROJECT_SOURCE_DIR}/include attention/src attention/test)
target_link_libraries(test_aot_attention PRIVATE CUDA::cudart cudnn_frontend_aot_loader)
add_dependencies(test_aot_attention cudnn_frontend_aot_attention)

add_executable(bench_aot_attention attention/test/bench_aot_attention.cpp)
target_include_directories(bench_aot_attention PRIVATE ${PROJECT_SOURCE_DIR}/include attention/src attention/test)
target_link_libraries(bench_aot_attention PRIVATE CUDA::cudart cudnn_frontend_aot_loader)
target_compile_features(bench_aot_attention PRIVATE cxx_std_17)
add_dependencies(bench_aot_attention cudnn_frontend_aot_attention)

# Both find the library through --lib or the loader path.
message(STATUS "AOT SM120: run ${CMAKE_CURRENT_BINARY_DIR}/test_aot_attention --lib ${AOT_ATTENTION_LIB}")
