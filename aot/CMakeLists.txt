# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ahead-of-time compiled kernels (opt-in: -DCUDNN_FRONTEND_BUILD_AOT_SM120=ON). See docs/fe-oss-apis/aot_sm120_attention.md.
#
# libcudnn_frontend_aot_sm120.so is produced by aot/sm120/build.py, which needs a Python with this checkout importable
# (e.g. `pip install -e .`, which brings the CuTeDSL wheel). Point CMake at it with -DPython3_EXECUTABLE=<python>.
# No GPU is needed to build; the test and benchmark need an sm_120 GPU and skip without one.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CUDNN_FRONTEND_AOT_SM120_JOBS "" CACHE STRING "Parallel export jobs for aot/sm120/build.py (empty: its default)")
set(AOT_SM120_OUT ${CMAKE_CURRENT_BINARY_DIR}/sm120)
set(AOT_SM120_LIB ${AOT_SM120_OUT}/lib/libcudnn_frontend_aot_sm120.so)

file(GLOB AOT_SM120_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/sm120/*.py ${CMAKE_CURRENT_SOURCE_DIR}/sm120/src/*)
file(GLOB_RECURSE AOT_SM120_TEMPLATES ${PROJECT_SOURCE_DIR}/python/cudnn/sdpa/fwd/*.py)

add_custom_command(
    OUTPUT ${AOT_SM120_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${PROJECT_SOURCE_DIR}/python" ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/sm120/build.py --out ${AOT_SM120_OUT} --cuda ${CUDAToolkit_LIBRARY_ROOT}
            --cxx ${CMAKE_CXX_COMPILER} --reuse $<$<BOOL:${CUDNN_FRONTEND_AOT_SM120_JOBS}>:--jobs=${CUDNN_FRONTEND_AOT_SM120_JOBS}>
    DEPENDS ${AOT_SM120_SOURCES} ${AOT_SM120_TEMPLATES} ${PROJECT_SOURCE_DIR}/include/cudnn_frontend/experimental/sm120_aot_attention_abi.h
    COMMENT "Exporting the SM120 FROST kernels ahead of time and linking libcudnn_frontend_aot_sm120.so"
    USES_TERMINAL
    VERBATIM
)
add_custom_target(cudnn_frontend_aot_sm120 ALL DEPENDS ${AOT_SM120_LIB})

# The test's reference kernels are CUDA; the root's warning flags are host-compiler flags, so route them per language.
enable_language(CUDA)
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>"
)

add_executable(test_sdpa_sm120 sm120/test/test_sdpa_sm120.cu)
set_target_properties(test_sdpa_sm120 PROPERTIES CUDA_ARCHITECTURES 120 CUDA_STANDARD 17)
target_include_directories(test_sdpa_sm120 PRIVATE ${PROJECT_SOURCE_DIR}/include sm120/src sm120/test)
target_link_libraries(test_sdpa_sm120 PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
add_dependencies(test_sdpa_sm120 cudnn_frontend_aot_sm120)

add_executable(bench_sdpa_sm120 sm120/test/bench_sdpa_sm120.cpp)
target_include_directories(bench_sdpa_sm120 PRIVATE ${PROJECT_SOURCE_DIR}/include sm120/src sm120/test)
target_link_libraries(bench_sdpa_sm120 PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
target_compile_features(bench_sdpa_sm120 PRIVATE cxx_std_17)
add_dependencies(bench_sdpa_sm120 cudnn_frontend_aot_sm120)

# Both find the library through --lib, $CUDNN_FRONTEND_AOT_SM120_LIB or the loader path.
message(STATUS "AOT SM120: run ${CMAKE_CURRENT_BINARY_DIR}/test_sdpa_sm120 --lib ${AOT_SM120_LIB}")
