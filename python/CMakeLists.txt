# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.26)

Include(FetchContent)

# Fetch and build dlpack
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_VERSION_MINIMUM 3.18)
set(BUILD_MOCK OFF)

option(CUDNN_FRONTEND_USE_SYSTEM_DLPACK "Whether dlpack should use the system version or fetched." OFF)

IF(CUDNN_FRONTEND_USE_SYSTEM_DLPACK)
  # 1.3 is where DLPackExchangeAPI is declared, which pygraph/variant_pack.cpp
  # needs. The wire structs are byte-identical to 1.1 -- sizeof and every
  # offsetof of DLTensor and DLManagedTensor match -- so this is a
  # compile-time requirement only; capsules exchanged with a consumer built
  # against 1.1 are unaffected.
  find_package(dlpack 1.3 REQUIRED)
  if(dlpack_FOUND)
    message(STATUS "Found system dlpack ${dlpack_VERSION}")
  else()
    message(FATAL_ERROR "dlpack >= 1.3 not found (needed for DLPackExchangeAPI); unset CUDNN_FRONTEND_USE_SYSTEM_DLPACK to fetch it")
  endif()
else()

  # Read dlpack version from dlpack_version.txt
  file(READ ${PROJECT_SOURCE_DIR}/dlpack_version.txt DLPACK_VERSION)
  string(STRIP "${DLPACK_VERSION}" DLPACK_VERSION)

  FetchContent_Declare(
    dlpack
    GIT_REPOSITORY https://github.com/dmlc/dlpack
    GIT_TAG        v${DLPACK_VERSION}
  )
  FetchContent_MakeAvailable(dlpack)
endif()

# Find python
option(CUDNN_FRONTEND_PYTHON_STABLE_ABI "Require the stable ABI for Python bindings." OFF)
if(CUDNN_FRONTEND_PYTHON_STABLE_ABI)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
else()
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
endif()
set(binding_options)
if(CUDNN_FRONTEND_PYTHON_STABLE_ABI)
    set(binding_options STABLE_ABI)
endif()

# Find cudnn
include(${PROJECT_SOURCE_DIR}/cmake/cuDNN.cmake)

option(CUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE "Whether cmake build system should fetch pybinds." ON)

if(CUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v3.0.1
    )
    FetchContent_MakeAvailable(nanobind)
else()
    find_package(nanobind CONFIG REQUIRED)
endif()

nanobind_add_module(
    _compiled_module
    ${binding_options}

    pycudnn.cpp
    gnn.cpp
    properties.cpp

    pygraph/pygraph.cpp
    pygraph/norm.cpp
    pygraph/sdpa.cpp
    pygraph/pointwise.cpp
    pygraph/variant_pack.cpp
    pygraph/sdpa_thd_binding.cpp
    pygraph/sdpa_dense_binding.cpp
    pygraph/sdpa_bwd_binding.cpp
    pygraph/sdpa_sm80_binding.cpp
    pygraph/sdpa_sm80_thd_binding.cpp

)

get_target_property(binding_libraries _compiled_module LINK_LIBRARIES)
set(binding_runtime nanobind-static)
if("nanobind-static-abi3" IN_LIST binding_libraries)
    set(binding_runtime nanobind-static-abi3)
elseif(CUDNN_FRONTEND_PYTHON_STABLE_ABI)
    # nanobind may silently fall back, but ON requires an abi3 extension.
    message(FATAL_ERROR "Stable ABI was requested, but nanobind did not enable it for this Python interpreter.")
endif()

target_compile_features(_compiled_module PRIVATE cxx_std_20)

# The bindings and nanobind runtime require RTTI even when
# CUDNN_FRONTEND_ENABLE_RTTI is off for the rest of the build.
if(MSVC)
    target_compile_options(_compiled_module PRIVATE /GR)
    # Suppress shadowing warnings only for the nanobind runtime.
    target_compile_options(${binding_runtime} PRIVATE /GR /wd4457 /wd4459)
else()
    target_compile_options(_compiled_module PRIVATE -frtti)
    target_compile_options(${binding_runtime} PRIVATE -frtti)
endif()

target_include_directories(
    _compiled_module
    PRIVATE $<TARGET_PROPERTY:cudnn_frontend,INTERFACE_INCLUDE_DIRECTORIES>
    PRIVATE $<TARGET_PROPERTY:CUDNN::cudnn_all,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(_compiled_module PRIVATE NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)

if(CUDNN_FRONTEND_USE_SYSTEM_DLPACK)
    target_link_libraries(
        _compiled_module

        PRIVATE dlpack::dlpack
    )
else()
    target_link_libraries(
        _compiled_module

        PRIVATE dlpack
    )
endif()

if(MSVC)
    target_compile_options(_compiled_module PRIVATE /wd4127)
    target_compile_options(_compiled_module PRIVATE /wd4244)
    target_compile_options(_compiled_module PRIVATE /wd4458)
    target_compile_options(_compiled_module PRIVATE /wd4505)
    # nanobind's cast wrappers may exceed MSVC's forceinline budget.
    target_compile_options(_compiled_module PRIVATE /wd4714)
    target_compile_options(_compiled_module PRIVATE /bigobj)
else()
    add_compile_options(-fvisibility=hidden)
endif()

set_target_properties(
    _compiled_module

    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed"
    LINK_FLAGS "-Wl,--enable-new-dtags"
    LINK_FLAGS "-Wl,-rpath,'$ORIGIN',-rpath,'$ORIGIN/../lib',-rpath,'$ORIGIN/../nvidia/cudnn/lib'"
    LINK_WHAT_YOU_USE TRUE
)

# using python bindings directly with cmake build system is not supported
# Temparorily use below parameter
option(CUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR "Whether pybinds should be kept inside build/cudnn directory." ON)
if(CUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR)
set_target_properties(
        _compiled_module

        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cudnn
    )
    file(COPY ${PROJECT_SOURCE_DIR}/python/cudnn DESTINATION ${PROJECT_BINARY_DIR})
endif()
