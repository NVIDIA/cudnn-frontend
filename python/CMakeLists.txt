# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Fetch and build dlpack
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_VERSION_MINIMUM 3.18)
set(BUILD_MOCK OFF)

option(CUDNN_FRONTEND_USE_SYSTEM_DLPACK "Whether dlpack should use the system version or fetched." OFF)

IF(CUDNN_FRONTEND_USE_SYSTEM_DLPACK)
  find_package(dlpack REQUIRED)
  if(dlpack_FOUND)
    message(STATUS "Found system dlpack")
  else()
    message(FATAL_ERROR "dlpack not found")
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
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Find cudnn
include(${PROJECT_SOURCE_DIR}/cmake/cuDNN.cmake)

# The GNN direct-call API is newer than the first cudnn_gnn_ops.h revisions,
# so probe for the declarations themselves instead of relying on a cuDNN
# version number or header-presence check.
include(CheckCXXSourceCompiles)
set(_CUDNN_FRONTEND_SAVED_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
set(_CUDNN_FRONTEND_SAVED_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_INCLUDES ${CUDNN_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
check_cxx_source_compiles(
    "#include <cudnn.h>
     int main() {
       auto forward = &cudnnGnnAggSimpleForward;
       auto backward = &cudnnGnnAggSimpleBackward;
       (void)forward;
       (void)backward;
       return 0;
     }"
    CUDNN_FRONTEND_HAS_GNN_AGG_SIMPLE
)
set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_CUDNN_FRONTEND_SAVED_TRY_COMPILE_TARGET_TYPE})
set(CMAKE_REQUIRED_INCLUDES ${_CUDNN_FRONTEND_SAVED_REQUIRED_INCLUDES})

option(CUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE "Whether cmake build system should fetch pybinds." ON)
set(PYBIND11_FINDPYTHON ON)


if(CUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
else()
    find_package(pybind11 CONFIG REQUIRED)
endif()


# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(
    _compiled_module
    
    MODULE
    pycudnn.cpp
    gnn.cpp
    properties.cpp

    pygraph/pygraph.cpp
    pygraph/norm.cpp
    pygraph/sdpa.cpp
    pygraph/pointwise.cpp

    WITH_SOABI
)
target_link_libraries(_compiled_module PRIVATE pybind11::headers)

target_compile_features(_compiled_module PRIVATE cxx_std_20)

# pybind11's type registry is typeid-based, so this module requires RTTI even when
# CUDNN_FRONTEND_ENABLE_RTTI is off for the rest of the build.
if(MSVC)
    target_compile_options(_compiled_module PRIVATE /GR)
else()
    target_compile_options(_compiled_module PRIVATE -frtti)
endif()

target_include_directories(
    _compiled_module
    PRIVATE $<TARGET_PROPERTY:cudnn_frontend,INTERFACE_INCLUDE_DIRECTORIES>
    PRIVATE $<TARGET_PROPERTY:CUDNN::cudnn_all,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(_compiled_module PRIVATE NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
if(CUDNN_FRONTEND_HAS_GNN_AGG_SIMPLE)
    target_compile_definitions(_compiled_module PRIVATE CUDNN_FRONTEND_HAS_GNN_AGG_SIMPLE=1)
endif()

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
