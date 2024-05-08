cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Fetch and build dlpack
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

find_package(dlpack)
if(NOT dlpack_FOUND)
    set(BUILD_MOCK OFF)
    FetchContent_Declare(
      dlpack
      GIT_REPOSITORY https://github.com/dmlc/dlpack
      GIT_TAG        v0.8
    )
    FetchContent_MakeAvailable(dlpack)
endif()

# Find python
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_package(cudnn_frontend REQUIRED)
# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(
    _compiled_module
    
    MODULE
    pycudnn.cpp
    properties.cpp

    pygraph/pygraph.cpp
    pygraph/norm.cpp
    pygraph/sdpa.cpp
    pygraph/pointwise.cpp

    WITH_SOABI
)

target_link_libraries(
    _compiled_module
    PRIVATE
    pybind11::headers
    dlpack::dlpack
    cudnn_frontend::cudnn_frontend
)

target_compile_features(_compiled_module PRIVATE cxx_std_17)

target_compile_definitions(_compiled_module PRIVATE NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)


set_target_properties(
    _compiled_module

    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed"
    LINK_FLAGS "-Wl,--enable-new-dtags"
    LINK_FLAGS "-Wl,-rpath,'$ORIGIN',-rpath,'$ORIGIN/../lib',-rpath,'$ORIGIN/../nvidia/cudnn/lib'"
    LINK_WHAT_YOU_USE TRUE
)

install(
    TARGETS _compiled_module
    LIBRARY DESTINATION python/cudnn
)
