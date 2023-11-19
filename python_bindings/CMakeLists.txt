cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Fetch and build dlpack
FetchContent_Declare(
  dlpack
  GIT_REPOSITORY https://github.com/dmlc/dlpack
  GIT_TAG        v0.8
)
FetchContent_MakeAvailable(dlpack)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.4
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(
    cudnn
    
    pycudnn.cpp
    properties.cpp
    
    pygraph/pygraph.cpp
    pygraph/norm.cpp
    pygraph/sdpa.cpp
    pygraph/pointwise.cpp
)

target_link_libraries(
    cudnn
    
    PRIVATE cudnn_frontend
    PRIVATE dlpack
)

set_target_properties(
    cudnn
    PROPERTIES LINK_FLAGS "-Wl,--no-as-needed"
    LINK_WHAT_YOU_USE TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib64
)
