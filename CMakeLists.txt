cmake_minimum_required(VERSION 3.17)

project(cudnn_frontend VERSION 1.1.2)

option(CUDNN_FRONTEND_BUILD_SAMPLES "Defines if samples are built or not." ON)
option(CUDNN_FRONTEND_BUILD_UNIT_TESTS "Defines if unittests are built or not." OFF)

if(MSVC OR MSYS OR MINGW)
    option(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS "Defines if python bindings are built or not." OFF)
    add_compile_options(/W4 /WX)
else()
    option(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS "Defines if python bindings are built or not." ON)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-error=attributes -Wno-attributes -Wno-error=unused-function -Wno-unused-function)
endif()

add_library(cudnn_frontend INTERFACE)

target_include_directories(
    cudnn_frontend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Find the cuda compiler
find_package(CUDAToolkit)

# Find cudnn
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuDNN.cmake)

target_link_libraries(
    cudnn_frontend INTERFACE

    CUDA::cudart
    CUDA::nvrtc
    
    # cuDNN dlopen's its libraries
    # Add all libraries in link line as NEEDED
    # This forces the executable itself to find all cudnn sublibraries initially
    CUDNN::cudnn_all
)

target_compile_features(cudnn_frontend INTERFACE cxx_std_17)

if (CUDNN_FRONTEND_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

if (CUDNN_FRONTEND_BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

if (CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()
