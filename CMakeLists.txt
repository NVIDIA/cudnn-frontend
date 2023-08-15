cmake_minimum_required(VERSION 3.17)

project(cudnn_frontend VERSION 1.0.0)

option(CUDNN_FRONTEND_BUILD_SAMPLES "Defines if samples are built or not." ON)

if(MSVC OR MSYS OR MINGW)
    option(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS "Defines if python bindings are built or not." OFF)
else()
    option(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS "Defines if python bindings are built or not." ON)
endif()

add_library(cudnn_frontend INTERFACE)

target_include_directories(
    cudnn_frontend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(cudnn_frontend INTERFACE cxx_std_17)

if (CUDNN_FRONTEND_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

if (CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()
