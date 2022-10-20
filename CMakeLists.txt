cmake_minimum_required(VERSION 3.17)

project(cudnn_frontend VERSION 0.7)

add_library(cudnn_frontend INTERFACE)

target_include_directories(
    cudnn_frontend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(cudnn_frontend INTERFACE cxx_std_11)

add_subdirectory(samples)
