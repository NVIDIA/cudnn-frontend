cmake_minimum_required(VERSION 3.23)
project(cudnn_frontend_doc_sample LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
if(CUDNN_FRONTEND_SOURCE)
    set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CUDNN_FRONTEND_SOURCE} cudnn_frontend-source EXCLUDE_FROM_ALL)
else()
    find_package(cudnn_frontend CONFIG REQUIRED)
endif()
find_library(CUDNN_LIBRARY NAMES cudnn REQUIRED)

add_executable(cudnn_fe_matmul ../cpp/matmul.cc)
target_compile_features(cudnn_fe_matmul PRIVATE cxx_std_17)
target_link_libraries(
    cudnn_fe_matmul
    PRIVATE cudnn_frontend ${CUDNN_LIBRARY} CUDA::cudart CUDA::nvrtc
)
