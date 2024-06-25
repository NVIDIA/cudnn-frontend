cmake_minimum_required(VERSION 3.17)

project(cudnn_frontend VERSION 1.5.2)

option(CUDNN_FRONTEND_SKIP_JSON_LIB "Defines whether FE should not include nlohmann/json.hpp." OFF)
option(CUDNN_FRONTEND_BUILD_SAMPLES "Defines if samples are built or not." ON)
option(CUDNN_FRONTEND_BUILD_UNIT_TESTS "Defines if unittests are built or not." ON)

if(MSVC OR MSYS OR MINGW)
    option(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS "Defines if python bindings are built or not." OFF)
    add_compile_options(/W4 /WX)
else()
    option(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS "Defines if python bindings are built or not." OFF)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-error=attributes -Wno-attributes -Wno-error=unused-function -Wno-unused-function)
endif()

add_library(cudnn_frontend INTERFACE)

target_compile_definitions(
    cudnn_frontend INTERFACE
    $<$<BOOL:${CUDNN_FRONTEND_SKIP_JSON_LIB}>:CUDNN_FRONTEND_SKIP_JSON_LIB>
)

target_include_directories(
    cudnn_frontend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Find the cuda compiler
find_package(CUDAToolkit)

target_link_libraries(
    cudnn_frontend INTERFACE

    CUDA::cudart
    CUDA::nvrtc
)

target_compile_features(cudnn_frontend INTERFACE cxx_std_17)

# Make PCH for targets to link against
add_library(_cudnn_frontend_pch INTERFACE)
target_precompile_headers(_cudnn_frontend_pch INTERFACE ${PROJECT_SOURCE_DIR}/include/cudnn_frontend.h)

if (CUDNN_FRONTEND_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

if (CUDNN_FRONTEND_BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

if (CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
