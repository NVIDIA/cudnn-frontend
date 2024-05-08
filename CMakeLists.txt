cmake_minimum_required(VERSION 3.17)

project(cudnn_frontend VERSION 1.4.0)

option(CUDNN_FRONTEND_SKIP_NLOHMANN_JSON "Defines whether FE should not include nlohmann/json.hpp." OFF)
option(CUDNN_FRONTEND_BUILD_SAMPLES "Defines if samples are built or not." ON)
option(CUDNN_FRONTEND_BUILD_UNIT_TESTS "Defines if unittests are built or not." ON)

if(MSVC OR MSYS OR MINGW)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-error=attributes -Wno-attributes -Wno-error=unused-function -Wno-unused-function)
endif()

add_library(cudnn_frontend INTERFACE)

target_compile_definitions(
    cudnn_frontend INTERFACE
    $<$<BOOL:${CUDNN_FRONTEND_SKIP_NLOHMANN_JSON}>:CUDNN_FRONTEND_SKIP_NLOHMANN_JSON>
)

target_include_directories(
    cudnn_frontend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Find the cuda compiler
find_package(CUDAToolkit)
# Find cudnn
include(${CMAKE_SOURCE_DIR}/cmake/cuDNN.cmake)

target_link_libraries(
    cudnn_frontend INTERFACE
    CUDNN::cudnn_all
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
    enable_testing()
    add_subdirectory(test)
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

# Export the target
install(TARGETS cudnn_frontend EXPORT cudnn_frontendTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create the cmake directory
install(EXPORT cudnn_frontendTargets
    FILE cudnn_frontendTargets.cmake
    NAMESPACE cudnn_frontend::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudnn_frontend
)

# Create the package configuration file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cudnn_frontend/cudnn_frontendConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cudnn_frontend/cudnn_frontendConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudnn_frontend
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cudnn_frontend/cudnn_frontendConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cudnn_frontend/cudnn_frontendConfigVersion.cmake"
    "${CMAKE_SOURCE_DIR}/cmake/cuDNN.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudnn_frontend
)

