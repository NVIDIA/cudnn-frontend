cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Find the cuda compiler
find_package(CUDAToolkit)

# Fetch and build catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(Catch2)

# Find cudnn
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/cuDNN.cmake)

add_executable(
    samples
    conv_sample.cpp
    resnet_test_list.cpp
    resnet_sample.cpp
    test_list.cpp 
    fp16_emu.cpp 
    helpers.cpp 
    fusion_sample.cpp 
    fp8_sample.cpp 
    mha_sample.cpp
    norm_samples.cpp
    fused_mha_sample.cpp
    f16_flash_mha_sample.cpp
    fp8_flash_mha_sample.cpp
)

if(DEFINED ENV{NO_DEFAULT_IN_SWITCH})
    message("Default case in the switch is disabled")
    add_compile_definitions(NO_DEFAULT_IN_SWITCH)
endif()

if (MSVC)
    target_compile_options(
        samples PRIVATE
        /W4 /WX # warning level 3 and all warnings as errors
        /wd4100 # allow unused parameters
        /wd4458 # local hides class member (currently a problem for all inline setters)
        /wd4505 # unreferenced function with internal linkage has been removed
        /wd4101 /wd4189 # unreferenced local
    )
else()
    target_compile_options(
        samples PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wno-unused-function
    )
endif()
target_include_directories(samples PRIVATE "resnet_block/include")

target_link_libraries(
    samples
    cudnn_frontend
    Catch2::Catch2WithMain

    CUDA::cudart
    CUDA::cublas
    CUDA::nvrtc
    
    CUDNN::cudnn_all
)

# cuDNN dlopen's its libraries
# Add all libraries in link line as NEEDED
set_target_properties(
    samples
    PROPERTIES
    LINK_WHAT_YOU_USE TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
