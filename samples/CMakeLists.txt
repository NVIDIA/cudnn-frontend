cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Fetch and build catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.2
)
FetchContent_MakeAvailable(Catch2)

add_executable(
    samples
    
    cpp/mha.cpp
    cpp/convolutions.cpp
    cpp/dgrads.cpp
    cpp/matmuls.cpp
    cpp/batchnorm.cpp
    cpp/layernorm.cpp
    cpp/rmsnorm.cpp
    cpp/wgrads.cpp
    cpp/serialization.cpp
    cpp/pointwise.cpp

    legacy_samples/conv_sample.cpp 
    legacy_samples/resnet_test_list.cpp
    legacy_samples/resnet_sample.cpp
    legacy_samples/test_list.cpp 
    legacy_samples/fp16_emu.cpp 
    legacy_samples/helpers.cpp 
    legacy_samples/fusion_sample.cpp 
    legacy_samples/fp8_sample.cpp 
    legacy_samples/norm_samples.cpp
    legacy_samples/fused_mha_sample.cpp
    legacy_samples/f16_flash_mha_sample.cpp
    legacy_samples/fp8_flash_mha_sample.cpp
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

target_link_libraries(
    samples

    cudnn_frontend
    Catch2::Catch2WithMain
)

# cuDNN dlopen's its libraries
# Add all libraries in link line as NEEDED
set_target_properties(
    samples
    PROPERTIES
    LINK_WHAT_YOU_USE TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_source_files_properties(
    legacy_samples/resnet_test_list.cpp legacy_samples/resnet_sample.cpp
    PROPERTIES 
    INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/samples/legacy_samples/resnet_block/include"
)