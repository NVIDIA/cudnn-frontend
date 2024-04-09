cmake_minimum_required(VERSION 3.18)

find_package(Catch2 QUIET)

if(NOT Catch2_FOUND)
    Include(FetchContent)

    # Fetch and build catch2
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.3.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Find cudnn
include(${CMAKE_SOURCE_DIR}/cmake/cuDNN.cmake)

add_executable(
    unit_tests

    serialize.cpp
    validate.cpp
    version.cpp
)

if (MSVC)
    target_compile_options(
        unit_tests PRIVATE
        /W4 /WX # warning level 3 and all warnings as errors
        /wd4100 # allow unused parameters
        /wd4458 # local hides class member (currently a problem for all inline setters)
        /wd4505 # unreferenced function with internal linkage has been removed
        /wd4101 /wd4189 # unreferenced local
        /bigobj # increase number of sections in .Obj file
    )
else()
    target_compile_options(
        unit_tests PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wno-unused-function
    )
endif()

target_link_libraries(
    unit_tests
    cudnn_frontend
    _cudnn_frontend_pch
    Catch2::Catch2WithMain

    CUDNN::cudnn_all
)

# cuDNN dlopen's its libraries
# Add all libraries in link line as NEEDED
set_target_properties(
    unit_tests
    PROPERTIES
    LINK_WHAT_YOU_USE TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
