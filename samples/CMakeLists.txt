cmake_minimum_required(VERSION 3.18)

find_package(Threads)

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)

    # Fetch and build catch2
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.3.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/cuDNN.cmake)

if(DEFINED ENV{NO_DEFAULT_IN_SWITCH})
    message("Default case in the switch is disabled")
    add_compile_definitions(NO_DEFAULT_IN_SWITCH)
endif()

# Add subdirectories for samples and legacy_samples
add_subdirectory(cpp)
add_subdirectory(legacy_samples)