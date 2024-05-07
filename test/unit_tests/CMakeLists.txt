cmake_minimum_required(VERSION 3.18)

include(CTest)

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


SET(CUDNN_FRONTEND_UNIT_TESTS
    serialize.cpp
    validate.cpp
    version.cpp
)

if (MSVC)
    SET(platform_compile_options
        /W4 /WX # warning level 4 and all warnings as errors
        /wd4100 # allow unused parameters
        /wd4458 # local hides class member (currently a problem for all inline setters)
        /wd4505 # unreferenced function with internal linkage has been removed
        /wd4101 /wd4189 # unreferenced local
        /bigobj # increase number of sections in .Obj file
    )
else()
    SET(platform_compile_options
        -Wall
        -Wextra
        -Werror
        -Wno-unused-function
    )
endif()

foreach (test_source ${CUDNN_FRONTEND_UNIT_TESTS})
    get_filename_component (test_name ${test_source} NAME_WE)

    add_executable(
        ${test_name}
        ${test_source}
    )

    target_compile_options(
        ${test_name}
        PRIVATE
        ${platform_compile_options}
    )

    target_link_libraries(
        ${test_name}
        PRIVATE

        cudnn_frontend
        _cudnn_frontend_pch
        Catch2::Catch2WithMain
    )

    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
    )

endforeach ()