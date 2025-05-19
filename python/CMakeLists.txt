cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Fetch and build dlpack
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_VERSION_MINIMUM 3.18)
set(BUILD_MOCK OFF)

option(CUDNN_FRONTEND_USE_SYSTEM_DLPACK "Whether dlpack should use the system version or fetched." OFF)

IF(CUDNN_FRONTEND_USE_SYSTEM_DLPACK)
  find_package(dlpack REQUIRED)
  if(dlpack_FOUND)
    message(STATUS "Found system dlpack")
  else()
    message(FATAL_ERROR "dlpack not found")
  endif()
else()
  FetchContent_Declare(
    dlpack
    GIT_REPOSITORY https://github.com/dmlc/dlpack
    GIT_TAG        v1.1
  )
  FetchContent_MakeAvailable(dlpack)
endif()

# Find python
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Find cudnn
include(${PROJECT_SOURCE_DIR}/cmake/cuDNN.cmake)

option(CUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE "Whether cmake build system should fetch pybinds." ON)


if(CUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
else()
    find_package(pybind11 CONFIG REQUIRED)
endif()


# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(
    _compiled_module
    
    MODULE
    pycudnn.cpp
    properties.cpp

    pygraph/pygraph.cpp
    pygraph/norm.cpp
    pygraph/sdpa.cpp
    pygraph/pointwise.cpp

    WITH_SOABI
)
target_link_libraries(_compiled_module PRIVATE pybind11::headers)

target_compile_features(_compiled_module PRIVATE cxx_std_20)

target_include_directories(
    _compiled_module
    PRIVATE $<TARGET_PROPERTY:cudnn_frontend,INTERFACE_INCLUDE_DIRECTORIES>
    PRIVATE $<TARGET_PROPERTY:CUDNN::cudnn_all,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(_compiled_module PRIVATE NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)

if(CUDNN_FRONTEND_USE_SYSTEM_DLPACK)
    target_link_libraries(
        _compiled_module

        PRIVATE dlpack::dlpack
    )
else()
    target_link_libraries(
        _compiled_module

        PRIVATE dlpack
    )
endif()

if(MSVC)
    target_compile_options(_compiled_module PRIVATE /wd4127)
    target_compile_options(_compiled_module PRIVATE /wd4244)
    target_compile_options(_compiled_module PRIVATE /wd4458)
    target_compile_options(_compiled_module PRIVATE /wd4505)
    target_compile_options(_compiled_module PRIVATE /bigobj)
else()
    add_compile_options(-fvisibility=hidden)
endif()

set_target_properties(
    _compiled_module

    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed"
    LINK_FLAGS "-Wl,--enable-new-dtags"
    LINK_FLAGS "-Wl,-rpath,'$ORIGIN',-rpath,'$ORIGIN/../lib',-rpath,'$ORIGIN/../nvidia/cudnn/lib'"
    LINK_WHAT_YOU_USE TRUE
)

# using python bindings directly with cmake build system is not supported
# Temparorily use below parameter
option(CUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR "Whether pybinds should be kept inside build/cudnn directory." ON)
if(CUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR)
set_target_properties(
        _compiled_module

        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cudnn
    )
    file(COPY ${PROJECT_SOURCE_DIR}/python/cudnn DESTINATION ${PROJECT_BINARY_DIR})
endif()
