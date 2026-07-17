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

  # Read dlpack version from dlpack_version.txt
  file(READ ${PROJECT_SOURCE_DIR}/dlpack_version.txt DLPACK_VERSION)
  string(STRIP "${DLPACK_VERSION}" DLPACK_VERSION)

  FetchContent_Declare(
    dlpack
    GIT_REPOSITORY https://github.com/dmlc/dlpack
    GIT_TAG        v${DLPACK_VERSION}
  )
  FetchContent_MakeAvailable(dlpack)
endif()

# Find python
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Find cudnn
include(${PROJECT_SOURCE_DIR}/cmake/cuDNN.cmake)

option(CUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE "Whether cmake build system should fetch pybinds." ON)
set(PYBIND11_FINDPYTHON ON)


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

option(CUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR "Whether pybinds should be kept inside build/cudnn directory." ON)
if(CUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR)
set_target_properties(
        _compiled_module

        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cudnn
    )
    file(COPY ${PROJECT_SOURCE_DIR}/python/cudnn DESTINATION ${PROJECT_BINARY_DIR})
endif()

# Optionally install the python bindings with `cmake --install`, for system
# packagers and users who cannot go through pip/setup.py (see issue #149).
# The layout matches what the wheel produces: the `cudnn` package (python
# sources + compiled module) plus a PEP 376 `.dist-info` directory so the
# install is visible to pip, importlib.metadata and other packaging tools.
option(CUDNN_FRONTEND_INSTALL_PYTHON_BINDINGS "Whether `cmake --install` should install the python bindings into the python site-packages." OFF)
if(CUDNN_FRONTEND_INSTALL_PYTHON_BINDINGS)
    # Destination: absolute path, or path relative to CMAKE_INSTALL_PREFIX.
    # Defaults to the platlib (site-packages) of the interpreter found above,
    # which honors virtual environments.
    set(CUDNN_FRONTEND_PYTHON_INSTALL_DIR "${Python_SITEARCH}" CACHE PATH "Directory the cudnn python package is installed to.")

    if(IS_ABSOLUTE "${CUDNN_FRONTEND_PYTHON_INSTALL_DIR}")
        set(_cudnn_fe_python_install_dir "${CUDNN_FRONTEND_PYTHON_INSTALL_DIR}")
    else()
        set(_cudnn_fe_python_install_dir "${CMAKE_INSTALL_PREFIX}/${CUDNN_FRONTEND_PYTHON_INSTALL_DIR}")
    endif()
    message(STATUS "cudnn python bindings will be installed to ${_cudnn_fe_python_install_dir}")

    # Pure python sources of the `cudnn` package
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/python/cudnn
        DESTINATION ${_cudnn_fe_python_install_dir}
        FILES_MATCHING PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
    )

    # Compiled extension module. cudnn/__init__.py imports it from
    # `cudnn.Release` on Windows (multi-config layout), from `cudnn` elsewhere.
    if(WIN32)
        set(_cudnn_fe_module_subdir cudnn/Release)
    else()
        set(_cudnn_fe_module_subdir cudnn)
    endif()
    install(
        TARGETS _compiled_module
        LIBRARY DESTINATION ${_cudnn_fe_python_install_dir}/${_cudnn_fe_module_subdir}
    )

    # PEP 376 .dist-info so the install is discoverable by packaging tools
    set(_cudnn_fe_dist_info "nvidia_cudnn_frontend-${PROJECT_VERSION}.dist-info")
    file(CONFIGURE
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist-info/METADATA
        CONTENT
"Metadata-Version: 2.1
Name: nvidia-cudnn-frontend
Version: @PROJECT_VERSION@
Summary: NVIDIA cuDNN Frontend — Python and C++ Graph API
Home-page: https://github.com/NVIDIA/cudnn-frontend
License: MIT
Requires-Python: >=3.9
"
        @ONLY
    )
    file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist-info/INSTALLER CONTENT "cmake\n" @ONLY)
    file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist-info/top_level.txt CONTENT "cudnn\n" @ONLY)
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/dist-info/METADATA
            ${CMAKE_CURRENT_BINARY_DIR}/dist-info/INSTALLER
            ${CMAKE_CURRENT_BINARY_DIR}/dist-info/top_level.txt
        DESTINATION ${_cudnn_fe_python_install_dir}/${_cudnn_fe_dist_info}
    )

    # Generate the RECORD file at install time, once all files are in place,
    # so that `pip uninstall nvidia-cudnn-frontend` works.
    file(CONFIGURE
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist-info/GenerateRecord.cmake
        CONTENT
[[set(_site_dir "$ENV{DESTDIR}@_cudnn_fe_python_install_dir@")
set(_dist_info "@_cudnn_fe_dist_info@")
file(GLOB_RECURSE _installed_files LIST_DIRECTORIES false
    RELATIVE "${_site_dir}"
    "${_site_dir}/cudnn/*"
    "${_site_dir}/${_dist_info}/*"
)
list(REMOVE_ITEM _installed_files "${_dist_info}/RECORD")
set(_record "")
foreach(_file IN LISTS _installed_files)
    string(APPEND _record "${_file},,\n")
endforeach()
string(APPEND _record "${_dist_info}/RECORD,,\n")
file(WRITE "${_site_dir}/${_dist_info}/RECORD" "${_record}")
message(STATUS "Generated: ${_site_dir}/${_dist_info}/RECORD")
]]
        @ONLY
    )
    install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/dist-info/GenerateRecord.cmake)
endif()
