# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

stable_abi_value = os.environ.get("CUDNN_FRONTEND_PYTHON_STABLE_ABI", "OFF").strip().upper()
if stable_abi_value not in ("", "0", "OFF", "NO", "FALSE", "1", "ON", "YES", "TRUE"):
    raise ValueError("CUDNN_FRONTEND_PYTHON_STABLE_ABI must be ON or OFF")
stable_abi_requested = stable_abi_value in ("1", "ON", "YES", "TRUE")
stable_abi = stable_abi_requested and sys.implementation.name == "cpython" and sys.version_info >= (3, 12) and not sysconfig.get_config_var("Py_GIL_DISABLED")


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[], py_limited_api=stable_abi)
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        is_windows = os.name == "nt"
        cmake_args = [
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DCUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=ON",
            f"-DCUDNN_FRONTEND_PYTHON_STABLE_ABI={'ON' if stable_abi else 'OFF'}",
            # There's no need to build cpp samples and tests with python
            f"-DCUDNN_FRONTEND_BUILD_SAMPLES=OFF",
            f"-DCUDNN_FRONTEND_BUILD_TESTS=OFF",
            # All these are handled by pip
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR=OFF",
        ]

        if is_windows:
            cmake_args += [
                f"-DCUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE=ON",
            ]
        else:
            import nanobind

            cmake_args += [
                f"-DCUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE=OFF",
                f"-Dnanobind_DIR={nanobind.cmake_dir()}",
            ]
        if "CUDA_PATH" in os.environ:
            cmake_args.append(f"-DCUDAToolkit_ROOT={os.environ['CUDA_PATH']}")

        if "CUDAToolkit_ROOT" in os.environ:
            cmake_args.append(f"-DCUDAToolkit_ROOT={os.environ['CUDAToolkit_ROOT']}")

        if "CUDNN_PATH" in os.environ:
            cmake_args.append(f"-DCUDNN_PATH={os.environ['CUDNN_PATH']}")

        if "FETCHCONTENT_SOURCE_DIR_DLPACK" in os.environ:
            cmake_args.append(f"-DFETCHCONTENT_SOURCE_DIR_DLPACK={os.environ['FETCHCONTENT_SOURCE_DIR_DLPACK']}")

        # Using Ninja-build since it a) is available as a wheel and b)
        # multithreads automatically. MSVC would require all variables be
        # exported for Ninja to pick it up, which is a little tricky to do.
        # Users can override the generator with CMAKE_GENERATOR in CMake
        # 3.15+.
        if is_windows == False:
            try:
                import ninja

                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except ImportError:
                pass

        build_args = []
        if is_windows:
            build_args += [f"--config Release"]
        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]
            else:
                # Without an explicit -j, `cmake --build` compiles one TU at a
                # time, so pip installs built the extension serially. Cap at 8:
                # the extension has ~6 TUs, each peaking at 1-2 GB of compiler
                # memory, so higher values buy nothing and only add pressure.
                build_args += [f"-j{min(os.cpu_count() or 1, 8)}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print(" ".join(cmake_args))
        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", ".", *build_args], cwd=build_temp, check=True)


setup(
    ext_modules=[CMakeExtension("cudnn._compiled_module")],
    cmdclass={"build_ext": CMakeBuild},
    options={"bdist_wheel": {"py_limited_api": "cp312"}} if stable_abi else {},
)
