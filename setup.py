import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
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

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        cmake_args = [
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DCUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=ON",
            # There's no need to build cpp samples and tests with python
            f"-DCUDNN_FRONTEND_BUILD_SAMPLES=OFF",
            f"-DCUDNN_FRONTEND_BUILD_UNIT_TESTS=OFF",
            # All these are handled by pip
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR=OFF",
            f"-DCUDNN_FRONTEND_FETCH_PYBINDS_IN_CMAKE=OFF",
        ]

        if "CUDA_PATH" in os.environ:
            cmake_args.append(f"-DCUDAToolkit_ROOT={os.environ['CUDA_PATH']}")

        if "CUDAToolkit_ROOT" in os.environ:
            cmake_args.append(f"-DCUDAToolkit_ROOT={os.environ['CUDAToolkit_ROOT']}")

        if "CUDNN_PATH" in os.environ:
            cmake_args.append(f"-DCUDNN_PATH={os.environ['CUDNN_PATH']}")

        # Using Ninja-build since it a) is available as a wheel and b)
        # multithreads automatically. MSVC would require all variables be
        # exported for Ninja to pick it up, which is a little tricky to do.
        # Users can override the generator with CMAKE_GENERATOR in CMake
        # 3.15+.
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
        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


setup(
    ext_modules=[CMakeExtension("cudnn/_compiled_module")],
    cmdclass={"build_ext": CMakeBuild},
)
