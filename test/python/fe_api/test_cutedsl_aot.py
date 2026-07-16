import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from test_utils import torch_fork_set_rng

pytestmark = pytest.mark.L0


@torch_fork_set_rng(seed=0)
def test_gemm_amax_aot_cpp_smoke(tmp_path, monkeypatch):
    pytest.importorskip("cutlass", reason="CuTe DSL is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if shutil.which("g++") is None:
        pytest.skip("g++ is not available for the TVM-FFI C++ smoke")

    from cudnn import gemm_amax_wrapper_sm100
    from cudnn._cutedsl_aot import read_metadata
    from cudnn.gemm_amax import api as gemm_amax_api
    from fe_api.test_gemm_amax_utils import (
        allocate_input_tensors,
    )

    if importlib.util.find_spec("tvm_ffi") is None:
        pytest.skip("TVM-FFI Python package is not installed")

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip(f"GemmAmax AOT requires SM100+, found SM{major}{minor}")

    a_torch, _, b_torch, _, sfa_torch, _, sfb_torch, _ = allocate_input_tensors(
        512,
        256,
        256,
        1,
        torch.float8_e5m2,
        torch.float8_e8m0fnu,
        32,
        "k",
        "k",
    )
    gemm_amax_api._cache_of_GemmAmaxSm100Objects.clear()
    monkeypatch.setenv("NV_CUDNN_FE_AOT_MODE", "write")
    monkeypatch.setenv("NV_CUDNN_FE_AOT_DIR", str(tmp_path))
    gemm_amax_wrapper_sm100(a_torch, b_torch, sfa_torch, sfb_torch)
    metadata_files = list(tmp_path.glob("*.json"))
    assert len(metadata_files) == 1
    metadata = read_metadata(metadata_files[0])

    smoke_src = tmp_path / "cutedsl_aot_cpp_smoke.cpp"
    smoke_src.write_text(
        (Path(__file__).parent / "cutedsl_aot_cpp_smoke.cpp").read_text(),
        encoding="utf-8",
    )
    exe = tmp_path / "cutedsl_aot_cpp_smoke"
    tvm_ffi_spec = importlib.util.find_spec("tvm_ffi")
    if tvm_ffi_spec is None or tvm_ffi_spec.origin is None:
        pytest.skip("TVM-FFI Python package path is unavailable for C++ smoke")
    tvm_ffi_root = Path(tvm_ffi_spec.origin).parent
    tvm_ffi_include = tvm_ffi_root / "include"
    tvm_ffi_lib = tvm_ffi_root / "lib"
    if not (tvm_ffi_include / "tvm" / "ffi" / "extra" / "module.h").exists():
        pytest.skip(f"TVM-FFI C++ module headers are unavailable under {tvm_ffi_include}")
    if not (tvm_ffi_lib / "libtvm_ffi.so").exists():
        pytest.skip(f"TVM-FFI C++ runtime library is unavailable under {tvm_ffi_lib}")

    compile_cmd = [
        "g++",
        "-std=c++17",
        str(smoke_src),
        "-I",
        str(tvm_ffi_include),
        "-L",
        str(tvm_ffi_lib),
        f"-Wl,-rpath,{tvm_ffi_lib}",
        "-ltvm_ffi",
        "-ldl",
        "-o",
        str(exe),
    ]
    compile_result = subprocess.run(compile_cmd, text=True, capture_output=True)
    if compile_result.returncode != 0:
        pytest.skip("TVM-FFI C++ smoke compiler setup is unavailable: " + compile_result.stderr.strip())

    import cutlass.cute as cute

    finder = getattr(cute.runtime, "find_runtime_libraries", None)
    runtime_libraries = ()
    if finder is not None:
        try:
            runtime_libraries = finder(enable_tvm_ffi=True) or ()
        except TypeError:
            runtime_libraries = finder() or ()
    runtime_library_dirs = [str(Path(library).parent) for library in runtime_libraries]
    env = None
    if runtime_library_dirs:
        env = {
            **os.environ,
            "LD_LIBRARY_PATH": ":".join(runtime_library_dirs + ([os.environ["LD_LIBRARY_PATH"]] if "LD_LIBRARY_PATH" in os.environ else [])),
        }

    run_result = subprocess.run(
        [str(exe), str(tmp_path / metadata.shared_library), metadata.symbol],
        text=True,
        capture_output=True,
        env=env,
    )
    assert run_result.returncode == 0, run_result.stderr
