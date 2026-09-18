#!/bin/bash
# Build the two C++ pieces:
#   flow2_import_execute            standalone binary, no Python in the process
#   flow3_global_execute_nanobind   python extension, loaded into the process that registers
#
#   CUDA_PATH=... CUDNN_PATH=... ./build.sh
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FE_ROOT="$(cd "$HERE/../.." && pwd)"

: "${CUDA_PATH:?set CUDA_PATH to the CUDA toolkit}"
: "${CUDNN_PATH:?set CUDNN_PATH to the cuDNN install}"
PYTHON="${PYTHON:-python3}"

read -r TVM_INC TVM_DLPACK TVM_LIB NB_DIR NB_INC PY_INC <<EOF
$($PYTHON - <<'PY'
import os, sysconfig, nanobind, tvm_ffi, tvm_ffi.libinfo as libinfo

tvm = os.path.dirname(tvm_ffi.__file__)
nb = os.path.dirname(nanobind.__file__)
print(
    os.path.join(tvm, "include"),
    os.path.join(tvm, "3rdparty", "dlpack", "include"),
    libinfo.find_libtvm_ffi(),
    nb,
    nanobind.include_dir(),
    sysconfig.get_paths()["include"],
)
PY
)
EOF

COMMON=(-std=c++17 -O2 -Wall -Wextra
        -I"$FE_ROOT/include" -isystem "$TVM_INC" -isystem "$TVM_DLPACK"
        -I"$CUDNN_PATH/include" -I"$CUDA_PATH/include"
        -L"$CUDNN_PATH/lib64" -L"$CUDNN_PATH/lib" -L"$CUDA_PATH/lib64")

echo "== flow2_import_execute (standalone) =="
g++ "${COMMON[@]}" "$HERE/flow2_import_execute.cpp" -o "$HERE/flow2_import_execute" \
    "$TVM_LIB" -lcudnn -lcudart

echo "== bench_cpu_costs (standalone) =="
# The floor arms need nvcc. Compiled twice: an object for the cudaLaunchKernel
# arm, and a cubin the driver-API arm loads at runtime.
SM_ARCH="${SM_ARCH:-$($PYTHON -c 'import torch;a,b=torch.cuda.get_device_capability();print(f"sm_{a}{b}a")' 2>/dev/null || echo sm_100a)}"
"$CUDA_PATH/bin/nvcc" -O2 -std=c++17 -arch="$SM_ARCH" -c "$HERE/bench_native_add.cu" -o "$HERE/bench_native_add.o"
"$CUDA_PATH/bin/nvcc" -O2 -std=c++17 -arch="$SM_ARCH" -cubin "$HERE/bench_native_add.cu" -o "$HERE/bench_native_add.cubin"
g++ "${COMMON[@]}" "$HERE/bench_cpu_costs.cpp" "$HERE/bench_native_add.o" -o "$HERE/bench_cpu_costs" \
    "$TVM_LIB" -lcudnn -lcudart -lcuda

echo "== bench_sdpa_cpu_costs (standalone) =="
g++ "${COMMON[@]}" "$HERE/bench_sdpa_cpu_costs.cpp" -o "$HERE/bench_sdpa_cpu_costs" \
    "$TVM_LIB" -lcudnn -lcudart -lcuda -lnvrtc

echo "== flow3_global_execute_nanobind (python extension) =="
EXT_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# -fvisibility=hidden on purpose: this is exactly how FE's own bindings are
# built, so the extension proves the registry really is shared across shared
# objects rather than accidentally working because everything is exported.
g++ "${COMMON[@]}" -shared -fPIC -fvisibility=hidden \
    "$HERE/flow3_global_execute_nanobind.cpp" "$NB_DIR/src/nb_combined.cpp" \
    -I"$NB_INC" -I"$NB_DIR/src" -I"$NB_DIR/ext/robin_map/include" -I"$PY_INC" \
    -o "$HERE/flow3_global_execute_nanobind${EXT_SUFFIX}" \
    "$TVM_LIB" -lcudnn -lcudart

echo
echo "built:"
ls -1 "$HERE/flow2_import_execute" "$HERE/bench_cpu_costs" "$HERE/bench_sdpa_cpu_costs" "$HERE"/flow3_global_execute_nanobind*.so
