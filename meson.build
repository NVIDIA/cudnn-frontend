# === @begin: General ===
project(
    'cuda-cudnn-frontend',
    'c', 'cpp',
    default_options: [
        'c_std=gnu17',
        'cpp_std=gnu++23',
        'buildtype=release',
    ],
    meson_version: '>=1.3.0',
    version: run_command([
        'sed', '-n',
        's/^__version__\\s*=\\s*"\([^"]*\)"/\\1/p',
        meson.project_source_root() / 'python' / 'cudnn' / '__init__.py'
    ], check: true).stdout().strip()
)
# === @end: General ===


# === @begin: Meson modules imports ===
fs = import('fs')
py = import('python').find_installation(pure: false, modules: ['pybind11'])
# === @end: Meson modules imports ===



# === @begin: Project-wide settings ===
add_project_arguments(
    # From root, find the `include` directory
    '-I' + meson.project_source_root() / 'include',
    # From root, find the `python/pygraph` directory
    '-I' + meson.project_source_root() / 'python' / 'pygraph',
    language: ['c', 'cpp']
)
# === @end: Project-wide settings ===


# === @begin: Dependencies ===
# By default, `meson` prefers `cudart_static` and we don't want to alter that.
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#cudart-none-shared-static-cudart
#
# Actually, only `cudart` and headers are used from CUDAToolkit. All other components and libraries are ignored.
# Moreover, this package doesn't involve any CUDA code compilation (i.e., .cu and .cuh files).
cuda = dependency('cuda')

# Find cudnn backend using their cmake file.
# We created a symlink to `cuDNN.cmake` in the `cmake` directory to our custom `FindcuDNN.cmake` file (it's a naming convention forced by CMake)
cudnn_backend = dependency('cuDNN', cmake_module_path: meson.project_source_root() / 'cmake')

# Find our system installed `dlpack` using their cmake file and ...
dlpack = dependency('dlpack', method: 'cmake', modules: ['dlpack::dlpack'])

# `CUDNN_FRONTEND_SKIP_JSON_LIB` is not needed because we always use system-provided version
nlohmann_json = dependency('nlohmann_json', method: 'cmake')

# Find the `pybind11` package using their built in approach: https://mesonbuild.com/Dependencies.html#pybind11:~:text=pkg%2Dconfig.-,Pybind11,-(added%201.1.0)
pybind11 = dependency('Pybind11')
# === @end: Dependencies ===


# === @begin: Native ===
cudnn_frontend_native_header_files = files(
    'include/cudnn_frontend_Reorder_Tensor.h',
    'include/cudnn_frontend_Operation.h',
    'include/cudnn_frontend_EngineConfigGenerator.h',
    'include/cudnn_frontend.h',
    'include/cudnn_frontend_OperationGraph.h',
    'include/cudnn_frontend_find_plan.h',
    'include/cudnn_frontend_Rng.h',
    'include/cudnn_frontend_MatMulDesc.h',
    'include/cudnn_frontend_get_plan.h',
    'include/cudnn_frontend_Tensor.h',
    'include/cudnn_frontend_Filters.h',
    'include/cudnn_frontend_VariantPack.h',
    'include/cudnn_frontend_ReductionDesc.h',
    'include/cudnn_frontend_utils.h',
    'include/cudnn_frontend_ConvDesc.h',
    'include/cudnn_frontend_Engine.h',
    'include/cudnn_frontend_PointWiseDesc.h',
    'include/cudnn_frontend_Heuristics.h',
    'include/cudnn_frontend_EngineConfig.h',
    'include/cudnn_frontend/node_interface.h',
    'include/cudnn_frontend/graph_interface.h',
    'include/cudnn_frontend/cudnn_interface.h',
    'include/cudnn_frontend/graph_helpers.h',
    'include/cudnn_frontend/context.h',
    'include/cudnn_frontend/plans.h',
    'include/cudnn_frontend/utils/attn_score_modifiers.h',
    'include/cudnn_frontend/utils/serialize.h',
    'include/cudnn_frontend/backend/execution_helpers.h',
    'include/cudnn_frontend/backend/kernel_cache.h',
    'include/cudnn_frontend/backend/plan_helpers.h',
    'include/cudnn_frontend/backend/backend_descriptor.h',
    'include/cudnn_frontend/graph_properties.h',
    'include/cudnn_frontend/node/softmax.h',
    'include/cudnn_frontend/node/matmul.h',
    'include/cudnn_frontend/node/resample.h',
    'include/cudnn_frontend/node/dbn_weight.h',
    'include/cudnn_frontend/node/rmsnorm.h',
    'include/cudnn_frontend/node/conv_dgrad.h',
    'include/cudnn_frontend/node/scaled_dot_product_flash_attention.h',
    'include/cudnn_frontend/node/conv_fprop.h',
    'include/cudnn_frontend/node/batchnorm_inference.h',
    'include/cudnn_frontend/node/batchnorm.h',
    'include/cudnn_frontend/node/layernorm.h',
    'include/cudnn_frontend/node/block_scale_quantize.h',
    'include/cudnn_frontend/node/block_scale_dequantize.h',
    'include/cudnn_frontend/node/genstats.h',
    'include/cudnn_frontend/node/instancenorm.h',
    'include/cudnn_frontend/node/slice.h',
    'include/cudnn_frontend/node/sdpa_fp8_bwd.h',
    'include/cudnn_frontend/node/sdpa_fp8.h',
    'include/cudnn_frontend/node/concatenate.h',
    'include/cudnn_frontend/node/matmul_fp8.h',
    'include/cudnn_frontend/node/rng.h',
    'include/cudnn_frontend/node/paged_cache_load.h',
    'include/cudnn_frontend/node/reshape.h',
    'include/cudnn_frontend/node/bn_finalize.h',
    'include/cudnn_frontend/node/dbn.h',
    'include/cudnn_frontend/node/conv_wgrad.h',
    'include/cudnn_frontend/node/pointwise.h',
    'include/cudnn_frontend/node/reduction.h',
    'include/cudnn_frontend/node/dln.h',
    'include/cudnn_frontend/knobs.h',
    'include/cudnn_backend_base.h',
    'include/cudnn_frontend_version.h',
    'include/cudnn_frontend_EngineFallbackList.h',
    'include/cudnn_frontend_shim.h',
    'include/cudnn_frontend_Resample.h',
    'include/cudnn_frontend_Errata.h',
    'include/cudnn_frontend_ExecutionPlan.h',
    'include/cudnn_frontend_ExecutionPlanCache.h',
    'include/cudnn_frontend_Logging.h'
)

# @TODO+:1
install_headers(
    cudnn_frontend_native_header_files,
    preserve_path: true
)

cudnn_frontend_native = include_directories('include')
cudnn_frontend_native_dep = declare_dependency(
  include_directories: cudnn_frontend_native,
  dependencies: [cuda, cudnn_backend]
)
# === @end: Native ===


# === @begin: Python, compilable ===
py.extension_module(
    '_compiled_module',
    [
        'python/pycudnn.cpp',
        'python/properties.cpp',

        'python/pygraph/pygraph.cpp',
        'python/pygraph/norm.cpp',
        'python/pygraph/sdpa.cpp',
        'python/pygraph/pointwise.cpp'
    ],
    install: true,
    subdir: 'cudnn/',
    dependencies: [
        cuda,
        cudnn_backend,
        dlpack,
        nlohmann_json,
        pybind11,
        cudnn_frontend_native_dep
    ],
    # Nvidia upstream does the same but for unknown reason PyPi-published wheel still exposes all symbols.
    gnu_symbol_visibility: 'hidden',

    # This library is one of a few which requires direct interaction with the GPU driver, so we need to link this library explicitly.
    # `meson` fails to find that in their `cuda` finder, so we do it here. Noteworthy, `libcuda.so` doesn't exist at buildtime, it only
    # becomes inserted by NVIDIA driver runtime. At buildtime, we just have stubs library allowing to build but useless at runtime.
    link_args: ['-lcuda']
)
# === @end: Python, compilable ===


# === @begin: Python, pure ===
install_subdir(
    'python/cudnn/',
    install_dir: py.get_install_dir(),
    install_tag: 'python-runtime',

    follow_symlinks: true,
)
# === @end: Python, pure ===


# @HELP TODO+
# 1. Try using cmake mode to find cuda and it portions to simplify our build and only use that we really need.
#    1. Probably we don't need `cudart_static` even!
# 2. Headers into /usr/local/include
#    1. Probably put into `cudnn` package (and not into site-packages directly) and just expose via cmake-config file
# 3. Generate CMake configuration files
#    1. cmake.write_basic_package_version_file
#    OR
#    2. [more preferrably] cmake.<some function> -> and that Conors merged into upstream
#    3. Review the changes in the pull request for further insights
     1. https://github.com/NVIDIA/cudnn-frontend/pull/125/files#diff-f716df17f273769aad66b5665797f033f4ae752ee785e426203f02d41bed8147
