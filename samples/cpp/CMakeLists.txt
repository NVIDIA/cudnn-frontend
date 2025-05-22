# target sources
add_executable(
    samples

    sdpa/fp16_fwd.cpp
    sdpa/fp16_bwd.cpp
    sdpa/fp16_cached.cpp
    sdpa/fp16_benchmark.cpp
    sdpa/fp16_fwd_with_flexible_graphs.cpp
    sdpa/fp16_bwd_with_flexible_graphs.cpp
    sdpa/fp16_fwd_with_custom_dropout.cpp
    sdpa/fp16_fwd_with_paged_caches.cpp
    sdpa/fp16_fwd_paged_decode_and_prefill.cpp
    sdpa/fp16_fwd_with_cudagraphs.cpp
    sdpa/fp16_bwd_with_cudagraphs.cpp
    sdpa/fp8_fwd.cpp
    sdpa/fp8_bwd.cpp
    sdpa/fp8_fwd_bottom_right_causal_mask.cpp
    sdpa/fp8_bwd_bottom_right_causal_mask.cpp

    convolution/fprop.cpp
    convolution/fp8_fprop.cpp
    convolution/int8_fprop.cpp
    convolution/dgrads.cpp
    convolution/wgrads.cpp
    convolution/conv_dynamic_shape_benchmark.cpp

    matmul/matmuls.cpp
    matmul/fp8_matmul.cpp
    matmul/int8_matmul.cpp
    matmul/mixed_matmul.cpp
    matmul/nvfp4_mxfp8_matmul.cpp

    norm/batchnorm.cpp
    norm/layernorm.cpp
    norm/adaptive_layernorm.cpp
    norm/norm_zero_centered_gamma.cpp
    norm/rmsnorm.cpp
    norm/norm_block_scale.cpp

    misc/serialization.cpp
    misc/autotuning.cpp
    misc/custom_plan.cpp
    misc/parallel_compilation.cpp
    misc/pointwise.cpp
    misc/resample.cpp
    misc/slice.cpp
    misc/sm_carveout.cpp
    misc/cudagraphs.cpp
)

# target flags
if(MSVC)
    target_compile_options(
        samples PRIVATE
        /W4 /WX # warning level 3 and all warnings as errors
        /wd4100 # allow unused parameters
        /wd4458 # local hides class member (currently a problem for all inline setters)
        /wd4505 # unreferenced function with internal linkage has been removed
        /wd4101 /wd4189 # unreferenced local
        /bigobj # increase number of sections in .Obj file
    )
else()
    target_compile_options(
        samples PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wno-unused-function
    )
endif()

# target links
target_link_libraries(
    samples PRIVATE
    Threads::Threads
    Catch2::Catch2WithMain
    cudnn_frontend
    _cudnn_frontend_pch
    CUDNN::cudnn

    CUDA::cublasLt
    CUDA::cudart
    CUDA::cuda_driver # Needed as calls all CUDA calls will eventually move to driver
    CUDA::nvrtc
)

# target cmake properties
set_target_properties(
    samples PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
