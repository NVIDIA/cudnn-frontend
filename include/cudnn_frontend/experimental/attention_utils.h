#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cudnn_frontend::experimental {

// ============================================================
// Internal types matching generated kernel parameter layouts
// ============================================================

struct AttentionShape_t {
    uint32_t b, q_h, k_h, v_h, s_q, s_kv, d_qk, d_v;
};

struct AttentionDescriptor_t {
    uint32_t b, q_h, k_h, v_h, s_q, s_kv, d_qk, d_v;
    uint16_t q_heads_per_k, q_heads_per_v, min_q_heads_per_kv;
};

struct FastDivisor_t {
    uint32_t val, shr, mul;
};

struct tensor_descriptor {
    static const int MAX_DIMS = 12;
    int64_t num_dims;
    int64_t dims[MAX_DIMS];
    int64_t strides[MAX_DIMS];
};

// ============================================================
// Utility functions
// ============================================================

inline int
div_up(int a, int b) {
    return (a + b - 1) / b;
}

// floor(log2(x)) for x > 0.
inline int
find_log_2_floor(uint32_t x) {
    if (x <= 1) return 0;
    int a = 0;
    while ((1u << (a + 1)) <= x) a++;
    return a;
}

// Compute FastDivisor_t for the kernel's fastDivMod which uses:
//   div = __umulhi(2 * val, mul) >> shr
// This matches cuDNN's find_divisor_v2 (xmma/fast_math.h:118-125).
inline FastDivisor_t
make_fast_divisor(uint32_t divisor) {
    FastDivisor_t d;
    d.val = divisor;

    if (divisor <= 1) {
        // Division by 1: umulhi(2*val, 0x80000000) >> 0 = val, mod = 0
        d.shr = 0;
        d.mul = 0x80000000u;
        return d;
    }

    // find_log_2(2 * divisor, round_up=true)
    uint32_t x2 = 2u * divisor;
    int a       = 0;
    {
        uint32_t tmp = x2;
        while (tmp > 1) {
            tmp >>= 1;
            a++;
        }
    }
    // round up if not a power of 2
    if (x2 & (x2 - 1)) a++;

    uint32_t p = 31 + static_cast<uint32_t>(a);
    d.mul      = static_cast<uint32_t>(((1ULL << p) + static_cast<uint64_t>(x2) - 1) / static_cast<uint64_t>(x2));
    d.shr      = p - 32;
    return d;
}

inline std::vector<std::string>
parse_flags_string(const char* data, size_t len) {
    std::vector<std::string> flags;
    std::string content(data, len);
    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        line       = line.substr(start, end - start + 1);
        if (!line.empty()) {
            flags.push_back(line);
        }
    }
    return flags;
}

// ============================================================
// Kernel specification (compile-time metadata per kernel variant)
// ============================================================

struct KernelSpec {
    const char* source;
    size_t source_len;
    const char* flags_raw;
    size_t flags_len;
    const char* kernel_name;
    int tile_m, tile_n, tile_k;
    int smem_bytes;
};

// ============================================================
// CUDA driver API wrappers (using NV_FE_CALL_TO_CU)
// ============================================================

namespace detail {

// NV_FE_CALL_TO_CU/CUDA macros reference symbols in cudnn_frontend::detail
// via unqualified lookup. Import them into this namespace.
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
using cudnn_frontend::detail::CudaLibrary;
using cudnn_frontend::detail::get_cuda_symbol;
#endif

// Re-export CUDA runtime wrappers from the main shim so the engine
// never calls cudaMalloc/cudaFree/etc. directly (required for dynamic loading).
using cudnn_frontend::detail::cuda_get_device;
using cudnn_frontend::detail::cuda_get_device_properties;
using cudnn_frontend::detail::cuda_get_error_string;
using cudnn_frontend::detail::cuda_mem_set_async;

// Import cuGetErrorString wrapper for human-readable CUresult messages
using cudnn_frontend::detail::cu_get_error_string;

// Convert CUresult to a descriptive string (e.g., "CUDA_ERROR_INVALID_VALUE")
inline std::string
cu_result_to_string(CUresult err) {
    const char* str = nullptr;
    cu_get_error_string(err, &str);
    return str ? std::string(str) : ("CUresult=" + std::to_string(static_cast<int>(err)));
}

// Convert cudaError_t to a descriptive string (e.g., "invalid argument")
inline std::string
cuda_error_to_string(cudaError_t err) {
    const char* str = cuda_get_error_string(err);
    return str ? std::string(str) : ("cudaError=" + std::to_string(static_cast<int>(err)));
}

// cudaGetLastError takes zero arguments — can't use NV_FE_CALL_TO_CUDA
// (variadic macro requires at least one arg). Handle both paths manually.
inline cudaError_t
cuda_get_last_error() {
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    using fn_t = cudaError_t (*)();
    auto _fn = reinterpret_cast<fn_t>(cudnn_frontend::detail::get_cuda_symbol(CudaLibrary::CUDART, "cudaGetLastError"));
    return _fn();
#else
    return cudaGetLastError();
#endif
}

inline CUresult
cu_library_load_data(CUlibrary* library,
                     const void* code,
                     CUjit_option* jitOptions,
                     void** jitOptionsValues,
                     unsigned int numJitOptions,
                     CUlibraryOption* libraryOptions,
                     void** libraryOptionsValues,
                     unsigned int numLibraryOptions) {
    NV_FE_CALL_TO_CU(cu_library_load_data,
                     cuLibraryLoadData,
                     library,
                     code,
                     jitOptions,
                     jitOptionsValues,
                     numJitOptions,
                     libraryOptions,
                     libraryOptionsValues,
                     numLibraryOptions);
}

inline CUresult
cu_library_get_kernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    NV_FE_CALL_TO_CU(cu_library_get_kernel, cuLibraryGetKernel, pKernel, library, name);
}

inline CUresult
cu_library_unload(CUlibrary library) {
    NV_FE_CALL_TO_CU(cu_library_unload, cuLibraryUnload, library);
}

inline CUresult
cu_kernel_set_attribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) {
    NV_FE_CALL_TO_CU(cu_kernel_set_attribute, cuKernelSetAttribute, attrib, val, kernel, dev);
}

inline CUresult
cu_launch_kernel(CUfunction f,
                 unsigned int gridDimX,
                 unsigned int gridDimY,
                 unsigned int gridDimZ,
                 unsigned int blockDimX,
                 unsigned int blockDimY,
                 unsigned int blockDimZ,
                 unsigned int sharedMemBytes,
                 CUstream hStream,
                 void** kernelParams,
                 void** extra) {
    NV_FE_CALL_TO_CU(cu_launch_kernel,
                     cuLaunchKernel,
                     f,
                     gridDimX,
                     gridDimY,
                     gridDimZ,
                     blockDimX,
                     blockDimY,
                     blockDimZ,
                     sharedMemBytes,
                     hStream,
                     kernelParams,
                     extra);
}

inline CUresult
cu_tensor_map_encode_tiled(CUtensorMap* tensorMap,
                           CUtensorMapDataType tensorDataType,
                           cuuint32_t tensorRank,
                           void* globalAddress,
                           const cuuint64_t* globalDim,
                           const cuuint64_t* globalStrides,
                           const cuuint32_t* boxDim,
                           const cuuint32_t* elementStrides,
                           CUtensorMapInterleave interleave,
                           CUtensorMapSwizzle swizzle,
                           CUtensorMapL2promotion l2Promotion,
                           CUtensorMapFloatOOBfill oobFill) {
    NV_FE_CALL_TO_CU(cu_tensor_map_encode_tiled,
                     cuTensorMapEncodeTiled,
                     tensorMap,
                     tensorDataType,
                     tensorRank,
                     globalAddress,
                     globalDim,
                     globalStrides,
                     boxDim,
                     elementStrides,
                     interleave,
                     swizzle,
                     l2Promotion,
                     oobFill);
}

inline CUresult
cu_device_get(CUdevice* device, int ordinal) {
    NV_FE_CALL_TO_CU(cu_device_get, cuDeviceGet, device, ordinal);
}

inline CUresult
cu_init(unsigned int flags) {
    NV_FE_CALL_TO_CU(cu_init, cuInit, flags);
}

}  // namespace detail

// ============================================================
// TMA descriptor creation (4D)
// ============================================================
inline error_t
create_tma_desc_4d(CUtensorMap* desc,
                   void* globalAddress,
                   CUtensorMapDataType dataType,
                   uint32_t dim0,
                   uint32_t dim1,
                   uint32_t dim2,
                   uint32_t dim3,
                   uint64_t stride1_bytes,
                   uint64_t stride2_bytes,
                   uint64_t stride3_bytes,
                   uint32_t boxDim0,
                   uint32_t boxDim1,
                   CUtensorMapSwizzle swizzle) {
    uint64_t globalDims[4]    = {dim0, dim1, dim2, dim3};
    uint64_t globalStrides[3] = {stride1_bytes, stride2_bytes, stride3_bytes};
    uint32_t boxDims[4]       = {boxDim0, boxDim1, 1, 1};
    uint32_t elemStrides[4]   = {1, 1, 1, 1};

    CUresult err_status = detail::cu_tensor_map_encode_tiled(desc,
                                                             dataType,
                                                             4,
                                                             globalAddress,
                                                             globalDims,
                                                             globalStrides,
                                                             boxDims,
                                                             elemStrides,
                                                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                                                             swizzle,
                                                             CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

    RETURN_CUDNN_FRONTEND_ERROR_IF(
        err_status != CUDA_SUCCESS, error_code_t::CUDA_API_FAILED, "cuTensorMapEncodeTiled failed");
    return {error_code_t::OK, ""};
}

}  // namespace cudnn_frontend::experimental