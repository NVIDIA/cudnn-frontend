#pragma once

#include <vector>
#include <stdexcept>
#include <sstream>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <catch2/catch_test_macros.hpp>

static int64_t
div_up(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

#define CUDA_CHECK(status)                                                                                    \
    {                                                                                                         \
        cudaError_t err = status;                                                                             \
        if (err != cudaSuccess) {                                                                             \
            std::stringstream err_msg;                                                                        \
            err_msg << "CUDA Error: " << cudaGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                              \
            FAIL(err_msg.str());                                                                              \
        }                                                                                                     \
    }

#define CUDNN_CHECK(status)                                                                                     \
    {                                                                                                           \
        cudnnStatus_t err = status;                                                                             \
        if (err != CUDNN_STATUS_SUCCESS) {                                                                      \
            std::stringstream err_msg;                                                                          \
            err_msg << "cuDNN Error: " << cudnnGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                                \
            FAIL(err_msg.str());                                                                                \
        }                                                                                                       \
    }

// Custom deleter for cudnnHandle_t
struct CudnnHandleDeleter {
    void
    operator()(cudnnHandle_t* handle) const {
        if (handle) {
            CUDNN_CHECK(cudnnDestroy(*handle));
            delete handle;
        }
    }
};

// Function to create a unique_ptr for cudnnHandle_t
inline std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>
create_cudnn_handle() {
    auto handle = std::make_unique<cudnnHandle_t>();
    CUDNN_CHECK(cudnnCreate(handle.get()));
    return std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>(handle.release(), CudnnHandleDeleter());
}

inline size_t
get_compute_capability() {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, current_device));
    return prop.major * 10 + prop.minor;
}

inline bool
is_ampere_arch() {
    auto cc = get_compute_capability();
    return (80 <= cc) && (cc < 89);
}

inline bool
is_ada_arch() {
    auto cc = get_compute_capability();
    return (cc == 89);
}

inline bool
is_hopper_arch() {
    auto cc = get_compute_capability();
    return (90 <= cc) && (cc < 100);
}

inline bool
is_blackwell_arch() {
    auto cc = get_compute_capability();
    return (100 <= cc);
}

inline bool
is_blackwell_computing_arch() {
    auto cc = get_compute_capability();
    return (100 <= cc && cc < 110);
}

inline bool
is_arch_supported_by_cudnn() {
    if (cudnnGetVersion() < 8600 && (is_hopper_arch() || is_ada_arch())) {
        return false;
    }
    return true;
}

inline bool
check_device_arch_newer_than(std::string const& arch) {
    size_t arch_major = 6;
    size_t arch_minor = 0;
    if (arch == "blackwell") {
        arch_major = 10;
    }
    if (arch == "hopper") {
        arch_major = 9;
    }
    if (arch == "ampere") {
        arch_major = 8;
    }
    if (arch == "turing") {
        arch_major = 7;
        arch_minor = 5;
    }
    if (arch == "volta") {
        arch_major = 7;
    }
    if (arch == "pascal") {
        arch_major = 6;
    }

    auto queried_version = arch_major * 10 + arch_minor;
    if (get_compute_capability() >= queried_version) {
        return true;
    }
    return false;
}

static half
cpu_float2half_rn(float f) {
    void* f_ptr = &f;
    unsigned x  = *((int*)f_ptr);
    unsigned u  = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    __half_raw hr;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        hr.x = 0x7fffU;
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<half*>(hr_ptr);
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        hr.x = static_cast<unsigned short>(sign | 0x7c00U);
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<half*>(hr_ptr);
    }
    if (u < 0x33000001) {
        hr.x = static_cast<unsigned short>(sign | 0x0000U);
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<half*>(hr_ptr);
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift    = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb    = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    hr.x = static_cast<unsigned short>((sign | (exponent << 10) | mantissa));

    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half*>(hr_ptr);
}

static float
cpu_half2float(half h) {
    // Add an indirection to get around type aliasing check
    void* h_ptr   = &h;
    __half_raw hr = *reinterpret_cast<__half_raw*>(h_ptr);

    unsigned sign     = ((hr.x >> 15) & 1);
    unsigned exponent = ((hr.x >> 10) & 0x1f);
    unsigned mantissa = ((hr.x & 0x3ff) << 13);

    if (exponent == 0x1f) { /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) { /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1; /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    // Add an indirection to get around type aliasing check
    void* temp_ptr = &temp;
    float* res_ptr = reinterpret_cast<float*>(temp_ptr);
    return *res_ptr;
}

// Generate uniform numbers [0,1)
static void
initHostImage(float* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10f;  // 2^-32
    }
}

static void
initHostImage(half* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = cpu_float2half_rn(float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
static void
initHostImage(int8_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate random integers [0, 50] to avoid uint8 overflow
static void
initHostImage(uint8_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 50]
        image[index] = (uint8_t)(50 * float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
static void
initHostImage(int32_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int32_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
static void
initHostImage(int64_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int64_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32
    }
}

// Currently set to generate booleans
static void
initHostImage(bool* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        int64_t val = ((int32_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32

        // val is 0 or 1
        image[index] = (val == 1);
    }
}

template <typename T>
static void
initImage(T* devPtr, size_t imageSize) {
    if (imageSize == 0) {
        return;
    }

    std::vector<T> host(imageSize);
    initHostImage(host.data(), static_cast<int64_t>(imageSize));
    CUDA_CHECK(cudaMemcpy(devPtr, host.data(), sizeof(host[0]) * imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
static void
fillImage(T* devPtr, size_t imageSize, T fillValue) {
    if (imageSize == 0) {
        return;
    }

    std::vector<T> host(imageSize, fillValue);
    CUDA_CHECK(cudaMemcpy(devPtr, host.data(), sizeof(host[0]) * imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T_ELEM>
struct Surface {
    T_ELEM* devPtr = NULL;
    size_t size    = 0;

   protected:
    explicit Surface() {}

   public:
    explicit Surface(size_t size) : size(size) {
        if (size == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(&devPtr, size * sizeof(T_ELEM)));
        initImage(devPtr, size);
    }

    explicit Surface(size_t size, T_ELEM fillValue) : size(size) {
        if (size == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(&devPtr, size * sizeof(T_ELEM)));
        fillImage(devPtr, size, fillValue);
    }

    Surface(const Surface& other) : size(other.size) {
        if (size == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(&devPtr, size * sizeof(T_ELEM)));
        CUDA_CHECK(cudaMemcpy(devPtr, other.devPtr, sizeof(devPtr[0]) * size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    Surface(Surface&& other) noexcept : Surface() { swap(*this, other); }

    Surface&
    operator=(Surface other) {
        swap(*this, other);
        return *this;
    }

    friend void
    swap(Surface& first, Surface& second) {
        std::swap(first.size, second.size);
        std::swap(first.devPtr, second.devPtr);
    }

    ~Surface() {
        if (devPtr) {
            cudaFree(devPtr);
            devPtr = nullptr;
        }
    }
};
