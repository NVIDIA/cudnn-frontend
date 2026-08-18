// The driver floor for the CPU-cost table: the same elementwise add, written by
// hand, with no FFI and no frontend anywhere in the call.
//
// Separate from bench_cpu_costs.cpp because it needs nvcc, and the rest of the
// benchmark is g++ over the FE and tvm-ffi headers. build.sh compiles this
// twice: once to an object linked into the benchmark (the cudaLaunchKernel
// arm), and once to a .cubin loaded through cuModuleLoad at runtime (the
// driver-API arm).
//
// The block size matches the CuTeDSL kernel's so the two launches configure the
// same grid. This is the one arm whose device code is NOT the cutlass-authored
// kernel; it is here to answer "what does a launch cost at all", not to compare
// device performance.

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

extern "C" __global__ void
native_add(float const *a, float const *b, float *c, int n) {
    int const tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// Launched from here rather than from the benchmark so the whole arm stays in
// the translation unit nvcc owns.
extern "C" void
native_add_launch(void const *a, void const *b, void *c, int n, cudaStream_t stream) {
    int const blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    native_add<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        static_cast<float const *>(a), static_cast<float const *>(b), static_cast<float *>(c), n);
}
