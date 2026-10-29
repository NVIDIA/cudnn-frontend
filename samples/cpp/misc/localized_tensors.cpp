/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils/helpers.h"
#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

#include <cuda.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

/*
Run this example by using command:
bin/samples "[localized]"

This example runs a cuDNN graph on tensors whose memory sits in one locality
domain of the GPU.

Some GPUs, Rubin among them, divide their SMs and memory into locality domains.
The CUDA virtual memory management (VMM) API can place an allocation in one
chosen domain: cuMemCreate with CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN.
cuDNN needs no special support for this. The application allocates the memory
and passes cuDNN an ordinary device pointer, so everything locality-specific
happens in the allocator, above cuDNN. From validate() to execute() this is an
ordinary cuDNN frontend program, and one graph serves every domain.

On a device with a single locality domain, a localized allocation is the same
as an ordinary full-device allocation, so the example skips there.

The matmul inputs are constants, so the expected output is known without a
reference implementation.
*/

#define A_UID 0
#define B_UID 1
#define C_UID 2

// The CUDA driver API counterpart of CUDA_CHECK in helpers.h.
#define CU_CHECK(status)                                                                                              \
    {                                                                                                                 \
        CUresult err = status;                                                                                        \
        if (err != CUDA_SUCCESS) {                                                                                    \
            std::stringstream err_msg;                                                                                \
            err_msg << "CUDA Driver Error: " << driver_error_string(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                                      \
            FAIL(err_msg.str());                                                                                      \
        }                                                                                                             \
    }

#if CUDA_VERSION >= 13040
namespace {

// The CUDA driver functions this sample calls. They are looked up at run time through the CUDA runtime, so
// the samples executable does not link libcuda.so and still starts on a machine without the driver. Each
// type is the function's signature as of the CUDA version in its suffix, the version that is requested.
struct DriverApi {
    PFN_cuGetErrorString_v6000 cuGetErrorString;
    PFN_cuInit_v2000 cuInit;
    PFN_cuDeviceGet_v2000 cuDeviceGet;
    PFN_cuDeviceGetAttribute_v2000 cuDeviceGetAttribute;
    PFN_cuMemGetAllocationGranularity_v10020 cuMemGetAllocationGranularity;
    PFN_cuMemCreate_v10020 cuMemCreate;
    PFN_cuMemAddressReserve_v10020 cuMemAddressReserve;
    PFN_cuMemMap_v10020 cuMemMap;
    PFN_cuMemSetAccess_v10020 cuMemSetAccess;
    PFN_cuCtxSynchronize_v2000 cuCtxSynchronize;
    PFN_cuMemUnmap_v10020 cuMemUnmap;
    PFN_cuMemAddressFree_v10020 cuMemAddressFree;
    PFN_cuMemRelease_v10020 cuMemRelease;
    PFN_cuPointerGetAttribute_v4000 cuPointerGetAttribute;
};
DriverApi driver = {};

// Fills `driver`. Returns the name of the first function the driver does not provide, or nullptr.
char const *
load_driver_api() {
    char const *missing = nullptr;
    auto load           = [&missing](char const *name, unsigned int version, auto &function) {
        void *address                          = nullptr;
        cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
        cudaError_t err = cudaGetDriverEntryPointByVersion(name, &address, version, cudaEnableDefault, &status);
        if ((err != cudaSuccess || status != cudaDriverEntryPointSuccess) && missing == nullptr) {
            missing = name;
        }
        function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(address);
    };
    load("cuGetErrorString", 6000, driver.cuGetErrorString);
    load("cuInit", 2000, driver.cuInit);
    load("cuDeviceGet", 2000, driver.cuDeviceGet);
    load("cuDeviceGetAttribute", 2000, driver.cuDeviceGetAttribute);
    load("cuMemGetAllocationGranularity", 10020, driver.cuMemGetAllocationGranularity);
    load("cuMemCreate", 10020, driver.cuMemCreate);
    load("cuMemAddressReserve", 10020, driver.cuMemAddressReserve);
    load("cuMemMap", 10020, driver.cuMemMap);
    load("cuMemSetAccess", 10020, driver.cuMemSetAccess);
    load("cuCtxSynchronize", 2000, driver.cuCtxSynchronize);
    load("cuMemUnmap", 10020, driver.cuMemUnmap);
    load("cuMemAddressFree", 10020, driver.cuMemAddressFree);
    load("cuMemRelease", 10020, driver.cuMemRelease);
    load("cuPointerGetAttribute", 4000, driver.cuPointerGetAttribute);
    return missing;
}

char const *
driver_error_string(CUresult status) {
    char const *text = nullptr;
    driver.cuGetErrorString(status, &text);
    return text ? text : "unknown error";
}

// Device memory placed in one locality domain. Freed when it goes out of scope.
class LocalizedBuffer {
   public:
    LocalizedBuffer(size_t bytes, CUdevice device, int domain) {
        CUmemAllocationProp prop = {};
        prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
        // Set the location type first. `location.id` and `location.localized` share storage, so
        // writing one and reading the other gives a plausible but wrong number, not an error.
        prop.location.type                       = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
        prop.location.localized.deviceId         = static_cast<unsigned char>(device);
        prop.location.localized.localityDomainId = static_cast<unsigned char>(domain);

        size_t granularity = 0;
        CU_CHECK(driver.cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size = (std::max<size_t>(bytes, 1) + granularity - 1) / granularity * granularity;

        // The destructor does not run when a constructor throws, so undo the steps that succeeded here.
        try {
            CU_CHECK(driver.cuMemCreate(&handle, size, &prop, 0));
            created = true;
            CU_CHECK(driver.cuMemAddressReserve(&ptr, size, granularity, 0, 0));
            reserved = true;
            CU_CHECK(driver.cuMemMap(ptr, size, 0, handle, 0));
            mapped = true;

            // Access is granted to the device, whichever domain holds the memory.
            CUmemAccessDesc access = {};
            access.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id     = device;
            access.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(driver.cuMemSetAccess(ptr, size, &access, 1));
        } catch (...) {
            release();
            throw;
        }
    }

    LocalizedBuffer(LocalizedBuffer const &) = delete;
    LocalizedBuffer &
    operator=(LocalizedBuffer const &) = delete;

    ~LocalizedBuffer() { release(); }

    // The device address, which is all cuDNN needs.
    void *
    get() const {
        return reinterpret_cast<void *>(ptr);
    }

   private:
    // Undo each allocation step that succeeded, in reverse order. This never throws, so a failure here cannot
    // replace the error that may have led to it. A failed call is reported on stderr instead.
    void
    release() {
        if (mapped) {
            // After an error, copies or kernels using this memory may still be queued: wait before unmapping.
            report(driver.cuCtxSynchronize(), "cuCtxSynchronize");
            report(driver.cuMemUnmap(ptr, size), "cuMemUnmap");
        }
        if (reserved) {
            report(driver.cuMemAddressFree(ptr, size), "cuMemAddressFree");
        }
        if (created) {
            report(driver.cuMemRelease(handle), "cuMemRelease");
        }
    }

    static void
    report(CUresult status, char const *call) {
        if (status != CUDA_SUCCESS) {
            std::cerr << "LocalizedBuffer cleanup: " << call << " failed: " << driver_error_string(status) << " ("
                      << status << ")" << std::endl;
        }
    }

    CUmemGenericAllocationHandle handle = 0;
    CUdeviceptr ptr                     = 0;
    size_t size                         = 0;
    bool created                        = false;
    bool reserved                       = false;
    bool mapped                         = false;
};

// A non-blocking stream for a cuDNN handle. The destructor detaches and destroys it on every exit
// path, including a failed REQUIRE.
struct StreamOwner {
    explicit StreamOwner(cudnnHandle_t handle) : handle(handle) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    StreamOwner(StreamOwner const &) = delete;
    StreamOwner &
    operator=(StreamOwner const &) = delete;
    ~StreamOwner() {
        cudnnSetStream(handle, nullptr);
        cudaStreamDestroy(stream);
    }

    cudnnHandle_t handle;
    cudaStream_t stream = nullptr;
};

// The locality domain holding `ptr`, or -1 if the memory is not localized.
int
locality_domain_of(void const *ptr) {
    int domain = 0;
    CU_CHECK(driver.cuPointerGetAttribute(
        &domain, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, reinterpret_cast<CUdeviceptr>(ptr)));
    return domain;
}

}  // namespace
#endif  // CUDA_VERSION >= 13040

TEST_CASE("Matmul on localized tensors", "[localized][matmul][graph]") {
#if CUDA_VERSION < 13040
    SKIP("Locality domains need CUDA 13.4 or above (CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN)");
#else
    namespace fe = cudnn_frontend;

    // The driver functions themselves are older than CUDA 13.4; the locality-domain values passed to them are not.
    char const *missing = load_driver_api();
    int driver_version  = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    if (driver_version < 13040) {
        SKIP("Locality domains need a CUDA 13.4 or newer driver; this driver is " << driver_version);
    }
    if (missing != nullptr) {
        SKIP("The CUDA driver does not provide " << missing);
    }
    CU_CHECK(driver.cuInit(0));

    int ordinal = 0;
    CUDA_CHECK(cudaGetDevice(&ordinal));
    CUdevice device = 0;
    CU_CHECK(driver.cuDeviceGet(&device, ordinal));

    int num_domains = 0, sms_per_domain = 0, sms_total = 0;
    CU_CHECK(driver.cuDeviceGetAttribute(&num_domains, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device));
    CU_CHECK(
        driver.cuDeviceGetAttribute(&sms_per_domain, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT, device));
    CU_CHECK(driver.cuDeviceGetAttribute(&sms_total, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    std::cout << "device " << ordinal << ": " << num_domains << " locality domain(s), " << sms_per_domain
              << " SMs per domain, " << sms_total << " SMs on the device" << std::endl;
    if (num_domains < 2) {
        SKIP("Device " << ordinal << " reports " << num_domains
                       << " locality domain(s), so a localized allocation would be an ordinary full-device one");
    }

    // An ordinary allocation is not localized.
    Surface<half> ordinary(1);
    REQUIRE(locality_domain_of(ordinary.devPtr) == -1);

    //// Build the graph. Nothing here refers to a locality domain, so one graph serves every domain.
    int64_t const b = 4, m = 64, n = 32, k = 64;
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A = graph->tensor(
        fe::graph::Tensor_attributes().set_name("A").set_uid(A_UID).set_dim({b, m, k}).set_stride({m * k, k, 1}));
    auto B = graph->tensor(
        fe::graph::Tensor_attributes().set_name("B").set_uid(B_UID).set_dim({b, k, n}).set_stride({k * n, n, 1}));
    auto C = graph->matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true).set_uid(C_UID);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    StreamOwner stream_owner(handle);
    cudaStream_t stream = stream_owner.stream;
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
    REQUIRE(graph->check_support().is_good());
    REQUIRE(graph->build_plans().is_good());

    // Every element of C is a dot product of k ones and k halves.
    half const a_value   = __float2half(1.f);
    half const b_value   = __float2half(.5f);
    float const expected = static_cast<float>(k) * __half2float(a_value) * __half2float(b_value);
    std::vector<half> a_host(b * m * k, a_value);
    std::vector<half> b_host(b * k * n, b_value);
    std::vector<half> c_host(b * m * n);
    size_t const a_bytes = sizeof(half) * a_host.size();
    size_t const b_bytes = sizeof(half) * b_host.size();
    size_t const c_bytes = sizeof(half) * c_host.size();

    //// Run the same graph with its tensors in each domain in turn.
    for (int domain = 0; domain < num_domains; ++domain) {
        INFO("locality domain " << domain);
        LocalizedBuffer a_gpu(a_bytes, device, domain);
        LocalizedBuffer b_gpu(b_bytes, device, domain);
        LocalizedBuffer c_gpu(c_bytes, device, domain);
        // Optional: only the tensors need localizing. The workspace is scratch memory and an
        // ordinary allocation works just as well; it is localized here for consistency.
        LocalizedBuffer workspace(graph->get_workspace_size(), device, domain);

        // Ask the driver where the memory landed rather than trusting the request.
        for (LocalizedBuffer const *buffer : {&a_gpu, &b_gpu, &c_gpu, &workspace}) {
            REQUIRE(locality_domain_of(buffer->get()) == domain);
        }

        CUDA_CHECK(cudaMemcpyAsync(a_gpu.get(), a_host.data(), a_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(b_gpu.get(), b_host.data(), b_bytes, cudaMemcpyHostToDevice, stream));
        // Fill C with NaNs so that a kernel writing elsewhere cannot pass.
        CUDA_CHECK(cudaMemsetAsync(c_gpu.get(), 0xFF, c_bytes, stream));

        // Ordinary device pointers: cuDNN never learns the memory was localized.
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
            {A_UID, a_gpu.get()}, {B_UID, b_gpu.get()}, {C_UID, c_gpu.get()}};
        REQUIRE(graph->execute(handle, variant_pack, workspace.get()).is_good());

        CUDA_CHECK(cudaMemcpyAsync(c_host.data(), c_gpu.get(), c_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t mismatches = 0;
        for (half value : c_host) {
            mismatches += (__half2float(value) != expected) ? 1 : 0;
        }
        REQUIRE(mismatches == 0);
        std::cout << "locality domain " << domain << ": matmul on localized tensors gives the expected result"
                  << std::endl;
    }
#endif  // CUDA_VERSION < 13040
}
