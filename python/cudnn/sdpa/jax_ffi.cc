// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <cudnn_frontend.h>
#include <xla/ffi/api/ffi.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace ffi = xla::ffi;
struct Plan {
    cudnn_frontend::graph::Graph graph;
    cudnnHandle_t handle = nullptr;
    int device;
    std::vector<int64_t> uids;
    std::vector<int64_t> bytes;
    std::mutex mutex;
    ~Plan() {
        if (handle) cudnnDestroy(handle);
    }
};
// Executables may outlive Python tracing caches. Native plans remain valid until exit.
static std::mutex registry_mutex;
static std::unordered_map<int64_t, std::unique_ptr<Plan>> plans;
static thread_local std::string last_error;
extern "C" const char*
cudnn_jax_error() {
    return last_error.c_str();
}
extern "C" int64_t
cudnn_jax_prepare(const uint8_t* blob, size_t size, const int64_t* uids, const int64_t* bytes, size_t count) {
    try {
        auto plan = std::make_unique<Plan>();
        if (cudaGetDevice(&plan->device) != cudaSuccess || cudnnCreate(&plan->handle) != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error("cannot create cuDNN JAX handle");
        auto status = plan->graph.deserialize(plan->handle, std::vector<uint8_t>(blob, blob + size));
        if (status.is_bad()) throw std::runtime_error(status.get_message());
        plan->uids.assign(uids, uids + count);
        plan->bytes.assign(bytes, bytes + count);
        std::lock_guard<std::mutex> lock(registry_mutex);
        int64_t id = plans.size() + 1;
        plans.emplace(id, std::move(plan));
        return id;
    } catch (const std::exception& e) {
        last_error = e.what();
        return 0;
    }
}
static ffi::Error
Execute(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets, int64_t id) {
    Plan* plan;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it = plans.find(id);
        if (it == plans.end()) return ffi::Error::InvalidArgument("unknown cuDNN JAX plan");
        plan = it->second.get();
    }
    int device;
    if (cudaGetDevice(&device) != cudaSuccess || device != plan->device)
        return ffi::Error::InvalidArgument("cuDNN JAX plan belongs to another device");
    if (args.size() + rets.size() != plan->uids.size() + 1)
        return ffi::Error::InvalidArgument("cuDNN JAX operand count mismatch");
    std::unordered_map<int64_t, void*> pointers;
    for (size_t i = 0; i < plan->uids.size(); ++i) {
        void* ptr;
        size_t bytes;
        if (i < args.size()) {
            auto buf = args.get<ffi::AnyBuffer>(i);
            if (!buf.has_value()) return ffi::Error::InvalidArgument("invalid input");
            ptr   = buf->untyped_data();
            bytes = buf->size_bytes();
        } else {
            auto buf = rets.get<ffi::AnyBuffer>(i - args.size());
            if (!buf.has_value()) return ffi::Error::InvalidArgument("invalid output");
            ptr   = (*buf)->untyped_data();
            bytes = (*buf)->size_bytes();
        }
        if (bytes != static_cast<size_t>(plan->bytes[i])) return ffi::Error::InvalidArgument("operand size mismatch");
        pointers.emplace(plan->uids[i], ptr);
    }
    auto workspace = rets.get<ffi::AnyBuffer>(rets.size() - 1);
    if (!workspace.has_value() || (*workspace)->size_bytes() < static_cast<size_t>(plan->graph.get_workspace_size()))
        return ffi::Error::InvalidArgument("insufficient cuDNN workspace");
    std::lock_guard<std::mutex> lock(plan->mutex);
    if (cudnnSetStream(plan->handle, stream) != CUDNN_STATUS_SUCCESS)
        return ffi::Error::Internal("cannot set cuDNN stream");
    auto status = plan->graph.execute(plan->handle, pointers, (*workspace)->untyped_data());
    return status.is_bad() ? ffi::Error::Internal(status.get_message()) : ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cudnn_jax_execute,
    Execute,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets().Attr<int64_t>("plan"));
