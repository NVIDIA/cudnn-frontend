// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <xla/ffi/api/ffi.h>

#include <cstdint>
#include <array>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ffi = xla::ffi;

struct Argument {
    int64_t kind, index, offset, ndim, shape[8], code, bits, integer;
    double floating;
};
struct StageSpec {
    void* function;
    int64_t count;
    const Argument* arguments;
};
struct Stage {
    void* function;
    std::vector<Argument> arguments;
};
struct Plan {
    int device;
    int inputs;
    std::vector<Stage> stages;
    std::vector<Argument> zeros;
};

// Plans stay alive for every compiled XLA executable in this process.
static std::mutex plans_mutex;
static std::unordered_map<int64_t, const Plan*> plans;

extern "C" void*
cake_jax_plan(int device, int inputs, int count, const StageSpec* stages, int zero_count, const Argument* zeros) {
    auto* plan = new Plan{device, inputs, {}, {}};
    for (int i = 0; i < count; ++i) {
        plan->stages.push_back({stages[i].function, {stages[i].arguments, stages[i].arguments + stages[i].count}});
        TVMFFIObjectIncRef(stages[i].function);
    }
    if (zero_count) plan->zeros.assign(zeros, zeros + zero_count);
    std::lock_guard<std::mutex> lock(plans_mutex);
    plans.emplace(reinterpret_cast<int64_t>(plan), plan);
    return plan;
}

static ffi::Error
Launch(cudaStream_t stream, ffi::RemainingArgs inputs, ffi::RemainingRets outputs, int64_t plan_address) {
    const Plan* prepared;
    {
        std::lock_guard<std::mutex> lock(plans_mutex);
        auto found = plans.find(plan_address);
        if (found == plans.end())
            return ffi::Error::InvalidArgument("unknown Cake JAX plan; prepare it in this process");
        prepared = found->second;
    }
    const auto& plan = *prepared;
    if (inputs.size() != static_cast<size_t>(plan.inputs)) return ffi::Error::InvalidArgument("wrong Cake input count");
    int device;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
    if (device != plan.device) return ffi::Error::InvalidArgument("Cake JAX plan was compiled for another GPU");
    std::vector<void*> buffers;
    std::vector<size_t> capacities;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto value = inputs.get<ffi::AnyBuffer>(i);
        if (!value.has_value()) return ffi::Error::InvalidArgument("invalid Cake input buffer");
        buffers.push_back(value->untyped_data());
        capacities.push_back(value->size_bytes());
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto value = outputs.get<ffi::AnyBuffer>(i);
        if (!value.has_value()) return ffi::Error::InvalidArgument("invalid Cake output buffer");
        buffers.push_back((*value)->untyped_data());
        capacities.push_back((*value)->size_bytes());
    }
    auto valid = [&](const Argument& arg) {
        if (arg.kind != kTVMFFIDLTensorPtr) return true;
        if (arg.index < 0 || static_cast<size_t>(arg.index) >= buffers.size() || arg.offset < 0) return false;
        size_t bytes = arg.bits / 8;
        for (int i = 0; i < arg.ndim; ++i) bytes *= arg.shape[i];
        auto offset = static_cast<size_t>(arg.offset);
        return offset <= capacities[arg.index] && bytes <= capacities[arg.index] - offset &&
               (reinterpret_cast<uintptr_t>(buffers[arg.index]) + offset) % 16 == 0;
    };
    for (const auto& stage : plan.stages)
        for (const auto& arg : stage.arguments)
            if (!valid(arg)) return ffi::Error::InvalidArgument("invalid Cake buffer capacity or alignment");
    for (const auto& zero : plan.zeros)
        if (!valid(zero)) return ffi::Error::InvalidArgument("invalid Cake workspace capacity or alignment");
    for (const auto& zero : plan.zeros) {
        size_t bytes = zero.bits / 8;
        for (int i = 0; i < zero.ndim; ++i) bytes *= zero.shape[i];
        status = cudaMemsetAsync(static_cast<char*>(buffers.at(zero.index)) + zero.offset, 0, bytes, stream);
        if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
    }
    TVMFFIStreamHandle previous;
    if (TVMFFIEnvSetStream(kDLCUDA, device, stream, &previous))
        return ffi::Error::Internal("cannot set Cake CUDA stream");
    struct Restore {
        int device;
        TVMFFIStreamHandle previous;
        ~Restore() { TVMFFIEnvSetStream(kDLCUDA, device, previous, nullptr); }
    } restore{device, previous};
    for (const auto& stage : plan.stages) {
        std::vector<TVMFFIAny> args(stage.arguments.size());
        std::vector<DLTensor> tensors(stage.arguments.size());
        std::vector<std::array<int64_t, 8>> strides(stage.arguments.size());
        for (size_t i = 0; i < args.size(); ++i) {
            const auto& arg    = stage.arguments[i];
            args[i].type_index = arg.kind;
            if (arg.kind == kTVMFFIDLTensorPtr) {
                auto& tensor   = tensors[i];
                tensor.data    = static_cast<char*>(buffers.at(arg.index)) + arg.offset;
                tensor.device  = {kDLCUDA, device};
                tensor.ndim    = arg.ndim;
                tensor.dtype   = {static_cast<uint8_t>(arg.code), static_cast<uint8_t>(arg.bits), 1};
                tensor.shape   = const_cast<int64_t*>(arg.shape);
                int64_t stride = 1;
                for (int d = arg.ndim - 1; d >= 0; --d) {
                    strides[i][d] = stride;
                    stride *= arg.shape[d];
                }
                tensor.strides = strides[i].data();
                args[i].v_ptr  = &tensor;
            } else if (arg.kind == kTVMFFIFloat) {
                args[i].v_float64 = arg.floating;
            } else {
                args[i].v_int64 = arg.integer;
            }
        }
        TVMFFIAny result{};
        if (TVMFFIFunctionCall(stage.function, args.data(), args.size(), &result)) {
            TVMFFIObjectHandle error = nullptr;
            TVMFFIErrorMoveFromRaised(&error);
            std::string message = "Cake kernel launch failed";
            if (error) {
                auto* cell = TVMFFIErrorGetCellPtr(error);
                message.assign(cell->message.data, cell->message.size);
                TVMFFIObjectDecRef(error);
            }
            return ffi::Error::Internal(message);
        }
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cake_jax_launch,
    Launch,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets().Attr<int64_t>("plan"));
