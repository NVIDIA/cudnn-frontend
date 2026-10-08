/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "../../cudnn_frontend_shim.h"

namespace cudnn_frontend {
namespace detail {

// Device memory holding constant data that work reads when it runs: uploaded once, then read in place by
// kernels launched directly or from CUDA graphs. Shared by its owner and by the CUDA graphs that keep it
// alive (as part of a CudaGraphRetainedResource payload).
//
// Freeing it takes CUDA calls, which CUDA user-object destructors may not make, and must not overtake work
// launched directly that still reads it. So dropping the last reference makes no CUDA call: it queues the
// memory, and CudaGraphRetainedResource::drain_deferred_releases() frees it once every stream recorded with
// record_use() has passed that point.
class DeviceConstantBuffer {
   public:
    DeviceConstantBuffer(DeviceConstantBuffer const &) = delete;
    DeviceConstantBuffer &
    operator=(DeviceConstantBuffer const &) = delete;

    // Upload `size` bytes from `host` to new device memory on the current device. Safe while a stream on
    // this or another thread is being captured: relaxed capture mode, private non-blocking stream.
    static cudaError_t
    create(void const *host, size_t size, std::shared_ptr<DeviceConstantBuffer> &out) {
        out.reset();
        std::shared_ptr<DeviceConstantBuffer> buffer(new DeviceConstantBuffer, queue_free);
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        bool const relaxed         = cuda_thread_exchange_stream_capture_mode(&mode) == cudaSuccess;
        cudaError_t status         = cuda_get_device(&buffer->device);
        if (status == cudaSuccess) {
            status = cuda_malloc(&buffer->ptr, size);
        }
        cudaStream_t stream = nullptr;
        if (status == cudaSuccess) {
            status = cuda_stream_create_with_flags(&stream, cudaStreamNonBlocking);
        }
        if (status == cudaSuccess) {
            status = cuda_mem_cpy_async(buffer->ptr, host, size, cudaMemcpyHostToDevice, stream);
            if (status == cudaSuccess) {
                status = cuda_stream_synchronize(stream);
            }
            (void)cuda_stream_destroy(stream);
        }
        if (relaxed) {
            (void)cuda_thread_exchange_stream_capture_mode(&mode);
        }
        if (status == cudaSuccess) {
            out = std::move(buffer);
        }
        return status;  // On failure, `buffer` is queued for a free like any other.
    }

    void *
    data() const {
        return ptr;
    }

    // Order this memory's eventual release after the work queued so far on `stream`, which must not be
    // capturing. Call it after launching work that reads the memory.
    cudaError_t
    record_use(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex);
        cudaEvent_t event = nullptr;
        for (auto const &[used_stream, used_event] : uses) {
            if (used_stream == stream) {
                event = used_event;
            }
        }
        if (event == nullptr) {
            cudaError_t const status = cuda_event_create_with_flags(&event, cudaEventDisableTiming);
            if (status != cudaSuccess) {
                return status;
            }
            uses.emplace_back(stream, event);
        }
        return cuda_event_record(event, stream);
    }

    // Whether released memory awaits a free.
    static bool
    has_deferred_frees() {
        return deferred_frees().pending.load(std::memory_order_acquire);
    }

    // Free released memory that no recorded use still reads. Call in relaxed capture mode, from a thread
    // whose stream is not being captured.
    static void
    drain_deferred_frees() {
        DeferredFrees &deferred = deferred_frees();
        if (!deferred.pending.load(std::memory_order_acquire)) {
            return;
        }
        std::vector<DeviceConstantBuffer *> buffers;
        {
            std::lock_guard<std::mutex> lock(deferred.mutex);
            buffers.swap(deferred.buffers);
            deferred.pending.store(false, std::memory_order_release);
        }
        std::vector<DeviceConstantBuffer *> in_use;
        for (DeviceConstantBuffer *buffer : buffers) {
            bool busy = false;
            for (auto const &[stream, event] : buffer->uses) {
                (void)stream;
                busy = busy || cuda_event_query(event) == cudaErrorNotReady;
            }
            (void)cuda_get_last_error();
            if (busy) {
                in_use.push_back(buffer);
                continue;
            }
            // The draining thread may have another device current.
            int current_device       = buffer->device;
            bool const switch_device = cuda_get_device(&current_device) == cudaSuccess &&
                                       current_device != buffer->device &&
                                       cuda_set_device(buffer->device) == cudaSuccess;
            for (auto const &[stream, event] : buffer->uses) {
                (void)stream;
                (void)cuda_event_destroy(event);
            }
            if (buffer->ptr != nullptr) {
                (void)cuda_free(buffer->ptr);
            }
            if (switch_device) {
                (void)cuda_set_device(current_device);
            }
            (void)cuda_get_last_error();
            delete buffer;
        }
        if (!in_use.empty()) {
            std::lock_guard<std::mutex> lock(deferred.mutex);
            deferred.buffers.insert(deferred.buffers.end(), in_use.begin(), in_use.end());
            deferred.pending.store(true, std::memory_order_release);
        }
    }

    // For tests: released memory not freed yet.
    static size_t
    deferred_free_count() {
        DeferredFrees &deferred = deferred_frees();
        std::lock_guard<std::mutex> lock(deferred.mutex);
        return deferred.buffers.size();
    }

   private:
    DeviceConstantBuffer()  = default;
    ~DeviceConstantBuffer() = default;

    struct DeferredFrees {
        std::mutex mutex;
        std::vector<DeviceConstantBuffer *> buffers;
        std::atomic<bool> pending{false};
    };

    // Intentionally leaked, like CudaGraphRetainedResource's queue.
    static DeferredFrees &
    deferred_frees() {
        static DeferredFrees &instance = *new DeferredFrees;
        return instance;
    }

    // shared_ptr deleter; may run in a CUDA user-object destructor, so it makes no CUDA call.
    static void
    queue_free(DeviceConstantBuffer *buffer) {
        DeferredFrees &deferred = deferred_frees();
        std::lock_guard<std::mutex> lock(deferred.mutex);
        deferred.buffers.push_back(buffer);
        deferred.pending.store(true, std::memory_order_release);
    }

    void *ptr  = nullptr;
    int device = 0;
    std::mutex mutex;
    std::vector<std::pair<cudaStream_t, cudaEvent_t>> uses;  // one event per stream
};

// Keeps a host-side resource alive for as long as any CUDA graph that was recorded against it
// exists, including graphExecs instantiated from it and clones of it.
//
// The graph holds the resource through a CUDA user object whose payload is a shared_ptr to the
// resource. One user object is created per CudaGraphRetainedResource on first use; every
// graph that is recorded against the resource then adds one reference to it, so retaining on
// the same graph repeatedly (populate, then update every iteration) costs a reference count,
// not an allocation. The resource itself holds one reference until it is destroyed, which is
// what makes reusing the object across graphs race-free: it cannot reach zero while its owner
// can still record work.
//
// The payload is deliberately type-erased so this serves every resource whose lifetime must
// follow the graphs that reference it: the cuDNN execution plan behind recorded kernel launches
// (its runtime-compiled code is released with the plan), or host buffers that memcpy nodes
// read from.
//
// CUDA forbids CUDA API calls in user-object destructors, and releasing a payload may make them
// (e.g. cudnnBackendDestroyDescriptor frees device memory). The destructor therefore only
// queues the payload. Those calls would also invalidate an active stream capture, so the queue
// is drained by the next execution that is NOT being captured. A payload queued after the last
// such execution stays alive until process exit.
class CudaGraphRetainedResource {
   public:
    CudaGraphRetainedResource() = default;

    // Give `graph` one reference to the payload. `make_payload()` is invoked once, on the first
    // call, and must return a non-null std::shared_ptr<void> (or convertible) that owns
    // everything the graph has to keep alive; the payload is then fixed for this object's life.
    template <typename MakePayload>
    cudaError_t
    retain_on_graph(cudaGraph_t graph, MakePayload &&make_payload) {
        if (graph == nullptr) {
            return cudaErrorInvalidValue;
        }
        std::lock_guard<std::mutex> lock(*mutex);
        if (state == nullptr) {
            std::shared_ptr<void> payload = make_payload();
            if (payload == nullptr) {
                return cudaErrorInvalidValue;
            }
            state = std::make_shared<State>(std::move(payload));
        }
        if (state->user_object == nullptr) {
            auto *payload_ref               = new std::shared_ptr<void>(state->payload);
            cudaError_t const create_status = cuda_user_object_create(
                &state->user_object, payload_ref, release_graph_ref, 1, cudaUserObjectNoDestructorSync);
            if (create_status != cudaSuccess) {
                delete payload_ref;
                state->user_object = nullptr;
                return create_status;
            }
        }
        return cuda_graph_retain_user_object(graph, state->user_object, 1, /*flags=*/0);
    }

    // Same as retain_on_graph() for the graph `stream` is currently capturing into. When the
    // stream is not capturing, this is the point at which queued payloads are released instead.
    template <typename MakePayload>
    cudaError_t
    retain_on_capturing_stream(cudaStream_t stream, MakePayload &&make_payload) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cudaGraph_t capture_graph              = nullptr;
        cudaError_t const query_status         = cuda_stream_get_capture_info(stream, &capture_status, &capture_graph);
        if (query_status == cudaErrorStreamCaptureImplicit) {
            // Legacy default stream while another stream captures in global mode: not our
            // capture, and not a safe moment to release anything either. The recorded work
            // itself reports the problem if there is one.
            (void)cuda_get_last_error();
            return cudaSuccess;
        }
        if (query_status != cudaSuccess) {
            return query_status;
        }
        if (capture_status == cudaStreamCaptureStatusNone) {
            drain_deferred_releases();
            return cudaSuccess;
        }
        if (capture_status != cudaStreamCaptureStatusActive || capture_graph == nullptr) {
            return cudaSuccess;
        }
        return retain_on_graph(capture_graph, std::forward<MakePayload>(make_payload));
    }

    // Whether user-object destructors have handed back payloads that still await release.
    static bool
    has_deferred_releases() {
        return deferred_releases().pending.load(std::memory_order_acquire) ||
               DeviceConstantBuffer::has_deferred_frees();
    }

    // Release queued payloads if `stream` is not being captured. For callers that retain nothing
    // themselves; cheap (no CUDA call) when nothing is queued.
    static cudaError_t
    drain_deferred_releases_on_stream(cudaStream_t stream) {
        if (!has_deferred_releases()) {
            return cudaSuccess;
        }
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cudaGraph_t capture_graph              = nullptr;
        cudaError_t const query_status         = cuda_stream_get_capture_info(stream, &capture_status, &capture_graph);
        if (query_status == cudaErrorStreamCaptureImplicit) {
            (void)cuda_get_last_error();
            return cudaSuccess;
        }
        if (query_status != cudaSuccess) {
            return query_status;
        }
        if (capture_status == cudaStreamCaptureStatusNone) {
            drain_deferred_releases();
        }
        return cudaSuccess;
    }

    // Release payloads handed back by user-object destructors (see class comment).
    static void
    drain_deferred_releases() {
        DeferredReleases &deferred = deferred_releases();
        if (!has_deferred_releases()) {
            return;
        }
        std::vector<std::shared_ptr<void>> releases;
        {
            std::lock_guard<std::mutex> lock(deferred.mutex);
            releases.swap(deferred.payloads);
            deferred.pending.store(false, std::memory_order_release);
        }
        // The caller's stream is not capturing, but another stream (on this or another thread)
        // may be capturing in global mode, which forbids potentially unsafe CUDA calls from every
        // thread that is not in relaxed mode. Release in relaxed mode so the payload destructors
        // (device frees, kernel unloads) cannot invalidate someone else's capture.
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        bool const relaxed         = cuda_thread_exchange_stream_capture_mode(&mode) == cudaSuccess;
        releases.clear();
        DeviceConstantBuffer::drain_deferred_frees();  // including memory the payloads just released
        if (relaxed) {
            (void)cuda_thread_exchange_stream_capture_mode(&mode);
        }
    }

   private:
    struct State {
        explicit State(std::shared_ptr<void> payload_) : payload(std::move(payload_)) {}
        ~State() {
            if (user_object != nullptr) {
                // Drop the owner's reference. Graphs that still hold references keep the payload
                // alive; release_graph_ref() runs once the last of them is gone.
                (void)cuda_user_object_release(user_object, 1);
            }
        }
        State(State const &) = delete;
        State &
        operator=(State const &) = delete;

        std::shared_ptr<void> payload;
        cudaUserObject_t user_object = nullptr;
    };

    struct DeferredReleases {
        std::mutex mutex;
        std::vector<std::shared_ptr<void>> payloads;
        std::atomic<bool> pending{false};
    };

    // Intentionally leaked: CUDA may run user-object destructors during process teardown,
    // after static destruction, and destroying the queue could itself run payload destructors
    // that call into CUDA.
    static DeferredReleases &
    deferred_releases() {
        static DeferredReleases &instance = *new DeferredReleases;
        return instance;
    }

    static void CUDART_CB
    release_graph_ref(void *opaque) {
        auto *payload_ref          = static_cast<std::shared_ptr<void> *>(opaque);
        DeferredReleases &deferred = deferred_releases();
        {
            std::lock_guard<std::mutex> lock(deferred.mutex);
            deferred.payloads.push_back(std::move(*payload_ref));
            deferred.pending.store(true, std::memory_order_release);
        }
        delete payload_ref;
    }

    // Copies of the owning object (e.g. copied execution plans) share one state and one user
    // object; the reference the owner holds is released when the last copy goes away.
    std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();
    std::shared_ptr<State> state;
};

}  // namespace detail
}  // namespace cudnn_frontend
