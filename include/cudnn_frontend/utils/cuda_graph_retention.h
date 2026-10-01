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

    // Release payloads handed back by user-object destructors (see class comment).
    static void
    drain_deferred_releases() {
        DeferredReleases &deferred = deferred_releases();
        if (!deferred.pending.load(std::memory_order_acquire)) {
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
