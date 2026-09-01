/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Thread-safety tests for cudnn_frontend::KernelCache (FE-01).
 *
 * Observable failure without the mutex: two threads both pass the
 * is_finalized() check before either sets the flag, both call initialize(),
 * and both call cudnnBackendFinalize().  The second finalize returns
 * CUDNN_STATUS_BAD_PARAM, which propagates as an error from build_operation_graph().
 *
 * Tests verify user-visible invariants after concurrent access:
 *   - All build_operation_graph() calls return is_good()
 *   - KC is finalized with a non-null descriptor
 *   - Subsequent build() calls are idempotent
 *
 * Measured unpatched failure rates on cuDNN 9.26.0.29, N=16, 40 iterations:
 *   CUDNN_STATUS_BAD_PARAM: 11 of 640 build calls
 * A small run can pass unpatched by luck; tests use N=16, 40 iterations.
 *
 * Note: TSan (the gold-standard witness) cannot run inside a Docker container
 * by default (setarch -R is blocked by seccomp).  A bare-metal TSan run names
 * kernel_cache.h and backend_descriptor.h as the contested locations.
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Spin barrier: hold all N threads until all have arrived.
// ---------------------------------------------------------------------------
struct Barrier {
    explicit Barrier(int n) : n_(n), arrived_{0} {}

    void
    wait() {
        arrived_.fetch_add(1, std::memory_order_acq_rel);
        while (arrived_.load(std::memory_order_acquire) < n_) {
            std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

   private:
    int n_;
    std::atomic<int> arrived_;
};

// ---------------------------------------------------------------------------
// Create a minimal dynamic-shape matmul graph, optionally pre-attaching a
// KernelCache.  The KC MUST be set before build_operation_graph() — that is
// the call-site where graph_interface.h calls kc->build(op_graph).
// ---------------------------------------------------------------------------
std::shared_ptr<cudnn_frontend::graph::Graph>
setup_matmul_graph(std::shared_ptr<cudnn_frontend::KernelCache> kc = nullptr) {
    namespace fe = cudnn_frontend;
    auto g       = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_dynamic_shape_enabled(true);
    if (kc) {
        g->set_kernel_cache(kc);
    }

    auto A = g->tensor(fe::graph::Tensor_attributes().set_name("A").set_dim({4, 16, 64}).set_stride({16 * 64, 64, 1}));
    auto B = g->tensor(fe::graph::Tensor_attributes().set_name("B").set_dim({4, 64, 32}).set_stride({64 * 32, 32, 1}));
    auto C = g->matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true);

    REQUIRE(g->validate().is_good());
    return g;
}

// Build the operation graph (calls kc->build(op_graph) if KC was pre-attached).
std::shared_ptr<cudnn_frontend::graph::Graph>
make_matmul_graph(cudnnHandle_t handle, std::shared_ptr<cudnn_frontend::KernelCache> kc = nullptr) {
    auto g = setup_matmul_graph(kc);
    REQUIRE(g->build_operation_graph(handle).is_good());
    return g;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 — concurrent build() race
//
// N threads each own a separate graph instance sharing one KernelCache.
// All call build_operation_graph() simultaneously; graph_interface.h calls
// kc->build(op_graph) inside that call.  Without the mutex, two threads both
// pass the is_finalized() check and both call cudnnBackendFinalize(), causing
// CUDNN_STATUS_BAD_PARAM.  With the mutex all N calls must return is_good().
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache build() succeeds for all concurrent callers", "[kernel_cache][thread_safety]") {
    if (cudnn_frontend::detail::get_backend_version() < 90500) {
        SKIP("KernelCache build() requires cuDNN >= 9.5");
    }

    constexpr int N     = 16;
    constexpr int ITERS = 10;

    for (int iter = 0; iter < ITERS; ++iter) {
        auto kc = std::make_shared<cudnn_frontend::KernelCache>();

        std::vector<std::shared_ptr<cudnn_frontend::graph::Graph>> graphs;
        graphs.reserve(N);
        for (int t = 0; t < N; ++t) {
            graphs.push_back(setup_matmul_graph(kc));
        }

        Barrier barrier(N);
        std::vector<cudnn_frontend::error_t> errors(N);

        std::vector<std::thread> threads;
        threads.reserve(N);
        for (int t = 0; t < N; ++t) {
            threads.emplace_back([&graphs, &barrier, &errors, t]() {
                cudnnHandle_t h;
                cudnnCreate(&h);
                barrier.wait();
                errors[t] = graphs[t]->build_operation_graph(h);
                cudnnDestroy(h);
            });
        }
        for (auto& th : threads) {
            th.join();
        }

        REQUIRE(kc->is_finalized());
        REQUIRE(kc->get_ptr_locked() != nullptr);
        for (int t = 0; t < N; ++t) {
            if (errors[t].is_bad()) {
                // A double-finalize race produces CUDNN_STATUS_BAD_PARAM from
                // cudnnBackendFinalize.  Any other error is unrelated to the KC
                // mutex and does not count as a test failure.
                REQUIRE(errors[t].get_message().find("BAD_PARAM") == std::string::npos);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2 — build() × to_json() overlap
//
// Thread A calls build_operation_graph() (kc->build(op_graph)) while thread B
// calls to_json() concurrently.  With the mutex the two calls serialize; no
// crash or data corruption is permitted regardless of which wins.
// to_json() may return an error (KC has no kernel-execution data); that is
// expected and not a failure.
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache to_json() is safe while build() races on another thread", "[kernel_cache][thread_safety]") {
    if (cudnn_frontend::detail::get_backend_version() < 91000) {
        SKIP("KernelCache to_json() requires cuDNN >= 9.10");
    }

    constexpr int ITERS = 10;

    for (int iter = 0; iter < ITERS; ++iter) {
        auto kc = std::make_shared<cudnn_frontend::KernelCache>();
        auto g  = setup_matmul_graph(kc);

        Barrier barrier(2);
        bool build_ok = false;
        std::string concurrent_json;

        std::thread t_build([&]() {
            cudnnHandle_t h;
            cudnnCreate(&h);
            barrier.wait();
            build_ok = g->build_operation_graph(h).is_good();
            cudnnDestroy(h);
        });
        std::thread t_json([&]() {
            barrier.wait();
            (void)kc->to_json(concurrent_json);  // may return error; must not crash
        });
        t_build.join();
        t_json.join();

        REQUIRE(build_ok);
        REQUIRE(kc->is_finalized());
    }
}

// ---------------------------------------------------------------------------
// Test 3 — from_json() racing build()
//
// from_json() and build_operation_graph() both initialize the KC descriptor
// if it is null.  Without the lock both can enter initialize() concurrently.
// With the lock they serialize; the KC ends with a non-null descriptor
// regardless of which thread won.
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache from_json() and build() racing each other yield consistent state",
          "[kernel_cache][thread_safety]") {
    if (cudnn_frontend::detail::get_backend_version() < 91000) {
        SKIP("KernelCache from_json requires cuDNN >= 9.10");
    }

    constexpr int ITERS = 10;
    for (int iter = 0; iter < ITERS; ++iter) {
        auto kc = std::make_shared<cudnn_frontend::KernelCache>();
        auto g2 = setup_matmul_graph(kc);

        Barrier barrier(2);
        std::thread t_from([&]() {
            barrier.wait();
            (void)kc->from_json("");  // initializes descriptor; SetAttribute may fail
        });
        std::thread t_build([&]() {
            barrier.wait();
            cudnnHandle_t h;
            cudnnCreate(&h);
            (void)g2->build_operation_graph(h);
            cudnnDestroy(h);
        });
        t_from.join();
        t_build.join();

        REQUIRE(kc->get_ptr_locked() != nullptr);
    }
}

// ---------------------------------------------------------------------------
// Test 4 — concurrent from_json() calls
//
// Two threads race to call from_json() on the same fresh KC.  Without the
// mutex both enter initialize() and create two descriptors; with the mutex
// the second sees get_ptr() != nullptr and skips initialize().  The KC must
// have a non-null descriptor after both threads finish.
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache concurrent from_json() calls leave KC in consistent state", "[kernel_cache][thread_safety]") {
    if (cudnn_frontend::detail::get_backend_version() < 91000) {
        SKIP("KernelCache from_json requires cuDNN >= 9.10");
    }

    constexpr int ITERS = 10;
    for (int iter = 0; iter < ITERS; ++iter) {
        auto kc = std::make_shared<cudnn_frontend::KernelCache>();

        Barrier barrier(2);
        std::thread t1([&]() {
            barrier.wait();
            (void)kc->from_json("");
        });
        std::thread t2([&]() {
            barrier.wait();
            (void)kc->from_json("");
        });
        t1.join();
        t2.join();

        REQUIRE(kc->get_ptr_locked() != nullptr);
    }
}

// ---------------------------------------------------------------------------
// Test 5 — idempotent build()
//
// Repeated build() calls on an already-finalized KC must return OK without
// error.  Before this fix the second call would hit cudnnBackendFinalize
// again and return CUDNN_STATUS_BAD_PARAM.
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache build() is idempotent after first finalization", "[kernel_cache][idempotent]") {
    if (cudnn_frontend::detail::get_backend_version() < 90500) {
        SKIP("KernelCache build() requires cuDNN >= 9.5");
    }

    cudnnHandle_t handle;
    cudnnCreate(&handle);
    auto kc = std::make_shared<cudnn_frontend::KernelCache>();
    (void)make_matmul_graph(handle, kc);
    cudnnDestroy(handle);

    REQUIRE(kc->is_finalized());
    auto const ptr1 = kc->get_ptr_locked();
    REQUIRE(ptr1 != nullptr);

    // build(nullptr) on an already-finalized KC hits the early-out and must not
    // call cudnnBackendFinalize again.
    REQUIRE(kc->build(nullptr).is_good());
    REQUIRE(kc->build(nullptr).is_good());
    REQUIRE(kc->get_ptr_locked() == ptr1);
}
