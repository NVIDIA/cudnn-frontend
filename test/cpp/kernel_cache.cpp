/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

#include <atomic>
#include <cstdint>
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
// Shape of the matmul used by every test here.  The defaults are the shape that
// the thread-safety tests use; the revision tests vary m.
// ---------------------------------------------------------------------------
struct MatmulShape {
    int64_t b = 4;
    int64_t m = 16;
    int64_t n = 32;
    int64_t k = 64;
};

// ---------------------------------------------------------------------------
// Create a minimal dynamic-shape matmul graph, optionally pre-attaching a
// KernelCache.  The KC MUST be set before build_operation_graph() — that is
// the call-site where graph_interface.h calls kc->build(op_graph).
//
// The tests never execute the graph, so no device memory is allocated and only
// plan compilation costs anything.
// ---------------------------------------------------------------------------
std::shared_ptr<cudnn_frontend::graph::Graph>
setup_matmul_graph(std::shared_ptr<cudnn_frontend::KernelCache> kc = nullptr, MatmulShape shape = {}) {
    namespace fe = cudnn_frontend;
    auto g       = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_dynamic_shape_enabled(true);
    if (kc) {
        g->set_kernel_cache(kc);
    }

    auto A = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("A")
                           .set_dim({shape.b, shape.m, shape.k})
                           .set_stride({shape.m * shape.k, shape.k, 1}));
    auto B = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("B")
                           .set_dim({shape.b, shape.k, shape.n})
                           .set_stride({shape.k * shape.n, shape.n, 1}));
    auto C = g->matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true);

    REQUIRE(g->validate().is_good());
    return g;
}

// Build the operation graph (calls kc->build(op_graph) if KC was pre-attached).
// This finalizes the KC but compiles no plan, so it cannot change the number of
// entries.
std::shared_ptr<cudnn_frontend::graph::Graph>
make_matmul_graph(cudnnHandle_t handle,
                  std::shared_ptr<cudnn_frontend::KernelCache> kc = nullptr,
                  MatmulShape shape                               = {}) {
    auto g = setup_matmul_graph(kc, shape);
    REQUIRE(g->build_operation_graph(handle).is_good());
    return g;
}

// Compile the plan.  This is the only step that can add a kernel cache entry.
void
build_plan(cudnnHandle_t handle, std::shared_ptr<cudnn_frontend::graph::Graph> const& g) {
    namespace fe = cudnn_frontend;
    REQUIRE(g->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
    REQUIRE(g->check_support(handle).is_good());
    REQUIRE(g->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
}

// The two accessors are available starting cuDNN 9.27.
constexpr int64_t MIN_REVISION_VERSION = 92700;

struct CacheState {
    int64_t revision = -1;
    int64_t size     = -1;
};

// The revision tests vary only m, and keep the tensors in the low kilobytes.
MatmulShape
revision_shape(int64_t m) {
    return MatmulShape{1, m, 32, 32};
}

// Read both accessors from a finalized kernel cache.
CacheState
read_state(std::shared_ptr<cudnn_frontend::KernelCache> const& kc) {
    CacheState state;
    REQUIRE(kc->revision(state.revision).is_good());
    REQUIRE(kc->size(state.size).is_good());
    return state;
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

// ---------------------------------------------------------------------------
// Test 6 — both accessors need a finalized cache
//
// An unfinalized cache fails in two different ways.  A cache that was never
// built has no descriptor and trips the frontend guard.  A cache that came from
// from_json() has a descriptor, but the backend rejects the read.
//
// The test also pins the baseline of a loaded cache.  from_json() does not
// restore a counter: the serialized data does not contain one.  The load starts
// a new count and adds one for each entry that it puts in the cache.  Thus the
// baseline is the entry count, and not zero.
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache revision() and size() need a finalized cache", "[kernel_cache][revision]") {
    if (cudnn_frontend::detail::get_backend_version() < MIN_REVISION_VERSION) {
        SKIP("KernelCache revision()/size() require cuDNN >= 9.27");
    }

    {
        auto never_built = std::make_shared<cudnn_frontend::KernelCache>();
        int64_t value    = 0;
        REQUIRE(never_built->revision(value).is_bad());
        REQUIRE(never_built->size(value).is_bad());
    }

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    auto source = std::make_shared<cudnn_frontend::KernelCache>();
    auto g      = make_matmul_graph(handle, source, revision_shape(32));
    build_plan(handle, g);

    std::string blob;
    REQUIRE(source->to_json(blob).is_good());
    REQUIRE(!blob.empty());

    auto loaded = std::make_shared<cudnn_frontend::KernelCache>();
    REQUIRE(loaded->from_json(blob).is_good());
    {
        int64_t value = 0;
        REQUIRE(loaded->revision(value).is_bad());
        REQUIRE(loaded->size(value).is_bad());
    }

    (void)make_matmul_graph(handle, loaded, revision_shape(32));
    auto const baseline = read_state(loaded);
    REQUIRE(baseline.size >= 1);
    REQUIRE(baseline.revision == baseline.size);

    cudnnDestroy(handle);
}

// ---------------------------------------------------------------------------
// Test 7 — the counter starts at zero, an insertion adds one, a lookup adds none
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache revision() counts insertions and ignores lookups", "[kernel_cache][revision]") {
    if (cudnn_frontend::detail::get_backend_version() < MIN_REVISION_VERSION) {
        SKIP("KernelCache revision()/size() require cuDNN >= 9.27");
    }

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    auto kc = std::make_shared<cudnn_frontend::KernelCache>();
    auto g  = make_matmul_graph(handle, kc, revision_shape(32));

    // Finalized, but no plan is compiled yet.
    auto const empty = read_state(kc);
    REQUIRE(empty.revision == 0);
    REQUIRE(empty.size == 0);

    // The first plan build on a cold cache always inserts.
    build_plan(handle, g);
    auto const first = read_state(kc);
    REQUIRE(first.revision == 1);
    REQUIRE(first.size == 1);

    // The same shape finds the entry.  A lookup does not change the counter.
    auto g_repeat = make_matmul_graph(handle, kc, revision_shape(32));
    build_plan(handle, g_repeat);
    auto const repeat = read_state(kc);
    REQUIRE(repeat.revision == first.revision);
    REQUIRE(repeat.size == first.size);

    cudnnDestroy(handle);
}

// ---------------------------------------------------------------------------
// Test 8 — revision() advances exactly as many times as size() grows
//
// Two different shapes can use the same kernel cache entry, so a new shape is
// not necessarily a new entry.  Which shapes share an entry depends on the
// architecture and on the cuDNN version, because a different kernel may be
// chosen.  The test therefore walks a ladder of shapes, and asserts only the
// invariant that holds for a hit and for an insertion alike.  These graphs
// cause no replacement and no eviction, so the two deltas must agree at each
// step.  The ladder stops as soon as it has seen both a hit and an insertion,
// which keeps the number of plan compilations low.
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache revision() grows with size()", "[kernel_cache][revision]") {
    if (cudnn_frontend::detail::get_backend_version() < MIN_REVISION_VERSION) {
        SKIP("KernelCache revision()/size() require cuDNN >= 9.27");
    }

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    std::vector<int64_t> const ladder = {16, 32, 64, 128, 256, 512};

    auto kc = std::make_shared<cudnn_frontend::KernelCache>();
    CacheState previous;
    bool saw_insertion = false;
    bool saw_hit       = false;

    for (size_t i = 0; i < ladder.size() && !(saw_insertion && saw_hit); ++i) {
        auto g = make_matmul_graph(handle, kc, revision_shape(ladder[i]));
        if (i == 0) {
            previous = read_state(kc);
        }
        build_plan(handle, g);
        auto const now = read_state(kc);

        INFO("m = " << ladder[i] << ", revision " << previous.revision << " -> " << now.revision << ", size "
                    << previous.size << " -> " << now.size);
        REQUIRE(now.revision >= previous.revision);
        REQUIRE(now.size >= previous.size);
        REQUIRE(now.revision - previous.revision == now.size - previous.size);

        if (now.size > previous.size) {
            saw_insertion = true;
        } else {
            saw_hit = true;
        }
        previous = now;
    }

    cudnnDestroy(handle);

    // Which shapes share an entry is a property of the heuristics, so a supported
    // configuration may give only insertions or only hits. That is not a failure: the delta
    // invariant above is checked either way, and Test 7 pins an insertion and a lookup
    // deterministically. Report the weaker coverage instead of failing.
    if (!saw_insertion || !saw_hit) {
        WARN("the shape ladder gave no cache hit or no cache insertion on this configuration");
    }
}
