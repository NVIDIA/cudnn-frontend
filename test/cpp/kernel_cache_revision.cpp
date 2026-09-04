/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

// The two accessors are available starting cuDNN 9.27.
constexpr int64_t MIN_BACKEND_VERSION = 92700;

struct CacheState {
    int64_t revision = -1;
    int64_t size     = -1;
};

// Read both accessors from a finalized kernel cache.
CacheState
read_state(std::shared_ptr<fe::KernelCache> const& kc) {
    CacheState state;
    REQUIRE(kc->revision(state.revision).is_good());
    REQUIRE(kc->size(state.size).is_good());
    return state;
}

// ---------------------------------------------------------------------------
// A minimal dynamic-shape matmul graph.  Only m changes between the shapes, and
// the tensors stay in the low kilobytes.  The tests never execute the graph, so
// no device memory is allocated and only plan compilation costs anything.
//
// The kernel cache MUST be attached before build_operation_graph() -- that is
// the call site where graph_interface.h calls kc->build(op_graph).
// ---------------------------------------------------------------------------
std::shared_ptr<fe::graph::Graph>
make_matmul_graph(std::shared_ptr<fe::KernelCache> kc, int64_t m) {
    constexpr int64_t b = 1;
    constexpr int64_t n = 32;
    constexpr int64_t k = 32;

    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_dynamic_shape_enabled(true);
    g->set_kernel_cache(kc);

    auto A = g->tensor(fe::graph::Tensor_attributes().set_name("A").set_dim({b, m, k}).set_stride({m * k, k, 1}));
    auto B = g->tensor(fe::graph::Tensor_attributes().set_name("B").set_dim({b, k, n}).set_stride({k * n, n, 1}));
    auto C = g->matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true);

    REQUIRE(g->validate().is_good());
    return g;
}

// Build the operation graph.  This finalizes the kernel cache but compiles no
// plan, so it cannot change the number of entries.
std::shared_ptr<fe::graph::Graph>
finalize_cache(cudnnHandle_t handle, std::shared_ptr<fe::KernelCache> kc, int64_t m) {
    auto g = make_matmul_graph(kc, m);
    REQUIRE(g->build_operation_graph(handle).is_good());
    return g;
}

// Compile the plan.  This is the only step that can add an entry.
void
build_plan(cudnnHandle_t handle, std::shared_ptr<fe::graph::Graph> const& g) {
    REQUIRE(g->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
    REQUIRE(g->check_support(handle).is_good());
    REQUIRE(g->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1 -- both accessors need a finalized cache
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
    if (fe::detail::get_backend_version() < MIN_BACKEND_VERSION) {
        SKIP("KernelCache revision()/size() require cuDNN >= 9.27");
    }

    {
        auto never_built = std::make_shared<fe::KernelCache>();
        int64_t value    = 0;
        REQUIRE(never_built->revision(value).is_bad());
        REQUIRE(never_built->size(value).is_bad());
    }

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    auto source = std::make_shared<fe::KernelCache>();
    auto g      = finalize_cache(handle, source, 32);
    build_plan(handle, g);

    std::string blob;
    REQUIRE(source->to_json(blob).is_good());
    REQUIRE(!blob.empty());

    auto loaded = std::make_shared<fe::KernelCache>();
    REQUIRE(loaded->from_json(blob).is_good());
    {
        int64_t value = 0;
        REQUIRE(loaded->revision(value).is_bad());
        REQUIRE(loaded->size(value).is_bad());
    }

    (void)finalize_cache(handle, loaded, 32);
    auto const baseline = read_state(loaded);
    REQUIRE(baseline.size >= 1);
    REQUIRE(baseline.revision == baseline.size);

    cudnnDestroy(handle);
}

// ---------------------------------------------------------------------------
// Test 2 -- the counter starts at zero, an insertion adds one, a lookup adds none
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache revision() counts insertions and ignores lookups", "[kernel_cache][revision]") {
    if (fe::detail::get_backend_version() < MIN_BACKEND_VERSION) {
        SKIP("KernelCache revision()/size() require cuDNN >= 9.27");
    }

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    auto kc = std::make_shared<fe::KernelCache>();
    auto g  = finalize_cache(handle, kc, 32);

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
    auto g_repeat = finalize_cache(handle, kc, 32);
    build_plan(handle, g_repeat);
    auto const repeat = read_state(kc);
    REQUIRE(repeat.revision == first.revision);
    REQUIRE(repeat.size == first.size);

    cudnnDestroy(handle);
}

// ---------------------------------------------------------------------------
// Test 3 -- revision() advances exactly as many times as size() grows
//
// The kernel cache keys its entries on the engine configuration, and not on the
// tensor shape.  Thus two different shapes can share one entry, and a new shape
// is not necessarily a new entry.  A test that expects a given shape to add an
// entry is a test of the heuristics, and it breaks on a new architecture or a
// new cuDNN version.
//
// This test asserts only the invariant that holds for a hit and for an
// insertion.  These graphs cause no replacement and no eviction, so the two
// deltas must agree at each step.  The ladder stops as soon as it has seen both
// a hit and an insertion, which keeps the number of plan compilations low.
// ---------------------------------------------------------------------------
TEST_CASE("KernelCache revision() grows with size()", "[kernel_cache][revision]") {
    if (fe::detail::get_backend_version() < MIN_BACKEND_VERSION) {
        SKIP("KernelCache revision()/size() require cuDNN >= 9.27");
    }

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    std::vector<int64_t> const ladder = {16, 32, 64, 128, 256, 512};

    auto kc = std::make_shared<fe::KernelCache>();
    CacheState previous;
    bool saw_insertion = false;
    bool saw_hit       = false;

    for (size_t i = 0; i < ladder.size() && !(saw_insertion && saw_hit); ++i) {
        auto g = finalize_cache(handle, kc, ladder[i]);
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

    if (!saw_insertion || !saw_hit) {
        FAIL(
            "the shape ladder gave no cache hit or no cache insertion, so the invariant was not "
            "tested on both paths");
    }
}
