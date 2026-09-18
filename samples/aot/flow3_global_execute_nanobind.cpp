// Flow 3, execute step (C++): fetch a published kernel by name and run it,
// from a nanobind extension loaded into the SAME process that registered it.
//
// This is the half of flow 3 that a pure-Python sample cannot show. The kernel
// is compiled by Python; this is a separate shared object, and it reaches the
// same graph through the process-global registry with no file, no container and
// no serialisation anywhere.
//
// That only works because cudnn_frontend::detail::global_registry() is exported
// with default visibility. It is a function-local static in a header, and FE's
// own Python bindings are built with -fvisibility=hidden, so without that this
// module would get a private, empty registry and get_global() would report
// nothing registered. See kernel_library.h.
//
// Built by build.sh into flow3_global_execute_nanobind*.so.

#include <cudnn_frontend/kernel_library.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace fe = cudnn_frontend;

namespace {

// FE returns error_t and never throws; the Python half of the boundary wants an
// exception, so this is where the two conventions meet. Taken by value because
// error_object::is_bad() / get_message() are non-const.
void
check(fe::error_t status, char const *what) {
    if (status.is_bad()) {
        throw std::runtime_error(std::string(what) + ": " + status.get_message());
    }
}

}  // namespace

NB_MODULE(flow3_global_execute_nanobind, m) {
    m.doc() = "C++ executor for a kernel published by cudnn.register_global() in this process";

    m.def(
        "registered_names",
        [] { return fe::registered_global_names(); },
        "What this shared object can see in the process-global registry. Proves it is the same "
        "registry Python wrote to, not a private copy.");

    m.def(
        "variant_pack_uids_sorted",
        [](std::string const &name) {
            fe::graph::Graph graph;
            check(fe::get_global(name, &graph), "get_global");
            return graph.get_variant_pack_uids_sorted();
        },
        nb::arg("name"),
        "The uid order execute() wants pointers in. Resolved once, at startup.");

    m.def(
        "workspace_size",
        [](std::string const &name) {
            fe::graph::Graph graph;
            check(fe::get_global(name, &graph), "get_global");
            int64_t size = 0;
            check(graph.get_workspace_size(size), "get_workspace_size");
            return size;
        },
        nb::arg("name"));

    m.def(
        "execute",
        [](std::string const &name, std::vector<uintptr_t> const &ptrs, uintptr_t workspace, uintptr_t handle) {
            // A snapshot, exactly as in Python: this keeps a reference to the
            // compiled kernel it was fetched with.
            fe::graph::Graph graph;
            check(fe::get_global(name, &graph), "get_global");

            std::vector<void *> raw(ptrs.size());
            for (size_t i = 0; i < ptrs.size(); i++) {
                raw[i] = reinterpret_cast<void *>(ptrs[i]);
            }

            // Identical to the flow-2 C++ executor from here down.
            check(graph.execute(reinterpret_cast<cudnnHandle_t>(handle),
                                raw.data(),
                                static_cast<int>(raw.size()),
                                reinterpret_cast<void *>(workspace)),
                  "execute");
        },
        nb::arg("name"),
        nb::arg("ptrs"),
        nb::arg("workspace"),
        nb::arg("handle"),
        "Execute the named kernel from a pointer array gathered in variant_pack_uids_sorted() order.");
}
