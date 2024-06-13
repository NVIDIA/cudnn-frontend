#pragma once

#include <vector>

#include "cudnn.h"

#include "backend_descriptor.h"

namespace cudnn_frontend::detail {
/**
 * @brief Creates a CUDNN backend variant pack descriptor.
 *
 * This function creates a `backend_descriptor` object representing a CUDNN backend variant pack
 * descriptor. The variant pack descriptor is configured with the provided device pointers, unique
 * IDs, and a workspace pointer.
 *
 * @param[out] variant_pack The created `backend_descriptor` object representing the variant pack.
 * @param device_ptrs A vector of device pointers to be associated with the variant pack.
 * @param uids A vector of unique IDs to be associated with the variant pack.
 * @param workspace_ptr A pointer to the workspace memory to be associated with the variant pack.
 * @return `error_t` A tuple containing the error code and an optional error message.
 *         The error code is `error_code_t::OK` on success, or an appropriate error code on failure.
 */
inline error_t
get_workspace_size(ManagedOpaqueDescriptor& engine_config, int64_t& workspace) {
#if CUDNN_VERSION >= 90200
    CHECK_CUDNN_ERROR(detail::get_attribute(engine_config->get_backend_descriptor(),
                                            CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                            CUDNN_TYPE_INT64,
                                            1,
                                            nullptr,
                                            &workspace));
    return {error_code_t::OK, ""};
#else
    (void)engine_config;
    (void)workspace;
    return {error_code_t::CUDNN_BACKEND_API_FAILED,
            "CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE is only available starting 9.2."};
#endif
}

inline error_t
get_shared_memory_size(ManagedOpaqueDescriptor& engine_config, int32_t& shared_memory_size) {
#if CUDNN_VERSION >= 90200
    CHECK_CUDNN_ERROR(detail::get_attribute(engine_config->get_backend_descriptor(),
                                            CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED,
                                            CUDNN_TYPE_INT32,
                                            1,
                                            nullptr,
                                            &shared_memory_size));
    return {error_code_t::OK, ""};
#else
    (void)engine_config;
    (void)shared_memory_size;
    return {error_code_t::CUDNN_BACKEND_API_FAILED,
            "CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED is only available starting 9.2."};
#endif
}

}  // namespace cudnn_frontend::detail
