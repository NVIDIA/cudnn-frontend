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
create_variant_pack(backend_descriptor& variant_pack,
                    std::vector<void*>& device_ptrs,
                    std::vector<int64_t> const& uids,
                    void* workspace_ptr) {
    CHECK_CUDNN_ERROR(detail::set_attribute(
        variant_pack.get_ptr(), CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace_ptr));

    CHECK_CUDNN_ERROR(detail::set_attribute(variant_pack.get_ptr(),
                                            CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                            CUDNN_TYPE_VOID_PTR,
                                            device_ptrs.size(),
                                            device_ptrs.data()));

    CHECK_CUDNN_ERROR(detail::set_attribute(
        variant_pack.get_ptr(), CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data()));

    CHECK_CUDNN_ERROR(detail::finalize(variant_pack.get_ptr()));

    return {error_code_t::OK, ""};
}

}  // namespace cudnn_frontend::detail
