/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>

namespace cudnn_frontend {

///
/// OpaqueBackendPointer class
/// Holds the raws pointer to backend_descriptor
/// Usage is to wrap this into a smart pointer as
/// it helps to create and destroy the backendpointer

class OpaqueBackendPointer {
    cudnnBackendDescriptor_t m_desc = nullptr;               //!< Raw void pointer
    cudnnStatus_t status            = CUDNN_STATUS_SUCCESS;  //!< status of creation of the Descriptor

   public:
    OpaqueBackendPointer(const OpaqueBackendPointer&) = delete;  //!< Delete the copy constructor to prevent bad copies
    OpaqueBackendPointer&
    operator=(const OpaqueBackendPointer&)       = delete;
    OpaqueBackendPointer(OpaqueBackendPointer&&) = default;

    /**
     * OpaqueBackendPointer constructor.
     * Calls the cudnnBackendCreateDescriptor. Allocates memory according to the type.
     */
    OpaqueBackendPointer(cudnnBackendDescriptorType_t type) { status = detail::create_descriptor(type, &m_desc); }
    /**
     * OpaqueBackendPointer destructor.
     * Calls the cudnnBackendDestroyDescriptor. Frees memory allocated in the constructor.
     */
    ~OpaqueBackendPointer() { detail::destroy_descriptor(m_desc); };
    /**
     * Accessor.
     * Returns the const reference to raw underlying descriptor.
     * Treat it like the data() function of a smart pointer. Can be freed behind the back.
     */
    cudnnBackendDescriptor_t const&
    get_backend_descriptor() const {
        return m_desc;
    }
    /**
     * Accessor.
     * Queries the status of the descriptor after calling the cudnnCreate.
     */
    cudnnStatus_t
    get_status() const {
        return status;
    }
    /**
     * Accessor.
     * Queries the status of the descriptor returns true if all good.
     */
    bool
    is_good() const {
        return status == CUDNN_STATUS_SUCCESS;
    }
};

/*! \var A shared_ptr wrapper on top of the OpaqueBackendPointer */
using ManagedOpaqueDescriptor = std::shared_ptr<OpaqueBackendPointer>;

/*! \fn A wrapper on top of the std::make_shared for the OpaqueBackendPointer */
static ManagedOpaqueDescriptor
make_shared_backend_pointer(cudnnBackendDescriptorType_t type) {
    return std::make_shared<OpaqueBackendPointer>(type);
}

///
/// BackendDescriptor class
/// Holds a Managed pointer to OpaqueBackendPointer class
/// Contains the status and error message if set after any operation.
/// If exception is disabled the user must query the status after
/// build operation in order to check if the cudnn construct was built
/// correctly.
class BackendDescriptor {
   public:
    //! Return a string describing the backend Descriptor
    virtual std::string
    describe() const = 0;

    //! Get a copy of the raw descriptor pointer. Ownership is reatined and
    //! gets deleted when out of scope
    cudnnBackendDescriptor_t
    get_raw_desc() const {
        return pointer->get_backend_descriptor();
    }

    //! Current status of the descriptor
    cudnnStatus_t
    get_status() const {
        return status;
    }

    //! Set status of the descriptor
    void
    set_status(cudnnStatus_t const status_) const {
        status = status_;
    }

    //! Set Diagonistic error message.
    void
    set_error(const char* message) const {
        err_msg = message;
    }

    //! Diagonistic error message if any
    const char*
    get_error() const {
        return err_msg.c_str();
    }

    //! Returns a copy of underlying managed descriptor
    ManagedOpaqueDescriptor
    get_desc() const {
        return pointer;
    }

    //! Initializes the underlying managed descriptor
    cudnnStatus_t
    initialize_managed_backend_pointer(cudnnBackendDescriptorType_t type) {
        pointer = make_shared_backend_pointer(type);
        return pointer->get_status();
    }

   protected:
    /**
     * BackendDescriptor constructor.
     * Initializes the member variables as passed.
     */
    BackendDescriptor(ManagedOpaqueDescriptor pointer_, cudnnStatus_t status_, std::string err_msg_)
        : pointer(pointer_), status(status_), err_msg(err_msg_) {}
    BackendDescriptor() = default;

    virtual ~BackendDescriptor() {};

    ManagedOpaqueDescriptor pointer;  //! Shared pointer of the OpaqueBackendPointer

    mutable cudnnStatus_t status = CUDNN_STATUS_SUCCESS;  //!< Error code if any being set
    mutable std::string err_msg;                          //!< Error message if any being set
};

}  // namespace cudnn_frontend
