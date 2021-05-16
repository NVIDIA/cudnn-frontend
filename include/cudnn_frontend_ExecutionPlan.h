/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_Engine.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// ExecutionPlan_v8 Class
/// This class tells the Configuration of the Engine in terms of the knob
/// choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine
///
/// Use ExecutionPlanBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class ExecutionPlan_v8 : public BackendDescriptor {
   public:
    friend class ExecutionPlanBuilder_v8;

    ExecutionPlan_v8(ExecutionPlan_v8 &&from)
        : BackendDescriptor(from.get_desc(), from.get_status(), from.get_error()),
          engine_config(from.engine_config),
          handle(from.handle),
          planTag(from.planTag) {}
    ~ExecutionPlan_v8() = default;
    /** @defgroup ExecutionPlanQuery
     *  Query individual property of ExecutionPlan_v8 class
     *  @{
     */
    //! Query the workspace requirement for the given plan
    auto
    getWorkspaceSize(void) const -> int64_t {
        std::int64_t workSpaceSize = 0;
        auto status            = cudnnBackendGetAttribute(pointer->get_backend_descriptor(),
                                               CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                               CUDNN_TYPE_INT64,
                                               1,
                                               NULL,
                                               &workSpaceSize);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE Failed");
            return workSpaceSize;
        }
        if (workSpaceSize < 0) {
            set_error_and_throw_exception(
                this, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute Workspace Size Invalid");
            return workSpaceSize;
        }
        return workSpaceSize;
    }

    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR :";
        return ss.str();
    }

    std::string const &
    getTag() const {
        return planTag;
    }

   private:
    void
    computeTag() {
        // Compute a unique tag for execution plan:
        auto status = CUDNN_STATUS_SUCCESS;
        std::stringstream tag{""};
        int64_t elemCount = 0, engineId = 0, numKnobs = 0;

        ManagedOpaqueDescriptor extractedEngine = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        status                                  = extractedEngine->get_status();
        std::array<ManagedOpaqueDescriptor, CUDNN_KNOB_TYPE_COUNTS> extractedKnobs{{nullptr}};
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                this, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed when compute tag");
        }

        for (auto &knob : extractedKnobs) {
            knob   = make_shared_backend_pointer(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
            status = knob->get_status();
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    this, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed when compute tag");
            }
        }

        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
        std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> extractedKnobs_{{nullptr}};
        for (std::uint32_t i = 0; i < extractedKnobs.size(); i++) {
            extractedKnobs_[i] = extractedKnobs[i]->get_backend_descriptor();
        }

        status = cudnnBackendGetAttribute(engine_config->get_backend_descriptor(),
                                          CUDNN_ATTR_ENGINECFG_ENGINE,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &elemCount,
                                          &extractedEngine_);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_ENGINE Failed");
        }
        status = cudnnBackendGetAttribute(
            extractedEngine_, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &elemCount, &engineId);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINE_GLOBAL_INDEX Failed");
        }
        tag << "eng" << engineId;

        status = cudnnBackendGetAttribute(engine_config->get_backend_descriptor(),
                                          CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          CUDNN_KNOB_TYPE_COUNTS,
                                          &numKnobs,
                                          &(extractedKnobs_[0]));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES Failed");
        }
        if (numKnobs > CUDNN_KNOB_TYPE_COUNTS) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "numKnobs exceed the CUDNN_KNOB_TYPE_COUNTS");
        }
        for (int64_t idx = 0; idx < numKnobs; ++idx) {
            const cudnnBackendDescriptor_t &knob = extractedKnobs_[idx];
            cudnnBackendKnobType_t type          = CUDNN_KNOB_TYPE_COUNTS;
            int64_t choice                       = -2;
            status                               = cudnnBackendGetAttribute(
                knob, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, nullptr, &type);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this,
                                              status,
                                              "computeTag CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE Failed");
            }
            status = cudnnBackendGetAttribute(
                knob, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, nullptr, &choice);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this,
                                              status,
                                              "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE Failed");
            }
            tag << "_k" << type << "=" << choice;
        }
        planTag += tag.str();
    }

    ExecutionPlan_v8()                         = default;
    ExecutionPlan_v8(ExecutionPlan_v8 const &) = delete;
    ExecutionPlan_v8 &
    operator=(ExecutionPlan_v8 const &) = delete;

    ManagedOpaqueDescriptor engine_config = nullptr;
    cudnnHandle_t handle                  = nullptr;
    std::string planTag;
};

///
/// ExecutionPlanBuilder_v8 Class
/// Helper class used to build ExecutionPlan_v8 class
class ExecutionPlanBuilder_v8 {
   public:
    /** @defgroup ExecutionPlanBuilder_v8
     *  Set individual property of ExecutionPlan_v8 class
     *  @{
     */
    //! Set engine for the ExecutionPlan_v8
    auto
    setHandle(cudnnHandle_t handle_) -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.handle = handle_;
        return *this;
    }
    //! Set engine Config for the Plan
    auto
    setEngineConfig(EngineConfig_v8 const &engine_config_) -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.engine_config = engine_config_.get_desc();
        m_execution_plan.planTag       = engine_config_.getTag();
        return *this;
    }

    //! Set engine Config for the Plan
    auto
    setEngineConfig(ManagedOpaqueDescriptor &desc, std::string const &opGraphTag_ = "") -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.engine_config = desc;
        m_execution_plan.planTag       = opGraphTag_;
        return *this;
    }
    /** @} */

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    ExecutionPlan_v8 &&
    build() {
        if (m_execution_plan.handle == nullptr) {
            set_error_and_throw_exception(
                &m_execution_plan,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set the CUDNN_ATTR_EXECUTION_PLAN_HANDLE");
            return std::move(m_execution_plan);
        };
        if (m_execution_plan.engine_config == nullptr) {
            set_error_and_throw_exception(
                &m_execution_plan,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set the CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG");
            return std::move(m_execution_plan);
        };

        // Create a descriptor. Memory allocation happens here.
        auto status = m_execution_plan.initialize_managed_backend_pointer(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_execution_plan);
        }

        status = cudnnBackendSetAttribute(m_execution_plan.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &(m_execution_plan.engine_config->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG Failed");
            return std::move(m_execution_plan);
        }
        status = cudnnBackendSetAttribute(m_execution_plan.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                          CUDNN_TYPE_HANDLE,
                                          1,
                                          &m_execution_plan.handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute CUDNN_ATTR_EXECUTION_PLAN_HANDLE Failed");
            return std::move(m_execution_plan);
        }
        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_execution_plan.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed");
            return std::move(m_execution_plan);
        }

        m_execution_plan.computeTag();

        return std::move(m_execution_plan);
    }

    explicit ExecutionPlanBuilder_v8()                       = default;
    ~ExecutionPlanBuilder_v8()                               = default;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8 &&)      = delete;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8 const &) = delete;
    ExecutionPlanBuilder_v8 &
    operator=(ExecutionPlanBuilder_v8 const &) = delete;

   private:
    ExecutionPlan_v8 m_execution_plan;
};
}
