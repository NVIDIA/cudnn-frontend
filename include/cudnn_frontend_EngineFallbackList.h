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

#include <numeric>
#include "cudnn_frontend_Heuristics.h"

namespace cudnn_frontend {

[[maybe_unused]] auto static get_fallback_engine_list(DescriptorType_t mode, const std::string &opGraphTag)
    -> std::vector<int> {
    auto major_version = cudnnGetVersion() / 1000;

    auto minor_version = (cudnnGetVersion() / 100) % 10;
    if (major_version >= 8) {
        if (minor_version <= 2) {
            /// Here we are using the term "bias" in the operationGraph as a proxy for
            /// the conv*bias* operation graph. We are not strictly checking the order of
            /// the operations in the graph. We propose this as a temporary workaround until
            /// the backend API supports querying the fallback list directly from cudnn
            if (mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
                if (opGraphTag.find("bias") == std::string::npos) {
                    std::vector<int> engine_list(50);
                    std::iota(engine_list.begin(), engine_list.end(), 0);
                    return engine_list;
                } else {
                    return {11, 0};
                }
            } else if (mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
                std::vector<int> engine_list(61);
                std::iota(engine_list.begin(), engine_list.end(), 0);
                return engine_list;
            } else if (mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
                return {0, 1, 20};
            } else {
                return {};
            }
        } else {
            if (mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
                if (opGraphTag.find("bias") == std::string::npos) {
                    return {0, 1, 28};
                } else {
                    return {};
                }
            } else if (mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
                return {0, 1, 25};
            } else if (mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
                return {0, 1, 20};
            } else {
                return {};
            }
        }
    } else {
        return {};
    }
}

class EngineFallbackList_v8 : public BackendDescriptor {
   public:
    friend class EngineFallbackListBuilder_v8;

    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_FALLBACK ENGINES :";
        return ss.str();
    }

    auto
    getFallbackList() -> std::vector<ManagedOpaqueDescriptor> & {
        return m_engine_configs;
    }

    ~EngineFallbackList_v8() = default;

    EngineFallbackList_v8(EngineFallbackList_v8 &&from)
        : BackendDescriptor(from.get_desc(), from.get_status(), from.get_error()),
          opGraph(from.opGraph),
          mode(from.mode),
          num_ops(from.num_ops),
          opGraphTag(from.opGraphTag) {
        m_engine_configs.swap(from.m_engine_configs);
    }

   private:
    EngineFallbackList_v8()                              = default;
    EngineFallbackList_v8(EngineFallbackList_v8 const &) = delete;
    EngineFallbackList_v8 &
    operator=(EngineFallbackList_v8 const &) = delete;

    ManagedOpaqueDescriptor opGraph = nullptr;
    DescriptorType_t mode;
    uint64_t num_ops;
    std::vector<ManagedOpaqueDescriptor> m_engine_configs;
    std::string opGraphTag;
};

///
/// EngineFallBackListBuilder Class
/// Helper class used to build EngineFallBackList class
class EngineFallbackListBuilder_v8 {
   public:
    /** @defgroup EngineFallbackListBuilder_v8
     *  Set individual property of EngineFallbackList_v8 class
     *  @{
     */
    //! Set operationGraph for the engine (opGraph is not destroyed)
    auto
    setOperationGraph(OperationGraph_v8 &opGraph_) -> EngineFallbackListBuilder_v8 & {
        m_fallback_list.opGraph    = opGraph_.get_desc();
        m_fallback_list.opGraphTag = opGraph_.getTag();
        m_fallback_list.num_ops    = opGraph_.getOpCount();
        return *this;
    }

    auto
    setOperation(DescriptorType_t mode) -> EngineFallbackListBuilder_v8 & {
        m_fallback_list.mode = mode;
        return *this;
    }

    auto
    setOperation(cudnnBackendDescriptorType_t mode) -> EngineFallbackListBuilder_v8 & {
        m_fallback_list.mode = detail::convert_from_cudnn_type(mode);
        return *this;
    }
    /** @} */

    //! constructs the EngineFallbackList_v8 by calling the cudnn API
    //! Throws the appropriate error message
    EngineFallbackList_v8 &&
    build() {
        if (m_fallback_list.opGraph == nullptr) {
            set_error_and_throw_exception(&m_fallback_list,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH field for heuristic");
            return std::move(m_fallback_list);
        };
#if (CUDNN_VERSION >= 8400)
        auto fallback_heuristics = EngineHeuristicsBuilder_v8()
                                       .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                       .setOperationGraph(m_fallback_list.opGraph, m_fallback_list.opGraphTag)
                                       .build();
        auto count                       = fallback_heuristics.getEngineConfigCount();
        m_fallback_list.m_engine_configs = fallback_heuristics.getEngineConfig(count);
#else
        auto fallback_engine_list = get_fallback_engine_list(m_fallback_list.mode, m_fallback_list.opGraphTag);
        for (std::uint32_t i = 0; i < fallback_engine_list.size(); i++) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
#endif
                auto engine = cudnn_frontend::EngineBuilder_v8()
                                  .setGlobalEngineIdx(fallback_engine_list[i])
                                  .setOperationGraph(m_fallback_list.opGraph)
                                  .build();
                auto engine_config = cudnn_frontend::EngineConfigBuilder_v8().setEngine(engine).build();
                m_fallback_list.m_engine_configs.emplace_back(engine_config.get_desc());
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException &e) {
                CUDNN_FRONTEND_UNUSED(e);
                continue;
            }
#endif
        }
#endif
        getLogger() << "[cudnn_frontend] " << m_fallback_list << std::endl;
        return std::move(m_fallback_list);
    }

    explicit EngineFallbackListBuilder_v8()                            = default;
    ~EngineFallbackListBuilder_v8()                                    = default;
    EngineFallbackListBuilder_v8(EngineFallbackListBuilder_v8 &&)      = delete;
    EngineFallbackListBuilder_v8(EngineFallbackListBuilder_v8 const &) = delete;
    EngineFallbackListBuilder_v8 &
    operator=(EngineFallbackListBuilder_v8 const &) = delete;

   private:
    EngineFallbackList_v8 m_fallback_list;
};
}  // namespace cudnn_frontend
