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

#include <vector>
#include <mutex>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_EngineConfig.h"
#if (CUDNN_VERSION < 8400)
#include "cudnn_frontend_EngineFallbackList.h"
#endif
#include "cudnn_frontend_utils.h"
#include "cudnn_frontend_Filters.h"

namespace cudnn_frontend {
///
/// Engine Heuristic Class
/// This class helps determine the engine from the operation graph
/// based on the heuristics
/// Properties:
///    - heuristic mode
///    - operation graph
///
/// Use EngineHeuristicsBuilder_v8 to build this class.
/// Describe returns a string describing the EngineHeuristics_v8 class
///
class EngineHeuristics_v8 : public BackendDescriptor {
   public:
    friend class EngineHeuristicsBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR :";
        return ss.str();
    }

    EngineHeuristics_v8(EngineHeuristics_v8 &&from) = default;
    EngineHeuristics_v8 &
    operator= (EngineHeuristics_v8 &&from) = default;

    ~EngineHeuristics_v8() = default;

    /** @defgroup EngineHeuristicsQuery
     *  Query individual property of EngineHeuristics_v8 class
     *  @{
     */
    //! Query the total count of the engines for the Operation Set
    auto
    getEngineConfig(int64_t count = 1) -> std::vector<ManagedOpaqueDescriptor> & {
        cudnnStatus_t status;
        for (auto i = 0u; i < count; ++i) {
            ManagedOpaqueDescriptor engConfig = nullptr;
            engConfig                         = make_shared_backend_pointer(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
            if (engConfig->is_good() == false) {
                set_error_and_throw_exception(
                    this,
                    engConfig->get_status(),
                    "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: CUDNN_BACKEND_ENGINECFG_DESCRIPTOR cudnnCreate Failed");
                return m_heuristic_results;
            };
            m_heuristic_results.emplace_back(engConfig);
        }
        std::vector<cudnnBackendDescriptor_t> heuristic_results_;
        for (std::uint32_t i = 0; i < m_heuristic_results.size(); i++) {
            heuristic_results_.emplace_back(m_heuristic_results[i]->get_backend_descriptor());
        }
        int64_t result = -1;
        status         = cudnnBackendGetAttribute(pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          count,
                                          &result,
                                          heuristic_results_.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                this, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: GetAttribute CUDNN_ATTR_ENGINEHEUR_RESULTS Failed");
        };
        return m_heuristic_results;
    }

    //! Query the total count of the engine config for the Operation Set
    auto
    getEngineConfigCount(void) const -> int64_t {
        cudnnStatus_t status;
        int64_t count = -1;
        status        = cudnnBackendGetAttribute(pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          0,
                                          &count,
                                          nullptr);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                this,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: GetAttribute CUDNN_ATTR_ENGINEHEUR_RESULTS Count Failed");
        };
        return count;
    }
    /** @} */

   private:
    EngineHeuristics_v8()                            = default;
    EngineHeuristics_v8(EngineHeuristics_v8 const &) = delete;
    EngineHeuristics_v8 &
    operator=(EngineHeuristics_v8 const &) = delete;

    cudnnBackendHeurMode_t mode     = CUDNN_HEUR_MODE_INSTANT;
    ManagedOpaqueDescriptor opGraph = nullptr;
    std::vector<ManagedOpaqueDescriptor> m_heuristic_results;  //! storage of heuristic results
    std::string opGraphTag;

    static std::mutex & get_heur_b_mutex() {
        static std::mutex heur_b_mutex;
        return heur_b_mutex;
    }
};

///
/// EngineHeuristicsBuilder_v8 Class
/// Helper class used to build EngineHeuristics_v8 class
class EngineHeuristicsBuilder_v8 {
   public:
    /** @defgroup EngineHeuristicsBuilder_v8
     *  Set individual property of EngineHeuristics_v8 class
     *  @{
     */
    //! Set operationGraph for the engine (opGraph is not destroyed)
    auto
    setOperationGraph(OperationGraph_v8 &opGraph_) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.opGraph    = opGraph_.get_desc();
        m_heuristics.opGraphTag = opGraph_.getTag();
        return *this;
    }
    auto
    setOperationGraph(ManagedOpaqueDescriptor opGraph, std::string tag) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.opGraph    = opGraph;
        m_heuristics.opGraphTag = tag;
        return *this;
    }
    //! Set cudnnHandle for the operations
    auto
    setHeurMode(cudnnBackendHeurMode_t mode_) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.mode = mode_;
        return *this;
    }
    /** @} */

    //! constructs the EngineHeuristics_v8 by calling the cudnn API
    //! Throws the appropriate error message
    EngineHeuristics_v8 &&
    build() {
        if (m_heuristics.opGraph == nullptr) {
            set_error_and_throw_exception(&m_heuristics,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH field for heuristic");
            return std::move(m_heuristics);
        };

        // Create a descriptor. Memory allocation happens here.
        auto status = m_heuristics.initialize_managed_backend_pointer(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_heuristics);
        };

        status = cudnnBackendSetAttribute(m_heuristics.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &(m_heuristics.opGraph->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute  CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH Failed");
            return std::move(m_heuristics);
        };
        status = cudnnBackendSetAttribute(m_heuristics.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_ENGINEHEUR_MODE,
                                          CUDNN_TYPE_HEUR_MODE,
                                          1,
                                          &m_heuristics.mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINEHEUR_MODE Failed");
            return std::move(m_heuristics);
        };

#if (CUDNN_VERSION >= 8401)
        if (m_heuristics.mode == CUDNN_HEUR_MODE_B) {
            EngineHeuristics_v8::get_heur_b_mutex().lock();
        }
#endif
        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_heuristics.pointer->get_backend_descriptor());
#if (CUDNN_VERSION >= 8401)
        if (m_heuristics.mode == CUDNN_HEUR_MODE_B) {
            EngineHeuristics_v8::get_heur_b_mutex().unlock();
        }
#endif
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: cudnn Finalize failed");
            return std::move(m_heuristics);
        };

        getLogger() << "[cudnn_frontend] " << m_heuristics << std::endl;
        return std::move(m_heuristics);
    }

    explicit EngineHeuristicsBuilder_v8()                          = default;
    ~EngineHeuristicsBuilder_v8()                                  = default;
    EngineHeuristicsBuilder_v8(EngineHeuristicsBuilder_v8 &&)      = delete;
    EngineHeuristicsBuilder_v8(EngineHeuristicsBuilder_v8 const &) = delete;
    EngineHeuristicsBuilder_v8 &
    operator=(EngineHeuristicsBuilder_v8 const &) = delete;

   private:
    EngineHeuristics_v8 m_heuristics;
};

template<std::size_t SIZE>
EngineConfigList
get_heuristics_list(std::array<cudnnBackendHeurMode_t, SIZE> modes,
    OperationGraph_v8 &opGraph,
    std::function<bool(cudnnBackendDescriptor_t)> filter_fn) {
    CUDNN_FRONTEND_UNUSED(modes);
    EngineConfigList filtered_configs;

    for (auto mode : modes) {
        if (mode == CUDNN_HEUR_MODES_COUNT) {continue;}
        auto heuristics = EngineHeuristicsBuilder_v8()
            .setOperationGraph(opGraph)
            .setHeurMode(mode)
            .build();
        getLogger() << "Heuristic Mode " << mode << " has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
        cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
    }
    return filtered_configs;
}

template<std::size_t SIZE>
std::vector<cudnnStatus_t>
get_heuristics_list(std::array<std::string, SIZE> modes,
    OperationGraph_v8 &opGraph,
    std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
    EngineConfigList &filtered_configs,
    bool evaluate_all = false) {
    
    std::vector<cudnnStatus_t> statuses;

    for (auto &mode : modes) {
        if (mode.find("heuristics_instant") != std::string::npos) {
heuristics_instant:
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
#endif
                auto heuristics = EngineHeuristicsBuilder_v8()
                    .setOperationGraph(opGraph)
                    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                    .build();
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                auto num_config = heuristics.getEngineConfigCount();
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                getLogger() << "Heuristic Mode " << mode << " has " << num_config << " configurations " << std::endl;
                auto& engine_config = heuristics.getEngineConfig(num_config);
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
                statuses.push_back(heuristics.get_status());
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException &e) {
                statuses.push_back(e.getCudnnStatus());
                if (evaluate_all) 
                    continue;
                else 
                    break;
            }
#endif
        } else if (mode.find("heuristics_fallback") != std::string::npos) {
#if (CUDNN_VERSION < 8400)
            if (opGraph.getOpCount() > 1) {
                std::vector<ManagedOpaqueDescriptor> engine_configs;
                std::array<int32_t, 2> fallback_engine_list = {0, 1};
                for (std::uint32_t i = 0; i < fallback_engine_list.size(); i++) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
                    try {
#endif
                        auto engine = cudnn_frontend::EngineBuilder_v8()
                                          .setGlobalEngineIdx(fallback_engine_list[i])
                                          .setOperationGraph(opGraph)
                                          .build();
                        if (engine.get_status() != CUDNN_STATUS_SUCCESS) {
                            continue;
                        }
                        auto engine_config = cudnn_frontend::EngineConfigBuilder_v8().setEngine(engine).build();
                        if (engine_config.get_status() != CUDNN_STATUS_SUCCESS) {
                            continue;
                        }
                        engine_configs.emplace_back(engine_config.get_desc());
#ifndef NV_CUDNN_DISABLE_EXCEPTION
                    } catch (cudnn_frontend::cudnnException &e) {
                        continue;
                    }
#endif
                }
                cudnn_frontend::filter(engine_configs, filtered_configs, filter_fn);
                cudnnStatus_t status_ = CUDNN_STATUS_NOT_SUPPORTED;
                if (filtered_configs.size() > 0) {
                    status_ = CUDNN_STATUS_SUCCESS;
                }
                statuses.push_back(status_);
            }
#endif
#if (CUDNN_VERSION >= 8300)
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
#endif
                auto heuristics = EngineHeuristicsBuilder_v8()
                    .setOperationGraph(opGraph)
                    .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                    .build();
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                auto num_config = heuristics.getEngineConfigCount();
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                getLogger() << "Heuristic Mode " << mode << " has " << num_config << " configurations " << std::endl;
                auto& engine_config = heuristics.getEngineConfig(num_config);
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
                statuses.push_back(heuristics.get_status());
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException &e) {
                statuses.push_back(e.getCudnnStatus());
                if (evaluate_all) 
                    continue;
                else 
                    break;
            }
#endif
#else
            cudnnBackendDescriptorType_t op_type = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
            CUDNN_FRONTEND_UNUSED(op_type);
            std::string tag_ = opGraph.getTag();
            if (tag_.find("ConvFwd") != std::string::npos) {
                op_type = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
            } else if (tag_.find("ConvBwdFilter") != std::string::npos) {
                op_type = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
            }
            auto fallback = cudnn_frontend::EngineFallbackListBuilder_v8()
                                .setOperationGraph(opGraph)
                                .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                                .build();
            if (fallback.get_status() != CUDNN_STATUS_SUCCESS) {
                statuses.push_back(fallback.get_status());
                if (evaluate_all) 
                    continue;
                else 
                    break;
            }            
            auto& fallback_list = fallback.getFallbackList();
            if (fallback.get_status() != CUDNN_STATUS_SUCCESS) {
                statuses.push_back(fallback.get_status());
                if (evaluate_all) 
                    continue;
                else 
                    break;
            }     
            getLogger() << "Fallback List has " << fallback_list.size() << " configurations " << std::endl;
            cudnn_frontend::filter(fallback_list, filtered_configs, filter_fn);
            statuses.push_back(fallback.get_status());
#endif
        } else if (mode.find("heuristics_mode_b") != std::string::npos) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
#endif
                auto heuristics = EngineHeuristicsBuilder_v8()
                    .setOperationGraph(opGraph)
                    .setHeurMode(CUDNN_HEUR_MODE_B)
                    .build();
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    goto heuristics_instant;
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                auto num_config = heuristics.getEngineConfigCount();
                getLogger() << "Heuristic Mode " << mode << " has " << num_config << " configurations " << std::endl;
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                auto& engine_config = heuristics.getEngineConfig(num_config);
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
                statuses.push_back(heuristics.get_status());
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException &e) {
                goto heuristics_instant;
                statuses.push_back(e.getCudnnStatus());
                if (evaluate_all) 
                    continue;
                else 
                    break;
            }
#endif
#if (CUDNN_VERSION >= 8300)
        } else if (mode.find("heuristics_mode_a") != std::string::npos) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
#endif
                auto heuristics = EngineHeuristicsBuilder_v8()
                    .setOperationGraph(opGraph)
                    .setHeurMode(CUDNN_HEUR_MODE_A)
                    .build();
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                auto num_config = heuristics.getEngineConfigCount();
                getLogger() << "Heuristic Mode " << mode << " has " << num_config << " configurations " << std::endl;
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                auto& engine_config = heuristics.getEngineConfig(num_config);
                if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) {
                    statuses.push_back(heuristics.get_status());
                    if (evaluate_all) 
                        continue;
                    else 
                        break;
                }
                cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
                statuses.push_back(heuristics.get_status());
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException &e) {
                statuses.push_back(e.getCudnnStatus());
                if (evaluate_all) 
                    continue;
                else 
                    break;
            }
#endif
#endif
        }
    }
    return statuses;
}

}
