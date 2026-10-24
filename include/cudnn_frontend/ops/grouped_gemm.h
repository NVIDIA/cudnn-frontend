/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// The CuTeDSL grouped (MoE) GEMMs of cudnn.grouped_gemm_<kind>_wrapper_sm100,
// executed from C++. The Python wrappers run their repeated calls through
// this op, and C++ callers in the same process call it directly.
//
// One op is one wrapper call: its kind (glu, dglu, quant, wgrad, gemm), its
// options (the wrapper's other keyword arguments, as JSON) and its operands,
// named as the wrapper names its tensor arguments and the outputs it
// allocates (a_tensor, sfa_tensor, ..., c_tensor, d_tensor, ...). The kernel
// is the one the wrapper compiles for them; Python registers it, with how its
// API passes each operand to it, the second time the wrapper is called with
// them (KernelLibrary::registry()). create() binds the operands the same way
// and never compiles; execute() launches with the operands' addresses.

#include "kernel_library.h"

#include <array>
#include <string>
#include <vector>

namespace cudnn_frontend {
namespace ops {

struct GroupedGemmOperand {
    std::string name;
    TensorView view;
};

class GroupedGemm {
   public:
    // The configuration create() looks the kernel up by.
    static error_t
    config(std::string const &kind, std::vector<GroupedGemmOperand> const &operands, json const &options, json &out) {
        json described = json::object();
        for (auto const &o : operands) {
            auto const &v = o.view;
            RETURN_CUDNN_FRONTEND_ERROR_IF(v.rank < 0 || v.rank > 8,
                                           error_code_t::INVALID_VALUE,
                                           "GroupedGemm: operand " + o.name + " has rank above 8");
            json dim = json::array(), stride = json::array();
            for (int i = 0; i < v.rank; i++) {
                dim.push_back(v.dim[i]);
                stride.push_back(v.stride[i]);
            }
            described[o.name] = {{"dtype", v.dtype}, {"dim", dim}, {"stride", stride}};
        }
        int device = 0, cc = 0, num_sm = 0;
        CHECK_CUDNN_FRONTEND_ERROR(detail::current_device(device, cc, num_sm));
        out = {{"op", "grouped_gemm"},
               {"kind", kind},
               {"options", options},
               {"operands", described},
               {"target", {{"cc", cc}, {"num_sm", num_sm}}}};
        return {error_code_t::OK, ""};
    }

    // Takes the kernel from lib and binds the operands; never compiles. For the current device.
    static error_t
    create(std::string const &kind,
           std::vector<GroupedGemmOperand> const &operands,
           json const &options,
           KernelLibrary const &lib,
           GroupedGemm &out) {
        GroupedGemm p;
        json key;
        CHECK_CUDNN_FRONTEND_ERROR(config(kind, operands, options, key));
        std::shared_ptr<detail::Kernel const> kernel;
        CHECK_CUDNN_FRONTEND_ERROR(lib.find(key, kernel));
        int cc = 0, num_sm = 0;
        CHECK_CUDNN_FRONTEND_ERROR(detail::current_device(p.device_, cc, num_sm));
        for (auto const &o : operands) {
            p.views_.push_back(o.view);
        }
        CHECK_CUDNN_FRONTEND_ERROR(p.bind(operands, kernel->meta));
        p.kernel_ = std::move(kernel);
        out       = std::move(p);
        return {error_code_t::OK, ""};
    }

    // Bytes, 128-byte aligned; not shared by calls in flight.
    int64_t
    workspace_size() const {
        return workspace_size_;
    }

    // The operands of create(), by name, with the same dtypes, shapes and strides; only addresses may differ.
    error_t
    execute(std::vector<GroupedGemmOperand> const &operands, void *workspace, cudaStream_t stream) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(operands.size() != names_.size(),
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "GroupedGemm::execute: operands differ from create()");
        std::vector<void *> data(operands.size());
        for (size_t i = 0; i < operands.size(); i++) {
            auto const &a = operands[i].view, &b = views_[i];
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                operands[i].name != names_[i] || a.dtype != b.dtype || a.rank != b.rank || a.dim != b.dim ||
                    a.stride != b.stride,
                error_code_t::INVALID_VARIANT_PACK,
                "GroupedGemm::execute: operand " + operands[i].name + " differs from create()");
            data[i] = a.data;
        }
        return execute(data.data(), data.size(), workspace, stream);
    }

    // The operands' addresses, in create()'s order. A misaligned operand returns INVALID_VARIANT_PACK and launches
    // nothing.
    error_t
    execute(void *const *data, size_t count, void *workspace, cudaStream_t stream) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(count != views_.size(),
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "GroupedGemm::execute: operands differ from create()");
        for (size_t i = 0; i < count; i++) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(reinterpret_cast<uintptr_t>(data[i]) % align_[i] != 0,
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "GroupedGemm::execute: operand " + names_[i] + " is less aligned than " +
                                               std::to_string(align_[i]) + " bytes");
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(workspace_size_ > 0 && reinterpret_cast<uintptr_t>(workspace) % 128 != 0,
                                       error_code_t::INVALID_VALUE,
                                       "GroupedGemm::execute: the workspace must be 128-byte aligned");
        detail::ffi::Any args[MAX_ARGS];
        detail::ffi::DLTensor tensors[MAX_ARGS];
        std::copy(proto_.begin(), proto_.end(), args);
        int t = 0;
        for (auto const &b : binds_) {
            uintptr_t const base = b.operand >= 0           ? reinterpret_cast<uintptr_t>(data[b.operand])
                                   : b.operand == WORKSPACE ? reinterpret_cast<uintptr_t>(workspace)
                                                            : 0;
            void *p              = reinterpret_cast<void *>(base + b.offset);
            switch (b.as) {
                case As::TENSOR: {
                    auto &d     = tensors[t++];
                    d           = {};
                    d.data      = p;
                    d.device    = {2, device_};
                    d.ndim      = b.rank;
                    d.dtype     = b.dtype;
                    d.shape     = const_cast<int64_t *>(b.dim.data());
                    d.strides   = const_cast<int64_t *>(b.stride.data());
                    args[b.arg] = detail::any_tensor(&d);
                    break;
                }
                case As::INT:
                    args[b.arg] = detail::any_int(static_cast<int64_t>(reinterpret_cast<intptr_t>(p)));
                    break;
                case As::POINTER:
                    args[b.arg] = detail::any_pointer(p);
                    break;
                case As::STREAM:
                    args[b.arg] = detail::any_pointer(stream);
                    break;
            }
        }
        return kernel_->call(args, static_cast<int>(proto_.size()));
    }

   private:
    static constexpr size_t MAX_ARGS = 64;
    static constexpr int WORKSPACE   = -1;
    static constexpr int ABSOLUTE    = -2;

    enum class As { TENSOR, INT, POINTER, STREAM };

    // An argument that changes per call: an operand, or a view into one or into the workspace.
    struct Bind {
        As as          = As::TENSOR;
        int arg        = 0;
        int operand    = WORKSPACE;  // an operand's index, WORKSPACE or ABSOLUTE
        int64_t offset = 0;          // bytes from the buffer's start; the address itself when ABSOLUTE
        int rank       = 0;
        detail::ffi::DLDataType dtype{};
        std::array<int64_t, 8> dim{}, stride{};
    };

    // The kernel's arguments, as the registry's metadata binds them:
    //   null, {"int": v}, {"float": v}, {"bool": v}, {"pointer": v}  constants
    //   {"stream": true}                                             the stream
    //   {"operand": name | "workspace": true, "offset": bytes, "as": "tensor" | "int" | "pointer",
    //    "dim": [...], "stride": [...], "dtype": [code, bits, lanes]}  an address in a buffer; a tensor
    //                                                                  without "dim" is the operand itself
    //   {"address": a, "as": "tensor", "dim": ..., ...}              a tensor the API owns, at a fixed address
    // plus "align" (bytes each operand's address is a multiple of) and "workspace_bytes".
    error_t
    bind(std::vector<GroupedGemmOperand> const &operands, json const &meta) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!meta.contains("args") || !meta.at("args").is_array(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "GroupedGemm: the registered kernel has no argument binding");
        auto const &recorded = meta.at("args");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            recorded.size() > MAX_ARGS,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "GroupedGemm: the kernel takes more than " + std::to_string(MAX_ARGS) + " arguments");
        workspace_size_  = meta.value("workspace_bytes", int64_t{0});
        auto const align = meta.value("align", json::object());
        auto index       = [&](std::string const &name) {
            for (size_t i = 0; i < operands.size(); i++) {
                if (operands[i].name == name) return static_cast<int>(i);
            }
            return -1;
        };
        for (auto const &o : operands) {
            names_.push_back(o.name);
            align_.push_back(std::max<int64_t>(1, align.value(o.name, int64_t{1})));
        }
        proto_.assign(recorded.size(), detail::any_none());
        for (size_t i = 0; i < recorded.size(); i++) {
            auto const &a = recorded[i];
            if (a.is_null()) continue;
            if (a.contains("int")) {
                proto_[i] = detail::any_int(a.at("int").get<int64_t>());
                continue;
            }
            if (a.contains("float")) {
                proto_[i] = detail::any_float(a.at("float").get<double>());
                continue;
            }
            if (a.contains("bool")) {
                proto_[i] = detail::any_bool(a.at("bool").get<bool>());
                continue;
            }
            if (a.contains("pointer")) {
                proto_[i] = detail::any_pointer(reinterpret_cast<void *>(a.at("pointer").get<int64_t>()));
                continue;
            }
            Bind b;
            b.arg = static_cast<int>(i);
            if (a.contains("stream")) {
                b.as = As::STREAM;
                binds_.push_back(b);
                continue;
            }
            if (a.contains("operand")) {
                auto const name = a.at("operand").get<std::string>();
                b.operand       = index(name);
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    b.operand < 0, error_code_t::INVALID_VALUE, "GroupedGemm: the kernel needs operand " + name);
            } else if (a.contains("address")) {
                b.operand = ABSOLUTE;
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(!a.contains("workspace"),
                                               error_code_t::GRAPH_NOT_SUPPORTED,
                                               "GroupedGemm: unknown argument binding " + a.dump());
            }
            b.offset        = b.operand == ABSOLUTE ? a.at("address").get<int64_t>() : a.value("offset", int64_t{0});
            auto const kind = a.value("as", std::string("tensor"));
            b.as            = kind == "int" ? As::INT : kind == "pointer" ? As::POINTER : As::TENSOR;
            if (b.as == As::TENSOR && !a.contains("dim")) {
                RETURN_CUDNN_FRONTEND_ERROR_IF(b.operand < 0,
                                               error_code_t::GRAPH_NOT_SUPPORTED,
                                               "GroupedGemm: a tensor that is no operand needs its dims " + a.dump());
                auto const &v = operands[b.operand].view;
                RETURN_CUDNN_FRONTEND_ERROR_IF(!detail::dl_dtype(v.dtype, b.dtype),
                                               error_code_t::GRAPH_NOT_SUPPORTED,
                                               "GroupedGemm: operand " + a.at("operand").get<std::string>() +
                                                   " has a dtype without a DLPack equivalent");
                b.rank   = v.rank;
                b.dim    = v.dim;
                b.stride = v.stride;
            } else if (b.as == As::TENSOR) {
                auto const &dim = a.at("dim"), &stride = a.at("stride"), &dtype = a.at("dtype");
                b.rank = static_cast<int>(dim.size());
                RETURN_CUDNN_FRONTEND_ERROR_IF(b.rank > 8 || stride.size() != dim.size(),
                                               error_code_t::GRAPH_NOT_SUPPORTED,
                                               "GroupedGemm: malformed tensor binding " + a.dump());
                for (int k = 0; k < b.rank; k++) {
                    b.dim[k]    = dim[k].get<int64_t>();
                    b.stride[k] = stride[k].get<int64_t>();
                }
                b.dtype = {dtype[0].get<uint8_t>(), dtype[1].get<uint8_t>(), dtype[2].get<uint16_t>()};
            }
            binds_.push_back(b);
        }
        return {error_code_t::OK, ""};
    }

    std::shared_ptr<detail::Kernel const> kernel_;
    std::vector<std::string> names_;
    std::vector<TensorView> views_;
    std::vector<int64_t> align_;
    std::vector<detail::ffi::Any> proto_;
    std::vector<Bind> binds_;
    int device_             = 0;
    int64_t workspace_size_ = 0;
};

}  // namespace ops
}  // namespace cudnn_frontend
