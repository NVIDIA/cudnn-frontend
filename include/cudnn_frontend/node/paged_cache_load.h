#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "pointwise.h"
#include "reduction.h"

namespace cudnn_frontend::graph {

class PagedCacheLoadNode : public NodeCRTP<PagedCacheLoadNode> {
   public:
    PagedCacheLoad_attributes attributes;

    PagedCacheLoadNode(PagedCacheLoad_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::PAGED_CACHE_LOAD;
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);

        auto&& paged_cache_load_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(container, PagedCacheLoad_attributes::input_names::container);
        paged_cache_load_operation_builder.setcontainerDesc(*(tensors.at(container->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(pageTable, PagedCacheLoad_attributes::input_names::pageTable);
        paged_cache_load_operation_builder.setpageTableDesc(*(tensors.at(pageTable->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(seqLen, PagedCacheLoad_attributes::input_names::seqLen);
        paged_cache_load_operation_builder.setsequenceDesc(*(tensors.at(seqLen->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(yOut, PagedCacheLoad_attributes::output_names::yOut);
        paged_cache_load_operation_builder.setyDesc(*(tensors.at(yOut->second->get_uid())));

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = paged_cache_load_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = paged_cache_load_operation_builder.build();
            operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
        } catch (cudnn_frontend::cudnnException& e) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::CUDNN_BACKEND_API_FAILED, e.what());
        }
#endif

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating PagedCacheLoadNode " << attributes.name << "...");
        auto const yOut_dims      = attributes.outputs.at(PagedCacheLoad_attributes::output_names::yOut)->get_dim();
        auto const yOut_strides   = attributes.outputs.at(PagedCacheLoad_attributes::output_names::yOut)->get_stride();
        auto const container_dims = attributes.inputs.at(PagedCacheLoad_attributes::input_names::container)->get_dim();
        auto const pageTable_dims = attributes.inputs.at(PagedCacheLoad_attributes::input_names::pageTable)->get_dim();

        // In the backend, the k-cache is passed as K^T and has dims [B,H,D,S], while v-cache has dims [B,H,S,D]
        // Use the strides to distinguish.
        auto yIsTransposed = yOut_strides[2] == 1;
        auto s_kv          = !yIsTransposed ? yOut_dims[2] : yOut_dims[3];

        auto block_size = container_dims[2];
        auto table_size = pageTable_dims[2];
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (s_kv + (block_size - 1)) / block_size != table_size,
            error_code_t::INVALID_VALUE,
            "Paged cache load: mismatch between max sequence length, block size and page table size");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
#endif
};

inline void
INode::paged_cache_load(std::shared_ptr<Tensor_attributes> container,
                        std::shared_ptr<Tensor_attributes> seqLen,
                        std::shared_ptr<Tensor_attributes> pageTable,
                        PagedCacheLoad_attributes attributes,
                        std::shared_ptr<Tensor_attributes> yOut) {
    attributes.inputs[PagedCacheLoad_attributes::input_names::container] = std::move(container);
    attributes.inputs[PagedCacheLoad_attributes::input_names::seqLen]    = std::move(seqLen);
    attributes.inputs[PagedCacheLoad_attributes::input_names::pageTable] = std::move(pageTable);
    attributes.outputs[PagedCacheLoad_attributes::output_names::yOut]    = std::move(yOut);
    sub_nodes.emplace_back(std::make_unique<PagedCacheLoadNode>(std::move(attributes), context));
}
}  // namespace cudnn_frontend::graph