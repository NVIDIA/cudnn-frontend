#pragma once

#include "../cudnn_frontend_utils.h"

namespace cudnn_frontend::detail {

class Context {
    DataType_t compute_data_type      = DataType_t::NOT_SET;
    DataType_t intermediate_data_type = DataType_t::NOT_SET;
    DataType_t io_data_type           = DataType_t::NOT_SET;
    int32_t target_sm_count           = -1;

    std::string name = "";

   public:
    Context&
    set_intermediate_data_type(DataType_t const type) {
        intermediate_data_type = type;
        return *this;
    }

    Context&
    set_io_data_type(DataType_t const type) {
        io_data_type = type;
        return *this;
    }

    Context&
    set_compute_data_type(DataType_t const type) {
        compute_data_type = type;
        return *this;
    }

    DataType_t
    get_io_data_type() const {
        return io_data_type;
    }

    DataType_t
    get_intermediate_data_type() const {
        return intermediate_data_type;
    }

    DataType_t
    get_compute_data_type() const {
        return compute_data_type;
    }

    Context&
    set_name(std::string const& name_) {
        name = name_;
        return *this;
    }

    std::string
    get_name() const {
        return name;
    }

    Context&
    set_target_sm_count(int32_t count) {
        target_sm_count = count;
        return *this;
    }

    int32_t
    get_target_sm_count() const {
        return target_sm_count;
    }

    Context&
    fill_missing_properties(Context const& global_context) {
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(global_context.get_compute_data_type());
        }
        if (get_intermediate_data_type() == DataType_t::NOT_SET) {
            set_intermediate_data_type(global_context.get_intermediate_data_type());
        }
        if (get_io_data_type() == DataType_t::NOT_SET) {
            set_io_data_type(global_context.get_io_data_type());
        }
        return *this;
    }
};

}  // namespace cudnn_frontend::detail