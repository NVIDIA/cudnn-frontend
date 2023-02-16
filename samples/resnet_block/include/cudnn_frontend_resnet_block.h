#pragma once

#include <unordered_map>
#include <map>
#include <vector>
#include <iomanip>
#include <string>
#include <queue>

#include "cudnn_frontend.h"

#if (CUDNN_VERSION >= 8600)
#include "layers/common/include/cudnn_frontend_block_factory.h"
#include "layers/common/cudnn_frontend_resnet_block_helpers.h"

#include "layers/models/resnet/block_params/cudnn_frontend_residual_block_params.h"
#include "layers/models/resnet/block_device_pointer_stores/cudnn_frontend_residual_block_dev_ptr_store.h"

#endif