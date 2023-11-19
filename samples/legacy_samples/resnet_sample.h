#pragma once

#include "cudnn_frontend.h"
#include "resnet_block/include/cudnn_frontend_resnet_block.h"
#include "../utils/helpers.h"

void
RunResidualBlock(cudnn_frontend::ResidualBlockParams const &params,
                 cudnn_frontend::ResidualBlockDevPtrStore *devPtrStore,
                 const std::string &type);