# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def tensor_list(tensors):
    return [entry for _, entry in sorted(tensors.items(), key=lambda item: int(item[0]))]
