# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Plan-time physical-origin contracts for the half SM100 THD forward bodies."""

PORTS = ("q", "k", "v", "o", "stats")
ROLES = tuple("ragged_" + name for name in PORTS)


def graph_origin_specs(facts):
    """Keep offset dtype, multiplier and token stride; never inspect device data.

    Offset values are absolute in each bound port and must identify whole token
    rows. Lengths describe live rows independently of gaps in those buffers.
    """
    import cudnn

    specs = []
    for row, name in enumerate(PORTS):
        tensor = getattr(facts, name + "_t")
        offset = getattr(tensor, "ragged_offset", None) if tensor is not None else None
        if offset is None:
            specs.append(None)
            continue
        dtype = offset.get_data_type()
        if dtype not in (cudnn.data_type.INT32, cudnn.data_type.INT64):
            raise NotImplementedError("THD physical offsets must be INT32 or INT64")
        stride = int(tensor.get_stride()[2])
        multiplier = int(tensor.get_ragged_offset_multiplier() or 1)
        if stride <= 0 or multiplier <= 0 or stride % multiplier:
            raise NotImplementedError("THD offset multiplier must divide the port's token stride")
        specs.append((row, multiplier, stride, 64 if dtype == cudnn.data_type.INT64 else 32))
    return tuple(specs)


def descriptor_bytes(batch):
    # Four arrays of B maps and one valid-address dead-unit sentinel per array.
    # Five INT64 origin rows follow the aligned tensor-map storage.
    return 4 * (batch + 1) * 128 + ((5 * batch * 8 + 127) // 128) * 128
