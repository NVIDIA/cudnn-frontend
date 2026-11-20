# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target admission and resources of the stream used by SE launches."""


def require_arch(capability):
    from cudnn.frost.buffers import cutedsl_arch_requirement_error

    error = cutedsl_arch_requirement_error(capability)
    if error:
        raise NotImplementedError(error)
    if capability == (10, 7):
        from cutlass.base_dsl.version_info import CUDA_VERSION

        if (CUDA_VERSION.major, CUDA_VERSION.minor) < (13, 4):
            raise NotImplementedError(
                "SE causal_conv1d on SM107 requires a CuTe DSL CUDA compiler >= 13.4; "
                f"found {CUDA_VERSION.major}.{CUDA_VERSION.minor}. Install nvidia-cutlass-dsl[cu13]."
            )


def stream_multiprocessors(device):
    """Read current-stream resources without synchronizing or retaining stream handles."""
    import torch
    from cuda.bindings import driver as cuda
    from cudnn._device import execution_context

    def checked(result):
        if result[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"SE stream resource query failed: {result[0]}")
        return result[1]

    with torch.cuda.device(device):
        version = checked(cuda.cuDriverGetVersion())
        if version < 12040:
            # Green contexts were introduced in CUDA 12.4.
            return torch.cuda.get_device_properties(device).multi_processor_count
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        with execution_context(int(stream), torch.cuda.current_device()):
            kind = cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            if version >= 13010 and hasattr(cuda, "cuStreamGetDevResource"):
                resource = checked(cuda.cuStreamGetDevResource(stream, kind))
            else:
                # CUDA 12 bindings expose the stream's converted green context.
                context = checked(cuda.cuStreamGetCtx(stream))
                resource = checked(cuda.cuCtxGetDevResource(context, kind))
            count = int(resource.sm.smCount)
            if count <= 0:
                raise RuntimeError("SE launch stream exposes no SM resources")
            return count
