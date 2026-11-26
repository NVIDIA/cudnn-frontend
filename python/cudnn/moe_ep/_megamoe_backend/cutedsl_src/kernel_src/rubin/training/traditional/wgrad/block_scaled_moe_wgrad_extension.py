"""Block-scaled MoE wgrad per-expert GMEM views for TMA with instruction-level descriptor overrides.

Every TMA descriptor of the kernel is built once on the host from a template tensor and passed as a
kernel parameter. The fields that differ per expert (base address, token extent of the K tail, the
per-expert strides of the SF plane and of ragged K-major inputs, the output block address of a
discrete C) travel in the TMA instruction itself (PTX ``cp.async.bulk.tensor ...
.override::global_address{.override::global_dim_stride}``). Nothing is written to the workspace,
so there is no init kernel and no ``fence.proxy.tensormap`` in the main kernel.
"""
# Needs the TMA descriptor overrides (cute.copy on the *Override* TMA atoms with a residue
# tensor); helpers/dsl_helpers.py builds them on the public cuTeDSL release wheel.

from typing import ClassVar, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cute.typing import AddressSpace, Pointer
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from ......api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc
from ......helpers.device_workspace import DeviceWorkspace
from ......helpers.dsl_helpers import make_tma_residue_tensor
from .....schedulers.wgrad_mapping import WgradWorkTileInfo, compute_expert_token_range

# Scale factors are expert-ragged in both modes. Each expert's SFA/SFB block is
# padded and 32x4x4-swizzled independently, so a global SF descriptor plus a K
# domain offset cannot describe the concatenated storage. SFA/SFB therefore
# always carry their expert's base, extent and strides in the TMA instruction.
# ``WgradInputOrder`` does not change SF storage; it only decides whether A/B
# are expert-packed as well.
#
# ``tensor2d`` keeps A/B as global 2-D tensors. Their expert range is a K
# domain offset into ``(fixed_dimension, sum_tokens)`` (plain TMA coordinates;
# the per-expert token runs are padded to ``fix_pad_size`` so a K tail never
# reads the next expert, and the zero-filled SF masks the pad rows).
#
# ``tensor_ragged`` makes A/B follow the expert segmentation already required
# by SF: independently contiguous ``(fixed_dimension, expert_tokens)`` blocks
# are concatenated, so A/B also take their base, extent and stride from the
# instruction. The host tensor's unit-stride axis selects K-major versus
# MN-major; its non-unit stride does not describe the cross-expert packing.
WgradInputOrder = Literal["tensor2d", "tensor_ragged"]

# Weight-gradient output storage across experts:
#
# ``dense`` uses one contiguous ``(experts, hidden, intermediate)`` allocation.
# The kernel exposes it as GEMM ``(M=hidden, N=intermediate, L=experts)`` and
# selects the current expert through an L-domain offset of the launch-time
# C TMA descriptor.
#
# ``discrete`` accepts a device ``Int64`` array of independent expert output
# base addresses plus one single-expert tensor template that defines dtype,
# shape, and stride. The template builds one C TMA descriptor; every store or
# reduce-add overrides its global address with the expert's block, so outputs
# may live in unrelated allocations.
WgradOutputStorageMode = Literal["dense", "discrete"]
# Optional tensor-wise A/B scales applied after block-scaled MMA.
WgradGlobalScaleMode = Literal["none", "per_expert"]
WgradTensorRole = Literal["a", "b", "sfa", "sfb", "c"]


class BlockScaledMoeWgradExtension(KernelComponent):
    """Per-expert GMEM views (data tensor + TMA residue tensor) and the kernel's tail reset."""

    work_tile_type: ClassVar[type] = WgradWorkTileInfo
    # Mixed-CGA scheduling claims work through atomic counters in the local workspace. They are
    # ``tail_reset`` regions: the last CTA to finish zeroes them for the next launch (see
    # ``arrive_and_reset_tail``), which needs this counter of finished CTAs. One 16-byte vector so
    # ``DeviceWorkspace.reset_tail`` can clear it with the other counters.
    done_cta_counter_region: ClassVar[str] = "rubin.training.traditional.block_scaled_moe_wgrad.done_ctas"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": int,
            "ab_sf_vec_size": int,
            "a_major_mode": OperandMajorMode,
            "b_major_mode": OperandMajorMode,
            "input_order": str,
            "output_storage_mode": str,
            "accumulate_on_output": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {"fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple])}

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.ab_sf_vec_size = problem_desc["ab_sf_vec_size"]
        self.a_major_mode = problem_desc["a_major_mode"]
        self.b_major_mode = problem_desc["b_major_mode"]
        self.input_order = problem_desc["input_order"]
        self.output_storage_mode = problem_desc["output_storage_mode"]
        self.accumulate_on_output = problem_desc["accumulate_on_output"]
        self.is_mixed_cga = impl_desc.get("fallback_cluster_shape_mn") is not None

        if isinstance(self.expert_count, bool) or self.expert_count <= 0:
            raise ValueError("expert_count must be a positive int.")
        if isinstance(self.ab_sf_vec_size, bool) or self.ab_sf_vec_size <= 0:
            raise ValueError("ab_sf_vec_size must be a positive int.")
        if self.input_order not in ("tensor2d", "tensor_ragged"):
            raise ValueError(f"input_order must be 'tensor2d' or 'tensor_ragged', got {self.input_order!r}.")
        if self.output_storage_mode not in ("dense", "discrete"):
            raise ValueError(f"output_storage_mode must be 'dense' or 'discrete', got {self.output_storage_mode!r}.")

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "BlockScaledMoeWgradExtension":
        if values:
            raise ValueError(f"BlockScaledMoeWgradExtension expected no MLIR values, got {len(values)}.")
        return self

    # ------------------------------------------------------------------ roles

    def uses_override(self, tensor_role: WgradTensorRole) -> bool:
        """Whether this operand's TMA copies carry per-expert descriptor fields in the instruction."""
        if tensor_role in ("sfa", "sfb"):
            return True
        if tensor_role in ("a", "b"):
            return self.input_order == "tensor_ragged"
        if tensor_role == "c":
            return self.output_storage_mode == "discrete"
        raise ValueError(f"Unknown wgrad tensor role: {tensor_role!r}.")

    @property
    def override_roles(self) -> Tuple[WgradTensorRole, ...]:
        """Mainloop operands whose launch-time atoms must be override variants."""
        return tuple(role for role in ("a", "b", "sfa", "sfb") if self.uses_override(role))

    @property
    def discrete_reduce_output(self) -> bool:
        """Discrete C with beta = 1: TMA reduce-add with the address override (inline PTX, no DSL op)."""
        return self.output_storage_mode == "discrete" and self.accumulate_on_output

    @property
    def needs_tail_reset(self) -> bool:
        """Whether the kernel must zero ``tail_reset`` workspace regions itself.

        Only a scheduler that claims work through workspace atomics (``atomic_counter`` work IDs)
        registers such regions. The wgrad mixed-CGA scheduler is CLC based and keeps no device
        state, so this is False in every current configuration; see ``tail_reset_regions_present``.
        """
        return self._tail_reset_regions_present

    _tail_reset_regions_present: bool = False

    def observe_scheduler_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Record whether the scheduler registered ``tail_reset`` regions (call before finalize)."""
        self._tail_reset_regions_present = any(
            region.reset == "tail_reset" for region in device_workspace.regions("local")
        )
        if self._tail_reset_regions_present and self.is_mixed_cga:
            # CLC cancels the surplus clusters of the ticket grid, so ``grid_dim`` is not the
            # number of CTAs that run and the finished-CTA count below could never complete.
            raise NotImplementedError(
                "In-kernel tail reset is not supported together with CLC mixed-CGA scheduling."
            )

    # -------------------------------------------------------------- workspace

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register the finished-CTA counter that drives the in-kernel tail reset (mixed CGA only)."""
        if self.needs_tail_reset:
            device_workspace.register(
                self.done_cta_counter_region, cutlass.Int32, (4,), buffer_space="local",
                byte_alignment=16, reset="tail_reset",
            )

    @cute.jit
    def arrive_and_reset_tail(self, device_workspace: DeviceWorkspace) -> None:
        """Count this CTA as finished; the last one zeroes every ``tail_reset`` workspace region.

        Called by one thread per CTA after the CTA's last workspace access (the scheduler's
        final claim). Counters are only read by claims, and a CTA increments the counter after
        its last claim, so when the count reaches the grid size no CTA touches the counters
        any more and the reset cannot race. The release/acquire pair orders every CTA's claims
        before the reset; the kernel boundary publishes the zeros to the next launch.
        """
        if cutlass.const_expr(not self.needs_tail_reset):
            return
        grid_m, grid_n, grid_l = cute.arch.grid_dim()
        total_ctas = grid_m * grid_n * grid_l
        counter = device_workspace.ptr(self.done_cta_counter_region)
        finished = cute.arch.atomic_add(counter, Int32(1), sem="acq_rel", scope="gpu")
        if finished + Int32(1) == total_ctas:
            cute.arch.fence_acq_rel_gpu()
            device_workspace.reset_tail(Int32(0), 1)

    # ------------------------------------------------------------ expert views

    @cute.jit
    def expert_token_range(self, expert_token_prefix_sum: cute.Tensor, expert_idx: Int32) -> Tuple[Int32, Int32]:
        """(token offset, exact token count) of one expert from the cumulative ends."""
        return compute_expert_token_range(expert_token_prefix_sum, expert_idx)

    @cute.jit
    def operand_view(
        self,
        tensor_role: WgradTensorRole,
        tma_atom: cute.CopyAtom,
        tensor: cute.Tensor,
        real_tensor: Optional[cute.Tensor],
        token_offset: Int32,
        token_count: Int32,
    ) -> Tuple[cute.Tensor, Optional[cute.Tensor]]:
        """Resolve one expert's A / B / SFA / SFB view.

        Returns ``(data, residue)``. For plain TMA operands ``residue`` is None and ``data`` is a
        coordinate view of the launch-time TMA tensor ``tensor``. For override operands ``data`` is
        the expert's real GMEM tensor (pointer into ``real_tensor``, exact token extent, per-expert
        strides) and ``residue`` the matching TMA residue tensor (remaining elements per TMA mode)
        that ``cute.copy`` turns into the instruction's address / size / stride operands.
        ``token_count`` must be the exact count: the K tail zero-fills past it. ``real_tensor`` is
        the launch-time GMEM tensor behind ``tensor`` (None for plain TMA operands).
        """
        shape = tensor.shape
        singleton = Int32(1)

        if cutlass.const_expr(tensor_role in ("a", "b")):
            per_expert_shape = (shape[0], token_count, singleton)
            if cutlass.const_expr(self.input_order == "tensor_ragged"):
                major_mode = self.a_major_mode if tensor_role == "a" else self.b_major_mode
                dtype = real_tensor.element_type
                fixed_dimension = cute.size(tensor, mode=[0])
                # Byte addressing avoids invalid arithmetic on sub-byte fat-pointer lowering.
                byte_offset = (Int64(fixed_dimension) * Int64(token_offset) * dtype.width) // 8
                uint8_pointer = cute.recast_ptr(real_tensor.iterator, dtype=cutlass.Uint8)
                expert_pointer = cute.recast_ptr(uint8_pointer + byte_offset, dtype=dtype)
                if cutlass.const_expr(major_mode == OperandMajorMode.K):
                    expert_stride = (token_count, singleton, Int32(0))
                else:
                    expert_stride = (singleton, fixed_dimension, Int32(0))
                data = cute.make_tensor(expert_pointer, cute.make_layout(per_expert_shape, stride=expert_stride))
                return data, make_tma_residue_tensor(tma_atom, data)
            data = cute.domain_offset((0, token_offset, 0), tensor)
            data = cute.make_tensor(data.iterator, cute.make_layout(per_expert_shape, stride=data.stride))
            return data, None

        if cutlass.const_expr(tensor_role in ("sfa", "sfb")):
            fixed_dimension = cute.size(tensor, mode=[0])
            element_offset = Int64(fixed_dimension) * Int64(token_offset) // self.ab_sf_vec_size
            expert_pointer = real_tensor.iterator + element_offset
            data = cute.make_tensor(
                expert_pointer, tile_atom_to_shape_SF((shape[0], token_count, singleton), self.ab_sf_vec_size)
            )
            return data, make_tma_residue_tensor(tma_atom, data)

        raise ValueError(f"operand_view does not handle tensor role {tensor_role!r}.")

    @cute.jit
    def output_pointer(self, real_output: cute.Tensor, expert_idx: Int32) -> Pointer:
        """Discrete C: the expert's output block base from the Int64 pointer array behind ``real_output``."""
        pointer_tensor = cute.make_tensor(
            cute.make_ptr(cutlass.Int64, real_output.iterator.toint(), AddressSpace.gmem, assumed_align=8),
            cute.make_layout((self.expert_count,)),
        )
        return cute.make_ptr(real_output.element_type, pointer_tensor[expert_idx], AddressSpace.gmem, assumed_align=16)

    @cute.jit
    def output_view(
        self,
        tma_atom_c: cute.CopyAtom,
        output_template: cute.Tensor,
        real_output: Optional[cute.Tensor],
        expert_idx: Int32,
    ) -> Tuple[cute.Tensor, Optional[cute.Tensor]]:
        """Resolve one expert's C view: ``(data, residue)`` as in ``operand_view``.

        ``output_template`` is the launch-time TMA tensor; ``real_output`` the GMEM tensor behind
        it (discrete: its iterator is the Int64 pointer array; None for dense C).

        Dense C is an L-domain offset of the launch-time tensor (plain TMA coordinates). Discrete C
        is the expert block as a real tensor plus its TMA residue: the S2G override store (beta = 0)
        or the inline-PTX reduce-add override (beta = 1) take address, sizes and strides from them.
        """
        shape = output_template.shape
        per_expert_shape = (shape[0], shape[1], Int32(1))
        if cutlass.const_expr(self.output_storage_mode == "dense"):
            data = cute.domain_offset((0, 0, expert_idx), output_template)
            data = cute.make_tensor(data.iterator, cute.make_layout(per_expert_shape, stride=data.stride))
            return data, None
        # Discrete C (store or reduce-add): the expert block as a real tensor plus its TMA residue.
        # Memory strides come from the real tensor; the template's strides are TMA coordinates.
        data = cute.make_tensor(
            self.output_pointer(real_output, expert_idx),
            cute.make_layout(per_expert_shape, stride=real_output.stride),
        )
        return data, make_tma_residue_tensor(tma_atom_c, data)


__all__ = [
    "BlockScaledMoeWgradExtension",
    "WgradGlobalScaleMode",
    "WgradInputOrder",
    "WgradOutputStorageMode",
    "WgradTensorRole",
]
