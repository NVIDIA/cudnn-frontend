# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Process-level runtime ownership for the private MegaMoE backend.

This module is import-light: importing it does not import CUDA Python or
NVSHMEM and does not initialize CUDA. Optional runtime modules are loaded only
when a distributed runtime is actually acquired.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Protocol

import torch
import torch.distributed as dist

from .._contracts import ForwardConfig

_logger = logging.getLogger(__name__)


def _runtime_debug_enabled() -> bool:
    return os.environ.get("MOE_EP_DEBUG_RUNTIME", "0") == "1"


def _runtime_debug(event: str, **details: object) -> None:
    if not _runtime_debug_enabled():
        return
    fields = {
        "time": f"{time.monotonic():.6f}",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", "?"),
        "local_rank": os.environ.get("LOCAL_RANK", "?"),
        "event": event,
        **details,
    }
    print(
        "[moe-ep-runtime] "
        + " ".join(f"{name}={value}" for name, value in fields.items()),
        file=sys.stderr,
        flush=True,
    )


def _runtime_debug_init_status(core) -> object:
    if not _runtime_debug_enabled():
        return "debug-disabled"
    try:
        status = core.init_status()
    except (AttributeError, RuntimeError):
        return "unavailable"
    return getattr(status, "name", status)


class _RuntimeWatchdog:
    """Emit Python stacks and kernel wait channels while NVSHMEM init is blocked."""

    def __init__(self, event: str) -> None:
        self._event = event
        self._stopped = threading.Event()
        try:
            self._interval = float(
                os.environ.get("MOE_EP_RUNTIME_WATCHDOG_SECONDS", "30")
            )
        except ValueError:
            self._interval = 30.0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not _runtime_debug_enabled() or self._interval <= 0:
            return
        # faulthandler's timer is implemented outside the Python interpreter
        # lock, so it still emits the main-thread stack if a native NVSHMEM
        # call holds the GIL. The Python thread adds /proc wait-channel data
        # whenever the GIL remains schedulable.
        faulthandler.dump_traceback_later(
            self._interval,
            repeat=True,
            file=sys.stderr,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="moe-ep-runtime-watchdog",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stopped.set()
        if _runtime_debug_enabled() and self._interval > 0:
            faulthandler.cancel_dump_traceback_later()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _run(self) -> None:
        sample = 0
        while not self._stopped.wait(self._interval):
            sample += 1
            wait_channels: list[str] = []
            for task_dir in sorted(Path("/proc/self/task").glob("[0-9]*")):
                try:
                    thread_name = (task_dir / "comm").read_text().strip()
                    wait_channel = (task_dir / "wchan").read_text().strip()
                except OSError as exc:
                    wait_channels.append(f"{task_dir.name}:unavailable({exc.errno})")
                else:
                    wait_channels.append(
                        f"{task_dir.name}:{thread_name}:{wait_channel or '-'}"
                    )
            _runtime_debug(
                "watchdog",
                blocked_event=self._event,
                sample=sample,
                threads=";".join(wait_channels),
            )
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


class RuntimeUnavailableError(RuntimeError):
    """The requested runtime cannot be loaded or initialized."""


class RuntimeInitState(Enum):
    """Normalized NVSHMEM initialization state."""

    NOT_INITIALIZED = "not_initialized"
    INITIALIZED = "initialized"
    PARTIAL = "partial"


@dataclass(frozen=True)
class RuntimeWorld:
    """Group-relative geometry and ordered membership used for bootstrap."""

    rank: int
    size: int
    group: object
    global_ranks: tuple[int, ...]

    @property
    def identity(self) -> tuple[int, int, tuple[int, ...]]:
        """Stable process-group identity independent of ProcessGroup objects."""

        return self.rank, self.size, self.global_ranks


class NvshmemRuntimeProvider(Protocol):
    """Injectable NVSHMEM lifecycle boundary used by :class:`RuntimeManager`."""

    def initialization_state(self) -> RuntimeInitState: ...

    def initialize(self, device: torch.device, world: RuntimeWorld) -> None: ...

    def rank(self) -> int: ...

    def world_size(self) -> int: ...

    def device(self) -> torch.device: ...

    def finalize(self) -> None: ...


def _resolve_world(config: ForwardConfig) -> RuntimeWorld:
    if config.ep_group is None:
        if (
            config.ep_size != 1
            or config.ep_rank != 0
            or config.ep_global_ranks
        ):
            raise ValueError(
                "ep_group=None requires ep_size=1, ep_rank=0, and no "
                "distributed rank membership"
            )
        return RuntimeWorld(rank=0, size=1, group=None, global_ranks=())

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "distributed MegaMoE runtime requires torch.distributed to be initialized"
        )

    group = config.ep_group
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    global_ranks = tuple(
        dist.get_global_rank(group, group_rank)
        for group_rank in range(size)
    )
    if (rank, size) != (config.ep_rank, config.ep_size):
        raise RuntimeError(
            "ForwardConfig EP geometry does not match its process group: "
            f"config=({config.ep_rank}, {config.ep_size}), "
            f"runtime=({rank}, {size})"
        )
    if global_ranks != config.ep_global_ranks:
        raise RuntimeError(
            "ForwardConfig EP membership does not match its process group: "
            f"config={config.ep_global_ranks}, runtime={global_ranks}"
        )
    return RuntimeWorld(
        rank=rank,
        size=size,
        group=group,
        global_ranks=global_ranks,
    )


def _canonical_cuda_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"MegaMoE runtime requires a CUDA device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _spans_default_distributed_world(world: RuntimeWorld) -> bool:
    """Whether ``world`` has the default world's complete ordered membership."""

    if not dist.is_available() or not dist.is_initialized():
        return False
    return world.global_ranks == tuple(range(dist.get_world_size()))


def _load_nvshmem_core():
    try:
        import nvshmem.core as core
    except (ImportError, OSError) as exc:
        raise RuntimeUnavailableError(
            "MegaMoE distributed runtime requires nvshmem4py and NVSHMEM libraries"
        ) from exc
    return core


def _normalize_nvshmem_init_state(status) -> RuntimeInitState:
    """Normalize enum and integer forms used across nvshmem4py releases."""

    name = getattr(status, "name", "")
    if name.endswith("NOT_INITIALIZED"):
        return RuntimeInitState.NOT_INITIALIZED
    if name.endswith("IS_INITIALIZED") or name.endswith(
        ("LIMITED_MPG", "FULL_MPG")
    ):
        return RuntimeInitState.INITIALIZED
    if name.endswith("IS_BOOTSTRAPPED"):
        return RuntimeInitState.PARTIAL

    try:
        value = int(getattr(status, "value", status))
    except (TypeError, ValueError):
        return RuntimeInitState.PARTIAL
    if value == 0:
        return RuntimeInitState.NOT_INITIALIZED
    if value in {2, 3, 4}:
        return RuntimeInitState.INITIALIZED
    return RuntimeInitState.PARTIAL


class _DefaultNvshmemRuntimeProvider:
    """Lazy adapter over the installed ``nvshmem.core`` API."""

    def initialization_state(self) -> RuntimeInitState:
        core = _load_nvshmem_core()
        try:
            status = core.init_status()
        except Exception as exc:
            raise RuntimeUnavailableError(
                "failed to query NVSHMEM initialization status"
            ) from exc
        return _normalize_nvshmem_init_state(status)

    def initialize(self, device: torch.device, world: RuntimeWorld) -> None:
        if world.size <= 1:
            raise ValueError("NVSHMEM initialization requires a distributed subgroup")
        if world.group is None:
            raise ValueError("NVSHMEM initialization requires a process group")

        core = _load_nvshmem_core()
        started_at = time.monotonic()
        _runtime_debug(
            "initialize.begin",
            device=device,
            ep_rank=world.rank,
            ep_size=world.size,
            global_ranks=world.global_ranks,
        )
        try:
            import numpy as np

            try:
                from cuda.core.experimental import Device
            except ImportError:
                from cuda.core import Device

            torch.cuda.set_device(device)
            cuda_device = Device(device.index)
            cuda_device.set_current()
            _runtime_debug("initialize.cuda-current", device=device)

            uid = core.get_unique_id(empty=(world.rank != 0))
            _runtime_debug("initialize.uid-created")
            uid_bytes = uid._data.view(np.uint8).copy()
            uid_tensor = torch.from_numpy(uid_bytes)
            group_backend = dist.get_backend(world.group)
            if (
                group_backend == dist.Backend.NCCL
                or str(group_backend).lower() == "nccl"
            ):
                uid_tensor = uid_tensor.to(device=device)
            root_global_rank = dist.get_global_rank(world.group, 0)
            if root_global_rank != world.global_ranks[0]:
                raise RuntimeError(
                    "EP subgroup root changed during NVSHMEM bootstrap"
                )
            _runtime_debug(
                "initialize.uid-broadcast.begin",
                backend=group_backend,
                root_global_rank=root_global_rank,
                tensor_device=uid_tensor.device,
                tensor_bytes=uid_tensor.numel() * uid_tensor.element_size(),
            )
            dist.broadcast(
                uid_tensor,
                src=root_global_rank,
                group=world.group,
            )
            _runtime_debug("initialize.uid-broadcast.end")
            _runtime_debug("initialize.torch-barrier.begin")
            dist.barrier(group=world.group)
            _runtime_debug("initialize.torch-barrier.end")
            uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

            watchdog = _RuntimeWatchdog("core.init")
            _runtime_debug("initialize.core-init.begin")
            watchdog.start()
            try:
                core.init(
                    device=cuda_device,
                    uid=uid,
                    rank=world.rank,
                    nranks=world.size,
                    initializer_method="uid",
                )
            finally:
                watchdog.close()
            _runtime_debug(
                "initialize.core-init.end",
                elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
                init_status=_runtime_debug_init_status(core),
            )
        except RuntimeUnavailableError:
            raise
        except Exception as exc:
            _runtime_debug(
                "initialize.error",
                error_type=type(exc).__name__,
                error=repr(exc),
                elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
            )
            raise RuntimeUnavailableError(
                "failed to initialize the NVSHMEM EP subgroup runtime"
            ) from exc

    def rank(self) -> int:
        try:
            return int(_load_nvshmem_core().my_pe())
        except Exception as exc:
            raise RuntimeUnavailableError("failed to query the NVSHMEM PE rank") from exc

    def world_size(self) -> int:
        try:
            return int(_load_nvshmem_core().n_pes())
        except Exception as exc:
            raise RuntimeUnavailableError(
                "failed to query the NVSHMEM PE world size"
            ) from exc

    def device(self) -> torch.device:
        try:
            from nvshmem.core.memory import _cached_device

            cached = _cached_device["device"]
            if cached is None:
                raise RuntimeError("NVSHMEM cached device is empty")
            return torch.device("cuda", int(cached.device_id))
        except Exception as exc:
            raise RuntimeUnavailableError(
                "failed to query the NVSHMEM initialization device"
            ) from exc

    def finalize(self) -> None:
        core = _load_nvshmem_core()
        started_at = time.monotonic()
        _runtime_debug(
            "finalize.begin",
            init_status=_runtime_debug_init_status(core),
        )
        watchdog = _RuntimeWatchdog("core.finalize")
        watchdog.start()
        try:
            core.finalize()
        except Exception as exc:
            _runtime_debug(
                "finalize.error",
                error_type=type(exc).__name__,
                error=repr(exc),
                elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
            )
            raise RuntimeUnavailableError("failed to finalize NVSHMEM") from exc
        finally:
            watchdog.close()
        _runtime_debug(
            "finalize.end",
            elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
            init_status=_runtime_debug_init_status(core),
        )


@dataclass
class _ActiveRuntime:
    token: object
    device: torch.device
    world: RuntimeWorld
    provider: Optional[NvshmemRuntimeProvider]
    owns_runtime: bool
    ref_count: int = 1
    cleanup_required: bool = False


class _ProcessRuntimeRegistry:
    """State shared by every RuntimeManager instance in this process."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.active: Optional[_ActiveRuntime] = None


_PROCESS_RUNTIME_REGISTRY = _ProcessRuntimeRegistry()


class RuntimeHandle:
    """Per-backend lease on process-level runtime state."""

    def __init__(
        self,
        manager: "RuntimeManager",
        token: object,
        device: torch.device,
        world: RuntimeWorld,
        owns_runtime: bool,
    ) -> None:
        self._manager = manager
        self._token = token
        self.device = device
        self.rank = world.rank
        self.world_size = world.size
        self.group = world.group
        self.global_ranks = world.global_ranks
        self.owns_runtime = owns_runtime
        self._closed = False
        self._close_lock = threading.Lock()

    @property
    def nvshmem_enabled(self) -> bool:
        return self.world_size > 1

    @property
    def closed(self) -> bool:
        return self._closed

    def ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MegaMoE runtime handle is closed")

    def current_stream(self) -> torch.cuda.Stream:
        self.ensure_open()
        return torch.cuda.current_stream(self.device)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._manager._release(self._token)
            self._closed = True


class RuntimeManager:
    """Reference-counted owner for one process-global runtime subgroup."""

    def __init__(
        self,
        *,
        provider_factory: Callable[[], NvshmemRuntimeProvider] = (
            _DefaultNvshmemRuntimeProvider
        ),
        world_resolver: Callable[[ForwardConfig], RuntimeWorld] = _resolve_world,
        keep_alive: bool = False,
    ) -> None:
        self._provider_factory = provider_factory
        self._world_resolver = world_resolver
        self._keep_alive = keep_alive

    @property
    def ref_count(self) -> int:
        with _PROCESS_RUNTIME_REGISTRY.lock:
            active = _PROCESS_RUNTIME_REGISTRY.active
            return 0 if active is None else active.ref_count

    @property
    def active_device(self) -> Optional[torch.device]:
        with _PROCESS_RUNTIME_REGISTRY.lock:
            active = _PROCESS_RUNTIME_REGISTRY.active
            return None if active is None else active.device

    def acquire(
        self,
        config: ForwardConfig,
        device: torch.device,
    ) -> RuntimeHandle:
        device = _canonical_cuda_device(device)
        world = self._world_resolver(config)

        with _PROCESS_RUNTIME_REGISTRY.lock:
            if _PROCESS_RUNTIME_REGISTRY.active is not None:
                active = _PROCESS_RUNTIME_REGISTRY.active
                _runtime_debug(
                    "manager.acquire-reuse.begin",
                    ref_count=active.ref_count,
                    owns_runtime=active.owns_runtime,
                    cleanup_required=active.cleanup_required,
                )
                if active.cleanup_required:
                    raise RuntimeError(
                        "MegaMoE process runtime requires cleanup before reacquire"
                    )
                if active.device != device:
                    raise ValueError(
                        f"MegaMoE process runtime is bound to {active.device}; "
                        f"cannot acquire it for {device}"
                    )
                if active.world.identity != world.identity:
                    raise RuntimeError(
                        "MegaMoE process runtime is already bound to a different "
                        "EP subgroup"
                    )
                active.ref_count += 1
                _runtime_debug(
                    "manager.acquire-reuse.end",
                    ref_count=active.ref_count,
                )
                return RuntimeHandle(
                    self,
                    active.token,
                    active.device,
                    active.world,
                    active.owns_runtime,
                )

            provider: Optional[NvshmemRuntimeProvider] = None
            owns_runtime = False
            _runtime_debug(
                "manager.acquire-new.begin",
                device=device,
                ep_rank=world.rank,
                ep_size=world.size,
            )
            if world.size > 1:
                provider = self._provider_factory()
                status = provider.initialization_state()
                _runtime_debug(
                    "manager.acquire-new.state",
                    init_status=status.value,
                )
                if status is RuntimeInitState.PARTIAL:
                    raise RuntimeError(
                        "cannot attach to a partially initialized NVSHMEM runtime"
                    )
                if (
                    status is RuntimeInitState.INITIALIZED
                    and not _spans_default_distributed_world(world)
                ):
                    raise RuntimeError(
                        "cannot safely attach an externally initialized NVSHMEM "
                        "runtime to a non-WORLD EP subgroup because its ordered "
                        "membership cannot be verified"
                    )
                if status is RuntimeInitState.NOT_INITIALIZED:
                    try:
                        provider.initialize(device, world)
                    except Exception as initialization_error:
                        self._rollback_failed_initialization(
                            provider,
                            device,
                            world,
                            initialization_error,
                        )
                        raise
                    owns_runtime = True

                try:
                    provider_device = provider.device()
                    provider_rank = provider.rank()
                    provider_size = provider.world_size()
                except Exception as validation_error:
                    if owns_runtime:
                        self._cleanup_owned_runtime_after_error(
                            provider,
                            device,
                            world,
                            validation_error,
                        )
                    raise

                if provider_device != device:
                    if owns_runtime:
                        self._cleanup_owned_runtime_after_error(
                            provider,
                            device,
                            world,
                            RuntimeError(
                                "NVSHMEM initialization device does not match "
                                f"the requested device: nvshmem={provider_device}, "
                                f"requested={device}"
                            ),
                        )
                    ownership = (
                        "owned"
                        if owns_runtime
                        else "externally initialized"
                    )
                    raise RuntimeError(
                        f"{ownership} NVSHMEM runtime is bound to "
                        f"{provider_device}, not the requested device {device}"
                    )
                if (provider_rank, provider_size) != (world.rank, world.size):
                    if owns_runtime:
                        self._cleanup_owned_runtime_after_error(
                            provider,
                            device,
                            world,
                            RuntimeError("NVSHMEM PE geometry mismatch"),
                        )
                    raise RuntimeError(
                        "NVSHMEM PE geometry does not match the EP subgroup: "
                        f"nvshmem=({provider_rank}, {provider_size}), "
                        f"torch=({world.rank}, {world.size})"
                    )

            token = object()
            _PROCESS_RUNTIME_REGISTRY.active = _ActiveRuntime(
                token=token,
                device=device,
                world=world,
                provider=provider,
                owns_runtime=owns_runtime,
            )
            _runtime_debug(
                "manager.acquire-new.end",
                owns_runtime=owns_runtime,
                ref_count=1,
            )
            return RuntimeHandle(self, token, device, world, owns_runtime)

    @staticmethod
    def _mark_cleanup_required(
        provider: NvshmemRuntimeProvider,
        device: torch.device,
        world: RuntimeWorld,
    ) -> None:
        _PROCESS_RUNTIME_REGISTRY.active = _ActiveRuntime(
            token=object(),
            device=device,
            world=world,
            provider=provider,
            owns_runtime=True,
            ref_count=0,
            cleanup_required=True,
        )

    @classmethod
    def _rollback_failed_initialization(
        cls,
        provider: NvshmemRuntimeProvider,
        device: torch.device,
        world: RuntimeWorld,
        initialization_error: Exception,
    ) -> None:
        try:
            state = provider.initialization_state()
            if state is not RuntimeInitState.NOT_INITIALIZED:
                provider.finalize()
        except Exception as cleanup_error:
            cls._mark_cleanup_required(provider, device, world)
            raise RuntimeError(
                "NVSHMEM initialization failed and rollback requires retry"
            ) from cleanup_error
        _logger.debug(
            "rolled back failed NVSHMEM initialization: %s",
            initialization_error,
        )

    @classmethod
    def _cleanup_owned_runtime_after_error(
        cls,
        provider: NvshmemRuntimeProvider,
        device: torch.device,
        world: RuntimeWorld,
        original_error: Exception,
    ) -> None:
        try:
            provider.finalize()
        except Exception as cleanup_error:
            cls._mark_cleanup_required(provider, device, world)
            raise RuntimeError(
                "NVSHMEM validation failed and cleanup requires retry"
            ) from cleanup_error
        _logger.debug(
            "finalized owned NVSHMEM after validation failure: %s",
            original_error,
        )

    def retry_cleanup(self) -> None:
        """Retry cleanup after an acquire-time rollback failure."""

        with _PROCESS_RUNTIME_REGISTRY.lock:
            active = _PROCESS_RUNTIME_REGISTRY.active
            if active is None:
                return
            if not active.cleanup_required or active.ref_count != 0:
                raise RuntimeError(
                    "MegaMoE process runtime does not have retryable cleanup"
                )
            if active.provider is None:
                raise RuntimeError(
                    "retryable MegaMoE runtime cleanup has no provider"
                )
            active.provider.finalize()
            _PROCESS_RUNTIME_REGISTRY.active = None

    def shutdown(self) -> None:
        """Finalize an idle process runtime at a caller-controlled collective point."""

        with _PROCESS_RUNTIME_REGISTRY.lock:
            active = _PROCESS_RUNTIME_REGISTRY.active
            if active is None:
                return
            if active.ref_count != 0:
                raise RuntimeError(
                    "cannot shut down the MegaMoE process runtime while "
                    f"{active.ref_count} handles remain active"
                )
            if active.provider is not None and (
                active.owns_runtime or active.cleanup_required
            ):
                _runtime_debug(
                    "manager.shutdown-finalize.begin",
                    cleanup_required=active.cleanup_required,
                )
                active.provider.finalize()
                _runtime_debug("manager.shutdown-finalize.end")
            _PROCESS_RUNTIME_REGISTRY.active = None
            _runtime_debug("manager.shutdown.end")

    def _release(self, token: object) -> None:
        with _PROCESS_RUNTIME_REGISTRY.lock:
            active = _PROCESS_RUNTIME_REGISTRY.active
            if active is None or active.token is not token:
                _runtime_debug("manager.release-stale")
                return
            if active.ref_count <= 0:
                raise RuntimeError(
                    "MegaMoE process runtime has invalid release state"
                )

            _runtime_debug(
                "manager.release.begin",
                ref_count=active.ref_count,
                owns_runtime=active.owns_runtime,
                cleanup_required=active.cleanup_required,
            )
            if active.cleanup_required:
                if active.ref_count != 1 or active.provider is None:
                    raise RuntimeError(
                        "MegaMoE process runtime has invalid retry state"
                    )
                active.provider.finalize()
                _PROCESS_RUNTIME_REGISTRY.active = None
                _runtime_debug("manager.release.cleanup-retry.end")
                return

            if active.ref_count > 1:
                active.ref_count -= 1
                _runtime_debug(
                    "manager.release-retained",
                    ref_count=active.ref_count,
                )
                return

            if self._keep_alive and not active.cleanup_required:
                active.ref_count = 0
                _runtime_debug(
                    "manager.release-idle",
                    owns_runtime=active.owns_runtime,
                )
                return

            if active.owns_runtime and active.provider is not None:
                _runtime_debug("manager.release-finalize.begin")
                try:
                    active.provider.finalize()
                except Exception:
                    active.cleanup_required = True
                    _runtime_debug("manager.release-finalize.error")
                    raise
                _runtime_debug("manager.release-finalize.end")
            _PROCESS_RUNTIME_REGISTRY.active = None
            _runtime_debug("manager.release.end", ref_count=0)


_DEFAULT_RUNTIME_MANAGER = RuntimeManager(keep_alive=True)


def get_runtime_manager() -> RuntimeManager:
    """Return the process-level manager used by the default MegaMoE backend."""

    return _DEFAULT_RUNTIME_MANAGER


__all__ = [
    "NvshmemRuntimeProvider",
    "RuntimeHandle",
    "RuntimeInitState",
    "RuntimeManager",
    "RuntimeUnavailableError",
    "RuntimeWorld",
    "get_runtime_manager",
]
