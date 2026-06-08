import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, Optional, Tuple


_DIGEST_LENGTH = 16
_SYMBOL_RE = re.compile(r"[^0-9A-Za-z_]")
_REGISTRY_RE = re.compile(r"[^0-9A-Za-z_.]")


@dataclasses.dataclass(frozen=True)
class AOTIdentity:
    kernel_name: str
    payload: Dict[str, Any]
    digest: str


@dataclasses.dataclass(frozen=True)
class AOTArtifactPaths:
    object_file: Path
    shared_library: Path
    metadata_file: Path


@dataclasses.dataclass(frozen=True)
class AOTMetadata:
    identity: AOTIdentity
    symbol: str
    registry_name: Optional[str]
    object_file: str
    shared_library: str
    metadata_file: str


@dataclasses.dataclass(frozen=True)
class AOTExportedArtifact:
    metadata: AOTMetadata
    paths: AOTArtifactPaths


@dataclasses.dataclass(frozen=True)
class AOTLoadedArtifact:
    metadata: AOTMetadata
    paths: AOTArtifactPaths
    module: Any
    function: Callable[..., Any]


def _canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"))


def _package_versions() -> Dict[str, Optional[str]]:
    versions = {}
    for package in ("nvidia-cudnn-frontend", "nvidia-cutlass-dsl", "apache-tvm-ffi", "tvm-ffi"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_aot_identity(
    *,
    kernel_name: str,
    cache_key: Any,
    compile_options: str,
    compute_capability: Any,
) -> AOTIdentity:
    payload = {
        "kernel_name": kernel_name,
        "cache_key": _canonicalize(cache_key),
        "compile_options": compile_options,
        "compute_capability": str(compute_capability),
        "package_versions": _package_versions(),
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    return AOTIdentity(kernel_name=kernel_name, payload=payload, digest=digest)


def build_symbol_name(prefix: str, identity: AOTIdentity) -> str:
    safe_prefix = _SYMBOL_RE.sub("_", prefix).strip("_")
    if not safe_prefix:
        safe_prefix = "cudnnfe_cutedslaot"
    if safe_prefix[0].isdigit():
        safe_prefix = f"_{safe_prefix}"
    return f"{safe_prefix}_{identity.digest}"


def build_registry_name(prefix: str, identity: AOTIdentity) -> str:
    safe_prefix = _REGISTRY_RE.sub(".", prefix).strip(".")
    if not safe_prefix:
        safe_prefix = "cudnnfe.cutedsl_aot"
    return f"{safe_prefix}.{identity.digest}"


def artifact_paths(artifact_dir: os.PathLike[str] | str, symbol: str) -> AOTArtifactPaths:
    base_dir = Path(artifact_dir)
    base_name = _SYMBOL_RE.sub("_", symbol).strip("_")
    if not base_name:
        base_name = "cudnnfe_cutedslaot"
    return AOTArtifactPaths(
        object_file=base_dir / f"{base_name}.o",
        shared_library=base_dir / f"{base_name}.so",
        metadata_file=base_dir / f"{base_name}.json",
    )


def _identity_from_dict(value: Dict[str, Any]) -> AOTIdentity:
    return AOTIdentity(
        kernel_name=value["kernel_name"],
        payload=value["payload"],
        digest=value["digest"],
    )


def _metadata_to_dict(metadata: AOTMetadata) -> Dict[str, Any]:
    return {
        "identity": dataclasses.asdict(metadata.identity),
        "symbol": metadata.symbol,
        "registry_name": metadata.registry_name,
        "object_file": metadata.object_file,
        "shared_library": metadata.shared_library,
        "metadata_file": metadata.metadata_file,
    }


def _metadata_from_dict(value: Dict[str, Any]) -> AOTMetadata:
    return AOTMetadata(
        identity=_identity_from_dict(value["identity"]),
        symbol=value["symbol"],
        registry_name=value.get("registry_name"),
        object_file=value["object_file"],
        shared_library=value["shared_library"],
        metadata_file=value["metadata_file"],
    )


def write_metadata_atomic(metadata: AOTMetadata, path: os.PathLike[str] | str) -> None:
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(_metadata_to_dict(metadata), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, metadata_path)


def read_metadata(path: os.PathLike[str] | str) -> AOTMetadata:
    return _metadata_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _artifact_paths_from_metadata(metadata: AOTMetadata, metadata_path: os.PathLike[str] | str) -> AOTArtifactPaths:
    base_dir = Path(metadata_path).parent
    return AOTArtifactPaths(
        object_file=base_dir / metadata.object_file,
        shared_library=base_dir / metadata.shared_library,
        metadata_file=base_dir / metadata.metadata_file,
    )


def _runtime_libraries() -> Tuple[str, ...]:
    import cutlass.cute as cute

    finder = getattr(cute.runtime, "find_runtime_libraries", None)
    if finder is None:
        return ()
    try:
        libraries = finder(enable_tvm_ffi=True)
    except TypeError:
        libraries = finder()
    if libraries is None:
        return ()
    return tuple(str(library) for library in libraries)


def _link_shared_library(object_file: Path, shared_library: Path) -> None:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Cannot link CuTe DSL AOT artifact: no C++ compiler found")

    cmd = [compiler, "-shared", "-o", str(shared_library), str(object_file)]
    cmd.extend(_runtime_libraries())
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to link CuTe DSL AOT shared library with command "
            f"{cmd!r}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def export_compiled_module(
    compiled: Any,
    paths: AOTArtifactPaths,
    symbol: str,
    metadata: AOTMetadata,
    *,
    force: bool = False,
) -> None:
    paths.object_file.parent.mkdir(parents=True, exist_ok=True)
    if not force and paths.metadata_file.exists():
        existing_metadata = read_metadata(paths.metadata_file)
        if existing_metadata != metadata:
            raise RuntimeError(
                "Existing CuTe DSL AOT artifact metadata does not match the requested export "
                f"for {symbol}; pass force=True to overwrite"
            )
        if paths.object_file.exists() and paths.shared_library.exists():
            return
    elif not force and (paths.object_file.exists() or paths.shared_library.exists()):
        raise FileExistsError(
            f"Partial AOT artifact exists for {symbol} without metadata; pass force=True to overwrite"
        )
    if not force and (paths.object_file.exists() or paths.shared_library.exists()):
        paths.object_file.unlink(missing_ok=True)
        paths.shared_library.unlink(missing_ok=True)

    export_to_c = getattr(compiled, "export_to_c", None)
    if export_to_c is None:
        raise TypeError("Compiled CuTe DSL object does not expose export_to_c")

    export_to_c(str(paths.object_file), function_name=symbol)
    _link_shared_library(paths.object_file, paths.shared_library)
    write_metadata_atomic(metadata, paths.metadata_file)


def _get_module_function(module: Any, symbol: str) -> Callable[..., Any]:
    candidates = (symbol, f"__tvm_ffi_{symbol}")
    for candidate in candidates:
        try:
            func = module[candidate]
        except (KeyError, TypeError, AttributeError):
            pass
        else:
            if func is not None:
                return func

        for getter_name in ("get_function", "get_global_func", "get"):
            getter = getattr(type(module), getter_name, None)
            if getter is None:
                continue
            try:
                func = getter(module, candidate)
            except TypeError:
                continue
            if func is not None:
                return func

        try:
            func = getattr(module, candidate)
        except (AttributeError, RuntimeError):
            continue
        if func is not None:
            return func
    if callable(module):
        return module
    raise RuntimeError(f"Loaded CuTe DSL AOT module does not expose symbol {symbol}")


def load_exported_module(paths: AOTArtifactPaths, symbol: str) -> Tuple[Any, Callable[..., Any]]:
    import cutlass.cute as cute

    module = cute.runtime.load_module(str(paths.shared_library), enable_tvm_ffi=True)
    return module, _get_module_function(module, symbol)


def load_aot_artifact(
    metadata_path: os.PathLike[str] | str,
    *,
    registry_name: Optional[str] = None,
    register: bool = False,
    override: bool = False,
) -> AOTLoadedArtifact:
    metadata = read_metadata(metadata_path)
    paths = _artifact_paths_from_metadata(metadata, metadata_path)
    module, function = load_exported_module(paths, metadata.symbol)
    resolved_registry_name = registry_name if registry_name is not None else metadata.registry_name
    if register and resolved_registry_name is not None:
        register_loaded_function(resolved_registry_name, function, override=override)
    return AOTLoadedArtifact(metadata=metadata, paths=paths, module=module, function=function)


def _registry_module() -> Any:
    import tvm_ffi

    return getattr(tvm_ffi, "registry", tvm_ffi)


def register_loaded_function(name: str, function: Callable[..., Any], *, override: bool = False) -> None:
    registry = _registry_module()
    register = getattr(registry, "register_global_func", None)
    if register is None:
        register = getattr(registry, "register_func", None)
    if register is None:
        raise RuntimeError("TVM-FFI registry does not expose a global registration API")
    try:
        register(name, function, override=override)
    except TypeError:
        register(name, function, allow_override=override)


def get_registered_function(name: str, *, allow_missing: bool = False) -> Optional[Callable[..., Any]]:
    registry = _registry_module()
    getter = getattr(registry, "get_global_func", None)
    if getter is None:
        getter = getattr(registry, "get_func", None)
    if getter is None:
        raise RuntimeError("TVM-FFI registry does not expose a global lookup API")
    try:
        return getter(name, allow_missing=allow_missing)
    except TypeError:
        if allow_missing:
            try:
                return getter(name)
            except (KeyError, ValueError, RuntimeError):
                return None
        return getter(name)


def remove_registered_function(name: str, *, allow_missing: bool = False) -> None:
    registry = _registry_module()
    remover = getattr(registry, "remove_global_func", None)
    if remover is None:
        remover = getattr(registry, "remove_func", None)
    if remover is None:
        if allow_missing:
            return
        raise RuntimeError("TVM-FFI registry does not expose a global removal API")
    try:
        remover(name)
    except (KeyError, ValueError, RuntimeError):
        if not allow_missing:
            raise
