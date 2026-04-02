"""Shared utility functions for cuDNN repro tool."""

import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


def sha1_seed(raw: str) -> int:
    """Generate a deterministic seed from a string using SHA1."""
    value = int(hashlib.sha1(raw.encode("utf-8")).hexdigest(), 16) % ((1 << 31) - 1)
    return value if value != 0 else 1


def parse_hex_float(value: Any) -> Optional[float]:
    """Parse a float from hex string or numeric value."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if not isinstance(value, str):
        return None
    hex_str = value.strip().lower()
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    if len(hex_str) == 8:
        try:
            return struct.unpack("<f", bytes.fromhex(hex_str))[0]
        except (ValueError, struct.error):
            pass
    try:
        return float(value)
    except ValueError:
        return None


def torch_dtype(io_type: Optional[str]) -> Optional[str]:
    """Convert cuDNN DataType_t string to torch dtype string."""
    if io_type is None:
        return "torch.float16"
    mapping = {
        "BFLOAT16": "torch.bfloat16",
        "HALF": "torch.float16",
        "FLOAT16": "torch.float16",
        "FLOAT": "torch.float32",
        "FLOAT32": "torch.float32",
        "FP8_E4M3": "torch.float8_e4m3fn",
        "FP8_E5M2": "torch.float8_e5m2",
    }
    return mapping.get(io_type.upper(), "torch.float16")


def node_by_tag(payload: dict, *tags: str) -> Optional[dict]:
    """Find the first node whose tag matches one of the requested tags."""
    requested = set(tags)
    for candidate in payload.get("nodes", []):
        if candidate.get("tag") in requested:
            return candidate
    return None


def parse_optional_int(value: Any) -> Optional[int]:
    """Parse an optional integer value."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def tensor_entry(tensors: dict, node_name: Optional[str], label: str, hint: Optional[str]) -> Optional[dict]:
    """Find a tensor entry in the tensors dict by various lookup strategies."""
    if not tensors:
        return None

    def _from_key(key: Any) -> Optional[dict]:
        if key is None:
            return None
        str_key = str(int(key)) if isinstance(key, (int, float)) else str(key)
        return tensors.get(str_key)

    def _from_uid(uid: Any) -> Optional[dict]:
        try:
            uid_int = int(uid) if uid is not None else None
        except (TypeError, ValueError):
            return None
        for value in tensors.values():
            if value.get("uid") == uid_int:
                return value
        return None

    candidates = []
    if hint:
        candidates.append(hint)
        candidates.append(str(hint))
    if node_name:
        candidates.append(f"{node_name}::{label}")
        candidates.append(f"{node_name}::{label.lower()}")
        candidates.append(f"{node_name}::{label.upper()}")
    candidates.extend([label, label.lower(), label.upper()])
    for key in candidates:
        entry = _from_key(key)
        if entry:
            return entry
    direct_uid = _from_uid(hint)
    if direct_uid:
        return direct_uid
    suffix = f"::{label}"
    for key, value in tensors.items():
        skey = str(key)
        if skey.endswith(suffix) or skey == label:
            return value
    return None


def tensor_dtype(entry: Optional[dict]) -> Optional[str]:
    """Return the torch dtype string for a serialized tensor entry."""
    if not entry:
        return None
    return torch_dtype(entry.get("data_type"))


def shape(entry: Optional[dict]) -> Optional[Tuple[int, ...]]:
    """Extract shape tuple from tensor entry."""
    if not entry:
        return None
    dims = entry.get("dim")
    if not dims:
        return None
    return tuple(int(d) for d in dims)


def stride(entry: Optional[dict]) -> Optional[Tuple[int, ...]]:
    """Extract stride tuple from tensor entry."""
    if not entry:
        return None
    strides = entry.get("stride")
    if not strides:
        return None
    return tuple(int(s) for s in strides)


def normalize_k_layout(shape_k: Optional[Tuple[int, ...]], stride_k: Optional[Tuple[int, ...]]) -> tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Convert logged KT layout [B,H,D,S] into BHSD when detected by stride pattern."""
    if shape_k and stride_k and len(shape_k) == 4 and len(stride_k) == 4:
        if int(stride_k[2]) == 1 and int(stride_k[3]) != 1:
            shape_k = (shape_k[0], shape_k[1], shape_k[3], shape_k[2])
            stride_k = (stride_k[0], stride_k[1], stride_k[3], stride_k[2])
    return shape_k, stride_k


def flatten_pass_by_value(value: Any) -> list[int]:
    """Flatten pass_by_value data to a list of integers."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        if value.startswith("0x"):
            return [int(value, 16)]
        try:
            return [int(value)]
        except ValueError:
            return []
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(flatten_pass_by_value(item))
        return result
    return []


def seq_len(entry: Optional[dict]) -> list[int]:
    """Extract seq_len list from tensor entry."""
    if not entry:
        return []
    return flatten_pass_by_value(entry.get("pass_by_value"))


def bool_from_inputs(inputs: dict, target: str) -> Optional[bool]:
    """Check if a target tensor is present in inputs dict."""
    if not inputs:
        return None
    return target in inputs


def infer_block_size(page_table_entry: Optional[dict], seq_len_kv: list[int], k_entry: Optional[dict]) -> Optional[int]:
    """Infer paged-attention block size from serialized tensors."""
    if page_table_entry is None or k_entry is None:
        return None
    page_table_shape = shape(page_table_entry)
    k_shape = shape(k_entry)
    if not page_table_shape or len(page_table_shape) < 3 or not k_shape or len(k_shape) < 3:
        return None
    blocks_per_batch = page_table_shape[2]
    if blocks_per_batch <= 0:
        return None
    max_seq = max(seq_len_kv) if seq_len_kv else None
    if max_seq:
        return max(1, (max_seq + blocks_per_batch - 1) // blocks_per_batch)
    return k_shape[2]


def is_mxfp8_payload(payload: dict, node: dict) -> bool:
    """Detect MXFP8 payloads from scale-factor tensor metadata."""
    tensors = payload.get("tensors", {})
    node_name = node.get("name")
    for label, hint in node.get("inputs", {}).items():
        if not label.startswith(("Descale_", "Scale_")):
            continue
        entry = tensor_entry(tensors, node_name, label, hint)
        if entry is None:
            continue
        data_type = (entry.get("data_type") or "").upper()
        reordering = (entry.get("reordering_type") or "").upper()
        if data_type == "FP8_E8M0" or reordering == "F8_128X4":
            return True
    return False


def parse_ragged_tensor_names(log_text: Optional[str]) -> list[str]:
    """Extract tensor names that have ragged offsets enabled from FE log text."""
    if not log_text:
        return []

    uid_to_name = {}
    current_descriptor_uid = None
    ragged_tensor_names = []

    for line in log_text.splitlines():
        match = re.search(r"Backend Tensor named '([^']+)' with UID (\d+)", line)
        if match:
            uid_to_name[int(match.group(2))] = match.group(1)
            continue

        match = re.search(r"Id:\s+(\d+)", line)
        if match:
            current_descriptor_uid = int(match.group(1))
            continue

        if current_descriptor_uid is not None and "raggedOffset: Enabled UID:" in line:
            tensor_name = uid_to_name.get(current_descriptor_uid)
            if tensor_name is not None:
                ragged_tensor_names.append(tensor_name)

    return ragged_tensor_names


def json_with_max_indent(value: Any, depth: int = 0, indent: int = 2, max_indent_level: int = 3) -> str:
    """Format JSON with limited indentation depth."""
    if isinstance(value, dict):
        if not value:
            return "{}"
        if depth >= max_indent_level:
            return json.dumps(value, separators=(", ", ": "), sort_keys=False)
        pad = " " * (depth * indent)
        child_pad = " " * ((depth + 1) * indent)
        parts = []
        for k, v in value.items():
            rendered = json_with_max_indent(v, depth + 1, indent, max_indent_level)
            parts.append(f"{child_pad}{json.dumps(str(k))}: {rendered}")
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if depth >= max_indent_level:
            return json.dumps(value, separators=(", ", ": "), sort_keys=False)
        pad = " " * (depth * indent)
        child_pad = " " * ((depth + 1) * indent)
        parts = [f"{child_pad}{json_with_max_indent(v, depth + 1, indent, max_indent_level)}" for v in value]
        return "[\n" + ",\n".join(parts) + "\n" + pad + "]"
    return json.dumps(value, sort_keys=False)


def format_json_pretty(value: Any) -> str:
    """Format JSON with pretty indentation."""
    return json_with_max_indent(value, depth=0, indent=2, max_indent_level=3)


def write_text(path: Path, text: str) -> None:
    """Write text to a file, ensuring it ends with newline."""
    path.write_text(text + ("" if text.endswith("\n") else "\n"))


def try_write_text(path: Path, text: str) -> None:
    """Try to write text to a file, printing warning on failure."""
    try:
        write_text(path, text)
    except OSError as exc:
        print(f"warning: failed to write {path}: {exc}", file=sys.stderr)
