#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and run every canonical documentation sample."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
SAMPLE_ROOT = DOCS_ROOT / "samples"
FERN_ROOT = DOCS_ROOT / "fern"
REQUIRED_FIELDS = {
    "id",
    "path",
    "language",
    "mode",
    "owners",
    "apis",
    "regions",
    "command",
    "requirements",
    "timeout_seconds",
    "expected",
    "verification",
    "purpose",
}
EXECUTABLE_LANGUAGES = {"python", "py", "cpp", "c++", "cuda", "bash", "sh", "shell", "cmake"}
FENCE_LANGUAGES = EXECUTABLE_LANGUAGES | {"console", "json", "mermaid", "puml", "text", "toml", "yaml"}
CODE_COMPONENT_PATTERN = re.compile(r"<Code\b(?P<attributes>.*?)/>", re.DOTALL)
MARKDOWN_COMPONENT_PATTERN = re.compile(r"<Markdown\b(?P<attributes>.*?)/>", re.DOTALL)
COMPONENT_ATTRIBUTE_PATTERN = re.compile(r'(?P<name>[A-Za-z][A-Za-z0-9]*)="(?P<value>[^"]*)"')


class ManifestError(RuntimeError):
    pass


def load_sample_manifest(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load {path}: {exc}") from exc
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ManifestError("sample manifest must contain a 'samples' list")
    ids: set[str] = set()
    paths: set[str] = set()
    for index, sample in enumerate(samples):
        missing = REQUIRED_FIELDS - sample.keys()
        if missing:
            raise ManifestError(f"sample row {index} is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(sample["expected"], str) or not sample["expected"].strip():
            raise ManifestError(f"sample row {index} must declare a nonblank expected result")
        if "fallback_command" in sample and (
            not isinstance(sample["fallback_command"], list)
            or not sample["fallback_command"]
            or not all(isinstance(part, str) and part for part in sample["fallback_command"])
        ):
            raise ManifestError(f"sample row {index} has an invalid fallback_command")
        if sample["id"] in ids:
            raise ManifestError(f"duplicate sample id: {sample['id']}")
        if sample["path"] in paths:
            raise ManifestError(f"duplicate sample path: {sample['path']}")
        ids.add(sample["id"])
        paths.add(sample["path"])
    return samples


def snippet_language(target: str) -> str | None:
    path = target.split(":", 1)[0]
    name = Path(path).name
    if name == "CMakeLists.txt" or path.endswith(".cmake"):
        return "cmake"
    suffix = Path(path).suffix
    return {
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cu": "cuda",
        ".h": "cpp",
        ".hpp": "cpp",
        ".json": "json",
        ".py": "python",
        ".sh": "bash",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(suffix)


def _component_attributes(attributes: str) -> dict[str, str]:
    return {match.group("name"): match.group("value") for match in COMPONENT_ATTRIBUTE_PATTERN.finditer(attributes)}


def fern_sample_references(page: Path) -> list[str]:
    """Return documentation-relative sample paths included by Fern components."""
    text = page.read_text(encoding="utf-8")
    references: list[str] = []
    for match in CODE_COMPONENT_PATTERN.finditer(text):
        source = _component_attributes(match.group("attributes")).get("src")
        if not source or source.startswith(("http://", "https://")):
            continue
        target = (FERN_ROOT / source.lstrip("/")) if source.startswith("/") else (page.parent / source)
        try:
            relative = target.resolve().relative_to(DOCS_ROOT.resolve()).as_posix()
        except ValueError:
            continue
        if relative.startswith("samples/"):
            references.append(relative)
    return references


def fern_region_references(page: Path) -> list[str]:
    """Return absolute Fern snippet paths used for sample regions."""
    text = page.read_text(encoding="utf-8")
    references: list[str] = []
    for match in MARKDOWN_COMPONENT_PATTERN.finditer(text):
        source = _component_attributes(match.group("attributes")).get("src")
        if source and source.startswith("/snippets/generated/"):
            references.append(source)
    return references


def scan_markdown_code(path: Path) -> tuple[list[dict], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    section = ""
    blocks: list[dict] = []
    problems: list[str] = []
    current: dict | None = None
    for line_number, line in enumerate(lines, 1):
        if current is not None:
            marker = current["marker"]
            if re.match(rf"^[ \t]*{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$", line):
                blocks.append(current)
                current = None
            else:
                current["content"].append((line_number, line))
            continue
        heading = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if heading:
            section = heading.group(1).casefold()
        fence = re.match(r"^[ \t]*(?P<marker>`{3,}|~{3,})(?P<info>.*)$", line)
        if fence:
            info = fence.group("info").strip()
            language = info.split()[0].casefold() if info else ""
            if not language:
                problems.append(f"{path}:{line_number}: opening code fence has no language label")
            elif language not in FENCE_LANGUAGES:
                problems.append(f"{path}:{line_number}: unsupported code-fence language {language!r}")
            current = {
                "line": line_number,
                "marker": fence.group("marker"),
                "language": language,
                "section": section,
                "content": [],
            }
            continue
    if current is not None:
        problems.append(f"{path}:{current['line']}: code fence is not closed")
    return blocks, problems


def find_inline_executable_examples(path: Path) -> list[int]:
    blocks, _ = scan_markdown_code(path)
    findings: list[int] = []
    for block in blocks:
        if not block["section"].startswith("examples") or block["language"] not in EXECUTABLE_LANGUAGES:
            continue
        meaningful = [line for _, line in block["content"] if line.strip()]
        if meaningful:
            findings.append(block["line"])
    return findings


def _requirements_met(requirements: list[str]) -> tuple[bool, str]:
    probe_env = os.environ.copy()
    probe_env["PYTHONPATH"] = str(REPO_ROOT / "python") + os.pathsep + probe_env.get("PYTHONPATH", "")
    for requirement in requirements:
        if requirement in {"python", "linux"}:
            if requirement == "linux" and not sys.platform.startswith("linux"):
                return False, "requires Linux"
            continue
        if requirement in {"bash", "nvcc", "cmake", "ninja"} and shutil.which(requirement) is None:
            return False, f"requires {requirement}"
        if requirement in {"bash", "nvcc", "cmake", "ninja"}:
            continue
        if requirement == "cuda":
            probe = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True)
            if probe.returncode != 0:
                return False, "requires a visible NVIDIA GPU"
            continue
        if requirement == "cutedsl":
            probe = subprocess.run([sys.executable, "-c", "import cutlass"], cwd=REPO_ROOT, env=probe_env, text=True, capture_output=True)
            if probe.returncode != 0:
                return False, "requires CUTLASS DSL"
            continue
        if requirement.startswith("sm"):
            minimum = int(requirement.removeprefix("sm"))
            probe = subprocess.run(
                [sys.executable, "-c", "import torch; p=torch.cuda.get_device_capability(); print(p[0]*10+p[1])"],
                cwd=REPO_ROOT,
                env=probe_env,
                text=True,
                capture_output=True,
            )
            if probe.returncode != 0 or int(probe.stdout.strip()) < minimum:
                return False, f"requires {requirement}+"
            continue
        if requirement.startswith("cudnn>="):
            minimum = int(requirement.partition(">=")[2])
            probe = subprocess.run(
                [sys.executable, "-c", "import cudnn; print(cudnn.backend_version())"],
                cwd=REPO_ROOT,
                env=probe_env,
                text=True,
                capture_output=True,
            )
            if probe.returncode != 0 or int(probe.stdout.strip()) < minimum:
                return False, f"requires cuDNN {minimum // 10000}.{(minimum // 100) % 100} or newer"
            continue
        if requirement.startswith("cublaslt>="):
            minimum = int(requirement.partition(">=")[2])
            version = 0
            for library in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
                try:
                    version = ctypes.CDLL(library).cublasLtGetVersion()
                    break
                except OSError:
                    continue
            if version < minimum:
                return False, f"requires cublasLt {minimum // 10000}.{(minimum // 100) % 100} or newer"
            continue
        if requirement == "causal_conv1d_bindings":
            probe = subprocess.run(
                [sys.executable, "-c", "import cudnn; raise SystemExit(0 if hasattr(cudnn, 'causal_conv1d_forward') else 1)"],
                cwd=REPO_ROOT,
                env=probe_env,
                text=True,
                capture_output=True,
            )
            if probe.returncode != 0:
                return False, "requires an extension built with Causal Conv1d bindings"
            continue
        if requirement == "disposable_python_environment":
            return False, "requires a disposable Python environment for package installation"
        return False, f"unknown or unmet requirement {requirement}"
    return True, ""


def validate_manifest(samples: list[dict]) -> None:
    registered = {sample["path"] for sample in samples}
    for sample in samples:
        path = DOCS_ROOT / sample["path"]
        if not path.is_file():
            raise ManifestError(f"missing sample file: {sample['path']}")
        for owner in sample["owners"]:
            if not (DOCS_ROOT / owner).is_file():
                raise ManifestError(f"sample {sample['id']} has missing owner {owner}")
    actual = {
        path.relative_to(DOCS_ROOT).as_posix()
        for path in SAMPLE_ROOT.rglob("*")
        if path.is_file() and (path.suffix in {".py", ".cc", ".cpp", ".cu", ".sh", ".cmake"} or path.name == "CMakeLists.txt") and path.name != "__init__.py"
    }
    extra = actual - registered
    missing = registered - actual
    if extra:
        raise ManifestError(f"unregistered sample files: {', '.join(sorted(extra))}")
    if missing:
        raise ManifestError(f"registered sample files not found: {', '.join(sorted(missing))}")

    inline: list[str] = []
    fence_problems: list[str] = []
    fern_samples: set[str] = set()
    fern_regions: set[str] = set()
    for page in DOCS_ROOT.rglob("*.md"):
        _, page_fence_problems = scan_markdown_code(page)
        fence_problems.extend(page_fence_problems)
        for line in find_inline_executable_examples(page):
            inline.append(f"{page.relative_to(DOCS_ROOT)}:{line}")
        text = page.read_text(encoding="utf-8")
        if re.search(r'^\s*--8<--\s+["\']', text, re.MULTILINE):
            fence_problems.append(f"{page}: legacy sample inclusion directive remains")
        fern_samples.update(fern_sample_references(page))
        fern_regions.update(fern_region_references(page))
    if fence_problems:
        raise ManifestError("markdown code-fence problems: " + "; ".join(fence_problems))
    if inline:
        raise ManifestError(f"inline executable examples: {', '.join(inline)}")
    stale_fern = fern_samples - registered
    if stale_fern:
        raise ManifestError(f"Fern Code sources are not registered samples: {', '.join(sorted(stale_fern))}")
    registered_regions = {
        f"/snippets/generated/{Path(sample['path']).parent.name}/{Path(sample['path']).stem}/{region}.mdx" for sample in samples for region in sample["regions"]
    }
    stale_regions = fern_regions - registered_regions
    if stale_regions:
        raise ManifestError(f"Fern Markdown sources are not registered sample regions: {', '.join(sorted(stale_regions))}")
    for source in fern_regions:
        target = FERN_ROOT / source.lstrip("/")
        if not target.is_file():
            raise ManifestError(f"missing generated Fern sample region: {source}")


def _command(sample: dict, build_dir: Path, field: str = "command") -> list[str]:
    sample_path = DOCS_ROOT / sample["path"]
    binary = build_dir / sample["id"]
    values = {
        "python": sys.executable,
        "path": str(sample_path),
        "repo": str(REPO_ROOT),
        "build_dir": str(build_dir),
        "binary": str(binary),
    }
    return [part.format(**values) for part in sample[field]]


def run_static_fallback(sample: dict, build_dir: Path, requirement_reason: str, started: float) -> dict:
    env = os.environ.copy()
    env["CUDNN_FRONTEND_CUDART_LIB_NAME"] = "libcudart.so.13"
    env["PYTHONPATH"] = str(REPO_ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONPYCACHEPREFIX"] = str(build_dir / "pycache")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    command = _command(sample, build_dir, "fallback_command")
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=sample["timeout_seconds"],
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "id": sample["id"],
            "status": "failed",
            "reason": f"static fallback timed out after {sample['timeout_seconds']} seconds",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "seconds": round(time.monotonic() - started, 3),
        }
    status = "compiled" if result.returncode == 0 else "failed"
    reason = f"static fallback because sample {requirement_reason}" if status == "compiled" else f"static fallback exited with status {result.returncode}"
    return {
        "id": sample["id"],
        "status": status,
        "reason": reason,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "seconds": round(time.monotonic() - started, 3),
    }


def run_sample(sample: dict, build_dir: Path, *, require_execution: bool) -> dict:
    started = time.monotonic()
    met, reason = _requirements_met(sample["requirements"])
    if not met:
        if sample.get("fallback_command"):
            return run_static_fallback(sample, build_dir, reason, started)
        status = "failed" if require_execution else "skipped"
        return {"id": sample["id"], "status": status, "reason": reason, "seconds": 0.0}
    if require_execution and sample["mode"] in {"syntax", "static", "compile"}:
        return {
            "id": sample["id"],
            "status": "failed",
            "reason": "compatible sample is configured for static validation only",
            "seconds": 0.0,
        }
    env = os.environ.copy()
    env["CUDNN_FRONTEND_CUDART_LIB_NAME"] = "libcudart.so.13"
    env["PYTHONPATH"] = str(REPO_ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONPYCACHEPREFIX"] = str(build_dir / "pycache")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    command = _command(sample, build_dir)
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=sample["timeout_seconds"],
        )
        if result.returncode == 0 and sample["mode"] == "compile-run":
            execute_result = subprocess.run(
                [str(build_dir / sample["id"])],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=sample["timeout_seconds"],
            )
            result = subprocess.CompletedProcess(
                command,
                execute_result.returncode,
                result.stdout + execute_result.stdout,
                result.stderr + execute_result.stderr,
            )
        elif result.returncode == 0 and sample["mode"] == "configure-build":
            build_result = subprocess.run(
                ["cmake", "--build", str(build_dir)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=sample["timeout_seconds"],
            )
            result = subprocess.CompletedProcess(
                command,
                build_result.returncode,
                result.stdout + build_result.stdout,
                result.stderr + build_result.stderr,
            )
    except subprocess.TimeoutExpired as exc:
        return {
            "id": sample["id"],
            "status": "failed",
            "reason": f"timeout after {sample['timeout_seconds']} seconds",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "seconds": time.monotonic() - started,
        }
    output = result.stdout + result.stderr
    if result.returncode != 0:
        status, reason = "failed", f"exit status {result.returncode}"
    elif sample["expected"] != "exit:0" and sample["expected"] not in output:
        status, reason = "failed", f"expected output not found: {sample['expected']!r}"
    elif "SKIP:" in output:
        status, reason = "failed", "sample reported an undeclared skip"
    else:
        status, reason = ("executed" if sample["mode"] in {"execute", "compile-run"} else "compiled"), ""
    return {
        "id": sample["id"],
        "status": status,
        "reason": reason,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "seconds": round(time.monotonic() - started, 3),
    }


def print_result(result: dict) -> None:
    suffix = f": {result['reason']}" if result["reason"] else ""
    print(f"{result['status'].upper():8} {result['id']} ({result['seconds']:.3f}s){suffix}", flush=True)
    if result["status"] == "failed":
        if result.get("stdout"):
            print(result["stdout"].rstrip(), flush=True)
        if result.get("stderr"):
            print(result["stderr"].rstrip(), file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--language")
    parser.add_argument("--jobs", type=int, default=1, help="Run independent samples concurrently")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-execution", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    try:
        samples = load_sample_manifest(SAMPLE_ROOT / "manifest.json")
        validate_manifest(samples)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.ids:
        wanted = set(args.ids)
        unknown = wanted - {sample["id"] for sample in samples}
        if unknown:
            print(f"ERROR: unknown sample IDs: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        samples = [sample for sample in samples if sample["id"] in wanted]
    if args.language:
        samples = [sample for sample in samples if sample["language"] == args.language]
    if args.list:
        for sample in samples:
            print(f"{sample['id']}\t{sample['language']}\t{sample['mode']}\t{sample['path']}")
        return 0

    with tempfile.TemporaryDirectory(prefix="cudnn-doc-samples-") as temporary:
        build_dir = Path(temporary)
        if args.jobs == 1:
            results = []
            for sample in samples:
                result = run_sample(sample, build_dir, require_execution=args.require_execution)
                results.append(result)
                print_result(result)
        else:
            order = {sample["id"]: index for index, sample in enumerate(samples)}
            results = []
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(run_sample, sample, build_dir, require_execution=args.require_execution): sample["id"] for sample in samples}
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print_result(result)
            results.sort(key=lambda result: order[result["id"]])
    totals = {status: sum(result["status"] == status for result in results) for status in ("executed", "compiled", "skipped", "failed")}
    print("summary: " + ", ".join(f"{key}={value}" for key, value in totals.items()))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"results": results, "totals": totals}, indent=2) + "\n", encoding="utf-8")
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
