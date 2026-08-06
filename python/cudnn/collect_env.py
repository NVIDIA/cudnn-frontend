# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect environment information for cuDNN frontend bug reports.

Usage (any of):

    python -m cudnn.collect_env
    python -m cudnn.collect_env --json
    # If `import cudnn` itself fails, download and run standalone:
    curl -OL https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/python/cudnn/collect_env.py
    python collect_env.py

Design constraints (please preserve when editing):
- Only stdlib imports at module level, so the file runs standalone even when
  cudnn / torch fail to import or are absent.
- Every probe is individually guarded; a failure becomes a value in the
  report, never an exception. The report must always print.
- Strictly offline and read-only.
- Beyond versions, the report disambiguates *loaded* vs *installed* GPU
  libraries (via /proc/self/maps) and lists every on-disk copy it can find,
  because "multiple cuDNN/CUDA copies installed, the loaded one is not the
  one you think" is the most common unreproducible-issue root cause.
"""

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys

# Library families whose loaded-vs-installed identity we disambiguate.
# Keys are display names, values are regexes matched against .so basenames.
# libcudnn intentionally also matches the sublibraries (libcudnn_ops,
# libcudnn_engines_precompiled, ...) — an install with mixed sublibrary
# versions are a known inconsistent state this report must surface.
_LIB_FAMILIES = {
    "libcudnn": r"libcudnn(?:_[a-z_]+)?\.so",
    "libcublas": r"libcublas(?:Lt)?\.so",
    "libcudart": r"libcudart\.so",
    "libnvrtc": r"libnvrtc\.so",
    "libcuda (driver)": r"libcuda\.so",
}

# Distribution-name patterns for the "Relevant packages" section.
_PACKAGE_PATTERNS = [
    r"^nvidia-cudnn",  # backend wheels and this frontend
    r"^cudnn",
    r"^torch",
    r"^triton",
    r"^nvidia-",
    r"^cuda-",
    r"^transformer-engine",
    r"^flashinfer",
    r"^onnxruntime",
    r"^jax",
    r"^tensorrt",
    r"^looseversion$",
    r"^numpy$",
    r"^ninja$",
    r"^pybind11$",
]

# Packages whose declared cuDNN backend pin we annotate in the Relevant
# Packages section — host/backend version skew is a recurring source of
# "cudnn bug" reports that are really an environment mismatch.
_HOST_FRAMEWORKS = ("torch", "transformer-engine")

# Environment variable prefixes worth reporting (plus a few exact names).
_ENV_PREFIXES = (
    "CUDNN_",  # includes CUDNN_FRONTEND_*
    "CUDA_",
    "NVIDIA_",
    "TORCH_",
    "PYTORCH_",
    "NVTE_",
)
_ENV_EXACT = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
)


def _run(cmd, timeout=30):
    """Run a shell command, return stdout on success else None."""
    try:
        out = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if out.returncode != 0:
            return None
        # Strip ANSI escapes (e.g. nvidia-smi topo underlines) for clean paste.
        return re.sub(r"\x1b\[[0-9;]*m", "", out.stdout).strip()
    except Exception:
        return None


def _guard(fn, default="<probe failed>"):
    try:
        return fn()
    except Exception as e:
        return f"{default}: {type(e).__name__}: {e}"


def _installed_distributions():
    """{normalized_name: version} for every installed distribution."""
    import importlib.metadata

    dists = {}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if name:
            dists[name.lower()] = dist.version
    return dists


def _fe_backend_search():
    """Replicate the search order of cudnn/__init__.py:_dlopen_cudnn()
    read-only: LD_LIBRARY_PATH first, then site-packages nvidia/cudnn[_jit],
    then the dynamic-linker soname fallback. Shows which libcudnn the
    frontend picks *before* any torch import gets a say."""
    if sys.platform.startswith("win"):
        return "<not traced on Windows>"
    import sysconfig

    for library_dir in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not library_dir:
            continue
        for library_name in ("libcudnn.so.9", "libcudnn.so"):
            library_path = os.path.join(library_dir, library_name)
            if os.path.exists(library_path):
                return f"{os.path.realpath(library_path)} (via LD_LIBRARY_PATH)"
    for sub in ("nvidia/cudnn/lib", "nvidia/cudnn_jit/lib"):
        hits = glob.glob(os.path.join(sysconfig.get_path("purelib"), sub, "libcudnn.so.*[0-9]"))
        if hits:
            return f"{os.path.realpath(hits[0])} (via site-packages {sub})"
    return "dynamic-linker soname fallback (ldconfig / RPATH)"


def _get_cudnn_frontend_info():
    info = {}
    dists = _guard(_installed_distributions, None)
    try:
        import cudnn  # dlopens the cuDNN backend on import
    except Exception as e:
        info["cudnn-frontend"] = f"<import failed: {type(e).__name__}: {e}>"
        if isinstance(dists, dict):
            info["nvidia-cudnn-frontend (pip)"] = dists.get("nvidia-cudnn-frontend", "not installed")
        info["backend search order would pick"] = _guard(_fe_backend_search)
        return info

    fe_version = _guard(lambda: cudnn.__version__, "?")
    info["cudnn-frontend"] = fe_version
    if isinstance(dists, dict):
        pip_version = dists.get("nvidia-cudnn-frontend")
        if pip_version is None:
            info["nvidia-cudnn-frontend (pip)"] = "not installed (source checkout on sys.path?)"
        elif pip_version != fe_version:
            # Stale build, or a source checkout shadowing the installed wheel.
            info["nvidia-cudnn-frontend (pip)"] = f"{pip_version}  ⚠ MISMATCH vs imported cudnn.__version__ == {fe_version}"
        else:
            info["nvidia-cudnn-frontend (pip)"] = pip_version
    info["cudnn-frontend file"] = _guard(lambda: cudnn.__file__, "?")
    info["compiled module"] = _guard(lambda: cudnn._pybind_module.__file__, "?")
    info["cudnn backend (loaded)"] = _guard(lambda: f"{cudnn.backend_version_string()} ({cudnn.backend_version()})", "?")
    info["backend search order picks"] = _guard(_fe_backend_search)
    return info


def _get_platform_info():
    info = {}
    info["Python"] = sys.version.replace("\n", " ")
    info["Python executable"] = sys.executable
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        info["Virtual env"] = sys.prefix
    info["Platform"] = platform.platform()
    libc = platform.libc_ver()
    info["libc"] = " ".join(v for v in libc if v) or "n/a"
    os_release = _run("grep PRETTY_NAME /etc/os-release")
    if os_release:
        info["OS"] = os_release.split("=", 1)[-1].strip('"')
    in_container = os.path.exists("/.dockerenv") or bool(_run("grep -sq -e docker -e containerd -e kubepods /proc/1/cgroup && echo 1"))
    info["Container"] = "yes" if in_container else "no / not detected"
    return info


def _get_gpu_info():
    """Per-GPU properties. torch is authoritative for enumeration order
    (CUDA order != nvidia-smi order); nvidia-smi is the no-torch fallback and
    supplies the driver version."""
    info = {}
    smi = _run("nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader")
    driver = None
    if smi:
        driver = smi.splitlines()[0].rsplit(",", 1)[-1].strip()
    info["Driver version"] = driver or "<nvidia-smi unavailable>"
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")

    def _smi_fallback():
        for line in smi.splitlines():
            idx, rest = line.split(",", 1)
            info[f"GPU {idx.strip()} (nvidia-smi order)"] = rest.strip()

    try:
        import torch

        if not torch.cuda.is_available():
            info["GPUs (torch)"] = "torch.cuda.is_available() == False"
            if smi:
                _smi_fallback()
            return info
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info[f"GPU {i} (CUDA order)"] = f"{p.name} | SM{p.major}{p.minor} | {p.multi_processor_count} SMs | {p.total_memory / (1 << 30):.1f} GiB"
    except Exception as e:
        info["GPUs (torch)"] = f"<torch probe failed: {type(e).__name__}: {e}>"
        if smi:
            _smi_fallback()
    return info


def _get_cuda_toolkit_info():
    info = {}
    import shutil

    nvcc_on_path = shutil.which("nvcc")
    info["nvcc on PATH"] = nvcc_on_path or "not found"

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    info["CUDA_HOME (env)"] = cuda_home or "<unset>"

    nvcc = nvcc_on_path
    if not nvcc and cuda_home:
        cand = os.path.join(cuda_home, "bin", "nvcc")
        nvcc = cand if os.path.isfile(cand) else None
    if nvcc:
        out = _run([nvcc, "-V"])
        if out:
            m = re.search(r"release [\d.]+, V[\d.]+", out)
            info["nvcc version"] = m.group(0) if m else out.splitlines()[-1]
    return info


def _get_torch_info():
    info = {}
    try:
        import torch
    except Exception as e:
        info["torch"] = f"<import failed: {type(e).__name__}: {e}>"
        return info
    info["torch"] = torch.__version__
    info["torch.version.cuda"] = str(torch.version.cuda)
    info["torch file"] = _guard(lambda: torch.__file__, "?")

    torch_cudnn = _guard(lambda: torch.backends.cudnn.version(), None)
    line = str(torch_cudnn)
    if isinstance(torch_cudnn, int):

        def _fe_backend():
            import cudnn

            return cudnn.backend_version()

        fe_backend = _guard(_fe_backend, None)
        if isinstance(fe_backend, int) and fe_backend != torch_cudnn:
            # Same process, two different libcudnn copies: torch resolved one
            # via its own linkage, the frontend dlopen'ed another.
            line += f"  ⚠ differs from cudnn-frontend's loaded backend ({fe_backend}) — two libcudnn copies in this process"
    info["torch.backends.cudnn.version()"] = line
    return info


def _check_pin(req_line, dists):
    """Evaluate one Requires-Dist line against installed versions.
    Returns (satisfied, installed_version) or None if not evaluable
    (packaging unavailable, target not installed, no specifier)."""
    try:
        from packaging.requirements import Requirement

        req = Requirement(req_line.split(";")[0].strip())
        target = re.sub(r"[-_.]+", "-", req.name).lower()
        installed = dists.get(target)
        if installed is None or not req.specifier:
            return None
        return bool(req.specifier.contains(installed, prereleases=True)), installed
    except Exception:
        return None


def _framework_pin_note(name, dists):
    """For an installed host framework, render its declared cuDNN backend pin
    vs what is installed — stated factually; overriding a pin is often
    intentional and judging it is not this tool's job."""
    import importlib.metadata

    reqs = _guard(lambda: importlib.metadata.requires(name), None) or []
    rendered = []
    for pin in [r for r in reqs if "cudnn" in r.lower()]:
        verdict = _check_pin(pin, dists)
        suffix = ""
        if verdict is not None:
            ok, installed = verdict
            if not ok:
                suffix = f"; installed {installed}"
        rendered.append(pin + suffix)
    return f"  (declares {'; '.join(rendered)})" if rendered else ""


def _force_load_gpu_libs():
    """Trigger lazy loading of the GPU libraries torch actually uses, so that
    /proc/self/maps reflects reality. Every step is optional."""

    def _f():
        import torch

        torch.backends.cudnn.version()  # loads torch's libcudnn
        if torch.cuda.is_available():
            x = torch.randn(8, 8, device="cuda", dtype=torch.float16)
            torch.mm(x, x)  # loads libcublas/Lt
            y = torch.randn(1, 1, 8, 8, device="cuda")
            w = torch.randn(1, 1, 3, 3, device="cuda")
            torch.nn.functional.conv2d(y, w)  # exercises cudnn
            torch.cuda.synchronize()

    _guard(_f, None)


def _loaded_gpu_libs():
    """{family: set of realpaths} of GPU libraries mapped into this process."""
    loaded = {name: set() for name in _LIB_FAMILIES}
    try:
        with open("/proc/self/maps") as f:
            maps = f.read()
    except OSError:
        return loaded
    for path in set(re.findall(r"\S*/lib\S+\.so\S*", maps)):
        base = os.path.basename(path)
        for family, pat in _LIB_FAMILIES.items():
            if re.match(pat, base):
                loaded[family].add(os.path.realpath(path))
    return loaded


def _candidate_lib_dirs():
    """Directories where conflicting copies of GPU libraries typically hide."""
    dirs = []
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if d:
            dirs.append(d)
    try:
        import site
        import sysconfig

        sp = set(site.getsitepackages() + [site.getusersitepackages()])
        sp.add(sysconfig.get_paths()["purelib"])
        for p in sp:
            dirs.extend(glob.glob(os.path.join(p, "nvidia", "*", "lib")))
            dirs.append(os.path.join(p, "torch", "lib"))
    except Exception:
        pass
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        dirs.append(os.path.join(cuda_home, "lib64"))
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


# For pip provenance of a directory holding a family's .so files. The plain
# names (no -cuXX suffix) are the CUDA 13 mega-wheel packages installed under
# site-packages/nvidia/cu13/lib.
_FAMILY_PIP_NAMES = {
    "libcudnn": ["cudnn"],
    "libcublas": ["cublas"],
    "libcudart": ["cuda-runtime"],
    "libnvrtc": ["cuda-nvrtc"],
    "libcuda (driver)": [],
}


def _describe_lib_dir(family, dirpath, paths, dists):
    """One line of provenance for a directory holding this family's .so files:
    filename-embedded version and/or the owning pip package."""
    tags = []
    versions = set()
    for p in paths:
        m = re.search(r"\.so\.(\d+(?:\.\d+)+)$", p)
        if m:
            versions.add(m.group(1))
    if versions:
        tags.append("v" + " / v".join(sorted(versions)))
    if len(versions) > 1:
        # e.g. libcudnn sublibraries from two different backend releases mixed
        # into one directory by an interrupted pip upgrade or manual copy.
        tags.append("⚠ mixed versions in one directory")
    if isinstance(dists, dict) and "/site-packages/" in dirpath:
        for base in _FAMILY_PIP_NAMES.get(family, []):
            for pkg in (
                f"nvidia-{base}-cu13",
                f"nvidia-{base}-cu12",
                f"nvidia-{base}-cu11",
                f"nvidia-{base}",
            ):
                if pkg in dists:
                    tags.append(f"pip {pkg}=={dists[pkg]}")
                    break
            else:
                continue
            break
    return f"{dirpath}/" + (f"  [{', '.join(tags)}]" if tags else "")


def _get_gpu_library_conflicts():
    """The headline section: for each library family, list where the copies
    actually loaded into this process live and every other install visible on
    disk. Aggregated by directory — the .so files of one install (e.g. the
    libcudnn_* sublibraries) share a directory, so distinct directories are
    the unit of "conflicting copies"."""
    _force_load_gpu_libs()
    _guard(_get_cudnn_frontend_info, None)  # frontend import also dlopens libcudnn
    loaded = _loaded_gpu_libs()
    dists = _guard(_installed_distributions, None)

    on_disk = {name: set() for name in _LIB_FAMILIES}
    for d in _candidate_lib_dirs():
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for base in entries:
            for family, pat in _LIB_FAMILIES.items():
                if re.match(pat, base):
                    on_disk[family].add(os.path.realpath(os.path.join(d, base)))
    ldconfig = _run("ldconfig -p") or ""
    for line in ldconfig.splitlines():
        base = line.strip().split(" ", 1)[0]
        path = line.rsplit("=> ", 1)[-1].strip() if "=> " in line else None
        if not path:
            continue
        for family, pat in _LIB_FAMILIES.items():
            if re.match(pat, base):
                on_disk[family].add(os.path.realpath(path))

    info = {}
    for family in _LIB_FAMILIES:
        by_dir = {}
        for p in loaded[family] | on_disk[family]:
            by_dir.setdefault(os.path.dirname(p), set()).add(p)
        if not by_dir:
            continue
        loaded_dirs = {os.path.dirname(p) for p in loaded[family]}
        lines = []
        for d in sorted(by_dir, key=lambda d: (d not in loaded_dirs, d)):
            state = "LOADED " if d in loaded_dirs else "on disk"
            lines.append(f"{state}  {_describe_lib_dir(family, d, by_dir[d], dists)}")
        header = family
        if len(loaded_dirs) > 1:
            header += "  ⚠ LOADED FROM MULTIPLE DIRECTORIES (likely conflict)"
        elif loaded_dirs and len(by_dir) > 1:
            header += "  ⚠ other installs on disk (check which one you expect)"
        elif len(by_dir) > 1:
            header += "  ⚠ multiple installs on disk"
        info[header] = "\n" + "\n".join(f"    {line}" for line in lines)
    if not info:
        info["note"] = "no GPU libraries loaded or found (torch missing / CPU-only?)"
    return info


def _get_relevant_packages():
    dists = _guard(_installed_distributions, None)
    if not isinstance(dists, dict):
        return {"packages": str(dists)}
    pats = [re.compile(p) for p in _PACKAGE_PATTERNS]
    pkgs = {name: ver for name, ver in sorted(dists.items()) if any(p.search(name) for p in pats)}
    for name in _HOST_FRAMEWORKS:
        if name in pkgs:
            pkgs[name] += _framework_pin_note(name, dists)
    return pkgs


def _get_env_vars():
    info = {}
    for k in sorted(os.environ):
        if k.startswith(_ENV_PREFIXES) or k in _ENV_EXACT:
            info[k] = os.environ[k]
    return info or {"(none set)": ""}


def _get_topology():
    topo = _run("nvidia-smi topo -m", timeout=60)
    if not topo:
        return {}
    # Drop the static legend boilerplate; keep just the matrix.
    return {"nvidia-smi topo -m": "\n" + topo.split("\n\nLegend:")[0].rstrip()}


def collect_env_info():
    """Collect everything into an ordered {section: {key: value}} dict."""
    sections = [
        ("cuDNN Frontend", _get_cudnn_frontend_info),
        ("Python / Platform", _get_platform_info),
        ("GPU / Driver", _get_gpu_info),
        ("CUDA Toolkit", _get_cuda_toolkit_info),
        ("PyTorch", _get_torch_info),
        ("GPU Libraries: loaded vs on disk", _get_gpu_library_conflicts),
        ("Relevant Packages", _get_relevant_packages),
        ("Environment Variables", _get_env_vars),
        ("GPU Topology", _get_topology),
    ]
    report = {}
    for title, fn in sections:
        result = _guard(fn, None)
        report[title] = result if isinstance(result, dict) else {"error": str(result)}
    return report


def format_report(report):
    lines = [
        "### cuDNN frontend environment report",
        "<!-- generated by `python -m cudnn.collect_env`; paste into your issue -->",
    ]
    for title, entries in report.items():
        if not entries:
            continue
        lines.append("")
        lines.append(f"==== {title} ====")
        width = max((len(k) for k in entries), default=0)
        for k, v in entries.items():
            v = str(v)
            if v.startswith("\n"):
                lines.append(f"{k}:{v}")
            else:
                lines.append(f"{k:<{width}} : {v}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect environment information for cuDNN frontend bug reports.")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    report = collect_env_info()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))


if __name__ == "__main__":
    # When invoked by path from a source checkout (python python/cudnn/collect_env.py),
    # sys.path[0] is this file's directory — the cudnn package dir — so its
    # modules (graph.py, datatypes.py, ...) would shadow same-named top-level
    # imports inside the probes. Probe with a clean path; nothing is imported
    # from this script's own directory.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]
    main()
