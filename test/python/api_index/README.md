# Python API index

`api_index_modules.txt` defines the checked modules: sorted, unique dotted names,
with no wildcard or recursive entries. `api_index.txt` records their runtime
exports. Additions and removals require a reviewed baseline update.

The scanner imports only the listed modules, resolves every public name in each
module's `__all__`, and inspects public members of exported cuDNN classes.
Inherited members, properties, enums, native bindings, and generated methods are
inspected directly. Properties are not evaluated. Private names are excluded.
Module aliases and external classes are recorded without recursively expanding
them. Namespace instances and unlisted lazy exports are not traversed; expose
supported names through a listed module's `__all__`.

Each listed module must declare `__all__`. Missing modules, import failures,
invalid declarations, and unresolved exports fail the scan, including `--write`.
Unlisted modules are loaded only when a listed export needs them. There is no
AST parser, kernel-template discovery, or broad exposure report.

## Run

Use the wheel artifact from `build:dev:linux:amd64`, in the same Linux
development CI image with its CUDA/cuDNN libraries and Python dependencies.
The inspection itself needs no GPU.

```bash
python3 -m pip install --no-deps --target api_index_wheel build/wheel/*.whl
CUDA_VISIBLE_DEVICES='' python3 test/python/api_index/api_index.py --package-root api_index_wheel/cudnn
CUDA_VISIBLE_DEVICES='' CUDNN_API_INDEX_PACKAGE_ROOT="$PWD/api_index_wheel/cudnn" \
  python3 -m unittest discover -s test/python/api_index -p test_api_index.py -v
```

The scanner defaults to `build/cudnn` for local CMake builds; `--package-root`
selects the installed wheel above.
It checks that the selected package is used, even with an editable cuDNN install
present. Run it in a fresh Python process.

After reviewing an intentional API change, in the same environment:

```bash
CUDA_VISIBLE_DEVICES='' python3 test/python/api_index/api_index.py --package-root api_index_wheel/cudnn --write
git diff -- test/python/api_index/api_index_modules.txt test/python/api_index/api_index.txt
```

`--modules` and `--index` select alternative files. Regeneration replaces only
the API baseline, not the module allowlist.

## CI and tests

`analysis:api_index` downloads the wheel from `build:dev:linux:amd64` and runs
the checks on a CPU runner with GPU visibility disabled. The analysis stage
follows build, and the job explicitly needs that build's wheel artifact. The
development image supplies PyTorch and CuTeDSL; the job installs JAX for the
listed JAX facade. Release and Windows builds can expose different native names
and do not define this baseline.

Use `unittest` for CPU execution; the parent `test/python` pytest configuration
requires a GPU. Fixture tests also run in the analysis job and can be run locally
using only Python 3.10+ and the standard library:

```bash
python3 -S -m unittest discover -s test/python/api_index -p test_api_index.py -v
```

The repository test is enabled by `CUDNN_API_INDEX_PACKAGE_ROOT`; without it,
only fixture tests run. Shared files live in `test/python/api_index/`, outside the
protected `ci/**` tree, so mirroring and release overlays update them together.
