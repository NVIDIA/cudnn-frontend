# Python API index

`api_index_modules.txt` defines the checked modules, one sorted, unique dotted
module name per line. It starts with the package and 26 public API facades.
There are no wildcard or recursive entries: adding a submodule does not
automatically expand the gate. Missing/private modules and malformed lists fail
the check.

`api_index.txt` gates their intentional exports, one sorted dotted name per line.
Additions and removals require reviewing and updating the baseline in the same
PR. Importability alone does not make a name part of this contract.

```bash
python3 -S tests/analysis/api_index.py
python3 -S -m unittest discover -s tests/analysis -p test_api_index.py -v
```

After reviewing an intentional API change:

```bash
python3 -S tests/analysis/api_index.py --write
git diff -- tests/analysis/api_index_modules.txt tests/analysis/api_index.txt
```

## Enforced surface

- `cudnn.__all__`, top-level lazy export tables, and explicit top-level
  `__getattr__` entry points.
- `__all__` exports from listed modules, including modules inaccessible after
  plain `import cudnn`. Empty declarations expose only the module name.
- Listed modules without `__all__` contribute locally defined public functions
  and classes, plus explicit cuDNN re-exports. This covers `cudnn.torch` without
  counting its third-party imports or incidental variables.
- Public members of exported classes, including inherited members, properties,
  enum values, pybind declarations, and generated `pygraph` methods. Aliases of
  private implementation classes are checked through their public names.

Unlisted module helpers, dependency imports, kernel templates, generated
configuration catalogs, and unexported classes do not enter the enforced index.
A deliberate export from a listed module still counts. Implementations of
exported classes can live outside the module list; their public members remain
checked through the exported class name. Use `__all__` to declare a
supported module entry point and add its module to `api_index_modules.txt`; the scanner does not infer support from mentions
in documentation or from incidental imports.

Discovery uses AST and literal pybind declarations without importing cuDNN.
Conditional exports contribute their union, independent of optional dependency
versions and backend/platform differences. Type-checking-only and script-only
bodies are excluded. Unknown export expressions and malformed source fail the
scan. Signatures, runtime-created instance attributes, and arbitrary
metaprogramming are outside this inventory; new export mechanisms need scanner
coverage.

## Informational exposure report

```bash
python3 -S tests/analysis/api_index.py --report > /tmp/cudnn-api-exposure.txt
```

`--report` prints the broader inventory of public source declarations, including
incidental imports, implementation modules, templates, and generated GEMM
configurations. It neither compares nor writes the enforced baseline. It cannot
be combined with `--write`.

Only the broad report evaluates the pure geometry catalog in an isolated
namespace, replacing its occupancy import with constants read from source.
The enforced scan does not execute this catalog. Neither mode imports cuDNN or
requires installed optional dependencies. Module aliases are recorded as names;
their descendants appear at their defining module to avoid circular paths.

## CI

The scanner requires Python 3.10+ and its standard library. GitLab Linux and
Windows build jobs run the unittest suite before compilation. The API check
requires no CUDA toolkit, cuDNN build, driver, or GPU; the existing build jobs
retain their own dependencies.

Shared scanner files live in `tests/analysis/` so GitHub mirroring and release
overlays update them together. GitLab owns the wiring and sync regression under
its protected `ci/**` tree.
