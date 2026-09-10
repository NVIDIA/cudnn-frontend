# Python API index

`tests/analysis/api_index.txt` records the Python API names shipped by this source tree, one
sorted, unique dotted name per line. Changes to public names require an explicit
index update in the same PR. Additions and removals both fail the check.

```bash
python3 -S tests/analysis/api_index.py
python3 -S -m unittest discover -s tests/analysis -p test_api_index.py -v
```

After reviewing an intentional API change:

```bash
python3 -S tests/analysis/api_index.py --write
git diff -- tests/analysis/api_index.txt
```

The scanner requires only Python 3.10+ and its standard library. No cuDNN build,
package installation, optional framework, CUDA toolkit, driver, or GPU is needed.
GitLab's Linux and Windows build jobs run the unittest suite before compilation.
The repository-index test compares the scanned names with `api_index.txt` and
fails the build job on a mismatch. The standalone scanner lives beside the tests
in `tests/analysis/`, so mirroring and release overlays update it with the index
and tests.
GitLab's `ci/**` files are protected from release overlays.

## Coverage

- Every non-underscore module under `python/cudnn`, including namespace packages,
  modules unreachable after `import cudnn`, and modules whose imports would fail.
- Non-underscore functions, classes, assignments, imports/re-exports, and declared
  class methods/properties/fields. Imported third-party names count as exposed
  names; third-party objects' members are not recursively inventoried.
- `__all__` declarations and the package's lazy export tables/attribute hooks.
  `__all__` does not hide other public declarations. Platform/optional branches
  contribute their union; type-checking-only and script-only bodies are excluded.
- Re-exported private classes and inherited members; literal pybind class, enum,
  method, property and exception declarations exposed by the Python package.
- Generated `pygraph` methods and lazy engine factories from their source tables.
- Generated GEMM geometry configuration names. This pure CPU catalog is evaluated
  in an isolated module namespace, replacing its occupancy import with constants
  read from source. It never imports `cudnn` or initializes a device. Errors fail
  the check rather than silently removing names.

Module aliases are recorded as names; their descendants are inventoried at the
defining module, avoiding infinite paths through circular module references.
Underscore paths (including dunder names) are excluded unless a class is
re-exported through a public path. Runtime-created instance attributes, arbitrary
user monkey-patches, and signatures are outside this declaration inventory.

This is deliberately broader than the API index report's curated module list.
An entry records exposure, not an endorsement that the API should remain public.
Prefer private names for implementation details.

Source-based discovery cannot infer arbitrary Python metaprogramming. New export
mechanisms must extend the scanner and its fixtures in the same change.
Unresolved `__all__`, unsupported star imports, malformed Python, and unresolved
generated-method tables fail rather than being skipped.
