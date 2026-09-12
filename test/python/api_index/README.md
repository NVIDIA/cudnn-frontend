# Python API index

`api_index_modules.txt` lists the checked modules as sorted, unique dotted names.
The scanner compares their public `__all__` exports and public members of
exported cuDNN classes against `api_index.txt`. Import failures, invalid
`__all__` declarations, unresolved exports, and API additions or removals fail
the check.

## Run

From the repository root, install a wheel built from the same source revision.
The environment needs the CUDA/cuDNN libraries and Python dependencies required
by the listed modules. No GPU is needed.

```bash
python3 -m pip install --no-deps --target api_index_wheel build/wheel/*.whl
CUDA_VISIBLE_DEVICES='' CUDNN_API_INDEX_PACKAGE_ROOT="$PWD/api_index_wheel/cudnn" \
  python3 -m unittest discover -s test/python/api_index -p test_api_index.py -v
```

`CUDNN_API_INDEX_PACKAGE_ROOT` selects the package to inspect. Without it, only
fixture tests run and the runtime baseline check is skipped.

## Update the baseline

For an intentional API change, regenerate the baseline and review the diff:

```bash
CUDA_VISIBLE_DEVICES='' python3 test/python/api_index/api_index.py \
  --package-root api_index_wheel/cudnn --write
git diff -- test/python/api_index/api_index.txt
```

Edit `api_index_modules.txt` separately to change which modules are checked.
