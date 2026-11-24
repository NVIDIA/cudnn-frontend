# Reinforcement learning tests

Run the suite from `test/python` on a supported GPU. Install a FlashInfer build
with cuDNN RL support to include the rollout integration tests.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/rl/goldens" \
    python -m pytest -n0 -q -rs rl/
```

- [Plan selection](test_rl_dispatch.py) covers the RL numerical note, engine discovery, replay and unsupported graphs.
- [BF16 matmul](test_rl_matmul_determinism.py) covers batch invariance, pipeline stress, known output bytes and RL forward paired with native cuDNN matmul and reduction plans for the gradients.
- [FlashInfer integration](test_flashinfer_integration.py) compares a rollout row bitwise with the corresponding training row and the original FlashInfer implementation.

The [goldens](goldens/) are grouped by the test runtime hash. Each JSON file records its GPU and software versions.
