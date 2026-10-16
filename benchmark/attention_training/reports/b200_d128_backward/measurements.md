# Recorded backward measurements

Generated from [measurements.json](measurements.json) by [plot.py](plot.py). All values include the complete backward chain.

## Dense MHA, paced duty 0.5

| B | S | Mask | Baseline | Baseline ms | Paired FROST ms | FROST speed-up | SOL % (MMA) |
|---:|---:|---|---|---:|---:|---:|---|
| 2 | 2048 | causal | cuDNN BE | 0.054672 | 0.052718 | +3.71% | unavailable |
| 2 | 2048 | none | cuDNN BE | 0.091157 | 0.089416 | +1.95% | unavailable |
| 2 | 2048 | causal | FA4 | 0.076423 | 0.052180 | +46.46% | unavailable |
| 2 | 2048 | none | FA4 | 0.089811 | 0.089850 | -0.04% | unavailable |
| 2 | 4096 | causal | cuDNN BE | 0.163251 | 0.160757 | +1.55% | unavailable |
| 2 | 4096 | none | cuDNN BE | 0.312260 | 0.310540 | +0.55% | unavailable |
| 2 | 4096 | causal | FA4 | 0.201341 | 0.160800 | +25.21% | unavailable |
| 2 | 4096 | none | FA4 | 0.303526 | 0.310240 | -2.16% | unavailable |
| 2 | 8192 | causal | cuDNN BE | 0.537144 | 0.535144 | +0.37% | unavailable |
| 2 | 8192 | none | cuDNN BE | 1.010424 | 1.006976 | +0.34% | unavailable |
| 2 | 8192 | causal | FA4 | 0.584536 | 0.535376 | +9.18% | unavailable |
| 2 | 8192 | none | FA4 | 0.945768 | 0.995936 | -5.04% | unavailable |
| 2 | 16384 | causal | cuDNN BE | 1.907456 | 1.909504 | -0.11% | unavailable |
| 2 | 16384 | none | cuDNN BE | 3.730872 | 3.732752 | -0.05% | unavailable |
| 2 | 16384 | causal | FA4 | 1.913280 | 1.910592 | +0.14% | unavailable |
| 2 | 16384 | none | FA4 | 3.477968 | 3.737312 | -6.94% | unavailable |

## CP8 THD/GQA, paced duty 0.2

| B | S | Mask | Baseline | Baseline ms | Paired FROST ms | FROST speed-up | SOL % (MMA) |
|---:|---:|---|---|---:|---:|---:|---|
| 256 | 2048 | causal | cuDNN BE | 3.819296 | 3.150432 | +21.23% | unavailable |
| 256 | 2048 | causal | FA4 | 4.019080 | 3.151968 | +27.51% | unavailable |
| 128 | 4096 | causal | cuDNN BE | 5.600128 | 4.887088 | +14.59% | unavailable |
| 128 | 4096 | causal | FA4 | 5.324040 | 4.862128 | +9.50% | unavailable |
| 64 | 8192 | causal | cuDNN BE | 9.380952 | 8.238864 | +13.86% | unavailable |
| 64 | 8192 | causal | FA4 | 8.405552 | 8.239472 | +2.02% | unavailable |

## CP2 THD/GQA, sustained 1200 W

| B | S | Mask | Baseline | Baseline ms | Paired FROST ms | FROST speed-up | SOL % (MMA) |
|---:|---:|---|---|---:|---:|---:|---|
| 64 | 8192 | causal | FA4 | 38.413812 | 40.079000 | -4.03% | unavailable |

Dense/CP8 speed-up is the ratio of summarized latencies. CP2 uses the median of six paired baseline/FROST ratios; dividing its displayed latency medians is a different estimator. See the [methodology](README.md#measurement-method-and-limits).
