# SHAP importance performance

`calculate_shap_importance` aggregates SHAP values for feature elimination, model
interpretation, and sample similarity. It now skips standard deviation when the
variance penalty is disabled, constructs only the two public report columns, and
releases the full multiclass absolute-value array after reducing over classes.
Sorting still uses pandas' default ordering, including ties and missing values.
Input arrays are not modified, and output columns remain float64.

## Measurements

Measured on 2026-09-08 against baseline `d07a6b0` (Probatus 3.1.5), on Linux
x86-64 with an AMD EPYC 9645 CPU and eight virtual CPUs available. Both versions
used Python 3.12.3, NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.0, SHAP 0.52.0,
and joblib 1.6.0 from the same locked environment. Native thread pools were
limited to one thread, and feature elimination used `n_jobs=1`.
This was a shared virtual machine with other workloads active, so the raw timing
samples include scheduling noise.

Times below are medians of nine batches, targeting 0.3 seconds per batch
(calibrated from a warm call). Inputs are deterministic and generated outside the
timed region. Peak allocations are measured in a separate call with `tracemalloc`;
they exclude existing inputs and are not total process memory. The final optimized
run preceded the baseline run; earlier trials also measured baseline first.

| Workload | Baseline (ms) | Optimized (ms) | Speedup | Peak allocations, before → after (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 100 × 20, float64, no penalty | 0.319 | 0.194 | 1.64× | 0.048 → 0.024 |
| 20,000 × 200, float64, no penalty | 15.101 | 4.144 | 3.64× | 61.102 → 30.541 |
| 20,000 × 200, float32, no penalty | 5.743 | 2.410 | 2.38× | 30.552 → 15.280 |
| 20,000 × 200, float64, penalty 0.5 | 9.773 | 9.713 | 1.01× | 61.102 → 61.102 |
| 3 classes × 5,000 × 100, no penalty | 3.533 | 2.124 | 1.66× | 22.954 → 15.260 |
| 3 classes × 5,000 × 100, penalty 0.5 | 3.811 | 3.027 | 1.26× | 22.954 → 15.260 |
| Fitted `ShapModelInterpreter.compute()` | 14.068 | 6.895 | 2.04× | 30.594 → 15.287 |
| Complete `ShapRFECV.fit_compute()` | 360.859 | 347.482 | 1.04× | 7.720 → 7.695 |

The interpreter benchmark uses a real fitted depth-3 decision tree and SHAP
values for two 10,000 × 200 datasets. Its timing covers report computation;
model fitting and SHAP explanation are excluded. Feature elimination includes
fitting, scoring, explanation, aggregation, and reporting on 2,000 × 100 inputs,
with three folds and `step=0.5`. Both model workloads use deterministic depth-3
decision trees and `feature_perturbation="tree_path_dependent"` on both revisions.
No approximation is enabled.

Model fitting and scoring dominate the full feature-elimination benchmark. Its
timing ranges overlap (347–370 ms before, 340–365 ms after), so the 1.04× result
is not evidence of a reliable end-to-end speedup. The enabled binary penalty
likewise shows no material speed change. These results describe the measured
workloads, not every model or dataset.

The initial profile attributed roughly half the large aggregation's runtime to
the unused standard deviation. Separate trials measured the zero-penalty change
alone at 4.374 ms for the large float64 input and 0.295 ms for the small input;
removing temporary report columns then reduced the small-input trial to 0.187 ms.
Raw final samples, allocation peaks, dependency versions, and result fingerprints
are in [`scripts/benchmark_results`](https://github.com/ing-bank/probatus/tree/main/scripts/benchmark_results).
All eight baseline and optimized report fingerprints match. Regression tests
also compare exact values and ordering across float32/float64, multiclass,
contiguous/Fortran/strided layouts, duplicate labels, ties, and enabled penalties.
With a disabled penalty, very large finite SHAP values no longer get a NaN sort
key from overflowing an unused variance calculation.

## Reproduce

From a checkout containing the benchmark script:

```shell
git worktree add --detach /tmp/probatus-perf-baseline d07a6b0
uv run --locked --all-extras python scripts/benchmark_shap_importance.py \
  --source-root /tmp/probatus-perf-baseline --output /tmp/probatus-before.json \
  --repeats 9 --min-time 0.3
uv run --locked --all-extras python scripts/benchmark_shap_importance.py \
  --output /tmp/probatus-after.json --repeats 9 --min-time 0.3
```

Use the same interpreter and dependency environment for both commands. Run them
sequentially on an otherwise idle machine, then repeat in reverse order. JSON
output records each batch, its loop count, traced allocations, and a SHA-256
fingerprint of the report serialized to JSON with 15 decimal places. Compare
`result_sha256` as well as timing; the fingerprints supplement the exact-value
regression tests. Timings are intentionally not assertions in the test suite.
