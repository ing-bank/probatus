"""Measure SHAP aggregation and feature elimination against a chosen source checkout."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path


def measure(run, repeats, min_time):
    """Warm up, calibrate batches, then measure time and allocation peaks separately."""
    result = run()
    started = time.perf_counter()
    run()
    elapsed = time.perf_counter() - started
    loops = max(1, min(1000, int(min_time / elapsed)))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(loops):
            run()
        samples.append((time.perf_counter() - started) / loops)

    tracemalloc.start()
    run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "loops_per_sample": loops,
        "peak_allocated_bytes": peak,
        "result_sha256": hashlib.sha256(result.to_json(double_precision=15).encode()).hexdigest(),
    }


def main():
    """Use identical deterministic inputs and the same interpreter for both checkouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-time", type=float, default=0.2, help="Target seconds per timing batch.")
    args = parser.parse_args()
    if args.repeats < 1 or args.min_time <= 0:
        parser.error("repeats and min-time must be positive")
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))

    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from threadpoolctl import threadpool_limits

    import probatus
    from probatus.feature_elimination import ShapRFECV
    from probatus.interpret import ShapModelInterpreter
    from probatus.utils import calculate_shap_importance

    assert Path(probatus.__file__).resolve().parent.parent == source_root
    rng = np.random.default_rng(42)
    results = {
        "source_root": str(source_root),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip(),
        "tracked_diff": subprocess.check_output(["git", "diff", "HEAD", "--", "probatus"], cwd=source_root, text=True),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "versions": {
            name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "shap", "joblib"]
        },
        "repeats": args.repeats,
        "min_time": args.min_time,
        "workloads": {},
    }

    def record(name, run):
        measurement = measure(run, args.repeats, args.min_time)
        results["workloads"][name] = measurement
        print(f"{name}: {measurement['median_seconds'] * 1000:.3f} ms", flush=True)

    with threadpool_limits(limits=1):
        for name, shape, dtype, penalty in [
            ("small", (100, 20), "float64", None),
            ("large", (20000, 200), "float64", None),
            ("large_float32", (20000, 200), "float32", None),
            ("large_penalized", (20000, 200), "float64", 0.5),
            ("multiclass", (3, 5000, 100), "float64", None),
            ("multiclass_penalized", (3, 5000, 100), "float64", 0.5),
        ]:
            values = rng.normal(size=shape).astype(dtype)
            columns = [f"feature_{i}" for i in range(shape[-1])]
            record(name, lambda: calculate_shap_importance(values, columns, shap_variance_penalty_factor=penalty))

        # Fit a real model and explainer outside the compute-only timing.
        X = pd.DataFrame(rng.normal(size=(20000, 200)))
        y = pd.Series((X[0] + X[1] > 0).astype(int))
        model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X.iloc[:10000], y.iloc[:10000])
        interpreter = ShapModelInterpreter(model, random_state=42)
        interpreter.fit(
            X.iloc[:10000],
            X.iloc[10000:],
            y.iloc[:10000],
            y.iloc[10000:],
            feature_perturbation="tree_path_dependent",
        )
        record("interpreter_compute", interpreter.compute)

        # Include model fitting, scoring, SHAP calculation and reporting in this timing.
        X = pd.DataFrame(rng.normal(size=(2000, 100)))
        y = pd.Series((X[0] + X[1] > 0).astype(int))

        def elimination():
            return ShapRFECV(
                DecisionTreeClassifier(max_depth=3, random_state=42),
                step=0.5,
                cv=3,
                n_jobs=1,
                random_state=42,
            ).fit_compute(X, y, feature_perturbation="tree_path_dependent")

        record("rfecv_fit_compute", elimination)

    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
