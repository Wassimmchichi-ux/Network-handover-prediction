from __future__ import annotations

import argparse
import statistics
import time
import sys
from pathlib import Path

import numpy as np

# Allow running as `python src/validation/benchmark_latency.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from production.temporal_deepset_data import build_or_load_temporal_deepset_artifacts
from production.temporal_deepset_model import load_temporal_deepset


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Latency benchmark (single-sample inference).")
    ap.add_argument("--root", default=None)
    ap.add_argument("--label", choices=["optimal", "target"], default="optimal")
    ap.add_argument("--model", default="models/best_temporal_deepset.keras")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "TensorFlow is not installed. Install `tensorflow-cpu` (edge CPU) or `tensorflow` (GPU) "
            "then rerun."
        ) from e

    arts = build_or_load_temporal_deepset_artifacts(root_dir=args.root, label_mode=args.label)
    model = load_temporal_deepset(args.model)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(arts.y_te), size=min(args.samples, len(arts.y_te)), replace=False)
    X = arts.X_te[idx]
    M = arts.M_te[idx]

    # Warmup (graph + caches)
    for i in range(min(args.warmup, len(idx))):
        _ = model.predict({"cells": X[i : i + 1], "mask": M[i : i + 1]}, verbose=0)

    lat_ms: list[float] = []
    for i in range(len(idx)):
        t0 = time.perf_counter()
        _ = model.predict({"cells": X[i : i + 1], "mask": M[i : i + 1]}, verbose=0)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    print(
        "[benchmark_latency] "
        f"n={len(lat_ms)}  "
        f"p50={_percentile(lat_ms,50):.3f}ms  "
        f"p95={_percentile(lat_ms,95):.3f}ms  "
        f"p99={_percentile(lat_ms,99):.3f}ms  "
        f"mean={statistics.mean(lat_ms):.3f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
