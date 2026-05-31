from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as `python src/validation/robustness_tests.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from production.temporal_deepset_data import build_or_load_temporal_deepset_artifacts
from production.temporal_deepset_model import load_temporal_deepset


def _eval_top1(model, X: np.ndarray, M: np.ndarray, y: np.ndarray, *, batch_size: int = 256) -> float:
    probs = model.predict({"cells": X, "mask": M}, batch_size=batch_size, verbose=0)
    return float((probs.argmax(axis=1).astype(np.int32) == y).mean())


def _add_noise(X: np.ndarray, M: np.ndarray, *, rsrp_db_std: float, sinr_db_std: float, load_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    valid = (M == 1.0)
    if valid.sum() == 0:
        return Xn
    # Feature layout: [rsrp, sinr, load]
    noise = np.zeros_like(Xn[valid], dtype=np.float32)
    noise[:, :, 0] = rng.normal(0.0, rsrp_db_std, size=noise[:, :, 0].shape)
    noise[:, :, 1] = rng.normal(0.0, sinr_db_std, size=noise[:, :, 1].shape)
    noise[:, :, 2] = rng.normal(0.0, load_std, size=noise[:, :, 2].shape)
    Xn[valid] = (Xn[valid] + noise).astype(np.float32)
    return Xn


def _drop_neighbors(X: np.ndarray, M: np.ndarray, *, drop_prob: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xd = X.copy()
    Md = M.copy()
    keep = rng.random(size=Md.shape) >= drop_prob
    Md = (Md * keep.astype(np.float32)).astype(np.float32)
    # zero-out dropped cells so masks remain consistent with training convention
    dropped = (keep == 0) & (M == 1.0)
    if dropped.any():
        Xd[dropped] = 0.0
    return Xd, Md


def main() -> int:
    ap = argparse.ArgumentParser(description="Robustness tests: noise + missing neighbors.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--label", choices=["optimal", "target"], default="optimal")
    ap.add_argument("--model", default="models/best_temporal_deepset.keras")
    ap.add_argument("--out", default="metrics/robustness_report.json")
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

    base = _eval_top1(model, arts.X_te, arts.M_te, arts.y_te)

    # Noise sweeps (rough, tune per UE measurement spec)
    noise_1 = _eval_top1(
        model,
        _add_noise(arts.X_te, arts.M_te, rsrp_db_std=1.0, sinr_db_std=1.0, load_std=0.01, seed=1),
        arts.M_te,
        arts.y_te,
    )
    noise_3 = _eval_top1(
        model,
        _add_noise(arts.X_te, arts.M_te, rsrp_db_std=3.0, sinr_db_std=2.0, load_std=0.03, seed=2),
        arts.M_te,
        arts.y_te,
    )

    # Missing neighbor sweeps
    Xd10, Md10 = _drop_neighbors(arts.X_te, arts.M_te, drop_prob=0.10, seed=3)
    Xd30, Md30 = _drop_neighbors(arts.X_te, arts.M_te, drop_prob=0.30, seed=4)
    drop10 = _eval_top1(model, Xd10, Md10, arts.y_te)
    drop30 = _eval_top1(model, Xd30, Md30, arts.y_te)

    report = {
        "label_mode": args.label,
        "model": args.model,
        "n_test": int(len(arts.y_te)),
        "top1_base": base,
        "top1_noise_rsrp1_sinr1": noise_1,
        "top1_noise_rsrp3_sinr2": noise_3,
        "top1_drop10": drop10,
        "top1_drop30": drop30,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[robustness_tests] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
