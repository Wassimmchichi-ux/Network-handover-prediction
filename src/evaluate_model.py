from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from production.temporal_deepset_data import build_or_load_temporal_deepset_artifacts
from production.temporal_deepset_model import load_temporal_deepset


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a TemporalDeepSet model on the cached test split.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--label", choices=["optimal", "target"], default="optimal")
    ap.add_argument("--model", default="models/best_temporal_deepset.keras")
    ap.add_argument("--out", default="metrics/temporal_deepset_eval.json")
    args = ap.parse_args()

    arts = build_or_load_temporal_deepset_artifacts(root_dir=args.root, label_mode=args.label)

    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "TensorFlow is not installed. Install `tensorflow-cpu` (edge CPU) or `tensorflow` (GPU) "
            "then rerun."
        ) from e

    model = load_temporal_deepset(args.model)
    probs = model.predict({"cells": arts.X_te, "mask": arts.M_te}, batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1).astype(np.int32)
    top1 = float((y_pred == arts.y_te).mean())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "top1_acc": top1,
                "n_test": int(len(arts.y_te)),
                "label_mode": args.label,
                "model": args.model,
            },
            indent=2,
        )
    )
    print(f"[evaluate_model] top1_acc={top1:.4f}  n_test={len(arts.y_te):,}  → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
