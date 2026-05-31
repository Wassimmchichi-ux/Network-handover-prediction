from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from production.mh_transformer_data import build_or_load_mh_cache


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate multi-horizon Transformer (per horizon metrics).")
    ap.add_argument("--root", default=None)
    ap.add_argument("--label", choices=["optimal", "target"], default="optimal")
    ap.add_argument("--model", default="models/best_mh_transformer.keras")
    ap.add_argument("--out", default="metrics/mh_transformer_eval.json")
    args = ap.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise SystemExit("TensorFlow is not installed. Activate `tda-handover` and rerun.") from e

    arts = build_or_load_mh_cache(root_dir=args.root, label_mode=args.label)
    model = tf.keras.models.load_model(args.model, compile=False)

    probs = model.predict({"cells": arts.X_te, "mask": arts.M_te}, batch_size=256, verbose=0)  # (N, H, K)
    y = arts.y_te
    H = y.shape[1]

    out = {
        "model": args.model,
        "label_mode": args.label,
        "n_test": int(len(y)),
        "horizons": [],
    }
    for h in range(H):
        top1 = float((probs[:, h, :].argmax(axis=1).astype(np.int32) == y[:, h]).mean())
        out["horizons"].append({"h": int(h + 1), "top1_acc": top1})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[evaluate_mh_transformer] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

