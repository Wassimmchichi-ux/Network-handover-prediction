from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from production.temporal_deepset_data import build_or_load_temporal_deepset_artifacts
from production.temporal_deepset_model import TemporalDeepSetHP, build_temporal_deepset


def main() -> int:
    ap = argparse.ArgumentParser(description="Train TemporalDeepSet (production).")
    ap.add_argument("--root", default=None, help="Project root (defaults to repo root).")
    ap.add_argument("--label", choices=["optimal", "target"], default="optimal", help="Training label source.")
    ap.add_argument("--force-rebuild", action="store_true", help="Rebuild cached arrays and scaler.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-out", default="models/best_temporal_deepset.keras")
    args = ap.parse_args()

    arts = build_or_load_temporal_deepset_artifacts(
        root_dir=args.root,
        label_mode=args.label,
        force_rebuild=args.force_rebuild,
    )

    # Lazy import (TensorFlow is optional at repo level)
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "TensorFlow is not installed. Install `tensorflow-cpu` (edge CPU) or `tensorflow` (GPU) "
            "then rerun."
        ) from e

    hp = TemporalDeepSetHP(
        max_cells=arts.X_tr.shape[1],
        obs_steps=arts.X_tr.shape[2],
        n_feats=arts.X_tr.shape[3],
    )
    model = build_temporal_deepset(hp)

    y_tr_oh = tf.one_hot(arts.y_tr, depth=hp.max_cells).numpy().astype(np.float32)
    y_va_oh = tf.one_hot(arts.y_va, depth=hp.max_cells).numpy().astype(np.float32)

    ds_tr = tf.data.Dataset.from_tensor_slices(({"cells": arts.X_tr, "mask": arts.M_tr}, y_tr_oh))
    ds_va = tf.data.Dataset.from_tensor_slices(({"cells": arts.X_va, "mask": arts.M_va}, y_va_oh))
    ds_tr = ds_tr.shuffle(len(arts.y_tr), seed=42, reshuffle_each_iteration=True).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    ds_va = ds_va.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_out),
            monitor="val_top1_acc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_top1_acc",
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=callbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
