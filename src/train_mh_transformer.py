from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from production.mh_transformer_data import build_or_load_mh_cache
from production.mh_transformer_model import MHTransformerHP, build_mh_transformer


def main() -> int:
    ap = argparse.ArgumentParser(description="Train multi-horizon Transformer (production experiment).")
    ap.add_argument("--root", default=None)
    ap.add_argument("--label", choices=["optimal", "target"], default="optimal")
    ap.add_argument("--include-global", action="store_true", default=True)
    ap.add_argument("--no-global", dest="include_global", action="store_false")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--t", type=int, default=25)
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--lead", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-out", default="models/best_mh_transformer.keras")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "TensorFlow is not installed in this environment. Activate your `tda-handover` env and rerun."
        ) from e

    arts = build_or_load_mh_cache(
        root_dir=args.root,
        k=args.k,
        t=args.t,
        h=args.h,
        lead=args.lead,
        label_mode=args.label,
        include_global=args.include_global,
        force_rebuild=args.force_rebuild,
    )

    hp = MHTransformerHP(
        k=arts.X_tr.shape[1],
        t=arts.X_tr.shape[2],
        f=arts.X_tr.shape[3],
        h=arts.y_tr.shape[1],
    )
    model = build_mh_transformer(hp)

    y_tr_oh = tf.one_hot(arts.y_tr, depth=hp.k).numpy().astype(np.float32)
    y_va_oh = tf.one_hot(arts.y_va, depth=hp.k).numpy().astype(np.float32)

    ds_tr = tf.data.Dataset.from_tensor_slices(({"cells": arts.X_tr, "mask": arts.M_tr}, y_tr_oh))
    ds_va = tf.data.Dataset.from_tensor_slices(({"cells": arts.X_va, "mask": arts.M_va}, y_va_oh))
    ds_tr = ds_tr.shuffle(len(arts.X_tr), seed=42, reshuffle_each_iteration=True).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    ds_va = ds_va.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out),
            monitor="val_top1_acc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_top1_acc",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
    ]

    model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=callbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

