from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from production.temporal_deepset_decision import DecisionConfig, TemporalDecisionGate
from production.temporal_deepset_model import load_temporal_deepset


_BRACKETS = re.compile(r"[\[\]]")


def _parse_float_list(s, k: int) -> np.ndarray:
    if pd.isna(s):
        return np.zeros(k, dtype=np.float32)
    cleaned = _BRACKETS.sub("", str(s)).strip()
    if not cleaned:
        return np.zeros(k, dtype=np.float32)
    parts = re.split(r"[,;]", cleaned)
    out = []
    for p in parts[:k]:
        try:
            out.append(float(p.strip()))
        except ValueError:
            out.append(0.0)
    out += [0.0] * (k - len(out))
    return np.asarray(out, dtype=np.float32)


def _parse_int_list(s, k: int) -> list[int]:
    if pd.isna(s):
        return [0] * k
    cleaned = _BRACKETS.sub("", str(s)).strip()
    if not cleaned:
        return [0] * k
    parts = re.split(r"[,;]", cleaned)
    out = []
    for p in parts[:k]:
        try:
            out.append(int(float(p.strip())))
        except (ValueError, TypeError):
            out.append(0)
    out += [0] * (k - len(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline inference + decision gating for TemporalDeepSet.")
    ap.add_argument("--csv", default="dataset/raw/handover_dataset.csv")
    ap.add_argument("--model", default="models/best_temporal_deepset.keras")
    ap.add_argument("--scaler", default="dataset/temporal_deepset_cache/scaler.pkl")
    ap.add_argument("--ue-id", default=None, help="Filter to a single UE ID.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--t", type=int, default=25)
    ap.add_argument("--ttt", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--cooldown", type=int, default=5)
    args = ap.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "TensorFlow is not installed. Install `tensorflow-cpu` (edge CPU) or `tensorflow` (GPU) "
            "then rerun."
        ) from e

    model = load_temporal_deepset(args.model)
    scaler = pickle.loads(Path(args.scaler).read_bytes())

    df = pd.read_csv(args.csv, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    if args.ue_id is not None:
        df = df[df["ue_id"].astype(str) == str(args.ue_id)]
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    gate = TemporalDecisionGate(DecisionConfig(ttt_steps=args.ttt, margin_threshold=args.margin, cooldown_steps=args.cooldown))

    # Rolling window per UE
    buffers: dict[str, list[np.ndarray]] = {}

    for _, row in df.iterrows():
        ue = str(row["ue_id"])
        nb_ids = _parse_int_list(row["nb_cell_ids"], args.k)
        nb_r = _parse_float_list(row["nb_rsrps"], args.k)
        nb_s = _parse_float_list(row["nb_sinrs"], args.k)
        nb_l = _parse_float_list(row["nb_loads"], args.k)

        feats = np.stack([nb_r, nb_s, nb_l], axis=1).astype(np.float32)  # (K,3)
        buffers.setdefault(ue, []).append(feats)
        if len(buffers[ue]) < args.t:
            continue
        if len(buffers[ue]) > args.t:
            buffers[ue] = buffers[ue][-args.t :]

        win = np.stack(buffers[ue], axis=0)  # (T,K,3)
        win = win.transpose(1, 0, 2)  # (K,T,3)
        mask = (win[:, -1, 0] != 0.0).astype(np.float32)

        # Scale valid cells only (same convention as training cache)
        v = mask == 1.0
        win_scaled = win.copy()
        if v.sum() > 0:
            win_scaled[v] = scaler.transform(win[v].reshape(-1, 3)).reshape(-1, args.t, 3)

        probs = model.predict({"cells": win_scaled[None, ...], "mask": mask[None, ...]}, verbose=0)[0]

        decision = gate.update(
            ue_id=ue,
            serving_cell_id=int(row["serving_cell_id"]),
            nb_cell_ids=nb_ids,
            cell_probs=np.asarray(probs, dtype=np.float32),
        )

        if decision.action == "handover":
            print(
                f"{row['timestamp']} ue={ue} HO → target={decision.target_cell_id} "
                f"margin={decision.margin:.3f} p_best={decision.best_prob:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
