from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


_BRACKETS = re.compile(r"[\[\]]")


def parse_float_list(
    s,
    *,
    max_len: int,
    fill: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """Parse a CSV cell containing a list of floats.

    Supports:
      - Python-like lists: "[-44.0, -47.1, ...]"
      - Semicolon lists : "[-44.0;-47.1;...]"
    Pads to `max_len` with `fill`.
    """
    if pd.isna(s):
        return np.full(max_len, fill, dtype=dtype)
    cleaned = _BRACKETS.sub("", str(s)).strip()
    if not cleaned:
        return np.full(max_len, fill, dtype=dtype)
    parts = re.split(r"[,;]", cleaned)
    out: list[float] = []
    for p in parts[:max_len]:
        p = p.strip()
        if p == "" or p.lower() in ("nan", "none"):
            out.append(fill)
        else:
            try:
                out.append(float(p))
            except ValueError:
                out.append(fill)
    out += [fill] * (max_len - len(out))
    return np.asarray(out, dtype=dtype)


def parse_int_list(s, *, max_len: int) -> list[int]:
    """Parse a CSV cell containing a list of cell IDs."""
    if pd.isna(s):
        return [0] * max_len
    cleaned = _BRACKETS.sub("", str(s)).strip()
    if not cleaned:
        return [0] * max_len
    parts = re.split(r"[,;]", cleaned)
    out: list[int] = []
    for p in parts[:max_len]:
        p = p.strip()
        try:
            out.append(int(float(p)))
        except (ValueError, TypeError):
            out.append(0)
    out += [0] * (max_len - len(out))
    return out


LabelMode = Literal["optimal", "target"]


@dataclass(frozen=True)
class TemporalDeepSetArtifacts:
    X_tr: np.ndarray
    M_tr: np.ndarray
    y_tr: np.ndarray
    ybin_tr: np.ndarray
    X_va: np.ndarray
    M_va: np.ndarray
    y_va: np.ndarray
    ybin_va: np.ndarray
    X_te: np.ndarray
    M_te: np.ndarray
    y_te: np.ndarray
    ybin_te: np.ndarray
    scaler: StandardScaler
    ue_split: dict


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_ue_split(root: Path, cache_dir: Path, *, seed: int, val_frac: float, test_frac: float) -> dict:
    """Prefer the canonical UE split if it exists, otherwise create one."""
    canonical = root / "dataset" / "processed" / "ue_split.json"
    if canonical.exists():
        return json.loads(canonical.read_text())

    split_path = cache_dir / "ue_split.json"
    if split_path.exists():
        return json.loads(split_path.read_text())

    df = pd.read_csv(root / "dataset" / "raw" / "handover_dataset.csv", usecols=["ue_id"])
    ue_list = df["ue_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ue_list)

    n_test = max(1, int(len(ue_list) * test_frac))
    n_val = max(1, int(len(ue_list) * val_frac))

    ue_test = sorted(str(u) for u in ue_list[:n_test])
    ue_val = sorted(str(u) for u in ue_list[n_test : n_test + n_val])
    ue_train = sorted(str(u) for u in ue_list[n_test + n_val :])

    ue_split = {"train_ues": ue_train, "val_ues": ue_val, "test_ues": ue_test}
    split_path.write_text(json.dumps(ue_split, indent=2))
    return ue_split


def build_or_load_temporal_deepset_artifacts(
    *,
    root_dir: str | Path | None = None,
    cache_subdir: str = "temporal_deepset_cache",
    max_cells: int = 10,
    obs_steps: int = 25,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    label_mode: LabelMode = "optimal",
    force_rebuild: bool = False,
) -> TemporalDeepSetArtifacts:
    """Create or load cached artifacts for TemporalDeepSet training.

    Dataset semantics
    -----------------
    - Inputs are built from UE windows of length `obs_steps`.
    - Neighbor order leakage is prevented by applying a random permutation
      to the K neighbor axis per training sample.
    - Labels are the index of the chosen cell *after permutation*.
    """
    root = _default_root() if root_dir is None else Path(root_dir).resolve()
    raw_csv = root / "dataset" / "raw" / "handover_dataset.csv"
    cache_dir = root / "dataset" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "meta.json"
    scaler_path = cache_dir / "scaler.pkl"
    train_path = cache_dir / "train.npz"
    val_path = cache_dir / "val.npz"
    test_path = cache_dir / "test.npz"

    want_meta = {
        "max_cells": max_cells,
        "obs_steps": obs_steps,
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "label_mode": label_mode,
        "features": ["nb_rsrps", "nb_sinrs", "nb_loads"],
    }

    if not force_rebuild and all(p.exists() for p in (meta_path, scaler_path, train_path, val_path, test_path)):
        meta = json.loads(meta_path.read_text())
        if meta == want_meta:
            tr = np.load(train_path)
            va = np.load(val_path)
            te = np.load(test_path)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            ue_split = _get_ue_split(root, cache_dir, seed=seed, val_frac=val_frac, test_frac=test_frac)
            return TemporalDeepSetArtifacts(
                X_tr=tr["X"].astype(np.float32),
                M_tr=tr["M"].astype(np.float32),
                y_tr=tr["y"].astype(np.int32),
                ybin_tr=tr["y_bin"].astype(np.float32),
                X_va=va["X"].astype(np.float32),
                M_va=va["M"].astype(np.float32),
                y_va=va["y"].astype(np.int32),
                ybin_va=va["y_bin"].astype(np.float32),
                X_te=te["X"].astype(np.float32),
                M_te=te["M"].astype(np.float32),
                y_te=te["y"].astype(np.int32),
                ybin_te=te["y_bin"].astype(np.float32),
                scaler=scaler,
                ue_split=ue_split,
            )

    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_csv}")

    t0 = time.time()
    df = pd.read_csv(raw_csv, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    df["nb_ids"] = df["nb_cell_ids"].apply(lambda x: parse_int_list(x, max_len=max_cells))
    df["nb_r"] = df["nb_rsrps"].apply(lambda x: parse_float_list(x, max_len=max_cells, fill=0.0))
    df["nb_s"] = df["nb_sinrs"].apply(lambda x: parse_float_list(x, max_len=max_cells, fill=0.0))
    df["nb_l"] = df["nb_loads"].apply(lambda x: parse_float_list(x, max_len=max_cells, fill=0.0))

    nb_ids = np.asarray(df["nb_ids"].to_list(), dtype=np.int64)  # (N, K)
    nb_r = np.stack(df["nb_r"].to_numpy()).astype(np.float32)  # (N, K)
    nb_s = np.stack(df["nb_s"].to_numpy()).astype(np.float32)  # (N, K)
    nb_l = np.stack(df["nb_l"].to_numpy()).astype(np.float32)  # (N, K)
    cell_feat = np.stack([nb_r, nb_s, nb_l], axis=2).astype(np.float32)  # (N, K, 3)

    if label_mode == "optimal":
        label_ids = df["optimal_cell_id"].to_numpy().astype(np.int64)
    elif label_mode == "target":
        label_ids = df["target_cell_id"].to_numpy().astype(np.int64)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    serving_ids = df["serving_cell_id"].to_numpy().astype(np.int64)
    y_bin_row = (label_ids != serving_ids).astype(np.float32)

    ue_split = _get_ue_split(root, cache_dir, seed=seed, val_frac=val_frac, test_frac=test_frac)
    ue_train = set(ue_split["train_ues"])
    ue_val = set(ue_split["val_ues"])
    ue_test = set(ue_split["test_ues"])

    rng = np.random.default_rng(seed)

    out = {
        "train": {"X": [], "M": [], "y": [], "y_bin": []},
        "val": {"X": [], "M": [], "y": [], "y_bin": []},
        "test": {"X": [], "M": [], "y": [], "y_bin": []},
    }

    for ue_id, grp in df.groupby("ue_id", sort=False):
        ue_key = str(ue_id)
        if ue_key in ue_train:
            split = "train"
        elif ue_key in ue_val:
            split = "val"
        elif ue_key in ue_test:
            split = "test"
        else:
            # Should not happen, but be safe and ignore unknown UE.
            continue

        idx = grp.index.to_numpy()
        n_rows = len(idx)
        if n_rows < obs_steps + 1:
            continue

        for t in range(obs_steps, n_rows):
            rows = idx[t - obs_steps : t]  # observation window (len=obs_steps)
            X_w = cell_feat[rows]  # (obs, K, 3)

            p = rng.permutation(max_cells)
            X_w = X_w[:, p, :]  # (obs, K, 3)
            X_w = X_w.transpose(1, 0, 2)  # (K, obs, 3)

            M_w = (X_w[:, -1, 0] != 0.0).astype(np.float32)  # based on RSRP at last step

            chosen_cell_id = int(label_ids[idx[t]])
            nb_ids_t = nb_ids[idx[t]].tolist()
            try:
                orig_idx = nb_ids_t.index(chosen_cell_id)
                y_w = int(np.where(p == orig_idx)[0][0])
            except ValueError:
                # Fallback to serving cell if chosen cell not present
                srv = int(serving_ids[idx[t]])
                try:
                    orig_idx = nb_ids_t.index(srv)
                    y_w = int(np.where(p == orig_idx)[0][0])
                except ValueError:
                    y_w = 0

            out[split]["X"].append(X_w)
            out[split]["M"].append(M_w)
            out[split]["y"].append(y_w)
            out[split]["y_bin"].append(float(y_bin_row[idx[t]]))

    def _stack(split: str):
        return (
            np.asarray(out[split]["X"], dtype=np.float32),
            np.asarray(out[split]["M"], dtype=np.float32),
            np.asarray(out[split]["y"], dtype=np.int32),
            np.asarray(out[split]["y_bin"], dtype=np.float32),
        )

    X_tr, M_tr, y_tr, yb_tr = _stack("train")
    X_va, M_va, y_va, yb_va = _stack("val")
    X_te, M_te, y_te, yb_te = _stack("test")

    # Scale using TRAIN only, on valid cells.
    scaler = StandardScaler()
    valid = (M_tr == 1.0)  # (N_tr, K)
    if valid.sum() == 0:
        scaler.fit(X_tr.reshape(-1, X_tr.shape[-1]))
        X_tr_n = scaler.transform(X_tr.reshape(-1, 3)).reshape(X_tr.shape).astype(np.float32)
        X_va_n = scaler.transform(X_va.reshape(-1, 3)).reshape(X_va.shape).astype(np.float32)
        X_te_n = scaler.transform(X_te.reshape(-1, 3)).reshape(X_te.shape).astype(np.float32)
    else:
        scaler.fit(X_tr[valid].reshape(-1, 3))

        def _transform(X: np.ndarray, M: np.ndarray) -> np.ndarray:
            Xn = X.copy()
            v = (M == 1.0)
            if v.sum() > 0:
                Xn[v] = scaler.transform(X[v].reshape(-1, 3)).reshape(-1, obs_steps, 3)
            return Xn.astype(np.float32)

        X_tr_n = _transform(X_tr, M_tr)
        X_va_n = _transform(X_va, M_va)
        X_te_n = _transform(X_te, M_te)

    np.savez_compressed(train_path, X=X_tr_n, M=M_tr, y=y_tr, y_bin=yb_tr)
    np.savez_compressed(val_path, X=X_va_n, M=M_va, y=y_va, y_bin=yb_va)
    np.savez_compressed(test_path, X=X_te_n, M=M_te, y=y_te, y_bin=yb_te)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    meta_path.write_text(json.dumps(want_meta, indent=2))

    elapsed = time.time() - t0
    print(
        f"[temporal_deepset_data] built cache in {elapsed:.1f}s "
        f"(train={len(y_tr):,}, val={len(y_va):,}, test={len(y_te):,})"
    )

    return TemporalDeepSetArtifacts(
        X_tr=X_tr_n,
        M_tr=M_tr,
        y_tr=y_tr,
        ybin_tr=yb_tr,
        X_va=X_va_n,
        M_va=M_va,
        y_va=y_va,
        ybin_va=yb_va,
        X_te=X_te_n,
        M_te=M_te,
        y_te=y_te,
        ybin_te=yb_te,
        scaler=scaler,
        ue_split=ue_split,
    )

