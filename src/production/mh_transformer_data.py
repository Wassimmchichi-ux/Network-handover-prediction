from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


_BRACKETS = re.compile(r"[\[\]]")


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_float_list(s, *, k: int, fill: float = 0.0) -> np.ndarray:
    if pd.isna(s):
        return np.full(k, fill, dtype=np.float32)
    cleaned = _BRACKETS.sub("", str(s)).strip()
    if not cleaned:
        return np.full(k, fill, dtype=np.float32)
    parts = re.split(r"[,;]", cleaned)
    out: list[float] = []
    for p in parts[:k]:
        p = p.strip()
        if p == "" or p.lower() in ("nan", "none"):
            out.append(fill)
        else:
            try:
                out.append(float(p))
            except ValueError:
                out.append(fill)
    out += [fill] * (k - len(out))
    return np.asarray(out, dtype=np.float32)


def parse_int_list(s, *, k: int) -> list[int]:
    if pd.isna(s):
        return [0] * k
    cleaned = _BRACKETS.sub("", str(s)).strip()
    if not cleaned:
        return [0] * k
    parts = re.split(r"[,;]", cleaned)
    out: list[int] = []
    for p in parts[:k]:
        p = p.strip()
        try:
            out.append(int(float(p)))
        except (ValueError, TypeError):
            out.append(0)
    out += [0] * (k - len(out))
    return out


LabelMode = Literal["optimal", "target"]


@dataclass(frozen=True)
class MultiHorizonArtifacts:
    X_tr: np.ndarray
    M_tr: np.ndarray
    y_tr: np.ndarray  # (N, H)
    X_va: np.ndarray
    M_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    M_te: np.ndarray
    y_te: np.ndarray
    scaler: StandardScaler
    meta: dict


def build_or_load_mh_cache(
    *,
    root_dir: str | Path | None = None,
    cache_subdir: str = "mh_cache",
    k: int = 10,
    t: int = 25,
    h: int = 5,
    lead: int = 0,
    seed: int = 42,
    label_mode: LabelMode = "optimal",
    include_global: bool = True,
    include_scores: bool = True,
    force_rebuild: bool = False,
) -> MultiHorizonArtifacts:
    """Leakage-safe multi-horizon dataset builder.

    Label semantics
    --------------
    For each window ending at time index (t-1), we predict the best cell at
    future rows (t+lead+h_i). Labels are mapped into the **anchor neighbor list**
    from the last observation row (t-1), then remapped after a per-sample
    neighbor-axis permutation (to remove order leakage).
    """
    root = _default_root() if root_dir is None else Path(root_dir).resolve()
    raw_csv = root / "dataset" / "raw" / "handover_dataset.csv"
    ue_split = root / "dataset" / "processed" / "ue_split.json"
    cache_dir = root / "dataset" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "k": k,
        "t": t,
        "h": h,
        "lead": lead,
        "seed": seed,
        "label_mode": label_mode,
        "include_global": include_global,
        "include_scores": include_scores,
        "cell_features": ["nb_rsrps", "nb_sinrs", "nb_loads"],
        "global_features": ["speed", "direction_cos", "direction_sin", "cell_load"] if include_global else [],
    }

    meta_path = cache_dir / "meta.json"
    scaler_path = cache_dir / "scaler.pkl"
    train_path = cache_dir / "train.npz"
    val_path = cache_dir / "val.npz"
    test_path = cache_dir / "test.npz"

    if (
        not force_rebuild
        and meta_path.exists()
        and scaler_path.exists()
        and train_path.exists()
        and val_path.exists()
        and test_path.exists()
    ):
        have = json.loads(meta_path.read_text())
        if have == meta:
            tr = np.load(train_path)
            va = np.load(val_path)
            te = np.load(test_path)
            scaler = pickle.loads(scaler_path.read_bytes())
            return MultiHorizonArtifacts(
                X_tr=tr["X"].astype(np.float32),
                M_tr=tr["M"].astype(np.float32),
                y_tr=tr["y"].astype(np.int32),
                X_va=va["X"].astype(np.float32),
                M_va=va["M"].astype(np.float32),
                y_va=va["y"].astype(np.int32),
                X_te=te["X"].astype(np.float32),
                M_te=te["M"].astype(np.float32),
                y_te=te["y"].astype(np.int32),
                scaler=scaler,
                meta=meta,
            )

    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_csv}")
    if not ue_split.exists():
        raise FileNotFoundError(f"UE split not found: {ue_split} (run `python src/preprocess.py`)")

    t0 = time.time()
    df = pd.read_csv(raw_csv, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    split = json.loads(ue_split.read_text())
    ue_tr = set(split["train_ues"])
    ue_va = set(split["val_ues"])
    ue_te = set(split["test_ues"])

    df["nb_ids"] = df["nb_cell_ids"].apply(lambda x: parse_int_list(x, k=k))
    df["nb_r"] = df["nb_rsrps"].apply(lambda x: parse_float_list(x, k=k, fill=0.0))
    df["nb_s"] = df["nb_sinrs"].apply(lambda x: parse_float_list(x, k=k, fill=0.0))
    df["nb_l"] = df["nb_loads"].apply(lambda x: parse_float_list(x, k=k, fill=0.0))
    if include_scores:
        df["nb_sc"] = df["nb_scores"].apply(lambda x: parse_float_list(x, k=k, fill=0.0))

    nb_ids = np.asarray(df["nb_ids"].to_list(), dtype=np.int64)  # (N, K)
    nb_r = np.stack(df["nb_r"].to_numpy()).astype(np.float32)
    nb_s = np.stack(df["nb_s"].to_numpy()).astype(np.float32)
    nb_l = np.stack(df["nb_l"].to_numpy()).astype(np.float32)
    if include_scores:
        nb_sc = np.stack(df["nb_sc"].to_numpy()).astype(np.float32)
        base = np.stack([nb_r, nb_s, nb_l, nb_sc], axis=2).astype(np.float32)  # (N, K, 4)
    else:
        base = np.stack([nb_r, nb_s, nb_l], axis=2).astype(np.float32)  # (N, K, 3)

    if include_global:
        speed = df["speed"].to_numpy(np.float32)
        dr = np.radians(df["direction"].to_numpy(np.float32))
        glob = np.stack(
            [speed, np.cos(dr), np.sin(dr), df["cell_load"].to_numpy(np.float32)],
            axis=1,
        ).astype(np.float32)  # (N, 4)
        gk = np.repeat(glob[:, None, :], k, axis=1)  # (N, K, 4)
        cell_feat = np.concatenate([base, gk], axis=2).astype(np.float32)  # (N, K, 7)
    else:
        cell_feat = base

    if label_mode == "optimal":
        label_ids = df["optimal_cell_id"].to_numpy(np.int64)
    elif label_mode == "target":
        label_ids = df["target_cell_id"].to_numpy(np.int64)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    serving_ids = df["serving_cell_id"].to_numpy(np.int64)

    rng = np.random.default_rng(seed)
    out = {s: {"X": [], "M": [], "y": []} for s in ("train", "val", "test")}

    for ue_id, grp in df.groupby("ue_id", sort=False):
        ue_key = str(ue_id)
        if ue_key in ue_tr:
            split_name = "train"
        elif ue_key in ue_va:
            split_name = "val"
        elif ue_key in ue_te:
            split_name = "test"
        else:
            continue

        idx = grp.index.to_numpy()
        n = len(idx)
        if n < t + lead + h:
            continue

        for end in range(t, n - (lead + h) + 1):
            obs_rows = idx[end - t : end]  # len=t, last obs row = end-1
            anchor_row = idx[end - 1]
            anchor_ids = nb_ids[anchor_row].tolist()

            p = rng.permutation(k)
            Xw = cell_feat[obs_rows]  # (t, k, f)
            Xw = Xw[:, p, :].transpose(1, 0, 2)  # (k, t, f)
            Mw = (Xw[:, -1, 0] != 0.0).astype(np.float32)

            yh = np.zeros((h,), dtype=np.int32)
            for step in range(h):
                future_row = idx[end + lead + step]
                chosen_id = int(label_ids[future_row])
                try:
                    orig = anchor_ids.index(chosen_id)
                    yh[step] = int(np.where(p == orig)[0][0])
                except ValueError:
                    srv = int(serving_ids[future_row])
                    try:
                        orig = anchor_ids.index(srv)
                        yh[step] = int(np.where(p == orig)[0][0])
                    except ValueError:
                        yh[step] = 0

            out[split_name]["X"].append(Xw)
            out[split_name]["M"].append(Mw)
            out[split_name]["y"].append(yh)

    def _stack(split_name: str):
        return (
            np.asarray(out[split_name]["X"], dtype=np.float32),
            np.asarray(out[split_name]["M"], dtype=np.float32),
            np.asarray(out[split_name]["y"], dtype=np.int32),
        )

    X_tr, M_tr, y_tr = _stack("train")
    X_va, M_va, y_va = _stack("val")
    X_te, M_te, y_te = _stack("test")

    # Scale on TRAIN only, valid cells only.
    fdim = X_tr.shape[-1]
    scaler = StandardScaler()
    valid_tr = (M_tr == 1.0)
    if valid_tr.sum() == 0:
        scaler.fit(X_tr.reshape(-1, fdim))

        def _scale(X):
            return scaler.transform(X.reshape(-1, fdim)).reshape(X.shape).astype(np.float32)

        X_trn, X_van, X_ten = _scale(X_tr), _scale(X_va), _scale(X_te)
    else:
        scaler.fit(X_tr[valid_tr].reshape(-1, fdim))

        def _scale_masked(X, M):
            Xn = X.copy()
            v = (M == 1.0)
            if v.sum() > 0:
                Xn[v] = scaler.transform(X[v].reshape(-1, fdim)).reshape(-1, t, fdim)
            return Xn.astype(np.float32)

        X_trn = _scale_masked(X_tr, M_tr)
        X_van = _scale_masked(X_va, M_va)
        X_ten = _scale_masked(X_te, M_te)

    np.savez_compressed(train_path, X=X_trn, M=M_tr, y=y_tr)
    np.savez_compressed(val_path, X=X_van, M=M_va, y=y_va)
    np.savez_compressed(test_path, X=X_ten, M=M_te, y=y_te)
    scaler_path.write_bytes(pickle.dumps(scaler))
    meta_path.write_text(json.dumps(meta, indent=2))

    elapsed = time.time() - t0
    print(
        f"[mh_transformer_data] built cache in {elapsed:.1f}s "
        f"(train={len(y_tr):,}, val={len(y_va):,}, test={len(y_te):,})"
    )

    return MultiHorizonArtifacts(
        X_tr=X_trn,
        M_tr=M_tr,
        y_tr=y_tr,
        X_va=X_van,
        M_va=M_va,
        y_va=y_va,
        X_te=X_ten,
        M_te=M_te,
        y_te=y_te,
        scaler=scaler,
        meta=meta,
    )
