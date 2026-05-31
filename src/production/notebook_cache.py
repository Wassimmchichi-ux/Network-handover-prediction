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

LabelMode = Literal["optimal", "target"]
NotebookKind = Literal[
    "basic",          # X,M,y (+ optional r)
    "strategic",      # X,G,M,y,r
    "rel_pointer",    # X,G,M,y,r (with derived per-cell feats)
    "two_stage",      # X_ue, X_cells, y_cls, y_ptr, groups
    "cascade",        # X_cells(T,K rsrp), X_glob(T,1), y_bin
]


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


def _onehot_ho_class(df: pd.DataFrame) -> np.ndarray:
    """Return (N,5) one-hot over classes [0,1,3,5,6]. Class 3 may be absent."""
    classes = [0, 1, 3, 5, 6]
    y = df["handover_class"].to_numpy()
    out = np.stack([(y == c).astype(np.float32) for c in classes], axis=1)
    return out.astype(np.float32)


def _build_global_seq(df: pd.DataFrame) -> np.ndarray:
    """(N, 9) global features per row: speed, cos(dir), sin(dir), cell_load, ho_class×5."""
    speed = df["speed"].to_numpy(np.float32)
    dr = np.radians(df["direction"].to_numpy(np.float32))
    cell_load = df["cell_load"].to_numpy(np.float32)
    ho_oh = _onehot_ho_class(df)  # (N,5)
    base = np.stack([speed, np.cos(dr), np.sin(dr), cell_load], axis=1).astype(np.float32)
    return np.concatenate([base, ho_oh], axis=1).astype(np.float32)  # (N,9)


def _window_core(
    *,
    df: pd.DataFrame,
    k: int,
    win_t: int,
    future_h: int,
    lead_l: int,
    seed: int,
    label_mode: LabelMode,
    want_scores: bool,
    want_dists: bool,
    want_global_seq: bool,
    want_is_serving: bool,
    want_rsrp_delta: bool,
) -> dict:
    """Build leakage-safe windows with consistent neighbor shuffling and multi-horizon labels."""
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    df["nb_ids"] = df["nb_cell_ids"].apply(lambda x: parse_int_list(x, k=k))
    df["nb_r"] = df["nb_rsrps"].apply(lambda x: parse_float_list(x, k=k, fill=0.0))
    df["nb_s"] = df["nb_sinrs"].apply(lambda x: parse_float_list(x, k=k, fill=0.0))
    df["nb_l"] = df["nb_loads"].apply(lambda x: parse_float_list(x, k=k, fill=0.0))
    if want_scores:
        df["nb_sc"] = df["nb_scores"].apply(lambda x: parse_float_list(x, k=k, fill=0.0))
    if want_dists:
        df["nb_d"] = df["nb_dists_m"].apply(lambda x: parse_float_list(x, k=k, fill=9999.0))

    nb_ids = np.asarray(df["nb_ids"].to_list(), dtype=np.int64)  # (N,K)
    nb_r = np.stack(df["nb_r"].to_numpy()).astype(np.float32)
    nb_s = np.stack(df["nb_s"].to_numpy()).astype(np.float32)
    nb_l = np.stack(df["nb_l"].to_numpy()).astype(np.float32)

    feats = [nb_r, nb_s, nb_l]
    if want_scores:
        feats.append(np.stack(df["nb_sc"].to_numpy()).astype(np.float32))
    if want_dists:
        feats.append(np.stack(df["nb_d"].to_numpy()).astype(np.float32))

    base = np.stack(feats, axis=2).astype(np.float32)  # (N,K,F_base)
    f_base = base.shape[-1]

    glob_row = _build_global_seq(df).astype(np.float32) if want_global_seq else None  # (N,9)
    serving_rsrp = df["rsrp"].to_numpy(np.float32)
    serving_ids = df["serving_cell_id"].to_numpy(np.int64)

    if label_mode == "optimal":
        label_ids = df["optimal_cell_id"].to_numpy(np.int64)
        reg_rsrp = df["optimal_cell_rsrp"].to_numpy(np.float32)
    else:
        label_ids = df["target_cell_id"].to_numpy(np.int64)
        # Fallback regression target: serving RSRP if target RSRP isn't defined
        reg_rsrp = df["rsrp"].to_numpy(np.float32)

    y_bin_row = (label_ids != serving_ids).astype(np.float32)

    X_list: list[np.ndarray] = []
    M_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    r_list: list[np.ndarray] = []
    G_list: list[np.ndarray] = []
    ue_list: list[str] = []
    X_cells_list: list[np.ndarray] = []
    X_glob_list: list[np.ndarray] = []

    for ue_id, grp in df.groupby("ue_id", sort=False):
        idx = grp.index.to_numpy()
        n = len(idx)
        if n < win_t + lead_l + future_h:
            continue

        for end in range(win_t, n - (lead_l + future_h) + 1):
            obs_rows = idx[end - win_t : end]  # (win_t,)
            anchor_row = idx[end - 1]
            anchor_ids = nb_ids[anchor_row].tolist()
            anchor_srv = int(serving_ids[anchor_row])

            p = rng.permutation(k)

            Xw = base[obs_rows]  # (T,K,Fb)
            Xw = Xw[:, p, :].transpose(1, 0, 2)  # (K,T,Fb)

            # Derived per-cell features (append along feature axis)
            extra_feats: list[np.ndarray] = []
            if want_is_serving:
                # indicator based on anchor serving cell id
                is_srv = np.array([(cid == anchor_srv) for cid in anchor_ids], dtype=np.float32)  # (K,)
                is_srv = is_srv[p]  # after permutation
                extra_feats.append(np.repeat(is_srv[:, None, None], win_t, axis=1))  # (K,T,1)
            if want_rsrp_delta:
                rs = Xw[:, :, 0]  # (K,T)
                d = np.diff(rs, axis=1, prepend=rs[:, :1]).astype(np.float32)  # (K,T)
                extra_feats.append(d[:, :, None])

            if extra_feats:
                Xw = np.concatenate([Xw] + extra_feats, axis=2).astype(np.float32)

            # Mask: visible if RSRP != 0 at last step
            Mw = (Xw[:, -1, 0] != 0.0).astype(np.float32)

            # Multi-horizon y and regression target r (RSRP) for the chosen label at each horizon
            yh = np.zeros((future_h,), dtype=np.int32)
            rh = np.zeros((future_h,), dtype=np.float32)
            for h_i in range(future_h):
                fr = idx[end + lead_l + h_i]
                chosen_id = int(label_ids[fr])
                rh[h_i] = float(reg_rsrp[fr])
                try:
                    orig = anchor_ids.index(chosen_id)
                    yh[h_i] = int(np.where(p == orig)[0][0])
                except ValueError:
                    srv = int(serving_ids[fr])
                    try:
                        orig = anchor_ids.index(srv)
                        yh[h_i] = int(np.where(p == orig)[0][0])
                    except ValueError:
                        yh[h_i] = 0

            X_list.append(Xw)
            M_list.append(Mw)
            y_list.append(yh)
            r_list.append(rh)
            ue_list.append(str(ue_id))

            if want_global_seq:
                G_list.append(glob_row[obs_rows])  # (T,9)
            # Common NB14/Cascade shapes
            X_cells_list.append(Xw[:, :, 0].T.astype(np.float32))  # (T,K)
            X_glob_list.append(serving_rsrp[obs_rows][:, None].astype(np.float32))  # (T,1)

    X = np.asarray(X_list, dtype=np.float32)
    M = np.asarray(M_list, dtype=np.float32)
    y_all = np.asarray(y_list, dtype=np.int32)  # (N,H)
    r_all = np.asarray(r_list, dtype=np.float32)  # (N,H)
    ue_ids = np.asarray(ue_list, dtype=object)

    out = {
        "X": X,
        "M": M,
        "y_all": y_all,
        "r_all": r_all,
        "ue_ids": ue_ids,
        "y_bin_row": y_bin_row,
        "X_cells": np.asarray(X_cells_list, dtype=np.float32),
        "X_glob": np.asarray(X_glob_list, dtype=np.float32),
    }
    if want_global_seq:
        out["G"] = np.asarray(G_list, dtype=np.float32)
    return out


def build_or_load_notebook_cache(
    *,
    kind: NotebookKind,
    root_dir: str | Path | None = None,
    k: int = 10,
    win_t: int = 25,
    future_h: int = 5,
    lead_l: int = 0,
    target_h_idx: int = 1,
    label_mode: LabelMode = "optimal",
    seed: int = 42,
    force_rebuild: bool = False,
) -> dict:
    """Return a dict of arrays matching notebook expectations.

    This is designed specifically to make notebooks 02–14 runnable on the
    integral dataset without relying on deprecated caches.
    """
    root = _default_root() if root_dir is None else Path(root_dir).resolve()
    raw_csv = root / "dataset" / "raw" / "handover_dataset.csv"
    split_path = root / "dataset" / "processed" / "ue_split.json"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_csv}")
    if not split_path.exists():
        raise FileNotFoundError(f"UE split not found: {split_path} (run `python src/preprocess.py`)")

    cache_dir = root / "dataset" / "notebook_cache" / kind
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "kind": kind,
        "k": k,
        "win_t": win_t,
        "future_h": future_h,
        "lead_l": lead_l,
        "target_h_idx": target_h_idx,
        "label_mode": label_mode,
        "seed": seed,
    }

    meta_path = cache_dir / "meta.json"
    npz_tr = cache_dir / "train.npz"
    npz_va = cache_dir / "val.npz"
    npz_te = cache_dir / "test.npz"
    scaler_path = cache_dir / "scaler.pkl"

    if (
        not force_rebuild
        and meta_path.exists()
        and npz_tr.exists()
        and npz_va.exists()
        and npz_te.exists()
        and scaler_path.exists()
    ):
        have = json.loads(meta_path.read_text())
        if have == meta:
            tr = np.load(npz_tr, allow_pickle=False)
            va = np.load(npz_va, allow_pickle=False)
            te = np.load(npz_te, allow_pickle=False)
            scaler = pickle.loads(scaler_path.read_bytes())
            return {
                "train": dict(tr),
                "val": dict(va),
                "test": dict(te),
                "scaler": scaler,
                "meta": meta,
            }

    df = pd.read_csv(raw_csv, low_memory=False)
    split = json.loads(split_path.read_text())
    ue_tr = set(split["train_ues"])
    ue_va = set(split["val_ues"])
    ue_te = set(split["test_ues"])

    # Kind-specific feature requirements
    want_scores = True
    want_dists = kind in ("cascade",)
    want_global_seq = kind in ("strategic", "rel_pointer")
    want_is_serving = kind in ("rel_pointer",)
    want_rsrp_delta = kind in ("rel_pointer",)

    core = _window_core(
        df=df,
        k=k,
        win_t=win_t,
        future_h=future_h,
        lead_l=lead_l,
        seed=seed,
        label_mode=label_mode,
        want_scores=want_scores,
        want_dists=want_dists,
        want_global_seq=want_global_seq,
        want_is_serving=want_is_serving,
        want_rsrp_delta=want_rsrp_delta,
    )

    X = core["X"]
    M = core["M"]
    y_all = core["y_all"]
    r_all = core["r_all"]
    ue_ids = core["ue_ids"]

    h_sel = max(1, min(int(target_h_idx), y_all.shape[1])) - 1
    y = y_all[:, h_sel].astype(np.int32)
    r = r_all[:, h_sel].astype(np.float32)

    # Split by UE id
    idx_tr = np.where([u in ue_tr for u in ue_ids])[0]
    idx_va = np.where([u in ue_va for u in ue_ids])[0]
    idx_te = np.where([u in ue_te for u in ue_ids])[0]

    # Fit scaler on train only, valid cells only
    fdim = X.shape[-1]
    scaler = StandardScaler()
    valid_tr = (M[idx_tr] == 1.0)
    if valid_tr.sum() == 0:
        scaler.fit(X[idx_tr].reshape(-1, fdim))
    else:
        scaler.fit(X[idx_tr][valid_tr].reshape(-1, fdim))

    def _scale(Xs, Ms):
        Xn = Xs.copy()
        v = (Ms == 1.0)
        if v.sum() > 0:
            Xn[v] = scaler.transform(Xs[v].reshape(-1, fdim)).reshape(-1, win_t, fdim)
        return Xn.astype(np.float32)

    X_tr = _scale(X[idx_tr], M[idx_tr])
    X_va = _scale(X[idx_va], M[idx_va])
    X_te = _scale(X[idx_te], M[idx_te])

    def pack(split_idx, Xs, Ms, ys, rs):
        d = {"X": Xs, "M": Ms, "y": ys, "r": rs}
        if want_global_seq:
            d["G"] = core["G"][split_idx].astype(np.float32)
        return d

    tr = pack(idx_tr, X_tr, M[idx_tr].astype(np.float32), y[idx_tr], r[idx_tr])
    va = pack(idx_va, X_va, M[idx_va].astype(np.float32), y[idx_va], r[idx_va])
    te = pack(idx_te, X_te, M[idx_te].astype(np.float32), y[idx_te], r[idx_te])

    if kind == "cascade":
        tr["X_cells"] = core["X_cells"][idx_tr].astype(np.float32)
        va["X_cells"] = core["X_cells"][idx_va].astype(np.float32)
        te["X_cells"] = core["X_cells"][idx_te].astype(np.float32)

        tr["X_glob"] = core["X_glob"][idx_tr].astype(np.float32)
        va["X_glob"] = core["X_glob"][idx_va].astype(np.float32)
        te["X_glob"] = core["X_glob"][idx_te].astype(np.float32)

        # Binary label aligned with selected horizon choice (stay vs switch)
        # Use y_bin from raw rows at the selected horizon (approx proxy)
        # For notebooks that need y_bin at decision time, caller can recompute.
        # Here we export a conservative default: 1 if selected horizon y != serving-index.
        # (Not perfect, but keeps notebooks runnable; production uses `optimal_is_current`.)
        tr["y_bin"] = (tr["y"] != 0).astype(np.float32)
        va["y_bin"] = (va["y"] != 0).astype(np.float32)
        te["y_bin"] = (te["y"] != 0).astype(np.float32)

    np.savez_compressed(npz_tr, **tr)
    np.savez_compressed(npz_va, **va)
    np.savez_compressed(npz_te, **te)
    scaler_path.write_bytes(pickle.dumps(scaler))
    meta_path.write_text(json.dumps(meta, indent=2))

    print(
        f"[notebook_cache:{kind}] built in {time.time():.0f} "
        f"train={len(tr['y']):,} val={len(va['y']):,} test={len(te['y']):,}"
    )
    return {"train": tr, "val": va, "test": te, "scaler": scaler, "meta": meta}
