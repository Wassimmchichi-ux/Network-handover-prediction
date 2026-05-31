"""
preprocess.py  —  Generate all split artifacts from handover_dataset.csv
========================================================================

Outputs
-------
dataset/processed/
    train.npz, val.npz, test.npz   – X[N,K,T,F], M[N,K], y[N]  (NB01, NB05)
    scaler.pkl                     – fitted StandardScaler
    ue_split.json                  – {"train_ues":[], "val_ues":[], "test_ues":[]}
                                     (NB02, NB03, NB04, NB06 reuse this split)

notebooks/modeling/
    X_cells.npy  [N, T, K]        – per-neighbour RSRP sequences  (NB14)
    X_glob.npy   [N, T, 1]        – serving-cell RSRP sequence    (NB14)
    y.npy        [N]              – pointer label 0..K-1          (NB14)

Usage
-----
    # From project root:
    python src/preprocess.py

    # Regenerate even if cached files exist:
    python src/preprocess.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────────
K          = 10        # max neighbour cells
T          = 25        # observation window (timesteps)
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SEED       = 42

# ── Paths (relative to project root) ──────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent          # src/
ROOT       = _HERE.parent                             # project root
RAW_CSV    = ROOT / "dataset" / "raw" / "handover_dataset.csv"
PROC_DIR   = ROOT / "dataset" / "processed"
NB_DIR     = ROOT / "notebooks" / "modeling"

# ── Array parser ──────────────────────────────────────────────────────────────

def parse_nb_array(s, *, max_len: int = K, fill: float = 0.0,
                   dtype=np.float32) -> np.ndarray:
    """Parse a neighbour float-array cell from the CSV.

    Handles both formats emitted by the data-generation pipeline:
      • Python list  : ``[-94.25, -66.88, -74.88, ...]``   (current CSV)
      • Semicolon    : ``-94.25;-66.88;-74.88;...``        (legacy)
    Missing values and NaN strings are replaced with *fill*.
    """
    if pd.isna(s):
        return np.full(max_len, fill, dtype=dtype)
    cleaned = re.sub(r"[\[\]]", "", str(s)).strip()
    parts   = re.split(r"[,;]", cleaned)
    vals: list[float] = []
    for p in parts[:max_len]:
        p = p.strip()
        if p == "" or p.lower() in ("nan", "none"):
            vals.append(fill)
        else:
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(fill)
    # Pad to max_len
    vals += [fill] * (max_len - len(vals))
    return np.array(vals, dtype=dtype)


def parse_nb_ids(s, max_len: int = K) -> list[int]:
    """Parse neighbour cell-ID array → list of ints (length max_len, 0-padded)."""
    if pd.isna(s):
        return [0] * max_len
    cleaned = re.sub(r"[\[\]]", "", str(s)).strip()
    parts   = re.split(r"[,;]", cleaned)
    ids: list[int] = []
    for p in parts[:max_len]:
        p = p.strip()
        try:
            ids.append(int(float(p)))
        except (ValueError, TypeError):
            ids.append(0)
    ids += [0] * (max_len - len(ids))
    return ids


def derive_ptr_label(serving_id: int, nb_ids: list[int]) -> int:
    """Find position of *serving_id* in the score-sorted *nb_ids* list.

    Label semantics
    ---------------
    The ``nb_cell_ids`` list is always emitted score-sorted with the optimal
    cell at index 0. Therefore ``optimal_cell_idx_in_k`` is always 0 (a known
    data artefact — documented in the raw CSV comments).

    The meaningful classification target is the **rank of the current serving
    cell** in the sorted neighbour list:
      0       → serving cell IS the best cell (no handover needed)
      1..K-1  → serving cell is at rank k  (handover gap = k positions)

    If the serving cell is not found in the neighbour list, default to 0
    (treat as no-handover — conservative fallback).
    """
    try:
        idx = nb_ids.index(int(serving_id))
        return min(idx, K - 1)
    except (ValueError, TypeError):
        return 0


# ── Feature extraction helpers ─────────────────────────────────────────────────

def _extract_cell_arrays(df: pd.DataFrame
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cell_tens, dist_arr, y_bin).

    cell_tens  : [N_rows, K, 4]  — (rsrp, sinr, load, score) per neighbour
                                   Row 0 = optimal cell (score-sorted by data gen)
    dist_arr   : [N_rows, K]     — distance in metres (9999 = not visible)
    y_bin      : [N_rows]        — binary HO flag (1 = HO needed, 0 = stay)
                                   True if serving_cell_id != optimal_cell_id
    """
    print("  Parsing neighbour arrays …", flush=True)
    rsrps  = np.stack(df["nb_rsrps"].apply(parse_nb_array).values)            # [N, K]
    sinrs  = np.stack(df["nb_sinrs"].apply(parse_nb_array).values)            # [N, K]
    loads  = np.stack(df["nb_loads"].apply(parse_nb_array).values)            # [N, K]
    scores = np.stack(df["nb_scores"].apply(
        lambda x: parse_nb_array(x, fill=0.0)).values)                        # [N, K]
    dists  = np.stack(df["nb_dists_m"].apply(
        lambda x: parse_nb_array(x, fill=9999.0)).values)                     # [N, K]

    # Binary HO flag: serving cell != optimal cell
    y_bin = (df["serving_cell_id"].values != df["optimal_cell_id"].values).astype(np.float32)

    print(f"  HO rate: {y_bin.mean():.3f}  ({int(y_bin.sum())} / {len(y_bin)})", flush=True)

    # Stack features → [N, K, 4]
    cell_tens = np.stack([rsrps, sinrs, loads, scores], axis=2).astype(np.float32)
    return cell_tens, dists.astype(np.float32), y_bin


# ── Window builder ─────────────────────────────────────────────────────────────

def _build_windows(df: pd.DataFrame,
                   cell_tens: np.ndarray,
                   dists: np.ndarray,
                   y_bin_row: np.ndarray,
                   rsrp_col: np.ndarray,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows per UE and apply spatial shuffling.

    Returns
    -------
    X        : [N_win, K, T, 4]  cell feature windows (K-axis shuffled)
    M        : [N_win, K]        visibility mask (K-axis shuffled)
    y        : [N_win]           pointer label (index of optimal cell after shuffle)
    y_bin    : [N_win]           binary HO flag (1 = serving not optimal)
    X_cells  : [N_win, T, K]    RSRP-only (for NB14, K-axis shuffled)
    X_glob   : [N_win, T, 1]    serving RSRP (for NB14)
    ue_ids   : [N_win]          UE identifier for split
    """
    all_X, all_M, all_y, all_y_bin = [], [], [], []
    all_Xcells, all_Xglob  = [], []
    all_ues                = []

    df_idx  = df.index.values          # original DataFrame indices
    row_map = {orig: pos for pos, orig in enumerate(df_idx)}

    for ue_id, grp in df.groupby("ue_id", sort=False):
        grp = grp.sort_values("timestamp")
        idx = grp.index.values         # original indices into df
        pos = np.array([row_map[i] for i in idx])   # positions in cell_tens

        N_grp = len(pos)
        if N_grp < T + 1:
            continue

        ct = cell_tens[pos]            # [N_grp, K, 4]
        di = dists[pos]               # [N_grp, K]
        yb = y_bin_row[pos]           # [N_grp]
        rs = rsrp_col[pos]            # [N_grp]  serving rsrp

        for t in range(T, N_grp):
            # Cell window [K, T, 4]  (legacy shape for NB01/05)
            X_w = ct[t - T:t].transpose(1, 0, 2)          # [K, T, 4]
            # Mask: cell visible if dist < 9999 at last timestep
            M_w = (di[t - 1] < 9999.0).astype(np.float32) # [K]
            
            # --- SHUFFLING LOGIC ---
            # Generate random permutation for the K neighbors
            # We apply the same permutation across the whole time window T
            # so the LSTM tracks a consistent neighbor trajectory.
            p = np.random.permutation(K)
            X_w_shuf = X_w[p, :, :]
            M_w_shuf = M_w[p]
            
            # The optimal cell is ALWAYS at index 0 before shuffling (data gen artifact).
            # We must track where index 0 moved to in the permutation to get the true label.
            y_shuf = int(np.where(p == 0)[0][0])
            
            # For NB14 shapes
            X_cells_w = ct[t - T:t, :, 0]          # [T, K]
            X_cells_w_shuf = X_cells_w[:, p]

            all_X.append(X_w_shuf)
            all_M.append(M_w_shuf)
            all_y.append(y_shuf)
            all_y_bin.append(float(yb[t]))
            all_ues.append(ue_id)

            all_Xcells.append(X_cells_w_shuf)
            all_Xglob.append(rs[t - T:t, np.newaxis])     # [T, 1]  serving RSRP

    X       = np.array(all_X,      dtype=np.float32)   # [N, K, T, 4]
    M       = np.array(all_M,      dtype=np.float32)   # [N, K]
    y       = np.array(all_y,      dtype=np.int32)     # [N]
    y_bin   = np.array(all_y_bin,  dtype=np.float32)   # [N]
    X_cells = np.array(all_Xcells, dtype=np.float32)   # [N, T, K]
    X_glob  = np.array(all_Xglob,  dtype=np.float32)   # [N, T, 1]
    ue_ids  = np.array(all_ues)

    return X, M, y, y_bin, X_cells, X_glob, ue_ids


# ── UE-level train / val / test split ─────────────────────────────────────────

def _split_ues(ue_ids: np.ndarray) -> tuple[set, set, set]:
    ue_list = np.unique(ue_ids)
    rng     = np.random.default_rng(SEED)
    rng.shuffle(ue_list)
    n_test  = max(1, int(len(ue_list) * TEST_FRAC))
    n_val   = max(1, int(len(ue_list) * VAL_FRAC))
    ue_test  = set(ue_list[:n_test])
    ue_val   = set(ue_list[n_test:n_test + n_val])
    ue_train = set(ue_list[n_test + n_val:])
    return ue_train, ue_val, ue_test


# ── Main preprocessing routine ─────────────────────────────────────────────────

def run(force: bool = False) -> None:
    proc_done = all(
        (PROC_DIR / f).exists()
        for f in ("train.npz", "val.npz", "test.npz", "scaler.pkl", "ue_split.json")
    )
    nb14_done = all(
        (NB_DIR / f).exists()
        for f in ("X_cells.npy", "X_glob.npy", "y.npy")
    )

    if proc_done and nb14_done and not force:
        print("✅  All artifacts already exist. Use --force to regenerate.")
        return

    if not RAW_CSV.exists():
        sys.exit(f"❌  Raw CSV not found: {RAW_CSV}")

    t0 = time.time()
    print(f"📂  Loading {RAW_CSV} …", flush=True)
    df = pd.read_csv(str(RAW_CSV), low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df.sort_values(["ue_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    {len(df):,} rows · {df['ue_id'].nunique()} UEs  ({time.time()-t0:.1f}s)")

    # Serving-cell RSRP column for NB14 X_glob
    rsrp_col = df["rsrp"].values.astype(np.float32)

    # ── Parse all cell arrays (once, vectorised) ───────────────────────────────
    cell_tens, dists, y_bin_row = _extract_cell_arrays(df)

    # ── Build sliding windows ─────────────────────────────────────────────────
    print("🔨  Building windows (T=%d, K=%d) …" % (T, K), flush=True)
    X, M, y, y_bin, X_cells, X_glob, ue_ids = _build_windows(
        df, cell_tens, dists, y_bin_row, rsrp_col)
    del cell_tens, dists, y_bin_row, rsrp_col
    print(f"    {len(y):,} windows total  ({time.time()-t0:.1f}s)")

    # ── UE-level split ─────────────────────────────────────────────────────────
    ue_train, ue_val, ue_test = _split_ues(ue_ids)
    idx_tr = np.where([g in ue_train for g in ue_ids])[0]
    idx_va = np.where([g in ue_val   for g in ue_ids])[0]
    idx_te = np.where([g in ue_test  for g in ue_ids])[0]
    print(f"    split  train={len(idx_tr):,}  val={len(idx_va):,}  test={len(idx_te):,}")

    # ── Fit scaler on TRAIN only (no leakage) ─────────────────────────────────
    _, K_, T_, F_ = X.shape
    scaler = StandardScaler()
    valid_mask_tr = (M[idx_tr] == 1.0)   # [N_tr, K]
    if valid_mask_tr.sum() == 0:
        print("  ⚠️  All-zero mask in train — falling back to global fit")
        scaler.fit(X[idx_tr].reshape(-1, F_))
    else:
        scaler.fit(X[idx_tr][valid_mask_tr].reshape(-1, F_))

    # Apply scaler to valid cells only
    X_n = X.copy()
    valid_all = (M == 1.0)
    if valid_all.sum() > 0:
        X_n[valid_all] = scaler.transform(
            X[valid_all].reshape(-1, F_)).reshape(-1, T_, F_)
    else:
        X_n = scaler.transform(X.reshape(-1, F_)).reshape(X.shape)

    # ── Save dataset/processed/ ───────────────────────────────────────────────
    os.makedirs(PROC_DIR, exist_ok=True)
    print(f"💾  Writing {PROC_DIR} …", flush=True)
    np.savez_compressed(PROC_DIR / "train.npz",
                        X=X_n[idx_tr], M=M[idx_tr], y=y[idx_tr], y_bin=y_bin[idx_tr])
    np.savez_compressed(PROC_DIR / "val.npz",
                        X=X_n[idx_va], M=M[idx_va], y=y[idx_va], y_bin=y_bin[idx_va])
    np.savez_compressed(PROC_DIR / "test.npz",
                        X=X_n[idx_te], M=M[idx_te], y=y[idx_te], y_bin=y_bin[idx_te])
    with open(PROC_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    ue_split = {
        "train_ues": sorted(str(u) for u in ue_train),
        "val_ues":   sorted(str(u) for u in ue_val),
        "test_ues":  sorted(str(u) for u in ue_test),
    }
    with open(PROC_DIR / "ue_split.json", "w") as f:
        json.dump(ue_split, f, indent=2)
    print(f"    ✅  train.npz  val.npz  test.npz  scaler.pkl  ue_split.json")

    # ── Save notebooks/modeling/ NB14 files ───────────────────────────────────
    print(f"💾  Writing NB14 files → {NB_DIR} …", flush=True)
    np.save(NB_DIR / "X_cells.npy", X_cells)   # [N, T, K]
    np.save(NB_DIR / "X_glob.npy",  X_glob)    # [N, T, 1]
    np.save(NB_DIR / "y.npy",       y)          # [N]
    print(f"    ✅  X_cells.npy {X_cells.shape}  "
          f"X_glob.npy {X_glob.shape}  "
          f"y.npy {y.shape}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n✅  Done in {elapsed:.1f}s")
    print(f"   X shape : {X.shape}  (N, K={K_}, T={T_}, F={F_})")
    print(f"   y unique : {np.unique(y, return_counts=True)}")
    print(f"   Scaler means (4 feats): {scaler.mean_.round(3)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if outputs already exist")
    args = ap.parse_args()
    run(force=args.force)
