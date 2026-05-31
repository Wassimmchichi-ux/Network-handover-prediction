import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Constants
K = 10
T = 25
VAL_UE_FRAC = 0.15
TEST_UE_FRAC = 0.15
SEED = 42

def parse_nb_array(s, dtype=np.float32, max_len=K, fill=0.0):
    """Parse neighbour arrays from CSV.

    Handles both formats produced by the data pipeline:
      - Python list:  [-94.25, -66.88, -74.88, ...]   (current CSV)
      - Semicolon:    -94.25;-66.88;-74.88;...          (legacy)
    """
    if pd.isna(s):
        return np.full(max_len, fill, dtype=dtype)
    # Strip surrounding brackets / whitespace then split on comma OR semicolon
    cleaned = re.sub(r'[\[\]]', '', str(s)).strip()
    parts   = re.split(r'[,;]', cleaned)
    vals    = []
    for p in parts[:max_len]:
        p = p.strip()
        if p == '' or p.lower() in ('nan', 'none'):
            vals.append(fill)
        else:
            try:   vals.append(float(p))
            except: vals.append(fill)
    vals += [fill] * (max_len - len(vals))
    return np.array(vals, dtype=dtype)

def load_and_preprocess_data(root_dir=None, *, return_y_bin: bool = False):
    if root_dir is None:
        root_dir = Path("../../").resolve()
    else:
        root_dir = Path(root_dir)
        
    data_dir = root_dir / "dataset" / "processed"
    raw_path = root_dir / "dataset" / "raw" / "handover_dataset.csv"
    
    # Fast path: load from disk if exists
    if (data_dir / "train.npz").exists():
        print(f"Loading cached dataset from {data_dir}...")
        tr = np.load(str(data_dir / "train.npz"))
        va = np.load(str(data_dir / "val.npz"))
        te = np.load(str(data_dir / "test.npz"))
        
        with open(str(data_dir / "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
            
        y_bin_tr_c = tr["y_bin"].astype(np.float32) if "y_bin" in tr else (tr["y"] > 0).astype(np.float32)
        y_bin_va_c = va["y_bin"].astype(np.float32) if "y_bin" in va else (va["y"] > 0).astype(np.float32)
        y_bin_te_c = te["y_bin"].astype(np.float32) if "y_bin" in te else (te["y"] > 0).astype(np.float32)

        X_tr = tr["X"].astype(np.float32); M_tr = tr["M"].astype(np.float32); y_tr = tr["y"].astype(np.int32)
        X_va = va["X"].astype(np.float32); M_va = va["M"].astype(np.float32); y_va = va["y"].astype(np.int32)
        X_te = te["X"].astype(np.float32); M_te = te["M"].astype(np.float32); y_te = te["y"].astype(np.int32)

        # Backward-compatible default: return the original 10-tuple used by early notebooks.
        if not return_y_bin:
            return (X_tr, M_tr, y_tr,
                    X_va, M_va, y_va,
                    X_te, M_te, y_te,
                    scaler)

        return (X_tr, M_tr, y_tr, y_bin_tr_c,
                X_va, M_va, y_va, y_bin_va_c,
                X_te, M_te, y_te, y_bin_te_c,
                scaler)

    print(f"Parsing raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    
    print("Parsing neighbour arrays...")
    df["nb_rsrps_arr"] = df["nb_rsrps"].apply(parse_nb_array)
    df["nb_sinrs_arr"] = df["nb_sinrs"].apply(parse_nb_array)
    df["nb_loads_arr"] = df["nb_loads"].apply(parse_nb_array)
    df["nb_dists_arr"] = df["nb_dists_m"].apply(lambda x: parse_nb_array(x, fill=9999.0))
    df["ptr_label"]    = df["optimal_cell_idx_in_k"].clip(0, K-1).astype(int)
    
    df.sort_values(["ue_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    cell_arrays = ["nb_rsrps_arr", "nb_sinrs_arr", "nb_loads_arr", "nb_dists_arr"]
    
    print("Building sequences...")
    all_X, all_M, all_y, all_y_bin, groups = [], [], [], [], []
    
    for ue_id, grp in df.groupby("ue_id", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        N_grp = len(grp)
        if N_grp < T + 1:
            continue
            
        cell_tens = np.stack(
            [np.stack(grp[ca].values, axis=0) for ca in cell_arrays],
            axis=2) # [N, K, F]
            
        for t in range(T, N_grp):
            X_w = cell_tens[t-T : t] # [T, K, F]
            # Transpose to [K, T, F] for legacy models
            X_w = X_w.transpose(1, 0, 2)
            
            # Mask based on distance not being 9999.0
            M_w = (X_w[:, -1, 3] != 9999.0).astype(np.float32)
            
            y_w = int(grp["ptr_label"].iloc[t])
            
            # Binary HO label: 1 = handover needed (optimal != current serving cell)
            # optimal_is_current==0 means the best cell differs from the serving cell
            y_bin_w = float(grp["optimal_is_current"].iloc[t] == 0)
            
            all_X.append(X_w)
            all_M.append(M_w)
            all_y.append(y_w)
            all_y_bin.append(y_bin_w)
            groups.append(ue_id)
            
    X = np.array(all_X, dtype=np.float32)
    M = np.array(all_M, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    y_bin = np.array(all_y_bin, dtype=np.float32)
    groups = np.array(groups)
    print(f"HO rate in full dataset: {y_bin.mean():.3f} ({y_bin.sum():.0f} / {len(y_bin)})")
    
    print(f"Total sequences: {len(y)}")
    
    # Split
    ue_list = np.unique(groups)
    rng = np.random.default_rng(SEED)
    rng.shuffle(ue_list)

    n_test = max(1, int(len(ue_list) * TEST_UE_FRAC))
    n_val  = max(1, int(len(ue_list) * VAL_UE_FRAC))

    ue_test  = set(ue_list[:n_test])
    ue_val   = set(ue_list[n_test : n_test + n_val])
    ue_train = set(ue_list[n_test + n_val:])

    idx_tr = np.where([g in ue_train for g in groups])[0]
    idx_va = np.where([g in ue_val   for g in groups])[0]
    idx_te = np.where([g in ue_test  for g in groups])[0]
    
    # Normalise [N, K, T, F] -> fit on train set valid cells
    _, K_, T_, F_ = X.shape
    
    scaler = StandardScaler()
    # Mask [N, K]
    valid_mask_tr = (M[idx_tr] == 1.0)
    print(f"X.shape: {X.shape}, M.shape: {M.shape}")
    print(f"idx_tr size: {len(idx_tr)}, valid_mask_tr sum: {valid_mask_tr.sum()}")
    
    if valid_mask_tr.sum() == 0:
        print("WARNING: valid_mask_tr sum is 0! Using all cells for scaling fallback.")
        scaler.fit(X[idx_tr].reshape(-1, F_))
    else:
        scaler.fit(X[idx_tr][valid_mask_tr].reshape(-1, F_))
    
    X_n = X.copy()
    valid_mask_all = (M == 1.0)
    if valid_mask_all.sum() > 0:
        X_n[valid_mask_all] = scaler.transform(X[valid_mask_all].reshape(-1, F_)).reshape(-1, T_, F_)
    else:
        X_n = scaler.transform(X.reshape(-1, F_)).reshape(X.shape)
    
    os.makedirs(data_dir, exist_ok=True)

    print("Saving .npz files...")
    np.savez_compressed(data_dir / "train.npz", X=X_n[idx_tr], M=M[idx_tr], y=y[idx_tr], y_bin=y_bin[idx_tr])
    np.savez_compressed(data_dir / "val.npz",   X=X_n[idx_va], M=M[idx_va], y=y[idx_va], y_bin=y_bin[idx_va])
    np.savez_compressed(data_dir / "test.npz",  X=X_n[idx_te], M=M[idx_te], y=y[idx_te], y_bin=y_bin[idx_te])

    with open(data_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save UE split so other notebooks (NB02, NB03, etc.) share the same split
    ue_split = {
        "train_ues": [str(u) for u in ue_train],
        "val_ues":   [str(u) for u in ue_val],
        "test_ues":  [str(u) for u in ue_test],
    }
    with open(data_dir / "ue_split.json", "w") as f:
        json.dump(ue_split, f, indent=2)

    print("Data successfully generated.")
    X_tr = X_n[idx_tr]; M_tr = M[idx_tr]; y_tr = y[idx_tr]; yb_tr = y_bin[idx_tr]
    X_va = X_n[idx_va]; M_va = M[idx_va]; y_va = y[idx_va]; yb_va = y_bin[idx_va]
    X_te = X_n[idx_te]; M_te = M[idx_te]; y_te = y[idx_te]; yb_te = y_bin[idx_te]

    if not return_y_bin:
        return (X_tr, M_tr, y_tr,
                X_va, M_va, y_va,
                X_te, M_te, y_te,
                scaler)

    return (X_tr, M_tr, y_tr, yb_tr,
            X_va, M_va, y_va, yb_va,
            X_te, M_te, y_te, yb_te,
            scaler)

if __name__ == "__main__":
    load_and_preprocess_data(root_dir=".")
