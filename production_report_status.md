# Production Report — Handover Target Selection (Status)

Date: 2026-05-24  
Repo: `/home/wassimmchichi/Downloads/Handover_projects`

This report summarizes what has been implemented so far to move from research notebooks to a production-oriented edge-AI handover decision pipeline, using the latest integral dataset:
`dataset/raw/handover_dataset.csv`.

---

## 0) Context and goals

Project goal:
- Predict the **best target cell** for the UE handover decision, early enough to be actionable before RLF.
- Deploy in a **5G Edge/MEC** architecture (not cloud), so inference + decisioning must fit tight timing budgets.

Key operational timing model:
- `T_total = T_measure + T_uplink + T_transport + T_prediction + T_decision + T_handover`
- The prediction must arrive early enough (before RLF), with scenario-dependent useful decision times.

---

## 1) Dataset audit (latest integral CSV)

File:
- `dataset/raw/handover_dataset.csv`

Critical dataset property (order leakage risk):
- `optimal_cell_idx_in_k` is **constant 0** for all rows.
- The neighbor lists are score-sorted such that the **optimal cell is always at index 0** in `nb_cell_ids`.

Implication:
- Any model trained directly to predict `optimal_cell_idx_in_k` without shuffling will learn trivial “always 0” behavior (inflated accuracy).
- **Mitigation**: shuffle the neighbor axis `K` per sample/window and map labels after shuffling (same idea used in `notebooks/modeling/01_temporal_deepset.ipynb`).

Split policy:
- We reuse UE-level splitting (`dataset/processed/ue_split.json`) to avoid time leakage across train/val/test.

---

## 2) Production constraints documentation (Task 1)

Created:
- `docs/production_constraints.md`

Contents:
- End-to-end edge-AI pipeline stages (UE measurement → gNB → MEC → inference → decision → HO prep/execution).
- Scenario timing targets and typical per-step latencies.
- Practical implications: latency targets, stability gating, robustness requirements, safe fallbacks.

---

## 3) Production pipeline (baseline “single-step” TemporalDeepSet) (Task 2)

Objective:
- Convert the best-performing notebook approach into reproducible scripts and modules.
- Ensure leakage-safe shuffling and UE-level split reuse.

Added modules:
- `src/production/temporal_deepset_data.py`
  - Builds/loads a cache for TemporalDeepSet training (K-neighbor shuffling per sample).
  - Produces arrays + scaler for reproducible runs.
- `src/production/temporal_deepset_model.py`
  - Keras model builder + loader (with custom pooling).
- `src/production/temporal_deepset_decision.py`
  - Production stability gate: **margin + Time-To-Trigger (TTT) + cooldown**.

Added scripts:
- `src/train_model.py` — train TemporalDeepSet (requires TensorFlow env).
- `src/evaluate_model.py` — evaluate and write a metrics JSON.
- `src/inference.py` — offline streaming inference + stability gating.

Validation run (in your `tda-handover` conda env):
- `conda run -n tda-handover python src/evaluate_model.py --label optimal --model models/best_temporal_deepset.keras`
- Produced: `metrics/temporal_deepset_eval_optimal.json`
- Observed: top-1 accuracy ≈ **0.9989** on the cached test split.

Notes:
- This “single-step” model predicts the best cell for the current decision instant using the last `T` samples.
- For strict edge timing (HO prep/execution/signaling), this is often not sufficient; multi-horizon is needed.

---

## 4) Robustness / stability / scalability plan + harnesses (Task 3)

Created:
- `docs/robustness_scalability_plan.md`

Focus:
- Treat notebooks 02–14 as patterns/harnesses (not as insights), because prior runs used deprecated data.
- Define robustness tests (noise, missing neighbors/timesteps, domain shift).
- Define stability tests (flip-rate, ping-pong) and gating policy (hysteresis/margin + TTT + cooldown).
- Define scalability tests (p50/p95/p99 single-sample inference, throughput under concurrency).

Added runnable validation utilities:
- `src/validation/robustness_tests.py`
  - Noise injection + neighbor-drop stress tests (reports top-1 degradation).
- `src/validation/benchmark_latency.py`
  - Single-sample inference latency benchmark (p50/p95/p99).

---

## 5) Explainability + fine-tuning guidance for ray tracing (Task 4)

Created:
- `docs/explainability_and_finetuning.md`

Includes:
- Explainability goals: why this cell, which features/timesteps/cells mattered, uncertainty.
- Practical methods:
  - score introspection (top-k + margins),
  - occlusion/ablation (edge-friendly),
  - integrated gradients / grad×input (offline),
  - calibration (temperature scaling).
- Fine-tuning strategy for ray tracing:
  - freeze encoder first, retrain head first,
  - progressively unfreeze,
  - small LR, early stopping, distillation/label smoothing/focal for noisy labels,
  - optional domain adapter blocks.

---

## 6) Notebook hygiene: update 02–14 safely (no misleading conclusions)

Added patch script:
- `scripts/patch_notebooks_for_new_data.py`

Applied change:
- Inserted a warning banner at the top of every notebook `notebooks/modeling/02..14` describing:
  - new dataset location,
  - `optimal_cell_idx_in_k == 0` / optimal-at-index-0 leakage,
  - recommendation to treat notebooks as robustness/stability/scalability harnesses,
  - pointers to production docs and leakage-safe pipelines.

---

## 7) Edge “production-style” inference notebooks (50 ms timestep)

Created:
- `notebooks/inference/01_edge_production_pipeline.ipynb`
  - Assumes **50 ms per row**.
  - Rolling-window per-UE streaming inference for TemporalDeepSet.
  - Applies stability gate (TTT/margin/cooldown).
  - Benchmarks model inference latency and compares to timing budgets.

---

## 8) Multi-step horizon (multi-horizon) production experiment (Transformer, no LSTM)

Rationale:
- You requested **multi-step horizon** prediction and prefer to avoid LSTMs here.
- Implemented a Transformer over the **time axis** (per cell, shared weights), then DeepSet pooling over cells.

Notebook experiment:
- `notebooks/modeling/15_multihorizon_transformer_production.ipynb`
  - Builds leakage-safe multi-horizon cache (`dataset/mh_cache/`).
  - Outputs probabilities with shape `(B, H, K)` for horizons `h=1..H`.
  - Neighbor-axis shuffling per sample is included to remove order leakage.

Matching edge streaming notebook (choose horizon for decision):
- `notebooks/inference/02_edge_production_multihorizon.ipynb`
  - Uses the multi-horizon model output.
  - Example: `USE_H = 3` → **150 ms ahead** decisions (given 50 ms per timestep).
  - Applies the same stability gate.

Script equivalents (for reproducibility / production training runs):
- `src/production/mh_transformer_data.py` — leakage-safe multi-horizon cache builder/loader.
- `src/production/mh_transformer_model.py` — MH Transformer + DeepSet model.
- `src/train_mh_transformer.py` — train multi-horizon model.
- `src/evaluate_mh_transformer.py` — per-horizon evaluation output to JSON.

---

## 9) Docker / requirements scaffolding

Added:
- `requirements-edge.txt` — minimal edge runtime deps (includes `tensorflow-cpu` placeholder).
- `Dockerfile` — basic container skeleton (can be tuned to your deployment baseline).

Updated usage docs:
- `README.md`

---

## 10) How to run (recommended commands)

Pre-step (ensures UE split + processed artifacts exist):
- `python src/preprocess.py`

### TemporalDeepSet (single-step)
- Train: `conda run -n tda-handover python src/train_model.py --label optimal`
- Eval: `conda run -n tda-handover python src/evaluate_model.py --label optimal --model models/best_temporal_deepset.keras`
- Offline streaming inference + gating: `conda run -n tda-handover python src/inference.py --ue-id 0`
- Robustness: `conda run -n tda-handover python src/validation/robustness_tests.py`
- Latency benchmark: `conda run -n tda-handover python src/validation/benchmark_latency.py`

### Multi-horizon Transformer (multi-step)
- Train: `conda run -n tda-handover python src/train_mh_transformer.py --label optimal --h 5`
- Eval: `conda run -n tda-handover python src/evaluate_mh_transformer.py --label optimal --model models/best_mh_transformer.keras`

---

## 11) Known issues / warnings observed

TensorFlow emits repeated warnings/errors at import time inside the conda env:
- `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`
- CUDA diagnostics warnings when GPU is not available/usable.

Despite this, model loading/evaluation completed successfully in the observed run.

---

## 12) Next actions (what’s left to finalize production)

1) Choose the training label for production:
   - `optimal_cell_id` (oracle “best cell”) vs `target_cell_id` (policy-executed target).
2) Pick the horizon `H` (and which `h` is used operationally) to satisfy:
   - signaling + HO prep + execution time, and RLF margin.
3) Add calibration (temperature scaling) so probability margins are meaningful for gating.
4) Run the scalability benchmark under expected concurrency (edge throughput / tail latency).
5) Fine-tune on ray-tracing data following `docs/explainability_and_finetuning.md`.

