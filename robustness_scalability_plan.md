# Robustness, Stability, and Scalability Plan

The notebooks `notebooks/modeling/02..14` were run on deprecated / non-integral data, so their *metrics* should not be reused as conclusions. They are still useful as **test patterns** and **stress harnesses** for the current dataset.

This document defines the production-facing validation suite to keep the handover model safe under edge constraints.

## 1) Robustness tests (data / RF perturbations)

Run on the **latest** dataset (`dataset/raw/handover_dataset.csv`) and the current production model.

**Measurement noise**
- Add realistic noise to RSRP/SINR (e.g., ±1–3 dB) and re-evaluate accuracy/top-k.
- Validate that small noise does not cause large decision flips.

**Missing data / partial neighbor list**
- Randomly drop neighbors (simulate measurement gaps / filtering) and measure degradation.
- Randomly drop timesteps inside the observation window (simulate packet loss / jitter).

**Mask correctness**
- Verify that padded cells (missing neighbors) never get selected as target.

**Domain shift**
- Split evaluation by scenario/mobility type (pedestrian vs vehicle vs train-like) to catch regime-specific failures.

## 2) Stability tests (handover ping-pong / oscillation)

Production must include decision stabilization:
- Hysteresis on confidence margin (only switch if `p(best) - p(serving)` exceeds threshold).
- Time-to-trigger (TTT): require N consecutive timesteps predicting the same target.
- Cooldown: once a handover is triggered, suppress new decisions for a short window.

Stability metrics to track:
- Flip rate per UE (how often predicted target changes).
- Ping-pong rate (A→B→A patterns).
- Average “dwell time” on a target.

## 3) Scalability tests (edge throughput)

We need to measure:
- p50/p95 inference latency for **single-sample** inference (streaming).
- Throughput at target UE counts (e.g., 1k–50k UEs depending on deployment).
- Tail latency under concurrency (batching vs no batching).

Practical knobs:
- Quantization (TFLite int8 / fp16) where possible.
- Model size (LSTM units, embedding dims).
- Feature count and observation window length.

## 4) How notebooks map to the validation suite

Use these notebooks as templates (logic, not metrics):
- `02_set_transformer.ipynb`: permutation invariance checks; variable neighbor count stress.
- `03_mtl_transformer*.ipynb`: multi-task loss weighting; calibration and auxiliary heads for stability.
- `06_temporal_smoothing.ipynb`: post-processing patterns (TTT / smoothing) for decision stability.
- `11_6G_Handover_Ablation_Study.ipynb`: ablation protocol to confirm what features matter.
- `14_Cascade_Handover.ipynb`: two-stage “trigger + target” design (reduce unnecessary HOs).

