
# 5G-to-6G Multi-Horizon Handover & Cell Selection Prediction Engine

An intelligent, deep-learning-powered **Multi-Horizon Handover (HO) and Cell Selection Prediction Engine** designed for Next-Generation Radio Access Networks (NG-RAN). Running containerized on 5G New Radio (NR) servers, this engine proactively manages user equipment (UE) transitions across **10 heterogeneous cells** (including ground macro-stations and dynamic aerial drone base stations).

Optimized for **5G Non-Standalone (NSA)** deployments, this framework shifts the networking paradigm from reactive event-triggered handovers to predictive, multi-horizon candidate pool pre-allocation—a key enabler for early 6G Ultra-Reliable Low-Latency Communication (URLLC).

---

## 🚀 Key Performance Milestones

The prediction engine outputs a **5-step multi-horizon projection** into the future. Sampling telemetry every **50ms**, the model accurately predicts cell suitability **250ms** ahead of time.

| Scenario / Topology | Lookahead Horizon | Top-1 Accuracy | Top-3 Accuracy | Operational Application |
| --- | --- | --- | --- | --- |
| **Perfect Hexagonal Topology** | 1-Step (50ms) | **98.0%** | — | Theoretical validation & tracking baseline |
| **Realistic Urban Micro (UMi) + Drones** | 5-Steps (250ms) | **56.0%** | **85.0%** | **Production Target** for Conditional Handover (CHO) |

### 📊 Why 85% Top-3 Accuracy Matters for Telco Deployments

Predicting the absolute single best cell (Top-1) 250ms in advance under complex urban topologies is highly volatile due to fast-fading and sudden shadow blockages. However, **3GPP Release 16/17 Conditional Handover (CHO)** allows the gNodeB to prepare a pool of target candidates.

Achieving an **85% Top-3 Accuracy** means that 85% of the time, the ideal target cell is already prepared. The NR server can pre-allocate random-access channel (RACH) resources and sync context vectors *before* link degradation occurs, dropping Handover Failures (HOF) to near-zero.

---

## 🛠️ Advanced Simulation Pipeline

To bridge the gap between theoretical math and real-world field deployment, this model was trained and fine-tuned using a high-fidelity hybrid pipeline:

```text
[ ns-3 Framework ] ---------> Microscopic mobility & macro UE trajectories
       |
       v
[ Sionna (Python) ] --------> Differentiable PHY layer & 3GPP UMi channel profiles
       ^
       | (Fine-Tuning Layer via Custom XML Scene Scripts)
[ Blender 3D + CFC ] -------> Precision urban geometry & Ray-Tracing data generation

```

1. **Mobility (ns-3):** Packet-level network simulation and historical UE trajectories.
2. **Aerial Nodes:** Integrated Unmanned Aerial Vehicles (UAVs) running a **Load-Aware Strategy** to dynamically adjust cell selection boundaries according to live ground density.
3. **Physical Layer (Sionna):** Link-level tensor simulations running 3GPP Urban Micro (UMi) profiles.
4. **Ray-Tracing Fine-Tuning:** Urban maps were modeled using **Computational Fluid Dynamics (CFC) meshes in Blender**, exported via XML to map multi-path propagation, diffractions, and site-specific blockages accurately into Sionna.

---




---

## ⚡ Quick Start

### Prerequisites

* Python 3.10+
* NVIDIA GPU (CUDA 11.8+ required for Sionna Ray-Tracing acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Wassimmchichi-ux/Network-handover-prediction.git

```


2. Install Python dependencies:
```bash
pip install -r requirements.txt

```


3. Run a baseline 1-step prediction evaluation:
```bash
python src/inference/evaluate.py --topology hexagonal --steps 1

```


4. Run the production-spec 5-step multi-horizon simulation:
```bash
python src/inference/evaluate.py --topology umi_drones --steps 5

```



---


## Production pipeline
``Note : serve just as a benchmark``

This repository trains and evaluates handover target-cell predictors on:
`dataset/raw/handover_dataset.csv`

### Production docs
- `docs/production_constraints.md`
- `docs/robustness_scalability_plan.md`
- `docs/explainability_and_finetuning.md`

### Reproducible scripts (TemporalDeepSet)

These scripts follow the same core approach as `notebooks/modeling/01_temporal_deepset.ipynb`:
shuffle neighbor order per sample to prevent order leakage.

- Train: `python src/train_model.py --label optimal`
- Evaluate: `python src/evaluate_model.py --label optimal --model models/best_temporal_deepset.keras`
- Offline inference + stability gating: `python src/inference.py --ue-id 0`
- Robustness: `python src/validation/robustness_tests.py`
- Latency benchmark: `python src/validation/benchmark_latency.py`

Note: training/inference requires TensorFlow (`tensorflow-cpu` on edge CPU).



## 📖 Full Documentation

For detailed guides on how to export the Blender CFC meshes, configure the load-aware drone strategy, or interface the containerized ONNX model with a live gNodeB Centralized Unit Control Plane (CU-CP), please visit our comprehensive documentation portal:

👉 **[Read the Docs Configuration & Deployment Manual](https://network-handover-prediction.readthedocs.io/)**

---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
## Handover target-cell selection
This serve as a benchmark for the dvc pipelines in the production.
