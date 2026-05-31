# Overview: Time-Series Based Handover Cell Selection for 5G-to-6G Migration

## 1. Introduction

The transition from 5G to 6G requires a paradigm shift in how we manage User Equipment (UE) mobility. As network density and frequency spectrum complexity increase, traditional reactive handover (HO) mechanisms based on instantaneous signal thresholds are becoming obsolete. This study focuses on a **proactive paradigm**, where handover cell selection is driven by **time-series prediction of network indicators**, enabling the network to anticipate movement patterns and channel quality fluctuations before they occur.

---

## 2. The Core Problem: Beyond Threshold-Based Handovers

Current 5G handover protocols rely on the "A3 event" (RSRP-based triggering). This approach is fundamentally limited in a 6G context for the following reasons:

* **Reactive Latency:** Waiting for signal degradation to cross a threshold creates a "dead zone" of connectivity, incompatible with 6G’s strict URLLC (Ultra-Reliable Low-Latency Communication) requirements.
* **Signal Instability:** High-frequency 6G bands (mmWave/THz) are highly susceptible to blockage and rapid fading. Instantaneous values (current RSRP) are poor predictors of signal viability just 500ms into the future.
* **Oscillation Risks:** Relying on current snapshots leads to frequent "ping-pong" handovers, as the network fails to distinguish between transient interference and genuine mobility-driven signal loss.

---

## 3. The Research Focus: Time-Series Predictive Decision Making

Our research shifts the focus from **"What is the signal now?"** to **"What will the indicators be at time $t + \Delta t$?"**

By treating network indicators as continuous time-series data, we can leverage temporal dependencies to make proactive handover decisions.

### Key Indicators for Prediction:

* **Channel State Information (CSI):** Predicting future fading patterns and SNR.
* **Reference Signal Received Power (RSRP):** Forecasting long-term trends to avoid HOs triggered by momentary shadowing.
* **Network Load/Traffic Density:** Predicting future cell congestion to perform load-balancing-aware handovers.
* **UE Trajectory/Velocity:** Using historical coordinate data to estimate the next likely target cells.

---

## 4. Proposed Framework Architecture

The proposed system utilizes a predictive pipeline to inform the selection engine.

### The Decision-Making Workflow:

1. **Data Collection:** Continuous streaming of temporal KPI data from the UE and gNB.
2. **Temporal Feature Engineering:** Normalizing and windowing data to capture trends, seasonality, and mobility patterns.
3. **Predictive Modeling:** Utilizing Deep Learning architectures (e.g., LSTMs, Transformers, or Temporal Convolutional Networks) to forecast indicator values.
4. **Proactive Handover Logic:** A decision engine that selects the target cell by comparing the *predicted* states of neighboring cells at time $t + \Delta t$, rather than their current states.

---

## 5. Mathematical Formulation

The objective is to maximize a utility function $U$ based on predicted indicators $I$:

$$
P_{HO} = \arg\max_{c \in C} \mathbb{E}\left[ U\left(\hat{I}_c(t+\Delta t)\right) \right]
$$

**Where:**

- $\hat{I}_c(t+\Delta t)$ is the predicted indicator vector (RSRP, Load, Throughput) for cell $c$ at the near-future time step.

- $C$ is the set of candidate neighbor cells.

- $U(\cdot)$ is the multi-objective utility function, which penalizes:
  - high handover frequency  
  - low predicted SNR

---

## 6. Strategic Research Goals

* **Anticipatory Handover:** Initiate HO preparation *before* the current signal drops below the critical threshold.
* **Ping-Pong Mitigation:** Filter out noise and transient dips in the time-series data to reduce redundant HO signaling.
* **Seamless Transition:** Ensure target cell capacity is reserved based on predicted arrival, preventing congestion-related drops during the handover execution.
