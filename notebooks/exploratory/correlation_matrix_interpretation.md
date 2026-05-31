# Correlation Matrix Interpretation — Handover Ambiguity Features

## Overview

The correlation matrix appears globally consistent and realistic for a cellular handover prediction simulation.

The observed relationships suggest that the dataset captures:
- radio signal behavior,
- mobility effects,
- neighbor-cell competition,
- multi-factor handover dynamics,
- and ambiguity-related features.

The absence of extreme or unrealistic correlations is a positive indicator of simulation quality.

---

# Key Interpretations

## 1. Moderate Correlation Between RSRP and SINR

Observed correlation:
- `RSRP ↔ SINR ≈ 0.49`

This is realistic in wireless systems.

Although SINR depends partially on received signal power (RSRP), it is also strongly affected by:
- inter-cell interference,
- network load,
- scheduling,
- neighboring cells,
- noise,
- and radio geometry.

Therefore:
- a UE may have strong RSRP but poor SINR due to interference,
- or moderate RSRP with relatively good SINR.

A moderate correlation (~0.5) is generally expected in realistic cellular environments.

Very high correlations (>0.9) would usually indicate an oversimplified radio model.

---

## 2. Strong Correlation Between `rsrp_gap_top2` and `score_gap_top2`

Observed correlation:
- `rsrp_gap_top2 ↔ score_gap_top2 ≈ 0.82`

This behavior is expected.

The cell selection score appears to depend significantly on radio quality metrics such as:
- RSRP,
- SINR,
- and possibly load balancing.

As a result:
- when the best candidate cell strongly dominates in RSRP,
- it also tends to dominate in the global cell score.

This indicates internal consistency in the scoring mechanism.

---

# 3. SINR Variability Among Neighbors

Observed correlation:
- `sinr_std_neighbors ↔ SINR ≈ 0.65`

This is also realistic.

High SINR variance among neighboring cells often reflects:
- unstable radio conditions,
- cell-edge regions,
- heterogeneous interference conditions,
- or ambiguous handover zones.

This feature is therefore relevant for ambiguity-aware handover prediction.

---

# 4. Handover Target (`is_handover`) Relationships

Observed correlations:
- `SINR ↔ is_handover ≈ -0.27`
- `RSRP ↔ is_handover ≈ -0.20`
- `hysteresis_margin ↔ is_handover ≈ -0.24`

These moderate negative correlations are coherent.

Handover events typically occur when:
- serving-cell quality decreases,
- neighboring cells become more attractive,
- or radio conditions become unstable.

The fact that no single feature completely dominates the handover decision suggests that the dataset reflects a multi-factor decision process, which is closer to real cellular systems.

---

# 5. Mobility Features

Observed correlations:
- `speed ↔ RSRP ≈ -0.12`
- `speed ↔ SINR ≈ -0.14`

These weak correlations are realistic.

User speed usually affects:
- handover frequency,
- link stability,
- channel variability,
- and ping-pong probability,

rather than directly controlling instantaneous radio power.

Therefore, low direct correlations between speed and radio metrics are expected.

---

# 6. Constant Features: TTT and Hysteresis

The features:
- `time_to_trigger`
- `hysteresis`

show near-zero or undefined correlations because they are currently constant in the simulation.

This is mathematically expected since Pearson correlation requires non-zero variance.

At the current stage of the project:
- TTT and hysteresis are fixed parameters,
- and adaptive optimization of these parameters is outside the current prediction objective.

This behavior is therefore normal and does not indicate an issue in the dataset.

---

# 7. Overall Assessment

The correlation matrix suggests that the simulation environment produces:
- realistic radio relationships,
- plausible mobility interactions,
- and non-trivial handover dynamics.

The dataset appears suitable for:
- handover prediction,
- ambiguity-aware mobility management,
- intelligent cell selection,
- and future adaptive handover optimization research.

The current feature set already captures multiple important dimensions of cellular decision-making:
- radio quality,
- mobility,
- neighboring-cell competition,
- load balancing,
- and ambiguity estimation.

This is significantly richer than simplified RSRP-only handover datasets.