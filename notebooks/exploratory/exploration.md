# Exploratory Data Analysis (EDA) Interpretation — Handover Prediction Dataset

# Overview

The exploratory analysis suggests that the simulated dataset captures realistic multi-factor handover dynamics in a cellular network environment.

The observed behaviors indicate that:
- handover decisions are influenced by multiple interacting factors,
- radio ambiguity exists between neighboring cells,
- mobility affects network stability,
- and cell load contributes to handover behavior without dominating the decision process.

Overall, the dataset appears suitable for:
- handover prediction,
- ambiguity-aware mobility management,
- load-aware cell selection,
- and AI-driven RAN optimization research.

---

# Part I — Dashboard Interpretation

## 1. Handover Class Distribution

The dataset is naturally imbalanced:
- approximately 79% of samples correspond to `no_handover`,
- while the remaining samples belong to different handover categories.

This is realistic because, in operational cellular systems:
- most UE states do not trigger handovers,
- and handover events remain relatively infrequent.

The imbalance therefore reflects plausible network behavior rather than a synthetic balanced dataset.

---

## 2. RSRP Distribution by Class

The RSRP distributions show that:
- handover-related samples generally occur at lower RSRP values,
- while non-handover samples tend to maintain stronger radio conditions.

This is consistent with real cellular mobility behavior where:
- degrading signal quality increases the probability of a handover event.

The overlap between distributions is also important:
- handovers are not triggered solely by a fixed RSRP threshold,
- indicating that the simulation includes multi-factor decision logic.

---

## 3. Hysteresis Margin Distribution

The hysteresis margin is strongly concentrated around 0 dB.

This behavior is meaningful because values near 0 dB typically correspond to:
- cell-edge regions,
- competitive neighboring cells,
- and ambiguous mobility situations.

Such regions are critical for:
- intelligent handover prediction,
- ambiguity detection,
- and ping-pong mitigation.

The presence of both positive and negative margins indicates realistic fluctuations in cell dominance.

---



## 4. Ping-Pong Rate vs Mobility Type

The analysis shows that:
- pedestrian users experience lower ping-pong rates,
- while high-speed mobility scenarios exhibit higher instability.

This behavior is coherent with real wireless systems because:
- fast-moving UEs cross cell boundaries more frequently,
- radio conditions change more rapidly,
- and handover timing becomes more difficult.

The observed mobility-related instability therefore supports the realism of the simulation.

---

## 5. Temporal Lag vs Configured TTT

The dashboard shows a significant difference between:
- configured `Time-To-Trigger (TTT)`,
- and observed handover lag.

This indicates that:
- handover conditions may persist over time before the actual event occurs,
- and that temporal dynamics are present in the dataset.

This is highly valuable because it introduces:
- temporal ambiguity,
- delayed transitions,
- and realistic non-instantaneous handover behavior.

Such properties are particularly useful for:
- sequential learning models,
- LSTM/Transformer architectures,
- and early handover anticipation research.

---

## 7. Overall Dashboard Assessment

The dashboard suggests that the simulator captures:
- realistic radio degradation,
- neighbor-cell competition,
- mobility-driven instability,
- temporal handover dynamics,
- ambiguity-aware conditions,
- and non-trivial decision boundaries.

The generated dataset therefore appears significantly more realistic than simplified threshold-based handover simulations.

---

# Part II — Cellular Load vs Handover Interpretation

## 1. Cell Load Distribution by Class

The `cell_load` distributions indicate that:
- handover classes tend to appear at slightly higher load levels,
- while `no_handover` samples are more concentrated at lower loads.

This behavior is realistic in load-aware mobility systems where:
- overloaded cells may encourage mobility toward neighboring cells.

However, the distributions still overlap considerably.

This is important because:
- cell load alone should not fully determine handover behavior,
- otherwise the problem would become artificially simple.

The overlap therefore indicates a more realistic multi-factor decision process.

---

## 2. Boxplot Analysis

The boxplots show:
- higher median load values for several handover categories,
- but also substantial variance within each class.

This means:
- handovers can still occur under moderate load,
- and stable connections may remain possible even under relatively high load conditions.

Such variability is expected in operational cellular networks where:
- radio quality,
- mobility,
- interference,
- and neighbor-cell conditions
all interact simultaneously.

---

## 3. Cell Load vs RSRP Scatter Plot

The scatter plot reveals:
- no perfectly separable regions,
- no obvious deterministic threshold,
- and significant overlap between handover and non-handover samples.

This is a strong indicator of realism.

The handover decision appears to depend jointly on:
- radio quality,
- cell load,
- mobility,
- and neighboring-cell competition.

The absence of artificial clustering suggests that:
- the simulator is not producing trivial decision patterns,
- and the machine learning task remains meaningful.

---

## 4. Discrete Vertical Bands in Cell Load

The vertical bands observed in the scatter plot likely indicate that:
- `cell_load` is updated discretely,
- quantized,
- or derived from UE occupancy levels.

This behavior is common in network simulations where:
- load evolves through discrete attachment/detachment events,
- or periodic scheduler updates.

Therefore, this pattern does not appear problematic.

---

# Final Assessment

The exploratory analysis indicates that the dataset:
- contains realistic radio and mobility relationships,
- includes meaningful ambiguity conditions,
- preserves non-trivial class overlap,
- and captures temporal as well as load-aware handover dynamics.

The generated simulation environment appears suitable for advanced research topics such as:
- AI-native mobility management,
- ambiguity-aware handover prediction,
- load-aware SON optimization,
- ping-pong mitigation,
- and future adaptive TTT/hysteresis optimization.