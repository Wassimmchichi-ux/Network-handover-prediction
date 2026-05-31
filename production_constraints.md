# Production Constraints (Edge-AI Handover)

This project predicts the best target cell for an upcoming handover decision, with the model running on a **5G edge (MEC) server**.

## End-to-end timing budget

Operational pipeline:
1. UE measures radio metrics (RSRP/RSRQ/SINR, etc.)
2. Measurements sent to serving gNB
3. Data forwarded to edge/MEC
4. ML inference runs
5. Handover decision returned
6. HO preparation + execution

Total decision latency:

`T_total = T_measure + T_uplink + T_transport + T_prediction + T_decision + T_handover`

The decision must arrive early enough **before Radio Link Failure (RLF)**; otherwise the handover is either too late or becomes a “recovery” procedure.

## Max useful decision time (rule-of-thumb)

| Scenario | Max useful decision time |
|---|---|
| Pedestrian | 100–500 ms |
| Urban vehicle | 50–150 ms |
| Highway | 20–80 ms |
| High-speed train | < 20–50 ms |

These values bound the **latest** moment a target-cell decision is still actionable.

## Realistic 5G edge-AI latency (typical)

| Step | Typical delay |
|---|---|
| UE measurements | 5–40 ms |
| UE → gNB | 1–10 ms |
| gNB → edge server (MEC) | 1–5 ms |
| ML inference | 1–20 ms |
| Decision signaling | 5–20 ms |
| Handover execution | 20–60 ms |

Expected total:
| Architecture | Total latency |
|---|---|
| Edge AI (MEC) | 20–80 ms |
| Cloud AI | 100–300+ ms |

## What this implies for the model

**Latency target**
- Budget `T_prediction` at **≤ 5–10 ms p95** on edge CPU (or ≤ 1–5 ms p95 on edge GPU), to stay within the 20–80 ms end-to-end window.

**Input availability**
- The model must use features that are available at the time of the decision (from UE measurement reports + gNB context).
- Avoid “oracle-only” features that leak the outcome (e.g., anything computed using future throughput or future cell choice).

**Decision stability**
- “Best cell” prediction alone can cause **ping-pong** if decisions flip across adjacent timesteps.
- Production should include a *stability layer* (e.g., hysteresis, time-to-trigger, confidence margin, and/or N-consecutive confirmations).

**Robustness**
- Must tolerate missing neighbors, measurement noise (RSRP/SINR), and variable neighbor counts.
- Must degrade safely: if the model is uncertain, default to “stay” or follow a conservative rule-based policy.

**Operational constraints**
- Support per-UE streaming inference (single-sample inference).
- Support high throughput (many UEs) while maintaining per-request tail latency.

