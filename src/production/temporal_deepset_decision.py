from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DecisionConfig:
    ttt_steps: int = 3
    margin_threshold: float = 0.15
    cooldown_steps: int = 5


@dataclass
class DecisionState:
    candidate_cell_id: Optional[int] = None
    confirm_count: int = 0
    cooldown_left: int = 0


@dataclass(frozen=True)
class HandoverDecision:
    action: str  # "stay" | "handover"
    target_cell_id: int
    best_prob: float
    serving_prob: float
    margin: float
    ttt_confirmations: int


class TemporalDecisionGate:
    """Stability layer: margin gating + time-to-trigger + cooldown."""

    def __init__(self, cfg: DecisionConfig):
        self._cfg = cfg
        self._state_by_ue: Dict[str, DecisionState] = {}

    def update(
        self,
        *,
        ue_id: str,
        serving_cell_id: int,
        nb_cell_ids: list[int],
        cell_probs: np.ndarray,
    ) -> HandoverDecision:
        if cell_probs.ndim != 1:
            raise ValueError(f"cell_probs must be 1D, got shape={cell_probs.shape}")

        st = self._state_by_ue.setdefault(ue_id, DecisionState())

        if st.cooldown_left > 0:
            st.cooldown_left -= 1
            return HandoverDecision(
                action="stay",
                target_cell_id=serving_cell_id,
                best_prob=float(cell_probs.max()),
                serving_prob=float(cell_probs[nb_cell_ids.index(serving_cell_id)]) if serving_cell_id in nb_cell_ids else 0.0,
                margin=0.0,
                ttt_confirmations=0,
            )

        best_idx = int(np.argmax(cell_probs))
        best_prob = float(cell_probs[best_idx])
        best_cell_id = int(nb_cell_ids[best_idx]) if best_idx < len(nb_cell_ids) else int(serving_cell_id)

        serving_prob = float(cell_probs[nb_cell_ids.index(serving_cell_id)]) if serving_cell_id in nb_cell_ids else 0.0
        margin = best_prob - serving_prob

        # Candidate selection: only consider switching if margin is strong enough.
        candidate = best_cell_id if (best_cell_id != serving_cell_id and margin >= self._cfg.margin_threshold) else serving_cell_id

        if candidate == serving_cell_id:
            st.candidate_cell_id = None
            st.confirm_count = 0
            return HandoverDecision(
                action="stay",
                target_cell_id=serving_cell_id,
                best_prob=best_prob,
                serving_prob=serving_prob,
                margin=margin,
                ttt_confirmations=0,
            )

        # Time-to-trigger accumulation
        if st.candidate_cell_id == candidate:
            st.confirm_count += 1
        else:
            st.candidate_cell_id = candidate
            st.confirm_count = 1

        if st.confirm_count >= self._cfg.ttt_steps:
            st.cooldown_left = self._cfg.cooldown_steps
            st.candidate_cell_id = None
            st.confirm_count = 0
            return HandoverDecision(
                action="handover",
                target_cell_id=candidate,
                best_prob=best_prob,
                serving_prob=serving_prob,
                margin=margin,
                ttt_confirmations=self._cfg.ttt_steps,
            )

        return HandoverDecision(
            action="stay",
            target_cell_id=serving_cell_id,
            best_prob=best_prob,
            serving_prob=serving_prob,
            margin=margin,
            ttt_confirmations=st.confirm_count,
        )

