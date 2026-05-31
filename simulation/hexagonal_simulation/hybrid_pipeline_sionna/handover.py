from __future__ import annotations

from .config import SimulationConfig
from .models import Cell, HandoverDecision, MeasurementSnapshot, UEState


HO_LABELS = {
    0: "no_handover",
    1: "intra_freq_ho",
    2: "inter_freq_ho",
    3: "inter_rat_ho",
    4: "macro_to_drone_ho",
    5: "drone_to_macro_ho",
    6: "emergency_ho",
}


class HandoverController:
    def __init__(self, config: SimulationConfig, cells_by_id: dict[int, Cell]) -> None:
        self.config = config
        self.cells_by_id = cells_by_id
        self.total_handovers = 0
        self.total_ping_pong = 0
        self.total_rlf = 0

    def serving_map(self, ues: list[UEState]) -> dict[str, int | None]:
        return {ue.ue_id: ue.runtime.serving_cell_id for ue in ues}

    def evaluate_all(
        self,
        timestamp_s: float,
        ues: list[UEState],
        snapshots: dict[str, MeasurementSnapshot],
    ) -> dict[str, HandoverDecision]:
        decisions: dict[str, HandoverDecision] = {}
        for ue in ues:
            decisions[ue.ue_id] = self._evaluate_one(timestamp_s, ue, snapshots[ue.ue_id])
        return decisions

    def _evaluate_one(self, timestamp_s: float, ue: UEState, snapshot: MeasurementSnapshot) -> HandoverDecision:
        runtime = ue.runtime
        if runtime.anchor_cell_id is None:
            runtime.anchor_cell_id = snapshot.best_lte_id or snapshot.best_cell().cell_id
        if runtime.secondary_cell_id is None and snapshot.best_nr_id is not None:
            nr_meas = snapshot.by_cell_id[snapshot.best_nr_id]
            if self._x2_allowed(runtime.anchor_cell_id, snapshot.best_nr_id) and nr_meas.rsrp_dbm >= self.config.a4_absolute_threshold_dbm:
                runtime.prepared_target_id = snapshot.best_nr_id
                runtime.prepared_since_s = timestamp_s
                runtime.secondary_cell_id = snapshot.best_nr_id
        runtime.serving_cell_id = runtime.secondary_cell_id or runtime.anchor_cell_id or snapshot.best_cell().cell_id

        best_nr = snapshot.by_cell_id.get(snapshot.best_nr_id) if snapshot.best_nr_id is not None else None
        best_lte = snapshot.by_cell_id.get(snapshot.best_lte_id) if snapshot.best_lte_id is not None else None
        serving = snapshot.by_cell_id.get(runtime.serving_cell_id, snapshot.best_cell())

        prepared_candidate_id = None
        if best_nr is not None and best_nr.cell_id != runtime.secondary_cell_id:
            if self._x2_allowed(runtime.anchor_cell_id, best_nr.cell_id) and best_nr.rsrp_dbm >= self.config.a4_absolute_threshold_dbm:
                prepared_candidate_id = best_nr.cell_id
        elif best_lte is not None and best_lte.cell_id != runtime.anchor_cell_id:
            if best_lte.rsrp_dbm >= self.config.a4_absolute_threshold_dbm - 3.0:
                prepared_candidate_id = best_lte.cell_id

        if prepared_candidate_id != runtime.prepared_target_id:
            runtime.prepared_target_id = prepared_candidate_id
            runtime.prepared_since_s = timestamp_s if prepared_candidate_id is not None else None

        candidate = None
        emergency = False
        if runtime.secondary_cell_id is not None and best_nr is not None and best_nr.cell_id != runtime.secondary_cell_id:
            current_secondary = snapshot.by_cell_id.get(runtime.secondary_cell_id, serving)
            if self._trigger_condition(current_secondary, best_nr):
                candidate = best_nr
        elif runtime.secondary_cell_id is None and runtime.prepared_target_id is not None:
            prepared = snapshot.by_cell_id.get(runtime.prepared_target_id)
            if prepared is not None and self._trigger_condition(serving, prepared):
                candidate = prepared

        if candidate is None and serving.rsrp_dbm <= self.config.a5_serving_threshold_dbm and best_lte is not None:
            if best_lte.cell_id != runtime.anchor_cell_id and self._trigger_condition(serving, best_lte):
                candidate = best_lte

        if serving.rsrp_dbm <= self.config.emergency_rsrp_dbm:
            fallback = snapshot.by_cell_id.get(runtime.prepared_target_id) if runtime.prepared_target_id is not None else None
            candidate = fallback or best_nr or best_lte or snapshot.best_cell()
            emergency = True

        if serving.rsrp_dbm <= self.config.rlf_rsrp_dbm and candidate is None:
            self.total_rlf += 1
            runtime.serving_cell_id = snapshot.best_cell().cell_id
            runtime.anchor_cell_id = snapshot.best_lte_id or snapshot.best_cell().cell_id
            runtime.secondary_cell_id = snapshot.best_nr_id if snapshot.best_nr_id is not None and self._x2_allowed(runtime.anchor_cell_id, snapshot.best_nr_id) else None
            return HandoverDecision(
                executed=False,
                source_cell_id=serving.cell_id,
                target_cell_id=runtime.serving_cell_id,
                rlf_flag=1,
                reason="radio_link_failure",
            )

        if candidate is None:
            runtime.serving_cell_id = runtime.secondary_cell_id or runtime.anchor_cell_id or snapshot.best_cell().cell_id
            return HandoverDecision(
                source_cell_id=serving.cell_id,
                target_cell_id=runtime.serving_cell_id,
                reason="steady_state",
            )

        start_s = runtime.candidate_since_s.get(candidate.cell_id)
        if start_s is None:
            runtime.candidate_since_s = {candidate.cell_id: timestamp_s}
            return HandoverDecision(
                source_cell_id=serving.cell_id,
                target_cell_id=serving.cell_id,
                reason="ttt_pending",
            )
        if not emergency and (timestamp_s - start_s) < self.config.a3_ttt_s:
            runtime.candidate_since_s = {candidate.cell_id: start_s}
            return HandoverDecision(
                source_cell_id=serving.cell_id,
                target_cell_id=serving.cell_id,
                reason="ttt_pending",
            )

        source_id = runtime.serving_cell_id or serving.cell_id
        target_id = candidate.cell_id
        source_cell = self.cells_by_id[source_id]
        target_cell = self.cells_by_id[target_id]

        if target_cell.net_type == "5G NR" and not target_cell.is_drone:
            runtime.secondary_cell_id = target_id
            runtime.anchor_cell_id = runtime.anchor_cell_id or snapshot.best_lte_id or source_id
        elif target_cell.is_drone:
            runtime.secondary_cell_id = target_id
            runtime.anchor_cell_id = runtime.anchor_cell_id or snapshot.best_lte_id or source_id
        else:
            runtime.anchor_cell_id = target_id
            if runtime.secondary_cell_id is not None and not self._x2_allowed(target_id, runtime.secondary_cell_id):
                runtime.secondary_cell_id = None
        runtime.serving_cell_id = runtime.secondary_cell_id or runtime.anchor_cell_id or target_id
        runtime.prepared_target_id = None
        runtime.prepared_since_s = None
        runtime.candidate_since_s.clear()

        ping_pong = self._is_ping_pong(runtime.ho_history, timestamp_s, source_id, target_id)
        runtime.ho_history = [
            event for event in runtime.ho_history if timestamp_s - event[0] <= self.config.ping_pong_window_s
        ]
        runtime.ho_history.append((timestamp_s, source_id, target_id))
        runtime.last_handover_time_s = timestamp_s

        ho_class = self._classify(source_cell, target_cell, emergency)
        decision = HandoverDecision(
            executed=True,
            source_cell_id=source_id,
            target_cell_id=target_id,
            handover_class=ho_class,
            label=HO_LABELS[ho_class],
            success=1,
            ping_pong_flag=1 if ping_pong else 0,
            delay_s=(timestamp_s - start_s) if not emergency else 0.0,
            rlf_flag=1 if emergency and serving.rsrp_dbm <= self.config.rlf_rsrp_dbm else 0,
            reason="executed",
        )
        self.total_handovers += 1
        if ping_pong:
            self.total_ping_pong += 1
        return decision

    def _trigger_condition(self, serving, candidate) -> bool:
        a3 = candidate.rsrp_dbm >= (serving.rsrp_dbm + self.config.a3_hysteresis_db)
        a5 = serving.rsrp_dbm <= self.config.a5_serving_threshold_dbm and candidate.rsrp_dbm >= self.config.a5_target_threshold_dbm
        return bool(a3 or a5)

    def _x2_allowed(self, source_id: int | None, target_id: int) -> bool:
        if source_id is None:
            return True
        source = self.cells_by_id[source_id]
        target = self.cells_by_id[target_id]
        dx = source.x - target.x
        dy = source.y - target.y
        return ((dx * dx) + (dy * dy)) ** 0.5 <= self.config.x2_radius_m

    def _is_ping_pong(self, history: list[tuple[float, int, int]], timestamp_s: float, source_id: int, target_id: int) -> bool:
        for event_time, prior_source, prior_target in history:
            if timestamp_s - event_time <= self.config.ping_pong_window_s and prior_source == target_id and prior_target == source_id:
                return True
        return False

    def _classify(self, source: Cell, target: Cell, emergency: bool) -> int:
        if emergency:
            return 6
        if source.is_drone and not target.is_drone:
            return 5
        if not source.is_drone and target.is_drone:
            return 4
        if source.net_type != target.net_type:
            return 3
        if abs(source.frequency_ghz - target.frequency_ghz) > 0.1:
            return 2
        return 1
