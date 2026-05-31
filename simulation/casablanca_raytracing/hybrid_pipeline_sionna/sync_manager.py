from __future__ import annotations

import json
import math

from .channel_model import UMiChannelModel
from .config import SimulationConfig
from .dataset_builder import DatasetBuilder
from .drone_controller import DroneController
from .handover import HandoverController
from .mobility_engine import HybridMobilityEngine
from .state_buffer import StateBuffer


class SyncManager:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.state_buffer = StateBuffer()
        self.engine = HybridMobilityEngine.bootstrap(config)
        self.handover = HandoverController(config, self.engine.cells_by_id)
        self.channel = UMiChannelModel(config, self.engine.cells)
        self.drone_controller = DroneController(config)
        self.dataset = DatasetBuilder(config)
        self.last_measurements = {}
        self.last_decisions = {}
        self.cell_loads = {cell.cell_id: 0.1 for cell in self.engine.cells}

    def run(self) -> dict[str, object]:
        total_steps = int(round(self.config.sim_time_s / self.config.mobility_dt_s))
        measurement_every = max(1, int(round(self.config.measurement_dt_s / self.config.mobility_dt_s)))
        ho_every = max(1, int(round(self.config.ho_check_dt_s / self.config.mobility_dt_s)))
        drone_every = max(1, int(round(self.config.drone_control_dt_s / self.config.mobility_dt_s)))
        dataset_every = max(1, int(round(self.config.dataset_dt_s / self.config.mobility_dt_s)))

        for step in range(total_steps + 1):
            timestamp_s = step * self.config.mobility_dt_s
            if step > 0:
                self.engine.step(self.config.mobility_dt_s)
            self.state_buffer.publish_positions(self.engine.positions_snapshot())

            if step % measurement_every == 0:
                self.cell_loads = self._estimate_cell_loads(timestamp_s)
                self.last_measurements = self.channel.compute(
                    self.engine.ues,
                    self.handover.serving_map(self.engine.ues),
                    self.cell_loads,
                    timestamp_s,
                )
                self.state_buffer.publish_metrics(self.last_measurements, self.cell_loads)

            if self.last_measurements and step % ho_every == 0:
                self.last_decisions = self.handover.evaluate_all(timestamp_s, self.engine.ues, self.last_measurements)
                self.state_buffer.publish_handover_state(self.last_decisions)

            if self.last_measurements and step % drone_every == 0:
                commands = self.drone_controller.compute_waypoints(self.engine, self.last_measurements, self.cell_loads)
                self.engine.set_drone_waypoints(commands)
                self.state_buffer.publish_drone_commands(commands)

            if self.last_measurements and step % dataset_every == 0:
                self.dataset.write_ues(timestamp_s, self.engine, self.last_measurements, self.last_decisions)
                self.dataset.write_drones(timestamp_s, self.engine)

        self.dataset.close()
        summary = self._build_summary(total_steps)
        with self.config.summary_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def _estimate_cell_loads(self, timestamp_s: float) -> dict[int, float]:
        attachment_count = {cell.cell_id: 0 for cell in self.engine.cells}
        serving_rsrp: dict[int, list[float]] = {cell.cell_id: [] for cell in self.engine.cells}
        for ue in self.engine.ues:
            serving = ue.runtime.serving_cell_id
            if serving is None:
                continue
            attachment_count[serving] += 1
            if ue.ue_id in self.last_measurements and serving in self.last_measurements[ue.ue_id].by_cell_id:
                serving_rsrp[serving].append(self.last_measurements[ue.ue_id].by_cell_id[serving].rsrp_dbm)

        loads: dict[int, float] = {}
        for cell in self.engine.cells:
            attached = attachment_count[cell.cell_id]
            cap = 10.0 if cell.cell_type == "macro" else 6.0
            if cell.is_drone:
                cap = 14.0
            occupancy = min(1.0, attached / cap)
            hour = math.fmod(timestamp_s / 3600.0, 24.0)
            diurnal = 0.30 + 0.35 * math.exp(-0.5 * ((hour - 10.0) / 3.0) ** 2) + 0.25 * math.exp(
                -0.5 * ((hour - 19.0) / 2.5) ** 2
            )
            oscillation = math.sin(cell.load_seed * 10.0 + timestamp_s * 0.005) * 0.08
            avg_rsrp = sum(serving_rsrp[cell.cell_id]) / len(serving_rsrp[cell.cell_id]) if serving_rsrp[cell.cell_id] else -90.0
            rf_stress = min(1.0, max(0.0, (-95.0 - avg_rsrp) / 20.0))
            load = (0.45 * occupancy) + (0.35 * diurnal) + (0.10 * rf_stress) + (0.10 * oscillation)
            if cell.is_drone:
                load *= 0.55
            loads[cell.cell_id] = float(min(0.99, max(0.05, load)))
        return loads

    def _build_summary(self, total_steps: int) -> dict[str, object]:
        total_rows = self.dataset.row_count
        ho_rate = (self.handover.total_handovers / total_rows) if total_rows else 0.0
        return {
            "output_csv": str(self.config.output_csv),
            "drone_output_csv": str(self.config.drone_output_csv),
            "summary_json": str(self.config.summary_json),
            "num_ground_cells": len(self.engine.ground_cells),
            "num_drones": len(self.engine.drones),
            "num_ues": len(self.engine.ues),
            "sim_time_s": self.config.sim_time_s,
            "mobility_dt_s": self.config.mobility_dt_s,
            "measurement_dt_s": self.config.measurement_dt_s,
            "ho_check_dt_s": self.config.ho_check_dt_s,
            "drone_control_dt_s": self.config.drone_control_dt_s,
            "dataset_dt_s": self.config.dataset_dt_s,
            "channel_backend": "analytic_umi_fallback",
            "sionna_available": self.channel.sionna_available,
            "handover_events": self.handover.total_handovers,
            "handover_rate": ho_rate,
            "target_ho_rate_min": self.config.target_ho_rate_min,
            "target_ho_rate_max": self.config.target_ho_rate_max,
            "ping_pong_events": self.handover.total_ping_pong,
            "rlf_events": self.handover.total_rlf,
            "rows_written": total_rows,
            "mobility_steps": total_steps,
        }
