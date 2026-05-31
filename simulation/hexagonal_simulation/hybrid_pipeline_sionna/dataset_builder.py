from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DATASET_COLUMNS, SimulationConfig
from .mobility_engine import HybridMobilityEngine
from .models import HandoverDecision, MeasurementSnapshot, UEState


class DatasetBuilder:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.base_timestamp = datetime.fromisoformat(config.base_timestamp_iso).astimezone(timezone.utc)
        self.output_handle = config.output_csv.open("w", newline="")
        self.output_writer = csv.DictWriter(self.output_handle, fieldnames=DATASET_COLUMNS)
        self.output_writer.writeheader()
        self.drone_handle = config.drone_output_csv.open("w", newline="")
        self.drone_writer = csv.DictWriter(
            self.drone_handle,
            fieldnames=["timestamp", "cell_id", "x", "y", "altitude", "speed_ms", "tx_dbm", "frequency_mhz"],
        )
        self.drone_writer.writeheader()
        self.row_count = 0

    def close(self) -> None:
        self.output_handle.close()
        self.drone_handle.close()

    def write_ues(
        self,
        timestamp_s: float,
        engine: HybridMobilityEngine,
        measurements: dict[str, MeasurementSnapshot],
        decisions: dict[str, HandoverDecision],
    ) -> None:
        for ue in engine.ues:
            snapshot = measurements[ue.ue_id]
            decision = decisions[ue.ue_id]
            self.output_writer.writerow(self._build_row(timestamp_s, engine, ue, snapshot, decision))
            self.row_count += 1

    def write_drones(self, timestamp_s: float, engine: HybridMobilityEngine) -> None:
        stamp = self._timestamp(timestamp_s)
        for drone in engine.drones:
            self.drone_writer.writerow(
                {
                    "timestamp": stamp,
                    "cell_id": drone.cell_id,
                    "x": f"{drone.x:.3f}",
                    "y": f"{drone.y:.3f}",
                    "altitude": f"{drone.altitude_m:.1f}",
                    "speed_ms": f"{drone.speed_ms:.3f}",
                    "tx_dbm": f"{drone.tx_power_dbm:.2f}",
                    "frequency_mhz": f"{drone.frequency_ghz * 1000.0:.1f}",
                }
            )

    def _build_row(
        self,
        timestamp_s: float,
        engine: HybridMobilityEngine,
        ue: UEState,
        snapshot: MeasurementSnapshot,
        decision: HandoverDecision,
    ) -> dict[str, object]:
        runtime = ue.runtime
        serving_id = runtime.serving_cell_id or snapshot.serving_cell_id
        # serving = snapshot.by_cell_id[serving_id]
        serving = snapshot.by_cell_id.get(serving_id, snapshot.best_cell())
        target_id = decision.target_cell_id or serving_id
        target = snapshot.by_cell_id.get(target_id, serving)
        optimal = snapshot.best_cell()
        top_neighbors = snapshot.top_neighbors
        optimal_idx = next((index for index, meas in enumerate(top_neighbors) if meas.cell_id == optimal.cell_id), 0)
        serving_cell = engine.cells_by_id[serving_id]
        target_cell = engine.cells_by_id[target.cell_id]
        optimal_cell = engine.cells_by_id[optimal.cell_id]
        lat, lon = engine.geo_ref.xy_to_geo(ue.mobility.x, ue.mobility.y)
        nearest_drone = engine.nearest_drone(ue.mobility.x, ue.mobility.y)
        visibility_mask = (1 << len(top_neighbors)) - 1 if top_neighbors else 0

        return {
            "timestamp": self._timestamp(timestamp_s),
            "ue_id": ue.ue_id,
            "scenario_id": ue.scenario_id,
            "rsrp": f"{serving.rsrp_dbm:.2f}",
            "sinr": f"{serving.sinr_db:.2f}",
            "rsrq": f"{serving.rsrq_db:.2f}",
            "cqi": serving.cqi,
            "position_x": f"{lon:.6f}",
            "position_y": f"{lat:.6f}",
            "altitude": f"{ue.mobility.z:.1f}",
            "speed": f"{ue.mobility.speed_ms:.3f}",
            "direction": f"{ue.mobility.direction_deg:.2f}",
            "mobility_type": ue.mobility_type,
            "serving_cell_id": serving_id,
            "serving_cell_type": serving_cell.cell_type,
            "serving_net_type": serving_cell.net_type,
            "target_cell_id": target.cell_id,
            "target_cell_type": target_cell.cell_type,
            "handover_class": decision.handover_class,
            "handover_label": decision.label,
            "handover_success": decision.success,
            "ping_pong_flag": decision.ping_pong_flag,
            "handover_delay": f"{decision.delay_s:.3f}",
            "optimal_cell_id": optimal.cell_id,
            "optimal_cell_type": optimal_cell.cell_type,
            "optimal_cell_idx_in_k": optimal_idx,
            "optimal_cell_score": f"{optimal.score:.4f}",
            "optimal_cell_rsrp": f"{optimal.rsrp_dbm:.2f}",
            "optimal_cell_sinr": f"{optimal.sinr_db:.2f}",
            "optimal_cell_load": f"{optimal.load:.3f}",
            "optimal_cell_tp_est": f"{optimal.throughput_mbps:.3f}",
            "optimal_is_current": int(optimal.cell_id == serving_id),
            "nb_cell_ids": _fmt_list([meas.cell_id for meas in top_neighbors], precision=0),
            "nb_cell_types": _fmt_list([meas.cell_type for meas in top_neighbors]),
            "nb_net_types": _fmt_list([meas.net_type for meas in top_neighbors]),
            "nb_rsrps": _fmt_list([meas.rsrp_dbm for meas in top_neighbors], precision=2),
            "nb_sinrs": _fmt_list([meas.sinr_db for meas in top_neighbors], precision=2),
            "nb_loads": _fmt_list([meas.load for meas in top_neighbors], precision=3),
            "nb_tp_ests": _fmt_list([meas.throughput_mbps for meas in top_neighbors], precision=3),
            "nb_dists_m": _fmt_list([meas.distance_m for meas in top_neighbors], precision=1),
            "nb_path_losses_db": _fmt_list([meas.path_loss_db for meas in top_neighbors], precision=2),
            "nb_scores": _fmt_list([meas.score for meas in top_neighbors], precision=4),
            "num_visible_cells": len(snapshot.visible_cells),
            "num_neighbor_cells": len(top_neighbors),
            "cell_load": f"{serving.load:.3f}",
            "interference_level": f"{serving.interference_dbm:.3f}",
            "scan_radius_m": f"{self.config.filter_radius_m:.1f}",
            "cell_visibility_mask": visibility_mask,
            "latency": f"{snapshot.latency_ms:.2f}",
            "throughput": f"{serving.throughput_mbps:.3f}",
            "packet_loss": f"{snapshot.packet_loss:.5f}",
            "jitter": f"{snapshot.jitter_ms:.2f}",
            "hysteresis": f"{self.config.a3_hysteresis_db:.2f}",
            "time_to_trigger": int(self.config.a3_ttt_ms),
            "measurement_interval": f"{self.config.measurement_dt_s:.2f}",
            "zone_density": serving_cell.zone_density,
            "rlf_flag": decision.rlf_flag,
            "drone_strategy": self.config.drone_strategy_name,
            "nearest_drone_lat": f"{nearest_drone.lat:.6f}" if nearest_drone else "0.000000",
            "nearest_drone_lon": f"{nearest_drone.lon:.6f}" if nearest_drone else "0.000000",
            "nearest_drone_alt": f"{nearest_drone.altitude_m:.1f}" if nearest_drone else f"{self.config.drone_altitude_m:.1f}",
            "nearest_drone_speed_ms": f"{nearest_drone.speed_ms:.3f}" if nearest_drone else "0.000",
            "serving_site_id": serving_cell.site_id,
            "serving_sector_id": serving_cell.sector_id,
            "serving_azimuth": f"{serving_cell.azimuth_deg:.1f}",
        }

    def _timestamp(self, timestamp_s: float) -> str:
        stamp = self.base_timestamp + timedelta(seconds=timestamp_s)
        if stamp.microsecond:
            return stamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_list(values, precision: int | None = None) -> str:
    if not values:
        return "[]"
    if precision is None:
        return "[" + ";".join(str(value) for value in values) + "]"
    if precision == 0:
        return "[" + ";".join(str(int(value)) for value in values) + "]"
    return "[" + ";".join(f"{float(value):.{precision}f}" for value in values) + "]"
