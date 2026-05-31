from __future__ import annotations

import math
import numpy as np
from .config import SimulationConfig
from .mobility_engine import HybridMobilityEngine
from .models import MeasurementSnapshot


class DroneController:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def compute_waypoints(
        self,
        engine: HybridMobilityEngine,
        measurements: dict[str, MeasurementSnapshot],
        cell_loads: dict[int, float],
    ) -> dict[int, tuple[float, float]]:

        # 1. Perception Layer
        hotspots = self._calculate_hotspots(engine, measurements, cell_loads)

        # Fallback
        if not hotspots:
            busiest = sorted(
                engine.ground_cells,
                key=lambda cell: cell_loads.get(cell.cell_id, 0.0),
                reverse=True
            )
            hotspots = [(1.0, cell.x, cell.y) for cell in busiest[:5]]

        commands: dict[int, tuple[float, float]] = {}

        for drone_index, drone in enumerate(engine.drones):
            target_x, target_y = self._compute_drone_target(drone, hotspots)

            # Motion Layer: Direction-Aware Dispersion
            base_angle = math.atan2(target_y - drone.y, target_x - drone.x)
            offset_angle = (drone_index * 0.75) % (2 * math.pi)
            final_angle = base_angle + offset_angle

            radius = 40.0 + (drone_index % 5) * 15.0

            commands[drone.cell_id] = (
                target_x + radius * math.cos(final_angle),
                target_y + radius * math.sin(final_angle),
            )

        return commands

    def _calculate_hotspots(self, engine, measurements, cell_loads):
        """Perception Layer: load + RF hybrid scoring (normalized)."""

        rf_gap_by_cell: dict[int, list[float]] = {
            cell.cell_id: [] for cell in engine.ground_cells
        }

        for ue in engine.ues:
            snapshot = measurements[ue.ue_id]
            serving = snapshot.serving_measurement()

            nearest_ground = min(
                engine.ground_cells,
                key=lambda cell: math.hypot(
                    cell.x - ue.mobility.x,
                    cell.y - ue.mobility.y
                ),
            )

            rf_gap_by_cell[nearest_ground.cell_id].append(
                max(0.0, -100.0 - serving.rsrp_dbm)
            )

        raw_hotspots = []
        max_score = 1e-9  # stability fix

        for cell in engine.ground_cells:
            load = cell_loads.get(cell.cell_id, 0.1)
            load_excess = max(0.0, load - self.config.drone_load_threshold)

            rf_gaps = rf_gap_by_cell[cell.cell_id]
            rf_score = (float(np.mean(rf_gaps)) / 20.0) if rf_gaps else 0.0

            score = (0.65 * load_excess) + (0.35 * rf_score)
            max_score = max(max_score, score)

            raw_hotspots.append((score, cell.x, cell.y))

        normalized = []
        for s, x, y in raw_hotspots:
            ns = s / max_score
            if ns > 0.1:
                normalized.append((ns, x, y))

        return normalized

    def _compute_drone_target(self, drone, hotspots):
        """Unified Influence Model (stable + single-pass)."""

        influence_data = []

        for score, hx, hy in hotspots:
            dist = math.hypot(hx - drone.x, hy - drone.y)
            influence = score * math.exp(-dist / 500.0)
            influence_data.append((influence, score, hx, hy))

        influence_data.sort(key=lambda x: x[0], reverse=True)

        K = min(len(influence_data), 5)
        top = influence_data[:K]

        if not top:
            return drone.x, drone.y

        influences = np.array([t[0] for t in top])

        # stability + entropy smoothing
        influences = np.power(influences, 0.8) # This line controls the sensitivity of the drone to hotspots. Higher values make the drone more responsive to slight changes in load, while lower values make it more stable.
        """0.8 in influences  ,it is a design choice can be impreved by intellegent allocations systems"""
        
        total = np.sum(influences)

        if total > 0:
            weights = influences / total
        else:
            weights = np.ones(len(top)) / len(top)

        target_x = 0.0
        target_y = 0.0

        for i, (_, _, hx, hy) in enumerate(top):
            target_x += weights[i] * hx
            target_y += weights[i] * hy

        return target_x, target_y
