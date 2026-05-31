from __future__ import annotations

import math
import random

from .config import ScenarioConfig, SimulationConfig
from .geo import GeoReference
from .models import Cell, MobilityState, UEState
from .towers import load_ground_cells, spawn_drones


class HybridMobilityEngine:
    def __init__(
        self,
        config: SimulationConfig,
        geo_ref: GeoReference,
        ground_cells: list[Cell],
        drones: list[Cell],
        ues: list[UEState],
    ) -> None:
        self.config = config
        self.geo_ref = geo_ref
        self.ground_cells = ground_cells
        self.drones = drones
        self.ues = ues
        self.cells = self.ground_cells + self.drones
        self.cells_by_id = {cell.cell_id: cell for cell in self.cells}
        xs = [cell.x for cell in self.ground_cells]
        ys = [cell.y for cell in self.ground_cells]
        pad = 250.0
        self.bounds = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
        self.time_s = 0.0

    @classmethod
    def bootstrap(cls, config: SimulationConfig) -> "HybridMobilityEngine":
        rng = random.Random(config.seed)
        geo_ref, ground_cells = load_ground_cells(config)
        drones = spawn_drones(config, geo_ref, ground_cells, rng)
        ues = _spawn_ues(config, ground_cells, rng)
        return cls(config, geo_ref, ground_cells, drones, ues)

    def step(self, dt_s: float) -> None:
        self.time_s += dt_s
        xmin, xmax, ymin, ymax = self.bounds
        for ue in self.ues:
            ue.mobility.x += ue.mobility.vx * dt_s
            ue.mobility.y += ue.mobility.vy * dt_s
            if ue.mobility.x < xmin or ue.mobility.x > xmax:
                ue.mobility.vx *= -1.0
                ue.mobility.x = min(max(ue.mobility.x, xmin), xmax)
            if ue.mobility.y < ymin or ue.mobility.y > ymax:
                ue.mobility.vy *= -1.0
                ue.mobility.y = min(max(ue.mobility.y, ymin), ymax)

        for drone in self.drones:
            target_x = drone.waypoint_x if drone.waypoint_x is not None else drone.x
            target_y = drone.waypoint_y if drone.waypoint_y is not None else drone.y
            dx = target_x - drone.x
            dy = target_y - drone.y
            distance = math.hypot(dx, dy)
            max_step = self.config.drone_speed_max_ms * dt_s
            if distance <= max_step or distance < 1e-9:
                move_x = dx
                move_y = dy
                drone.speed_ms = distance / dt_s if dt_s > 0.0 else 0.0
            else:
                scale = max_step / distance
                move_x = dx * scale
                move_y = dy * scale
                drone.speed_ms = self.config.drone_speed_max_ms
            drone.x = min(max(drone.x + move_x, xmin), xmax)
            drone.y = min(max(drone.y + move_y, ymin), ymax)
            drone.lat, drone.lon = self.geo_ref.xy_to_geo(drone.x, drone.y)

    def update_velocity(self, ue_id: str, vx: float, vy: float, vz: float = 0.0) -> None:
        for ue in self.ues:
            if ue.ue_id == ue_id:
                ue.mobility.vx = vx
                ue.mobility.vy = vy
                ue.mobility.vz = vz
                return
        raise KeyError(f"Unknown UE {ue_id}")

    def set_drone_waypoints(self, waypoints: dict[int, tuple[float, float]]) -> None:
        for drone in self.drones:
            if drone.cell_id in waypoints:
                drone.waypoint_x, drone.waypoint_y = waypoints[drone.cell_id]

    def apply_external_state(
        self,
        ue_states: list[tuple[float, float, float, float, float, float]],
        drone_states: list[tuple[float, float, float, float, float, float]],
        timestamp_s: float,
    ) -> None:
        self.time_s = timestamp_s
        if len(ue_states) != len(self.ues):
            raise ValueError(f"UE state count mismatch: expected {len(self.ues)}, got {len(ue_states)}")
        if len(drone_states) != len(self.drones):
            raise ValueError(f"Drone state count mismatch: expected {len(self.drones)}, got {len(drone_states)}")

        for ue, state in zip(self.ues, ue_states):
            x, y, z, vx, vy, vz = state
            ue.mobility.x = x
            ue.mobility.y = y
            ue.mobility.z = z
            ue.mobility.vx = vx
            ue.mobility.vy = vy
            ue.mobility.vz = vz

        for drone, state in zip(self.drones, drone_states):
            x, y, z, vx, vy, vz = state
            drone.x = x
            drone.y = y
            drone.altitude_m = z
            drone.speed_ms = (vx * vx + vy * vy + vz * vz) ** 0.5
            drone.lat, drone.lon = self.geo_ref.xy_to_geo(x, y)

    def positions_snapshot(self) -> dict[str, dict[str, float]]:
        snapshot: dict[str, dict[str, float]] = {}
        for ue in self.ues:
            lat, lon = self.geo_ref.xy_to_geo(ue.mobility.x, ue.mobility.y)
            snapshot[ue.ue_id] = {
                "x": ue.mobility.x,
                "y": ue.mobility.y,
                "lat": lat,
                "lon": lon,
                "speed_ms": ue.mobility.speed_ms,
            }
        return snapshot

    def nearest_drone(self, x: float, y: float) -> Cell | None:
        if not self.drones:
            return None
        return min(self.drones, key=lambda drone: math.hypot(drone.x - x, drone.y - y))

def _pick_scenario(config: SimulationConfig, rng: random.Random) -> ScenarioConfig:
    pick = rng.random()
    cumulative = 0.0
    for scenario in config.scenarios:
        cumulative += scenario.weight
        if pick <= cumulative:
            return scenario
    return config.scenarios[-1]


def _spawn_ues(config: SimulationConfig, ground_cells: list[Cell], rng: random.Random) -> list[UEState]:
    ues: list[UEState] = []
    for index in range(config.num_ues):
        scenario = _pick_scenario(config, rng)
        base_cell = rng.choice(ground_cells)
        speed = rng.uniform(scenario.speed_low_ms, scenario.speed_high_ms)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        ues.append(
            UEState(
                ue_id=f"MA_UE_{index + 1:04d}",
                scenario_id=scenario.scenario_id,
                mobility_type=scenario.label,
                mobility=MobilityState(
                    x=base_cell.x + rng.uniform(-500.0, 500.0),
                    y=base_cell.y + rng.uniform(-500.0, 500.0),
                    z=config.ue_height_m,
                    vx=speed * math.cos(angle),
                    vy=speed * math.sin(angle),
                ),
            )
        )
    return ues
