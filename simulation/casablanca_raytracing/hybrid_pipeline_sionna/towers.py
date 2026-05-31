from __future__ import annotations

import csv
import math
import random
from typing import Iterable

import numpy as np

from .config import SimulationConfig
from .geo import GeoReference, haversine_m
from .models import Cell


def _numeric_cell_id(unique_cell_id: str, fallback: int) -> int:
    digits = "".join(ch for ch in unique_cell_id if ch.isdigit())
    return int(digits) if digits else fallback


def _zone_density(xs: np.ndarray, ys: np.ndarray) -> list[str]:
    if len(xs) == 0:
        return []
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    counts = np.sum((dx * dx + dy * dy) <= (250.0 ** 2), axis=1)
    zones = []
    for count in counts:
        if count >= 25:
            zones.append("dense_urban")
        elif count >= 12:
            zones.append("urban")
        else:
            zones.append("suburban")
    return zones


def _cell_height(row: dict[str, str], config: SimulationConfig) -> float:
    return config.macro_height_m if row["cell_type"].strip().lower() == "macro" else config.micro_height_m


def _tx_power(row: dict[str, str], config: SimulationConfig) -> float:
    is_5g = row["is_5g"].strip() == "1" or "5G" in row["radio"]
    return config.nr_tx_dbm if is_5g else config.lte_tx_dbm


def _peak_tp(is_5g: bool) -> float:
    return 1200.0 if is_5g else 150.0


def load_ground_cells(config: SimulationConfig) -> tuple[GeoReference, list[Cell]]:
    with config.tower_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {config.tower_csv}")

    latitudes = np.array([float(row["lat"]) for row in rows], dtype=float)
    longitudes = np.array([float(row["lon"]) for row in rows], dtype=float)
    centroid_lat = float(np.mean(latitudes))
    centroid_lon = float(np.mean(longitudes))

    geo_ref = GeoReference(centroid_lat, centroid_lon, config.earth_radius_m)
    radial_distances = np.array(
        [
            haversine_m(float(row["lat"]), float(row["lon"]), centroid_lat, centroid_lon, config.earth_radius_m)
            for row in rows
        ],
        dtype=float,
    )
    in_cluster = radial_distances <= (config.cluster_km * 1000.0)
    cluster_rows = [row for row, keep in zip(rows, in_cluster) if keep]
    if not cluster_rows:
        cluster_rows = rows

    cluster_rows.sort(
        key=lambda row: haversine_m(
            float(row["lat"]),
            float(row["lon"]),
            centroid_lat,
            centroid_lon,
            config.earth_radius_m,
        )
    )
    selected = cluster_rows[: min(config.num_ground_cells, len(cluster_rows))]

    xs = []
    ys = []
    for row in selected:
        x, y = geo_ref.geo_to_xy(float(row["lat"]), float(row["lon"]))
        xs.append(x)
        ys.append(y)
    zones = _zone_density(np.array(xs), np.array(ys))

    cells: list[Cell] = []
    for idx, row in enumerate(selected):
        lat = float(row["lat"])
        lon = float(row["lon"])
        x, y = geo_ref.geo_to_xy(lat, lon)
        is_5g = row["is_5g"].strip() == "1" or "5G" in row["radio"]
        cells.append(
            Cell(
                cell_id=_numeric_cell_id(row["unique_cell_id"], idx + 1),
                unique_id=row["unique_cell_id"],
                radio=row["radio"],
                net_type="5G NR" if is_5g else "LTE-A",
                cell_type=row["cell_type"].strip().lower(),
                city=row["city"],
                cluster_id=int(row["cluster_id"]),
                is_5g=is_5g,
                is_drone=False,
                frequency_ghz=float(row["frequency_ghz"]),
                tx_power_dbm=_tx_power(row, config),
                peak_throughput_mbps=_peak_tp(is_5g),
                lat=lat,
                lon=lon,
                x=x,
                y=y,
                altitude_m=_cell_height(row, config),
                zone_density=zones[idx],
                load_seed=((idx + 1) * 0.013) % 1.0,
            )
        )
    return geo_ref, cells


def spawn_drones(
    config: SimulationConfig,
    geo_ref: GeoReference,
    ground_cells: Iterable[Cell],
    rng: random.Random,
) -> list[Cell]:
    candidates = [cell for cell in ground_cells if cell.is_5g]
    if not candidates:
        candidates = list(ground_cells)
    chosen: list[Cell] = []
    min_spacing_m = 150.0
    for candidate in candidates:
        if len(chosen) >= config.num_drones:
            break
        if all(math.hypot(candidate.x - existing.x, candidate.y - existing.y) >= min_spacing_m for existing in chosen):
            chosen.append(candidate)
    while len(chosen) < min(config.num_drones, len(candidates)):
        chosen.append(candidates[len(chosen)])

    next_id = max(cell.cell_id for cell in ground_cells) + 90_001
    drones: list[Cell] = []
    for index, candidate in enumerate(chosen[: config.num_drones]):
        jitter_x = rng.uniform(-75.0, 75.0)
        jitter_y = rng.uniform(-75.0, 75.0)
        x = candidate.x + jitter_x
        y = candidate.y + jitter_y
        lat, lon = geo_ref.xy_to_geo(x, y)
        drones.append(
            Cell(
                cell_id=next_id + index,
                unique_id=f"DRONE_{index:03d}",
                radio="5G NSA",
                net_type="5G NR",
                cell_type="drone",
                city=candidate.city,
                cluster_id=candidate.cluster_id,
                is_5g=True,
                is_drone=True,
                frequency_ghz=3.5,
                tx_power_dbm=config.drone_tx_dbm,
                peak_throughput_mbps=2000.0,
                lat=lat,
                lon=lon,
                x=x,
                y=y,
                altitude_m=config.drone_altitude_m,
                zone_density=candidate.zone_density,
                load_seed=rng.random(),
                waypoint_x=x,
                waypoint_y=y,
            )
        )
    return drones
