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
    c_type = row.get("cell_type", "macro").strip().lower()
    return config.macro_height_m if c_type == "macro" else config.micro_height_m


def _tx_power(row: dict[str, str], config: SimulationConfig) -> float:
    net_type = row.get("net_type") or ("5G NR" if row.get("is_5g") == "1" else "LTE")
    radio = row.get("radio", "")
    is_5g = (net_type == "5G NR") or ("5G" in radio)
    return config.nr_tx_dbm if is_5g else config.lte_tx_dbm


def _peak_tp(is_5g: bool) -> float:
    return 1200.0 if is_5g else 150.0


def load_ground_cells(config: SimulationConfig) -> tuple[GeoReference, list[Cell]]:
    with config.tower_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {config.tower_csv}")

    # Check if this is a Cartesian grid or Lat/Lon
    is_cartesian = "x" in rows[0] and "y" in rows[0]
    
    if is_cartesian:
        geo_ref = GeoReference(0.0, 0.0, config.earth_radius_m)
        selected = rows[:config.num_ground_cells]
        xs = [float(row["x"]) for row in selected]
        ys = [float(row["y"]) for row in selected]
    else:
        latitudes = np.array([float(row.get("lat", 0)) for row in rows], dtype=float)
        longitudes = np.array([float(row.get("lon", 0)) for row in rows], dtype=float)
        centroid_lat = float(np.mean(latitudes))
        centroid_lon = float(np.mean(longitudes))
        geo_ref = GeoReference(centroid_lat, centroid_lon, config.earth_radius_m)
        
        radial_distances = np.array(
            [
                haversine_m(float(row.get("lat", 0)), float(row.get("lon", 0)), centroid_lat, centroid_lon, config.earth_radius_m)
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
                float(row.get("lat", 0)),
                float(row.get("lon", 0)),
                centroid_lat,
                centroid_lon,
                config.earth_radius_m,
            )
        )
        selected = cluster_rows[: min(config.num_ground_cells, len(cluster_rows))]
        xs = [geo_ref.geo_to_xy(float(row.get("lat", 0)), float(row.get("lon", 0)))[0] for row in selected]
        ys = [geo_ref.geo_to_xy(float(row.get("lat", 0)), float(row.get("lon", 0)))[1] for row in selected]

    zones = _zone_density(np.array(xs), np.array(ys))
    cells: list[Cell] = []
    for idx, row in enumerate(selected):
        x = float(row.get("x", 0)) if is_cartesian else xs[idx]
        y = float(row.get("y", 0)) if is_cartesian else ys[idx]
        lat, lon = geo_ref.xy_to_geo(x, y)
        
        # Mapping column names from hex CSV or Casablanca CSV
        c_id_raw = row.get("cell_id") or row.get("unique_cell_id") or str(idx + 1)
        c_id = _numeric_cell_id(str(c_id_raw), idx + 1)
        net_type = row.get("net_type") or ("5G NR" if row.get("is_5g") == "1" else "LTE-A")
        is_5g = (net_type == "5G NR")
        
        cells.append(
            Cell(
                cell_id=c_id,
                unique_id=str(c_id_raw),
                radio=row.get("radio", net_type),
                net_type=net_type,
                cell_type=row.get("cell_type", "macro").strip().lower(),
                city=row.get("city", "Unknown"),
                cluster_id=int(row.get("cluster_id", 0)),
                is_5g=is_5g,
                is_drone=False,
                frequency_ghz=float(row.get("frequency_ghz", 2.1)),
                tx_power_dbm=float(row.get("tx_power_dbm", _tx_power(row, config))),
                peak_throughput_mbps=_peak_tp(is_5g),
                lat=lat,
                lon=lon,
                x=x,
                y=y,
                altitude_m=float(row.get("altitude_m", _cell_height(row, config))),
                zone_density=zones[idx],
                load_seed=((idx + 1) * 0.013) % 1.0,
                site_id=int(row.get("site_id", 0)),
                sector_id=int(row.get("sector_id", 0)),
                azimuth_deg=float(row.get("azimuth_deg", 0.0)),
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
