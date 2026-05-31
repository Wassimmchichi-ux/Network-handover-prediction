from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

DATASET_COLUMNS = [
    "timestamp",
    "ue_id",
    "scenario_id",
    "rsrp",
    "sinr",
    "rsrq",
    "cqi",
    "position_x",
    "position_y",
    "altitude",
    "speed",
    "direction",
    "mobility_type",
    "serving_cell_id",
    "serving_cell_type",
    "serving_net_type",
    "target_cell_id",
    "target_cell_type",
    "handover_class",
    "handover_label",
    "handover_success",
    "ping_pong_flag",
    "handover_delay",
    "optimal_cell_id",
    "optimal_cell_type",
    "optimal_cell_idx_in_k",
    "optimal_cell_score",
    "optimal_cell_rsrp",
    "optimal_cell_sinr",
    "optimal_cell_load",
    "optimal_cell_tp_est",
    "optimal_is_current",
    "nb_cell_ids",
    "nb_cell_types",
    "nb_net_types",
    "nb_rsrps",
    "nb_sinrs",
    "nb_loads",
    "nb_tp_ests",
    "nb_dists_m",
    "nb_path_losses_db",
    "nb_scores",
    "num_visible_cells",
    "num_neighbor_cells",
    "cell_load",
    "interference_level",
    "scan_radius_m",
    "cell_visibility_mask",
    "latency",
    "throughput",
    "packet_loss",
    "jitter",
    "hysteresis",
    "time_to_trigger",
    "measurement_interval",
    "zone_density",
    "rlf_flag",
    "drone_strategy",
    "nearest_drone_lat",
    "nearest_drone_lon",
    "nearest_drone_alt",
    "nearest_drone_speed_ms",
    "ray_tx_ids",
    "ray_rx_ids",
    "ray_tx_types",
    "ray_rx_types",
    "ray_path_vertices",
    "ray_path_types",
    "ray_path_powers_db",
    "ray_path_delays_s",
    "ray_los_flags",
]


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: int
    label: str
    speed_low_ms: float
    speed_high_ms: float
    weight: float


DEFAULT_SCENARIOS = (
    ScenarioConfig(1, "pedestrian", 1.0, 2.0, 0.10),
    ScenarioConfig(2, "urban_vehicle", 22.0, 28.0, 0.20),
    ScenarioConfig(3, "highway_vehicle", 33.0, 44.0, 0.40),
    ScenarioConfig(4, "high_speed", 40.0, 44.0, 0.30),
)


@dataclass(frozen=True)
class SimulationConfig:
    tower_csv: Path = Path("towers_densified.csv")
    output_csv: Path = Path("hybrid_handover_dataset.csv")
    drone_output_csv: Path = Path("hybrid_drone_positions.csv")
    summary_json: Path = Path("hybrid_handover_summary.json")
    seed: int = 42
    measurement_seed: int = 2024
    num_ground_cells: int = 20
    num_drones: int = 5
    num_ues: int = 10
    sim_time_s: float = 30
    mobility_dt_s: float = 0.01
    measurement_dt_s: float = 0.
    ho_check_dt_s: float = 0.02
    drone_control_dt_s: float = 0.2
    dataset_dt_s: float = 0.2
    cluster_km: float = 25.0
    filter_radius_m: float = 200
    x2_radius_m: float = 800.0
    k_neighbors: int = 4
    earth_radius_m: float = 6_371_000.0
    ue_height_m: float = 1.5
    macro_height_m: float = 30.0
    micro_height_m: float = 15.0
    drone_altitude_m: float = 120.0
    lte_tx_dbm: float = 46.0
    nr_tx_dbm: float = 40.0
    drone_tx_dbm: float = 30.0
    rsrp_floor_dbm: float = -125.0
    rsrp_ceiling_dbm: float = -44.0
    rlf_rsrp_dbm: float = -120.0
    emergency_rsrp_dbm: float = -115.0
    a3_hysteresis_db: float = 1.5
    a3_ttt_ms: float = 50.0
    a4_absolute_threshold_dbm: float = -110.0
    a5_serving_threshold_dbm: float = -114.0
    a5_target_threshold_dbm: float = -106.0
    conditional_margin_db: float = 0.5
    ping_pong_window_s: float = 5.0
    drone_speed_max_ms: float = 25.0
    drone_load_threshold: float = 0.55
    drone_strategy_name: str = "NSA_LOAD_RF_AWARE"
    target_ho_rate_min: float = 0.08
    target_ho_rate_max: float = 0.10
    enable_sionna: bool = True
    enable_raytracing: bool = True
    scene_path: Path | None = None
    base_timestamp_iso: str = "2024-09-01T00:00:00+00:00"
    static_rsrp_tolerance_db: float = 0.1
    scenarios: tuple[ScenarioConfig, ...] = DEFAULT_SCENARIOS

    def clone(self, **updates: object) -> "SimulationConfig":
        return replace(self, **updates)

    @property
    def a3_ttt_s(self) -> float:
        return self.a3_ttt_ms / 1000.0
