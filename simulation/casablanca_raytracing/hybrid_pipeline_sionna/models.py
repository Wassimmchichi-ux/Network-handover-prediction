from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MobilityState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float = 0.0

    @property
    def speed_ms(self) -> float:
        return (self.vx ** 2 + self.vy ** 2 + self.vz ** 2) ** 0.5

    @property
    def direction_deg(self) -> float:
        import math

        return math.degrees(math.atan2(self.vy, self.vx))


@dataclass
class Cell:
    cell_id: int
    unique_id: str
    radio: str
    net_type: str
    cell_type: str
    city: str
    cluster_id: int
    is_5g: bool
    is_drone: bool
    frequency_ghz: float
    tx_power_dbm: float
    peak_throughput_mbps: float
    lat: float
    lon: float
    x: float
    y: float
    altitude_m: float
    zone_density: str
    speed_ms: float = 0.0
    load_seed: float = 0.0
    waypoint_x: Optional[float] = None
    waypoint_y: Optional[float] = None


@dataclass
class HandoverRuntime:
    serving_cell_id: Optional[int] = None
    anchor_cell_id: Optional[int] = None
    secondary_cell_id: Optional[int] = None
    prepared_target_id: Optional[int] = None
    prepared_since_s: Optional[float] = None
    candidate_since_s: dict[int, float] = field(default_factory=dict)
    ho_history: list[tuple[float, int, int]] = field(default_factory=list)
    last_handover_time_s: float = -1e9
    l3_rsrp: dict[int, float] = field(default_factory=dict)
    l3_sinr: dict[int, float] = field(default_factory=dict)
    executing_target_id: Optional[int] = None
    executing_since_s: Optional[float] = None


@dataclass
class UEState:
    ue_id: str
    scenario_id: int
    mobility_type: str
    mobility: MobilityState
    runtime: HandoverRuntime = field(default_factory=HandoverRuntime)


@dataclass
class RayPath:
    tx_id: str
    rx_id: str
    tx_type: str
    rx_type: str
    path_vertices: list[list[float]]
    path_type: str
    path_power_db: float
    path_delay_s: float
    los_flag: bool


@dataclass
class CellMeasurement:
    cell_id: int
    cell_type: str
    net_type: str
    frequency_ghz: float
    rsrp_dbm: float
    sinr_db: float
    rsrq_db: float
    cqi: int
    path_loss_db: float
    distance_m: float
    load: float
    throughput_mbps: float
    score: float
    interference_dbm: float
    ray_paths: list[RayPath] = field(default_factory=list)


@dataclass
class MeasurementSnapshot:
    ue_id: str
    timestamp_s: float
    serving_cell_id: int
    visible_cells: list[CellMeasurement]
    top_neighbors: list[CellMeasurement]
    by_cell_id: dict[int, CellMeasurement]
    best_lte_id: Optional[int]
    best_nr_id: Optional[int]
    latency_ms: float
    packet_loss: float
    jitter_ms: float
    interference_dbm: float

    def best_cell(self) -> CellMeasurement:
        return self.top_neighbors[0]

    def serving_measurement(self) -> CellMeasurement:
        return self.by_cell_id[self.serving_cell_id]


@dataclass
class HandoverDecision:
    executed: bool = False
    source_cell_id: int = 0
    target_cell_id: int = 0
    handover_class: int = 0
    label: str = "no_handover"
    success: int = 0
    ping_pong_flag: int = 0
    delay_s: float = 0.0
    rlf_flag: int = 0
    reason: str = "steady_state"
