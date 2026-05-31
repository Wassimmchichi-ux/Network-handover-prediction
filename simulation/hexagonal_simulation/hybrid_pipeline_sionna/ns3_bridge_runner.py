from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .channel_model import UMiChannelModel
from .config import SimulationConfig
from .dataset_builder import DatasetBuilder
from .drone_controller import DroneController
from .handover import HandoverController
from .mobility_engine import HybridMobilityEngine
from .ns3_protocol import pack_control, unpack_state
from .state_buffer import StateBuffer
from .zeromq_ctypes import ZMQ_LINGER, ZMQ_RCVTIMEO, ZMQ_REP, ZMQ_SNDTIMEO, ZmqContext


class Ns3BridgeRunner:
    def __init__(
        self,
        config: SimulationConfig,
        ns3_root: Path,
        endpoint: str,
        auto_install: bool = True,
        auto_build: bool = True,
    ) -> None:
        self.config = config
        self.ns3_root = ns3_root
        self.endpoint = endpoint
        self.auto_install = auto_install
        self.auto_build = auto_build
        self.workspace_root = Path(__file__).resolve().parent.parent
        self.bridge_src = self.workspace_root / "ns3_bridge_src"
        self.bridge_dst = ns3_root / "scratch" / "hybrid-nsa-zmq"
        self.binary_path = ns3_root / "build" / "scratch" / "hybrid-nsa-zmq" / "ns3.41-ns3-hybrid-mobility-server-optimized"
        self.state_buffer = StateBuffer()
        self.engine = HybridMobilityEngine.bootstrap(config)
        self.handover = HandoverController(config, self.engine.cells_by_id)
        self.channel = UMiChannelModel(config, self.engine.cells)
        self.drone_controller = DroneController(config)
        self.dataset = DatasetBuilder(config)
        self.last_measurements = {}
        self.last_decisions = {}
        self.latched_decisions = {}
        self.cell_loads = {cell.cell_id: 0.1 for cell in self.engine.cells}

    def run(self) -> dict[str, object]:
        if self.auto_install:
            self.install_bridge_sources()
        if self.auto_build:
            self.build_ns3_bridge()
        if not self.binary_path.exists():
            raise FileNotFoundError(f"ns-3 bridge binary not found at {self.binary_path}")

        with tempfile.TemporaryDirectory(prefix="hybrid-ns3-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            active_cells_csv = temp_dir / "active_ground_cells.csv"
            active_drones_csv = temp_dir / "active_drones.csv"
            active_ues_csv = temp_dir / "active_ues.csv"
            self._write_active_cells(active_cells_csv)
            self._write_active_drones(active_drones_csv)
            self._write_active_ues(active_ues_csv)

            context = ZmqContext()
            socket = context.socket(ZMQ_REP)
            socket.set_int_option(ZMQ_LINGER, 0)
            socket.set_int_option(ZMQ_RCVTIMEO, 60_000)
            socket.set_int_option(ZMQ_SNDTIMEO, 60_000)
            socket.bind(self.endpoint)

            command = [
                str(self.binary_path),
                f"--activeCells={active_cells_csv}",
                f"--activeDrones={active_drones_csv}",
                f"--activeUes={active_ues_csv}",
                f"--endpoint={self.endpoint}",
                f"--simTime={self.config.sim_time_s}",
                f"--dtMs={int(round(self.config.mobility_dt_s * 1000.0))}",
            ]
            process = subprocess.Popen(command, cwd=self.ns3_root)
            try:
                summary = self._drive_loop(socket)
                return summary
            finally:
                socket.close()
                context.close()
                process.wait(timeout=30)

    def install_bridge_sources(self) -> None:
        if self.bridge_dst.exists():
            shutil.rmtree(self.bridge_dst)
        shutil.copytree(self.bridge_src, self.bridge_dst)

    def build_ns3_bridge(self) -> None:
        target = "scratch_hybrid-nsa-zmq_ns3-hybrid-mobility-server"
        subprocess.run(["cmake", "--build", "cmake-cache", "--target", target, "-j"], cwd=self.ns3_root, check=True)

    def _drive_loop(self, socket) -> dict[str, object]:
        total_steps = int(round(self.config.sim_time_s / self.config.mobility_dt_s))
        measurement_every = max(1, int(round(self.config.measurement_dt_s / self.config.mobility_dt_s)))
        ho_every = max(1, int(round(self.config.ho_check_dt_s / self.config.mobility_dt_s)))
        drone_every = max(1, int(round(self.config.drone_control_dt_s / self.config.mobility_dt_s)))
        dataset_every = max(1, int(round(self.config.dataset_dt_s / self.config.mobility_dt_s)))

        while True:
            payload = socket.recv()
            step, time_ms, ue_states, drone_states = unpack_state(payload)
            timestamp_s = time_ms / 1000.0
            self.engine.apply_external_state(ue_states, drone_states, timestamp_s)
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
                
                # Latch executed handovers so they aren't lost between dataset logging intervals
                for ue_id, decision in self.last_decisions.items():
                    if decision.executed or ue_id not in self.latched_decisions:
                        # Prioritize 'executed' decisions over 'steady_state' or 'pending'
                        if decision.executed:
                            self.latched_decisions[ue_id] = decision
                        elif ue_id not in self.latched_decisions:
                            self.latched_decisions[ue_id] = decision

            if self.last_measurements and step % drone_every == 0:
                commands = self.drone_controller.compute_waypoints(self.engine, self.last_measurements, self.cell_loads)
                self.engine.set_drone_waypoints(commands)
                self.state_buffer.publish_drone_commands(commands)
            else:
                commands = {}

            if self.last_measurements and step % dataset_every == 0:
                # Use latched decisions for the dataset, then clear the latch
                self.dataset.write_ues(timestamp_s, self.engine, self.last_measurements, self.latched_decisions)
                self.dataset.write_drones(timestamp_s, self.engine)
                self.latched_decisions = {}

            ue_updates = []
            drone_waypoints = []
            for drone_index, drone in enumerate(self.engine.drones):
                if drone.cell_id in commands:
                    target_x, target_y = commands[drone.cell_id]
                    drone_waypoints.append(
                        (drone_index, float(target_x), float(target_y), float(self.config.drone_altitude_m), float(self.config.drone_speed_max_ms))
                    )

            stop_flag = 1 if step >= total_steps else 0
            socket.send(pack_control(step, stop_flag, ue_updates, drone_waypoints))
            if stop_flag:
                break

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
            hour = (timestamp_s / 3600.0) % 24.0
            diurnal = 0.30 + 0.35 * __import__("math").exp(-0.5 * ((hour - 10.0) / 3.0) ** 2) + 0.25 * __import__("math").exp(
                -0.5 * ((hour - 19.0) / 2.5) ** 2
            )
            oscillation = __import__("math").sin(cell.load_seed * 10.0 + timestamp_s * 0.005) * 0.08
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
            "ns3_root": str(self.ns3_root),
            "ns3_binary": str(self.binary_path),
            "zmq_endpoint": self.endpoint,
            "num_ground_cells": len(self.engine.ground_cells),
            "num_drones": len(self.engine.drones),
            "num_ues": len(self.engine.ues),
            "sim_time_s": self.config.sim_time_s,
            "mobility_dt_s": self.config.mobility_dt_s,
            "measurement_dt_s": self.config.measurement_dt_s,
            "ho_check_dt_s": self.config.ho_check_dt_s,
            "drone_control_dt_s": self.config.drone_control_dt_s,
            "dataset_dt_s": self.config.dataset_dt_s,
            "channel_backend": "sionna_3gpp_tr38901" if self.channel.sionna_available else "analytic_umi_fallback",
            "sionna_available": self.channel.sionna_available,
            "handover_events": self.handover.total_handovers,
            "handover_rate": ho_rate,
            "ho_rate_per_ue_sec": self.handover.total_handovers / (len(self.engine.ues) * self.config.sim_time_s) if self.config.sim_time_s > 0 else 0.0,
            "target_ho_rate_min": self.config.target_ho_rate_min,
            "target_ho_rate_max": self.config.target_ho_rate_max,
            "ping_pong_events": self.handover.total_ping_pong,
            "rlf_events": self.handover.total_rlf,
            "rows_written": total_rows,
            "mobility_steps": total_steps,
            "mobility_engine": "ns3_zmq",
        }

    def _write_active_cells(self, path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell_id", "x", "y", "altitude_m", "lat", "lon", "cell_type", "net_type", "frequency_ghz"])
            for cell in self.engine.ground_cells:
                writer.writerow([cell.cell_id, cell.x, cell.y, cell.altitude_m, cell.lat, cell.lon, cell.cell_type, cell.net_type, cell.frequency_ghz])

    def _write_active_drones(self, path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell_id", "x", "y", "altitude_m", "lat", "lon", "speed_ms"])
            for drone in self.engine.drones:
                writer.writerow([drone.cell_id, drone.x, drone.y, drone.altitude_m, drone.lat, drone.lon, drone.speed_ms])

    def _write_active_ues(self, path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ue_id", "scenario_id", "mobility_type", "x", "y", "z", "vx", "vy", "vz"])
            for ue in self.engine.ues:
                writer.writerow(
                    [
                        ue.ue_id,
                        ue.scenario_id,
                        ue.mobility_type,
                        ue.mobility.x,
                        ue.mobility.y,
                        ue.mobility.z,
                        ue.mobility.vx,
                        ue.mobility.vy,
                        ue.mobility.vz,
                    ]
                )
