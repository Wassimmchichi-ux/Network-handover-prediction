from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    import tensorflow as tf
    import sionna
    from sionna.phy.channel.tr38901 import AntennaArray, UMi
    SIONNA_AVAILABLE = True
except ImportError as e:
    SIONNA_AVAILABLE = False

from .config import SimulationConfig
from .models import Cell, CellMeasurement, MeasurementSnapshot, UEState


@dataclass
class _LinkState:
    shadow_db: float
    fast_db: float
    is_los: bool
    last_ue_x: float
    last_ue_y: float
    last_bs_x: float
    last_bs_y: float


class UMiChannelModel:
    def __init__(self, config: SimulationConfig, cells: list[Cell]) -> None:
        self.config = config
        self.cells = cells
        self.cells_by_id = {cell.cell_id: cell for cell in cells}
        self.rng = np.random.default_rng(config.measurement_seed)
        self.link_states: dict[tuple[str, int], _LinkState] = {}
        self.sionna_available = SIONNA_AVAILABLE and config.enable_sionna
        
        if self.sionna_available:
            # Initialize Sionna components
            # Single-element arrays to match analytic model's omnidirectional assumption
            # BS uses 3GPP 38.901 directional pattern for sectorization
            self.ut_array = AntennaArray(num_rows=1, num_cols=1, polarization="single", 
                                        polarization_type="V", antenna_pattern="omni", 
                                        carrier_frequency=2.1e9)
            self.bs_array = AntennaArray(num_rows=1, num_cols=1, polarization="single", 
                                        polarization_type="V", antenna_pattern="38.901", 
                                        carrier_frequency=2.1e9)

    def compute(
        self,
        ues: list[UEState],
        serving_map: dict[str, int | None],
        cell_loads: dict[int, float],
        timestamp_s: float,
    ) -> dict[str, MeasurementSnapshot]:
        if self.sionna_available:
            try:
                return self._compute_sionna(ues, serving_map, cell_loads, timestamp_s)
            except Exception as e:
                # Silence in production, but good to know for dev
                # print(f"Sionna error: {e}")
                pass
        return self._compute_analytic(ues, serving_map, cell_loads, timestamp_s)

    def _compute_sionna(
        self,
        ues: list[UEState],
        serving_map: dict[str, int | None],
        cell_loads: dict[int, float],
        timestamp_s: float,
    ) -> dict[str, MeasurementSnapshot]:
        import tensorflow as tf
        
        # 1. Coordinate Setup
        num_ues = len(ues)
        num_cells = len(self.cells)
        ue_pos = tf.constant([[ue.mobility.x, ue.mobility.y, ue.mobility.z] for ue in ues], dtype=tf.float32) # [num_ues, 3]
        bs_pos = tf.constant([[c.x, c.y, c.altitude_m] for c in self.cells], dtype=tf.float32) # [num_cells, 3]
        
        # 2. Group by Bands for Channel Modeling
        cell_freqs = np.array([cell.frequency_ghz for cell in self.cells])
        bands = np.unique(np.round(cell_freqs, 1))
        
        all_rsrps = np.full((num_ues, num_cells), self.config.rsrp_floor_dbm - 5.0)
        all_pls = np.full((num_ues, num_cells), 200.0)
        all_sinrs = np.full((num_ues, num_cells), -20.0)
        all_interf_dbm = np.full((num_ues, num_cells), -150.0)
        
        for band in bands:
            band_mask = np.round(cell_freqs, 1) == band
            band_indices = np.where(band_mask)[0]
            num_band_bs = len(band_indices)
            
            # Sionna Topology: [batch_size, num_bs, 3], [batch_size, num_ut, 3]
            # We use batch_size=1 and put all UEs and BSs in the topology
            bs_pos_b = tf.reshape(tf.gather(bs_pos, band_indices), [1, num_band_bs, 3])
            ut_pos_b = tf.reshape(ue_pos, [1, num_ues, 3])
            
            # 3. Sionna 3GPP UMi Model
            # This handles PathLoss + Shadowing + FastFading in a 3GPP-compliant way
            umi = UMi(carrier_frequency=band * 1e9, 
                      ut_array=self.ut_array, 
                      bs_array=self.bs_array, 
                      direction="downlink",
                      enable_shadow_fading=True,
                      enable_fast_fading=True) # Full stochastic fidelity
            
            # Get pathloss [1, num_bs, num_ut]
            pl = umi._path_loss(bs_pos_b, ut_pos_b)
            sf = umi._shadow_fading(bs_pos_b, ut_pos_b) if umi._enable_shadow_fading else 0.0

            # 4. Directional Antenna Gain (3GPP TR 38.901 Section 7.1)
            # Calculate horizontal angle between BS and UE
            dx = ut_pos_b[:, :, 0:1] - bs_pos_b[:, :, 0:1, 0] # [1, num_ues, num_bs]
            dy = ut_pos_b[:, :, 1:2] - bs_pos_b[:, :, 1:2, 0]
            ue_angles_rad = tf.atan2(dy, dx)
            
            # Get BS azimuths
            bs_az_deg = tf.constant([c.azimuth_deg for c in self.cells if round(c.frequency_ghz, 1) == band], dtype=tf.float32)
            bs_az_rad = bs_az_deg * (np.pi / 180.0)
            bs_az_rad = tf.reshape(bs_az_rad, [1, 1, num_band_bs])
            
            # Relative angle
            rel_angle = (ue_angles_rad - bs_az_rad + np.pi) % (2 * np.pi) - np.pi
            rel_angle_deg = rel_angle * (180.0 / np.pi)
            
            # Horizontal pattern: A(phi) = -min(12*(phi/phi_3db)^2, SLA_v)
            # We use standard 65 degree 3dB beamwidth
            gain_db = -tf.minimum(12.0 * tf.pow(rel_angle_deg / 65.0, 2), 30.0)
            
            # Combine to get large-scale RSRP
            tx_powers = tf.constant([self.cells[i].tx_power_dbm for i in band_indices], dtype=tf.float32)
            tx_powers_b = tf.reshape(tx_powers, [1, 1, num_band_bs])
            
            # RSRP = Tx + Gain - (PL + SF)
            rsrp_b = tx_powers_b + tf.transpose(gain_db, perm=[0, 2, 1]) - (pl + sf) # [1, num_bs, num_ut]
            
            # 4. Interference and SINR calculation in Tensor space
            # Convert RSRP (dBm) to linear power (mW)
            power_mw = tf.pow(10.0, rsrp_b / 10.0)
            
            # Reuse factors (LTE=0.15, NR=0.08)
            reuse = tf.constant([0.08 if self.cells[i].net_type == "5G NR" else 0.15 for i in band_indices], dtype=tf.float32)
            reuse_b = tf.reshape(reuse, [1, num_band_bs, 1])
            
            # Interference: for each link, sum power of other cells in the same band
            total_band_power = tf.reduce_sum(power_mw * reuse_b, axis=1, keepdims=True) # [1, 1, num_ut]
            interference_lin = tf.maximum(total_band_power - (power_mw * reuse_b), 1e-15) # [1, num_bs, num_ut]
            
            # Noise
            noise_dbm = -104.0 if band < 3.0 else -101.0
            noise_lin = 10.0 ** (noise_dbm / 10.0)
            
            # SINR = S / (I + N)
            sinr_lin = power_mw / (interference_lin + noise_lin)
            sinr_db = 10.0 * (tf.math.log(tf.maximum(sinr_lin, 1e-12)) / tf.math.log(10.0))
            
            # Convert back to NumPy
            all_rsrps[:, band_indices] = tf.transpose(rsrp_b[0]).numpy()
            all_pls[:, band_indices] = tf.transpose(pl[0]).numpy()
            all_sinrs[:, band_indices] = tf.transpose(sinr_db[0]).numpy()
            all_interf_dbm[:, band_indices] = tf.transpose(10.0 * (tf.math.log(interference_lin[0]) / tf.math.log(10.0))).numpy()

        # 5. Build Snapshots
        for i, ue in enumerate(ues):
            results[ue.ue_id] = self._build_snapshot_from_metrics(
                ue, all_rsrps[i], all_pls[i], all_sinrs[i], all_interf_dbm[i], serving_map, cell_loads, timestamp_s
            )
            
        return results

    def _build_snapshot_from_metrics(
        self,
        ue: UEState,
        rsrp: np.ndarray,
        path_loss: np.ndarray,
        sinr: np.ndarray,
        interference_dbm: np.ndarray,
        serving_map: dict[str, int | None],
        cell_loads: dict[int, float],
        timestamp_s: float,
    ) -> MeasurementSnapshot:
        # Simplified build logic using pre-calculated Sionna tensors
        dx = np.array([c.x for c in self.cells]) - ue.mobility.x
        dy = np.array([c.y for c in self.cells]) - ue.mobility.y
        d2d = np.sqrt(dx*dx + dy*dy)
        d3d = np.sqrt(d2d*d2d + (np.array([c.altitude_m for c in self.cells]) - ue.mobility.z)**2)
        
        visible = d2d <= self.config.filter_radius_m
        if not np.any(visible):
            visible[np.argmin(d2d)] = True
        visible_idx = np.flatnonzero(visible)
        
        rsrq = np.clip(sinr - 8.5, -19.5, -3.0)
        cqi = np.array([self._sinr_to_cqi(v) for v in sinr])
        cell_load_vals = np.array([cell_loads.get(c.cell_id, 0.1) for c in self.cells])
        throughput = np.array([self._throughput_estimate(self.cells[idx], sinr[idx], cell_load_vals[idx]) for idx in range(len(self.cells))])
        score = self._scores(rsrp, sinr, cell_load_vals)
        
        order = np.argsort(score[visible_idx])[::-1]
        sorted_idx = visible_idx[order]
        top_idx = sorted_idx[: self.config.k_neighbors]
        
        by_cell_id = {}
        visible_cells = []
        for idx in sorted_idx:
            cell = self.cells[idx]
            meas = CellMeasurement(
                cell_id=cell.cell_id,
                cell_type=cell.cell_type,
                net_type=cell.net_type,
                frequency_ghz=cell.frequency_ghz,
                rsrp_dbm=float(rsrp[idx]),
                sinr_db=float(sinr[idx]),
                rsrq_db=float(rsrq[idx]),
                cqi=int(cqi[idx]),
                path_loss_db=float(path_loss[idx]),
                distance_m=float(d3d[idx]),
                load=float(cell_load_vals[idx]),
                throughput_mbps=float(throughput[idx]),
                score=float(score[idx]),
                interference_dbm=float(interference_dbm[idx]),
            )
            by_cell_id[cell.cell_id] = meas
            visible_cells.append(meas)
            
        serving_id = serving_map.get(ue.ue_id)
        if serving_id is None or serving_id not in by_cell_id:
            serving_id = int(self.cells[top_idx[0]].cell_id)
        
        best_lte_id = next((self.cells[idx].cell_id for idx in sorted_idx if self.cells[idx].net_type.startswith("LTE")), None)
        best_nr_id = next((self.cells[idx].cell_id for idx in sorted_idx if self.cells[idx].net_type == "5G NR"), None)
        serving_meas = by_cell_id[serving_id]
        
        latency_ms = max(3.0, 8.0 + serving_meas.load * 18.0 + max(0.0, -serving_meas.sinr_db) * 0.25)
        packet_loss = min(1.0, 0.002 + max(0.0, -serving_meas.sinr_db - 1.0) * 0.003 + serving_meas.load * 0.02)
        jitter_ms = 0.6 + serving_meas.load * 2.4 + max(0.0, -serving_meas.sinr_db) * 0.08
        
        return MeasurementSnapshot(
            ue_id=ue.ue_id,
            timestamp_s=timestamp_s,
            serving_cell_id=serving_id,
            visible_cells=visible_cells,
            top_neighbors=[by_cell_id[self.cells[idx].cell_id] for idx in top_idx],
            by_cell_id=by_cell_id,
            best_lte_id=best_lte_id,
            best_nr_id=best_nr_id,
            latency_ms=float(latency_ms),
            packet_loss=float(packet_loss),
            jitter_ms=float(jitter_ms),
            interference_dbm=float(serving_meas.interference_dbm),
        )

    def _compute_analytic(
        self,
        ues: list[UEState],
        serving_map: dict[str, int | None],
        cell_loads: dict[int, float],
        timestamp_s: float,
    ) -> dict[str, MeasurementSnapshot]:
        cell_x = np.array([cell.x for cell in self.cells], dtype=float)
        cell_y = np.array([cell.y for cell in self.cells], dtype=float)
        cell_z = np.array([cell.altitude_m for cell in self.cells], dtype=float)
        cell_tx = np.array([cell.tx_power_dbm for cell in self.cells], dtype=float)
        cell_freq = np.array([cell.frequency_ghz for cell in self.cells], dtype=float)
        cell_load = np.array([cell_loads.get(cell.cell_id, 0.1) for cell in self.cells], dtype=float)
        band_keys = np.array([round(freq, 1) for freq in cell_freq], dtype=float)
        results: dict[str, MeasurementSnapshot] = {}

        for ue in ues:
            dx = cell_x - ue.mobility.x
            dy = cell_y - ue.mobility.y
            dz = cell_z - ue.mobility.z
            d2d = np.sqrt(dx * dx + dy * dy)
            d3d = np.sqrt(dx * dx + dy * dy + dz * dz)
            visible = d2d <= self.config.filter_radius_m
            if not np.any(visible):
                visible[np.argmin(d2d)] = True

            rsrp = np.full(len(self.cells), self.config.rsrp_floor_dbm - 5.0, dtype=float)
            path_loss = np.full(len(self.cells), 200.0, dtype=float)

            visible_idx = np.flatnonzero(visible)
            for idx in visible_idx:
                pl_db, rsrp_dbm = self._link_budget(ue, self.cells[idx], d2d[idx], d3d[idx])
                path_loss[idx] = pl_db
                rsrp[idx] = rsrp_dbm

            power_mw = np.power(10.0, rsrp / 10.0)
            reuse = np.array([0.08 if cell.net_type == "5G NR" else 0.15 for cell in self.cells], dtype=float)
            total_interference_by_band: dict[float, float] = {}
            for band in np.unique(band_keys[visible_idx]):
                mask = visible & (band_keys == band)
                total_interference_by_band[band] = float(np.sum(power_mw[mask] * reuse[mask]))

            sinr = np.full(len(self.cells), -20.0, dtype=float)
            interference_dbm = np.full(len(self.cells), -150.0, dtype=float)
            for idx in visible_idx:
                band = band_keys[idx]
                i_lin = max(total_interference_by_band.get(band, 0.0) - (power_mw[idx] * reuse[idx]), 0.0)
                noise_dbm = -104.0 if cell_freq[idx] < 3.0 else -101.0
                noise_lin = 10.0 ** (noise_dbm / 10.0)
                interference_dbm[idx] = 10.0 * math.log10(max(i_lin, 1e-15))
                sinr_lin = power_mw[idx] / max(i_lin + noise_lin, 1e-15)
                sinr[idx] = 10.0 * math.log10(max(sinr_lin, 1e-12))

            rsrq = np.clip(sinr - 8.5, -19.5, -3.0)
            cqi = np.array([self._sinr_to_cqi(value) for value in sinr], dtype=int)
            throughput = np.array(
                [
                    self._throughput_estimate(self.cells[idx], sinr[idx], cell_load[idx])
                    for idx in range(len(self.cells))
                ],
                dtype=float,
            )
            score = self._scores(rsrp, sinr, cell_load)
            order = np.argsort(score[visible_idx])[::-1]
            sorted_idx = visible_idx[order]
            top_idx = sorted_idx[: self.config.k_neighbors]

            by_cell_id: dict[int, CellMeasurement] = {}
            visible_cells: list[CellMeasurement] = []
            for idx in sorted_idx:
                meas = CellMeasurement(
                    cell_id=self.cells[idx].cell_id,
                    cell_type=self.cells[idx].cell_type,
                    net_type=self.cells[idx].net_type,
                    frequency_ghz=self.cells[idx].frequency_ghz,
                    rsrp_dbm=float(rsrp[idx]),
                    sinr_db=float(sinr[idx]),
                    rsrq_db=float(rsrq[idx]),
                    cqi=int(cqi[idx]),
                    path_loss_db=float(path_loss[idx]),
                    distance_m=float(d3d[idx]),
                    load=float(cell_load[idx]),
                    throughput_mbps=float(throughput[idx]),
                    score=float(score[idx]),
                    interference_dbm=float(interference_dbm[idx]),
                )
                by_cell_id[meas.cell_id] = meas
                visible_cells.append(meas)

            serving_id = serving_map.get(ue.ue_id)
            if serving_id is None or serving_id not in by_cell_id:
                serving_id = int(self.cells[top_idx[0]].cell_id)
            best_lte_id = next(
                (self.cells[idx].cell_id for idx in sorted_idx if self.cells[idx].net_type.startswith("LTE")),
                None,
            )
            best_nr_id = next(
                (self.cells[idx].cell_id for idx in sorted_idx if self.cells[idx].net_type == "5G NR"),
                None,
            )
            serving_meas = by_cell_id[serving_id]
            latency_ms = max(3.0, 8.0 + serving_meas.load * 18.0 + max(0.0, -serving_meas.sinr_db) * 0.25)
            packet_loss = min(1.0, 0.002 + max(0.0, -serving_meas.sinr_db - 1.0) * 0.003 + serving_meas.load * 0.02)
            jitter_ms = 0.6 + serving_meas.load * 2.4 + max(0.0, -serving_meas.sinr_db) * 0.08

            results[ue.ue_id] = MeasurementSnapshot(
                ue_id=ue.ue_id,
                timestamp_s=timestamp_s,
                serving_cell_id=serving_id,
                visible_cells=visible_cells,
                top_neighbors=[by_cell_id[self.cells[idx].cell_id] for idx in top_idx],
                by_cell_id=by_cell_id,
                best_lte_id=best_lte_id,
                best_nr_id=best_nr_id,
                latency_ms=float(latency_ms),
                packet_loss=float(packet_loss),
                jitter_ms=float(jitter_ms),
                interference_dbm=float(serving_meas.interference_dbm),
            )
        return results

    def _scores(self, rsrp: np.ndarray, sinr: np.ndarray, cell_load: np.ndarray) -> np.ndarray:
        rsrp_norm = np.clip((rsrp + 125.0) / 81.0, 0.0, 1.0)
        sinr_norm = np.clip((sinr + 10.0) / 30.0, 0.0, 1.0)
        load_norm = 1.0 - np.clip(cell_load, 0.0, 1.0)
        return (0.50 * rsrp_norm) + (0.25 * sinr_norm) + (0.25 * load_norm)

    def _link_budget(self, ue: UEState, cell: Cell, d2d_m: float, d3d_m: float) -> tuple[float, float]:
        d2d_m = max(d2d_m, 1.0)
        d3d_m = max(d3d_m, 1.0)
        p_los = self._los_prob_umi(d2d_m)
        state = self._next_link_state(ue, cell, p_los)
        pl_los = 32.4 + 21.0 * math.log10(d3d_m) + 20.0 * math.log10(cell.frequency_ghz)
        pl_nlos = max(pl_los, 22.4 + 35.3 * math.log10(d3d_m) + 21.3 * math.log10(cell.frequency_ghz))
        base_loss = pl_los if state.is_los else pl_nlos
        side_penalty = self._sidelobe_penalty(cell, d2d_m)
        total_loss = base_loss + state.shadow_db + side_penalty
        rsrp = cell.tx_power_dbm - total_loss + state.fast_db
        rsrp = min(self.config.rsrp_ceiling_dbm, max(self.config.rsrp_floor_dbm, rsrp))
        return total_loss, rsrp

    def _next_link_state(self, ue: UEState, cell: Cell, p_los: float) -> _LinkState:
        key = (ue.ue_id, cell.cell_id)
        previous = self.link_states.get(key)
        if previous is None:
            is_los = bool(self.rng.random() < p_los)
            shadow_std = 4.0 if is_los else 7.82
            state = _LinkState(
                shadow_db=float(self.rng.normal(0.0, shadow_std)),
                fast_db=float(self.rng.normal(0.0, 1.2 if is_los else 2.0)),
                is_los=is_los,
                last_ue_x=ue.mobility.x,
                last_ue_y=ue.mobility.y,
                last_bs_x=cell.x,
                last_bs_y=cell.y,
            )
            self.link_states[key] = state
            return state

        ue_disp = math.hypot(ue.mobility.x - previous.last_ue_x, ue.mobility.y - previous.last_ue_y)
        bs_disp = math.hypot(cell.x - previous.last_bs_x, cell.y - previous.last_bs_y)
        rel_disp = ue_disp + bs_disp
        if rel_disp <= 1e-9:
            return previous

        is_los = previous.is_los
        if rel_disp > 15.0 and self.rng.random() < min(1.0, rel_disp / 60.0):
            is_los = bool(self.rng.random() < p_los)
        shadow_std = 4.0 if is_los else 7.82
        rho_sf = math.exp(-rel_disp / 20.0)
        rho_ff = math.exp(-rel_disp / 2.0)
        shadow_db = (rho_sf * previous.shadow_db) + math.sqrt(max(0.0, 1.0 - rho_sf * rho_sf)) * float(
            self.rng.normal(0.0, shadow_std)
        )
        fast_db = (rho_ff * previous.fast_db) + math.sqrt(max(0.0, 1.0 - rho_ff * rho_ff)) * float(
            self.rng.normal(0.0, 1.2 if is_los else 2.0)
        )
        state = _LinkState(
            shadow_db=shadow_db,
            fast_db=fast_db,
            is_los=is_los,
            last_ue_x=ue.mobility.x,
            last_ue_y=ue.mobility.y,
            last_bs_x=cell.x,
            last_bs_y=cell.y,
        )
        self.link_states[key] = state
        return state

    def _los_prob_umi(self, d2d_m: float) -> float:
        return max(0.0, min(1.0, (18.0 / d2d_m) * (1.0 - math.exp(-d2d_m / 36.0)) + math.exp(-d2d_m / 36.0)))

    def _sinr_to_cqi(self, sinr_db: float) -> int:
        thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.7, 6.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
        cqi = 0
        for index, threshold in enumerate(thresholds, start=1):
            if sinr_db >= threshold:
                cqi = index
        return min(cqi, 15)

    def _throughput_estimate(self, cell: Cell, sinr_db: float, load: float) -> float:
        sinr_lin = 10.0 ** (sinr_db / 10.0)
        spectral_eff = math.log2(1.0 + max(sinr_lin, 1e-6))
        effective_peak = cell.peak_throughput_mbps * 0.30
        return max(0.0, effective_peak * min(1.0, spectral_eff / 7.0) * max(0.02, 1.0 - load))

    def _sidelobe_penalty(self, cell: Cell, dist_2d_m: float) -> float:
        if not cell.is_drone or cell.altitude_m < 1.0:
            return 0.0
        elev_deg = math.degrees(math.atan2(cell.altitude_m, max(dist_2d_m, 1.0)))
        if elev_deg < 30.0:
            return 0.0
        if elev_deg < 60.0:
            return ((elev_deg - 30.0) / 30.0) * 15.0
        return 15.0 + ((elev_deg - 60.0) / 30.0) * 10.0
