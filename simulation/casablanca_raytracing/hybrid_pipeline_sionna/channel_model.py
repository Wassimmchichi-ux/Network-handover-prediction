import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import tensorflow as tf
    import sionna
    # Sionna 1.2.2: correct module path
    from sionna.phy.channel.tr38901 import AntennaArray, UMi
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False

from .config import SimulationConfig
from .models import Cell, CellMeasurement, MeasurementSnapshot, UEState, RayPath


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
    def __init__(self, config: SimulationConfig, cells: List[Cell]) -> None:
        self.config = config
        self.cells = cells
        self.cells_by_id = {cell.cell_id: cell for cell in cells}
        self.rng = np.random.default_rng(config.measurement_seed)
        self.link_states: Dict[Tuple[str, int], _LinkState] = {}
        self.sionna_available = SIONNA_AVAILABLE and config.enable_sionna

        # Pre-compute cell arrays for analytic model
        self._cell_x = np.array([c.x for c in cells], dtype=float)
        self._cell_y = np.array([c.y for c in cells], dtype=float)
        self._cell_z = np.array([c.altitude_m for c in cells], dtype=float)
        self._cell_freq = np.array([c.frequency_ghz for c in cells], dtype=float)
        self._cell_tx = np.array([c.tx_power_dbm for c in cells], dtype=float)

        # Pre-compute per-band cell groupings for interference
        self._band_groups: Dict[float, List[int]] = {}
        for idx, c in enumerate(cells):
            bk = round(c.frequency_ghz, 1)
            self._band_groups.setdefault(bk, []).append(idx)

        if self.sionna_available:
            self.ut_array = AntennaArray(
                num_rows=1, num_cols=1, polarization="single",
                polarization_type="V", antenna_pattern="omni",
                carrier_frequency=2.1e9)
            self.bs_array = AntennaArray(
                num_rows=1, num_cols=1, polarization="single",
                polarization_type="V", antenna_pattern="omni",
                carrier_frequency=2.1e9)

            nom_freq = float(np.mean(self._cell_freq)) * 1e9
            self.umi_stochastic = UMi(
                carrier_frequency=nom_freq,
                o2i_model="low",
                ut_array=self.ut_array,
                bs_array=self.bs_array,
                direction="downlink",
                enable_pathloss=True,
                enable_shadow_fading=True)

            self.rt_receivers: Dict[str, Receiver] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def compute(
        self,
        ues: List[UEState],
        serving_map: Dict[str, Optional[int]],
        cell_loads: Dict[int, float],
        timestamp_s: float,
    ) -> Dict[str, MeasurementSnapshot]:
        if self.sionna_available:
            try:
                if self.config.enable_raytracing and self.config.scene_path:
                    return self._compute_sionna_rt(ues, serving_map, cell_loads, timestamp_s)
                else:
                    return self._compute_sionna_stochastic(ues, serving_map, cell_loads, timestamp_s)
            except Exception as e:
                import traceback
                print(f"[ERROR] Sionna failed — NOT falling back silently:\n{traceback.format_exc()}")
                raise
        return self._compute_analytic(ues, serving_map, cell_loads, timestamp_s)

    # ------------------------------------------------------------------
    # Module 1: Analytic baseline (TR 38.901 closed-form)
    # ------------------------------------------------------------------
    def _compute_analytic(
        self, ues: List[UEState], serving_map: Dict[str, Optional[int]],
        cell_loads: Dict[int, float], timestamp_s: float,
    ) -> Dict[str, MeasurementSnapshot]:
        results: Dict[str, MeasurementSnapshot] = {}
        load_arr = np.array([cell_loads.get(c.cell_id, 0.1) for c in self.cells])

        for ue in ues:
            dx = self._cell_x - ue.mobility.x
            dy = self._cell_y - ue.mobility.y
            dz = self._cell_z - ue.mobility.z
            d2d = np.sqrt(dx * dx + dy * dy)
            d3d = np.sqrt(dx * dx + dy * dy + dz * dz)

            rsrp = np.full(len(self.cells), self.config.rsrp_floor_dbm - 5.0)
            path_loss = np.full(len(self.cells), 200.0)

            visible = d2d <= self.config.filter_radius_m
            if not np.any(visible):
                visible[np.argmin(d2d)] = True
            vis_idx = np.flatnonzero(visible)

            for idx in vis_idx:
                pl_db, rsrp_dbm = self._link_budget(ue, self.cells[idx], d2d[idx], d3d[idx])
                path_loss[idx] = pl_db
                rsrp[idx] = rsrp_dbm

            sinr, interf_dbm = self._sinr_per_band(rsrp)

            results[ue.ue_id] = self._build_snapshot(
                ue, rsrp, path_loss, sinr, interf_dbm, d3d,
                load_arr, serving_map, timestamp_s)
        return results

    # ------------------------------------------------------------------
    # Module 2: Sionna Stochastic (UMi)
    # ------------------------------------------------------------------
    def _compute_sionna_stochastic(
        self, ues: List[UEState], serving_map: Dict[str, Optional[int]],
        cell_loads: Dict[int, float], timestamp_s: float,
    ) -> Dict[str, MeasurementSnapshot]:
        num_ues = len(ues)
        num_bs = len(self.cells)
        load_arr = np.array([cell_loads.get(c.cell_id, 0.1) for c in self.cells])

        D = tf.float32  # matches umi.rdtype
        ut_loc = tf.constant(
            [[[ue.mobility.x, ue.mobility.y, ue.mobility.z] for ue in ues]],
            dtype=D)  # [1, num_ues, 3]
        bs_loc = tf.constant(
            [[[c.x, c.y, c.altitude_m] for c in self.cells]],
            dtype=D)  # [1, num_bs, 3]

        # All UEs outdoor, zero orientation/velocity
        in_state = tf.zeros([1, num_ues], dtype=tf.bool)
        ut_orient = tf.zeros([1, num_ues, 3], dtype=D)
        bs_orient = tf.zeros([1, num_bs, 3], dtype=D)
        ut_vel = tf.zeros([1, num_ues, 3], dtype=D)

        self.umi_stochastic.set_topology(
            ut_loc=ut_loc, bs_loc=bs_loc, in_state=in_state,
            ut_orientations=ut_orient, bs_orientations=bs_orient,
            ut_velocities=ut_vel)

        # h shape: [1, num_ues, 1, num_bs, 1, num_clusters, 1]
        h, delays = self.umi_stochastic(num_time_samples=1, sampling_frequency=1.0)

        # Sum |h|^2 over clusters(axis=5) and time(axis=6)
        h_power = tf.reduce_sum(tf.abs(h) ** 2, axis=[-1, -2])
        # Extract [num_ues, num_bs] by indexing out batch and antenna dims
        h_power = h_power[0, :, 0, :, 0]  # [num_ues, num_bs]

        gain_db = 10.0 * (tf.math.log(tf.maximum(h_power, 1e-15)) / tf.math.log(10.0))
        tx_p = tf.constant(self._cell_tx, dtype=D)
        rsrp_tf = gain_db + tx_p
        pl_tf = tx_p - gain_db

        rsrp_np = rsrp_tf.numpy()
        pl_np = pl_tf.numpy()

        # Distances for snapshot
        d3d_all = np.zeros((num_ues, len(self.cells)))
        for i, ue in enumerate(ues):
            dx = self._cell_x - ue.mobility.x
            dy = self._cell_y - ue.mobility.y
            dz = self._cell_z - ue.mobility.z
            d3d_all[i] = np.sqrt(dx*dx + dy*dy + dz*dz)

        results = {}
        for i, ue in enumerate(ues):
            sinr, interf = self._sinr_per_band(rsrp_np[i])
            results[ue.ue_id] = self._build_snapshot(
                ue, rsrp_np[i], pl_np[i], sinr, interf, d3d_all[i],
                load_arr, serving_map, timestamp_s)
        return results

    # ------------------------------------------------------------------
    # Module 3: Sionna Ray Tracing
    # ------------------------------------------------------------------
    def _compute_sionna_rt(
        self, ues: List[UEState], serving_map: Dict[str, Optional[int]],
        cell_loads: Dict[int, float], timestamp_s: float,
    ) -> Dict[str, MeasurementSnapshot]:
        num_ues = len(ues)
        load_arr = np.array([cell_loads.get(c.cell_id, 0.1) for c in self.cells])

        if not hasattr(self, "scene"):
            self.scene = load_scene(str(self.config.scene_path))

            # arrays (OK to set once)
            self.scene.tx_array = PlanarArray(
                num_rows=1, num_cols=1,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V")

            self.scene.rx_array = PlanarArray(
                num_rows=1, num_cols=1,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V")

            # add transmitters ONLY ONCE
            for cell in self.cells:
                tx = Transmitter(
                    name=f"cell_{cell.cell_id}",
                    position=[cell.x, cell.y, cell.altitude_m],
                    power_dbm=cell.tx_power_dbm)
                self.scene.add(tx)
            self.scene.scene_geometry_updated()

        from sionna.rt import PathSolver
        solver = PathSolver()
        
        # Batching with dynamic add/remove to avoid CUDA_ERROR_ILLEGAL_ADDRESS
        batch_size = 5
        rsrp_np = np.zeros((num_ues, len(self.cells)))
        pl_np = np.zeros((num_ues, len(self.cells)))
        all_a_list = []
        all_tau_list = []
        all_vertices_list = []
        all_interactions_list = []

        for i_batch in range(0, num_ues, batch_size):
            batch_indices = range(i_batch, min(i_batch + batch_size, num_ues))
            
            # Clear previous batch receivers to keep scene small
            for rx_name in list(self.scene.receivers.keys()):
                self.scene.remove(rx_name)
            
            # Add only this batch
            batch_ues = [ues[i] for i in batch_indices]
            for i_ue, ue in enumerate(batch_ues):
                rx = Receiver(name=f"batch_rx_{i_ue}", 
                              position=[ue.mobility.x, ue.mobility.y, ue.mobility.z])
                self.scene.add(rx)

            # Solve for this batch
            paths = solver(self.scene, max_depth=2, max_num_paths_per_src=5, samples_per_src=100)
            a, tau = paths.cir(num_time_steps=1, out_type="tf")
            
            # Extract gain/rsrp for this batch
            gain_lin = tf.reduce_sum(tf.abs(a) ** 2, axis=[1, 3, 4, 5]).numpy()
            batch_gain_db = 10.0 * np.log10(np.maximum(gain_lin, 1e-15))
            
            for j_slot, i_ue in enumerate(batch_indices):
                rsrp_np[i_ue] = batch_gain_db[j_slot] + self._cell_tx
                pl_np[i_ue] = self._cell_tx - batch_gain_db[j_slot]

            # Pad path dimension to max_num_paths_per_src (5) to ensure consistent shapes for concatenation
            max_p = 5
            
            a_val = a.numpy()
            if a_val.shape[4] < max_p:
                a_val = np.pad(a_val, ((0,0), (0,0), (0,0), (0,0), (0, max_p - a_val.shape[4]), (0,0)))
            all_a_list.append(a_val)
            
            tau_val = tau.numpy()
            if tau_val.shape[2] < max_p:
                tau_val = np.pad(tau_val, ((0,0), (0,0), (0, max_p - tau_val.shape[2])))
            all_tau_list.append(tau_val)
            
            v_val = np.array(paths.vertices)
            if v_val.shape[3] < max_p:
                v_val = np.pad(v_val, ((0,0), (0,0), (0,0), (0, max_p - v_val.shape[3]), (0,0)))
            all_vertices_list.append(v_val)
            
            int_val = np.array(paths.interactions)
            if int_val.shape[3] < max_p:
                int_val = np.pad(int_val, ((0,0), (0,0), (0,0), (0, max_p - int_val.shape[3])))
            all_interactions_list.append(int_val)

        a_np = np.concatenate(all_a_list, axis=0)
        tau_np = np.concatenate(all_tau_list, axis=0)
        vertices_np = np.concatenate(all_vertices_list, axis=1)
        interactions_np = np.concatenate(all_interactions_list, axis=1)
        path_power_lin = np.sum(np.abs(a_np)**2, axis=(1, 3, 5))



        d3d_all = np.zeros((num_ues, len(self.cells)))
        for i, ue in enumerate(ues):
            dx = self._cell_x - ue.mobility.x
            dy = self._cell_y - ue.mobility.y
            dz = self._cell_z - ue.mobility.z
            d3d_all[i] = np.sqrt(dx*dx + dy*dy + dz*dz)

        ray_paths_per_cell = {}
        for i, ue in enumerate(ues):
            ray_paths_per_cell[ue.ue_id] = {}
            for j, cell in enumerate(self.cells):
                powers = path_power_lin[i, j, :]
                valid_paths = np.where(powers > 0)[0]
                if len(valid_paths) == 0:
                    continue
                
                # Top K strongest paths per link
                K = min(5, len(valid_paths))
                top_indices = valid_paths[np.argsort(powers[valid_paths])[::-1][:K]]
                
                cell_rays = []
                for p_idx in top_indices:
                    p_power_db = 10.0 * float(np.log10(max(powers[p_idx], 1e-15))) + cell.tx_power_dbm
                    p_tau = float(tau_np[i, j, p_idx])
                    
                    interacts = interactions_np[:, i, j, p_idx]
                    los_flag = bool(np.all(interacts == 0))
                    
                    if los_flag:
                        p_type = "LOS"
                    elif 8 in interacts:
                        p_type = "DIFFRACTION"
                    elif 2 in interacts:
                        p_type = "SCATTERING"
                    elif 1 in interacts:
                        p_type = "REFLECTION"
                    else:
                        p_type = "UNKNOWN"
                        
                    verts = [[float(cell.x), float(cell.y), float(cell.altitude_m)]]
                    for d in range(vertices_np.shape[0]):
                        if interacts[d] != 0:
                            v = vertices_np[d, i, j, p_idx]
                            verts.append([float(v[0]), float(v[1]), float(v[2])])
                    verts.append([float(ue.mobility.x), float(ue.mobility.y), float(ue.mobility.z)])
                    
                    rp = RayPath(
                        tx_id=str(cell.cell_id),
                        rx_id=str(ue.ue_id),
                        tx_type=cell.cell_type,
                        rx_type="UE",
                        path_vertices=verts,
                        path_type=p_type,
                        path_power_db=p_power_db,
                        path_delay_s=p_tau,
                        los_flag=los_flag
                    )
                    cell_rays.append(rp)
                
                ray_paths_per_cell[ue.ue_id][cell.cell_id] = cell_rays

        results = {}
        for i, ue in enumerate(ues):
            sinr, interf = self._sinr_per_band(rsrp_np[i])
            results[ue.ue_id] = self._build_snapshot(
                ue, rsrp_np[i], pl_np[i], sinr, interf, d3d_all[i],
                load_arr, serving_map, timestamp_s,
                ray_paths_per_cell.get(ue.ue_id, {})
            )
        return results

    # ------------------------------------------------------------------
    # Shared: per-band interference & SINR
    # ------------------------------------------------------------------
    def _sinr_per_band(self, rsrp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SINR with per-band interference isolation."""
        power_mw = np.power(10.0, rsrp / 10.0)
        sinr = np.full(len(self.cells), -20.0)
        interf_dbm = np.full(len(self.cells), -150.0)
        noise_mw = 10.0 ** (-101.0 / 10.0)

        for _, indices in self._band_groups.items():
            band_power = power_mw[indices]
            total = np.sum(band_power)
            for local_i, cell_idx in enumerate(indices):
                i_lin = max(total - band_power[local_i], 1e-15)
                interf_dbm[cell_idx] = 10.0 * math.log10(i_lin)
                sinr_lin = band_power[local_i] / (i_lin + noise_mw)
                sinr[cell_idx] = 10.0 * math.log10(max(sinr_lin, 1e-12))
        return sinr, interf_dbm

    # ------------------------------------------------------------------
    # Shared: snapshot assembly
    # ------------------------------------------------------------------
    def _build_snapshot(
        self, ue: UEState,
        rsrp: np.ndarray, path_loss: np.ndarray,
        sinr: np.ndarray, interference_dbm: np.ndarray,
        d3d: np.ndarray, cell_loads: np.ndarray,
        serving_map: Dict[str, Optional[int]],
        timestamp_s: float,
        ray_paths_per_cell: Optional[Dict[int, List[RayPath]]] = None,
    ) -> MeasurementSnapshot:

        rsrq = np.clip(sinr - 8.5, -19.5, -3.0)
        cqi = np.array([self._sinr_to_cqi(v) for v in sinr])
        throughput = np.array([
            self._throughput_estimate(self.cells[i], sinr[i], cell_loads[i])
            for i in range(len(self.cells))])
        score = self._scores(rsrp, sinr, cell_loads)

        # Visibility filter
        dx = self._cell_x - ue.mobility.x
        dy = self._cell_y - ue.mobility.y
        d2d = np.sqrt(dx * dx + dy * dy)
        visible = d2d <= self.config.filter_radius_m
        if not np.any(visible):
            visible[np.argmin(d2d)] = True
        vis_idx = np.flatnonzero(visible)

        order = vis_idx[np.argsort(score[vis_idx])[::-1]]
        top_idx = order[: self.config.k_neighbors]

        by_cell_id = {}
        visible_cells = []
        for idx in order:
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
                load=float(cell_loads[idx]),
                throughput_mbps=float(throughput[idx]),
                score=float(score[idx]),
                interference_dbm=float(interference_dbm[idx]),
                ray_paths=ray_paths_per_cell.get(cell.cell_id, []) if ray_paths_per_cell else []
            )
            by_cell_id[cell.cell_id] = meas
            visible_cells.append(meas)

        serving_id = serving_map.get(ue.ue_id)
        if serving_id is None or serving_id not in by_cell_id:
            serving_id = int(self.cells[top_idx[0]].cell_id)

        best_lte_id = next(
            (self.cells[i].cell_id for i in order if self.cells[i].net_type.startswith("LTE")), None)
        best_nr_id = next(
            (self.cells[i].cell_id for i in order if self.cells[i].net_type == "5G NR"), None)
        sm = by_cell_id[serving_id]

        latency = max(3.0, 8.0 + sm.load * 18.0 + max(0.0, -sm.sinr_db) * 0.25)
        pkt_loss = min(1.0, 0.002 + max(0.0, -sm.sinr_db - 1.0) * 0.003 + sm.load * 0.02)
        jitter = 0.6 + sm.load * 2.4 + max(0.0, -sm.sinr_db) * 0.08

        return MeasurementSnapshot(
            ue_id=ue.ue_id, timestamp_s=timestamp_s,
            serving_cell_id=serving_id,
            visible_cells=visible_cells,
            top_neighbors=[by_cell_id[self.cells[i].cell_id] for i in top_idx],
            by_cell_id=by_cell_id,
            best_lte_id=best_lte_id, best_nr_id=best_nr_id,
            latency_ms=float(latency),
            packet_loss=float(pkt_loss),
            jitter_ms=float(jitter),
            interference_dbm=float(sm.interference_dbm),
        )

    # ------------------------------------------------------------------
    # Utilities (unchanged, well-tested)
    # ------------------------------------------------------------------
    def _scores(self, rsrp: np.ndarray, sinr: np.ndarray, load: np.ndarray) -> np.ndarray:
        rn = np.clip((rsrp + 125.0) / 81.0, 0.0, 1.0)
        sn = np.clip((sinr + 10.0) / 30.0, 0.0, 1.0)
        ln = 1.0 - np.clip(load, 0.0, 1.0)
        return 0.50 * rn + 0.25 * sn + 0.25 * ln

    def _link_budget(self, ue: UEState, cell: Cell, d2d_m: float, d3d_m: float) -> tuple:
        d2d_m = max(d2d_m, 1.0)
        d3d_m = max(d3d_m, 1.0)
        p_los = self._los_prob_umi(d2d_m)
        state = self._next_link_state(ue, cell, p_los)
        pl_los = 32.4 + 21.0 * math.log10(d3d_m) + 20.0 * math.log10(cell.frequency_ghz)
        pl_nlos = max(pl_los, 22.4 + 35.3 * math.log10(d3d_m) + 21.3 * math.log10(cell.frequency_ghz))
        base_loss = pl_los if state.is_los else pl_nlos
        total_loss = base_loss + state.shadow_db + self._sidelobe_penalty(cell, d2d_m)
        rsrp = cell.tx_power_dbm - total_loss + state.fast_db
        rsrp = min(self.config.rsrp_ceiling_dbm, max(self.config.rsrp_floor_dbm, rsrp))
        return total_loss, rsrp

    def _next_link_state(self, ue: UEState, cell: Cell, p_los: float) -> _LinkState:
        key = (ue.ue_id, cell.cell_id)
        prev = self.link_states.get(key)
        if prev is None:
            is_los = bool(self.rng.random() < p_los)
            ss = 4.0 if is_los else 7.82
            state = _LinkState(
                shadow_db=float(self.rng.normal(0.0, ss)),
                fast_db=float(self.rng.normal(0.0, 1.2 if is_los else 2.0)),
                is_los=is_los,
                last_ue_x=ue.mobility.x, last_ue_y=ue.mobility.y,
                last_bs_x=cell.x, last_bs_y=cell.y)
            self.link_states[key] = state
            return state

        disp = math.hypot(ue.mobility.x - prev.last_ue_x, ue.mobility.y - prev.last_ue_y) + \
               math.hypot(cell.x - prev.last_bs_x, cell.y - prev.last_bs_y)
        if disp <= 1e-9:
            return prev

        is_los = prev.is_los
        if disp > 15.0 and self.rng.random() < min(1.0, disp / 60.0):
            is_los = bool(self.rng.random() < p_los)
        ss = 4.0 if is_los else 7.82
        rho_sf = math.exp(-disp / 20.0)
        rho_ff = math.exp(-disp / 2.0)
        shadow = rho_sf * prev.shadow_db + math.sqrt(max(0.0, 1 - rho_sf**2)) * float(self.rng.normal(0, ss))
        fast = rho_ff * prev.fast_db + math.sqrt(max(0.0, 1 - rho_ff**2)) * float(self.rng.normal(0, 1.2 if is_los else 2.0))
        state = _LinkState(shadow_db=shadow, fast_db=fast, is_los=is_los,
                           last_ue_x=ue.mobility.x, last_ue_y=ue.mobility.y,
                           last_bs_x=cell.x, last_bs_y=cell.y)
        self.link_states[key] = state
        return state

    def _los_prob_umi(self, d2d: float) -> float:
        return max(0.0, min(1.0, (18.0/d2d)*(1-math.exp(-d2d/36.0)) + math.exp(-d2d/36.0)))

    def _sinr_to_cqi(self, sinr_db: float) -> int:
        thr = [-6.7,-4.7,-2.3,0.2,2.4,4.7,6.9,8.1,10.3,11.7,14.1,16.3,18.7,21.0,22.7]
        cqi = 0
        for i, t in enumerate(thr, 1):
            if sinr_db >= t:
                cqi = i
        return min(cqi, 15)

    def _throughput_estimate(self, cell: Cell, sinr_db: float, load: float) -> float:
        se = math.log2(1.0 + max(10.0**(sinr_db/10.0), 1e-6))
        return max(0.0, cell.peak_throughput_mbps * 0.30 * min(1.0, se/7.0) * max(0.02, 1.0 - load))

    def _sidelobe_penalty(self, cell: Cell, d2d: float) -> float:
        if not cell.is_drone or cell.altitude_m < 1.0:
            return 0.0
        elev = math.degrees(math.atan2(cell.altitude_m, max(d2d, 1.0)))
        if elev < 30.0:
            return 0.0
        if elev < 60.0:
            return ((elev - 30.0) / 30.0) * 15.0
        return 15.0 + ((elev - 60.0) / 30.0) * 10.0
