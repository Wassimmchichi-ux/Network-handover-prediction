#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

# IMPORTANT: Importing from the NEW folder
from hybrid_pipeline_sionna import SimulationConfig, SyncManager
from hybrid_pipeline_sionna.ns3_bridge_runner import Ns3BridgeRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SIONNA-BACKED hybrid NSA mobility + handover pipeline.")
    parser.add_argument("--tower-csv", type=Path, default=Path("hexagonal_towers.csv"))
    parser.add_argument("--output", type=Path, default=Path("hybrid_handover_dataset_sionna.csv"))
    parser.add_argument("--drone-output", type=Path, default=Path("hybrid_drone_positions_sionna.csv"))
    parser.add_argument("--summary", type=Path, default=Path("hybrid_handover_summary_sionna.json"))
    parser.add_argument("--num-ground-cells", type=int, default=300)
    parser.add_argument("--num-drones", type=int, default=100)
    parser.add_argument("--num-ues", type=int, default=300)
    parser.add_argument("--sim-time", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-sionna", action="store_true")
    parser.add_argument("--ns3-root", type=Path, default='/home/wassimmchichi/ns3/ns-allinone-3.41/ns-3.41')
    parser.add_argument("--ns3-endpoint", type=str, default="tcp://127.0.0.1:5558") # Changed port to avoid conflict
    parser.add_argument("--no-auto-install-ns3-bridge", action="store_true")
    parser.add_argument("--no-auto-build-ns3-bridge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        tower_csv=args.tower_csv,
        output_csv=args.output,
        drone_output_csv=args.drone_output,
        summary_json=args.summary,
        num_ground_cells=args.num_ground_cells,
        num_drones=args.num_drones,
        num_ues=args.num_ues,
        sim_time_s=args.sim_time,
        seed=args.seed,
        enable_sionna=not args.disable_sionna,
    )
    if args.ns3_root is not None:
        summary = Ns3BridgeRunner(
            config,
            ns3_root=args.ns3_root,
            endpoint=args.ns3_endpoint,
            auto_install=not args.no_auto_install_ns3_bridge,
            auto_build=not args.no_auto_build_ns3_bridge,
        ).run()
    else:
        summary = SyncManager(config).run()
        
    print("-" * 30)
    print("SIONNA PIPELINE SUMMARY")
    print("-" * 30)
    print(f"rows_written={summary['rows_written']}")
    print(f"handover_events={summary['handover_events']}")
    print(f"handover_rate={summary['handover_rate']:.4f}")
    if 'ho_rate_per_ue_sec' in summary:
        print(f"ho_rate_per_ue_sec={summary['ho_rate_per_ue_sec']:.4f} (HO/UE/s)")
    print(f"sionna_available={summary['sionna_available']}")
    print(f"channel_backend={summary.get('channel_backend', 'unknown')}")
    print(f"summary_json={summary['summary_json']}")


if __name__ == "__main__":
    main()
