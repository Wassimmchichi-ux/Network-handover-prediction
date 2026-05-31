import numpy as np
import pandas as pd
import tensorflow as tf
from sionna.sys.topology import gen_hexgrid_topology

def generate_hexagonal_grid(num_rings, isd):
    """
    Generates a hexagonal grid using Sionna's native topology helper.
    Explicitly structures sectors as children of physical sites.
    """
    topo = gen_hexgrid_topology(
        batch_size=1,
        num_rings=num_rings,
        num_ut_per_sector=1,
        scenario="umi",
        isd=isd
    )

    bs_loc = topo[1][0].numpy()  # [num_cells*3, 3]
    num_sectors = bs_loc.shape[0]

    towers = []
    for i in range(num_sectors):
        site_id = i // 3
        sector_id = i % 3

        # ALL sectors at a site share the same physical MAST location
        x, y, z = bs_loc[i]

        is_5g = (site_id % 2 == 0)

        towers.append({
            "site_id": 100 + site_id,
            "sector_id": sector_id,
            "cell_id": 1000 + i,
            
            # Explicit physical coordinates for the shared site structure
            "site_x": float(x),
            "site_y": float(y),
            "site_z": float(z),
            
            # Legacy/Mapping coordinates
            "x": float(x),
            "y": float(y),
            "altitude_m": float(z),

            "azimuth_deg": sector_id * 120.0,
            "net_type": "5G NR" if is_5g else "LTE",
            "frequency_ghz": 3.5 if is_5g else 2.1,
            "tx_power_dbm": 43.0 if is_5g else 46.0,
            "city": "HexCity"
        })

    return pd.DataFrame(towers)

if __name__ == "__main__":
    df = generate_hexagonal_grid(num_rings=3, isd=500.0)
    df.to_csv("hexagonal_towers.csv", index=False)
    print(f"Generated {len(df)} sectors across {len(df)//3} physical sites.")
