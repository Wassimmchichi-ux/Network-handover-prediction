This package is a new standalone control layer for the hybrid NSA handover pipeline.

Files:

- `config.py`: Morocco-baseline simulation parameters and the dataset schema.
- `mobility_engine.py`: UE and drone mobility loop with `10 ms` stepping.
- `channel_model.py`: 3GPP TR 38.901 UMi-style measurement engine with correlated fading fallback.
- `handover.py`: A3 + A4 + A5, conditional preparation, dual-connectivity-aware execution, ping-pong detection, and RLF handling.
- `drone_controller.py`: load and RF aware aerial-BS waypoint logic.
- `dataset_builder.py`: exact `schema-handover.csv` column order.
- `sync_manager.py`: end-to-end timestamp loop.

Run:

```bash
python run_hybrid_pipeline.py
```

Small smoke run:

```bash
python run_hybrid_pipeline.py \
  --num-ground-cells 20 \
  --num-drones 4 \
  --num-ues 16 \
  --sim-time 2
```

Static channel validation:

```bash
python validate_hybrid_pipeline.py
```
