# Concerns
- **Topographic Misalignment:** Static variables (DEM, TWI) were misaligned (different bounding boxes), resulting in `nodata` overlap and forcing the ML model to rely heavily on raw coordinates.
- **LULC Latency:** Current dataset linearly interpolates 5-year LULC updates, ignoring physical lag times in aquifer recharge.
- **Memory Footprint:** Loading 2.2M rows into Streamlit causes crashes; relying on pre-aggregation (`aggregate_data.py`) is mandatory.
