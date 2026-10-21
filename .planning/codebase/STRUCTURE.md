# Structure
- `app_v2.py`: The main Streamlit UI and Swarm Agent router.
- `raster_to_ml_pipeline.py`: Base extraction of GeoTIFFs to CSV/Parquet.
- `multi_variable_fusion.py`: Spatiotemporal joining of extracted features.
- `aggregate_data.py`: Pre-aggregation of the 2.2M dataset for UI speed.
- `ml_predictor.py`: Training script for the XGBoost model.
- `ml_master_thesis_data_v2.parquet`: The core verified ground truth dataset.
- `xgboost_groundwater_model.pkl`: The trained predictive model (R2=0.98).
- `brag-output/`: Contains the launch video and presentation assets.
