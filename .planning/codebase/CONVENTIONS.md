# Conventions
- **Data Engineering:** No missing values permitted in ML dataset. Forward-fill/backward-fill used for sparse temporal data (e.g., LULC, Subsidence).
- **Spatial Features:** `longitude` and `latitude` are mandatory features for XGBoost prediction.
- **UI:** Streamlit apps use dark mode (`#0E1117`) and high-contrast green (`#39ff14`) to indicate active predictions.
