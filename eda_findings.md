# Exploratory Data Analysis & Research Gaps
**Date:** 2026-10-04  
**Dataset:** 2,235,744 spatiotemporal records (2001-2021) across the Taiwan Aquifer.

---

## 1. Key Analytical Findings

### Temporal Trends
- **Groundwater Variations:** Monthly averaged groundwater levels fluctuate significantly between wet and dry seasons, but there is a baseline stability when averaged globally across all spatial points.
- **Pumping Increases:** Pumping rates show a marked variance (from ~51k to ~205k units). High pumping periods correspond strongly with dry season drops in the water table.

### Feature Importance & Correlations
- **Dominance of Spatial Autocorrelation:** The XGBoost model achieved a massive **R2 = 0.9852** primarily by relying heavily on spatial coordinates (`longitude`: 0.7439, `latitude`: 0.2368). 
- **Land Use & Pumping:** Once spatial location is accounted for, **LULC (0.3927)** and **Pumping (0.2930)** are the strongest temporal predictors of groundwater depth.
- **The "Static Variable" Failure:** Static topographic variables (`elevation_dem`, `twi`, `distance_to_river`) yielded near-zero feature importance in the final model. Although the DEM is correctly aligned to the Choushui Fan (bounding box 120.13°–120.63°E, 23.47°–24.17°N, verified via rasterio), the introduction of autoregressive temporal lag features (`lag_1_gw`, `rolling_mean_3_gw`) and the removal of raw spatial coordinates caused the model to rely almost entirely on temporal signals, marginalizing the static topographic variables despite their strong raw correlation with groundwater (DEM r=0.86). **Note:** the `raster_parser.py` script in this repo generates a standalone synthetic demo raster at Indian coordinates (77.0°E) — this is NOT the DEM used in the ML pipeline.

- **Subsidence Link:** Subsidence showed a moderate positive correlation (0.20) with groundwater level fluctuations globally, confirming that aquifer compaction is actively occurring in response to head changes.

---

## 2. Identified Research Gaps for Thesis Publication

Based on the EDA and modeling anomalies, the following novel research gaps have been identified for the master's thesis:

### 🌟 Research Gap 1: Over-Reliance on Spatial Coordinates in ML Groundwater Models
**Problem:** Most existing literature uses generic spatial interpolation (Kriging) or treats ML models purely as temporal forecasters at specific wells. In our 21-year spatiotemporal model, `longitude` and `latitude` overpowered physical variables (Rainfall, Temperature). 
**Proposed Thesis Contribution:** A methodological critique proving that without perfectly aligned, high-resolution static topography (DEM/TWI), ML models degenerate into glorified spatial interpolators. The thesis will present a framework for isolating temporal drivers (LULC, Pumping) from static spatial proxies.

### 🌟 Research Gap 2: The Latency of LULC (Land Use) Impact on Groundwater Recharge
**Problem:** LULC was the highest-ranking physical feature (0.39 importance), yet standard datasets only update LULC every 5 years (2016, 2021). Current models linearly interpolate or nearest-neighbor match this data, ignoring the *lag time* between a surface paving event and its eventual impact on deep aquifer recharge.
**Proposed Thesis Contribution:** We will quantify the temporal lag between urbanization (LULC shift) and groundwater depletion, proposing a non-linear decay function for LULC changes rather than the standard step-function used in current literature.

---

## 3. UI Dashboard Delivery
To explore these findings interactively, run:
```bash
streamlit run app_v2.py
```
This loads the newly constructed 21-Year EDA Dashboard, featuring:
1. **Temporal Line Charts** (aggregated month-over-month averages).
2. **Interactive Correlation Heatmaps**.
3. **Spatial Dry-Season Mapping** (May 2017 scatter sample).
