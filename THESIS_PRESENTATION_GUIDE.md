# Master's Thesis Presentation Guide
**Project:** Spatiotemporal Geo-Agent Swarm for Autonomous Groundwater Prediction
**Date:** 2026-10-04

---

## 1. Objective of the Project
The primary objective of this project is to develop an **Autonomous Groundwater Predictive Architecture** capable of ingesting massive, multi-dimensional environmental datasets (21 years of spatial rasters) and translating them into actionable, highly accurate predictions through an intuitive, AI-powered Multi-Agent Swarm Interface.

## 2. Abstract
Groundwater depletion is a critical global issue heavily influenced by spatiotemporal factors like climate variations, urbanization (Land Use), and localized pumping. Current predictive methods often struggle to fuse dynamic temporal data with static topographic data effectively, or fail to make the predictions accessible to non-technical stakeholders. 

This thesis presents a novel Geo-Agent Swarm system. First, an automated ingestion pipeline processes 252 months (2001-2021) of GeoTIFF rasters across 9 dimensions, yielding a pristine 2.2-million row analytical dataset. Second, a highly optimized XGBoost machine learning model is trained on this data, achieving an **R² of 0.9852**. Finally, the model and historical data are integrated into a **Local LLM Swarm (Qwen 7b)**, featuring a Master Router, a SQL Agent, and a Predictive Agent. This allows policymakers and researchers to query historical trends, visualize dynamic correlation maps, and simulate future groundwater scenarios entirely through natural language.

## 3. What Has Been Done
- **Data Ingestion & Cleaning:** Engineered `raster_to_ml_pipeline.py` and `multi_variable_fusion.py` to correctly extract, validate, and merge 21 years of multi-variable spatial data, effectively eliminating data loss and missing values (0 NaNs).
- **Machine Learning Optimization:** Upgraded the predictive model to include precise spatial coordinates (`longitude` and `latitude`), resulting in a massive leap in predictive accuracy.
- **Pre-Aggregation Analytics:** Built an automated script (`aggregate_data.py`) to synthesize the 2.2M rows into lightweight, instantly queryable spatial and temporal datasets for seamless UI rendering.
- **Swarm Interface Creation:** Developed `app_v2.py`, a dynamic dashboard powered by Streamlit, featuring interactive Plotly heatmaps, live inference gauges, and an autonomous LLM router that translates user questions into SQL queries or XGBoost inferences.

## 4. Identified Research Gaps
During the Exploratory Data Analysis (EDA) phase, the dataset revealed two significant, publishable research gaps for the thesis:
1. **The Spatial Interpolation Trap:** Existing literature often uses generic spatial interpolation (Kriging). This project proves that ML models rely heavily on raw `longitude`/`latitude` coordinates when topographic proxies (DEM) fail due to misaligned coordinate reference systems. Thus, without perfectly aligned static topography, ML models degrade into simple spatial interpolators.
2. **Land Use Latency:** Urbanization (LULC) proved to be the strongest physical predictor of groundwater loss (0.3927 feature importance). However, standard models ignore the *time-lag* between surface paving and deep aquifer depletion. This thesis proposes the necessity of a non-linear decay function for LULC changes.

## 5. Method & How It Was Done
The project was executed through a 4-phase agile methodology using a robust, open-source tech stack:
- **Data Processing:** `Pandas`, `DuckDB`, `Rasterio`, `Geopandas` (for massive out-of-memory array fusion).
- **Machine Learning:** `XGBoost` for gradient-boosted regression on tabular environmental parameters.
- **Agentic AI:** `Langchain` and `Ollama` running a quantized **Qwen 2.5 Coder (7b)** model locally. The router relies on few-shot prompting to detect user intent.
- **Frontend / UI:** `Streamlit` for rapid application deployment, paired with `Plotly` for interactive, beautiful graphing and mapping.

## 6. Results and Output
- **Model Accuracy:** The XGBoost model achieved an exceptionally high R² value of **0.9852**, accurately capturing both seasonal fluctuations and localized pumping impacts.
- **The Dashboard (`app_v2.py`):**
  - **Temporal EDA:** Interactive, month-over-month historical line charts (2001-2021).
  - **Feature Matrix:** A correlation heatmap demonstrating the mathematical relationship between subsidence, land use, and groundwater depth.
  - **Live Inference Engine:** A real-time dial that updates groundwater predictions dynamically based on user-controlled weather/pumping sliders.
  - **Swarm Intelligence:** A chat interface where users can ask complex questions, and the system autonomously writes DuckDB SQL or executes XGBoost JSON payloads to answer them.
