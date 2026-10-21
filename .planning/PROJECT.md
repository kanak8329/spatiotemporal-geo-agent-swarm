# 🌊 Spatiotemporal Geo-Agent Swarm for Groundwater Prediction
**Master's Thesis Project — GSD Project Context**

---

## Project Overview

A multi-agent AI system for spatiotemporal groundwater level prediction and analysis, built on 21 years of multi-variable GeoTIFF raster data from Taiwan's aquifer system (2001–2021). The system combines XGBoost machine learning, DuckDB spatial analytics, graph topology analysis, and RAG-based document querying — all orchestrated through a Streamlit interface and an LLM-based master router.

**Thesis Goal:** Demonstrate a novel Geo-Agent Swarm architecture capable of fusing heterogeneous geospatial data sources to predict groundwater levels and surface deformation, identify research gaps, and support academic publication.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (app.py) |
| ML Model | XGBoost (xgboost_groundwater_model.pkl) |
| Database | DuckDB + Parquet (GeoParquet) |
| LLM Router | Ollama / Qwen2.5-Coder 7b |
| Data Sources | GeoTIFF rasters: Groundwater, Rainfall, Temperature, Pumping, LULC, DEM, TWI, Distance-to-River, Subsidence |
| Data Coverage | 2001-2021 monthly (252 files per variable, ~9 variables) |
| Graph Layer | NetworkX (graph_topology.py) |
| Ingestion | raster_parser.py, raster_to_ml_pipeline.py, ingestion_pipeline.py |

---

## Current Status (Milestone 2 start)

### Working
- Streamlit UI (app.py) — 3 tabs: Swarm Intelligence, Data Comparator, Live Inference
- Agent routing logic (LLM -> SQL / PREDICT / GENERAL)
- DuckDB view over parquet (when data is correct)
- XGBoost model file exists
- All GeoTIFF source files present (2001-2021, all variables)

### CRITICAL BUGS
- Raster pipeline only reads 2001 data — other years (2002-2021) NOT ingested
- Missing values throughout parquet — multi-variable fusion incomplete
- Model NOT retrained on full 2001-2021 dataset
- No temporal visualization of the spatiotemporal trends

### Research Gap
- No systematic EDA of the full 21-year multi-variable dataset
- Unknown correlation structure between predictors and groundwater level
- No temporal trend analysis or seasonal decomposition

---

## Constraints

- Deadline: < 2 weeks (thesis defense / paper submission)
- Hardware: Local GPU GTX 1660 Ti, local Ollama inference
- Goal: Publishable paper identifying research gaps in Taiwan aquifer dynamics

---

## Success Criteria

**Milestone 1 (Complete):**
1. Full pipeline ingests all 252 monthly GeoTIFFs per variable (2001-2021)
2. XGBoost model trained on complete multi-variable dataset, no missing values
3. Streamlit app shows temporal visualizations of all variables
4. Research gap identified and documented
5. System demo-ready for thesis committee

**Milestone 2 (Active):**
1. ML model learns physical parameters (fixing false predictions and spatial bias).
2. Terminal visualizations implemented showing live training progress and metrics.
3. 100% of dataset features and rows are utilized correctly in training.
4. 4-stage V&V pipeline (Schema → Statistical → Physical → HITL Gate) replaces qa_supervisor.py.
5. Predictive anomaly detection via XGBoost residual analysis.
6. Scalable data ingestion manager with partitioned Parquet storage.
7. RL environment (Gymnasium) wrapping XGBoost world model for pumping policy optimization.
8. RL training pipeline (PPO, Stable Baselines3) with TensorBoard logging.
9. RL Policy Dashboard + Data Quality Audit tabs in Streamlit.

