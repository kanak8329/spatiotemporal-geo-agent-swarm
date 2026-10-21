# Architecture — Spatiotemporal Geo-Agent Swarm

The system is organized into a modular, multi-tier computational architecture that bridges multi-modal geospatial data, physics-grounded machine learning, reinforcement learning optimization, and local agentic orchestration.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & V&V                            │
│  unified_data_extractor.py ──> validation_pipeline.py ──> data/validated│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌───────────────────────┐                         ┌──────────────────────┐
│  PRECOMPUTED TABLES   │                         │ PHYSICAL SIMULATOR   │
│   aggregate_data.py   │                         │   ml_predictor.py    │
│ (monthly, corr, eda)  │                         │ (XGBoost World Model)│
└──────────┬────────────┘                         └──────────┬───────────┘
           │                                                 │
           │                                                 ▼
           │                                      ┌──────────────────────┐
           │                                      │  POLICY OPTIMIZER    │
           │                                      │   rl_trainer.py      │
           │                                      │ (Stable-Baselines3)  │
           │                                      └──────────┬───────────┘
           │                                                 │
           ▼                                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 SCIENTIFIC VALIDATION & GEODETIC ADVANCES              │
│  zone_classifier.py │ gnss_point_predictor.py │ spatial_cv_evaluator.py│
│  subsidence_coupling_analyzer.py (Sentinel-1 InSAR + THSR Buffer)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT SWARM & UI DASHBOARD                    │
│   app_v2.py (Streamlit 7-Tab Dashboard)                                │
│   ├─ 💬 Swarm Intelligence (Master Router + DuckDB + XGBoost + Chat)   │
│   ├─ 📈 Dynamic Data Comparator (21-Yr Temporal Trends & Heatmaps)     │
│   ├─ 🔮 Live Inference (XGBoost World Simulation Dial)                 │
│   ├─ 🤖 RL Policy Dashboard (PPO Pumping Policy Actions)               │
│   ├─ 🛡️ V&V Audit (Data Quality Logs & Quarantines)                    │
│   ├─ 🏞️ Hydrogeological Zonation & 26 GNSS Stations                   │
│   └─ 🛰️ InSAR Subsidence & THSR Infrastructure Hazards                 │
└────────────────────────────────────────────────────────────────────────┘
```

## Key Architectural Principles

1. **Decoupled Data Lakehouse (`DuckDB` + `GeoParquet`):**
   - 2.23M spatiotemporal records (2001–2021) partitioned by year (`data/validated/year=YYYY/`).
   - DuckDB views allow sub-second OLAP queries without loading the entire dataset into RAM.

2. **Physical World Surrogate Model (`XGBoost`):**
   - Coordinates (`longitude`, `latitude`) are stripped to prevent the Spatial Interpolation Trap.
   - Dynamic temporal features (`lag_1_gw`, `rolling_mean_3_gw`) and physical environmental variables drive predictions ($R^2 > 0.98$).

3. **Deterministic 4-Stage V&V Gate (`validation_pipeline.py`):**
   - Schema enforcement, statistical outlier detection ($Z$-score), physical domain law compliance, and human-in-the-loop (HITL) quarantine routing.

4. **Reinforcement Learning Policy (`Stable-Baselines3 PPO`):**
   - Formulated as an MDP where the XGBoost model acts as the environmental transition simulator $s' = f(s, a)$.
   - Balances groundwater sustainability, ground subsidence prevention, and municipal water extraction.

5. **Multi-Agent Orchestrator (`Qwen 2.5 Coder 7B`):**
   - Natural language master router dynamically dispatches requests to SQL querying, physical scenario simulation, or hydrogeological dialogue.
