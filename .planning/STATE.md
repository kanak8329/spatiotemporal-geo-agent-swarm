# STATE — Project Memory
_Last updated: 2026-10-06_

---

## Current Phase
**Milestone 2 Active — Phase 5: ML Pipeline Overhaul, V&V Pipeline & Anomaly Detection**
(Milestone 1 is 100% COMPLETE. 🎉)

## Decisions Made
- Full 21-year ingestion (2001-2021) is COMPLETE and clean (2.2M rows).
- XGBoost Model retrained successfully including longitude/latitude, achieving R2 = 0.9852.
- Research gap focus: temporal trend + correlation analysis of multi-variable aquifer data.
- UI replaced with `app_v2.py` featuring a 3-tab temporal EDA dashboard + live Swarm AI.
- Target: publishable paper in hydrogeology / remote sensing journal.

### RL Framework Architecture Decisions (2026-10-06)
- **RL Algorithm:** PPO via Stable Baselines3 (continuous action space, GPU-compatible).
- **World Model:** Existing XGBoost used as dynamics predictor `s' = f(s, a)` — no physics simulator needed.
- **MDP Design:** State = 12 env features + derived; Action = pumping adjustment [-1,+1]; Reward = sustainability - subsidence penalty - volatility.
- **V&V Pipeline:** 4-stage (Schema → Statistical → Physical → HITL Gate) replacing qa_supervisor.py.
- **Anomaly Detection:** XGBoost residual-based predictive flagging (catches sensor drift).
- **Dashboard:** TensorBoard for training + 2 new Streamlit tabs (RL Policy + Data Quality Audit).
- **Critical Prerequisite:** Must remove lon/lat spatial overfitting from XGBoost BEFORE RL training.
- **Architecture Document:** See `rl_framework_architecture.md` in conversation artifacts.

## Known Issues
- XGBoost feature importance: 74% from longitude, 24% from latitude — model is a spatial interpolator, not a physical learner.
- Static vars (DEM, TWI, distance_to_river) have 0.0 importance due to bounding box mismatch (India vs Taiwan).
- qa_supervisor.py uses LLM for validation — needs replacement with deterministic V&V pipeline.

## Data Inventory
- `ml_master_thesis_data_v2.parquet` (2,235,744 rows, 12 columns, zero NaNs)
- `xgboost_groundwater_model.pkl` (High accuracy R2=0.98, but spatially biased)
- 9 distinct feature variables loaded perfectly.
- Pre-aggregated parquets (`monthly_aggregate.parquet`, `spatial_sample_201705.parquet`) for instantaneous UI load.

## Next Action
Plan Phase 5 — ML Pipeline Overhaul, V&V Pipeline & Anomaly Detection.
