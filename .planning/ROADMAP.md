# ROADMAP — Geo-Agent Swarm Thesis
_Milestone 1: Thesis Completion Sprint_

---

## Phase 1 — Fix Raster Ingestion Pipeline [CRITICAL]
**Status:** ✅ COMPLETE (2026-10-04)
**Effort:** 1-2 days

### Goal
Fix raster_to_ml_pipeline.py (and related parsers) to ingest ALL years (2001-2021) for ALL variables, not just 2001.

### Tasks
- [x] Audit raster_to_ml_pipeline.py — found 5 root-cause bugs in multi_variable_fusion.py
- [x] Fix Subsidence filename: YYYY-MM dash vs YYYY_MM underscore mismatch
- [x] Fix LULC: moved to YEARLY_VARS with nearest-year matching (2016/2021 only)
- [x] Fix static variable merge: removed pre-init causing _x/_y column conflicts
- [x] Fix Pumping 2001-2006 gap: added bfill() after ffill()
- [x] Added end-of-pipeline validation report
- [x] Overwrote ml_master_thesis_data_v2.parquet with clean full dataset

### UAT Results (ALL PASSED ✅)
- PASS: Date range 2001-01-01 to 2021-12-01 (252 unique months)
- PASS: All 9 feature columns present, zero missing values
- PASS: DuckDB post-2001 count = 2,129,280 rows
- PASS: Subsidence populated for 2017+ (529,740 non-zero values)
- PASS: LULC populated for all timestamps
- Total rows: 2,235,744 across 12 clean columns

---

## Phase 2 — Retrain XGBoost on Full Dataset [CRITICAL]
**Status:** ✅ COMPLETE (2026-10-04)
**Effort:** 0.5 days

### Goal
Retrain the XGBoost model on the complete 21-year spatial dataset.

### Tasks
- [x] Load cleaned parquet from Phase 1
- [x] Feature engineering: select 8 predictor columns (added longitude/latitude as spatial features)
- [x] Train/test split (80/20, stratified by year)
- [x] Fit XGBoostRegressor, evaluate R2 and RMSE
- [x] Save new model to xgboost_groundwater_model.pkl
- [x] Log feature importances

### UAT Results (ALL PASSED ✅)
- R2 score = 0.9852 (>= 0.70 on test set)
- RMSE = 2.05 meters
- Model predicts for all features without errors
- Saved as xgboost_groundwater_model.pkl

---

## Phase 3 — Temporal EDA & Research Gap Report [HIGH]
**Status:** ✅ COMPLETE (2026-10-04)
**Effort:** 1-2 days

### Goal
Explore the full 21-year dataset visually and analytically. Identify the research gap for the paper.

### Tasks
- [x] Add temporal line chart tab to Streamlit (all variables, 2001-2021) -> created `app_v2.py` and `aggregate_data.py`
- [x] Compute correlation matrix between all 9 variables
- [x] Generate spatial heatmaps for selected months
- [x] Identify anomalies, trend shifts, seasonal patterns
- [x] Write eda_findings.md with research gap summary

### UAT Results (ALL PASSED ✅)
- Streamlit shows month-over-month line chart (`app_v2.py`)
- Correlation heatmap renders in the dashboard
- `eda_findings.md` exists and identifies 2 distinct research gaps (Over-reliance on coordinates, LULC latency)

---

## Phase 4 — Thesis Demo Polish [MEDIUM]
**Status:** ✅ COMPLETE (2026-10-04)
**Effort:** 1 day

### Goal
Make the full swarm demo bulletproof and visually impressive for the thesis committee.

### Tasks
- [x] Verify all 3 agent routes work end-to-end
- [x] Fix any UI bugs in app.py (Replaced with robust `app_v2.py`)
- [x] Add loading indicators and error handling
- [x] Update README with architecture description
- [x] Prepare 5 demo queries that showcase all agents

### UAT Results (ALL PASSED ✅)
- Interactive Dashboard handles data loading with `try/except` safeguards.
- 5 powerful Demo Queries (SQL, Predictive, Chat) are documented in the `README.md`.
- Architecture and findings summarized clearly for the committee.
- Ready for 10-minute crash-free demo session!

---

## Timeline

| Phase | Target | Notes |
|---|---|---|
| Phase 1 | Day 1-2 | UNBLOCKS everything else |
| Phase 2 | Day 3 | Fast once data is clean |
| Phase 3 | Day 3-5 | Core thesis contribution |
| Phase 4 | Day 6-7 | Polish before defense |
| Buffer | Day 8-14 | Writing, advisor review, contingency |

---

## Phase 5 — ML Pipeline Overhaul, V&V Pipeline & Anomaly Detection [CRITICAL]
**Status:** ✅ COMPLETE (2026-10-06)
**Effort:** 3-4 days

### Goal
Overhaul the ML training to learn physical relationships (not spatial coordinates), build a production-grade 4-stage Verification & Validation pipeline, and implement predictive anomaly detection. This phase is a prerequisite for Phase 6 (RL).

### Tasks
- [x] Refactor `ml_predictor.py` to remove spatial overfitting bias (drop lon/lat as features, add temporal lag features).
- [x] Ensure 100% data utilization from `data/validated/`.
- [x] Implement `rich` terminal progress bars and live metric logging during training.
- [x] Create `validation_pipeline.py` — 4-stage V&V: Schema → Statistical → Physical Constraints → HITL Gate.
- [x] Create `anomaly_detector.py` — XGBoost residual-based predictive anomaly flagging.
- [x] Create `data_ingestion_manager.py` — unified ingestion with partitioned Parquet storage and quarantine.
- [x] Validate model predictions on holdout data to ensure spatial bias is eliminated.
- [x] Export XGBoost as a "world model" API for the RL environment.

### Acceptance Criteria
- Model R² > 0.70 WITHOUT longitude/latitude as features (Achieved: 0.9978).
- V&V pipeline catches injected anomalies (Achieved).
- Anomaly detector flags >90% of known bad records from qa_supervisor test cases (Achieved).
- Terminal training output shows live progress with `rich` (Achieved).

---

## Phase 6 — Reinforcement Learning System & Dashboard [HIGH]
**Status:** ✅ COMPLETE (2026-10-06)
**Effort:** 4-5 days
**Depends on:** Phase 5 (world model + V&V pipeline)

### Goal
Implement a production-ready RL agent for groundwater pumping policy optimization using Stable Baselines3 PPO, with real-time monitoring via TensorBoard and Streamlit dashboard integration.

### Tasks
- [x] Create `rl_config.py` — centralized hyperparameters, reward weights, curriculum config.
- [x] Create `rl_environment.py` — Gymnasium env wrapping XGBoost world model (MDP: state=env vars, action=pumping adjustment, reward=sustainability).
- [x] Create `rl_trainer.py` — PPO training loop with TensorBoard logging, `rich` terminal metrics, checkpoint callbacks.
- [x] Implement curriculum learning (Phase A: stable → Phase B: seasonal → Phase C: drought + subsidence).
- [x] Create `rl_dashboard.py` — new Streamlit tab: reward curves, policy heatmaps, confidence gauge.
- [x] Add V&V audit log tab to `app_v2.py` — validation results table, anomaly map, human review queue.
- [x] Update `automate_mlops.py` to orchestrate V&V → XGBoost → RL training pipeline.
- [x] Save trained RL policy model and document policy interpretation.

### Acceptance Criteria
- RL agent trains for 100K+ timesteps without crashes (Achieved).
- TensorBoard shows improving reward curve over training (Achieved).
- Streamlit dashboard displays policy recommendations given slider inputs (Achieved).
- RL policy achieves better sustainability score than random/greedy baselines (Achieved).

