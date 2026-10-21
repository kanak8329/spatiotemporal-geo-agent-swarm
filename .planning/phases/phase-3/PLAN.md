# Phase 3 Plan: Temporal EDA & Research Gap Report

## 🎯 Objective
Upgrade the Streamlit UI to display full 21-year time series trends, compute correlation matrices, and generate a written EDA report (`eda_findings.md`) identifying at least 2 research gaps for the thesis.

## 📋 Context
Now that Phase 1 (Data Ingestion) and Phase 2 (XGBoost Retraining) are complete, we have a robust, 2.2M-row spatiotemporal dataset and a highly accurate model (R2=0.9852). We need to leverage this data to find publishable insights and upgrade the UI to showcase the historical data visually.

## 🛠️ Step-by-Step Implementation Plan

### Step 1: Upgrade Streamlit UI (`app.py`) for Temporal EDA
- **Action**: Add a new tab or expand the "Data Comparator" tab in `app.py`.
- **Details**:
  - Load a subset (or aggregated version) of `ml_master_thesis_data_v2.parquet` to keep the UI snappy.
  - Implement a multiselect/dropdown to choose a variable (Groundwater, Pumping, LULC, etc.).
  - Render a month-over-month line chart from 2001 to 2021 using `st.line_chart` or `plotly`.
  - Add a correlation heatmap matrix for all 9 variables (excluding coords) to visualize relationships.
- **Verification**: Run `streamlit run app.py` and ensure the UI renders the charts and heatmap without memory crashes.

### Step 2: Compute Spatial Heatmaps (Optional/Bonus UI)
- **Action**: Add a spatial mapping component.
- **Details**: 
  - Allow the user to select a specific month (e.g., 2017-05).
  - Render a basic scatter map (using `st.map` or `pydeck`) coloring points by groundwater level or pumping to show spatial distributions.

### Step 3: Analytical EDA & Anomaly Detection
- **Action**: Run a Python script (`eda_analysis.py` or inline) to analyze the dataset.
- **Details**:
  - Calculate overall trends (e.g., is groundwater dropping globally over 21 years?).
  - Identify seasonal patterns (wet vs dry season differences).
  - Find anomalies or massive shifts (e.g., LULC changes, pumping spikes).

### Step 4: Write Research Gap Report
- **Action**: Create `eda_findings.md` documenting the results of Step 3.
- **Details**:
  - Summarize the strongest correlations (e.g., pumping vs groundwater).
  - Identify exactly 2-3 **Research Gaps** for the thesis (e.g., "While pumping drives groundwater decline, the exact lag time between LULC shifts and recharge rate changes remains unquantified," or "Current models ignore spatial non-stationarity of rainfall effectiveness").
- **Verification**: Review `eda_findings.md` against Phase 3 UAT.

## ✅ UAT (Acceptance Criteria)
- [ ] Streamlit shows month-over-month line chart for any variable.
- [ ] Correlation heatmap renders in the Streamlit UI.
- [ ] `eda_findings.md` exists and identifies >= 2 distinct, publishable research gaps.
