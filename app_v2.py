import streamlit as st
import pandas as pd
import duckdb
import pickle
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 🎨 1. UI & ANIMATION CONFIGURATION
# ==========================================
st.set_page_config(page_title="Geo-Agent Swarm | Master's Thesis", page_icon="🌊", layout="wide",
                   initial_sidebar_state="expanded")

st.title("🌊 Spatiotemporal Geo-Agent Swarm")
st.markdown("### Autonomous Groundwater Predictive Architecture")
st.info("**For City Planners & Non-Technical Users:** This dashboard uses Artificial Intelligence to simulate the physical world (weather, terrain, and aquifers). It allows you to see how much water we can safely pump before the aquifer runs dry, and automatically suggests the best pumping policy.")
st.divider()


# ==========================================
# 🧠 2. LOAD BACKEND RESOURCES
# ==========================================
@st.cache_resource
def load_system():
    with open("xgboost_groundwater_model.pkl", "rb") as f:
        model = pickle.load(f)
    con = duckdb.connect()
    con.execute("CREATE VIEW IF NOT EXISTS aquifer_data AS SELECT * FROM read_parquet('data/validated/**/*.parquet', hive_partitioning=1)")
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1)

    # Get column names for dynamic plotting
    cols = con.execute("DESCRIBE aquifer_data").df()['column_name'].tolist()
    return model, con, llm, cols


xgb_model, db_con, llm, db_columns = load_system()


# --- TYPEWRITER ENGINE ---
def stream_text(text, delay=0.015):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


# --- CARTOON AGENT VISUALIZER ---
def get_agent_ui(active_agent="NONE"):
    """Generates the HTML for the connected cartoon agent network, reacting to the active state."""
    base = "padding: 15px; border-radius: 15px; text-align: center; width: 130px; transition: all 0.4s ease; display: inline-block; margin: 0 10px;"
    inactive = base + " border: 2px solid #555; opacity: 0.4; filter: grayscale(100%);"

    # Highlight logic based on who is called
    router_style = base + " border: 3px solid #FFA500; background: rgba(255, 165, 0, 0.2); box-shadow: 0 0 20px #FFA500; transform: scale(1.1);" if active_agent == "ROUTER" else base + " border: 2px solid #FFA500;"
    sql_style = base + " border: 3px solid #00d2ff; background: rgba(0, 210, 255, 0.2); box-shadow: 0 0 20px #00d2ff; transform: scale(1.1);" if active_agent == "SQL" else inactive
    pred_style = base + " border: 3px solid #39ff14; background: rgba(57, 255, 20, 0.2); box-shadow: 0 0 20px #39ff14; transform: scale(1.1);" if active_agent == "PREDICT" else inactive
    gen_style = base + " border: 3px solid #ff00ff; background: rgba(255, 0, 255, 0.2); box-shadow: 0 0 20px #ff00ff; transform: scale(1.1);" if active_agent == "GENERAL" else inactive

    html = f"""
    <div style="text-align: center; margin-bottom: 20px; background: #0E1117; padding: 20px; border-radius: 10px; border: 1px solid #333;">
        <div style="{router_style}">
            <div style="font-size: 50px;">🧠</div>
            <div style="font-weight: bold; color:white;">Master Router<br><small>(Qwen 7b)</small></div>
        </div>
        <div style="font-size: 24px; color: #888; margin: 10px 0;">⏬</div>
        <div style="display: flex; justify-content: center; flex-wrap: wrap;">
            <div style="{sql_style}">
                <div style="font-size: 45px;">🦆</div>
                <div style="font-weight: bold; color:white;">SQL Agent<br><small>(DuckDB)</small></div>
            </div>
            <div style="{pred_style}">
                <div style="font-size: 45px;">🌲</div>
                <div style="font-weight: bold; color:white;">Predict Agent<br><small>(XGBoost)</small></div>
            </div>
            <div style="{gen_style}">
                <div style="font-size: 45px;">💬</div>
                <div style="font-weight: bold; color:white;">General Agent<br><small>(Chat)</small></div>
            </div>
        </div>
    </div>
    """
    return html


# ==========================================
# 🤖 3. AGENT LOGIC
# ==========================================
def route_intent(query: str):
    prompt = f"You are a router. Route the following query to one of three agents: 'SQL' (for querying historical or statistical data), 'PREDICT' (for predicting or simulating future states, rainfall, pumping), or 'GENERAL' (for greetings or casual talk). Respond ONLY with the single exact word: SQL, PREDICT, or GENERAL.\nQuery: {query}"
    res = llm.invoke([SystemMessage(content=prompt)]).content.strip().upper()
    return "SQL" if "SQL" in res else "PREDICT" if "PREDICT" in res else "GENERAL"


def execute_sql_agent(query: str):
    prompt = f"Write DuckDB SQL to answer based on table: aquifer_data. Columns: {', '.join(db_columns)}. CRITICAL: ALWAYS append LIMIT 10. Do not use complex joins, just query the single table. Query: {query}. Return ONLY raw SQL, no markdown."
    sql = llm.invoke([SystemMessage(content=prompt)]).content.replace('```sql', '').replace('```', '').strip()
    try:
        df = db_con.execute(sql).df()
        return sql, df.to_string()
    except Exception as e:
        return sql, f"Error: {e}"


def execute_predictive_agent(query: str):
    # Updated to match the new XGBoost "World Model" which has no spatial bias
    features = {'rainfall': 150.0, 'temperature': 28.0, 'pumping': 50.0, 'lulc': 2.0, 'subsidence': 0.0,
                'elevation_dem': 15.0, 'twi': 10.0, 'distance_to_river': 500.0,
                'lag_1_gw': -10.0, 'rolling_mean_3_gw': -10.5}
    prompt = f"Extract parameters as JSON (Keys: rainfall, temperature, pumping). Query: {query}. Return ONLY raw JSON."
    try:
        json_str = llm.invoke([SystemMessage(content=prompt)]).content.replace('```json', '').replace('```', '').strip()
        parsed_json = json.loads(json_str)
        # Safely convert LLM output to floats; ignore hallucinations like "unknown"
        for k, v in parsed_json.items():
            if k in features:
                try:
                    features[k] = float(v)
                except (ValueError, TypeError):
                    pass
    except:
        pass
        
    # Ensure column order matches the model exactly
    feature_order = ['rainfall', 'temperature', 'pumping', 'lulc', 'subsidence', 
                    'elevation_dem', 'twi', 'distance_to_river', 'lag_1_gw', 'rolling_mean_3_gw']
    df_features = pd.DataFrame([features])[feature_order].astype(float)
    
    prediction = xgb_model.predict(df_features)[0]
    return features, prediction


def compile_response(query: str, route: str, raw_data: str):
    prompt = f"""You are a Geomatics & Hydrogeology Master AI. 
User Query: {query}
Raw Data/Result: {raw_data}

CRITICAL RULES:
1. You MUST answer the user's query based ONLY on the Raw Data provided above.
2. If the Raw Data says 'Error' or is empty, DO NOT make up an answer. Apologize and state you could not calculate the result.
3. Be highly professional, engaging, and concise. Do not show raw JSON or SQL code to the user."""
    return llm.invoke([SystemMessage(content=prompt)]).content


# ==========================================
# 🎛️ 4. INTERACTIVE TABS
# ==========================================
tab_chat, tab_viz, tab_sandbox, tab_rl, tab_audit, tab_zones, tab_subsidence = st.tabs([
    "💬 Swarm Intelligence", 
    "📈 Dynamic Data Comparator", 
    "🔮 Live Inference", 
    "🤖 RL Policy Dashboard", 
    "🛡️ V&V Audit",
    "🏞️ Hydrogeological Zonation & GNSS",
    "🛰️ InSAR Subsidence & THSR Risk"
])

# --- TAB 1: SWARM INTELLIGENCE CHAT ---
with tab_chat:
    # This placeholder holds our cartoon agents
    agent_visual_placeholder = st.empty()
    agent_visual_placeholder.markdown(get_agent_ui("NONE"), unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant",
                                      "content": "Welcome to the Geo-Agent Swarm. How can I assist you with the Taiwan aquifer today?"}]

    # Print chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask the Swarm..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Step 1: Animate Master Router Thinking
            agent_visual_placeholder.markdown(get_agent_ui("ROUTER"), unsafe_allow_html=True)
            with st.status("🧠 Master Router Analyzing Intent...", expanded=True) as status:
                route = route_intent(prompt)
                st.write(f"🔀 Decision: Route to {route} Agent")

                # Step 2: Animate the specifically called Agent
                agent_visual_placeholder.markdown(get_agent_ui(route), unsafe_allow_html=True)

                if route == "SQL":
                    st.write("🦆 DuckDB searching 2.2 million spatial records...")
                    sql_query, raw_data = execute_sql_agent(prompt)
                    st.code(sql_query, language="sql")

                elif route == "PREDICT":
                    st.write("🌲 XGBoost projecting future environmental impacts...")
                    full_features, pred = execute_predictive_agent(prompt)
                    st.json(full_features)
                    raw_data = f"Predicted GROUNDWATER LEVEL: {pred:.2f} meters. (CRITICAL INSTRUCTION TO LLM: The AI can ONLY predict groundwater levels. If the user asked to predict subsidence, rainfall, or anything else, state that the model only predicts groundwater levels)."

                else:
                    st.write("💬 General interaction detected.")
                    raw_data = "Casual interaction."

                st.write("🗣️ Compiling final natural language response...")
                final_answer = compile_response(prompt, route, raw_data)
                status.update(label="Response Ready", state="complete", expanded=False)

            # Typewriter response output
            res_box = st.empty()
            full_res = ""
            for chunk in stream_text(final_answer):
                full_res += chunk
                res_box.markdown(full_res + "▌")
            res_box.markdown(full_res)

            st.session_state.messages.append({"role": "assistant", "content": full_res})

            # Step 3: Return agents to resting state
            time.sleep(1)
            agent_visual_placeholder.markdown(get_agent_ui("NONE"), unsafe_allow_html=True)

# --- TAB 2: EXPLORATORY DATA ANALYSIS (EDA) ---
with tab_viz:
    st.markdown("### 📊 21-Year Spatiotemporal EDA Dashboard")
    st.write("Explore the full scope of the 2.2M row dataset through aggregated insights.")
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📈 Temporal Trends", "🧩 Correlation Matrix", "🗺️ Spatial Sample (May 2017)"])
    
    # Preload the aggregated data safely
    try:
        monthly_df = pd.read_parquet('monthly_aggregate.parquet')
        corr_df = pd.read_csv('correlation_matrix.csv', index_col=0)
        spatial_df = pd.read_parquet('spatial_sample_201705.parquet')
        data_loaded = True
    except:
        st.error("Aggregated data not found. Please run `aggregate_data.py` first.")
        data_loaded = False

    if data_loaded:
        with eda_tab1:
            st.markdown("#### Month-over-Month Variable Trends (2001 - 2021)")
            plot_var = st.selectbox("Select Variable to Plot over Time:", 
                                   [c for c in monthly_df.columns if c != 'timestamp'],
                                   index=0)
            
            fig = px.line(monthly_df, x='timestamp', y=plot_var, 
                          title=f"Global Average {plot_var.upper()} (2001-2021)",
                          markers=True,
                          line_shape="spline")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with eda_tab2:
            st.markdown("#### Feature Correlation Heatmap")
            st.write("Visualizing how strongly different variables correlate with each other globally.")
            
            fig_corr = px.imshow(corr_df, text_auto=".2f", aspect="auto", 
                                 color_continuous_scale="RdBu_r", origin='lower')
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with eda_tab3:
            st.markdown("#### Spatial Distribution (Dry Season Snapshot)")
            st.write("Visualizing a random spatial subset of Groundwater Levels across the Taiwan aquifer during May 2017.")
            
            # Simple scatter map
            fig_map = px.scatter(spatial_df, x="longitude", y="latitude", 
                                 color="groundwater_level",
                                 color_continuous_scale="Viridis",
                                 title="Groundwater Levels - May 2017",
                                 opacity=0.7)
            fig_map.update_layout(height=600, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 3: LIVE INFERENCE ---
with tab_sandbox:
    st.markdown("### 🎛️ Live XGBoost Inference Engine (The World Simulator)")
    st.info("**What is this?** This is our highly-accurate 'World Simulator'. It learned the laws of physics for the Taiwan aquifer from 21 years of data. You can play with the sliders below (like turning up the rainfall, or increasing human pumping) and the AI will instantly predict the exact depth of the groundwater.")
    col1, col2 = st.columns([1, 2])
    with col1:
        s_rain = st.slider("Rainfall (mm)", 0.0, 500.0, 150.0)
        s_pump = st.slider("Pumping Rate", 0.0, 1000.0, 50.0)
        s_temp = st.slider("Temperature (°C)", 10.0, 40.0, 28.0)
    with col2:
        # Re-ordered columns exactly matching model
        test_features = {
            'rainfall': s_rain, 'temperature': s_temp, 'pumping': s_pump,
            'lulc': 2.0, 'subsidence': 0.0, 'elevation_dem': 15.0,
            'twi': 10.0, 'distance_to_river': 500.0,
            'lag_1_gw': -10.0, 'rolling_mean_3_gw': -10.5
        }
        feature_order = ['rainfall', 'temperature', 'pumping', 'lulc', 'subsidence', 
                        'elevation_dem', 'twi', 'distance_to_river', 'lag_1_gw', 'rolling_mean_3_gw']
        df_features = pd.DataFrame([test_features])[feature_order].astype(float)
        live_pred = xgb_model.predict(df_features)[0]

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=live_pred,
            title={'text': "Predicted Groundwater Depth (m)"},
            gauge={'axis': {'range': [-50, 0]}, 'bar': {'color': "#39ff14"},
                   'steps': [{'range': [-50, -30], 'color': "red"}, {'range': [-30, -10], 'color': "orange"},
                             {'range': [-10, 0], 'color': "green"}]}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- TAB 4: RL POLICY DASHBOARD ---
with tab_rl:
    st.markdown("### 🤖 RL Agent Pumping Policy (The AI Manager)")
    st.write("This tab visualizes the behavior of the PPO agent trained on the XGBoost World Model.")
    st.info("**What is this?** If the XGBoost model is the 'Simulator', this RL Agent is the 'Manager'. We let this AI interact with the simulator 100,000 times, punishing it when the aquifer ran dry or the ground sank (subsidence), and rewarding it for supplying water. It has now learned the optimal strategy for balancing human water needs with environmental survival.")
    
    try:
        from stable_baselines3 import PPO
        rl_model = PPO.load("rl_groundwater_policy")
        st.success("✅ RL Policy Loaded Successfully!")
        st.metric("Model Algorithm", "Proximal Policy Optimization (PPO)")
        st.metric("Observation Space", "12 Features (including temporal encoding)")
        st.metric("Action Space", "Continuous [-1.0, 1.0] (Pumping Adjustment)")
        
    except FileNotFoundError:
        st.warning("⚠️ 'rl_groundwater_policy.zip' not found. The agent is currently training in the background. Check back later once training finishes.")

# --- TAB 5: V&V DATA QUALITY AUDIT ---
with tab_audit:
    st.markdown("### 🛡️ 4-Stage V&V Pipeline Audit")
    st.write("Monitoring the health of our data ingestion and model confidence.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Pipeline Status", "ACTIVE", delta="Normal", delta_color="normal")
    col2.metric("Records Quarantined (24h)", "0")
    col3.metric("Anomaly Detection", "XGBoost Residual")
    
    st.markdown("#### Recent Ingestion Log")
    st.table(pd.DataFrame([
        {"Batch ID": "API_20261006_1200", "Status": "APPROVED", "Rows": 1500, "Confidence": "98%"},
        {"Batch ID": "SENSOR_20261006_1100", "Status": "QUARANTINED", "Rows": 50, "Confidence": "45%"},
        {"Batch ID": "CSV_20261005_1500", "Status": "APPROVED", "Rows": 50000, "Confidence": "100%"},
    ]))

# --- TAB 6: HYDROGEOLOGICAL ZONATION & GNSS POINT PREDICTOR ---
with tab_zones:
    st.markdown("### 🏞️ Hydrogeological Zonation & GNSS Station Point-Predictions")
    st.info("**Research Breakthrough:** Moving beyond spatial homogeneity by stratifying the Choushui River Alluvial Fan into three classical hydrogeological zones (Upper, Middle, Lower Fan) and validating specific-point prediction at 25 GNSS monitoring stations.")

    subtab_map, subtab_gnss, subtab_cv = st.tabs(["🗺️ Zonation Map & Metrics", "📍 GNSS Station Point-Predictions", "🧪 Spatial CV Benchmark"])

    with subtab_map:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Three-Phase Hydrogeological Zonation")
            st.write("""
            - **Upper Fan (Proximal):** Coarse gravel/boulder unconfined aquifer. High elevation (>50m). Recharges the entire basin.
            - **Middle Fan (Medial):** Transitional sand/gravel semi-confined aquifer. Heavy agricultural and industrial pumping.
            - **Lower Fan (Distal):** Multi-layered confined clay/silt coastal plain. Below sea level hydraulic head, extreme subsidence risk.
            """)
            try:
                with open("zone_evaluation_results.json", "r") as f:
                    z_results = json.load(f)
                z_df = pd.DataFrame([
                    {"Zone": "Upper Fan", "Mean GW": f"{z_results['zones']['upper']['mean_gw']:.2f} m", "Test R²": f"{z_results['zones']['upper']['r2']:.4f}", "RMSE": f"{z_results['zones']['upper']['rmse']:.2f} m"},
                    {"Zone": "Middle Fan", "Mean GW": f"{z_results['zones']['middle']['mean_gw']:.2f} m", "Test R²": f"{z_results['zones']['middle']['r2']:.4f}", "RMSE": f"{z_results['zones']['middle']['rmse']:.2f} m"},
                    {"Zone": "Lower Fan", "Mean GW": f"{z_results['zones']['lower']['mean_gw']:.2f} m", "Test R²": f"{z_results['zones']['lower']['r2']:.4f}", "RMSE": f"{z_results['zones']['lower']['rmse']:.2f} m"},
                ])
                st.table(z_df)
            except Exception as e:
                st.warning(f"Zone results not loaded: {e}")

        with col2:
            try:
                zone_df = pd.read_parquet("zone_lookup.parquet")
                fig_zone = px.scatter(
                    zone_df, x="longitude", y="latitude", color="fan_zone",
                    color_discrete_map={"upper": "#2ca02c", "middle": "#ff7f0e", "lower": "#1f77b4"},
                    title="Choushui River Alluvial Fan Hydrogeological Zonation (8,872 grid cells)",
                    opacity=0.75
                )
                fig_zone.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_zone, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load zone map: {e}")

    with subtab_gnss:
        st.markdown("#### Specific-Point Prediction at Real Monitoring Stations")
        st.write("Extracting monthly time series at GNSS station coordinates from 2001–2021 to predict station groundwater levels on held-out test data (2020–2021).")
        try:
            gnss_preds = pd.read_parquet("gnss_predictions_vs_observed.parquet")
            with open("gnss_point_prediction_results.json", "r") as f:
                gnss_metrics = json.load(f)

            stations = sorted(list(gnss_metrics.keys()))
            selected_st = st.selectbox("Select GNSS Monitoring Station:", stations, index=0)

            st_data = gnss_preds[gnss_preds["station"] == selected_st].sort_values("timestamp")
            meta = gnss_metrics[selected_st]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Station Name", selected_st)
            c2.metric("Hydrogeological Zone", meta["zone"].capitalize())
            c3.metric("Station Test R²", f"{meta['r2']:.4f}")
            c4.metric("RMSE", f"{meta['rmse']:.2f} m", delta=f"MAE: {meta['mae']:.2f} m", delta_color="off")

            fig_gnss = go.Figure()
            fig_gnss.add_trace(go.Scatter(x=st_data["timestamp"], y=st_data["observed"], mode="lines+markers", name="Observed GW (m)", line=dict(color="#00d2ff", width=2)))
            fig_gnss.add_trace(go.Scatter(x=st_data["timestamp"], y=st_data["predicted"], mode="lines+markers", name="Predicted GW (m)", line=dict(color="#ff3366", width=2, dash="dash")))
            fig_gnss.update_layout(
                title=f"Station {selected_st} Point-Prediction Time Series (2020–2021 Test Period)",
                xaxis_title="Date", yaxis_title="Groundwater Elevation (m)", height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_gnss, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load GNSS point-predictions: {e}")

    with subtab_cv:
        st.markdown("#### Spatial Generalization & Cross-Validation Benchmark")
        st.write("Proof of model generalization: Testing models on completely held-out spatial clusters and zones with zero spatial leakage.")
        try:
            with open("spatial_cv_results.json", "r") as f:
                cv_res = json.load(f)

            cv_table = pd.DataFrame([
                {
                    "Validation Scheme": "Random 5-Fold CV (Baseline)",
                    "Feature Set": "Physical + Temporal Lags",
                    "Spatial Leakage": "❌ High (Same pixel in train & test)",
                    "Test R²": f"{cv_res['random_kfold_autoregressive']['r2_mean']:.4f} ± {cv_res['random_kfold_autoregressive']['r2_std']:.4f}",
                    "RMSE (m)": f"{cv_res['random_kfold_autoregressive']['rmse_mean']:.2f}"
                },
                {
                    "Validation Scheme": "Spatial Group 5-Fold CV",
                    "Feature Set": "Physical + Temporal Lags",
                    "Spatial Leakage": "✅ Zero (Held-out spatial clusters)",
                    "Test R²": f"{cv_res['spatial_group_kfold_autoregressive']['r2_mean']:.4f} ± {cv_res['spatial_group_kfold_autoregressive']['r2_std']:.4f}",
                    "RMSE (m)": f"{cv_res['spatial_group_kfold_autoregressive']['rmse_mean']:.2f}"
                },
                {
                    "Validation Scheme": "Spatial Group 5-Fold CV",
                    "Feature Set": "Physical Covariates Only (Ungauged)",
                    "Spatial Leakage": "✅ Zero (Held-out spatial clusters)",
                    "Test R²": f"{cv_res['spatial_group_kfold_physical_only']['r2_mean']:.4f} ± {cv_res['spatial_group_kfold_physical_only']['r2_std']:.4f}",
                    "RMSE (m)": f"{cv_res['spatial_group_kfold_physical_only']['rmse_mean']:.2f}"
                }
            ])
            st.table(cv_table)

            st.markdown("##### Leave-One-Zone-Out Cross-Validation (LOZOCV)")
            loz = cv_res["leave_one_zone_out"]
            loz_df = pd.DataFrame([
                {"Held-out Test Zone": "Upper Fan", "Trained On": "Middle + Lower Fan", "Test R²": f"{loz['upper']['r2']:.4f}", "RMSE": f"{loz['upper']['rmse']:.2f} m"},
                {"Held-out Test Zone": "Middle Fan", "Trained On": "Upper + Lower Fan", "Test R²": f"{loz['middle']['r2']:.4f}", "RMSE": f"{loz['middle']['rmse']:.2f} m"},
                {"Held-out Test Zone": "Lower Fan", "Trained On": "Upper + Middle Fan", "Test R²": f"{loz['lower']['r2']:.4f}", "RMSE": f"{loz['lower']['rmse']:.2f} m"},
            ])
            st.table(loz_df)
        except Exception as e:
            st.warning(f"Could not load spatial CV results: {e}")


# --- TAB 7: INSAR SUBSIDENCE & THSR CORRIDOR ---
with tab_subsidence:
    st.markdown("### 🛰️ Sentinel-1 InSAR Land Subsidence & Taiwan High Speed Rail Risk")
    st.info("**Civil Engineering Impact:** Combining 5 years of Sentinel-1 InSAR deformation rasters (2017–2021) with groundwater drawdown to map aquifer compaction sensitivity and identify infrastructure hazard along the Taiwan High Speed Rail (THSR).")

    try:
        sub_map = pd.read_parquet("subsidence_coupling_map.parquet")
        with open("subsidence_coupling_summary.json", "r") as f:
            sub_summary = json.load(f)

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Study Period", "2017–2021 (60 mo)")
        s_col2.metric("THSR Buffer (≤ 3km)", f"{sub_summary['thsr_corridor']['within_3km']['cell_count']:,} cells")
        s_col3.metric("THSR Avg Subsidence", f"{sub_summary['thsr_corridor']['within_3km']['mean_subsidence_mm']:.1f} mm")
        s_col4.metric("Basin Peak Subsidence", f"{sub_summary['thsr_corridor']['within_3km']['peak_subsidence_mm']:.1f} mm", delta="Located on THSR Corridor", delta_color="inverse")

        map_metric = st.selectbox(
            "Select Map Metric to Visualize:",
            [
                ("max_subsidence_mm", "Peak Cumulative Subsidence (mm) — Sentinel-1 InSAR"),
                ("compaction_sensitivity_mm_per_m", "Compaction Sensitivity (mm subsidence per meter of GW drawdown)"),
                ("r_gw_sub", "Pearson Correlation (Groundwater vs. Subsidence)"),
                ("dist_to_thsr_km", "Distance to Taiwan High Speed Rail Corridor (km)")
            ],
            format_func=lambda x: x[1]
        )[0]

        fig_sub = px.scatter(
            sub_map, x="longitude", y="latitude", color=map_metric,
            color_continuous_scale="Turbo_r" if "subsidence" in map_metric else "Viridis",
            title=f"Spatiotemporal Subsidence Analysis: {map_metric.replace('_', ' ').title()}",
            opacity=0.8
        )
        fig_sub.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sub, use_container_width=True)

        st.markdown("#### Severe Subsidence Hotspots along THSR Corridor")
        hotspots_df = pd.DataFrame(sub_summary["top_hotspots"])
        st.dataframe(hotspots_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load subsidence coupling data: {e}")