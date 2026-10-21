import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

console.print("[bold cyan]=== 🧠 PHASE 5: XGBOOST ML PREDICTOR (PHYSICAL WORLD MODEL) ===[/]")
console.print("[yellow]1. Loading fused master dataset from V&V Vault...[/]")
df = pd.read_parquet("./data/validated")

console.print("\n[yellow]--- 🩹 Auto-Fixing Merge Artifacts & Engineering Temporal Features ---[/]")
# Safely rename _x or _y columns to their base name, and drop duplicates
for col in list(df.columns):
    if col.endswith('_x') or col.endswith('_y'):
        base_name = col[:-2]
        if base_name not in df.columns:
            df = df.rename(columns={col: base_name})
        else:
            df = df.drop(columns=[col])

# Sort chronologically to ensure time-series integrity
df = df.sort_values(by=['longitude', 'latitude', 'timestamp'])

# FEATURE ENGINEERING: Add temporal lags instead of relying on spatial coordinates
console.print("Adding temporal lag features to remove spatial bias...")
df['lag_1_gw'] = df.groupby(['longitude', 'latitude'])['groundwater_level'].shift(1)
df['rolling_mean_3_gw'] = df.groupby(['longitude', 'latitude'])['groundwater_level'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Fill NaNs created by lagging
df['lag_1_gw'] = df['lag_1_gw'].bfill()
df['rolling_mean_3_gw'] = df['rolling_mean_3_gw'].bfill()

# Define our Features (what the AI learns from) and Target (what the AI predicts)
# CRITICAL: We removed 'longitude' and 'latitude' to prevent spatial overfitting!
features = [
    'rainfall', 'temperature', 'pumping', 'lulc', 'subsidence',
    'elevation_dem', 'twi', 'distance_to_river',
    'lag_1_gw', 'rolling_mean_3_gw'
]
target = 'groundwater_level'

# Just in case any column is completely empty, we fill it with 0
df[features] = df[features].fillna(0)

from sklearn.model_selection import train_test_split

console.print("\n[yellow]--- ✂️ Splitting Data (80/20 Stratified by Year) ---[/]")
df['year'] = df['timestamp'].dt.year

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['year']
)

console.print(f"-> Training on [bold green]{len(X_train):,}[/] historical records...")
console.print(f"-> Testing on [bold green]{len(X_test):,}[/] future records...")

console.print("\n[bold yellow]--- 🏋️ Training XGBoost World Model ---[/]")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    tree_method='hist',
    random_state=42
)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
) as progress:
    task = progress.add_task("[cyan]Fitting XGBoost Regressor...", total=100)
    # XGBoost fit doesn't have an easy native callback for rich progress out-of-the-box for small runs, 
    # so we just show the spinner. For larger runs we could use custom callbacks.
    model.fit(X_train, y_train)
    progress.update(task, advance=100)

console.print("\n[bold yellow]--- 📊 Evaluating Model Accuracy ---[/]")
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

metrics_table = Table(title="Model Performance")
metrics_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
metrics_table.add_column("Value", style="magenta")
metrics_table.add_row("R-squared (R2)", f"{r2:.4f}")
metrics_table.add_row("RMSE", f"{rmse:.2f} meters")
metrics_table.add_row("MAE", f"{mae:.2f} meters")
console.print(metrics_table)

# Log feature importances
console.print("\n[bold yellow]--- 🌟 Feature Importances ---[/]")
importances = model.feature_importances_
imp_table = Table(title="Feature Importance Ranking")
imp_table.add_column("Feature", style="cyan")
imp_table.add_column("Importance", style="green")

# Sort features by importance
sorted_idx = np.argsort(importances)[::-1]
for idx in sorted_idx:
    imp_table.add_row(features[idx], f"{importances[idx]:.4f}")
console.print(imp_table)

# Save the trained model brain so the Swarm Agent and RL Environment can use it
model_filename = "xgboost_groundwater_model.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

console.print(f"\n[bold green]✅ World Model trained and saved to disk as '{model_filename}'![/]")