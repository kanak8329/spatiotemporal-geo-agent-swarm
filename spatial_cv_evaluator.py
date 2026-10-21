"""
Advance 4: Spatial Cross-Validation & Spatial Generalization Evaluator
=====================================================================
Performs rigorous spatial cross-validation to test whether the ML model
truly learns physical hydrogeological relationships or merely memorizes
pixel locations through spatial autocorrelation and temporal autoregression.

Experiments:
1. Standard Random 5-Fold CV (Baseline with spatial leakage)
2. Spatial Group 5-Fold CV (K-Means spatial blocks: ZERO spatial leakage)
   - Evaluated for Exogenous Physical Model (Ungauged basin setting)
   - Evaluated for Autoregressive Model
3. Leave-One-Zone-Out Cross-Validation (LOZOCV) across Upper, Middle, Lower fan
"""

import json
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold
import xgboost as xgb
from rich.console import Console
from rich.table import Table

console = Console()


def run_spatial_cv():
    console.print("=" * 70, style="bold cyan")
    console.print("🧪 ADVANCE 4: SPATIAL CROSS-VALIDATION & GENERALIZATION BENCHMARK", style="bold cyan")
    console.print("Study Area: Choushui River Alluvial Fan (2001–2021)", style="bold cyan")
    console.print("=" * 70, style="bold cyan")

    # 1. Load Data
    console.print("\n[bold yellow][Step 1/5] Loading and Preparing Dataset...[/]")
    t0 = time.time()
    df = pd.read_parquet("data/validated")
    zone_lookup = pd.read_parquet("zone_lookup.parquet")
    df = df.merge(zone_lookup[["longitude", "latitude", "fan_zone"]], on=["longitude", "latitude"], how="left")
    df["fan_zone"] = df["fan_zone"].fillna("middle")
    console.print(f" Loaded {len(df):,} records in {time.time() - t0:.2f}s.")

    # 2. Build Features
    console.print("\n[bold yellow][Step 2/5] Engineering Physical and Autoregressive Features...[/]")
    df = df.sort_values(by=["longitude", "latitude", "timestamp"])
    df["lag_1_gw"] = df.groupby(["longitude", "latitude"])["groundwater_level"].shift(1).bfill()
    df["rolling_mean_3_gw"] = df.groupby(["longitude", "latitude"])["groundwater_level"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).bfill()

    physical_features = [
        "rainfall", "temperature", "pumping", "lulc", "subsidence",
        "elevation_dem", "twi", "distance_to_river"
    ]
    autoreg_features = physical_features + ["lag_1_gw", "rolling_mean_3_gw"]
    target = "groundwater_level"

    df[autoreg_features] = df[autoreg_features].fillna(0)

    # 3. Create Spatial Blocks (K-Means on coordinates to form spatial clusters)
    console.print("\n[bold yellow][Step 3/5] Partitioning Fan into 5 Spatial Clusters (Zero Spatial Leakage)...[/]")
    coords = df[["longitude", "latitude"]].drop_duplicates().reset_index(drop=True)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    coords["spatial_fold"] = kmeans.fit_predict(coords[["longitude", "latitude"]])
    df = df.merge(coords, on=["longitude", "latitude"], how="left")

    console.print(" Spatial cluster sizes (grid points per cluster):")
    for fold, count in coords["spatial_fold"].value_counts().sort_index().items():
        console.print(f"   Cluster {fold}: {count:,} locations ({count / len(coords) * 100:.1f}%)")

    # Subsample temporally by taking every 2nd month for rapid, reliable CV (126 time steps x 8,872 cells = 1.1M rows)
    console.print("\n Using representative temporal stride (1.1M samples) for high-precision cross-validation...")
    cv_df = df.iloc[::2].copy().reset_index(drop=True)

    # XGBoost fast training configuration
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42
    }

    # 4. Run Cross-Validation Experiments
    console.print("\n[bold yellow][Step 4/5] Running Cross-Validation Experiments...[/]")

    results = {}

    # Experiment 1: Standard Random 5-Fold (Baseline with spatial leakage)
    console.print("\n▶ Running Experiment 1: Standard Random 5-Fold CV (Spatial Leakage Present)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rand_r2_list, rand_rmse_list, rand_mae_list = [], [], []

    X_auto = cv_df[autoreg_features].values
    y_true = cv_df[target].values

    for train_idx, test_idx in kf.split(X_auto):
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_auto[train_idx], y_true[train_idx])
        preds = model.predict(X_auto[test_idx])
        rand_r2_list.append(r2_score(y_true[test_idx], preds))
        rand_rmse_list.append(np.sqrt(mean_squared_error(y_true[test_idx], preds)))
        rand_mae_list.append(mean_absolute_error(y_true[test_idx], preds))

    results["random_kfold_autoregressive"] = {
        "r2_mean": float(round(np.mean(rand_r2_list), 4)),
        "r2_std": float(round(np.std(rand_r2_list), 4)),
        "rmse_mean": float(round(np.mean(rand_rmse_list), 3)),
        "mae_mean": float(round(np.mean(rand_mae_list), 3))
    }
    console.print(f"  Random CV R²: {results['random_kfold_autoregressive']['r2_mean']:.4f} ± {results['random_kfold_autoregressive']['r2_std']:.4f}")

    # Experiment 2: Spatial Group 5-Fold (Autoregressive)
    console.print("\n▶ Running Experiment 2: Spatial Group 5-Fold CV (Autoregressive, Zero Spatial Leakage)...")
    spat_r2_auto, spat_rmse_auto, spat_mae_auto = [], [], []
    groups = cv_df["spatial_fold"].values

    for fold in range(5):
        test_mask = (groups == fold)
        train_mask = ~test_mask
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_auto[train_mask], y_true[train_mask])
        preds = model.predict(X_auto[test_mask])
        spat_r2_auto.append(r2_score(y_true[test_mask], preds))
        spat_rmse_auto.append(np.sqrt(mean_squared_error(y_true[test_mask], preds)))
        spat_mae_auto.append(mean_absolute_error(y_true[test_mask], preds))

    results["spatial_group_kfold_autoregressive"] = {
        "r2_mean": float(round(np.mean(spat_r2_auto), 4)),
        "r2_std": float(round(np.std(spat_r2_auto), 4)),
        "rmse_mean": float(round(np.mean(spat_rmse_auto), 3)),
        "mae_mean": float(round(np.mean(spat_mae_auto), 3)),
        "per_fold_r2": [float(round(v, 4)) for v in spat_r2_auto]
    }
    console.print(f"  Spatial CV (Autoregressive) R²: {results['spatial_group_kfold_autoregressive']['r2_mean']:.4f} ± {results['spatial_group_kfold_autoregressive']['r2_std']:.4f}")

    # Experiment 3: Spatial Group 5-Fold (Exogenous Physical Only - Ungauged Basin)
    console.print("\n▶ Running Experiment 3: Spatial Group 5-Fold CV (Physical Covariates Only, Ungauged Basin)...")
    X_phys = cv_df[physical_features].values
    spat_r2_phys, spat_rmse_phys, spat_mae_phys = [], [], []

    for fold in range(5):
        test_mask = (groups == fold)
        train_mask = ~test_mask
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_phys[train_mask], y_true[train_mask])
        preds = model.predict(X_phys[test_mask])
        spat_r2_phys.append(r2_score(y_true[test_mask], preds))
        spat_rmse_phys.append(np.sqrt(mean_squared_error(y_true[test_mask], preds)))
        spat_mae_phys.append(mean_absolute_error(y_true[test_mask], preds))

    results["spatial_group_kfold_physical_only"] = {
        "r2_mean": float(round(np.mean(spat_r2_phys), 4)),
        "r2_std": float(round(np.std(spat_r2_phys), 4)),
        "rmse_mean": float(round(np.mean(spat_rmse_phys), 3)),
        "mae_mean": float(round(np.mean(spat_mae_phys), 3)),
        "per_fold_r2": [float(round(v, 4)) for v in spat_r2_phys]
    }
    console.print(f"  Spatial CV (Physical Only) R²: {results['spatial_group_kfold_physical_only']['r2_mean']:.4f} ± {results['spatial_group_kfold_physical_only']['r2_std']:.4f}")

    # Experiment 4: Leave-One-Zone-Out Cross-Validation (LOZOCV)
    console.print("\n▶ Running Experiment 4: Leave-One-Zone-Out Cross-Validation (LOZOCV)...")
    lozocv_results = {}
    zones = ["upper", "middle", "lower"]

    for held_out in zones:
        test_mask = (cv_df["fan_zone"] == held_out).values
        train_mask = ~test_mask

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_auto[train_mask], y_true[train_mask])
        preds = model.predict(X_auto[test_mask])

        r2 = r2_score(y_true[test_mask], preds)
        rmse = np.sqrt(mean_squared_error(y_true[test_mask], preds))
        mae = mean_absolute_error(y_true[test_mask], preds)

        lozocv_results[held_out] = {
            "test_zone": held_out,
            "train_zones": [z for z in zones if z != held_out],
            "r2": float(round(r2, 4)),
            "rmse": float(round(rmse, 3)),
            "mae": float(round(mae, 3)),
            "test_points": int(test_mask.sum())
        }
        console.print(f"  Held-out {held_out.capitalize():<6} Fan: R² = {r2:.4f}, RMSE = {rmse:.2f} m, MAE = {mae:.2f} m")

    results["leave_one_zone_out"] = lozocv_results

    # 5. Output Results Table
    console.print("\n" + "=" * 70, style="bold cyan")
    console.print("📊 SUMMARY: SPATIAL CROSS-VALIDATION VS. RANDOM SPLIT", style="bold cyan")
    console.print("=" * 70, style="bold cyan")

    tbl = Table(title="Validation Scheme Comparison")
    tbl.add_column("Validation Scheme", style="cyan")
    tbl.add_column("Features Used", style="white")
    tbl.add_column("Spatial Leakage", style="magenta", justify="center")
    tbl.add_column("R² (Mean ± Std)", style="green", justify="right")
    tbl.add_column("RMSE (m)", style="yellow", justify="right")
    tbl.add_column("MAE (m)", style="yellow", justify="right")

    tbl.add_row(
        "Random 5-Fold CV",
        "Physical + Temporal Lags",
        "❌ High (Pixel overlap)",
        f"{results['random_kfold_autoregressive']['r2_mean']:.4f} ± {results['random_kfold_autoregressive']['r2_std']:.4f}",
        f"{results['random_kfold_autoregressive']['rmse_mean']:.2f}",
        f"{results['random_kfold_autoregressive']['mae_mean']:.2f}"
    )
    tbl.add_row(
        "Spatial Group 5-Fold",
        "Physical + Temporal Lags",
        "✅ None (Held-out blocks)",
        f"{results['spatial_group_kfold_autoregressive']['r2_mean']:.4f} ± {results['spatial_group_kfold_autoregressive']['r2_std']:.4f}",
        f"{results['spatial_group_kfold_autoregressive']['rmse_mean']:.2f}",
        f"{results['spatial_group_kfold_autoregressive']['mae_mean']:.2f}"
    )
    tbl.add_row(
        "Spatial Group 5-Fold",
        "Physical Only (Ungauged Basin)",
        "✅ None (Held-out blocks)",
        f"{results['spatial_group_kfold_physical_only']['r2_mean']:.4f} ± {results['spatial_group_kfold_physical_only']['r2_std']:.4f}",
        f"{results['spatial_group_kfold_physical_only']['rmse_mean']:.2f}",
        f"{results['spatial_group_kfold_physical_only']['mae_mean']:.2f}"
    )
    console.print(tbl)

    # Save to JSON
    with open("spatial_cv_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print("\n[green]✅ Spatial CV results saved to 'spatial_cv_results.json'[/]")


if __name__ == "__main__":
    run_spatial_cv()
