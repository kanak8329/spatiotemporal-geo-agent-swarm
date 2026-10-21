"""
Zone Classifier for Choushui River Alluvial Fan
================================================
Splits the 8,872 raster grid points into three hydrogeological zones:
  - Upper Fan (Proximal):  Inside Proximal fan.shp OR DEM > 50m
  - Middle Fan (Transition): Outside proximal, DEM 15–50m
  - Lower Fan (Distal/Coastal): DEM ≤ 15m (highest subsidence risk)

Then re-evaluates the existing XGBoost model per zone to show
whether performance varies across the fan.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rich.console import Console
from rich.table import Table

console = Console()

# =============================================
# 1. BUILD ZONE LOOKUP FROM DEM + PROXIMAL FAN
# =============================================
def build_zone_lookup():
    """Classify every unique (lon, lat) in the dataset into a fan zone."""
    console.print("[bold cyan]═══ ZONE CLASSIFIER: Choushui Alluvial Fan ═══[/]")

    # Load validated dataset to get unique coordinates
    console.print("[yellow]Loading validated dataset...[/]")
    df = pd.read_parquet("./data/validated")
    unique_coords = df[['longitude', 'latitude']].drop_duplicates().reset_index(drop=True)
    console.print(f"  → {len(unique_coords):,} unique grid points found")

    # Sample DEM at each coordinate
    console.print("[yellow]Sampling DEM elevation at each grid point...[/]")
    with rasterio.open("data/0_dropzone/DEM/DEM.tif") as src:
        dem_nodata = src.nodata
        coord_list = list(zip(unique_coords['longitude'], unique_coords['latitude']))
        sampled = [val[0] for val in src.sample(coord_list)]
        unique_coords['elevation'] = sampled
        # Replace nodata with NaN
        if dem_nodata is not None:
            unique_coords.loc[unique_coords['elevation'] == dem_nodata, 'elevation'] = np.nan

    console.print(f"  → DEM range: {unique_coords['elevation'].min():.1f}m to {unique_coords['elevation'].max():.1f}m")

    # Load Proximal Fan shapefile
    console.print("[yellow]Loading Proximal Fan boundary shapefile...[/]")
    proximal_gdf = gpd.read_file("data/0_dropzone/Choushui Shapefile/Proximal fan.shp")
    # Ensure CRS matches (convert to WGS84 if needed)
    if proximal_gdf.crs and proximal_gdf.crs.to_epsg() != 4326:
        proximal_gdf = proximal_gdf.to_crs(epsg=4326)
    proximal_union = proximal_gdf.geometry.unary_union
    console.print(f"  → Proximal fan polygon loaded ({len(proximal_gdf)} features)")

    # Classify each point
    console.print("[yellow]Classifying grid points into fan zones...[/]")
    zones = []
    for _, row in unique_coords.iterrows():
        pt = Point(row['longitude'], row['latitude'])
        elev = row['elevation']

        if proximal_union.contains(pt) or (not np.isnan(elev) and elev > 50):
            zones.append("upper")
        elif not np.isnan(elev) and elev > 15:
            zones.append("middle")
        else:
            zones.append("lower")

    unique_coords['fan_zone'] = zones

    # Summary
    zone_counts = unique_coords['fan_zone'].value_counts()
    console.print("\n[bold green]Zone Classification Results:[/]")
    for zone in ['upper', 'middle', 'lower']:
        count = zone_counts.get(zone, 0)
        pct = count / len(unique_coords) * 100
        console.print(f"  {zone.capitalize():>8} Fan: {count:>5} grid points ({pct:.1f}%)")

    # Save lookup
    lookup_path = "zone_lookup.parquet"
    unique_coords.to_parquet(lookup_path, index=False)
    console.print(f"\n[green]✅ Zone lookup saved to '{lookup_path}'[/]")

    return unique_coords


# =============================================
# 2. EVALUATE MODEL PER ZONE
# =============================================
def evaluate_per_zone(zone_lookup):
    """Load the trained XGBoost model and evaluate separately per zone."""
    console.print("\n[bold cyan]═══ ZONE-STRATIFIED MODEL EVALUATION ═══[/]")

    # Load data and model
    df = pd.read_parquet("./data/validated")

    # Add temporal lag features (same as ml_predictor.py)
    df = df.sort_values(by=['longitude', 'latitude', 'timestamp'])
    df['lag_1_gw'] = df.groupby(['longitude', 'latitude'])['groundwater_level'].shift(1)
    df['rolling_mean_3_gw'] = df.groupby(['longitude', 'latitude'])['groundwater_level'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['lag_1_gw'] = df['lag_1_gw'].bfill()
    df['rolling_mean_3_gw'] = df['rolling_mean_3_gw'].bfill()

    features = [
        'rainfall', 'temperature', 'pumping', 'lulc', 'subsidence',
        'elevation_dem', 'twi', 'distance_to_river',
        'lag_1_gw', 'rolling_mean_3_gw'
    ]
    target = 'groundwater_level'
    df[features] = df[features].fillna(0)

    # Merge zone labels
    df = df.merge(
        zone_lookup[['longitude', 'latitude', 'fan_zone', 'elevation']],
        on=['longitude', 'latitude'],
        how='left'
    )
    df['fan_zone'] = df['fan_zone'].fillna('unknown')

    # Load model
    with open("xgboost_groundwater_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Train/test split (same as ml_predictor.py)
    from sklearn.model_selection import train_test_split
    df['year'] = df['timestamp'].dt.year
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['year']
    )

    # Get zone labels for test set
    test_zones = df.loc[X_test.index, 'fan_zone']

    # Predict
    predictions = model.predict(X_test)

    # Global metrics
    global_r2 = r2_score(y_test, predictions)
    global_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    global_mae = mean_absolute_error(y_test, predictions)

    # Per-zone metrics
    results_table = Table(title="Zone-Stratified Model Performance")
    results_table.add_column("Zone", style="cyan", justify="center")
    results_table.add_column("Test Points", style="white", justify="right")
    results_table.add_column("R²", style="green", justify="right")
    results_table.add_column("RMSE (m)", style="yellow", justify="right")
    results_table.add_column("MAE (m)", style="yellow", justify="right")
    results_table.add_column("Mean GW (m)", style="magenta", justify="right")

    zone_results = {}
    for zone in ['upper', 'middle', 'lower']:
        mask = test_zones == zone
        if mask.sum() == 0:
            continue
        z_y = y_test[mask]
        z_pred = predictions[mask]
        z_r2 = r2_score(z_y, z_pred)
        z_rmse = np.sqrt(mean_squared_error(z_y, z_pred))
        z_mae = mean_absolute_error(z_y, z_pred)
        z_mean_gw = z_y.mean()

        results_table.add_row(
            f"{zone.capitalize()} Fan",
            f"{mask.sum():,}",
            f"{z_r2:.4f}",
            f"{z_rmse:.2f}",
            f"{z_mae:.2f}",
            f"{z_mean_gw:.2f}"
        )
        zone_results[zone] = {
            "r2": round(z_r2, 4), "rmse": round(z_rmse, 2),
            "mae": round(z_mae, 2), "mean_gw": round(z_mean_gw, 2),
            "test_points": int(mask.sum())
        }

    results_table.add_row(
        "[bold]Global[/]", f"{len(y_test):,}",
        f"[bold]{global_r2:.4f}[/]", f"[bold]{global_rmse:.2f}[/]",
        f"[bold]{global_mae:.2f}[/]", f"{y_test.mean():.2f}"
    )

    console.print(results_table)

    # Zone statistics from the full dataset
    console.print("\n[bold cyan]═══ ZONE HYDROGEOLOGICAL SUMMARY ═══[/]")
    stats_table = Table(title="Zone Physical Characteristics (Full Dataset)")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Upper Fan", style="green", justify="right")
    stats_table.add_column("Middle Fan", style="yellow", justify="right")
    stats_table.add_column("Lower Fan", style="red", justify="right")

    for metric, col in [
        ("Mean GW Level (m)", "groundwater_level"),
        ("Mean Elevation (m)", "elevation"),
        ("Mean Rainfall (mm)", "rainfall"),
        ("Mean Pumping", "pumping"),
        ("Mean Subsidence (mm)", "subsidence"),
    ]:
        vals = []
        for zone in ['upper', 'middle', 'lower']:
            mask = df['fan_zone'] == zone
            if mask.sum() > 0 and col in df.columns:
                vals.append(f"{df.loc[mask, col].mean():.2f}")
            else:
                vals.append("N/A")
        stats_table.add_row(metric, *vals)

    # GW trend (simple: last year mean minus first year mean)
    trend_vals = []
    for zone in ['upper', 'middle', 'lower']:
        zmask = df['fan_zone'] == zone
        if zmask.sum() > 0:
            zone_df = df[zmask]
            first_year = zone_df[zone_df['year'] == zone_df['year'].min()]['groundwater_level'].mean()
            last_year = zone_df[zone_df['year'] == zone_df['year'].max()]['groundwater_level'].mean()
            n_years = zone_df['year'].max() - zone_df['year'].min()
            if n_years > 0:
                trend = (last_year - first_year) / n_years
                trend_vals.append(f"{trend:+.3f}")
            else:
                trend_vals.append("N/A")
        else:
            trend_vals.append("N/A")
    stats_table.add_row("GW Trend (m/yr)", *trend_vals)

    console.print(stats_table)

    # Save results (convert numpy types to native Python for JSON)
    output = {
        "global": {"r2": float(round(global_r2, 4)), "rmse": float(round(global_rmse, 2)), "mae": float(round(global_mae, 2))},
        "zones": zone_results
    }
    with open("zone_evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    console.print("\n[green]✅ Results saved to 'zone_evaluation_results.json'[/]")

    return output


# =============================================
# 3. MAIN
# =============================================
if __name__ == "__main__":
    zone_lookup = build_zone_lookup()
    results = evaluate_per_zone(zone_lookup)

    console.print("\n[bold green]═══ ZONE CLASSIFICATION COMPLETE ═══[/]")
    console.print("Next steps:")
    console.print("  1. Review zone_evaluation_results.json for per-zone model metrics")
    console.print("  2. Run gnss_point_predictor.py for station-level predictions")
    console.print("  3. Use zone_lookup.parquet to add zone labels to any analysis")
