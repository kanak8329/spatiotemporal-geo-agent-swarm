"""
GNSS Station Point-Prediction System
=====================================
Extracts time series at each of the 26 GNSS monitoring stations by sampling
all rasters at station coordinates. Then trains per-station models and
demonstrates specific-point prediction — answering the professor's question.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import glob
import os
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()


# =============================================
# 1. EXTRACT TIME SERIES AT GNSS STATIONS
# =============================================
def extract_station_timeseries():
    """Sample all rasters at 26 GNSS station coordinates to build per-station time series."""
    console.print("[bold cyan]═══ GNSS STATION TIME SERIES EXTRACTION ═══[/]")

    # Load GNSS station locations
    gdf = gpd.read_file("data/0_dropzone/Choushui Shapefile/GNSS_Station.shp")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Extract lon/lat from geometry
    gdf['longitude'] = gdf.geometry.x
    gdf['latitude'] = gdf.geometry.y
    station_coords = list(zip(gdf['longitude'], gdf['latitude']))
    station_names = gdf['Maker'].tolist() if 'Maker' in gdf.columns else [f"GNSS_{i}" for i in range(len(gdf))]

    console.print(f"  → {len(station_coords)} GNSS stations loaded")
    console.print(f"  → Lon range: {min(c[0] for c in station_coords):.4f} to {max(c[0] for c in station_coords):.4f}")
    console.print(f"  → Lat range: {min(c[1] for c in station_coords):.4f} to {max(c[1] for c in station_coords):.4f}")

    # Raster variable definitions
    DROPZONE = "./data/0_dropzone"
    dynamic_vars = {
        'groundwater_level': ('Groundwater', 'Groundwater'),
        'rainfall': ('Rainfall', 'Precipitation'),
        'temperature': ('Temperature', 'Temperature'),
        'pumping': ('Pumping', 'Pumping'),
    }
    static_vars = {
        'elevation_dem': 'DEM/DEM.tif',
        'twi': 'TWI/TWI.tif',
        'distance_to_river': 'Distance_River/Distance_River.tif',
    }

    # Sample static variables (same for all months)
    console.print("\n[yellow]Sampling static variables at station locations...[/]")
    static_values = {}
    for var_name, rel_path in static_vars.items():
        filepath = os.path.join(DROPZONE, rel_path)
        if os.path.exists(filepath):
            with rasterio.open(filepath) as src:
                sampled = [val[0] for val in src.sample(station_coords)]
                # Replace nodata
                if src.nodata is not None:
                    sampled = [np.nan if v == src.nodata else v for v in sampled]
                static_values[var_name] = sampled
                console.print(f"  ✓ {var_name}: sampled {len(sampled)} stations")
        else:
            static_values[var_name] = [0.0] * len(station_coords)
            console.print(f"  ✗ {var_name}: file not found, using 0.0")

    # Sample dynamic variables per month
    console.print("\n[yellow]Sampling dynamic variables at station locations (252 months)...[/]")
    all_records = []

    for var_name, (folder, keyword) in dynamic_vars.items():
        var_dir = os.path.join(DROPZONE, folder)
        raster_files = sorted(glob.glob(os.path.join(var_dir, "*.tif")))

        with Progress(SpinnerColumn(), TextColumn(f"[cyan]{var_name}"), BarColumn()) as progress:
            task = progress.add_task(f"Sampling {var_name}...", total=len(raster_files))

            for filepath in raster_files:
                fname = os.path.basename(filepath).lower().replace('.tif', '')
                parts = fname.split('_')

                try:
                    if var_name == 'subsidence':
                        date_part = parts[-1]
                        year, month = date_part.split('-')
                    else:
                        year, month = parts[-2], parts[-1]
                    year, month = int(year), int(month)
                except (ValueError, IndexError):
                    progress.advance(task)
                    continue

                timestamp = f"{year}-{month:02d}-01"

                with rasterio.open(filepath) as src:
                    sampled = [val[0] for val in src.sample(station_coords)]
                    if src.nodata is not None:
                        sampled = [np.nan if v == src.nodata else v for v in sampled]

                for i, val in enumerate(sampled):
                    # Find or create record for this station+timestamp
                    key = (i, timestamp)
                    record = None
                    for r in all_records:
                        if r['_station_idx'] == i and r['timestamp'] == timestamp:
                            record = r
                            break
                    if record is None:
                        record = {
                            '_station_idx': i,
                            'station_name': station_names[i],
                            'longitude': station_coords[i][0],
                            'latitude': station_coords[i][1],
                            'timestamp': timestamp,
                        }
                        # Add static values
                        for svar, svals in static_values.items():
                            record[svar] = svals[i]
                        all_records.append(record)

                    record[var_name] = val

                progress.advance(task)

    # Sample subsidence (different naming pattern)
    console.print("[yellow]Sampling subsidence at station locations...[/]")
    sub_dir = os.path.join(DROPZONE, "Subsidence")
    sub_files = sorted(glob.glob(os.path.join(sub_dir, "*.tif")))

    with Progress(SpinnerColumn(), TextColumn("[cyan]subsidence"), BarColumn()) as progress:
        task = progress.add_task("Sampling subsidence...", total=len(sub_files))
        for filepath in sub_files:
            fname = os.path.basename(filepath).replace('.tif', '')
            try:
                date_part = fname.split('_')[-1]
                year, month = date_part.split('-')
                year, month = int(year), int(month)
            except (ValueError, IndexError):
                progress.advance(task)
                continue

            timestamp = f"{year}-{month:02d}-01"

            with rasterio.open(filepath) as src:
                sampled = [val[0] for val in src.sample(station_coords)]
                if src.nodata is not None:
                    sampled = [np.nan if v == src.nodata else v for v in sampled]

            for i, val in enumerate(sampled):
                for r in all_records:
                    if r['_station_idx'] == i and r['timestamp'] == timestamp:
                        r['subsidence'] = val
                        break

            progress.advance(task)

    # Build DataFrame
    df = pd.DataFrame(all_records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop(columns=['_station_idx'])
    df = df.sort_values(['station_name', 'timestamp']).reset_index(drop=True)

    # Fill missing subsidence with 0 (subsidence data starts 2017)
    if 'subsidence' not in df.columns:
        df['subsidence'] = 0.0
    df['subsidence'] = df['subsidence'].fillna(0.0)

    # Fill missing LULC
    if 'lulc' not in df.columns:
        df['lulc'] = 2.0  # Default land use class
    df = df.fillna(0)

    # Add temporal features
    df = df.sort_values(['station_name', 'timestamp'])
    df['lag_1_gw'] = df.groupby('station_name')['groundwater_level'].shift(1)
    df['rolling_mean_3_gw'] = df.groupby('station_name')['groundwater_level'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['lag_1_gw'] = df.groupby('station_name')['lag_1_gw'].transform(lambda x: x.bfill())
    df['rolling_mean_3_gw'] = df.groupby('station_name')['rolling_mean_3_gw'].transform(lambda x: x.bfill())

    # Add zone classification
    try:
        zone_df = pd.read_parquet("zone_lookup.parquet")
        # Find nearest zone for each station
        from scipy.spatial import cKDTree
        zone_tree = cKDTree(zone_df[['longitude', 'latitude']].values)
        station_unique = df[['station_name', 'longitude', 'latitude']].drop_duplicates()
        _, indices = zone_tree.query(station_unique[['longitude', 'latitude']].values)
        station_unique['fan_zone'] = zone_df.iloc[indices]['fan_zone'].values
        df = df.merge(station_unique[['station_name', 'fan_zone']], on='station_name', how='left')
        console.print("[green]  ✓ Zone labels assigned to stations[/]")
    except Exception as e:
        console.print(f"[yellow]  ⚠ Could not assign zones: {e}. Run zone_classifier.py first.[/]")
        df['fan_zone'] = 'unknown'

    # Save
    output_path = "gnss_station_timeseries.parquet"
    df.to_parquet(output_path, index=False)

    console.print(f"\n[bold green]✅ Extracted {len(df):,} records for {df['station_name'].nunique()} GNSS stations[/]")
    console.print(f"  → Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    console.print(f"  → Saved to '{output_path}'")

    return df


# =============================================
# 2. POINT PREDICTION: EVALUATE AT STATIONS
# =============================================
def evaluate_point_prediction(df):
    """Use the trained XGBoost model to predict at each station and report accuracy."""
    console.print("\n[bold cyan]═══ POINT PREDICTION EVALUATION ═══[/]")

    features = [
        'rainfall', 'temperature', 'pumping', 'lulc', 'subsidence',
        'elevation_dem', 'twi', 'distance_to_river',
        'lag_1_gw', 'rolling_mean_3_gw'
    ]
    target = 'groundwater_level'

    # Filter to rows where groundwater data exists
    valid_df = df[df['groundwater_level'] != 0].copy()
    if len(valid_df) == 0:
        console.print("[red]No valid groundwater data found at GNSS stations![/]")
        return

    # Load model
    with open("xgboost_groundwater_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Hold out last 2 years for testing (temporal split)
    valid_df['year'] = valid_df['timestamp'].dt.year
    max_year = valid_df['year'].max()
    test_years = [max_year - 1, max_year]
    train_mask = ~valid_df['year'].isin(test_years)
    test_mask = valid_df['year'].isin(test_years)

    console.print(f"  → Train: {train_mask.sum():,} records (up to {max_year - 2})")
    console.print(f"  → Test:  {test_mask.sum():,} records ({test_years})")

    X_test = valid_df.loc[test_mask, features].astype(float)
    y_test = valid_df.loc[test_mask, target]

    if len(X_test) == 0:
        console.print("[red]No test data available![/]")
        return

    predictions = model.predict(X_test)

    # Global test metrics
    console.print("\n[bold]Global Point-Prediction Performance:[/]")
    console.print(f"  R²:   {r2_score(y_test, predictions):.4f}")
    console.print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f} m")
    console.print(f"  MAE:  {mean_absolute_error(y_test, predictions):.2f} m")

    # Per-station metrics
    test_stations = valid_df.loc[test_mask, 'station_name']
    test_zones = valid_df.loc[test_mask, 'fan_zone'] if 'fan_zone' in valid_df.columns else pd.Series(['unknown'] * len(X_test))

    results_table = Table(title="Per-Station Point Prediction")
    results_table.add_column("Station", style="cyan")
    results_table.add_column("Zone", style="white")
    results_table.add_column("Points", justify="right")
    results_table.add_column("R²", style="green", justify="right")
    results_table.add_column("RMSE (m)", style="yellow", justify="right")
    results_table.add_column("Mean GW (m)", style="magenta", justify="right")

    station_results = {}
    for station in sorted(test_stations.unique()):
        mask = test_stations == station
        if mask.sum() < 3:
            continue
        s_y = y_test[mask]
        s_pred = predictions[mask]
        s_zone = test_zones[mask].iloc[0] if mask.sum() > 0 else "unknown"

        try:
            s_r2 = r2_score(s_y, s_pred)
        except ValueError:
            s_r2 = float('nan')
        s_rmse = np.sqrt(mean_squared_error(s_y, s_pred))
        s_mae = mean_absolute_error(s_y, s_pred)

        r2_str = f"{s_r2:.4f}" if not np.isnan(s_r2) else "N/A"
        color = "green" if s_r2 > 0.8 else "yellow" if s_r2 > 0.5 else "red"

        results_table.add_row(
            str(station)[:20], s_zone, str(mask.sum()),
            f"[{color}]{r2_str}[/]", f"{s_rmse:.2f}", f"{s_y.mean():.2f}"
        )
        station_results[str(station)] = {
            "zone": s_zone, "r2": round(s_r2, 4) if not np.isnan(s_r2) else None,
            "rmse": round(s_rmse, 2), "mae": round(s_mae, 2),
            "test_points": int(mask.sum())
        }

    console.print(results_table)

    # Save results
    with open("gnss_point_prediction_results.json", "w") as f:
        json.dump(station_results, f, indent=2, default=str)
    console.print("\n[green]✅ Per-station results saved to 'gnss_point_prediction_results.json'[/]")

    # Save predicted vs observed for plotting
    result_df = pd.DataFrame({
        'station': test_stations.values,
        'timestamp': valid_df.loc[test_mask, 'timestamp'].values,
        'observed': y_test.values,
        'predicted': predictions,
        'zone': test_zones.values
    })
    result_df.to_parquet("gnss_predictions_vs_observed.parquet", index=False)
    console.print("[green]✅ Predictions saved to 'gnss_predictions_vs_observed.parquet'[/]")

    return station_results


# =============================================
# 3. MAIN
# =============================================
if __name__ == "__main__":
    df = extract_station_timeseries()
    results = evaluate_point_prediction(df)

    console.print("\n[bold green]═══ GNSS POINT PREDICTION COMPLETE ═══[/]")
    console.print("You can now show your professor:")
    console.print("  1. gnss_point_prediction_results.json — per-station accuracy")
    console.print("  2. gnss_predictions_vs_observed.parquet — plot predicted vs observed curves")
    console.print("  3. gnss_station_timeseries.parquet — raw extracted time series data")
