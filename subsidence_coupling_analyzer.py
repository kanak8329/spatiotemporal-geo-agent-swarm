"""
Advance 3: Subsidence-Groundwater Spatiotemporal Coupling Analyzer
================================================================
Investigates the physical coupling between Sentinel-1 InSAR land subsidence
and groundwater drawdown across the Choushui River Alluvial Fan (2017–2021).

Key Hydrogeological Questions Answered:
1. Where does groundwater drawdown drive severe inelastic land compaction?
2. How does the coupling ratio (mm subsidence per meter of drawdown) differ
   across the Upper (proximal gravel), Middle (transitional), and Lower (distal clay) zones?
3. What is the subsidence hazard profile along the critical Taiwan High Speed Rail (THSR) corridor?
"""

import os
import json
import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd


def run_subsidence_coupling():
    print("=" * 70)
    print("🌊 ADVANCE 3: SUBSIDENCE-GROUNDWATER COUPLING ANALYZER")
    print("Study Area: Choushui River Alluvial Fan (2017–2021 InSAR + GW Overlap)")
    print("=" * 70)

    # 1. Load Zone Lookup
    print("\n[Step 1/5] Loading Hydrogeological Zones...")
    if not os.path.exists("zone_lookup.parquet"):
        raise FileNotFoundError("zone_lookup.parquet not found. Run zone_classifier.py first.")
    zone_df = pd.read_parquet("zone_lookup.parquet")
    print(f" Loaded {len(zone_df):,} grid cell zone classifications.")

    # 2. Compute Distance to Taiwan High Speed Rail (THSR)
    print("\n[Step 2/5] Spatial Overlay with Taiwan High Speed Rail (THSR.shp)...")
    thsr_path = "data/0_dropzone/Choushui Shapefile/THSR.shp"
    if os.path.exists(thsr_path):
        thsr_gdf = gpd.read_file(thsr_path).to_crs("EPSG:3826")
        union_thsr = thsr_gdf.union_all()

        pts_gdf = gpd.GeoDataFrame(
            zone_df,
            geometry=gpd.points_from_xy(zone_df.longitude, zone_df.latitude),
            crs="EPSG:4326"
        ).to_crs("EPSG:3826")

        zone_df["dist_to_thsr_km"] = (pts_gdf.distance(union_thsr) / 1000.0).round(2)
        zone_df["within_3km_thsr"] = zone_df["dist_to_thsr_km"] <= 3.0
        thsr_count = zone_df["within_3km_thsr"].sum()
        print(f" Calculated distances to THSR alignment. {thsr_count:,} cells are within the 3km statutory monitoring buffer.")
    else:
        print("  THSR shapefile not found; skipping THSR distance calculation.")
        zone_df["dist_to_thsr_km"] = 999.0
        zone_df["within_3km_thsr"] = False

    # 3. High-Performance Pixel-wise Regression in DuckDB
    print("\n[Step 3/5] Computing Pixel-wise Coupling & Drawdown via DuckDB...")
    con = duckdb.connect()
    
    # Register zones table in duckdb
    con.register("zones", zone_df)

    query = """
    WITH pixel_time_series AS (
        SELECT 
            v.longitude, 
            v.latitude,
            corr(v.groundwater_level, v.subsidence) AS r_gw_sub,
            regr_slope(v.subsidence, v.groundwater_level) AS compaction_sensitivity,
            min(v.subsidence) AS max_subsidence_mm,
            avg(v.subsidence) AS mean_subsidence_mm,
            avg(v.groundwater_level) AS mean_gw_m,
            max(v.groundwater_level) - min(v.groundwater_level) AS gw_fluctuation_range_m,
            max(v.subsidence) - min(v.subsidence) AS subsidence_span_mm
        FROM read_parquet('data/validated/**/*.parquet') v
        WHERE v.year >= 2017
        GROUP BY v.longitude, v.latitude
    )
    SELECT 
        p.longitude,
        p.latitude,
        z.fan_zone,
        z.dist_to_thsr_km,
        z.within_3km_thsr,
        round(p.r_gw_sub, 4) AS r_gw_sub,
        round(p.compaction_sensitivity, 4) AS compaction_sensitivity_mm_per_m,
        round(p.mean_subsidence_mm, 2) AS mean_subsidence_mm,
        round(p.max_subsidence_mm, 2) AS max_subsidence_mm,
        round(p.mean_gw_m, 2) AS mean_gw_m,
        round(p.gw_fluctuation_range_m, 2) AS gw_fluctuation_range_m,
        round(p.subsidence_span_mm, 2) AS subsidence_span_mm
    FROM pixel_time_series p
    JOIN zones z
      ON abs(p.longitude - z.longitude) < 0.0001 AND abs(p.latitude - z.latitude) < 0.0001
    """
    coupling_map_df = con.execute(query).df()
    print(f" Computed coupling metrics for {len(coupling_map_df):,} grid cells.")

    # Save coupling map parquet
    coupling_map_path = "subsidence_coupling_map.parquet"
    coupling_map_df.to_parquet(coupling_map_path, index=False)
    print(f" Saved full spatial coupling map -> {coupling_map_path}")

    # 4. Generate Summary Statistics
    print("\n[Step 4/5] Aggregating Hydrogeological Zone & THSR Statistics...")
    zone_stats = coupling_map_df.groupby("fan_zone").agg(
        cell_count=("longitude", "count"),
        mean_correlation=("r_gw_sub", "mean"),
        mean_sensitivity_mm_per_m=("compaction_sensitivity_mm_per_m", "mean"),
        mean_subsidence_mm=("mean_subsidence_mm", "mean"),
        peak_subsidence_mm=("max_subsidence_mm", "min"),
        mean_gw_level_m=("mean_gw_m", "mean")
    ).round(3).to_dict(orient="index")

    thsr_stats = coupling_map_df.groupby("within_3km_thsr").agg(
        cell_count=("longitude", "count"),
        mean_correlation=("r_gw_sub", "mean"),
        mean_sensitivity_mm_per_m=("compaction_sensitivity_mm_per_m", "mean"),
        mean_subsidence_mm=("mean_subsidence_mm", "mean"),
        peak_subsidence_mm=("max_subsidence_mm", "min")
    ).round(3).to_dict(orient="index")

    # Top 5 Subsidence Hotspots
    hotspots = coupling_map_df.sort_values("max_subsidence_mm").head(10)[
        ["longitude", "latitude", "fan_zone", "dist_to_thsr_km", "max_subsidence_mm", "r_gw_sub", "compaction_sensitivity_mm_per_m"]
    ].to_dict(orient="records")

    summary_results = {
        "study_period": "2017-2021 (60 months Sentinel-1 InSAR + Well Rasters)",
        "total_cells": int(len(coupling_map_df)),
        "zones": zone_stats,
        "thsr_corridor": {
            "within_3km": thsr_stats.get(True, {}),
            "outside_3km": thsr_stats.get(False, {})
        },
        "top_hotspots": hotspots
    }

    # Save summary JSON
    summary_path = "subsidence_coupling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f" Saved summary metrics -> {summary_path}")

    # 5. Display Formatted Results
    print("\n" + "=" * 70)
    print("📊 ADVANCE 3 RESULTS: ZONE-STRATIFIED SUBSIDENCE COUPLING")
    print("=" * 70)
    print(f"{'Zone':<12} | {'Cells':<8} | {'Avg Corr (r)':<14} | {'Sensitivity (mm/m)':<20} | {'Avg Sub (mm)':<14} | {'Peak Sub (mm)':<14}")
    print("-" * 92)
    for z_name in ["upper", "middle", "lower"]:
        info = zone_stats.get(z_name, {})
        print(f"{z_name.capitalize():<12} | {info.get('cell_count', 0):<8} | {info.get('mean_correlation', 0.0):<14.3f} | {info.get('mean_sensitivity_mm_per_m', 0.0):<20.3f} | {info.get('mean_subsidence_mm', 0.0):<14.2f} | {info.get('peak_subsidence_mm', 0.0):<14.2f}")

    print("\n" + "=" * 70)
    print("🚄 TAIWAN HIGH SPEED RAIL (THSR) CORRIDOR ANALYSIS (<= 3 km)")
    print("=" * 70)
    in_thsr = thsr_stats.get(True, {})
    out_thsr = thsr_stats.get(False, {})
    print(f"Inside 3km Buffer:  {in_thsr.get('cell_count', 0):,} cells | Avg Sub: {in_thsr.get('mean_subsidence_mm', 0):.2f} mm | Peak Sub: {in_thsr.get('peak_subsidence_mm', 0):.2f} mm | Corr: {in_thsr.get('mean_correlation', 0):.3f}")
    print(f"Outside 3km Buffer: {out_thsr.get('cell_count', 0):,} cells | Avg Sub: {out_thsr.get('mean_subsidence_mm', 0):.2f} mm | Peak Sub: {out_thsr.get('peak_subsidence_mm', 0):.2f} mm | Corr: {out_thsr.get('mean_correlation', 0):.3f}")
    print("=" * 70)
    print("✅ Advance 3 execution completed successfully.\n")


if __name__ == "__main__":
    run_subsidence_coupling()
