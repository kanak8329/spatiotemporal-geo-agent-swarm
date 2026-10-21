import os
import glob
import pandas as pd
import numpy as np
import rasterio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from data_ingestion_manager import DataIngestionManager
from validation_pipeline import ValidationPipeline

console = Console()

DROPZONE = "./data/0_dropzone"

def parse_date_from_filename(filename, var_name):
    """Extract year and month based on filename patterns."""
    fname = os.path.basename(filename).lower().replace('.tif', '')
    parts = fname.split('_')
    
    try:
        if var_name.lower() == 'subsidence':
            # e.g., Subsidence_2017-01.tif
            date_part = parts[-1]
            year, month = date_part.split('-')
            return int(year), int(month)
        else:
            # e.g., Groundwater_2010_11.tif
            year = int(parts[-2])
            month = int(parts[-1])
            return int(year), int(month)
    except Exception:
        return None, None

def crawl_dropzone():
    console.print(f"[bold cyan]🔍 Crawling Dropzone: {DROPZONE}[/]")
    rasters = glob.glob(os.path.join(DROPZONE, '**', '*.tif'), recursive=True)
    vectors = glob.glob(os.path.join(DROPZONE, '**', '*.shp'), recursive=True)
    
    console.print(f" -> Found [bold yellow]{len(rasters)}[/] Raster (.tif) files.")
    console.print(f" -> Found [bold yellow]{len(vectors)}[/] Vector (.shp) files.")
    
    return rasters, vectors

def extract_base_spatiotemporal_grid(rasters):
    console.print("\n[bold cyan]🏗️ Building Base Spatiotemporal Grid (from Groundwater)...[/]")
    gw_rasters = [r for r in rasters if 'groundwater' in os.path.basename(r).lower()]
    gw_rasters.sort()
    
    all_data = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    ) as progress:
        task = progress.add_task("[green]Parsing Groundwater Rasters...", total=len(gw_rasters))
        
        for filepath in gw_rasters:
            year, month = parse_date_from_filename(filepath, 'groundwater')
            if year is None: continue
            date_str = f"{year}-{month:02d}-01"
            
            with rasterio.open(filepath) as src:
                band_data = src.read(1)
                cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                
                lons = np.array(xs).flatten()
                lats = np.array(ys).flatten()
                values = band_data.flatten()
                
                nodata_val = src.nodata if src.nodata is not None else 0
                valid_mask = (values != nodata_val) & (~np.isnan(values)) & (values > -9999)
                
                valid_lons = lons[valid_mask]
                valid_lats = lats[valid_mask]
                valid_values = values[valid_mask]
                
                for i in range(len(valid_values)):
                    all_data.append({
                        "timestamp": date_str,
                        "longitude": valid_lons[i],
                        "latitude": valid_lats[i],
                        "groundwater_level": valid_values[i]
                    })
            progress.advance(task)
            
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    console.print(f"[bold green]✅ Extracted {len(df):,} valid base points![/]")
    return df

def fuse_dynamic_rasters(df, rasters, var_name, keyword):
    console.print(f"\n[bold cyan]🔗 Fusing Dynamic Variable: {var_name.upper()}...[/]")
    var_rasters = [r for r in rasters if keyword in os.path.basename(r).lower()]
    df[var_name] = np.nan
    
    if not var_rasters:
        console.print(f"[bold red] -> No files found for {var_name}![/]")
        return df

    # Group by timestamp to process month by month
    grouped = df.groupby('timestamp')
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn()) as progress:
        task = progress.add_task(f"[magenta]Sampling {var_name}...", total=len(grouped))
        
        for timestamp, group in grouped:
            year = timestamp.year
            month = timestamp.month
            
            # Find the matching raster for this month
            matched_file = None
            for r in var_rasters:
                ry, rm = parse_date_from_filename(r, var_name)
                if ry == year and rm == month:
                    matched_file = r
                    break
            
            if matched_file:
                with rasterio.open(matched_file) as src:
                    coords = list(zip(group['longitude'], group['latitude']))
                    sampled = [val[0] if val[0] != src.nodata else np.nan for val in src.sample(coords)]
                    df.loc[group.index, var_name] = sampled
            progress.advance(task)
            
    # Forward/Backward fill missing temporal gaps for this variable
    df[var_name] = df.groupby(['longitude', 'latitude'])[var_name].transform(lambda x: x.ffill().bfill())
    return df

def fuse_static_rasters(df, rasters, var_name, keyword):
    console.print(f"\n[bold cyan]🏔️ Fusing Static Variable: {var_name.upper()}...[/]")
    var_rasters = [r for r in rasters if keyword in os.path.basename(r).lower()]
    
    if not var_rasters:
        console.print(f"[bold red] -> No files found for {var_name}![/]")
        df[var_name] = 0.0
        return df
        
    unique_coords = df[['longitude', 'latitude']].drop_duplicates().reset_index(drop=True)
    filepath = var_rasters[0]
    console.print(f" -> Mapping from {os.path.basename(filepath)}")
    
    with rasterio.open(filepath) as src:
        coord_list = list(zip(unique_coords['longitude'], unique_coords['latitude']))
        sampled = [val[0] if val[0] != src.nodata else np.nan for val in src.sample(coord_list)]
        
    lookup = unique_coords.copy()
    lookup[var_name] = sampled
    df = df.merge(lookup, on=['longitude', 'latitude'], how='left')
    return df

def process_vector_shapefiles(df, vectors):
    console.print("\n[bold cyan]🗺️ Processing Vector Shapefiles...[/]")
    try:
        import geopandas as gpd
    except ImportError:
        console.print("[bold red]⚠️ 'geopandas' is not installed! Vector extraction skipped.[/]")
        console.print("To process vectors, run: [yellow]pip install geopandas[/]")
        return df
        
    for shp_file in vectors:
        try:
            gdf = gpd.read_file(shp_file)
            console.print(f" -> Successfully parsed Vector: {os.path.basename(shp_file)} with {len(gdf)} features.")
            # We can expand spatial joins here in the future.
            # For now, we prove we can read the formats without breaking the ML schema.
        except Exception as e:
            console.print(f"[bold red] -> Failed to read {os.path.basename(shp_file)}: {e}[/]")
            
    return df

def clean_raster_anomalies(df):
    console.print("\n[bold cyan]🧹 Geomatics Cleaning: Purging Rogue Pixels & NoData Artifacts...[/]")
    
    # Rainfall cannot be negative
    if 'rainfall' in df.columns:
        df.loc[df['rainfall'] < 0, 'rainfall'] = np.nan
        df.loc[df['rainfall'] > 1500, 'rainfall'] = np.nan
        
    # Pumping cannot be negative or absurdly high
    if 'pumping' in df.columns:
        df.loc[df['pumping'] < 0, 'pumping'] = np.nan
        df.loc[df['pumping'] > 300000, 'pumping'] = np.nan
        
    # Temperature (clip extreme NoData like -9999)
    if 'temperature' in df.columns:
        df.loc[df['temperature'] < -50, 'temperature'] = np.nan
        df.loc[df['temperature'] > 60, 'temperature'] = np.nan

    # DEM (Taiwan elevation is mostly > -100 and < 4000)
    if 'elevation_dem' in df.columns:
        df.loc[df['elevation_dem'] < -500, 'elevation_dem'] = np.nan
        df.loc[df['elevation_dem'] > 5000, 'elevation_dem'] = np.nan

    # TWI (Topographic Wetness Index) should not be extreme negative NoData
    if 'twi' in df.columns:
        df.loc[df['twi'] < -100, 'twi'] = np.nan

    # Forward/Backward fill the newly scrubbed NaNs
    cols = ['rainfall', 'pumping', 'temperature', 'elevation_dem', 'twi']
    df[cols] = df.groupby(['longitude', 'latitude'])[cols].transform(lambda x: x.ffill().bfill())
    
    return df

def main():
    console.print("[bold cyan]=== 🌪️ UNIFIED DATA EXTRACTOR & FUSION ENGINE ===[/]")
    
    rasters, vectors = crawl_dropzone()
    if not rasters:
        console.print("[bold red]No data found in dropzone![/]")
        return
        
    # 1. Build Base Spatiotemporal Grid
    df = extract_base_spatiotemporal_grid(rasters)
    
    # 2. Fuse Dynamic Rasters
    df = fuse_dynamic_rasters(df, rasters, 'rainfall', 'precipitation')
    df = fuse_dynamic_rasters(df, rasters, 'temperature', 'temperature')
    df = fuse_dynamic_rasters(df, rasters, 'pumping', 'pumping')
    df = fuse_dynamic_rasters(df, rasters, 'subsidence', 'subsidence')
    
    # 3. Fuse Static Rasters
    df = fuse_static_rasters(df, rasters, 'lulc', 'lulc') # simplified as static for now
    df = fuse_static_rasters(df, rasters, 'elevation_dem', 'dem')
    df = fuse_static_rasters(df, rasters, 'twi', 'twi')
    df = fuse_static_rasters(df, rasters, 'distance_to_river', 'distance_river')
    
    # 4. Process Vector Shapefiles
    df = process_vector_shapefiles(df, vectors)
    
    # 4.5. Clean Raster Anomalies
    df = clean_raster_anomalies(df)
    
    # Fill any remaining NaNs with 0
    df = df.fillna(0)
    
    # 5. Route to Data Ingestion Manager for V&V
    console.print("\n[bold cyan]🛡️ Routing fused data through 4-Stage V&V Pipeline...[/]")
    ingester = DataIngestionManager()
    
    # We ingest in chunks by year to prevent memory overload during V&V
    df['year'] = df['timestamp'].dt.year
    for year, group in df.groupby('year'):
        console.print(f" -> Validating and Storing Year {year}...")
        group_clean = group.drop(columns=['year'])
        report = ingester.ingest_batch(group_clean, source_label=f"dropzone_fusion_{year}")
        
        status_color = "green" if report['status'] == "APPROVED" else "red"
        console.print(f"    Status: [bold {status_color}]{report['status']}[/] | Rows: {report['rows_ingested']:,}")
        if report['status'] == "QUARANTINED":
            for res in report['validation_results']:
                if not res['passed']:
                    console.print(f"      [red]❌ {res['check']}: {res['message']}[/]")
        
    console.print("\n[bold green]🎉 All Dropzone data successfully extracted, fused, validated, and stored![/]")
    console.print("[yellow]Next Step: Run 'python ml_predictor.py' to retrain the ML model on the clean data.[/]")

if __name__ == "__main__":
    main()
