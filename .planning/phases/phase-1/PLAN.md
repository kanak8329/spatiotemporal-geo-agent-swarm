# PLAN — Phase 1: Fix Raster Ingestion Pipeline
_Status: READY TO EXECUTE_

---

## Diagnosis Summary

After auditing the codebase, the data pipeline is split across two scripts:

1. `raster_to_ml_pipeline.py` — reads ONLY Groundwater TIFFs. Already correct (iterates all years via os.listdir). Output: `ml_training_data.parquet` (groundwater_level + lon/lat/timestamp only)
2. `multi_variable_fusion.py` — joins other variables onto the groundwater data. **This is where missing values occur.**

### Root Causes Identified

| # | Bug | Location | Impact |
|---|---|---|---|
| 1 | `expected_prefix` uses `var_name.split("_")[0]` — so "subsidence" becomes "subsidence" but files are named `Subsidence_YYYY-MM.tif` (dashes, not underscores in date) | `multi_variable_fusion.py:42` | All Subsidence values = NaN |
| 2 | LULC is yearly (2016, 2021 only) but code looks for monthly files — prefix `LULC_2001_01` never matches | `multi_variable_fusion.py:42` | All LULC values = NaN |
| 3 | Pumping only starts 2007, but groundwater starts 2001 — 6 years of pumping are NaN (forward-fill only helps within a location, not backward) | `multi_variable_fusion.py:48,93` | 2001-2006 Pumping = NaN/0 |
| 4 | Static vars (DEM, TWI, Distance_River) each cause a `.merge()` that creates `_x/_y` columns — on 2nd+ merge, columns already exist, breaking the merge | `multi_variable_fusion.py:84-87` | Static vars partially missing |
| 5 | `ml_training_data.parquet` (input to fusion) is written by `raster_to_ml_pipeline.py` which only reads `./data/Groundwater` — but this part works correctly, it does read all 252 files | `raster_to_ml_pipeline.py:7` | Groundwater base is correct |

---

## Tasks

### TASK-1: Fix Subsidence filename prefix matching
**File:** `multi_variable_fusion.py`
**Lines:** 42-45

The Subsidence files use `Subsidence_YYYY-MM.tif` (dash separator between year and month, not underscore). The current code builds `subsidence_YYYY_MM` (underscore) so it never matches.

**Fix:** For the `subsidence` variable, build the prefix using dash: `Subsidence_YYYY-MM`

```python
# Current (broken):
expected_prefix = f"{var_name.split('_')[0]}_{year}_{month:02d}"

# Fixed — handle subsidence separately:
var_prefix = var_name.capitalize()
if var_name == "subsidence":
    expected_prefix = f"{var_prefix}_{year}-{month:02d}"
else:
    expected_prefix = f"{var_prefix}_{year}_{month:02d}"
```

**UAT:** After fix, Subsidence column has non-NaN values for 2017+

---

### TASK-2: Fix LULC — yearly files, not monthly
**File:** `multi_variable_fusion.py`
**Lines:** 29-58 (DYNAMIC_VARS loop)

LULC only has 2 files: `LULC_2016.tif` and `LULC_2021.tif`. These are NOT monthly. Move LULC to a new `YEARLY_VARS` dict and match on year only (nearest year).

**Fix:**
```python
YEARLY_VARS = {
    "lulc": "./data/LULC",
}
# Remove lulc from DYNAMIC_VARS

# New yearly matching logic:
for var_name, folder_path in YEARLY_VARS.items():
    df[var_name] = np.nan
    available_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
    # Parse available years
    available_years = {}
    for fname in available_files:
        try:
            yr = int(fname.replace('.tif','').split('_')[-1])
            available_years[yr] = os.path.join(folder_path, fname)
        except: pass
    all_years = sorted(available_years.keys())
    
    for timestamp, group in df.groupby('timestamp'):
        # Use nearest available year (find closest)
        target_yr = timestamp.year
        nearest_yr = min(all_years, key=lambda y: abs(y - target_yr))
        filepath = available_years[nearest_yr]
        with rasterio.open(filepath) as src:
            coords = list(zip(group['longitude'], group['latitude']))
            sampled = [val[0] if val[0] != src.nodata else np.nan for val in src.sample(coords)]
            df.loc[group.index, var_name] = sampled
```

**UAT:** LULC column has non-NaN values for all timestamps

---

### TASK-3: Fix static variable merge bug
**File:** `multi_variable_fusion.py`
**Lines:** 67-87

The `.merge()` approach for static vars creates `_x/_y` column conflicts when called multiple times. Replace with a direct coordinate-index lookup using a dict.

**Fix:**
```python
for var_name, folder_path in STATIC_VARS.items():
    df[var_name] = np.nan
    if not os.path.exists(folder_path):
        continue
    static_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    if not static_files:
        continue
    filepath = os.path.join(folder_path, static_files[0])
    print(f"  -> Mapping {var_name} from {static_files[0]}")
    with rasterio.open(filepath) as src:
        coord_list = list(zip(unique_coords['longitude'], unique_coords['latitude']))
        sampled = [val[0] if val[0] != src.nodata else np.nan for val in src.sample(coord_list)]
        unique_coords_copy = unique_coords.copy()
        unique_coords_copy[var_name] = sampled
        df = df.merge(unique_coords_copy[['longitude', 'latitude', var_name]],
                      on=['longitude', 'latitude'], how='left')
```

**UAT:** DEM, TWI, distance_to_river columns have non-NaN values

---

### TASK-4: Handle Pumping gap (2001-2006) with backward fill
**File:** `multi_variable_fusion.py`
**Lines:** 89-96

Pumping data starts 2007. For 2001-2006, backward-fill from 2007 (earliest available) so we don't zero-fill the first 6 years of training data.

**Fix:**
```python
df = df.sort_values(by=['longitude', 'latitude', 'timestamp'])
df[list(DYNAMIC_VARS.keys())] = df.groupby(['longitude', 'latitude'])[list(DYNAMIC_VARS.keys())].ffill()
df[list(DYNAMIC_VARS.keys())] = df.groupby(['longitude', 'latitude'])[list(DYNAMIC_VARS.keys())].bfill()
df.fillna(0, inplace=True)
```

**UAT:** Pumping column has zero NaN values after healing

---

### TASK-5: Add validation report at the end
**File:** `multi_variable_fusion.py`
**End of file**

```python
print("\n=== DATA VALIDATION REPORT ===")
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Unique months: {df['timestamp'].nunique()}")
print("\nMissing value audit:")
print(df.isnull().sum())
print(f"\nTarget column (groundwater_level) stats:")
print(df['groundwater_level'].describe())
```

**UAT:** Zero NaN values printed in validation report for all key columns

---

## Execution Order

```
TASK-3 (static merge fix)  ← independent, fix first to avoid column pollution
TASK-1 (subsidence prefix) ← independent
TASK-2 (LULC yearly)       ← independent  
TASK-4 (pumping bfill)     ← depends on TASK-1, 2, 3 being in place
TASK-5 (validation report) ← last, always
```

Then re-run:
```powershell
cd c:\Users\Kanak\groundwater_thesis
python raster_to_ml_pipeline.py    # regenerate base parquet
python multi_variable_fusion.py    # regenerate master dataset
```

---

## Acceptance Criteria (UAT)

- [ ] `ml_master_thesis_data_v2.parquet` date range spans 2001-01-01 to 2021-12-01
- [ ] All 9 columns present: groundwater_level, rainfall, temperature, pumping, lulc, subsidence, elevation_dem, twi, distance_to_river
- [ ] Zero NaN values in groundwater_level, rainfall, temperature
- [ ] DuckDB query `SELECT COUNT(*) FROM aquifer_data WHERE year(timestamp) > 2001` returns > 0
- [ ] Subsidence column has non-NaN values for 2017+ timestamps
- [ ] LULC column has non-NaN values for all timestamps
