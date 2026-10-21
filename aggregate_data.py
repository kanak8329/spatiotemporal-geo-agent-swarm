import pandas as pd
import duckdb

print("Reading 2.2M row dataset from V&V Vault...")
df = pd.read_parquet('./data/validated')

# 1. Monthly Aggregation for Time-Series Line Charts
print("Aggregating monthly data...")
monthly_df = df.groupby('timestamp').mean(numeric_only=True).reset_index()
monthly_df.to_parquet('monthly_aggregate.parquet', index=False)

# 2. Correlation Matrix
print("Computing correlation matrix...")
# Exclude timestamp, lat, lon from correlation
corr_cols = [c for c in df.columns if c not in ['timestamp', 'longitude', 'latitude']]
corr_matrix = df[corr_cols].corr()
corr_matrix.to_csv('correlation_matrix.csv')

# 3. Spatial Sample (May 2017, dry season)
print("Extracting spatial sample for May 2017...")
spatial_df = df[df['timestamp'] == '2017-05-01']
spatial_df.to_parquet('spatial_sample_201705.parquet', index=False)

# 4. Generate EDA text findings
print("Generating EDA statistical report...")
with open('eda_stats.txt', 'w') as f:
    f.write("=== TEMPORAL TRENDS ===\n")
    f.write(f"Groundwater range: {monthly_df['groundwater_level'].min():.2f} to {monthly_df['groundwater_level'].max():.2f}\n")
    f.write(f"Pumping range: {monthly_df['pumping'].min():.2f} to {monthly_df['pumping'].max():.2f}\n")
    
    f.write("\n=== CORRELATIONS WITH GROUNDWATER ===\n")
    f.write(corr_matrix['groundwater_level'].sort_values(ascending=False).to_string())

print("Pre-computation complete! Outputs saved.")
