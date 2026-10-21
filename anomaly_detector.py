import pandas as pd
import numpy as np

class PredictiveAnomalyDetector:
    """
    Flags data points where the XGBoost model's prediction
    diverges significantly from the observed value.
    This catches subtle sensor drift, calibration errors,
    and physically implausible data that passes basic checks.
    """

    def __init__(self, world_model, divergence_threshold: float = 3.0):
        self.model = world_model
        self.threshold = divergence_threshold  # in std deviations

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        features = [
            "rainfall", "temperature",
            "pumping", "lulc", "subsidence", "elevation_dem",
            "twi", "distance_to_river", "lag_1_gw", "rolling_mean_3_gw"
        ]
        
        # We need to make sure the required features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for anomaly detection: {missing_features}")
            
        predictions = self.model.predict(df[features].astype(float))
        residuals = df["groundwater_level"].values - predictions
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)

        df = df.copy()
        df["predicted_gw"] = predictions
        df["residual"] = residuals
        df["residual_zscore"] = (residuals - residual_mean) / (residual_std + 1e-8)
        df["is_anomaly"] = df["residual_zscore"].abs() > self.threshold
        df["anomaly_severity"] = df["residual_zscore"].abs().clip(0, 10) / 10

        flagged = df[df["is_anomaly"]].copy()
        if len(flagged) > 0:
            flagged["reason"] = flagged.apply(
                lambda row: (
                    f"Observed GW={row['groundwater_level']:.2f}m vs "
                    f"Predicted={row['predicted_gw']:.2f}m "
                    f"(Z={row['residual_zscore']:.1f}σ)"
                ), axis=1
            )
        return flagged
