import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger("vv_pipeline")


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKER = "blocker"


@dataclass
class ValidationResult:
    stage: str
    check_name: str
    passed: bool
    severity: Severity
    message: str
    affected_rows: int = 0
    details: dict = field(default_factory=dict)


class ValidationPipeline:
    """4-stage V&V pipeline for groundwater data."""

    SCHEMA = {
        "timestamp": "datetime64[ns]",
        "longitude": "float64",
        "latitude": "float64",
        "groundwater_level": "float64",
        "rainfall": "float64",
        "temperature": "float64",
        "pumping": "float64",
        "lulc": "float64",
        "subsidence": "float64",
        "elevation_dem": "float64",
        "twi": "float64",
        "distance_to_river": "float64",
    }

    PHYSICAL_BOUNDS = {
        "groundwater_level": (-500, 100),   # meters (below/above surface)
        "rainfall": (0, 1500),              # mm/month
        "temperature": (-10, 50),           # °C
        "pumping": (0, 300000),             # extraction units (observed up to 205k)
        "subsidence": (-500, 500),          # mm/year
        "latitude": (21.5, 26.0),           # Taiwan bounding box
        "longitude": (119.5, 122.5),        # Taiwan bounding box
    }

    def __init__(self, z_threshold: float = 4.0, hitl_threshold: float = 0.7):
        self.z_threshold = z_threshold
        self.hitl_threshold = hitl_threshold
        self.results = []
        self.flagged_for_review = pd.DataFrame()

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[ValidationResult]]:
        """Execute all 4 validation stages. Returns (cleaned_df, results)."""
        self.results = []

        df = self._stage1_schema(df)
        df = self._stage2_statistical(df)
        df = self._stage3_physical(df)
        self._stage4_hitl_gate(df)

        return df, self.results

    # --- STAGE 1: Schema Validation ---
    def _stage1_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check all required columns exist
        missing = set(self.SCHEMA.keys()) - set(df.columns)
        self.results.append(ValidationResult(
            stage="schema", check_name="column_presence",
            passed=len(missing) == 0,
            severity=Severity.BLOCKER if missing else Severity.INFO,
            message=f"Missing columns: {missing}" if missing else "All columns present",
        ))

        # Check for null values
        common_cols = list(set(self.SCHEMA.keys()) & set(df.columns))
        null_counts = df[common_cols].isnull().sum()
        total_nulls = null_counts.sum()
        self.results.append(ValidationResult(
            stage="schema", check_name="null_audit",
            passed=total_nulls == 0,
            severity=Severity.WARNING if total_nulls > 0 else Severity.INFO,
            message=f"Found {total_nulls} null values across {(null_counts > 0).sum()} columns",
            affected_rows=int(total_nulls),
            details=null_counts[null_counts > 0].to_dict(),
        ))

        return df

    # --- STAGE 2: Statistical Anomaly Detection ---
    def _stage2_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        anomaly_mask = pd.Series(False, index=df.index)

        for col in numeric_cols:
            if col in ["longitude", "latitude"]:
                continue
            z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-8))
            col_outliers = z_scores > self.z_threshold
            anomaly_mask |= col_outliers

            if col_outliers.sum() > 0:
                self.results.append(ValidationResult(
                    stage="statistical", check_name=f"zscore_{col}",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"{col}: {col_outliers.sum()} rows exceed Z={self.z_threshold}",
                    affected_rows=int(col_outliers.sum()),
                ))

        # Temporal autocorrelation check (detect sensor dropouts)
        if "timestamp" in df.columns:
            time_gaps = df.groupby(["longitude", "latitude"])["timestamp"].diff()
            large_gaps = time_gaps > pd.Timedelta(days=45)  # >45 days = suspicious gap
            if large_gaps.sum() > 0:
                self.results.append(ValidationResult(
                    stage="statistical", check_name="temporal_continuity",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"{large_gaps.sum()} temporal gaps > 45 days detected",
                    affected_rows=int(large_gaps.sum()),
                ))

        return df

    # --- STAGE 3: Physical Constraint Checker ---
    def _stage3_physical(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in self.PHYSICAL_BOUNDS.items():
            if col not in df.columns:
                continue
            violations = (df[col] < lo) | (df[col] > hi)
            if violations.sum() > 0:
                self.results.append(ValidationResult(
                    stage="physical", check_name=f"bounds_{col}",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message=f"{col}: {violations.sum()} values outside [{lo}, {hi}]",
                    affected_rows=int(violations.sum()),
                    details={"min": float(df[col].min()), "max": float(df[col].max())},
                ))

        # Conservation check: groundwater shouldn't jump >20m in one month
        if "groundwater_level" in df.columns and "timestamp" in df.columns:
            sorted_df = df.sort_values(["longitude", "latitude", "timestamp"])
            gw_delta = sorted_df.groupby(["longitude", "latitude"])["groundwater_level"].diff().abs()
            impossible_jumps = gw_delta > 20
            if impossible_jumps.sum() > 0:
                self.results.append(ValidationResult(
                    stage="physical", check_name="gw_jump_check",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message=f"{impossible_jumps.sum()} impossible GW jumps (>20m/month)",
                    affected_rows=int(impossible_jumps.sum()),
                ))

        return df

    # --- STAGE 4: Human-in-the-Loop Gate ---
    def _stage4_hitl_gate(self, df: pd.DataFrame):
        critical_failures = [r for r in self.results if r.severity in (Severity.CRITICAL, Severity.BLOCKER) and not r.passed]
        warning_failures = [r for r in self.results if r.severity == Severity.WARNING and not r.passed]

        # Compute confidence score (0=terrible, 1=perfect)
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        confidence = passed_checks / max(total_checks, 1)

        needs_human_review = (
            len(critical_failures) > 0 or
            confidence < self.hitl_threshold
        )

        self.results.append(ValidationResult(
            stage="hitl_gate", check_name="confidence_score",
            passed=not needs_human_review,
            severity=Severity.BLOCKER if needs_human_review else Severity.INFO,
            message=f"Confidence: {confidence:.2%} | Critical: {len(critical_failures)} | Warnings: {len(warning_failures)}",
            details={
                "confidence": confidence,
                "requires_human_review": needs_human_review,
                "critical_issues": [r.message for r in critical_failures],
            },
        ))

if __name__ == "__main__":
    print("Testing Validation Pipeline...")
    pipeline = ValidationPipeline()
    try:
        df = pd.read_parquet("data/validated")
        cleaned_df, results = pipeline.run(df)
        for r in results:
            icon = "✅" if r.passed else "❌"
            print(f"{icon} [{r.stage}] {r.check_name}: {r.message}")
    except Exception as e:
        print(f"Test failed: {e}")
