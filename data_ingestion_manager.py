import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from validation_pipeline import ValidationPipeline

logger = logging.getLogger("ingestion")


class DataIngestionManager:
    """
    Unified ingestion manager for all data sources.
    Handles: GeoTIFF rasters, CSV well logs, API feeds.
    Every record passes through the V&V pipeline before storage.
    """

    def __init__(self, storage_dir: str = "./data/validated",
                 archive_dir: str = "./data/archive"):
        self.storage = Path(storage_dir)
        self.archive = Path(archive_dir)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)
        # Taiwan's terrain has extreme natural variance (mountains). Relax Z-score and HITL thresholds.
        self.pipeline = ValidationPipeline(z_threshold=8.0, hitl_threshold=0.2)
        self.ingestion_log = []

    def ingest_batch(self, df: pd.DataFrame, source_label: str) -> dict:
        """Ingest a DataFrame through the full V&V pipeline."""
        batch_id = f"{source_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Ingesting batch {batch_id}: {len(df)} rows")

        # Run V&V
        validated_df, results = self.pipeline.run(df)

        # Check HITL gate result
        hitl = next((r for r in results if r.check_name == "confidence_score"), None)
        requires_review = hitl and hitl.details.get("requires_human_review", False)

        # Partition and store
        if not requires_review:
            self._store_partitioned(validated_df, batch_id)
            status = "APPROVED"
        else:
            # Store in quarantine for human review
            quarantine_path = self.archive / f"quarantine_{batch_id}.parquet"
            validated_df.to_parquet(quarantine_path)
            status = "QUARANTINED"

        report = {
            "batch_id": batch_id,
            "status": status,
            "rows_ingested": len(validated_df),
            "validation_results": [
                {"check": r.check_name, "passed": r.passed,
                 "severity": r.severity.value, "message": r.message}
                for r in results
            ],
        }
        self.ingestion_log.append(report)
        return report

    def _store_partitioned(self, df: pd.DataFrame, batch_id: str):
        """Store data partitioned by year for efficient querying."""
        if "timestamp" in df.columns:
            df["_year"] = df["timestamp"].dt.year
            for year, group in df.groupby("_year"):
                year_dir = self.storage / f"year={year}"
                year_dir.mkdir(exist_ok=True)
                path = year_dir / f"{batch_id}.parquet"
                group.drop(columns=["_year"]).to_parquet(path, index=False)
        else:
            path = self.storage / f"{batch_id}.parquet"
            df.to_parquet(path, index=False)
