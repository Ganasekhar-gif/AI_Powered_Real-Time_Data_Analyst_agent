# core/utils/dataframe_manager.py

import os
from typing import Optional
import pandas as pd

from core.executors.duckdb_exec import DuckRunner


class DataFrameManager:
    """Keeps the current working DataFrame in memory, synced with DuckDB, and persisted to disk."""

    def __init__(self, runner: DuckRunner, save_dir: str = "data/frames", table_name: str = "current_df"):
        self.current_df: Optional[pd.DataFrame] = None
        self.runner = runner
        self.save_dir = save_dir
        self.table_name = table_name
        os.makedirs(save_dir, exist_ok=True)

        # Try to auto-load last persisted dataframe from DuckDB
        try:
            if self.table_name in self.runner.list_tables():
                df = self.runner.load_dataframe(self.table_name)
                self.current_df = df
                print(f"[INFO] Restored dataframe '{self.table_name}' from DuckDB with {len(df)} rows")
            else:
                print("[INFO] No existing dataframe found in DuckDB, starting fresh")
        except Exception as e:
            print(f"[WARN] Could not restore dataframe from DuckDB: {e}")

    def set(self, df: pd.DataFrame, name: str = "current_df"):
        """Update current DataFrame, re-register in DuckDB, and persist snapshots."""
        self.current_df = df

        # Register inside DuckDB (persistent table)
        self.runner.register_dataframe(name, df, persist=True)

        # Save snapshot to disk (CSV + Parquet)
        csv_path = os.path.join(self.save_dir, f"{name}.csv")
        parquet_path = os.path.join(self.save_dir, f"{name}.parquet")

        try:
            df.to_csv(csv_path, index=False)
            df.to_parquet(parquet_path, index=False)
            print(f"[INFO] Saved dataframe snapshot → {csv_path}, {parquet_path}")
        except Exception as e:
            print(f"[WARN] Failed to save dataframe snapshot: {e}")

    def get(self) -> Optional[pd.DataFrame]:
        return self.current_df
