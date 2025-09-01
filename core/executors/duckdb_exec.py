import duckdb
import pandas as pd
import os


class DuckRunner:
    def __init__(self, db_path: str = "data/duckdb_store.duckdb"):
        # Persistent DB on disk instead of in-memory
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = duckdb.connect(database=db_path)

    # ---------------------------
    # Register DataFrame
    # ---------------------------
    def register_dataframe(self, table: str, df: pd.DataFrame, persist: bool = True):
        """
        Register a Pandas DataFrame into DuckDB.
        If persist=True, it creates a permanent table.
        If persist=False, it creates a temporary view.
        """
        if persist:
            self.con.execute(f"DROP TABLE IF EXISTS {table};")
            self.con.register("temp_df", df)
            self.con.execute(f"CREATE TABLE {table} AS SELECT * FROM temp_df;")
            self.con.unregister("temp_df")
        else:
            self.con.register(table, df)

    # Alias (shorter name)
    def register_df(self, table: str, df: pd.DataFrame, persist: bool = True):
        self.register_dataframe(table, df, persist=persist)

    # ---------------------------
    # Utilities
    # ---------------------------
    def get_schema(self, table: str) -> str:
        df = self.con.execute(f"PRAGMA table_info('{table}');").df()
        cols = ", ".join([f"{r['name']} ({r['type']})" for _, r in df.iterrows()])
        return f"{table} columns -> {cols}"

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def list_tables(self) -> list:
        """List all tables in DuckDB"""
        df = self.con.execute("SHOW TABLES;").df()
        return df["name"].tolist()

    def load_dataframe(self, table: str) -> pd.DataFrame:
        """Load a persisted DuckDB table into Pandas"""
        return self.con.execute(f"SELECT * FROM {table};").df()
