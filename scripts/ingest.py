# scripts/ingest.py
import duckdb
import pandas as pd
from pathlib import Path
from typing import Union, List
import uuid

# Use the SAME DB file as DuckRunner
DUCKDB_PATH = Path(__file__).resolve().parent.parent / "data" / "duckdb_store.duckdb"
DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _con():
    return duckdb.connect(str(DUCKDB_PATH))

def _norm_id(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_","-") else "_" for ch in s)

def ingest_file(file_path: Union[str, Path], user_id: str, session_id: str):
    """
    Ingest CSV/Excel -> DuckDB permanent table.
    Returns (df, table_name).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # load with pandas
    suf = file_path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(file_path)
    elif suf in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suf}")

    # user/session-tagged table name
    uid = _norm_id(user_id)
    sid = _norm_id(session_id)
    stem = _norm_id(file_path.stem)
    rand = uuid.uuid4().hex[:6]
    table_name = f"user_{uid}__sess_{sid}__{stem}_{rand}"

    con = _con()
    con.register("temp_df", df)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
    con.unregister("temp_df")
    con.close()

    print(f"✅ Ingested into DuckDB: {table_name}")
    return df, table_name

def list_session_tables(user_id: str, session_id: str) -> List[str]:
    con = _con()
    rows = con.execute("SHOW TABLES").fetchall()
    con.close()
    prefix = f"user_{_norm_id(user_id)}__sess_{_norm_id(session_id)}__"
    return [r[0] for r in rows if r[0].startswith(prefix)]

def list_user_tables(user_id: str) -> List[str]:
    con = _con()
    rows = con.execute("SHOW TABLES").fetchall()
    con.close()
    prefix = f"user_{_norm_id(user_id)}__"
    return [r[0] for r in rows if r[0].startswith(prefix)]

def drop_session_tables(user_id: str, session_id: str):
    con = _con()
    for t in list_session_tables(user_id, session_id):
        con.execute(f"DROP TABLE IF EXISTS {t}")
        print(f"🗑️ Dropped session table: {t}")
    con.close()

def drop_user_tables(user_id: str):
    con = _con()
    for t in list_user_tables(user_id):
        con.execute(f"DROP TABLE IF EXISTS {t}")
        print(f"🗑️ Dropped user table: {t}")
    con.close()

if __name__ == "__main__":
    uid = "user1"
    sid = "sessA"
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample.csv"
    if sample_csv.exists():
        df, t = ingest_file(sample_csv, user_id=uid, session_id=sid)
        print(df.head())
        print("Session tables:", list_session_tables(uid, sid))
    else:
        print("⚠️ No sample.csv found in data/")
