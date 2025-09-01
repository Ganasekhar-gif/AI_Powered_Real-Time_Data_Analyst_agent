from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from core.executors.duckdb_exec import DuckRunner
from core.agents.llm_sql_agent import run_qa
from scripts.ingest import list_session_tables

app = FastAPI()

runner = DuckRunner(db_path="data/duckdb_store.duckdb")

class ChatReq(BaseModel):
    user_id: str
    session_id: str
    message: str

class ChatResp(BaseModel):
    sql: str
    rows: int
    preview: List[dict]
    summary: str
    suggested_next: List[str]

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    tables = list_session_tables(req.user_id, req.session_id)
    if not tables:
        return {"sql":"", "rows":0, "preview":[], "summary":"No tables for this session.", "suggested_next":["Upload a dataset."]}
    out = run_qa(req.message, runner, tables)
    return {
        "sql": out["sql"],
        "rows": out["rows"],
        "preview": out["preview"].to_dict(orient="records"),
        "summary": out["summary"],
        "suggested_next": out["suggested_next"],
    }
