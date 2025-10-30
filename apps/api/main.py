from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from core.executors.duckdb_exec import DuckRunner
from core.llm.llm_sql_agent import run_qa
from scripts.ingest import list_session_tables, ingest_file
from pathlib import Path
import os
import pandas as pd

app = FastAPI()

# Allow local web app to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    executed_code: str
    download_url: str
    code_url: str

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    tables = list_session_tables(req.user_id, req.session_id)
    if not tables:
        return {"sql":"", "rows":0, "preview":[], "summary":"No tables for this session.", "suggested_next":["Upload a dataset."]}
    out = run_qa(req.message, runner, tables)

    # Persist executed code for this session (latest step)
    code = out.get("executed_code", "")
    # Prevent StatReload from triggering on code file writes (optional: set env var to skip)
    if not os.environ.get("SKIP_CODE_WRITE"):
        code_dir = Path("data/code")
        code_dir.mkdir(parents=True, exist_ok=True)
        latest_code_path = code_dir / f"latest_{req.user_id}_{req.session_id}.py"
        try:
            if code:
                latest_code_path.write_text(code, encoding="utf-8")
        except Exception:
            pass

    code_url = f"/code?user_id={req.user_id}&session_id={req.session_id}"

    # Sanitize preview (replace NaN/Inf with None for JSON compliance)
    preview_df = out["preview"]
    if isinstance(preview_df, pd.DataFrame):
        safe_preview = (
            preview_df.replace([float("inf"), float("-inf")], None)
            .where(pd.notna(preview_df), None)
            .to_dict(orient="records")
        )
    else:
        safe_preview = []

    return {
        "sql": out["sql"],
        "rows": out["rows"],
        "preview": safe_preview,
        "summary": out["summary"],
        "suggested_next": out["suggested_next"],
        "executed_code": out.get("executed_code", ""),
        "download_url": out.get("download_url", "/download"),
        "code_url": code_url,
    }

@app.post("/ingest")
async def ingest(
    user_id: str = Form(...),
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    # Validate extension
    filename = file.filename or "uploaded_file"
    suf = Path(filename).suffix.lower()
    if suf not in (".csv", ".xls", ".xlsx"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suf}")

    # Save to temp path under data/uploads
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    temp_path = uploads_dir / filename

    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        # Ingest into DuckDB
        df, table_name = ingest_file(temp_path, user_id=user_id, session_id=session_id)

        # Sanitize sample for JSON (replace NaN/Inf with None)
        sample_df = df.head(10).replace([float("inf"), float("-inf")], None).where(pd.notna(df), None)

        return {
            "table": table_name,
            "rows": int(len(df)),
            "columns": list(map(str, df.columns.tolist())),
            "sample": sample_df.to_dict(orient="records"),
        }
    finally:
        # Clean up temp file
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass

@app.get("/download")
def download_latest():
    # Serve the latest CSV snapshot saved by DataFrameManager
    csv_path = Path("data/frames/current_df.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No snapshot available. Perform an operation first.")
    return FileResponse(path=str(csv_path), media_type="text/csv", filename="current_df.csv")

@app.get("/code")
def get_latest_code(user_id: str, session_id: str):
    code_path = Path("data/code") / f"latest_{user_id}_{session_id}.py"
    if not code_path.exists():
        raise HTTPException(status_code=404, detail="No code available for this session. Perform an operation first.")
    try:
        content = code_path.read_text(encoding="utf-8")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read code file.")
    return {"user_id": user_id, "session_id": session_id, "code": content}
