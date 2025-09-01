# core/llm/llama_sql_agent.py

import os
import re
from typing import Dict, Any, List, Optional
import pandas as pd

from core.executors.duckdb_exec import DuckRunner
from core.llm.groq_client import GroqClient
from langgraph.graph import StateGraph, END
from core.rag.index import FaissIndex
from core.rag.embed import EmbeddingModel

# === Multi-agent imports (to be implemented in core/agents/*) ===
from core.agents.planner import PlannerAgent
from core.agents.sql import SQLAgent
from core.agents.cleaner import CleanerAgent
from core.agents.viz import VizAgent
from core.agents.explainer import ExplainerAgent

# -------------------
# Globals
# -------------------

SAFE_SQL = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)

SYSTEM_PROMPT = """You are an AI Data Analyst.
You can perform:
1. SQL queries on DuckDB
2. Data cleaning / preprocessing (Pandas)
3. Data visualization (Matplotlib/Plotly)
4. Feature engineering suggestions

Rules:
- If user asks about data in tables → generate SQL.
- If user asks for cleaning/processing/visualizations → return Python code.
- If user asks for advice → return text explanation.
- Always use the latest dataframe ("current_df") for analysis.
"""

# -------------------
# DataFrame Manager
# -------------------

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

# -------------------
# Helpers
# -------------------

def _build_schema_text(runner: DuckRunner, tables: List[str]) -> str:
    parts = []
    for t in tables:
        parts.append(runner.get_schema(t))
    return "\n".join(parts)

# -------------------
# LangGraph Workflow (Planner → Executors → Explainer)
# -------------------

def build_agent_graph(
    runner: DuckRunner,
    df_manager: DataFrameManager,
    tables: List[str],
    llm: GroqClient,
    faiss_index: FaissIndex,
):
    """
    Orchestrates:
        planner -> (sql_exec | clean_exec | viz_exec | text_response) -> explainer -> END
    """

    schema_text = _build_schema_text(runner, tables)

    # Instantiate agents
    planner = PlannerAgent(
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        faiss=faiss_index,
        schema_text=schema_text,
    )
    sql_agent = SQLAgent(runner=runner, faiss=faiss_index)
    cleaner_agent = CleanerAgent(df_manager=df_manager, faiss=faiss_index)
    viz_agent = VizAgent(df_manager=df_manager, faiss=faiss_index, artifacts_dir="artifacts")
    explainer_agent = ExplainerAgent(llm=llm, faiss=faiss_index)

    def planner_node(state: Dict[str, Any]) -> str:
        """
        Uses LLM + FAISS + schema to decide the route and produce raw content.
        Sets:
            state['raw_response']: str (SQL text, Python code, or prose)
            state['route']: one of {'sql_exec','clean_exec','viz_exec','text_response'}
        Returns the next node name (LangGraph dynamic routing).
        """
        question = state["question"]
        plan = planner.plan(question=question)  # dict: {'route': ..., 'raw_response': ..., 'context': ...}
        state.update(plan)

        # Dynamic routing
        return plan["route"]

    def sql_exec(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes SQL safely, returns preview, and stores memory in FAISS (inside SQLAgent).
        Expected state entries:
            - raw_response: SQL text
        Sets:
            - type, sql, rows, preview
        """
        sql_text = state["raw_response"]
        result = sql_agent.execute(question=state["question"], sql_text=sql_text)
        state.update(result)
        return state

    def clean_exec(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes Pandas cleaning code, updates DuckDB via DataFrameManager,
        and stores transformation facts in FAISS (inside CleanerAgent).
        Expected state:
            - raw_response: Python code
        Sets:
            - type, code, dataframe (preview)
        """
        code = state["raw_response"]
        result = cleaner_agent.execute(code=code)
        state.update(result)
        return state

    def viz_exec(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes viz code (Matplotlib/Plotly), saves artifact, stores step in FAISS.
        Expected state:
            - raw_response: Python viz code
        Sets:
            - type, code, artifacts: List[str] (paths), notes (optional)
        """
        code = state["raw_response"]
        result = viz_agent.execute(code=code)
        state.update(result)
        return state

    def text_response(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        For advice/explanations when neither SQL nor Python is required.
        The planner already put the prose into state['raw_response'].
        """
        state.update({
            "type": "text",
            "answer": state["raw_response"]
        })
        return state

    def explainer(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarizes what happened and suggests next steps; stores summary into FAISS.
        Uses ExplainerAgent.
        """
        summary = explainer_agent.summarize(
            question=state["question"],
            raw_response=state.get("raw_response", ""),
            exec_type=state.get("type", "text"),
            extras={
                "sql": state.get("sql"),
                "rows": state.get("rows"),
                "artifacts": state.get("artifacts"),
            }
        )
        state["summary"] = summary
        return state

    # Build LangGraph
    workflow = StateGraph(dict)

    # Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("sql_exec", sql_exec)
    workflow.add_node("clean_exec", clean_exec)
    workflow.add_node("viz_exec", viz_exec)
    workflow.add_node("text_response", text_response)
    workflow.add_node("explainer", explainer)

    # Entry
    workflow.set_entry_point("planner")

    # Dynamic routing from planner → one of the executors
    # (planner_node returns the next node name directly)
    # After any executor, go to explainer → END
    workflow.add_edge("sql_exec", "explainer")
    workflow.add_edge("clean_exec", "explainer")
    workflow.add_edge("viz_exec", "explainer")
    workflow.add_edge("text_response", "explainer")
    workflow.add_edge("explainer", END)

    return workflow.compile()

# -------------------
# Runner
# -------------------

def run_agent(
    question: str,
    runner: DuckRunner,
    df_manager: DataFrameManager,
    tables: List[str],
    llm: Optional[GroqClient] = None,
    faiss_index: Optional[FaissIndex] = None,
):
    llm = llm or GroqClient.from_env()
    faiss_index = faiss_index or FaissIndex(
        "data/faiss/index.bin",
        "data/faiss/meta.json",
        EmbeddingModel()
    )

    graph = build_agent_graph(runner, df_manager, tables, llm, faiss_index)
    state = {"question": question}
    return graph.invoke(state)
