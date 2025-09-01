# core/agents/python_agent.py

import pandas as pd
from typing import Dict, Any, Optional

from core.executors.duckdb_exec import DuckRunner
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex
from core.llm.llama_sql_agent import DataFrameManager


class PythonAgent:
    """
    Agent for data wrangling, cleaning, transformation, and feature engineering.
    """

    def __init__(
        self,
        runner: DuckRunner,
        df_manager: DataFrameManager,
        llm: GroqClient,
        faiss_index: FaissIndex,
    ):
        self.runner = runner
        self.df_manager = df_manager
        self.llm = llm
        self.faiss = faiss_index

    def _build_prompt(self, question: str, df: Optional[pd.DataFrame]) -> str:
        schema_text = (
            str(df.dtypes) if df is not None else "No dataframe currently loaded."
        )

        faiss_hits = self.faiss.search(question, k=3)
        faiss_context = "\n".join(
            [f"- {hit['text']} (score={hit['score']:.2f})" for hit in faiss_hits]
        )

        return f"""
You are an AI Data Wrangler that writes **Python Pandas code**.

Rules:
- Always assume the current working dataframe is called `df`.
- Perform cleaning, transformations, or feature engineering as requested.
- Do not invent columns that don't exist. Use schema below.
- Return **only code** (no explanation, no markdown).

# CURRENT SCHEMA (df.dtypes)
{schema_text}

# RELEVANT CONTEXT
{faiss_context}

# USER REQUEST
{question}

# RESPONSE (Python code only):
"""

    def ask(self, question: str) -> Dict[str, Any]:
        df = self.df_manager.get()
        prompt = self._build_prompt(question, df)
        code = self.llm.complete(prompt).strip()

        # Strip markdown fences if present
        if "```" in code:
            code = code.split("```")[1].strip()

        local_vars: Dict[str, Any] = {"pd": pd}
        if df is not None:
            local_vars["df"] = df.copy()

        try:
            exec(code, {}, local_vars)

            # Grab latest DataFrame from locals
            new_df = None
            for v in local_vars.values():
                if isinstance(v, pd.DataFrame):
                    new_df = v

            if new_df is not None:
                # Persist into DuckDB + disk snapshots
                self.df_manager.set(new_df)

                # Store transformation fact in FAISS
                doc_text = (
                    f"Python transformation run for: {question}\n"
                    f"Code:\n{code}\n"
                    f"Resulting shape: {new_df.shape}"
                )
                self.faiss.add([{"text": doc_text}])
                self.faiss.save()

                return {
                    "type": "python",
                    "question": question,
                    "code": code,
                    "rows": len(new_df),
                    "preview": new_df.head(20),
                }

            return {"error": "No DataFrame produced", "code": code}

        except Exception as e:
            return {"error": str(e), "code": code}
 
