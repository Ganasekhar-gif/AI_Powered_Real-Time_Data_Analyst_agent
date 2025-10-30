# core/agents/python_agent.py

import pandas as pd
from typing import Dict, Any, Optional

from core.executors.duckdb_exec import DuckRunner
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex
from core.utils.dataframe_manager import DataFrameManager
from core.rag.finance_kb import FinanceKnowledgeBase


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
        finance_kb: Optional[FinanceKnowledgeBase] = None,
    ):
        self.runner = runner
        self.df_manager = df_manager
        self.llm = llm
        self.faiss = faiss_index
        self.finance_kb = finance_kb

    def _build_prompt(self, question: str, df: Optional[pd.DataFrame]) -> str:
        schema_text = (
            str(df.dtypes) if df is not None else "No dataframe currently loaded."
        )

        # Keep FAISS context minimal to stay within token limits
        faiss_hits = self.faiss.search(question, k=1)
        faiss_context = "\n".join(
            [f"- {hit['text']} (score={hit['score']:.2f})" for hit in faiss_hits]
        )

        # Keep prompt minimal for code generation; omit finance_context here
        prompt = f"""
You are an AI Financial Data Wrangler that writes **Python Pandas code**.

Rules:
- Always assume the current working dataframe is called `df`.
- Perform financial data cleaning, transformations, or feature engineering as requested.
- Do not invent columns that don't exist. Use schema below.
- Apply financial best practices for data handling.
- Return **only code** (no explanation, no markdown).

# CURRENT SCHEMA (df.dtypes)
{schema_text}

# FINANCIAL KNOWLEDGE
<omitted>

# RELEVANT CONTEXT
{faiss_context}

# USER REQUEST
{question}

# RESPONSE (Python code only):
"""

        # Hard-cap prompt length to avoid model TPM/token limits
        if len(prompt) > 6000:
            prompt = prompt[:6000]
        return prompt

    def ask(self, question: str) -> Dict[str, Any]:
        df = self.df_manager.get()
        prompt = self._build_prompt(question, df)
        code = self.llm.complete(prompt).strip()

        # Strip markdown fences robustly and remove language tag like 'python'
        if "```" in code:
            first = code.find("```")
            last = code.rfind("```")
            if last > first:
                inner = code[first + 3:last].strip()
            else:
                inner = code[first + 3:].strip()
            # Drop a leading language identifier line if present
            if inner.lower().startswith("python\n"):
                inner = inner.split("\n", 1)[1]
            elif inner.lower().startswith("py\n"):
                inner = inner.split("\n", 1)[1]
            code = inner.strip()

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
 
