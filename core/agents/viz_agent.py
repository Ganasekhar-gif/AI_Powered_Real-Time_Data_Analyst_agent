# core/agents/viz_agent.py

import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Dict, Any, Optional

from core.executors.duckdb_exec import DuckRunner
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex
from core.llm.llama_sql_agent import DataFrameManager


class VisualizationAgent:
    """
    Agent for data visualization (Matplotlib / Plotly).
    Generates charts + Python code based on user queries.
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
You are an AI Data Visualization expert.

Rules:
- Assume the working DataFrame is named `df`.
- Use Matplotlib (`plt`) or Plotly (`px`) for visualization.
- Return **only Python code** that generates the chart (no explanations, no markdown).
- Do not invent columns. Use only schema below.

# CURRENT SCHEMA
{schema_text}

# RELEVANT CONTEXT
{faiss_context}

# USER REQUEST
{question}

# RESPONSE (Python code only):
"""

    def _render_matplotlib(self) -> str:
        """Capture latest Matplotlib plot as base64 string."""
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return encoded

    def _render_plotly(self, fig) -> str:
        """Capture Plotly figure as base64 PNG."""
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def ask(self, question: str) -> Dict[str, Any]:
        df = self.df_manager.get()
        prompt = self._build_prompt(question, df)
        code = self.llm.complete(prompt).strip()

        # Strip markdown fences if present
        if "```" in code:
            code = code.split("```")[1].strip()

        local_vars: Dict[str, Any] = {"pd": pd, "plt": plt, "px": px}
        if df is not None:
            local_vars["df"] = df.copy()

        chart_b64 = None

        try:
            exec(code, {}, local_vars)

            # Detect if Plotly fig was created
            if "fig" in local_vars and hasattr(local_vars["fig"], "to_dict"):
                chart_b64 = self._render_plotly(local_vars["fig"])
            else:
                chart_b64 = self._render_matplotlib()

            # Store visualization info in FAISS
            doc_text = (
                f"Visualization created for: {question}\n"
                f"Code:\n{code}\n"
            )
            self.faiss.add([{"text": doc_text}])
            self.faiss.save()

            return {
                "type": "visualization",
                "question": question,
                "code": code,
                "chart_base64": chart_b64,
            }

        except Exception as e:
            return {"error": str(e), "code": code}
 
