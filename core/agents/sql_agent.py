# core/agents/sql_agent.py

import re
import pandas as pd
from typing import Dict, Any, Optional

from core.executors.duckdb_exec import DuckRunner
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex

SAFE_SQL = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)


class SQLAgent:
    """
    Specialized agent for answering questions via SQL on DuckDB.
    """

    def __init__(self, runner: DuckRunner, llm: GroqClient, faiss_index: FaissIndex):
        self.runner = runner
        self.llm = llm
        self.faiss = faiss_index

    def _build_schema_text(self) -> str:
        parts = []
        for t in self.runner.list_tables():
            parts.append(self.runner.get_schema(t))
        return "\n".join(parts)

    def _build_prompt(self, question: str, schema_text: str) -> str:
        faiss_hits = self.faiss.search(question, k=3)
        faiss_context = "\n".join(
            [f"- {hit['text']} (score={hit['score']:.2f})" for hit in faiss_hits]
        )

        return f"""
You are an AI SQL generator for DuckDB.
Rules:
- Only produce SQL code (SELECT queries).
- Never hallucinate columns or tables not in schema.
- Optimize for clarity.

# SCHEMA
{schema_text}

# RELEVANT CONTEXT
{faiss_context}

# USER QUESTION
{question}

# RESPONSE (SQL only, no explanation):
"""

    def ask(self, question: str) -> Dict[str, Any]:
        schema_text = self._build_schema_text()
        prompt = self._build_prompt(question, schema_text)
        sql = self.llm.complete(prompt).strip()

        # Extract SQL block if wrapped
        if "```" in sql:
            sql = sql.split("```")[1].strip()

        if not SAFE_SQL.match(sql.lower()):
            return {"error": "Unsafe or non-SELECT SQL generated", "raw": sql}

        # Execute
        try:
            if " limit " not in sql.lower():
                sql = sql.rstrip(";") + " LIMIT 200;"
            df = self.runner.query(sql)

            # Store in FAISS
            try:
                doc_text = (
                    f"SQL run for: {question}\nQuery: {sql}\nPreview:\n{df.head(5).to_string(index=False)}"
                )
                self.faiss.add([{"text": doc_text}])
                self.faiss.save()
            except Exception as e:
                print(f"[WARN] Could not save SQL result to FAISS: {e}")

            return {
                "type": "sql",
                "question": question,
                "sql": sql,
                "rows": len(df),
                "preview": df.head(20),
            }

        except Exception as e:
            return {"error": str(e), "sql": sql}
 
