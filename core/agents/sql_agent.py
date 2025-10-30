# core/agents/sql_agent.py

import re
import pandas as pd
from typing import Dict, Any, Optional

from core.executors.duckdb_exec import DuckRunner
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex
from core.rag.finance_kb import FinanceKnowledgeBase

SAFE_SQL = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)


class SQLAgent:
    """
    Specialized agent for answering questions via SQL on DuckDB.
    """

    def __init__(self, runner: DuckRunner, llm: GroqClient, faiss_index: FaissIndex, finance_kb: Optional[FinanceKnowledgeBase] = None):
        self.runner = runner
        self.llm = llm
        self.faiss = faiss_index
        self.finance_kb = finance_kb

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

        # Get financial context if available
        finance_context = ""
        if self.finance_kb:
            finance_context = self.finance_kb.get_financial_context(question, k=3)

        return f"""
You are an AI Financial SQL Expert for DuckDB.
Rules:
- Only produce SQL code (SELECT queries).
- Never hallucinate columns or tables not in schema.
- Optimize for clarity and financial best practices.
- Use appropriate financial calculations and ratios.
- Use EXACT column names as shown in the schema (copy/paste including spaces or punctuation), and if a column name contains spaces or punctuation, wrap it in double quotes, e.g., "Estimated Sales, Thousands of Gallons".
- The active analysis table is named current_df. Always query from current_df.

# SCHEMA
{schema_text}

# FINANCIAL KNOWLEDGE
{finance_context}

# RELEVANT CONTEXT
{faiss_context}

# USER QUESTION
{question}

# RESPONSE (SQL only, no explanation):
"""

    def _quote_sql_identifiers(self, sql: str) -> str:
        """Auto-quote column identifiers that contain spaces/punctuation using DuckDB schema.
        This helps when the LLM forgets to add quotes around complex column names.
        """
        try:
            all_tables = self.runner.list_tables()
            cols = set()
            for t in all_tables:
                df = self.runner.query(f"PRAGMA table_info('{t}');")
                for _, r in df.iterrows():
                    name = str(r.get("name", ""))
                    # Identify names that likely need quoting
                    if any(ch for ch in name if not (ch.isalnum() or ch == "_")):
                        cols.add(name)
            # Replace longer names first to avoid partial overlaps
            for name in sorted(cols, key=len, reverse=True):
                quoted = f'"{name}"'
                if quoted in sql:
                    continue
                if name in sql:
                    sql = sql.replace(name, quoted)
            return sql
        except Exception:
            return sql

    def ask(self, question: str) -> Dict[str, Any]:
        schema_text = self._build_schema_text()
        prompt = self._build_prompt(question, schema_text)
        sql = self.llm.complete(prompt).strip()

        # Extract SQL block if wrapped and strip optional language tag
        if "```" in sql:
            first = sql.find("```")
            last = sql.rfind("```")
            if last > first:
                inner = sql[first + 3:last].strip()
            else:
                inner = sql[first + 3:].strip()
            # Drop leading language identifier line if present
            first_line, _, rest = inner.partition("\n")
            if first_line.strip().lower() in ("sql", "duckdb"):
                sql = rest.strip()
            else:
                sql = inner.strip()

        # Auto-quote identifiers with spaces/punctuation using schema
        sql = self._quote_sql_identifiers(sql)

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
 
