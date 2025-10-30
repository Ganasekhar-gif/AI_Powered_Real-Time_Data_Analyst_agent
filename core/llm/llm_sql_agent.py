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

# Import existing agents
from core.agents.sql_agent import SQLAgent
from core.agents.python_agent import PythonAgent
from core.agents.viz_agent import VisualizationAgent
from core.agents.explainer_agent import ExplainerAgent
from core.agents.critic_agent import CriticAgent
from core.agents.planner_agent import PlannerAgent
from core.agents.data_examiner_agent import DataExaminerAgent
from core.rag.finance_kb import FinanceKnowledgeBase
from core.utils.dataframe_manager import DataFrameManager

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

# (Using existing PlannerAgent from core/agents/planner_agent.py)

# -------------------
# DataFrame Manager (imported to avoid circular imports)
# -------------------

# -------------------
# Helpers
# -------------------

def _build_schema_text(runner: DuckRunner, tables: List[str]) -> str:
    parts = []
    for t in tables:
        parts.append(runner.get_schema(t))
    return "\n".join(parts)

def _clean_text(text: str) -> str:
    """Collapse newlines and markdown artifacts to make API output single-line friendly."""
    if not isinstance(text, str):
        return str(text)
    # Remove simple markdown emphasis and headings
    for ch in ["**", "__", "##", "###", "####", "*", "`", "\r"]:
        text = text.replace(ch, " ")
    # Collapse newlines and multiple spaces
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()

def _shorten(text: str, max_len: int = 300) -> str:
    if not isinstance(text, str):
        return str(text)
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"

def _cap(text: str, max_chars: int = 800) -> str:
    if not isinstance(text, str):
        return str(text)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"

def _format_number(n: Any) -> str:
    """Format numbers with comma separators and 2 decimal places if float"""
    if isinstance(n, (int, float)):
        if n == int(n):
            return f"{int(n):,}"
        return f"{n:,.2f}"
    return str(n)

def _fallback_summary(result_type: str, sql: str, rows: int, preview: pd.DataFrame, question: str) -> str:
    """Generate a clear, conversational summary of the analysis results."""
    def _analyze_sql(sql_text: str) -> Dict[str, Any]:
        s = (sql_text or "").lower()
        op = "query"
        if " sum(" in s or s.strip().startswith("sum(") or " sum (" in s:
            op = "sum"
        elif " avg(" in s:
            op = "average"
        elif " count(" in s:
            op = "count"
        elif " group by " in s:
            op = "grouped query"
        elif " where " in s:
            op = "filtered query"
        
        # Extract the column being operated on (for sum/avg/count)
        target_col = ""
        if op in ["sum", "average", "count"]:
            try:
                target_col = s.split("(")[1].split(")")[0].strip()
            except:
                pass
        
        # Extract GROUP BY column if present
        group_col = ""
        if "group by" in s:
            try:
                group_col = s.split("group by")[1].split(",")[0].strip().split()[0].strip()
            except:
                pass
        
        return {
            "operation": op,
            "target_column": target_col,
            "group_column": group_col,
            "has_where": " where " in s,
            "raw_sql": sql_text
        }

    def _extract_top_preview(df: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {"group_col": None, "metric_col": None, "pairs": []}
            
        # Identify metric (numeric) and group (non-numeric) columns
        metric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        group_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        
        metric_col = metric_cols[0] if metric_cols else None
        group_col = group_cols[0] if group_cols else (df.columns[0] if len(df.columns) else None)
        
        pairs = []
        try:
            if metric_col:
                df_sorted = df.sort_values(by=metric_col, ascending=False, kind="mergesort")
                head = df_sorted.head(3)
                for _, row in head.iterrows():
                    group_val = str(row.get(group_col, ""))
                    metric_val = row.get(metric_col)
                    if group_val and metric_val is not None:
                        pairs.append((group_val, metric_val))
        except Exception:
            pass
            
        return {
            "group_col": group_col,
            "metric_col": metric_col,
            "pairs": pairs,
            "total_rows": len(df)
        }

    # Start building the summary
    summary_parts = []
    
    if (result_type == "sql") or sql:
        info = _analyze_sql(sql or "")
        op = info["operation"]
        # Data quality/business logic check: natural, conversational, structured
        if any(w in (question or '').lower() for w in ["quality", "validation", "completeness", "business logic"]):
            lines = [
                "Here's a quick review of your dataset's health:",
                "",
            ]
            # Example: parse preview for issues if possible
            if preview is not None and hasattr(preview, 'to_dict'):
                d = preview.to_dict(orient="records")
                # Try to find keys with 'missing', 'duplicate', 'type', etc.
                for row in d:
                    for k, v in row.items():
                        if "missing" in k.lower():
                            lines.append(f"• Missing values: {v}")
                        if "duplicate" in k.lower():
                            lines.append(f"• Duplicates: {v}")
                        if "type" in k.lower():
                            lines.append(f"• Data types: {v}")
                if not lines[-1].startswith("•"):
                    lines.append("• No major issues detected.")
            else:
                lines.append("• No major issues detected.")
            lines.append("")
            lines.append("What you can do next:")
            lines.append("- No action needed if all checks passed.")
            lines.append("- If issues are listed above, consider cleaning or transforming those columns.")
            return "\n".join(lines)
        # Default SQL summary
        if op == "sum" and info["group_column"]:
            details = _extract_top_preview(preview)
            pairs = details.get("pairs", [])
            group_col = details["group_col"] or "category"
            metric_col = details["metric_col"] or "value"
            summary = [
                f"Based on your data, I explored how {info['target_column']} varies across different {group_col}s.",
            ]
            if pairs:
                summary.append(f"Here's what stands out:")
                for i, (group_val, metric_val) in enumerate(pairs, 1):
                    summary.append(f"- {group_val}: {metric_col} totals { _format_number(metric_val) }")
                total_groups = details.get("total_rows", 0)
                if total_groups > len(pairs):
                    summary.append(f"...plus {total_groups - len(pairs)} more {group_col}s with smaller totals.")
            summary.append(f"This suggests that certain {group_col}s contribute much more to the total {metric_col} than others.")
            summary.append("In short, you may want to focus on the top contributors for deeper analysis or optimization.")
            return " ".join(summary)
        elif op == "average" and info["group_column"]:
            details = _extract_top_preview(preview)
            pairs = details.get("pairs", [])
            group_col = details["group_col"] or "category"
            metric_col = details["metric_col"] or "value"
            summary = [
                f"Looking at your dataset, I compared the average {info['target_column']} among different {group_col}s.",
            ]
            if pairs:
                summary.append(f"The top averages are:")
                for i, (group_val, metric_val) in enumerate(pairs, 1):
                    summary.append(f"- {group_val}: average {metric_col} is { _format_number(metric_val) }")
                total_groups = details.get("total_rows", 0)
                if total_groups > len(pairs):
                    summary.append(f"...and {total_groups - len(pairs)} more {group_col}s with lower averages.")
            summary.append(f"This helps highlight which {group_col}s typically have higher values.")
            summary.append("In short, focusing on the groups with the highest averages could reveal important trends.")
            return " ".join(summary)
        elif op == "count" and info["group_column"]:
            details = _extract_top_preview(preview)
            pairs = details.get("pairs", [])
            group_col = details["group_col"] or "category"
            summary = [
                f"I reviewed how many records belong to each {group_col} in your dataset.",
            ]
            if pairs:
                summary.append("The largest groups are:")
                for i, (group_val, metric_val) in enumerate(pairs, 1):
                    summary.append(f"- {group_val}: { _format_number(metric_val) } records")
                total_groups = details.get("total_rows", 0)
                if total_groups > len(pairs):
                    summary.append(f"...plus {total_groups - len(pairs)} more {group_col}s.")
            summary.append(f"This breakdown gives you a sense of which {group_col}s are most common in your data.")
            summary.append("If needed, you can dig deeper into the largest or smallest groups for further insights.")
            return " ".join(summary)
        else:
            summary_parts.append("I explored your dataset as requested and summarized the key findings in a way that's easy to understand.")
        if rows == 0:
            summary_parts.append("No results were returned for this query. Double-check your filters or try broadening your search.")
        else:
            details = _extract_top_preview(preview)
            pairs = details.get("pairs", [])
            if pairs:
                group_col = details["group_col"] or "category"
                metric_col = details["metric_col"] or "value"
                summary_parts.append("Here's a quick look at the most notable results:")
                for i, (group_val, metric_val) in enumerate(pairs, 1):
                    summary_parts.append(f"- {group_col}: {group_val}, {metric_col}: {_format_number(metric_val)}")
                total_groups = details.get("total_rows", 0)
                if total_groups > len(pairs):
                    summary_parts.append(f"...and {total_groups - len(pairs)} more {group_col}s.")
            summary_parts.append("In short, these results can help you spot patterns or outliers worth a closer look.")
    else:
        summary_parts.append("I processed your request.")
        if rows is not None and rows > 0:
            summary_parts.append(f"Returned {rows} rows of results.")
    if not sql or "insert" not in sql.lower() and "update" not in sql.lower() and "delete" not in sql.lower():
        summary_parts.append("Your original dataset wasn't changed.")
    return "\n".join(summary_parts)


def _friendly_error(err: str) -> Dict[str, Any]:
    e = (err or "").strip()
    el = e.lower()
    summary = "An issue occurred while processing your request."
    suggestions: List[str] = []

    if any(k in el for k in ["column", "does not exist", "unknown column", "keyerror"]):
        summary = "A referenced column was not found in the dataset."
        suggestions = [
            "Check the exact column name and spelling (case-sensitive)",
            "List available columns or preview a few rows of current_df",
            "Adjust the query to use existing column names"
        ]
    elif "syntax error" in el or "parse" in el:
        summary = "There is a syntax issue in the generated query/code."
        suggestions = [
            "Rephrase the request more simply",
            "Avoid special characters in column names or quote them",
            "Ask for a smaller step to validate syntax"
        ]
    elif "kaleido" in el and "plotly" in el:
        summary = "Plotly image export failed due to missing Kaleido backend."
        suggestions = [
            "Install Kaleido (pip install kaleido) or use Matplotlib",
            "Ask for a Matplotlib chart instead of Plotly"
        ]
    elif "matplotlib" in el and "gui" in el:
        summary = "Matplotlib attempted to open a GUI window; switched to headless mode."
        suggestions = [
            "Proceed; the chart will be rendered to an image in the response",
            "If issue persists, request a static Matplotlib export"
        ]
    elif "nameerror" in el:
        summary = "Generated code referenced an undefined variable."
        suggestions = [
            "Ensure variable names match the context (use 'df' and available imports)",
            "Retry the request; the agent now sanitizes code fences and language tags"
        ]
    else:
        summary = e or summary
        suggestions = [
            "Try rephrasing the request",
            "Run a smaller step (e.g., preview columns) before the full operation",
            "Verify the dataset is ingested and session is active"
        ]
    return {"summary": summary, "suggested": suggestions}

def _suggestions_from_context(question: str, sql: str, preview: pd.DataFrame, rows: int) -> List[str]:
    ql = (question or "").lower()
    s = (sql or "").lower()
    sug: List[str] = []
    has_group = " group by " in s
    has_where = " where " in s
    
    # Custom suggestions based on query intent
    if rows == 0:
        if has_where:
            sug.append("Try relaxing or removing one of the filters to see if more data appears.")
        sug.append("Double-check your column names and filter values for typos or mismatches.")
        sug.append("Preview a few rows from your dataset to confirm data presence and format.")
        return sug[:3]

    if "trend" in ql or "over time" in ql or "monthly" in ql or "per month" in ql:
        sug.append("Plot a time series to visualize trends across months or years.")
        sug.append("Compare categories over time to spot seasonality or changes.")
    if "top" in ql or "highest" in ql or has_group:
        sug.append("Visualize the top groups or categories with a bar or pie chart.")
        sug.append("Drill down into the leading group to explore underlying details.")
    if "average" in ql or "mean" in ql:
        sug.append("Compare the averages between groups to spot outliers or unusual patterns.")
    if "sum" in ql or "total" in ql:
        sug.append("Look at the contribution of each group to the overall total.")
    if "count" in ql or "frequency" in ql:
        sug.append("See which groups appear most often and why.")
    if "outlier" in ql or "anomaly" in ql:
        sug.append("Investigate outliers further—are they errors or important signals?")
    if "distribution" in ql or "histogram" in ql:
        sug.append("Plot a histogram to visualize the distribution of values.")
    # Fallback: suggest next steps based on what was just done
    if not sug:
        if has_group:
            sug.append("Try filtering to a specific group or comparing two groups in detail.")
        if preview is not None and hasattr(preview, 'columns'):
            cols = list(preview.columns)
            if any('date' in c.lower() for c in cols):
                sug.append("Analyze trends over time using the Date column.")
            if any(pd.api.types.is_numeric_dtype(preview[c]) for c in cols):
                sug.append("Request summary statistics or visualize numeric columns.")
        sug.append("Ask a follow-up question to dig deeper into these results.")
    return sug[:3]


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
        planner -> (sql_exec | python_exec | viz_exec | text_response) -> critic -> explainer -> END
    """

    schema_text = _build_schema_text(runner, tables)

    # Initialize finance knowledge base
    finance_kb = FinanceKnowledgeBase(faiss_index)
    
    # Instantiate agents with finance knowledge
    sql_agent = SQLAgent(runner=runner, llm=llm, faiss_index=faiss_index, finance_kb=finance_kb)
    python_agent = PythonAgent(runner=runner, df_manager=df_manager, llm=llm, faiss_index=faiss_index, finance_kb=finance_kb)
    viz_agent = VisualizationAgent(runner=runner, df_manager=df_manager, llm=llm, faiss_index=faiss_index, finance_kb=finance_kb)
    critic_agent = CriticAgent(llm=llm, faiss_index=faiss_index)
    explainer_agent = ExplainerAgent(llm=llm, faiss_index=faiss_index)
    examiner_agent = DataExaminerAgent(llm=llm, faiss_index=faiss_index, finance_kb=finance_kb)
    
    planner = PlannerAgent(
        sql_agent=sql_agent,
        python_agent=python_agent,
        viz_agent=viz_agent,
        explainer_agent=explainer_agent,
        critic_agent=critic_agent,
        llm=llm,
        faiss_index=faiss_index,
    )

    def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses LLM + FAISS + schema to decide the route and produce raw content.
        Sets:
            state['raw_response']: str (SQL text, Python code, or prose)
            state['route']: one of {'examine_data','sql_exec','python_exec','viz_exec','text_response'}
        Returns the updated state (routing handled via add_conditional_edges).
        """
        question = state["question"]
        
        ql = question.lower()
        # Route schema/description requests to examination (not raw SQL)
        if any(w in ql for w in ["describe", "schema", "columns", "structure", "data dictionary", "profile", "profiling", "summarize dataset", "summary of dataset", "describe dataset"]):
            state["route"] = "examine_data"
            return state
        # Route data quality checks and explicit transformation / feature-engineering intents to python_exec
        if any(w in ql for w in [
            "missing", "null", "na", "nan", "duplicated", "duplicate", "duplicates",
            "handle outlier", "handle outliers", "remove outlier", "remove outliers",
            "clean outlier", "clean outliers", "normalize", "standardize", "impute",
            "fix", "drop", "fill", "transform", "feature engineering", "feature-engineering",
            "create column", "new column", "derive", "engineer feature", "add column"
        ]):
            state["route"] = "python_exec"
            return state
        # Route visualization requests explicitly to viz_exec
        if any(w in ql for w in [
            "chart", "plot", "visualize", "visualization", "bar chart", "line chart",
            "hist", "histogram", "scatter", "box plot", "heatmap"
        ]):
            state["route"] = "viz_exec"
            return state
        # Otherwise, examination-style intents
        if any(word in ql for word in ["examine", "check", "inspect", "quality", "issues", "problems"]):
            state["route"] = "examine_data"
            return state
        
        # Use PlannerAgent classification to choose route
        route = planner._classify(question)
        next_node = {
            "sql": "sql_exec",
            "python": "python_exec",
            "viz": "viz_exec",
        }.get(route, "sql_exec")
        state["route"] = next_node
        return state

    def examine_data(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Examine the dataset for quality issues and provide financial recommendations.
        Expected state entries:
            - question: str
        Sets:
            - type, analysis, recommendations, issues_found, next_steps
        """
        question = state["question"]
        df = df_manager.get()
        # If no current df is set, try loading the most recent session table
        if df is None:
            try:
                if tables:
                    # Use the last ingested table for this session
                    last_table = tables[-1]
                    loaded_df = runner.load_dataframe(last_table)
                    df_manager.set(loaded_df)
                    df = loaded_df
            except Exception as e:
                state.update({
                    "type": "examination",
                    "error": f"Failed to load session table: {e}",
                    "recommendations": ["Verify your session tables and try ingesting again."],
                })
                return state
        
        if df is None:
            state.update({
                "type": "examination",
                "error": "No dataset loaded. Please upload a dataset first.",
                "recommendations": ["Upload a financial dataset to examine"]
            })
            return state
        
        # Determine dataset type based on column names
        dataset_type = "financial"  # Default
        if any(col.lower() in ['revenue', 'sales', 'income'] for col in df.columns):
            dataset_type = "income_statement"
        elif any(col.lower() in ['assets', 'liabilities', 'equity'] for col in df.columns):
            dataset_type = "balance_sheet"
        elif any(col.lower() in ['cash_flow', 'operating_cash'] for col in df.columns):
            dataset_type = "cash_flow"
        
        result = examiner_agent.examine_dataset(df, dataset_type)
        state.update(result)
        return state

    def sql_exec(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes SQL safely, returns preview, and stores memory in FAISS (inside SQLAgent).
        Expected state entries:
            - question: str
        Sets:
            - type, sql, rows, preview
        """
        question = state["question"]
        result = sql_agent.ask(question)
        state.update(result)
        return state

    def python_exec(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes Pandas cleaning code, updates DuckDB via DataFrameManager,
        and stores transformation facts in FAISS (inside PythonAgent).
        Expected state:
            - question: str
        Sets:
            - type, code, rows, preview
        """
        question = state["question"]
        # Deterministic path for simple data-quality checks to avoid LLM errors
        ql = question.lower()
        df = df_manager.get()
        if df is None and tables:
            try:
                df = runner.load_dataframe(tables[-1])
                df_manager.set(df, name="current_df")
            except Exception:
                df = None
        if df is not None and (any(w in ql for w in ["missing", "null", "na", "nan"]) or any(w in ql for w in ["duplicate", "duplicated", "duplicates"])):
            try:
                import pandas as pd  # local import
                missing_df = df.isnull().sum().reset_index()
                missing_df.columns = ["column", "missing"]
                dup_count = int(df.duplicated().sum())
                preview_df = missing_df.sort_values(by="missing", ascending=False)
                summary = f"Checked data quality: missing values across {len(df.columns)} columns and duplicates in rows. Duplicates={dup_count}."
                state.update({
                    "type": "python",
                    "question": question,
                    "code": "missing_counts = df.isnull().sum(); duplicate_count = df.duplicated().sum()",
                    "rows": 0,
                    "preview": preview_df.head(50),
                    "dq": {"duplicate_count": dup_count},
                    "analysis": summary,
                })
                return state
            except Exception as e:
                state.update({"type": "python", "error": str(e)})
                return state

        # Fallback: use PythonAgent/LLM for arbitrary transformations
        result = python_agent.ask(question)
        state.update(result)
        return state

    def viz_exec(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes viz code (Matplotlib/Plotly), saves artifact, stores step in FAISS.
        Expected state:
            - question: str
        Sets:
            - type, code, chart_base64
        """
        question = state["question"]
        result = viz_agent.ask(question)
        state.update(result)
        return state

    def text_response(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        For advice/explanations when neither SQL nor Python is required.
        """
        state.update({
            "type": "text",
            "answer": "I can help you with data analysis. Please ask me to query data, clean data, or create visualizations."
        })
        return state

    def critic(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the result using CriticAgent.
        """
        result = {k: v for k, v in state.items() if k not in ["question", "route", "raw_response", "context"]}
        reviewed = critic_agent.review(result)
        state["critic_review"] = reviewed
        return state

    def explainer(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarizes what happened and suggests next steps; stores summary into FAISS.
        Uses ExplainerAgent.
        """
        # Extract the main result (excluding workflow state)
        main_result = {k: v for k, v in state.items() if k not in ["question", "route", "raw_response", "context", "critic_review"]}
        
        explanation = explainer_agent.explain(main_result)
        state["explanation"] = explanation
        return state

    # Build LangGraph
    workflow = StateGraph(dict)

    # Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("examine_data", examine_data)
    workflow.add_node("sql_exec", sql_exec)
    workflow.add_node("python_exec", python_exec)
    workflow.add_node("viz_exec", viz_exec)
    workflow.add_node("text_response", text_response)
    workflow.add_node("critic", critic)
    workflow.add_node("explainer", explainer)

    # Entry
    workflow.set_entry_point("planner")

    # Dynamic routing from planner → one of the executors using conditional edges
    workflow.add_conditional_edges(
        "planner",
        lambda s: s.get("route", "sql_exec"),
        {
            "examine_data": "examine_data",
            "sql_exec": "sql_exec",
            "python_exec": "python_exec",
            "viz_exec": "viz_exec",
            "text_response": "text_response",
        },
    )

    # After any executor, go to critic → explainer → END
    workflow.add_edge("examine_data", "explainer")  # Skip critic for examination
    workflow.add_edge("sql_exec", "critic")
    workflow.add_edge("python_exec", "critic")
    workflow.add_edge("viz_exec", "critic")
    workflow.add_edge("text_response", "critic")
    workflow.add_edge("critic", "explainer")
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


def run_qa(question: str, runner: DuckRunner, tables: List[str]) -> Dict[str, Any]:

    """
    Simplified QA function for API compatibility.
    Creates a DataFrameManager and runs the question through the orchestrator.
    """
    # Create DataFrameManager
    df_manager = DataFrameManager(runner)
    
    # Initialize LLM and FAISS
    llm = GroqClient.from_env()
    faiss_index = FaissIndex(
        "data/faiss/index.bin",
        "data/faiss/meta.json",
        EmbeddingModel()
    )
    
    # IMPORTANT: Ensure the active dataframe (current_df) reflects this session's latest table
    # so that downstream agents use the right data and /download returns the correct snapshot.
    try:
        if tables:
            last_table = tables[-1]
            session_df = runner.load_dataframe(last_table)
            df_manager.set(session_df, name="current_df")
    except Exception as e:
        print(f"[WARN] Could not set session dataframe as current_df: {e}")
    
    # Run through orchestrator
    result = run_agent(question, runner, df_manager, tables, llm, faiss_index)

    # If something unexpected came back (e.g., a string), coerce into a safe response
    if not isinstance(result, dict):
        safe_summary = str(result)
        return {
            "sql": "",
            "rows": 0,
            "preview": pd.DataFrame(),
            "summary": safe_summary,
            "suggested_next": [
                "Try rephrasing your question.",
                "Ensure a dataset is ingested for your session.",
            ],
            "executed_code": "",
            "download_url": "/download",
        }

    
    # Extract and format response for API
    if "error" in result:
        ferr = _friendly_error(str(result.get("error", "")))
        return {
            "sql": result.get("sql", ""),
            "rows": 0,
            "preview": pd.DataFrame(),
            "summary": ferr["summary"],
            "suggested_next": ferr["suggested"],
            "executed_code": result.get("code", ""),
            "download_url": "/download",
            "code_url": "/code",
        }
    
    # Handle different result types
    if result.get("type") == "examination":
        # Prefer explainer's structured output if available
        explanation = result.get("explanation", {})
        sql = ""
        rows = 0
        preview = pd.DataFrame()
        summary = explanation.get("summary") or result.get("analysis", "Data examination completed.")
        suggested_next = explanation.get("next_steps") or result.get("next_steps", [])
        executed_code = ""

        
        # Add examination-specific suggestions if not present
        if not suggested_next and result.get("recommendations"):
            suggested_next = [rec.get("recommendation", "") for rec in result.get("recommendations", [])]
    else:
        # Handle regular analysis results
        # Some states don't nest under 'result'; read directly from state
        explanation = result.get("explanation", {})
        state_keys = result.keys()
        sql = result.get("sql", "")
        rows = result.get("rows", 0)
        preview = result.get("preview", pd.DataFrame())
        executed_code = result.get("code", "")

        
        # Prefer structured fields from explainer
        summary = explanation.get("summary") or explanation.get("explanation", "Analysis completed successfully.")
        suggested_next = explanation.get("next_steps") or []
        
        # If explainer did not provide steps, add some based on type
        if not suggested_next:
            rtype = result.get("type")
            if rtype == "sql":
                suggested_next = [
                    "Try asking for data visualization",
                    "Ask for data cleaning or transformation",
                    "Request summary statistics"
                ]
            elif rtype == "python":
                suggested_next = [
                    "Visualize the transformed data",
                    "Run SQL queries on the cleaned data",
                    "Export the results"
                ]
            elif rtype == "viz":
                suggested_next = [
                    "Modify the chart parameters",
                    "Create additional visualizations",
                    "Export the chart"
                ]

        # For SQL steps, surface SQL as 'executed_code' for sandbox viewing
        if not executed_code and sql:
            executed_code = sql


    
    # If summary missing or generic, create a deterministic fallback from actual outputs
    generic = {"analysis completed successfully.", "analysis completed.", "done.", "success."}
    cleaned_for_check = _clean_text(summary).lower() if summary else ""
    if (not summary) or (cleaned_for_check in generic) or (len(cleaned_for_check) < 40):
        rtype = result.get("type") or ("sql" if sql else "python")
        summary = _fallback_summary(rtype, sql, rows, preview, question)

    # Clean and cap summary to ≤800 chars; Clean suggestions and cap to ≤3.
    summary = _clean_text(summary)
    summary = _cap(summary, 800)
    # Build contextual suggestions if the defaults look generic or empty
    if not suggested_next or all(ss.lower() in {"try asking for data visualization", "ask for data cleaning or transformation", "request summary statistics"} for ss in suggested_next):
        suggested_next = _suggestions_from_context(question, sql, preview, rows)
    # Remove redundant suggestions that repeat user's intent
    ql = question.lower()
    filtered = []
    for s in suggested_next:
        ss = _clean_text(s)
        if any(k in ql for k in ["outlier", "outliers"]) and "outlier" in ss.lower():
            continue
        filtered.append(ss)
    suggested_next = filtered[:3]

    return {
        "sql": sql,
        "rows": rows,
        "preview": preview,
        "summary": summary,
        "suggested_next": suggested_next,
        "executed_code": executed_code,
        "download_url": "/download",
        "code_url": "/code",
    }
