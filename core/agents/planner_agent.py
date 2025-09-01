# core/agents/planner_agent.py

from typing import Dict, Any
from core.agents.sql_agent import SQLAgent
from core.agents.python_agent import PythonAgent
from core.agents.viz_agent import VizAgent
from core.agents.explainer_agent import ExplainerAgent
from core.agents.critic_agent import CriticAgent
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex


class PlannerAgent:
    """
    High-level orchestrator that decides which specialized agent
    (SQL, Python, Viz) should handle a user request.
    - Routes user queries to the right agent
    - Passes results through CriticAgent for validation
    - If valid, sends to ExplainerAgent for summarization
    """

    def __init__(
        self,
        sql_agent: SQLAgent,
        python_agent: PythonAgent,
        viz_agent: VizAgent,
        explainer_agent: ExplainerAgent,
        critic_agent: CriticAgent,
        llm: GroqClient,
        faiss_index: FaissIndex,
    ):
        self.sql_agent = sql_agent
        self.python_agent = python_agent
        self.viz_agent = viz_agent
        self.explainer_agent = explainer_agent
        self.critic_agent = critic_agent
        self.llm = llm
        self.faiss = faiss_index

    def _classify(self, question: str) -> str:
        """
        Decide which agent to use based on the question.
        Uses a lightweight heuristic + LLM fallback.
        """
        q = question.lower()

        # Simple heuristics first
        if any(word in q for word in ["plot", "chart", "visualize", "graph", "histogram", "scatter", "bar"]):
            return "viz"
        if any(word in q for word in ["clean", "transform", "feature", "wrangle", "fillna", "dropna", "encode", "normalize"]):
            return "python"
        if any(word in q for word in ["select", "group by", "join", "count", "sum", "avg", "max", "min"]):
            return "sql"

        # Fallback: ask LLM to decide
        prompt = f"""
You are a router. Classify the following request into exactly one category:
- "sql" if it's about querying structured tables with SQL
- "python" if it's about cleaning, wrangling, or feature engineering
- "viz" if it's about charts, plots, or visualization

Request: {question}
Answer only one of: sql, python, viz
"""
        decision = self.llm.complete(prompt).strip().lower()
        if decision not in ["sql", "python", "viz"]:
            decision = "sql"  # fallback default

        return decision

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Route the question to the appropriate agent,
        validate with CriticAgent, then enrich with ExplainerAgent.
        """
        route = self._classify(question)

        if route == "sql":
            result = self.sql_agent.ask(question)
        elif route == "python":
            result = self.python_agent.ask(question)
        elif route == "viz":
            result = self.viz_agent.ask(question)
        else:
            return {"error": f"Unknown route {route}", "question": question}

        # First line of defense: check agent result
        if "error" in result:
            return result

        # Validation by CriticAgent
        reviewed = self.critic_agent.review(result)
        if not reviewed.get("approved", False):
            # Optionally attempt auto-correction
            retry = self.critic_agent.retry_if_failed(question, reviewed)
            return {
                "result": result,
                "critic_review": reviewed,
                "retry_suggestion": retry,
                "status": "rejected_by_critic",
            }

        # Send only validated result to ExplainerAgent
        explanation = self.explainer_agent.explain(reviewed["validated_result"])

        return {
            "result": reviewed["validated_result"],
            "explanation": explanation,
            "status": "approved",
        }
