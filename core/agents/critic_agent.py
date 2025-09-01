# core/agents/critic_agent.py

from typing import Dict, Any
import pandas as pd

from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex


class CriticAgent:
    """
    Validates and critiques outputs from SQLAgent, PythonAgent, and VizAgent
    before passing them to ExplainerAgent.

    Purpose:
    - Prevents garbage-in → garbage-out for ExplainerAgent.
    - Adds a safety net so that the user doesn’t see broken SQL, Python errors, or nonsense charts.
    - In the future: can auto-correct outputs by retrying with LLM if validation fails.
    """

    def __init__(self, llm: GroqClient, faiss_index: FaissIndex):
        self.llm = llm
        self.faiss = faiss_index

    # -------------------------------
    # Validation helpers
    # -------------------------------
    def _validate_sql(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in result:
            return {"approved": False, "error": f"SQL error: {result['error']}"}

        if result.get("rows", 0) == 0:
            return {"approved": False, "error": "SQL returned no rows"}

        if result.get("preview") is None or not isinstance(result["preview"], pd.DataFrame):
            return {"approved": False, "error": "Invalid SQL result preview"}

        return {"approved": True, "validated_result": result}

    def _validate_python(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in result:
            return {"approved": False, "error": f"Python error: {result['error']}"}

        if result.get("rows", 0) == 0:
            return {"approved": False, "error": "Python transformation produced empty dataframe"}

        if result.get("preview") is None or not isinstance(result["preview"], pd.DataFrame):
            return {"approved": False, "error": "No valid DataFrame after transformation"}

        return {"approved": True, "validated_result": result}

    def _validate_viz(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in result:
            return {"approved": False, "error": f"Viz error: {result['error']}"}

        if not result.get("chart"):
            return {"approved": False, "error": "No chart object produced"}

        return {"approved": True, "validated_result": result}

    # -------------------------------
    # Main validation entrypoint
    # -------------------------------
    def review(self, result: Dict[str, Any]) -> Dict[str, Any]:
        rtype = result.get("type")

        if rtype == "sql":
            return self._validate_sql(result)
        elif rtype == "python":
            return self._validate_python(result)
        elif rtype == "viz":
            return self._validate_viz(result)
        else:
            return {"approved": False, "error": f"Unknown result type: {rtype}"}

    # -------------------------------
    # (Future) Auto-correction
    # -------------------------------
    def retry_if_failed(self, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Future extension:
        If validation fails, ask LLM to regenerate a safer alternative.
        """
        if result.get("approved", True):
            return result  # Already valid

        error = result.get("error", "Unknown error")

        prompt = f"""
You are a critic & fixer agent.
The following attempt failed:

Question: {question}
Error: {error}

Please regenerate a safer, corrected response (SQL, Python, or Viz code depending on request).
Only return the corrected code, nothing else.
"""
        suggestion = self.llm.complete(prompt).strip()

        # Save into FAISS memory
        try:
            self.faiss.add([{"text": f"Critic correction for: {question}\n{suggestion}"}])
            self.faiss.save()
        except Exception as e:
            print(f"[WARN] Could not save critic correction: {e}")

        return {"approved": False, "error": error, "suggestion": suggestion}
 
