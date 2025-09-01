# core/agents/explainer_agent.py

from typing import Dict, Any
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex


class ExplainerAgent:
    """
    Agent that explains what was done, suggests next steps,
    and stores the summary in FAISS.
    """

    def __init__(self, llm: GroqClient, faiss_index: FaissIndex):
        self.llm = llm
        self.faiss = faiss_index

    def _build_prompt(self, result: Dict[str, Any]) -> str:
        """
        Build a prompt for explanation based on the result dict
        from SQLAgent, PythonAgent, or VizAgent.
        """
        context = ""
        if "sql" in result:
            context += f"\nSQL Query:\n{result['sql']}"
        if "code" in result:
            context += f"\nPython Code:\n{result['code']}"
        if "preview" in result:
            context += f"\nPreview:\n{result['preview'].head(5).to_string(index=False)}"

        return f"""
You are an AI Explainer.

Task:
1. Summarize what was done in simple words.
2. Suggest logical next steps the analyst might consider.

Here is the context:
Type: {result.get('type')}
Question: {result.get('question')}
{context}

Respond with:
- Summary (2–3 sentences max)
- Next Steps (bullet points)
"""

    def explain(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize the operation, suggest next steps, and store summary in FAISS.
        """
        try:
            prompt = self._build_prompt(result)
            explanation = self.llm.complete(prompt).strip()

            # Store explanation in FAISS for memory
            doc_text = (
                f"Explanation for {result.get('type')} task:\n"
                f"Question: {result.get('question')}\n"
                f"Explanation:\n{explanation}"
            )
            self.faiss.add([{"text": doc_text}])
            self.faiss.save()

            return {
                "type": "explanation",
                "question": result.get("question"),
                "explanation": explanation,
            }

        except Exception as e:
            return {"error": str(e), "context": result}
 
