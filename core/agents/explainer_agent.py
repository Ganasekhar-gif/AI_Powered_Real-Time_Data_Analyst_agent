# core/agents/explainer_agent.py

from typing import Dict, Any
import os
import httpx
import json
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex
from dotenv import load_dotenv


class ExplainerAgent:
    """
    Agent that explains what was done, suggests next steps,
    and stores the summary in FAISS.
    """

    def __init__(self, llm: GroqClient, faiss_index: FaissIndex, hf_model_repo: str = "Ganasekhar/explainer-llama3-fine-tuned"):
        self.llm = llm
        self.faiss = faiss_index
        self.hf_model_repo = hf_model_repo

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
1. Provide an action-centric summary of what the agent actually executed and the effect on the dataset (past tense).
2. If code was executed, briefly describe the technique (e.g., winsorization, IQR removal, z-score, normalization) and its impact (rows affected, columns touched, shape changes) based on the provided context. If SQL, mention the operation (e.g., SUM/AVG/COUNT), grouping/filters, and the table used (e.g., current_df).
3. Suggest logical next steps the analyst might consider next.

Constraints:
- Keep the summary concise but complete: prefer 4–7 short sentences within ~500–700 characters (max 800 characters). Do not stop mid‑sentence.
- Avoid repeating the user's instruction verbatim; focus on the actions and results.
- Provide at most 3 next steps, each short and actionable.

Here is the context:
Type: {result.get('type')}
Question: {result.get('question')}
{context}

Return STRICT JSON only with this schema (no markdown, no prose outside JSON):
{{
  "summary": "action-centric narrative of what was executed and its effects",
  "next_steps": ["<=3 short actionable steps"]
}}
"""

    def _hf_complete(self, prompt: str) -> str:
        load_dotenv()

        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN (or HUGGINGFACE_API_TOKEN) in environment")
        url = f"https://api-inference.huggingface.co/models/{self.hf_model_repo}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 400,
                "temperature": 0.6,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        with httpx.Client(timeout=120.0) as c:
            r = c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict) and "generated_text" in item:
                    return str(item["generated_text"])  # type: ignore
            if isinstance(data, dict):
                if "generated_text" in data:
                    return str(data["generated_text"])  # type: ignore
                if "error" in data:
                    raise RuntimeError(f"HF Inference API error: {data['error']}")
            raise RuntimeError("Unexpected response from HF Inference API")

    def explain(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize the operation, suggest next steps, and store summary in FAISS.
        """
        try:
            prompt = self._build_prompt(result)
            explanation_text = self._hf_complete(prompt).strip()

            # Try to parse structured JSON
            summary = explanation_text
            next_steps = []
            try:
                parsed = json.loads(explanation_text)
                if isinstance(parsed, dict):
                    if "summary" in parsed:
                        summary = str(parsed.get("summary", "")).strip()
                    if isinstance(parsed.get("next_steps"), list):
                        next_steps = [str(x).strip() for x in parsed.get("next_steps", [])]
            except Exception:
                # Fall back to raw text
                pass

            # Store explanation in FAISS for memory
            doc_text = (
                f"Explanation for {result.get('type')} task:\n"
                f"Question: {result.get('question')}\n"
                f"Summary:\n{summary}\n\nNext Steps:\n- " + "\n- ".join(next_steps or []) + ("\n\nRaw:\n" + explanation_text if explanation_text != summary else "")
            )
            self.faiss.add([{"text": doc_text}])
            self.faiss.save()

            return {
                "type": "explanation",
                "question": result.get("question"),
                "explanation": explanation_text,
                "summary": summary,
                "next_steps": next_steps,
            }

        except Exception as e:
            return {"error": str(e), "context": result}
