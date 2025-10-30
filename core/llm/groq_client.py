import os
from groq import Groq
from dotenv import load_dotenv

class GroqClient:
    def __init__(self, api_key: str, model: str = None):
        self.client = Groq(api_key=api_key)
        # Prefer env override, then provided arg, then a modern default
        env_model = os.getenv("GROQ_MODEL")
        self.model = env_model or model or "llama-3.1-8b-instant"

    @classmethod
    def from_env(cls):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        return cls(api_key)

    def complete(self, prompt: str) -> str:
        # Allow overriding max tokens via env, default to 512 to stay within limits
        try:
            max_toks = int(os.getenv("GROQ_MAX_TOKENS", "512"))
        except Exception:
            max_toks = 512
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_toks,
        )
        return resp.choices[0].message.content
