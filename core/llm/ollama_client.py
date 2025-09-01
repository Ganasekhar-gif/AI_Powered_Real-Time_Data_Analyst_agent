import os, httpx


class LLMNotAvailable(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.timeout = timeout
        self.is_available = self._check()


    @classmethod
    def from_env(cls):
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        return cls(base)


    def _check(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as c:
                r = c.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False


    def complete(self, prompt: str, temperature: float = 0.2) -> str:
        if not self.is_available:
            raise LLMNotAvailable("Ollama not reachable. Start Ollama or set OLLAMA_BASE_URL.")
        payload = {"model": self.model, "prompt": prompt, "options": {"temperature": temperature}, "stream": False}
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip() 
