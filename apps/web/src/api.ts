export type ChatRequest = {
  user_id: string;
  session_id: string;
  message: string;
};

export type ChatResponse = {
  sql: string;
  rows: number;
  preview: Array<Record<string, any>>;
  summary: string;
  suggested_next: string[];
  executed_code: string;
  download_url: string;
  code_url: string;
};

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function chat(req: ChatRequest): Promise<ChatResponse> {
  const r = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function ingest(user_id: string, session_id: string, file: File) {
  const fd = new FormData();
  fd.append("user_id", user_id);
  fd.append("session_id", session_id);
  fd.append("file", file, file.name);
  const r = await fetch(`${API_BASE}/ingest`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getCode(user_id: string, session_id: string) {
  const r = await fetch(`${API_BASE}/code?user_id=${encodeURIComponent(user_id)}&session_id=${encodeURIComponent(session_id)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ user_id: string; session_id: string; code: string }>;
}
