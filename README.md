# AI-Powered Real-Time Data Analyst

A modern, agentic platform for seamless, interactive data analysis and visualization. Upload your dataset, ask questions in natural language, and receive insightful explanations, code, and visualizations—just like working with an expert data analyst.

---

## 🚀 Features
- **Conversational Data Analysis**: Ask questions about your data and get clear, human-readable answers, actionable summaries, and code.
- **Session-Based Workflow**: Organize analyses into sessions, switch between them, and revisit past conversations.
- **Automatic Visualizations**: Request charts and plots—get code and images instantly, with no GUI popups.
- **Smart Explanations**: Every answer includes a concise, context-aware summary and next-step suggestions.
- **Downloadable Results**: Export the latest processed dataset as CSV.

---

## 🛠️ Tech Stack
- **Frontend**: React + Vite + TailwindCSS + Lucide Icons
- **Backend**: FastAPI (Python)
- **Agents/Orchestration**: LangGraph, custom Python agents
- **LLM APIs**: Groq, Hugging Face Inference API (for fine-tuned explainer)
- **Data Storage**: DuckDB (in-process OLAP), Pandas
- **Vector Search**: FAISS (for knowledge retrieval)
- **Visualization**: Matplotlib, Plotly (static image export)

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────┐
│        User (Web UI)        │
└─────────────┬────────────────┘
              │ 1. Ask question / Upload data
              ▼
┌──────────────────────────────┐
│      React Frontend App      │
└─────────────┬────────────────┘
              │ 2. REST API request
              ▼
┌──────────────────────────────┐
│     FastAPI Backend API      │
└─────────────┬────────────────┘
              │ 3. Agent Orchestration
              ▼
┌──────────────────────────────────────────────────────────────┐
│           Agent Workflow (Planner → SQL/Python/Viz)         │
│ ┌───────────────┐   ┌───────────────┐   ┌─────────────────┐ │
│ │   Planner     │→→ │  SQL Agent    │→→│   Python Agent   │ │
│ │ (Intent/Routing)│  │ (Query/Group)│  │ (Pandas/Logic)  │ │
│ └───────────────┘   └───────────────┘   └─────────────────┘ │
│         │                        │                          │
│         ▼                        ▼                          │
│   ┌─────────────┐         ┌──────────────┐                 │
│   │ DuckDB DB   │         │  Pandas DF   │                 │
│   └─────────────┘         └──────────────┘                 │
│         │                        │                          │
│         └─────────────┬──────────┘                          │
│                       ▼                                     │
│                ┌──────────────┐                             │
│                │ Viz Agent    │                             │
│                └─────┬────────┘                             │
│                      ▼                                      │
│              ┌─────────────────┐                            │
│              │ Matplotlib/Plotly│ (Chart/Export)            │
│              └─────────────────┘                            │
│         │                        │                          │
│         ▼                        ▼                          │
│   ┌─────────────┐         ┌──────────────┐                 │
│   │ FAISS Index │         │ Explainer    │ (LLM: HF/Groq)  │
│   └─────────────┘         └──────────────┘                 │
└──────────────────────────────────────────────────────────────┘
              │ 4. Response (summary, code, chart, suggestions)
              ▼
┌──────────────────────────────┐
│      React Frontend App      │
└─────────────┬────────────────┘
              │ 5. Display results to user
              ▼
┌──────────────────────────────┐
│        User (Web UI)        │
└──────────────────────────────┘
```

**Legend:**
- User interacts with the web UI to upload data and ask questions.
- Frontend sends requests to the backend API.
- Backend orchestrates agents (Planner, SQL, Python, Viz, Explainer) and uses DuckDB, Pandas, FAISS, and LLM APIs as needed.
- Visualizations and summaries are generated and returned to the frontend.
- User receives answers, code, charts, and suggestions in a chat-like interface.

---

## ⚡ Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/Ganasekhar-gif/AI_Powered_Real-Time_Data_Analyst_agent.git
cd AI_Powered_Real-Time_Data_Analyst_agent
python -m venv venv
venv\Scripts\activate  # (Windows) or source venv/bin/activate (Linux/Mac)
pip install -r requirements.txt
cd apps/web
npm install
```

### 2. Environment Setup
- Copy `.env.example` to `.env` and fill in your API keys (Groq, HuggingFace, etc.)

### 3. Start Backend (FastAPI)
```bash
uvicorn apps.api.main:app --reload
```

### 4. Start Frontend (React)
```bash
cd apps/web
npm run dev
```

- Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 💡 Usage
- **Upload** a CSV/XLS/XLSX to start a session.
- **Ask questions** about your data (e.g., "Show total sales by category").
- **Get answers** with summaries, code, previews, and suggestions.
- **Download** the processed dataset or code at any step.
- **Switch sessions** to revisit or organize your work.

---

## 🤖 Example Conversation

> **You:** Find the total transaction Amount grouped by Transaction Type
>
> **AI:**
> Based on your data, I explored how Amount varies across different Transaction Types. Here's what stands out:
> - Salary: Amount totals 165,000
> - Bonus: Amount totals 10,000
> - Dining: Amount totals -650
> ...plus 7 more Transaction Types with smaller totals.
> This suggests that certain Transaction Types contribute much more to the total Amount than others. In short, you may want to focus on the top contributors for deeper analysis or optimization.

---

## 📦 Project Structure

- `apps/api` — FastAPI backend
- `apps/web` — React frontend
- `core/agents` — Agent logic (explainer, viz, etc.)
- `core/llm` — LLM orchestration, SQL/Python agent
- `data/` — User data, snapshots, and code

---

## 🐳 Docker Setup

### 1. Build & Run with Docker Compose

```bash
docker-compose up --build
```

- This will build and start both the FastAPI backend and React frontend.
- The backend will be available at http://localhost:8000
- The frontend will be available at http://localhost:5173

### 2. Environment Variables

- Copy `.env.example` to `.env` in the project root and fill in your API keys (Groq, HuggingFace, etc.).
- The `.env` file is already included in `.gitignore` for safety.

---

## 📄 License
MIT License
