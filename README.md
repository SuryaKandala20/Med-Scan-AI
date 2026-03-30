# 🏥 MedScan AI v2 — React + FastAPI

## Quick Start (2 terminals needed)

### Terminal 1 — Backend (FastAPI):
```
cd backend
pip install -r requirements.txt
copy .env.example .env
# Edit .env → add your OpenAI key
uvicorn server:app --reload --port 8000
```

### Terminal 2 — Frontend (React):
```
cd frontend
npm install
npm start
```

Opens at http://localhost:3000

## What changed from v1 (Streamlit):
- ✅ Streaming responses (word-by-word like ChatGPT)
- ✅ PDF upload for report explainer
- ✅ Professional React UI with sidebar navigation
- ✅ FastAPI backend with proper REST endpoints
- ✅ Server-Sent Events for real-time streaming
- ✅ Same ChromaDB + SQLite + GPT-4o backend
