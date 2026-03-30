# MedScan AI v2

An AI-powered medical assistant that analyzes medical reports, performs skin lesion classification, answers medical questions, and checks drug interactions — built with React + FastAPI.

---

## Features

- **Medical Report Explainer** — Upload any PDF (ECG, ultrasound, lab report, pathology) and get a plain-language AI explanation with page-by-page multimodal analysis
- **Report Comparison** — Compare two reports side-by-side to track changes over time
- **Skin Lesion Classifier** — Upload a dermoscopy image; a PyTorch model (trained on HAM10000) classifies the lesion type with confidence scores
- **Medical Chat** — Ask any medical question; answers are grounded in a ChromaDB vector knowledge base with streaming word-by-word responses (like ChatGPT)
- **Drug Interaction Checker** — Enter multiple drug names and get AI-powered interaction warnings
- **Audit Logging** — Every interaction is logged to a local SQLite database, viewable from the Admin panel

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, React Router v6, Lucide icons, Server-Sent Events |
| Backend | FastAPI, Uvicorn, Python 3.11 |
| AI / LLM | OpenAI GPT-4o / GPT-4o-mini |
| Vector DB | ChromaDB + Sentence Transformers |
| Skin Model | PyTorch + timm (EfficientNet), trained on HAM10000 |
| PDF Parsing | PyMuPDF + Tesseract OCR (scanned PDFs supported) |
| Image Processing | OpenCV, Pillow, Albumentations |
| Audit DB | SQLite |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
Med-Scan-AI/
├── backend/
│   ├── server.py              # FastAPI app — all API routes
│   ├── config.py              # Loads .env keys
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Backend container
│   ├── .env.example           # Template for environment variables
│   ├── modules/
│   │   ├── vector_db.py       # ChromaDB setup + retrieval
│   │   ├── llm_chat.py        # OpenAI streaming chat
│   │   ├── report_explainer.py# Multimodal PDF analysis
│   │   ├── skin_classifier.py # PyTorch inference
│   │   ├── image_quality.py   # Pre-upload image checks
│   │   └── audit_logger.py    # SQLite audit trail
│   ├── models/                # Trained .pth model files (not in git — see below)
│   └── data/                  # ChromaDB + HAM10000 + audit DB (not in git)
├── frontend/
│   ├── src/
│   │   ├── App.js             # Routing + sidebar layout
│   │   ├── components/
│   │   │   ├── ReportPage.js        # Report explainer UI
│   │   │   ├── ReportComparePage.js # Report comparison UI
│   │   │   ├── SkinPage.js          # Skin classifier UI
│   │   │   ├── ChatPage.js          # Medical chat UI
│   │   │   ├── DrugCheckerPage.js   # Drug interaction UI
│   │   │   └── AdminPage.js         # Audit log viewer
│   │   ├── utils/api.js       # Axios + SSE helpers
│   │   └── styles/global.css  # App-wide styles
│   ├── Dockerfile             # Multi-stage: Node build → Nginx serve
│   └── nginx.conf             # Proxies /api/* to backend
└── docker-compose.yml         # One-command deployment
```

---

## Quick Start — Docker (Recommended)

**Prerequisites:** Docker Desktop installed and running.

```bash
# 1. Clone the repo
git clone https://github.com/SuryaKandala20/Med-Scan-AI.git
cd Med-Scan-AI

# 2. Set up environment variables
cp backend/.env.example backend/.env
# Open backend/.env and add your keys:
#   OPENAI_API_KEY=sk-...
#   OPENAI_MODEL=gpt-4o-mini

# 3. Build and start both services
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

To stop: `docker compose down`

---

## Quick Start — Local Development (2 terminals)

### Terminal 1 — Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
uvicorn server:app --reload --port 8000
```

### Terminal 2 — Frontend

```bash
cd frontend
npm install
npm start
```

Opens at http://localhost:3000

---

## Environment Variables

Create `backend/.env` (never commit this file):

```env
# Required
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini        # or gpt-4o for better quality

# Optional — needed only to download the HAM10000 dataset
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Get your OpenAI key at https://platform.openai.com/api-keys

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/report` | Upload PDF/image → streamed AI explanation |
| POST | `/api/report/compare` | Upload 2 PDFs → streamed comparison |
| POST | `/api/skin` | Upload image → skin lesion classification |
| POST | `/api/chat` | Send message → streamed AI reply |
| POST | `/api/drugs` | Check drug interactions |
| GET | `/api/audit` | Retrieve audit log entries |
| GET | `/health` | Health check |

Interactive API docs available at http://localhost:8000/docs when running.

---

## Skin Classifier — Model Training

The skin classifier uses EfficientNet trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection).

```bash
cd backend
python setup_data.py   # downloads HAM10000 via Kaggle API
python train_model.py  # trains and saves model to models/
```

Requires Kaggle credentials in `.env`.

---

## Data and Models (Not in Git)

Large files are excluded from the repository:

| Path | What it is | How to get it |
|------|-----------|---------------|
| `backend/data/ham10000/` | Training images (~1.5 GB) | Run `setup_data.py` |
| `backend/data/chromadb/` | Vector knowledge base | Auto-created on first run |
| `backend/data/medscan_audit.db` | Audit log database | Auto-created on first run |
| `backend/models/*.pth` | Trained model weights | Run `train_model.py` |

---

## What Changed from v1 (Streamlit)

- Streaming responses — word-by-word output like ChatGPT via Server-Sent Events
- Multimodal report analysis — model sees each page as both image and text; no hardcoded report types
- Professional React UI with sidebar navigation replacing Streamlit
- FastAPI backend with proper REST endpoints
- Scanned PDF support via Tesseract OCR fallback
- Audit logging with Admin panel
- Docker support for one-command deployment

---

## License

MIT
