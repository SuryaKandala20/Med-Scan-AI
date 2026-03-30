# MedScan AI v2

MedScan AI is a full-stack medical assistant app built to help people understand their medical reports, check skin conditions, ask health questions, and look up drug interactions — all in one place. The backend is FastAPI with GPT-4o powering the AI features, and the frontend is React with real-time streaming responses so you get answers word by word instead of waiting.

The idea behind this project was to make medical information more accessible. Most people receive a report from their doctor and have no idea what it means. MedScan lets you upload it and get a plain-language explanation instantly. The same goes for skin concerns, drug combinations, or just asking a medical question in your own words.

---

## Features

### Medical Report Explainer
Upload any medical PDF — ECG, ultrasound, blood work, pathology report, X-ray summary — and MedScan breaks it down page by page in plain language. It reads both the text and the images on each page, so it works on scanned PDFs too. You get a structured explanation of what the report says, what the key values mean, and what to discuss with your doctor. No more Googling medical terms one by one.

### Report Comparison
Have two versions of the same report from different dates? Upload both and MedScan compares them side by side. It highlights what changed, what improved, and what got worse — useful for tracking chronic conditions or post-treatment follow-ups.

### Skin Lesion Classifier
Upload a photo of a skin lesion or mole and the app runs it through a PyTorch deep learning model (EfficientNet trained on the HAM10000 dermatology dataset with 10,000+ real images). It returns the predicted lesion type with a confidence score. The 7 classes it can detect include melanoma, melanocytic nevi, basal cell carcinoma, actinic keratosis, and more. This is not a replacement for a dermatologist but it gives you an informed starting point.

### Medical Chatbot
Ask any health or medical question in plain English. The chatbot is powered by GPT-4o and backed by a ChromaDB vector knowledge base so answers are grounded rather than made up. Responses stream word by word in real time. You can ask things like "what does a high creatinine level mean", "what are the symptoms of hypothyroidism", or "is it safe to take ibuprofen with blood pressure medication".

### Drug Interaction Checker
Type in two or more medication names and MedScan checks for known interactions between them. It explains what the interaction is, how serious it is, and what symptoms to watch for. Useful for patients managing multiple prescriptions or caregivers tracking medications for elderly family members.

### Audit Log / Admin Panel
Every query made through the app is logged — what feature was used, what was uploaded or asked, and when. The Admin page shows the full history. This was added to keep track of usage and for anyone deploying this in a clinical or research setting where you need a record of interactions.

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

## What I Built

MedScan started as a simple Streamlit prototype. I rebuilt the entire thing from scratch into a full-stack web app to make it feel like a real product — not just a data science demo.

**New in v2:**
- Replaced Streamlit with a custom React frontend — full sidebar navigation, clean UI, mobile-friendly layout
- Rebuilt the backend with FastAPI to support proper REST endpoints and real-time streaming
- Added streaming responses using Server-Sent Events — the AI replies word by word, just like ChatGPT, instead of making you wait for the full answer
- Rewrote the report explainer to be fully multimodal — it now sends each page as both an image and extracted text to the model, so it works on any report type (ECG, ultrasound, labs, pathology) without any hardcoding
- Added support for scanned PDFs using Tesseract OCR as a fallback when there is no embedded text
- Built a Report Comparison page so you can upload two reports and see what changed between them
- Added a Drug Interaction Checker page
- Added an Admin panel that shows a full audit log of every query made through the app
- Containerized everything with Docker so anyone can run it with a single command

---

## License

MIT
