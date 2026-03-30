"""
backend/server.py - FastAPI Backend with Streaming + Generic Multimodal Report Understanding

What changed:
- /api/report now builds a page-wise multimodal payload
- no hardcoding for ECG / ultrasound / labs / pathology
- the model sees each page image + OCR/native text and decides the report type itself
- scanned PDFs and image uploads are supported
- /api/report/compare also accepts OCR/image-backed documents
"""

import base64
import io
import json
import os
import re
import time
import uuid
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pytesseract
from PIL import Image, ImageOps
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modules.vector_db import MedicalVectorDB
from modules.audit_logger import AuditLogger, init_db
from modules.report_explainer import ReportExplainer
from modules.skin_classifier import SkinClassifier
from modules.image_quality import ImageQualityChecker
from modules.llm_chat import MEDICAL_SYSTEM_PROMPT
from config import get_openai_key, get_openai_model, validate_config


sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[START] MedScan AI Backend starting...")
    init_db()

    vdb = MedicalVectorDB()
    vdb.initialize()
    app.state.vdb = vdb

    print("[READY] Backend ready")
    yield
    print("[STOP] Backend shutting down")


app = FastAPI(title="MedScan AI", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_vdb() -> MedicalVectorDB:
    vdb = getattr(app.state, "vdb", None)
    if vdb is None:
        vdb = MedicalVectorDB()
        vdb.initialize()
        app.state.vdb = vdb
    return vdb


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "conversation": [],
            "audit": AuditLogger(session_id=session_id),
        }
    return sessions[session_id]


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    language: str = "English"


class ResetRequest(BaseModel):
    session_id: str = ""


class DrugCheckRequest(BaseModel):
    drugs: List[str]
    language: str = "English"


SUPPORTED_LANGUAGES = [
    "English", "Hindi", "Spanish", "French", "German", "Telugu",
    "Tamil", "Bengali", "Marathi", "Urdu", "Arabic", "Chinese",
    "Japanese", "Korean", "Portuguese", "Russian",
]


def _normalize_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _extract_first(source: Any, keys: List[str]) -> Any:
    for key in keys:
        try:
            if key in source and source.get(key) not in (None, ""):
                return source.get(key)
        except Exception:
            continue
    return None


def _extract_json_from_text(content: str) -> Optional[dict]:
    if not content or not isinstance(content, str):
        return None

    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _extract_assessment(content: str) -> Optional[dict]:
    data = _extract_json_from_text(content)
    if isinstance(data, dict) and data.get("assessment") is True:
        return data
    return None


def _configure_tesseract() -> None:
    if os.name == "nt":
        env_path = os.getenv("TESSERACT_CMD")
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        if env_path:
            pytesseract.pytesseract.tesseract_cmd = env_path
        elif os.path.exists(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path


def _prepare_for_ocr(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    return image


def _ocr_pil_image(image: Image.Image) -> str:
    _configure_tesseract()
    image = _prepare_for_ocr(image)

    try:
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 3 --psm 6"
        )
        return (text or "").strip()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed. Make sure Tesseract OCR is installed on your system. Error: {str(e)}"
        )


def _resize_for_model(image: Image.Image, max_side: int = 1400) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image

    scale = max_side / largest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _pil_to_data_url(image: Image.Image) -> str:
    image = _resize_for_model(image, max_side=1400)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _combine_text(native_text: str, ocr_text: str) -> str:
    native_text = (native_text or "").strip()
    ocr_text = (ocr_text or "").strip()

    native_clean = re.sub(r"\s+", " ", native_text).strip()
    ocr_clean = re.sub(r"\s+", " ", ocr_text).strip()

    if native_clean and ocr_clean:
        if native_clean.lower() == ocr_clean.lower():
            return native_text
        return (
            "[Native extracted text]\n"
            f"{native_text}\n\n"
            "[OCR text]\n"
            f"{ocr_text}"
        ).strip()

    return native_text or ocr_text or ""


def _empty_report_payload(source_name: str = "user_submission") -> dict:
    return {
        "source_name": source_name,
        "file_kind": "unknown",
        "text_notes": [],
        "pages": [],
        "combined_text": "",
    }


def _refresh_combined_text(payload: dict) -> dict:
    parts: List[str] = []

    for idx, note in enumerate(payload.get("text_notes", []), start=1):
        if note and note.strip():
            parts.append(f"=== TEXT NOTE {idx} ===\n{note.strip()}")

    for page in payload.get("pages", []):
        page_num = page.get("page_number", "?")
        page_text = (page.get("page_text") or "").strip()
        if page_text:
            parts.append(f"=== PAGE {page_num} ===\n{page_text}")

    payload["combined_text"] = "\n\n".join(parts).strip()
    return payload


def _merge_payloads(text: str = "", file_payload: Optional[dict] = None) -> dict:
    payload = _empty_report_payload(
        source_name=(file_payload or {}).get("source_name", "user_submission")
    )

    if text and text.strip():
        payload["text_notes"].append(text.strip())

    if file_payload:
        payload["file_kind"] = file_payload.get("file_kind", "unknown")
        payload["pages"] = file_payload.get("pages", [])
        if not payload["source_name"]:
            payload["source_name"] = file_payload.get("source_name", "user_submission")

    return _refresh_combined_text(payload)


def _extract_pdf_payload(content: bytes, filename: str) -> dict:
    try:
        import fitz
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PDF parsing dependency missing. Run: pip install pymupdf"
        )

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open PDF: {str(e)}")

    pages: List[dict] = []

    try:
        for page_number, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))

            native_text = (page.get_text("text") or "").strip()
            native_clean = re.sub(r"\s+", " ", native_text).strip()

            ocr_text = ""
            if len(native_clean) < 250:
                ocr_text = _ocr_pil_image(page_image)

            page_text = _combine_text(native_text, ocr_text)

            pages.append(
                {
                    "page_number": page_number,
                    "source_name": filename,
                    "media_type": "pdf_page",
                    "page_text": page_text,
                    "image_data_url": _pil_to_data_url(page_image),
                }
            )

        payload = {
            "source_name": filename,
            "file_kind": "pdf",
            "text_notes": [],
            "pages": pages,
            "combined_text": "",
        }
        return _refresh_combined_text(payload)

    finally:
        doc.close()


def _extract_image_payload(content: bytes, filename: str) -> dict:
    try:
        image = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {str(e)}")

    ocr_text = _ocr_pil_image(image)

    payload = {
        "source_name": filename,
        "file_kind": "image",
        "text_notes": [],
        "pages": [
            {
                "page_number": 1,
                "source_name": filename,
                "media_type": "image",
                "page_text": ocr_text,
                "image_data_url": _pil_to_data_url(image),
            }
        ],
        "combined_text": "",
    }
    return _refresh_combined_text(payload)


async def _read_uploaded_document(file_obj: Any) -> dict:
    if not file_obj or not hasattr(file_obj, "filename") or not file_obj.filename:
        return _empty_report_payload()

    filename = file_obj.filename
    lower_name = filename.lower()
    content = await file_obj.read()

    if not content:
        return _empty_report_payload(source_name=filename)

    if lower_name.endswith(".pdf"):
        return _extract_pdf_payload(content, filename)

    if lower_name.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore").strip()
        return _merge_payloads(text=text, file_payload={"source_name": filename, "file_kind": "txt", "pages": []})

    if lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return _extract_image_payload(content, filename)

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Upload PDF, TXT, PNG, JPG, JPEG, or WEBP."
    )


async def _build_single_payload_from_text_and_file(text: str, file_obj: Any) -> dict:
    text = _normalize_text(text)
    file_payload = await _read_uploaded_document(file_obj) if file_obj else None

    if not text and not file_payload:
        raise HTTPException(
            status_code=400,
            detail="Please paste report text or upload a supported file."
        )

    payload = _merge_payloads(text=text, file_payload=file_payload)

    if not payload.get("text_notes") and not payload.get("pages"):
        raise HTTPException(
            status_code=400,
            detail="Could not extract any readable content from the provided input."
        )

    return payload


async def _parse_single_report_request(request: Request) -> dict:
    content_type = (request.headers.get("content-type") or "").lower()

    text = ""
    file_obj = None

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")

        if not isinstance(payload, dict):
            payload = {"text": str(payload)}

        text = _normalize_text(
            _extract_first(payload, ["text", "report_text", "content", "message"])
        )

    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse form data: {str(e)}")

        text = _normalize_text(
            _extract_first(form, ["text", "report_text", "content", "message"])
        )
        file_obj = _extract_first(form, ["file", "report_file", "document", "upload"])

    elif "text/plain" in content_type:
        raw = await request.body()
        text = raw.decode("utf-8", errors="ignore").strip()

    else:
        raw = await request.body()
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    payload = {"text": str(payload)}
                text = _normalize_text(
                    _extract_first(payload, ["text", "report_text", "content", "message"])
                )
            except Exception:
                text = raw.decode("utf-8", errors="ignore").strip()

    return await _build_single_payload_from_text_and_file(text, file_obj)


async def _parse_compare_request(request: Request) -> Dict[str, dict]:
    content_type = (request.headers.get("content-type") or "").lower()

    text1 = ""
    text2 = ""
    file1 = None
    file2 = None

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object.")

        text1 = _normalize_text(
            _extract_first(payload, ["text1", "report1", "older_report", "left_text"])
        )
        text2 = _normalize_text(
            _extract_first(payload, ["text2", "report2", "newer_report", "right_text"])
        )

    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse form data: {str(e)}")

        text1 = _normalize_text(
            _extract_first(form, ["text1", "report1", "older_report", "left_text"])
        )
        text2 = _normalize_text(
            _extract_first(form, ["text2", "report2", "newer_report", "right_text"])
        )
        file1 = _extract_first(form, ["file1", "report_file1", "older_file", "left_file"])
        file2 = _extract_first(form, ["file2", "report_file2", "newer_file", "right_file"])

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported content type for compare endpoint."
        )

    payload1 = await _build_single_payload_from_text_and_file(text1, file1)
    payload2 = await _build_single_payload_from_text_and_file(text2, file2)

    return {"report1": payload1, "report2": payload2}


def _payload_preview(payload: dict, limit: int = 500) -> str:
    preview = (payload.get("combined_text") or "").strip()
    if not preview:
        return "No extracted text"
    return preview[:limit]


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    from openai import OpenAI

    api_key = get_openai_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")

    session_id = req.session_id or str(uuid.uuid4())
    session = get_session(session_id)
    audit = session["audit"]

    session["conversation"].append({"role": "user", "content": req.message})
    audit.log_message(role="user", content=req.message)

    vdb = get_vdb()
    rag_context = vdb.get_context_for_symptoms(req.message)

    system_prompt = MEDICAL_SYSTEM_PROMPT
    if req.language and req.language != "English":
        system_prompt += (
            f"\n\nIMPORTANT: The user prefers {req.language}. "
            f"Respond ENTIRELY in {req.language}. Use {req.language} for all text including the "
            f"assessment JSON field values. Keep JSON keys in English but all values in {req.language}."
        )

    if rag_context:
        system_prompt += "\n\n## MEDICAL KNOWLEDGE BASE\n" + rag_context

    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(session["conversation"])

    client = OpenAI(api_key=api_key)

    def generate():
        full_response = ""
        start_time = time.time()

        try:
            stream = client.chat.completions.create(
                model=get_openai_model(),
                messages=api_messages,
                temperature=0.3,
                max_tokens=2000,
                stream=True,
            )

            yield "data: " + json.dumps({"type": "session", "session_id": session_id}) + "\n\n"

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    token = delta.content
                    full_response += token
                    yield "data: " + json.dumps({"type": "token", "content": token}) + "\n\n"

            assessment = _extract_assessment(full_response)
            if assessment:
                yield "data: " + json.dumps({"type": "assessment", "data": assessment}) + "\n\n"

                conditions = assessment.get("conditions", [])
                top = conditions[0] if conditions else {}
                doctor = assessment.get("doctor_referral", {})

                audit.log_assessment(
                    symptoms=[],
                    conditions=conditions,
                    top_condition=top.get("name", "Unknown"),
                    triage_level=assessment.get("triage", {}).get("level", "Routine"),
                    treatments=assessment.get("treatments", []),
                    specialist=doctor.get("specialty", ""),
                    model_used=get_openai_model(),
                    raw_response=full_response,
                )

            latency = int((time.time() - start_time) * 1000)
            yield "data: " + json.dumps({"type": "done", "latency_ms": latency}) + "\n\n"

            session["conversation"].append({"role": "assistant", "content": full_response})
            audit.log_message(
                role="bot",
                content=full_response[:500],
                latency_ms=latency,
                model_used=get_openai_model(),
                has_assessment=bool(assessment),
            )

        except Exception as e:
            error_msg = str(e)
            audit.log_error("api_error", "chat", error_msg, traceback.format_exc())
            yield "data: " + json.dumps({"type": "error", "message": error_msg}) + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat/reset")
async def reset_chat(req: Optional[ResetRequest] = None, session_id: str = ""):
    sid = session_id or (req.session_id if req else "")
    if sid in sessions:
        sessions[sid]["conversation"] = []
    return {"status": "ok", "session_id": sid}


@app.post("/api/report")
async def explain_report(request: Request):
    report_payload = await _parse_single_report_request(request)

    explainer = ReportExplainer()
    result = explainer.explain(report_payload)

    audit = AuditLogger()
    if isinstance(result, dict) and "error" not in result:
        audit.log_report_analysis(
            report_text=_payload_preview(report_payload, limit=300),
            urgency_level=result.get("urgency", {}).get("level", "Unknown"),
            terms_found=len(result.get("medical_terms", [])),
            key_findings=len(result.get("key_findings", [])),
            specialist=result.get("specialist", ""),
            model_used=get_openai_model(),
        )

    return result


@app.post("/api/skin/predict")
async def predict_skin(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content))

    checker = ImageQualityChecker()
    quality = checker.check(image)

    classifier = SkinClassifier()
    if not classifier.is_loaded:
        raise HTTPException(status_code=400, detail="Skin model not trained. Run: python train_model.py")

    results = classifier.predict(image, top_k=3)

    return {
        "quality": quality,
        "predictions": results,
    }


@app.get("/api/admin/stats")
async def admin_stats():
    stats = AuditLogger.get_stats()
    vdb = get_vdb()
    stats["vector_db"] = vdb.get_stats()
    return stats


@app.post("/api/drugs/check")
async def check_drug_interactions(req: DrugCheckRequest):
    from openai import OpenAI

    api_key = get_openai_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")

    if len(req.drugs) < 2:
        raise HTTPException(status_code=400, detail="Enter at least 2 medications to check interactions")

    drug_list = ", ".join(req.drugs)
    lang_instruction = f"Respond entirely in {req.language}." if req.language != "English" else ""

    prompt = f"""
You are a clinical pharmacist AI. Analyze the following medications for interactions.

Medications: {drug_list}

{lang_instruction}

Return ONLY valid JSON in this exact structure:
{{
  "drugs_analyzed": ["list of drug names"],
  "interactions": [
    {{
      "drug_pair": "Drug A + Drug B",
      "severity": "Major/Moderate/Minor/None",
      "description": "What happens when these are taken together",
      "mechanism": "How the interaction works pharmacologically",
      "recommendation": "What the patient should do"
    }}
  ],
  "overall_risk": "High/Moderate/Low/Safe",
  "overall_summary": "1-2 sentence plain-language summary of all interactions",
  "warnings": ["Any critical warnings"],
  "advice": "General advice for the patient"
}}

Be thorough. Check ALL possible pairs. If no interaction exists between a pair, still list it with severity "None".
IMPORTANT: This is educational only. Always recommend consulting a pharmacist or doctor.
""".strip()

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=get_openai_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )

    content = response.choices[0].message.content or ""
    parsed = _extract_json_from_text(content)
    return parsed if parsed is not None else {"raw_response": content}


@app.post("/api/report/compare")
async def compare_reports(request: Request):
    parsed = await _parse_compare_request(request)

    explainer = ReportExplainer()
    result = explainer.compare(parsed["report1"], parsed["report2"])
    return result


@app.get("/api/languages")
async def get_languages():
    return {"languages": SUPPORTED_LANGUAGES}


@app.get("/api/health")
async def health():
    issues = validate_config()
    return {
        "status": "ok" if not issues else "degraded",
        "openai_configured": bool(get_openai_key()),
        "model": get_openai_model(),
        "issues": issues,
    }