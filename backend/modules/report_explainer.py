"""
modules/report_explainer.py — Generic multimodal medical report explanation using OpenAI

Updated:
- adds detailed_key_findings for a second, more educational section
- keeps report-type detection generic
- tells the model not to invent exact numeric thresholds from a qualitative report
"""

import json
import re
from typing import Any, Dict, List

from openai import OpenAI
from config import get_openai_key, get_openai_model


REPORT_SYSTEM_PROMPT = """
You are a careful medical report explainer.

The user may provide:
- lab reports
- ultrasound reports
- radiology reports
- pathology reports
- ECG / EKG printouts
- prescriptions
- discharge summaries
- mixed multi-page documents
- scanned or noisy pages

Important:
- Do NOT rely on hardcoded report types.
- Infer what each page is by looking at both the image and the extracted text.
- Read the ENTIRE submission, including all pages.
- Some pages may be text-heavy, some may be image-heavy, and some may be noisy.
- If a page is unclear, say so calmly and avoid making up details.
- If the document contains multiple report types, explain all major sections.
- Use the image itself as a primary source when OCR text looks noisy or incomplete.
- Never invent a diagnosis that is not supported by the report.
- Use simple, calm, patient-friendly language.
- Clearly explain what looks normal, abnormal, or worth follow-up.
- Always recommend review with a doctor.

Very important for the educational section:
- Add a second layer of explanation using general medical context, but clearly separate
  what is directly shown in the report from what is general educational information.
- If a report is qualitative, do NOT pretend it gives an exact measurement.
- Example: if a liver ultrasound says fatty infiltration / steatosis, you may explain that
  fatty liver is generally associated with excess fat in the liver, but if the report does
  not provide a quantitative fat fraction, say that the exact amount cannot be determined
  from this report alone.
- For each major abnormal or clinically important finding, explain:
  1) what it means
  2) why it matters
  3) what people commonly do next
  4) what part is report-specific vs general educational context

Return ONLY valid JSON in this exact structure:
{
  "summary": "Plain-language explanation of the whole document",
  "urgency": {
    "level": "Low/Moderate/High",
    "message": "Short explanation of urgency"
  },
  "key_findings": [
    {
      "finding": "Name of finding",
      "status": "Normal/Abnormal/Needs Follow-up",
      "explanation": "Simple explanation"
    }
  ],
  "detailed_key_findings": [
    {
      "finding": "Name of finding",
      "report_specific_summary": "What this specific report says",
      "what_it_means": "Plain-language explanation of the condition/finding",
      "why_it_matters": "Why this can matter medically",
      "general_context": "General educational context, including typical thresholds only if appropriate",
      "what_to_do_next": [
        "Action 1",
        "Action 2",
        "Action 3"
      ],
      "red_flags": [
        "When to seek more urgent medical advice"
      ]
    }
  ],
  "medical_terms": [
    {
      "term": "Medical term",
      "explanation": "Simple explanation"
    }
  ],
  "questions_for_doctor": [
    "Question 1",
    "Question 2",
    "Question 3"
  ],
  "specialist": "Type of doctor to consult"
}
"""

COMPARE_SYSTEM_PROMPT = """
You are a careful medical report comparison assistant.

You will receive two medical report submissions. Each submission may contain:
- text notes
- OCR text
- multi-page mixed reports

Your job:
- compare the two reports
- identify meaningful changes
- explain them in simple language
- do not invent findings
- if a section is unclear, say so briefly

Return ONLY valid JSON in this exact structure:
{
  "summary": "2-3 sentence overview of important changes",
  "overall_trend": "Improved/Stable/Worsened/Mixed",
  "changes": [
    {
      "parameter": "Name of changed finding",
      "old_value": "Value in report 1",
      "new_value": "Value in report 2",
      "direction": "Improved/Worsened/Stable/New Finding",
      "significance": "High/Medium/Low",
      "explanation": "Plain-language explanation"
    }
  ],
  "unchanged": ["Item 1", "Item 2"],
  "concerns": ["Concern 1", "Concern 2"],
  "positive_changes": ["Positive 1", "Positive 2"],
  "follow_up": "Suggested next step",
  "specialist": "Type of doctor to discuss with"
}
"""


class ReportExplainer:
    def __init__(self):
        key = get_openai_key()
        self.client = OpenAI(api_key=key) if key else None
        self.model = get_openai_model()

    def _extract_json(self, content: str) -> dict:
        if not content:
            return {}

        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        fenced_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
        if fenced_match:
            try:
                return json.loads(fenced_match.group(1))
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"summary": content.strip(), "raw": True}

    def _normalize_explain_response(self, data: dict) -> dict:
        if not isinstance(data, dict):
            data = {"summary": str(data)}

        data.setdefault("summary", "No summary available.")
        data.setdefault(
            "urgency",
            {"level": "Moderate", "message": "Please review this report with your doctor."},
        )
        data.setdefault("key_findings", [])
        data.setdefault("detailed_key_findings", [])
        data.setdefault("medical_terms", [])
        data.setdefault(
            "questions_for_doctor",
            [
                "Can you explain the main findings in this report?",
                "Are any of these findings abnormal or in need of follow-up?",
                "Do I need any additional tests or repeat studies?",
            ],
        )
        data.setdefault("specialist", "Primary care doctor")
        return data

    def _normalize_compare_response(self, data: dict) -> dict:
        if not isinstance(data, dict):
            data = {"summary": str(data)}

        data.setdefault("summary", "No comparison summary available.")
        data.setdefault("overall_trend", "Mixed")
        data.setdefault("changes", [])
        data.setdefault("unchanged", [])
        data.setdefault("concerns", [])
        data.setdefault("positive_changes", [])
        data.setdefault("follow_up", "Please review both reports with your doctor.")
        data.setdefault("specialist", "Primary care doctor")
        return data

    def _build_explain_user_content(self, report_payload: dict) -> List[dict]:
        text_notes = report_payload.get("text_notes", []) or []
        pages = report_payload.get("pages", []) or []
        source_name = report_payload.get("source_name", "user_submission")

        content: List[dict] = []

        intro = f"""
Please explain this medical report submission.

Source name: {source_name}
Number of uploaded pages/images: {len(pages)}
Number of pasted text notes: {len(text_notes)}

Instructions:
- Infer the type of each page from the content itself.
- Do not ignore later pages.
- If the submission contains different kinds of medical pages, explain all important sections.
- Use page images as the main source when OCR text is noisy.
- Return JSON only.
- Populate both:
  1) key_findings for short explanations
  2) detailed_key_findings for deeper educational explanations
""".strip()

        content.append({"type": "text", "text": intro})

        for idx, note in enumerate(text_notes, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"Pasted text note {idx}:\n{note}",
                }
            )

        for page in pages:
            page_number = page.get("page_number", "?")
            page_text = (page.get("page_text") or "").strip()
            page_label = f"Document page {page_number}"

            if page_text:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"{page_label}\n"
                            f"Extracted text from this page. It may include native PDF text and/or OCR text.\n"
                            f"If this text conflicts with the image, trust the image more.\n\n"
                            f"{page_text}"
                        ),
                    }
                )
            else:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"{page_label}\n"
                            "No reliable extracted text was available for this page. "
                            "Please inspect the page image directly."
                        ),
                    }
                )

            image_data_url = page.get("image_data_url")
            if image_data_url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": "high",
                        },
                    }
                )

        return content

    def _build_text_only_explain_messages(self, report_payload: dict) -> List[dict]:
        text_notes = report_payload.get("text_notes", []) or []
        pages = report_payload.get("pages", []) or []
        source_name = report_payload.get("source_name", "user_submission")

        combined_parts: List[str] = [
            f"Source name: {source_name}",
            "The document may contain multiple page types. Infer the page type from the content itself.",
            "Explain the full submission, not just the first section.",
            "Populate both key_findings and detailed_key_findings.",
        ]

        for idx, note in enumerate(text_notes, start=1):
            combined_parts.append(f"=== TEXT NOTE {idx} ===\n{note}")

        for page in pages:
            page_number = page.get("page_number", "?")
            page_text = (page.get("page_text") or "").strip()
            combined_parts.append(f"=== PAGE {page_number} ===\n{page_text or '[No extracted text]'}")

        user_prompt = "\n\n".join(combined_parts).strip()

        return [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _create_multimodal_completion(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=2600,
        )
        return response.choices[0].message.content or ""

    def explain(self, report_payload: Any) -> dict:
        if not self.client:
            return {"error": "OpenAI API key not configured. Add it to .env file."}

        if isinstance(report_payload, str):
            report_payload = {
                "source_name": "text_input",
                "file_kind": "text",
                "text_notes": [report_payload],
                "pages": [],
                "combined_text": report_payload,
            }

        if not isinstance(report_payload, dict):
            return {"error": "Invalid report payload."}

        if not report_payload.get("text_notes") and not report_payload.get("pages"):
            return {"error": "No report content provided."}

        multimodal_messages = [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_explain_user_content(report_payload)},
        ]

        try:
            content = self._create_multimodal_completion(multimodal_messages)
            parsed = self._extract_json(content)
            return self._normalize_explain_response(parsed)

        except Exception as e:
            error_text = str(e).lower()

            vision_related = (
                "image" in error_text
                or "vision" in error_text
                or "content type" in error_text
                or "invalid image" in error_text
                or "unsupported" in error_text
            )

            if not vision_related:
                return {"error": f"API error: {str(e)}"}

            try:
                fallback_messages = self._build_text_only_explain_messages(report_payload)
                content = self._create_multimodal_completion(fallback_messages)
                parsed = self._extract_json(content)
                parsed = self._normalize_explain_response(parsed)

                summary = parsed.get("summary", "")
                note = " This explanation used text/OCR fallback because direct image analysis was unavailable."
                parsed["summary"] = (summary + note).strip()

                return parsed
            except Exception as second_error:
                return {"error": f"API error: {str(second_error)}"}

    def _payload_to_compare_text(self, payload: dict, label: str) -> str:
        parts: List[str] = [f"=== {label} ==="]

        for idx, note in enumerate(payload.get("text_notes", []), start=1):
            parts.append(f"TEXT NOTE {idx}:\n{note}")

        for page in payload.get("pages", []):
            page_number = page.get("page_number", "?")
            page_text = (page.get("page_text") or "").strip()
            parts.append(f"PAGE {page_number}:\n{page_text or '[No extracted text]'}")

        return "\n\n".join(parts).strip()

    def compare(self, report1_payload: dict, report2_payload: dict) -> dict:
        if not self.client:
            return {"error": "OpenAI API key not configured. Add it to .env file."}

        report1_text = self._payload_to_compare_text(report1_payload, "REPORT 1")
        report2_text = self._payload_to_compare_text(report2_payload, "REPORT 2")

        user_prompt = f"""
Please compare these two medical report submissions.

{report1_text}

{report2_text}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2200,
            )

            content = response.choices[0].message.content or ""
            parsed = self._extract_json(content)
            return self._normalize_compare_response(parsed)

        except Exception as e:
            return {"error": f"API error: {str(e)}"}