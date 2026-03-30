"""
modules/llm_chat.py — OpenAI GPT-4o powered medical chat engine

This is the REAL chatbot — it uses GPT-4o to:
1. Understand natural language symptom descriptions
2. Ask intelligent follow-up questions (like a doctor would)
3. Produce structured assessments with conditions, treatments, doctor referrals
4. Maintain conversation memory across turns

The LLM is constrained by a medical system prompt with guardrails:
- Never gives definitive diagnoses
- Always recommends consulting a doctor
- Specifies which type of specialist to see
- Provides triage urgency levels
"""

import json
import time
import traceback
from openai import OpenAI
from config import get_openai_key, get_openai_model
from modules.vector_db import MedicalVectorDB
from modules.audit_logger import AuditLogger

# ──────────────────────────────────────────────
# Medical System Prompt — This is the "brain"
# ──────────────────────────────────────────────
MEDICAL_SYSTEM_PROMPT = """You are MedScan AI, an empathetic and thorough medical assistant chatbot. You help users understand their symptoms and guide them toward appropriate care.

## YOUR ROLE
- You are an EDUCATIONAL medical assistant, NOT a doctor
- You NEVER give definitive diagnoses — only possible conditions
- You ALWAYS recommend consulting a real doctor
- You specify WHICH TYPE of specialist to see

## CONVERSATION FLOW
Follow this exact flow:

### Phase 1: GATHERING (first 2-3 exchanges)
When the user describes symptoms, ask targeted follow-up questions. Ask 3-4 questions per turn, covering:
- Severity (1-10 scale)
- Duration (when did it start?)
- Location (where exactly?)
- Character (sharp, dull, burning, throbbing?)
- Triggers (what makes it better/worse?)
- Associated symptoms (anything else going on?)
- Age, sex (if not provided)
- Medical history, medications
- Onset (sudden vs gradual)

Be conversational and empathetic, not robotic. Example:
"I'm sorry you're dealing with that. Let me ask a few things to better understand:
1. How would you rate the pain on a scale of 1-10?
2. Is it constant or does it come and go?
3. Have you noticed anything that makes it better or worse?"

### Phase 2: ASSESSMENT (after gathering enough info)
When you have enough information (usually after 2-3 exchanges), provide your assessment. You MUST respond with a JSON block wrapped in ```json``` markers containing:

```json
{
  "assessment": true,
  "intro": "Brief empathetic intro summarizing what you heard",
  "conditions": [
    {
      "rank": 1,
      "name": "Condition Name",
      "likelihood": "High/Moderate/Low",
      "description": "2-3 sentence explanation of this condition in simple terms",
      "matching_symptoms": ["symptom 1", "symptom 2"],
      "why": "Brief explanation of why this matches their symptoms"
    }
  ],
  "triage": {
    "level": "Emergency/Urgent/Same-day/Routine",
    "message": "What they should do about timing"
  },
  "treatments": [
    "Specific actionable care tip 1",
    "Specific actionable care tip 2",
    "Specific actionable care tip 3",
    "Specific actionable care tip 4",
    "Specific actionable care tip 5"
  ],
  "doctor_referral": {
    "specialty": "Neurologist/Cardiologist/Dermatologist/GP/etc",
    "icon": "🧠/❤️/🩺/etc",
    "urgency": "See within X days/hours",
    "message": "Please consult a [Specialty] before following any of the above suggestions. They can provide proper examination and definitive diagnosis."
  },
  "emergency_warning": "Only include this if symptoms could indicate emergency. E.g., 'Seek immediate emergency care if you experience X, Y, or Z'",
  "follow_up": "A caring follow-up question or offer"
}
```

ALWAYS include exactly 3 conditions ranked by likelihood.
ALWAYS include 5-6 specific treatment/care tips.
ALWAYS include a doctor referral with specific specialty.

### Phase 3: FOLLOW-UP
After assessment, be ready to:
- Explain conditions in more detail
- Answer questions about the assessment
- Discuss different conditions if asked
- Start new analysis for different symptoms

## SAFETY GUARDRAILS
- If symptoms suggest EMERGENCY (chest pain + arm pain, severe breathing difficulty, stroke symptoms, severe allergic reaction): IMMEDIATELY flag as Emergency triage and tell them to call 911/emergency services FIRST, then provide the assessment
- NEVER prescribe specific medications with dosages — say "your doctor may consider..." or "over-the-counter options include..."
- NEVER say "you have X" — say "this could possibly be X" or "your symptoms are consistent with X"
- For mental health: be extra gentle, always suggest professional support
- For children: note that symptoms may present differently and recommend pediatric care

## TONE
- Warm, empathetic, professional
- Use simple language (explain medical terms)
- Don't be overly formal or use bullet points in conversation (save structure for the assessment JSON)
- Show you're listening by referencing what they told you

## IMPORTANT
- If the user's message is just casual chat (greeting, thank you, etc.), respond naturally without medical analysis
- If the user wants to discuss a DIFFERENT set of symptoms, start fresh with gathering
- You can provide the assessment earlier if the user explicitly asks ("what do you think it could be?")
"""


class LLMChat:
    """OpenAI GPT-4o powered medical chat with RAG + audit logging."""

    def __init__(self, session_id: str = None):
        self.api_key = get_openai_key()
        self.model = get_openai_model()
        self.client = None
        self.is_configured = False

        # Initialize vector DB for RAG
        self.vector_db = MedicalVectorDB()
        try:
            self.vector_db.initialize()
        except Exception as e:
            print(f"Vector DB init warning: {e}")

        # Initialize audit logger
        self.audit = AuditLogger(session_id=session_id)

        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.is_configured = True
            except Exception as e:
                print(f"OpenAI init error: {e}")

    def chat(self, messages: list) -> dict:
        """
        Send conversation to GPT-4o with RAG context and get response.
        All interactions are audit-logged.
        """
        if not self.is_configured:
            return {
                "text": None, "assessment": None,
                "error": "OpenAI API key not configured. Please add your key to the .env file.",
            }

        try:
            start_time = time.time()

            # ── RAG: Retrieve relevant medical context ──
            # Extract latest user message for vector search
            last_user_msg = ""
            for m in reversed(messages):
                if m["role"] == "user":
                    last_user_msg = m["content"]
                    break

            rag_context = ""
            if last_user_msg:
                rag_context = self.vector_db.get_context_for_symptoms(last_user_msg)

            # Build system prompt with RAG context injected
            system_prompt = MEDICAL_SYSTEM_PROMPT
            if rag_context:
                system_prompt += f"""

## MEDICAL KNOWLEDGE BASE (retrieved for this patient's symptoms)
Use this verified medical information to inform your response. Prioritize this 
over general knowledge. Cite specific details from here when relevant.

{rag_context}
"""

            # ── Call OpenAI ──
            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.3,
                max_tokens=2000,
            )

            content = response.choices[0].message.content
            latency_ms = int((time.time() - start_time) * 1000)
            tokens = response.usage.total_tokens if response.usage else None

            # ── Log user message ──
            self.audit.log_message(
                role="user", content=last_user_msg,
                model_used=self.model, latency_ms=latency_ms, tokens_used=tokens,
            )

            # ── Parse assessment if present ──
            assessment = self._extract_assessment(content)

            if assessment:
                # Log bot assessment
                self.audit.log_message(
                    role="bot", content=assessment.get("intro", ""),
                    has_assessment=True, model_used=self.model,
                    latency_ms=latency_ms, tokens_used=tokens,
                )

                # Log structured assessment
                conditions = assessment.get("conditions", [])
                top_cond = conditions[0] if conditions else {}
                doctor = assessment.get("doctor_referral", {})
                self.audit.log_assessment(
                    symptoms=[],  # extracted from conversation
                    conditions=conditions,
                    top_condition=top_cond.get("name", "Unknown"),
                    triage_level=assessment.get("triage", {}).get("level", "Routine"),
                    treatments=assessment.get("treatments", []),
                    specialist=doctor.get("specialty", ""),
                    confidence=top_cond.get("likelihood", ""),
                    model_used=self.model,
                    raw_response=content,
                )

                return {"text": assessment.get("intro", ""), "assessment": assessment, "error": None}
            else:
                # Log bot follow-up question
                self.audit.log_message(
                    role="bot", content=content,
                    has_assessment=False, model_used=self.model,
                    latency_ms=latency_ms, tokens_used=tokens,
                )
                return {"text": content, "assessment": None, "error": None}

        except Exception as e:
            # Log the error
            self.audit.log_error(
                error_type="api_error", module="llm_chat",
                message=str(e), stack_trace=traceback.format_exc(),
            )

            error_msg = str(e)
            if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                return {"text": None, "assessment": None, "error": "Invalid OpenAI API key. Check .env file."}
            elif "rate_limit" in error_msg.lower():
                return {"text": None, "assessment": None, "error": "Rate limited. Wait a moment and try again."}
            else:
                return {"text": None, "assessment": None, "error": f"API error: {error_msg}"}

    def _extract_assessment(self, content: str) -> dict:
        """Try to extract JSON assessment from the response."""
        import re

        # Look for JSON block in ```json ... ``` markers
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if data.get("assessment") is True:
                    return data
            except json.JSONDecodeError:
                pass

        # Also try finding raw JSON (sometimes model omits markers)
        try:
            # Find the first { ... } block that looks like our assessment
            brace_match = re.search(r'\{[^{}]*"assessment"\s*:\s*true.*\}', content, re.DOTALL)
            if brace_match:
                # Find the matching closing brace
                start = content.index(brace_match.group(0)[:20])
                depth = 0
                end = start
                for i in range(start, len(content)):
                    if content[i] == '{':
                        depth += 1
                    elif content[i] == '}':
                        depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                data = json.loads(content[start:end])
                if data.get("assessment") is True:
                    return data
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def get_greeting(self) -> str:
        """Return the initial greeting (no API call needed)."""
        return (
            "Hi! I'm **MedScan AI**, your medical assistant. "
            "I'm powered by AI to help you understand your symptoms "
            "and guide you toward the right care.\n\n"
            "**Tell me what's bothering you** — describe how you're feeling "
            "in your own words, and I'll ask some follow-up questions "
            "before giving you my assessment.\n\n"
            "*Remember: I'm an educational tool, not a real doctor. "
            "Always consult a healthcare professional for actual medical advice.*"
        )
