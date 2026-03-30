"""
modules/audit_logger.py — SQLite Audit & Logging System

Tracks EVERYTHING that happens in the app:
- Every chat message (user + bot)
- Every symptom assessment with conditions returned
- Every skin image prediction
- Every report analysis
- User feedback (correct/incorrect)
- Session tracking
- Error logging

Database: data/medscan_audit.db (SQLite — zero config, runs locally)

Tables:
  sessions        — one row per user session
  chat_messages   — every message in every conversation
  assessments     — structured assessment results
  skin_predictions— skin image analysis results
  report_analyses — report explainer results
  feedback        — user feedback on predictions
  error_log       — all errors/failures
"""

import sqlite3
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path("data/medscan_audit.db")


def _ensure_db_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db():
    """Context manager for database connections."""
    _ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            -- Sessions table: one row per app session
            CREATE TABLE IF NOT EXISTS sessions (
                session_id      TEXT PRIMARY KEY,
                started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at        TIMESTAMP,
                ip_hash         TEXT,           -- hashed for privacy
                total_messages  INTEGER DEFAULT 0,
                total_assessments INTEGER DEFAULT 0,
                metadata        TEXT            -- JSON extra info
            );

            -- Chat messages: every user and bot message
            CREATE TABLE IF NOT EXISTS chat_messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role            TEXT NOT NULL,   -- 'user' or 'bot'
                content         TEXT NOT NULL,
                detected_symptoms TEXT,           -- JSON list of extracted symptoms
                has_assessment  BOOLEAN DEFAULT 0,
                tokens_used     INTEGER,
                model_used      TEXT,
                latency_ms      INTEGER,         -- response time in ms
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- Assessments: structured results from symptom analysis
            CREATE TABLE IF NOT EXISTS assessments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symptoms        TEXT NOT NULL,    -- JSON: all symptoms collected
                conditions      TEXT NOT NULL,    -- JSON: ranked conditions
                top_condition   TEXT NOT NULL,    -- name of #1 condition
                triage_level    TEXT NOT NULL,    -- Emergency/Urgent/Same-day/Routine
                treatments      TEXT,             -- JSON: treatment tips given
                specialist      TEXT,             -- recommended doctor type
                confidence      TEXT,             -- High/Moderate/Low
                patient_age     INTEGER,
                patient_sex     TEXT,
                duration        TEXT,
                severity        TEXT,
                model_used      TEXT,
                raw_response    TEXT,             -- full LLM response for debugging
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- Skin predictions: image-based analysis results
            CREATE TABLE IF NOT EXISTS skin_predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_hash      TEXT,             -- SHA256 of image (not the image itself)
                image_size      TEXT,             -- WxH
                quality_passed  BOOLEAN,
                quality_issues  TEXT,             -- JSON list
                top_prediction  TEXT,
                top_confidence  REAL,
                all_predictions TEXT,             -- JSON: all ranked predictions
                triage_level    TEXT,
                model_version   TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- Report analyses: medical report explanations
            CREATE TABLE IF NOT EXISTS report_analyses (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                report_hash     TEXT,             -- SHA256 of report text
                report_length   INTEGER,
                urgency_level   TEXT,
                terms_found     INTEGER,
                key_findings    INTEGER,
                specialist      TEXT,
                model_used      TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- User feedback on predictions
            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prediction_type TEXT NOT NULL,    -- 'chat_assessment', 'skin', 'report'
                prediction_id   INTEGER,          -- FK to relevant table
                rating          TEXT NOT NULL,     -- 'accurate', 'partially', 'inaccurate'
                user_comment    TEXT,
                correct_condition TEXT,            -- what user says it actually was
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- Error log
            CREATE TABLE IF NOT EXISTS error_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_type      TEXT NOT NULL,    -- 'api_error', 'model_error', 'validation_error'
                module          TEXT,             -- which module errored
                message         TEXT NOT NULL,
                stack_trace     TEXT,
                context         TEXT              -- JSON extra context
            );

            -- Create indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_assessments_session ON assessments(session_id);
            CREATE INDEX IF NOT EXISTS idx_assessments_condition ON assessments(top_condition);
            CREATE INDEX IF NOT EXISTS idx_assessments_triage ON assessments(triage_level);
            CREATE INDEX IF NOT EXISTS idx_skin_pred ON skin_predictions(top_prediction);
            CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(prediction_type, rating);
            CREATE INDEX IF NOT EXISTS idx_errors ON error_log(error_type, timestamp);
        """)


class AuditLogger:
    """Main audit logging class."""

    def __init__(self, session_id: str = None):
        init_db()
        self.session_id = session_id or str(uuid.uuid4())
        self._ensure_session()

    def _ensure_session(self):
        """Create session record if it doesn't exist."""
        with get_db() as conn:
            existing = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (self.session_id,)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO sessions (session_id) VALUES (?)",
                    (self.session_id,),
                )

    # ── Chat Logging ──

    def log_message(self, role: str, content: str, detected_symptoms: list = None,
                    has_assessment: bool = False, tokens_used: int = None,
                    model_used: str = None, latency_ms: int = None):
        """Log a chat message."""
        with get_db() as conn:
            conn.execute(
                """INSERT INTO chat_messages 
                   (session_id, role, content, detected_symptoms, has_assessment, 
                    tokens_used, model_used, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, role, content,
                    json.dumps(detected_symptoms) if detected_symptoms else None,
                    has_assessment, tokens_used, model_used, latency_ms,
                ),
            )
            conn.execute(
                "UPDATE sessions SET total_messages = total_messages + 1 WHERE session_id = ?",
                (self.session_id,),
            )

    def log_assessment(self, symptoms: list, conditions: list, top_condition: str,
                       triage_level: str, treatments: list = None, specialist: str = None,
                       confidence: str = None, patient_age: int = None,
                       patient_sex: str = None, duration: str = None,
                       severity: str = None, model_used: str = None,
                       raw_response: str = None) -> int:
        """Log an assessment. Returns the assessment ID."""
        with get_db() as conn:
            cursor = conn.execute(
                """INSERT INTO assessments
                   (session_id, symptoms, conditions, top_condition, triage_level,
                    treatments, specialist, confidence, patient_age, patient_sex,
                    duration, severity, model_used, raw_response)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id,
                    json.dumps(symptoms), json.dumps(conditions),
                    top_condition, triage_level,
                    json.dumps(treatments) if treatments else None,
                    specialist, confidence, patient_age, patient_sex,
                    duration, severity, model_used, raw_response,
                ),
            )
            conn.execute(
                "UPDATE sessions SET total_assessments = total_assessments + 1 WHERE session_id = ?",
                (self.session_id,),
            )
            return cursor.lastrowid

    # ── Skin Prediction Logging ──

    def log_skin_prediction(self, image_bytes: bytes, image_size: str,
                            quality_passed: bool, quality_issues: list,
                            predictions: list, model_version: str = None) -> int:
        """Log a skin image prediction."""
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        top = predictions[0] if predictions else {}

        with get_db() as conn:
            cursor = conn.execute(
                """INSERT INTO skin_predictions
                   (session_id, image_hash, image_size, quality_passed, quality_issues,
                    top_prediction, top_confidence, all_predictions, triage_level, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, image_hash, image_size,
                    quality_passed, json.dumps(quality_issues),
                    top.get("class_name", ""), top.get("confidence", 0),
                    json.dumps(predictions),
                    top.get("triage", ""), model_version,
                ),
            )
            return cursor.lastrowid

    # ── Report Analysis Logging ──

    def log_report_analysis(self, report_text: str, urgency_level: str,
                            terms_found: int, key_findings: int,
                            specialist: str = None, model_used: str = None) -> int:
        """Log a report analysis."""
        report_hash = hashlib.sha256(report_text.encode()).hexdigest()[:16]
        with get_db() as conn:
            cursor = conn.execute(
                """INSERT INTO report_analyses
                   (session_id, report_hash, report_length, urgency_level,
                    terms_found, key_findings, specialist, model_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, report_hash, len(report_text),
                    urgency_level, terms_found, key_findings,
                    specialist, model_used,
                ),
            )
            return cursor.lastrowid

    # ── Feedback ──

    def log_feedback(self, prediction_type: str, prediction_id: int,
                     rating: str, user_comment: str = None,
                     correct_condition: str = None):
        """Log user feedback on a prediction."""
        with get_db() as conn:
            conn.execute(
                """INSERT INTO feedback
                   (session_id, prediction_type, prediction_id, rating,
                    user_comment, correct_condition)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, prediction_type, prediction_id,
                    rating, user_comment, correct_condition,
                ),
            )

    # ── Error Logging ──

    def log_error(self, error_type: str, module: str, message: str,
                  stack_trace: str = None, context: dict = None):
        """Log an error."""
        with get_db() as conn:
            conn.execute(
                """INSERT INTO error_log
                   (session_id, error_type, module, message, stack_trace, context)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, error_type, module, message,
                    stack_trace, json.dumps(context) if context else None,
                ),
            )

    # ── Analytics Queries ──

    @staticmethod
    def get_stats() -> dict:
        """Get overall app statistics."""
        with get_db() as conn:
            stats = {}
            stats["total_sessions"] = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            stats["total_messages"] = conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]
            stats["total_assessments"] = conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
            stats["total_skin_predictions"] = conn.execute("SELECT COUNT(*) FROM skin_predictions").fetchone()[0]
            stats["total_report_analyses"] = conn.execute("SELECT COUNT(*) FROM report_analyses").fetchone()[0]
            stats["total_feedback"] = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            stats["total_errors"] = conn.execute("SELECT COUNT(*) FROM error_log").fetchone()[0]

            # Top conditions
            top_conditions = conn.execute(
                """SELECT top_condition, COUNT(*) as cnt 
                   FROM assessments GROUP BY top_condition 
                   ORDER BY cnt DESC LIMIT 10"""
            ).fetchall()
            stats["top_conditions"] = [{"condition": r[0], "count": r[1]} for r in top_conditions]

            # Triage distribution
            triage_dist = conn.execute(
                """SELECT triage_level, COUNT(*) as cnt 
                   FROM assessments GROUP BY triage_level"""
            ).fetchall()
            stats["triage_distribution"] = {r[0]: r[1] for r in triage_dist}

            # Feedback summary
            feedback_summary = conn.execute(
                """SELECT rating, COUNT(*) as cnt 
                   FROM feedback GROUP BY rating"""
            ).fetchall()
            stats["feedback_summary"] = {r[0]: r[1] for r in feedback_summary}

            # Recent errors
            recent_errors = conn.execute(
                """SELECT timestamp, error_type, module, message 
                   FROM error_log ORDER BY timestamp DESC LIMIT 5"""
            ).fetchall()
            stats["recent_errors"] = [
                {"time": r[0], "type": r[1], "module": r[2], "message": r[3]}
                for r in recent_errors
            ]

            return stats

    @staticmethod
    def get_recent_sessions(limit: int = 20) -> list:
        """Get recent sessions with summary info."""
        with get_db() as conn:
            rows = conn.execute(
                """SELECT session_id, started_at, total_messages, total_assessments
                   FROM sessions ORDER BY started_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
