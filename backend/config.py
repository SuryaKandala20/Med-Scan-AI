"""
config.py — Load and validate environment configuration

Reads API keys from .env file and validates they exist.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try loading from environment directly (e.g., Streamlit Cloud secrets)
    load_dotenv()


def get_openai_key() -> str:
    """Get OpenAI API key or raise clear error."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-xxxx"):
        return ""
    return key


def get_openai_model() -> str:
    """Get OpenAI model name."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_kaggle_credentials() -> tuple:
    """Get Kaggle username and API key."""
    username = os.getenv("KAGGLE_USERNAME", "")
    key = os.getenv("KAGGLE_KEY", "")
    return username, key


def validate_config():
    """Check all required config is present. Returns list of issues."""
    issues = []

    if not get_openai_key():
        issues.append(
            "❌ OPENAI_API_KEY not set. Get one at https://platform.openai.com/api-keys"
        )

    username, key = get_kaggle_credentials()
    if not username or not key or username == "your_kaggle_username":
        issues.append(
            "⚠️ Kaggle credentials not set. Needed for dataset download. "
            "Get them at https://www.kaggle.com/settings → API section"
        )

    return issues
