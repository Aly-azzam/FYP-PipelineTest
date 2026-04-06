"""Load backend configuration and secrets from environment variables (`.env`)."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def get_env(name: str, required: bool = True) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        if required:
            raise ValueError(
                f"Missing environment variable {name!r}. "
                f"Define it in the project root .env file."
            )
        return None

    value = raw.strip()
    if not value:
        if required:
            raise ValueError(
                f"Environment variable {name!r} is empty. "
                f"Set a non-empty value in the project root .env file."
            )
        return None

    return value


def get_gemini_api_key() -> str:
    key = get_env("GEMINI_API_KEY", required=True)
    if key is None:
        raise ValueError(
            "GEMINI_API_KEY is missing or empty. Add GEMINI_API_KEY=... to .env."
        )
    return key