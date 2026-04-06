"""Load backend configuration and secrets from environment variables (`.env`)."""

import os
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_DIR / ".env")


def get_env(name: str, required: bool = True) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        if required:
            raise ValueError(
                f"Missing environment variable {name!r}. "
                f"Define it in backend/.env (see project docs)."
            )
        return None
    value = raw.strip()
    if not value:
        if required:
            raise ValueError(
                f"Environment variable {name!r} is empty. "
                f"Set a non-empty value in backend/.env."
            )
        return None
    return value


def get_gemini_api_key() -> str:
    try:
        key = get_env("GEMINI_API_KEY", required=True)
    except ValueError as exc:
        raise ValueError(
            "GEMINI_API_KEY is missing or empty. "
            "Add GEMINI_API_KEY=... to backend/.env."
        ) from exc
    assert key is not None
    return key
