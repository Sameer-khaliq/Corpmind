"""corpmind.config now builds a module-level `settings` singleton at import
time (per the plan's checkpoint: `from corpmind.config import settings`).
That means the first test module that imports anything from corpmind.config
triggers Settings() construction before any test function — and therefore
before any monkeypatch — runs. Real dev/CI should have a real .env; this
just keeps `uv run pytest` runnable on a fresh clone with no .env yet.
"""
import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
