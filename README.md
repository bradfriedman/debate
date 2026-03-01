# Triad Debate

Multi-agent debate platform where three LLMs argue a topic in a structured, blind format.

## Quick Start

```bash
uv sync
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."; uv run streamlit run src/frontend/app.py
```

**Mac/Linux:**
```bash
PYTHONPATH=. uv run streamlit run src/frontend/app.py
```

Requires a `.env` file with API keys:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```
