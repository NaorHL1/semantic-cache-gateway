"""ASGI entry point: `granian --interface asgi src.main:app` (ADR-0003)."""

from fastapi import FastAPI

app = FastAPI(title="Semantic Cache Gateway", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
