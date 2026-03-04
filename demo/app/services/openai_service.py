"""
OpenAI API wrapper with Redis-backed embedding cache and Prometheus metrics.
"""

import asyncio
import hashlib
import struct
import time
from typing import List, Optional

import openai as openai_mod
import structlog
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram

from app.config import settings
from app.database import redis_client

logger = structlog.get_logger()

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

OPENAI_CALLS = Counter("openai_api_calls_total", "Total OpenAI API calls", ["model", "type"])
OPENAI_LATENCY = Histogram("openai_api_duration_seconds", "OpenAI API latency", ["model"])


def truncate_text(text: str, max_chars: int = 3000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _pack_vector(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(data: bytes) -> List[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


async def get_openai_embedding(text: str) -> List[float]:
    text = truncate_text(text, 2000)
    cache_key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"

    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return _unpack_vector(cached)

    OPENAI_CALLS.labels(settings.embedding_model, "embedding").inc()
    start = time.perf_counter()
    response = await openai_client.embeddings.create(model=settings.embedding_model, input=text)
    OPENAI_LATENCY.labels(settings.embedding_model).observe(time.perf_counter() - start)
    vec = response.data[0].embedding

    if redis_client:
        await redis_client.set(cache_key, _pack_vector(vec), ex=86400)

    return vec


async def openai_chat(
    messages: list,
    json_mode: bool = False,
    max_retries: int = 3,
    model: Optional[str] = None,
) -> Optional[str]:
    _model = model or settings.openai_model
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(max_retries):
        try:
            OPENAI_CALLS.labels(_model, "chat").inc()
            start = time.perf_counter()
            resp = await openai_client.chat.completions.create(
                model=_model, messages=messages, **kwargs
            )
            OPENAI_LATENCY.labels(_model).observe(time.perf_counter() - start)
            return resp.choices[0].message.content.strip()
        except (openai_mod.RateLimitError, openai_mod.APIConnectionError) as e:
            logger.warning("openai_retry", attempt=attempt + 1, error=str(e))
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error("openai_error", error=str(e))
            return None
    return None
