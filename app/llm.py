"""Shared Groq helper: one JSON-returning call with retry and model fallback."""
import asyncio
import json
import logging
import random

import httpx

from app.config import settings

URL = "https://api.groq.com/openai/v1/chat/completions"
log = logging.getLogger("review-agent")
_REQUEST_SEMAPHORE = asyncio.Semaphore(2)


async def ask_json(prompt: str, client: httpx.AsyncClient | None = None):
    """Returns parsed JSON from the model, or None if every attempt failed."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=45)
    try:
        async with _REQUEST_SEMAPHORE:
            for model in (settings.groq_model, settings.groq_fallback_model):
                body = {"model": model, "temperature": 0.1, "max_tokens": 2000,
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}]}
                for attempt in range(3):
                    try:
                        r = await client.post(URL, json=body,
                                              headers={"Authorization": f"Bearer {settings.groq_api_key}"})
                        if r.status_code in (429, 503):
                            delay = min(30.0, 2.0 ** attempt) * random.uniform(0.75, 1.25)
                            log.warning(
                                "Groq %s on %s, retrying in %.2fs",
                                r.status_code, model, delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        r.raise_for_status()
                        return json.loads(r.json()["choices"][0]["message"]["content"])
                    except httpx.TimeoutException:
                        log.warning("Groq timed out on %s, trying next model", model)
                        break
                    except httpx.HTTPStatusError as e:
                        log.error("Groq HTTP %s on %s: %s", e.response.status_code, model, e.response.text[:300])
                        break
                    except (ValueError, KeyError, IndexError):
                        log.error("Groq returned an unreadable response on %s", model)
                        break
        return None
    finally:
        if own:
            await client.aclose()