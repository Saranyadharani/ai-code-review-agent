"""Step 3: Bug Detection agent (Groq, OpenAI-compatible API). One LLM call per file."""
import asyncio
import json
import logging

import httpx
from pydantic import BaseModel

from app.config import settings

URL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT = """You are a senior engineer reviewing a pull request. Find LOGIC BUGS and missed edge cases
(off-by-one, division by zero, mutable defaults, wrong conditions, swallowed errors, None handling).
Do NOT report security issues (another agent does that). Only report bugs in CHANGED functions: {changed}.
Reply with a JSON object: {{"bugs": [ ... ]}}. Each bug has keys line (int, from the numbered source),
severity (high|medium|low), title (short), explanation (1-2 sentences), suggestion (the fix).
Example: {{"bugs": [{{"line": 9, "severity": "high", "title": "...", "explanation": "...", "suggestion": "..."}}]}}
Use {{"bugs": []}} if there are none.

File: {file}
Numbered source:
{source}
"""

log = logging.getLogger("review-agent")


class Bug(BaseModel):
    line: int
    severity: str
    title: str
    explanation: str
    suggestion: str


def parse_bugs(text: str) -> list[Bug]:
    data = json.loads(text)
    if isinstance(data, dict):
        items = data.get("bugs")
        if items is None:  # model used a different key
            items = next((v for v in data.values() if isinstance(v, list)), [])
        if isinstance(items, dict):
            items = [items]
    else:
        items = data
    bugs = []
    for b in items if isinstance(items, list) else []:
        try:
            bugs.append(Bug(**b))
        except (TypeError, ValueError):
            log.warning("Skipping malformed bug item: %.200r", b)
    return bugs


async def review_file(filename: str, source: str, ctx: dict, client: httpx.AsyncClient | None = None) -> list[Bug]:
    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(source.splitlines()[:400], 1))
    prompt = PROMPT.format(changed=", ".join(ctx["changed_functions"]) or "none", file=filename, source=numbered)
    own = client is None
    client = client or httpx.AsyncClient(timeout=45)
    try:
        for model in (settings.groq_model, settings.groq_fallback_model):
            body = {"model": model, "temperature": 0.1, "max_tokens": 2000,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}]}
            for _ in range(2):
                try:
                    r = await client.post(URL, json=body,
                                          headers={"Authorization": f"Bearer {settings.groq_api_key}"})
                    if r.status_code in (429, 503):
                        log.warning("Groq %s on %s, retrying", r.status_code, model)
                        await asyncio.sleep(3)
                        continue
                    r.raise_for_status()
                    return parse_bugs(r.json()["choices"][0]["message"]["content"])
                except httpx.TimeoutException:
                    log.warning("Groq timed out on %s, trying next model", model)
                    break
                except httpx.HTTPStatusError as e:
                    log.error("Groq HTTP %s on %s: %s", e.response.status_code, model, e.response.text[:300])
                    break
        log.error("No Groq model produced a result, skipping bug review")
        return []
    except Exception as e:  # a failed LLM call must never break the review pipeline
        log.error("Bug agent failed: %r", e)
        return []
    finally:
        if own:
            await client.aclose()