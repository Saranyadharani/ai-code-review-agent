import asyncio
import json

import httpx
import pytest

from app import llm
from app.llm import ask_json


@pytest.mark.asyncio
async def test_ask_json_limits_concurrent_requests_to_two():
    active = 0
    maximum_active = 0

    async def handler(request):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        results = await asyncio.gather(
            *(ask_json(f"prompt {index}", client=client) for index in range(6))
        )
    finally:
        await client.aclose()

    assert results == [{"ok": True}] * 6
    assert maximum_active == 2


@pytest.mark.asyncio
async def test_ask_json_retries_rate_limit_with_exponential_jitter(monkeypatch):
    requests = []
    delays = []

    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
            request=request,
        )

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm.random, "uniform", lambda low, high: high)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await ask_json("retry this request", client=client)
    finally:
        await client.aclose()

    assert result == {"ok": True}
    assert len(requests) == 2
    assert delays == [1.25]
