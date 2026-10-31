import asyncio
import json

import httpx
import pytest

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
