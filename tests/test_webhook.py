import json

from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.security import compute_signature, verify_signature

SECRET = "test-secret"


def test_signature_roundtrip():
    body = b'{"a": 1}'
    assert verify_signature(SECRET, body, compute_signature(SECRET, body))


def test_signature_rejects_tampering_and_missing():
    body = b'{"a": 1}'
    sig = compute_signature(SECRET, body)
    assert not verify_signature(SECRET, b'{"a": 2}', sig)
    assert not verify_signature(SECRET, body, None)
    assert not verify_signature("", body, sig)


def _post(client, payload, event, secret=SECRET):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-Hub-Signature-256": compute_signature(secret, body),
            "X-GitHub-Event": event,
            "Content-Type": "application/json",
        },
    )


def test_webhook(monkeypatch):
    monkeypatch.setattr(settings, "github_webhook_secret", SECRET)
    calls = []

    async def fake_review(owner, repo, number):
        calls.append((owner, repo, number))

    monkeypatch.setattr(main, "run_review", fake_review)
    client = TestClient(main.app)

    assert _post(client, {}, "ping").json() == {"msg": "pong"}
    assert _post(client, {}, "ping", secret="wrong").status_code == 401

    pr = {
        "action": "opened",
        "repository": {"full_name": "acme/widgets"},
        "pull_request": {"number": 7},
    }
    assert _post(client, pr, "pull_request").json()["msg"] == "review queued"
    assert calls == [("acme", "widgets", 7)]

    pr["action"] = "closed"
    assert "ignored" in _post(client, pr, "pull_request").json()["msg"]
