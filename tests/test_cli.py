import json

import pytest

from app import cli


def test_cli_reads_repository_and_pull_request_number(monkeypatch, tmp_path):
    event_file = tmp_path / "event.json"
    event_file.write_text(
        json.dumps({"pull_request": {"number": 37}}),
        encoding="utf-8",
    )
    calls = []

    async def fake_review(owner, repo, number):
        calls.append((owner, repo, number))

    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/widgets")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setattr(cli, "run_review", fake_review)

    cli.main()

    assert calls == [("acme", "widgets", 37)]
