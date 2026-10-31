"""GitHub Actions entry point for pull request reviews."""
import asyncio
import json
import os
from pathlib import Path

from app.main import run_review


def main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    owner, separator, repo = repository.partition("/")
    if not separator or not owner or not repo or "/" in repo:
        raise ValueError("GITHUB_REPOSITORY must be in owner/repo format")

    event_path = Path(os.environ["GITHUB_EVENT_PATH"])
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    number = payload["pull_request"]["number"]
    if not isinstance(number, int):
        raise ValueError("pull_request.number must be an integer")
    asyncio.run(run_review(owner, repo, number))


if __name__ == "__main__":
    main()
