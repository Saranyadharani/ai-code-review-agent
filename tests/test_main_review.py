import pytest

from app import cli, main
from app.config import settings


@pytest.mark.asyncio
async def test_file_graph_failure_posts_notice_and_fails_review(monkeypatch):
    notices = []

    class FakeGitHubClient:
        async def get_file_content(self, owner, repo, filename, ref):
            return "def changed():\n    return None\n"

        async def aclose(self):
            pass

    async def fake_collect(owner, repo, number):
        return {
            "repo": f"{owner}/{repo}",
            "title": "test pull request",
            "head_sha": "head-sha",
            "base_sha": "base-sha",
            "files": [{
                "filename": "broken.py",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": "@@ -1 +1 @@\n-return None\n+return False\n",
            }],
        }

    async def fake_context(*args, **kwargs):
        return {"changed_functions": ["changed"], "semgrep": [], "ruff": []}

    async def broken_graph(*args):
        raise TypeError("simulated graph failure")

    async def fake_notice(*args):
        notices.append(args)

    async def fake_index(*args):
        return None

    monkeypatch.setattr(main, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(main, "collect_pr_data", fake_collect)
    monkeypatch.setattr(main, "build_file_context", fake_context)
    monkeypatch.setattr(main, "run_graph", broken_graph)
    monkeypatch.setattr(main, "post_review_notice", fake_notice)
    monkeypatch.setattr(main, "create_repository_index", fake_index)
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "publish_reviews", False)

    with pytest.raises(RuntimeError, match="broken.py"):
        await main.run_review("acme", "widgets", 7)

    assert notices[0][0:4] == ("acme", "widgets", 7, "head-sha")
    assert notices[0][4] == ["broken.py"]


def test_cli_exits_nonzero_when_review_fails(monkeypatch, tmp_path):
    event_file = tmp_path / "event.json"
    event_file.write_text(
        '{"pull_request":{"number":7}}',
        encoding="utf-8",
    )

    async def failed_review(owner, repo, number):
        raise RuntimeError("review failed")

    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/widgets")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setattr(cli, "run_review", failed_review)

    with pytest.raises(RuntimeError, match="review failed"):
        cli.main()
