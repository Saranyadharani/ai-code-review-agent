import json
import hashlib

import httpx
import pytest

from app import github_client, main, publisher
from app.config import settings
from app.graph import Finding


@pytest.mark.asyncio
async def test_post_review_comments_only_on_changed_lines(monkeypatch):
    clients = []
    remote_comments = []

    class FakeGitHubClient:
        def __init__(self):
            self.reviews = []
            self.closed = False
            clients.append(self)

        async def create_pull_request_review(self, *args):
            self.reviews.append(args)
            remote_comments.extend(args[-1])

        async def list_pull_request_comments(self, owner, repo, number):
            return list(remote_comments)

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    findings = [
        {
            "line": 2,
            "category": "bug",
            "severity": "high",
            "title": "Incorrect result",
            "explanation": "The result is off by one.",
            "fix": {
                "original": "return count",
                "replacement": "return total",
                "verified": True,
            },
        },
        {
            "line": 9,
            "category": "security",
            "severity": "medium",
            "title": "Unverified change",
            "explanation": "This line is outside the patch.",
            "fix": {"replacement": "safe_call()", "verified": False},
        },
    ]
    patch = "@@ -1,1 +1,2 @@\n-old()\n+new()\n+return total\n"
    source = "old()\nreturn count\n"

    result = await publisher.post_review(
        "acme", "widgets", 7, "abc123", "src/app.py", findings, patch, source
    )

    assert result == {"published": True, "comments": 1, "summary_findings": 1}
    owner, repo, number, sha, body, comments = clients[0].reviews[0]
    assert (owner, repo, number, sha) == ("acme", "widgets", 7, "abc123")
    assert body.startswith("Findings in the review summary")
    assert "safe_call()" in body
    assert "```suggestion" not in body
    assert comments == [{
        "path": "src/app.py",
        "line": 2,
        "side": "RIGHT",
        "body": (
            "**HIGH / bug: Incorrect result**\n\n"
            "The result is off by one.\n\n"
            "```suggestion\nreturn total\n```\n\n"
            f"<!-- review-agent:{hashlib.sha1(b'src/app.py2Incorrect result').hexdigest()} -->"
        ),
    }]
    assert clients[0].closed


@pytest.mark.asyncio
async def test_suggestion_preserves_indentation_and_trailing_code(monkeypatch):
    comments = []

    class FakeGitHubClient:
        async def create_pull_request_review(self, *args):
            comments.extend(args[-1])

        async def list_pull_request_comments(self, owner, repo, number):
            return []

        async def aclose(self):
            pass

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    source = "    return eval(x)  # c\n"
    finding = {
        "line": 1,
        "category": "security",
        "severity": "high",
        "title": "Unsafe evaluation",
        "fix": {
            "original": "eval(x)",
            "replacement": "ast.literal_eval(x)",
            "verified": True,
        },
    }

    await publisher.post_review(
        "acme", "widgets", 8, "abc123", "app.py", [finding],
        "@@ -1 +1 @@\n-    return eval(x)  # c\n+    return ast.literal_eval(x)  # c\n",
        source,
    )

    assert "```suggestion\n    return ast.literal_eval(x)  # c\n```" in comments[0]["body"]
    assert comments[0]["line"] == 1
    assert "start_line" not in comments[0]


@pytest.mark.asyncio
async def test_multiline_suggestion_preserves_block_indentation_and_range(monkeypatch):
    comments = []

    class FakeGitHubClient:
        async def create_pull_request_review(self, *args):
            comments.extend(args[-1])

        async def list_pull_request_comments(self, owner, repo, number):
            return []

        async def aclose(self):
            pass

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    source = (
        "def f():\n"
        "    if condition:\n"
        "        first()\n"
        "        second()  # c\n"
    )
    finding = {
        "line": 3,
        "category": "bug",
        "severity": "medium",
        "title": "Replace block",
        "fix": {
            "original": "first()\n        second()",
            "replacement": "first_new()\n        second_new()\n        third()",
            "verified": True,
        },
    }

    await publisher.post_review(
        "acme", "widgets", 9, "abc123", "app.py", [finding],
        "@@ -3,2 +3,3 @@\n-        first()\n-        second()  # c\n"
        "+        first_new()\n+        second_new()\n+        third()  # c\n",
        source,
    )

    assert comments[0]["line"] == 4
    assert comments[0]["start_line"] == 3
    assert comments[0]["start_side"] == "RIGHT"
    assert (
        "```suggestion\n"
        "        first_new()\n"
        "        second_new()\n"
        "        third()  # c\n"
        "```"
    ) in comments[0]["body"]


@pytest.mark.asyncio
async def test_github_client_posts_review_request(monkeypatch):
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"id": 13}, request=request)

    mock_client = httpx.AsyncClient(
        base_url="https://api.github.test",
        transport=httpx.MockTransport(handler),
    )

    def fake_async_client(**kwargs):
        assert kwargs["timeout"] == 30.0
        return mock_client

    monkeypatch.setattr(github_client.httpx, "AsyncClient", fake_async_client)
    client = github_client.GitHubClient()
    comments = [{"path": "app.py", "line": 4, "side": "RIGHT", "body": "Finding"}]

    try:
        result = await client.create_pull_request_review(
            "acme", "widgets", 7, "abc123", "Summary", comments
        )
    finally:
        await client.aclose()

    assert result == {"id": 13}
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/repos/acme/widgets/pulls/7/reviews"
    assert json.loads(requests[0].content) == {
        "commit_id": "abc123",
        "body": "Summary",
        "event": "COMMENT",
        "comments": comments,
    }


@pytest.mark.asyncio
async def test_run_review_publishes_when_enabled(monkeypatch):
    review_calls = []

    class FakeGitHubClient:
        async def get_file_content(self, owner, repo, filename, ref):
            return "print('review me')\n"

        async def aclose(self):
            pass

    async def fake_collect(owner, repo, number):
        return {
            "repo": f"{owner}/{repo}",
            "title": "test pull request",
            "head_sha": "abc123",
            "files": [{
                "filename": "app.py",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": "@@ -1,0 +1 @@\n+print('review me')",
            }],
        }

    async def fake_graph(filename, source, context):
        return {
            "findings": [
                Finding(line=1, category="bug", title="Example finding")
            ],
            "verify": {},
        }

    async def fake_publish(*args):
        review_calls.append(args)

    monkeypatch.setattr(main, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(main, "collect_pr_data", fake_collect)
    async def fake_context(*args, **kwargs):
        return {"changed_functions": [], "semgrep": [], "ruff": []}

    monkeypatch.setattr(main, "build_file_context", fake_context)
    monkeypatch.setattr(main, "run_graph", fake_graph)
    monkeypatch.setattr(main, "post_reviews", fake_publish)
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "publish_reviews", True)

    await main.run_review("acme", "widgets", 7)

    assert len(review_calls) == 1
    assert review_calls[0][:4] == ("acme", "widgets", 7, "abc123")
    assert review_calls[0][4][0]["filename"] == "app.py"
    assert review_calls[0][4][0]["findings"][0].title == "Example finding"


@pytest.mark.asyncio
async def test_review_category_filter_and_quality_summary_only(monkeypatch):
    reviews = []

    class FakeGitHubClient:
        async def create_pull_request_review(self, *args):
            reviews.append(args)

        async def list_pull_request_comments(self, owner, repo, number):
            return []

        async def aclose(self):
            pass

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    findings = [
        {
            "line": 1, "category": "security", "severity": "high", "title": "Allowed",
            "fix": {"original": "unsafe()", "replacement": "safe()", "verified": True},
        },
        {
            "line": 2, "category": "quality", "severity": "high", "title": "Opt-in quality",
            "fix": {"original": "weak()", "replacement": "better()", "verified": True},
        },
        {
            "line": 3, "category": "security", "severity": "low", "title": "Low severity",
            "fix": {"original": "minor()", "replacement": "improved()", "verified": True},
        },
    ]
    source = "unsafe()\nweak()\nminor()\n"
    patch = "@@ -0,0 +1,3 @@\n+unsafe()\n+weak()\n+minor()\n"

    result = await publisher.post_review(
        "acme", "widgets", 10, "sha", "app.py", findings, patch, source,
        categories={"security", "quality"},
    )

    assert result == {"published": True, "comments": 1, "summary_findings": 2}
    body, comments = reviews[0][-2:]
    assert len(comments) == 1
    assert "```suggestion\nsafe()\n```" in comments[0]["body"]
    assert "Opt-in quality" in body
    assert "Low severity" in body
    assert "```suggestion" not in body

    filtered = await publisher.post_review(
        "acme", "widgets", 11, "sha", "app.py", findings, patch, source,
        categories={"bug"},
    )
    assert filtered == {"published": False, "comments": 0, "summary_findings": 0}
    assert len(reviews) == 1


@pytest.mark.asyncio
async def test_max_inline_comments_selects_highest_severity_first(monkeypatch):
    reviews = []

    class FakeGitHubClient:
        async def create_pull_request_review(self, *args):
            reviews.append(args)

        async def list_pull_request_comments(self, owner, repo, number):
            return []

        async def aclose(self):
            pass

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    findings = [
        {"line": line, "category": "bug", "severity": "medium", "title": f"medium-{line}"}
        for line in range(1, 5)
    ] + [
        {"line": line, "category": "security", "severity": "high", "title": f"high-{line}"}
        for line in range(5, 9)
    ]
    patch = "@@ -0,0 +1,8 @@\n" + "".join(f"+line {line}\n" for line in range(1, 9))

    result = await publisher.post_review(
        "acme", "widgets", 12, "sha", "app.py", findings, patch,
        max_inline_comments=3,
    )

    assert result == {"published": True, "comments": 3, "summary_findings": 5}
    _, comments = reviews[0][-2:]
    assert [comment["body"].splitlines()[0] for comment in comments] == [
        "**HIGH / security: high-5**",
        "**HIGH / security: high-6**",
        "**HIGH / security: high-7**",
    ]
    assert "medium-1" in reviews[0][-2]


@pytest.mark.asyncio
async def test_marker_dedupe_uses_existing_github_comments(monkeypatch):
    comments_on_github = []
    posted_reviews = []
    instances = []

    class FakeGitHubClient:
        def __init__(self):
            instances.append(self)

        async def list_pull_request_comments(self, owner, repo, number):
            return list(comments_on_github)

        async def create_pull_request_review(self, *args):
            posted_reviews.append(args)
            comments_on_github.extend(args[-1])

        async def aclose(self):
            pass

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    finding = {
        "line": 1,
        "category": "security",
        "severity": "high",
        "title": "Unsafe operation",
    }
    inputs = [{
        "filename": "app.py",
        "findings": [finding],
        "patch": "@@ -0,0 +1 @@\n+unsafe()\n",
        "source": "unsafe()\n",
    }]

    first = await publisher.post_reviews("acme", "widgets", 20, "sha", inputs)
    second = await publisher.post_reviews("acme", "widgets", 20, "sha", inputs)

    marker = f"<!-- review-agent:{hashlib.sha1(b'app.py1Unsafe operation').hexdigest()} -->"
    assert first == {"published": True, "comments": 1, "summary_findings": 0}
    assert marker in posted_reviews[0][-1][0]["body"]
    assert second == {"published": False, "comments": 0, "summary_findings": 0}
    assert len(posted_reviews) == 1
    assert len(instances) == 2
