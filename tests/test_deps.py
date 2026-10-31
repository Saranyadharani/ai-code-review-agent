import json

import httpx
import pytest

from app import main
from app.config import settings
from app.deps import parse_dependencies, scan_dependencies
from app.graph import Finding


def test_parse_pinned_and_unpinned_requirements():
    dependencies = parse_dependencies(
        "requirements-dev.txt",
        "requests==2.31.0\n"
        "flask>=3.0\n"
        "# comment\n"
        "urllib3==2.2.1 ; python_version >= '3.10'\n",
    )

    assert [(item.name, item.version, item.line) for item in dependencies] == [
        ("requests", "2.31.0", 1),
        ("flask", None, 2),
        ("urllib3", "2.2.1", 4),
    ]


def test_parse_pep621_and_poetry_dependency_tables():
    content = """
[project]
dependencies = [
    "requests==2.31.0",
    "flask>=3.0",
]

[project.optional-dependencies]
dev = ["pytest==8.3.0"]

[tool.poetry.dependencies]
python = "^3.12"
httpx = "==0.27.0"
rich = { version = ">=13.0" }

[tool.poetry.group.dev.dependencies]
ruff = { version = "==0.14.0" }
"""

    dependencies = parse_dependencies("pyproject.toml", content)

    assert [(item.name, item.version) for item in dependencies] == [
        ("requests", "2.31.0"),
        ("flask", None),
        ("pytest", "8.3.0"),
        ("httpx", "0.27.0"),
        ("rich", None),
        ("ruff", "0.14.0"),
    ]
    assert [item.line for item in dependencies] == [4, 5, 9, 13, 14, 17]


@pytest.mark.asyncio
async def test_osv_query_and_vulnerability_details_are_parsed():
    requests = []

    async def handler(request):
        requests.append(request)
        if request.url.path == "/v1/querybatch":
            return httpx.Response(
                200,
                json={"results": [{"vulns": [{"id": "GHSA-test"}]}, {"vulns": []}]},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "id": "GHSA-test",
                "summary": "Example vulnerability",
                "severity": [{"type": "CVSS_V3", "score": "8.1"}],
                "affected": [{
                    "package": {"ecosystem": "PyPI", "name": "requests"},
                    "ranges": [{"events": [{"introduced": "0"}, {"fixed": "2.32.0"}]}],
                }],
            },
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://api.osv.dev",
        transport=httpx.MockTransport(handler),
    )
    try:
        findings = await scan_dependencies(
            "requirements.txt",
            "requests==2.31.0\nflask==3.0.0\n",
            client,
        )
    finally:
        await client.aclose()

    assert len(requests) == 2
    assert requests[0].method == "POST"
    assert json.loads(requests[0].content) == {
        "queries": [
            {"package": {"name": "requests", "ecosystem": "PyPI"}, "version": "2.31.0"},
            {"package": {"name": "flask", "ecosystem": "PyPI"}, "version": "3.0.0"},
        ]
    }
    assert requests[1].url.path == "/v1/vulns/GHSA-test"
    assert len(findings) == 1
    finding = findings[0]
    assert finding.category == "dependency"
    assert finding.line == 1
    assert finding.severity == "high"
    assert finding.suggestion == "Update requests from ==2.31.0 to ==2.32.0."
    assert "GHSA-test" in finding.explanation


@pytest.mark.asyncio
async def test_unpinned_dependencies_are_reported_without_osv_call():
    async def handler(request):
        pytest.fail(f"OSV should not be queried for unpinned dependencies: {request.url}")

    client = httpx.AsyncClient(
        base_url="https://api.osv.dev",
        transport=httpx.MockTransport(handler),
    )
    try:
        findings = await scan_dependencies("requirements.txt", "requests>=2\n", client)
    finally:
        await client.aclose()

    assert len(findings) == 1
    assert findings[0].title == "Cannot check unpinned dependency: requests"
    assert findings[0].category == "dependency"


@pytest.mark.asyncio
async def test_run_review_scans_and_publishes_dependency_files(monkeypatch):
    published = []
    scanned = []

    class FakeGitHubClient:
        async def get_file_content(self, owner, repo, filename, ref):
            return "requests==2.31.0\n"

        async def aclose(self):
            pass

    async def fake_collect(owner, repo, number):
        return {
            "repo": f"{owner}/{repo}",
            "title": "dependency update",
            "head_sha": "abc123",
            "files": [{
                "filename": "requirements-dev.txt",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": "@@ -1 +1 @@\n-requests==2.30.0\n+requests==2.31.0\n",
            }],
        }

    async def fake_scan(filename, content):
        scanned.append((filename, content))
        return [Finding(
            line=1,
            category="dependency",
            title="Vulnerable dependency",
            severity="high",
        )]

    async def fake_publish(*args):
        published.append(args)

    monkeypatch.setattr(main, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(main, "collect_pr_data", fake_collect)
    monkeypatch.setattr(main, "scan_dependencies", fake_scan)
    monkeypatch.setattr(main, "post_reviews", fake_publish)
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "publish_reviews", True)

    await main.run_review("acme", "widgets", 13)

    assert scanned == [("requirements-dev.txt", "requests==2.31.0\n")]
    assert len(published) == 1
    assert published[0][4][0]["filename"] == "requirements-dev.txt"
    assert published[0][4][0]["findings"][0].category == "dependency"
