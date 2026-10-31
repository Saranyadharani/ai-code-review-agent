import pytest

from app import diffing, main
from app.config import settings


def test_semantic_diff_detects_signature_and_removed_public_function():
    base = (
        "from lib import old_name\n\n"
        "def public(value=1) -> int:\n"
        "    return value\n\n"
        "def removed():\n"
        "    return None\n\n"
        "class Existing(Base):\n"
        "    pass\n\n"
        "class RemovedClass:\n"
        "    pass\n"
    )
    head = (
        "from lib import new_name\n\n"
        "def public(value, other=2) -> str:\n"
        "    return str(value)\n\n"
        "class Existing(NewBase):\n"
        "    pass\n\n"
        "class AddedClass:\n"
        "    pass\n"
    )

    result = diffing.semantic_diff(base, head)

    assert result["removed_functions"] == ["removed"]
    assert result["removed_public_functions"] == ["removed"]
    assert result["added_classes"] == ["AddedClass"]
    assert result["removed_classes"] == ["RemovedClass"]
    assert result["modified_classes"] == ["Existing"]
    assert result["changed_imports"] == [{
        "base": "from lib import old_name",
        "head": "from lib import new_name",
    }]
    assert result["signature_changes"] == [{
        "name": "public",
        "base": ("(value=1)", "int"),
        "head": ("(value, other=2)", "str"),
    }]
    assert "Removed public function: removed" in result["breaking_change_candidates"]
    assert "Changed function signature: public" in result["breaking_change_candidates"]


def test_analyzer_findings_are_classified_by_rule_and_normalized_line():
    base_source = "unsafe_call(value)\nold_call()\n"
    head_source = "new line\n  unsafe_call(value)\n"
    findings = diffing.classify_findings(
        [
            {"rule": "R1", "line": 1, "message": "existing"},
            {"rule": "R2", "line": 2, "message": "fixed"},
        ],
        [
            {"rule": "R1", "line": 2, "message": "existing"},
            {"rule": "R3", "line": 1, "message": "new"},
        ],
        base_source,
        head_source,
    )

    assert [(item["rule"], item["status"]) for item in findings] == [
        ("R1", "pre-existing"),
        ("R3", "new"),
        ("R2", "fixed"),
    ]


@pytest.mark.asyncio
async def test_context_reports_only_new_head_analyzer_findings(monkeypatch):
    from app import context

    def fake_semgrep(source):
        if source == "old()\n":
            return [{"rule": "R1", "line": 1, "message": "old", "severity": "ERROR"}]
        return [
            {"rule": "R1", "line": 2, "message": "old", "severity": "ERROR"},
            {"rule": "R2", "line": 3, "message": "new", "severity": "ERROR"},
        ]

    def fake_ruff(source):
        return []

    monkeypatch.setattr(context, "run_semgrep", fake_semgrep)
    monkeypatch.setattr(context, "run_ruff", fake_ruff)
    head = "prefix()\nold()\nnew()\n"
    patch = "@@ -1 +1,3 @@\n+prefix()\n+old()\n+new()\n"

    result = await context.build_file_context("app.py", head, patch, "old()\n")

    assert [(item["rule"], item["line"]) for item in result["semgrep"]] == [("R2", 3)]
    assert result["baseline_comparison"] == {
        "new": 1,
        "pre-existing": 1,
        "fixed": 0,
    }


@pytest.mark.asyncio
async def test_run_review_fetches_renamed_file_base_from_previous_path(monkeypatch):
    fetches = []

    class FakeGitHubClient:
        async def get_file_content(self, owner, repo, path, ref):
            fetches.append((path, ref))
            return "def f():\n    return 1\n"

        async def aclose(self):
            pass

    async def fake_collect(owner, repo, number):
        return {
            "repo": f"{owner}/{repo}",
            "title": "rename",
            "head_sha": "head-sha",
            "base_sha": "base-sha",
            "files": [{
                "filename": "new_name.py",
                "previous_filename": "old_name.py",
                "status": "renamed",
                "additions": 1,
                "deletions": 1,
                "patch": "@@ -1 +1 @@\n-old\n+new\n",
            }],
        }

    async def fake_context(filename, source, patch, base_source=None):
        assert base_source == "def f():\n    return 1\n"
        return {"changed_functions": [], "semgrep": [], "ruff": []}

    monkeypatch.setattr(main, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(main, "collect_pr_data", fake_collect)
    monkeypatch.setattr(main, "build_file_context", fake_context)
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "publish_reviews", False)

    await main.run_review("acme", "widgets", 21)

    assert fetches == [
        ("new_name.py", "head-sha"),
        ("old_name.py", "base-sha"),
    ]
