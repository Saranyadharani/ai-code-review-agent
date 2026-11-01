import pytest

from app import publisher
from app.graph import Finding
from app.review_rules import (
    SECURITY_RULE_IDS,
    category_for_rule,
    correct_category,
    deduplicate_findings,
)


@pytest.mark.parametrize("rule_id", sorted(SECURITY_RULE_IDS) + ["S307", "S608"])
def test_security_rules_correct_mislabelled_findings(rule_id):
    finding = Finding(
        line=1, category="bug", title="Static analyzer finding", rule_id=rule_id
    )

    assert correct_category(finding) == "security"


@pytest.mark.parametrize(
    "title",
    [
        "SQL injection risk",
        "Command injection from input",
        "Path traversal in file access",
        "SSRF from user URL",
        "XXE in XML parsing",
        "Unsafe deserialization using pickle",
        "Weak crypto uses a weak hash",
        "Hardcoded credential found",
    ],
)
def test_security_keywords_correct_mislabelled_findings(title):
    finding = Finding(line=1, category="bug", title=title)

    assert correct_category(finding) == "security"


def test_nonsecurity_static_rule_remains_a_bug():
    finding = Finding(
        line=1, category="security", title="Bare except", rule_id="bare-except"
    )

    assert correct_category(finding) == "bug"
    assert category_for_rule("S110") == "bug"
    assert category_for_rule("SIM115", "performance") == "performance"


def test_rule_dedupe_keeps_security_finding_and_best_fix():
    bug = Finding(line=5, category="bug", title="Unsafe input handling")
    security = Finding(
        line=6, category="security", title="Unsafe input handling",
        rule_id="sql-fstring",
    )
    bug.fix = {
        "original": "unsafe()",
        "replacement": "safe()",
        "critic_safe": True,
        "verified": True,
    }

    result = deduplicate_findings([bug, security])

    assert result == [security]
    assert security.category == "security"
    assert security.fix["finding_id"] == security.id
    assert {
        key: value for key, value in security.fix.items() if key != "finding_id"
    } == bug.fix


def test_rule_dedupe_merges_similar_findings_and_shared_rule_ids():
    first = Finding(
        line=2, category="bug", title="Incorrect prediction shape",
        explanation="Returns an array with the wrong dimensions.",
    )
    similar = Finding(
        line=40, category="performance", title="Incorrect prediction dimensions",
        explanation="Returns an array with incorrect dimensions.",
    )
    same_rule = Finding(
        line=90, category="quality", title="Other report", rule_id="sql-fstring"
    )
    same_rule_duplicate = Finding(
        line=120, category="bug", title="Different report", rule_id="sql-fstring"
    )

    result = deduplicate_findings([first, similar, same_rule, same_rule_duplicate])

    assert result == [same_rule, first]


def test_rule_dedupe_merges_transitive_adjacent_findings():
    findings = [
        Finding(line=1, category="security", title="First report"),
        Finding(line=2, category="bug", title="Second report"),
        Finding(line=3, category="performance", title="Third report"),
    ]

    result = deduplicate_findings(findings)

    assert result == [findings[0]]


SAMPLES = [
    (
        "model_utils.py",
        "preds = np.argmax(p.predictions, axis=0)\n",
        "bug",
        "Incorrect prediction axis",
        None,
    ),
    (
        "pickle_utils.py",
        "result = pickle.load(stream)\n",
        "bug",
        "Unsafe deserialization with pickle",
        "pickle-load",
    ),
    (
        "eval_utils.py",
        "result = eval(expression)\n",
        "bug",
        "Unsafe eval of dynamic input",
        "eval-use",
    ),
    (
        "query_utils.py",
        "cursor.execute(f\"SELECT * FROM users WHERE name='{name}'\")\n",
        "bug",
        "SQL injection through string formatting",
        "sql-fstring",
    ),
    (
        "command_utils.py",
        "os.system(command)\n",
        "bug",
        "Command injection from shell execution",
        "os-system",
    ),
    (
        "path_utils.py",
        "open(root + '/' + user_path)\n",
        "bug",
        "Path traversal in user-controlled file path",
        None,
    ),
    (
        "http_utils.py",
        "requests.get(url)\n",
        "bug",
        "SSRF from user-controlled URL",
        None,
    ),
    (
        "crypto_utils.py",
        "digest = hashlib.md5(data).hexdigest()\n",
        "bug",
        "Weak hash used for sensitive data",
        "hashlib-weak-hash",
    ),
    (
        "format_utils.py",
        "value = 1\n",
        "quality",
        "PEP 8 style recommendation",
        None,
    ),
    (
        "nullable_utils.py",
        "value.name\n",
        "bug",
        "Possible None handling failure",
        None,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,source,category,title,rule_id", SAMPLES)
async def test_sample_review_publishes_one_categorized_issue_without_duplicate_summary(
    monkeypatch, filename, source, category, title, rule_id
):
    reviews = []

    class FakeGitHubClient:
        async def list_pull_request_comments(self, owner, repo, number):
            return []

        async def create_pull_request_review(self, *args):
            reviews.append(args)

        async def aclose(self):
            pass

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    issue = Finding(
        line=1, category=category, severity="high", title=title, rule_id=rule_id
    )
    duplicate = Finding(
        line=2, category=category, severity="medium", title=title, rule_id=rule_id
    )
    if filename == "model_utils.py":
        issue.fix = {
            "original": "axis=0",
            "replacement": "axis=1",
            "critic_safe": True,
            "verified": True,
        }
    if filename == "pickle_utils.py":
        issue.fix = {
            "original": "pickle.load(stream)",
            "replacement": "json.load(stream)",
            "critic_safe": False,
            "verified": False,
            "critic_reason": "The data format and return type may differ.",
        }
    if filename == "eval_utils.py":
        issue.fix = {
            "original": "eval(expression)",
            "replacement": "ast.literal_eval(expression)",
            "critic_safe": False,
            "verified": False,
            "critic_reason": "The proposed call changes behavior for valid expressions.",
        }
    summary_only = Finding(
        line=100, category="bug", severity="low", title="Separate summary-only finding"
    )

    result = await publisher.post_review(
        "acme", "review-samples", 1, "sha", filename,
        [issue, duplicate, summary_only],
        "@@ -1 +1 @@\n-old line\n+" + source.rstrip() + "\n",
        source,
        categories={"security", "bug", "performance", "quality"},
    )

    assert result["comments"] == (0 if category == "quality" else 1)
    assert result["summary_findings"] == (2 if category == "quality" else 1)
    body, comments = reviews[0][-2:]
    all_text = body + "\n" + "\n".join(comment["body"] for comment in comments)
    assert all_text.count(title) == 1
    assert all_text.count("Separate summary-only finding") == 1
    assert sum("```suggestion" in comment["body"] for comment in comments) <= 1
    assert title not in body or category == "quality"
    if filename == "model_utils.py":
        assert "```suggestion\npreds = np.argmax(p.predictions, axis=1)\n```" in comments[0]["body"]
    if filename in {"pickle_utils.py", "eval_utils.py"}:
        assert "```suggestion" not in all_text
        assert "json.load(stream)" not in all_text
        assert "ast.literal_eval(expression)" not in all_text
        assert "data format and return type may differ" not in all_text
        assert "changes behavior for valid expressions" not in all_text
    assert (
        f"/ {category}:" in all_text
        or "**HIGH / security:" in all_text
    )


@pytest.mark.asyncio
async def test_same_finding_decision_in_different_repository_layouts(monkeypatch):
    results = []

    class FakeGitHubClient:
        async def list_pull_request_comments(self, owner, repo, number):
            return []

        async def create_pull_request_review(self, *args):
            results.append(args)

        async def aclose(self):
            pass

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    for path in ("src/services/worker.py", "packages/core/runtime.py"):
        finding = Finding(
            line=1,
            category="security",
            severity="high",
            title="Unsafe shell execution",
            explanation="Untrusted input reaches a shell command.",
        )
        finding.fix = {
            "finding_id": finding.id,
            "original": "run(command)",
            "replacement": "run(command, shell=False)",
            "note": "Pass arguments without shell interpretation.",
            "verified": True,
            "critic_safe": True,
            "critic_verdict": {
                "fixes_defect": True,
                "breaks_valid_use": False,
                "reason": "Private critic rationale.",
            },
        }
        await publisher.post_review(
            "acme", "unrelated", 1, "sha", path, [finding],
            "@@ -1 +1 @@\n-run(command)\n+run(command, shell=False)\n",
            "run(command)\n",
            categories={"security"},
        )

    bodies = [comments[0]["body"] for _, _, _, _, _, comments in results]
    normalized = [body.split("\n\n<!-- review-agent:")[0] for body in bodies]
    assert normalized[0] == normalized[1]
    assert all("Private critic rationale" not in body for body in bodies)
    assert all("```suggestion\nrun(command, shell=False)\n```" in body for body in bodies)


@pytest.mark.asyncio
async def test_low_style_performance_noise_is_filtered(monkeypatch):
    class FakeGitHubClient:
        async def list_pull_request_comments(self, owner, repo, number):
            return []

        async def aclose(self):
            pass

    monkeypatch.setattr(publisher, "GitHubClient", FakeGitHubClient)
    result = await publisher.post_review(
        "acme", "review-samples", 2, "sha", "format.py",
        [Finding(
            line=1, category="performance", severity="low",
            title="Style: use a shorter expression",
        )],
        "@@ -1 +1 @@\n-old\n+new\n",
        "new\n",
        categories={"performance"},
    )

    assert result == {"published": False, "comments": 0, "summary_findings": 0}
