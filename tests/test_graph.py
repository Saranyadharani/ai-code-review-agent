import pytest

from app import graph


@pytest.mark.asyncio
async def test_new_nodes_use_matching_ruff_hints(monkeypatch):
    prompts = []

    async def fake_ask_json(prompt):
        prompts.append(prompt)
        return {
            "findings": [{
                "line": 4,
                "severity": "medium",
                "title": "Potential issue",
                "explanation": "A test finding.",
                "suggestion": "Use a safer form.",
            }]
        }

    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    state = {
        "filename": "app.py",
        "source": "def f():\n    return []\n",
        "ctx": {
            "changed_functions": ["f"],
            "ruff": [
                {"line": 4, "rule": "PERF401", "message": "Use list comprehension", "category": "performance"},
                {"line": 5, "rule": "D103", "message": "Missing docstring", "category": "quality"},
                {"line": 6, "rule": "B023", "message": "Late binding", "category": "bug"},
                {"line": 7, "rule": "S307", "message": "Use of eval", "category": "security"},
            ],
            "semgrep": [],
        },
    }

    performance = await graph.performance_node(state)
    quality = await graph.quality_node(state)
    bug = await graph.bug_node(state)
    security = await graph.security_node(state)

    assert performance["performance"][0].category == "performance"
    assert quality["quality"][0].category == "quality"
    assert "PERF401" in prompts[0] and "D103" not in prompts[0]
    assert "D103" in prompts[1] and "PERF401" not in prompts[1]
    assert "late-binding closures" in prompts[2]
    assert "B023" in prompts[2] and "S307" not in prompts[2]
    assert "S307" in prompts[3] and "B023" not in prompts[3]


@pytest.mark.asyncio
async def test_validate_deduplicates_static_hints_by_category_and_distance():
    state = {
        "bugs": [],
        "security": [
            graph.Finding(line=10, category="security", title="LLM security finding")
        ],
        "performance": [
            graph.Finding(line=20, category="performance", title="LLM performance finding")
        ],
        "quality": [],
        "ctx": {
            "semgrep": [
                {"line": 12, "message": "Covered security hint", "severity": "ERROR"},
                {"line": 30, "message": "Uncovered security hint", "severity": "WARNING"},
            ],
            "ruff": [
                {"line": 22, "rule": "PERF401", "message": "Covered performance hint", "category": "performance"},
                {"line": 40, "rule": "B023", "message": "Uncovered bug hint", "category": "bug"},
            ],
        },
    }

    result = await graph.validate_node(state)

    assert [(finding.category, finding.line) for finding in result["findings"]] == [
        ("security", 10),
        ("performance", 20),
        ("security", 30),
        ("bug", 40),
    ]


@pytest.mark.asyncio
async def test_validate_drops_impact_near_bug_or_security():
    state = {
        "bugs": [graph.Finding(line=10, category="bug", title="Wrong result")],
        "security": [graph.Finding(line=20, category="security", title="Unsafe input")],
        "performance": [],
        "quality": [],
        "impact": [
            graph.Finding(line=11, category="impact", title="Same issue"),
            graph.Finding(line=13, category="impact", title="Separate impact"),
            graph.Finding(line=21, category="impact", title="Duplicate security impact"),
            graph.Finding(line=23, category="impact", title="Other impact"),
        ],
        "ctx": {"semgrep": [], "ruff": []},
    }

    result = await graph.validate_node(state)

    assert [(finding.category, finding.line) for finding in result["findings"]] == [
        ("bug", 10),
        ("impact", 13),
        ("security", 20),
        ("impact", 23),
    ]


@pytest.mark.asyncio
async def test_impact_node_only_keeps_findings_with_retrieved_caller(monkeypatch):
    prompts = []

    async def fake_ask_json(prompt):
        prompts.append(prompt)
        return {
            "findings": [{
                "line": 2,
                "severity": "medium",
                "title": "Caller may break",
                "explanation": "handlers.py:8 still calls changed with the old signature.",
                "suggestion": "",
            }]
        }

    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    state = {
        "filename": "service.py",
        "source": "def changed(value):\n    return value\n",
        "ctx": {
            "changed_functions": ["changed"],
            "functions": [{"name": "changed", "start": 1, "end": 2}],
            "semantic_diff": {"signature_changes": [{"name": "changed"}]},
            "related_code": {
                "text": "handlers.py:8 calls changed()",
                "callers": [{
                    "symbol": "changed",
                    "file": "handlers.py",
                    "line": 8,
                    "function": "handle",
                }],
                "tests_changed": False,
            },
        },
    }

    result = await graph.impact_node(state)

    assert "handlers.py:8 calls changed()" in prompts[0]
    assert "signature_changes" in prompts[0]
    assert result["impact"][0].category == "impact"


@pytest.mark.asyncio
async def test_impact_node_returns_empty_when_changed_function_has_no_callers(monkeypatch):
    async def unexpected_ask_json(prompt):
        raise AssertionError("Impact agent should not run when there are no callers")

    monkeypatch.setattr(graph, "ask_json", unexpected_ask_json)
    state = {
        "filename": "service.py",
        "source": "def changed(value):\n    return value\n",
        "ctx": {
            "changed_functions": ["changed"],
            "semantic_diff": {"modified_functions": ["changed"]},
            "related_code": {"text": "", "callers": [], "tests_changed": False},
        },
    }

    result = await graph.impact_node(state)

    assert result == {"impact": []}


def test_header_includes_imports_and_only_changed_functions():
    state = {
        "filename": "app.py",
        "source": (
            "import os\n"
            "constant = 1\n"
            "def changed():\n"
            "    return os.getenv('X')\n"
            "def unchanged():\n"
            "    return constant\n"
        ),
        "ctx": {
            "imports": ["import os"],
            "changed_functions": ["changed"],
            "functions": [
                {"name": "changed", "start": 3, "end": 4},
                {"name": "unchanged", "start": 5, "end": 6},
            ],
        },
    }

    header = graph._header(state)

    assert "1: import os" in header
    assert "3: def changed():" in header
    assert "4:     return os.getenv('X')" in header
    assert "unchanged" not in header
    assert "constant = 1" not in header


def test_header_keeps_multiline_imports_complete():
    state = {
        "filename": "app.py",
        "source": (
            "from package import (\n"
            "    first,\n"
            "    second,\n"
            ")\n"
            "def changed():\n"
            "    return first\n"
        ),
        "ctx": {
            "imports": ["from package import (\n    first,\n    second,\n)"],
            "changed_functions": ["changed"],
            "functions": [{"name": "changed", "start": 5, "end": 6}],
        },
    }

    header = graph._header(state)

    assert "1: from package import (" in header
    assert "2:     first," in header
    assert "3:     second," in header
    assert "4: )" in header


def test_header_passes_unified_and_semantic_diffs():
    state = {
        "filename": "app.py",
        "source": "def changed():\n    return 1\n",
        "ctx": {
            "changed_functions": ["changed"],
            "functions": [{"name": "changed", "start": 1, "end": 2}],
            "unified_diff": "@@ -1 +1 @@\n-return 0\n+return 1\n",
            "semantic_diff": {"signature_changes": [{"name": "changed"}]},
        },
    }

    prompt_context = graph._header(state)

    assert "Unified PR diff:" in prompt_context
    assert "-return 0" in prompt_context
    assert '"signature_changes"' in prompt_context
