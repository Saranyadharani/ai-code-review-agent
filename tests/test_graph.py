import logging
import asyncio
import re

import pytest

from app import graph, publisher, verify
from app.context import parse_python


@pytest.mark.asyncio
async def test_new_nodes_use_matching_ruff_hints(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        graph.settings, "review_categories", "security,bug,performance,quality"
    )

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
    assert bug["bugs"][0].category == "bug"
    assert security["security"][0].category == "security"
    assert "PERF401" in prompts[0] and "D103" not in prompts[0]
    assert "D103" in prompts[1] and "PERF401" not in prompts[1]
    assert "late-binding closures" in prompts[2]
    assert "B023" in prompts[2] and "S307" not in prompts[2]
    assert "S307" in prompts[3] and "B023" not in prompts[3]
    assert all(
        "Only an excerpt of the file is shown. Do not report undefined names" in prompt
        for prompt in (prompts[2], prompts[3])
    )


@pytest.mark.asyncio
async def test_quality_agent_skips_when_disabled_and_caps_per_function(monkeypatch, caplog):
    calls = []

    async def fake_ask_json(prompt):
        calls.append(prompt)
        return {"findings": [
            {
                "line": line, "severity": "medium", "title": f"Quality issue {line}",
                "explanation": "Improve maintainability.",
            }
            for line in (2, 3, 4, 5)
        ]}

    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    monkeypatch.setattr(graph.settings, "review_categories", "security,bug")
    state = {
        "filename": "sample.py",
        "source": "def f():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n",
        "ctx": {
            "changed_functions": ["f"],
            "functions": [{"name": "f", "start": 1, "end": 5}],
            "ruff": [],
        },
    }

    assert await graph.quality_node(state) == {"quality": []}
    assert calls == []

    monkeypatch.setattr(graph.settings, "review_categories", "security,bug,quality")
    caplog.set_level(logging.INFO, logger="review-agent")
    result = await graph.quality_node(state)
    assert [finding.line for finding in result["quality"]] == [2, 3, 4]
    assert "no more than 3 findings per function" in calls[0]
    assert "reason=per_function_quality_cap:f" in caplog.text


@pytest.mark.asyncio
async def test_validate_deduplicates_static_hints_by_category_and_distance(monkeypatch):
    monkeypatch.setattr(
        graph.settings, "review_categories", "security,bug,performance,quality"
    )
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
                {
                    "line": 12, "rule": "eval-use",
                    "message": "Covered security hint", "severity": "ERROR",
                },
                {
                    "line": 30, "rule": "eval-use",
                    "message": "Uncovered security hint", "severity": "WARNING",
                },
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
async def test_validate_logs_id_and_reason_for_every_noise_drop(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="review-agent")
    monkeypatch.setattr(
        graph.settings, "review_categories", "security,bug,performance,dependency"
    )
    finding = graph.Finding(
        line=3,
        category="quality",
        severity="low",
        title="Potential quality improvement",
    )

    result = await graph.validate_node({
        "bugs": [],
        "security": [],
        "performance": [],
        "quality": [finding],
        "impact": [],
        "ctx": {"semgrep": [], "ruff": []},
    })

    assert result["findings"] == []
    assert f"finding_dropped id={finding.id}" in caplog.text
    assert "reason=quality_not_enabled" in caplog.text


@pytest.mark.asyncio
async def test_every_input_finding_is_retained_or_logged_as_dropped(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="review-agent")
    monkeypatch.setattr(
        graph.settings, "review_categories", "security,bug,performance,dependency"
    )
    findings = [
        graph.Finding(line=4, category="bug", title="Unsafe input handling"),
        graph.Finding(
            line=4, category="security", title="Unsafe input handling",
            rule_id="sql-fstring",
        ),
        graph.Finding(
            line=40, category="quality", severity="low",
            title="Potential quality improvement",
        ),
    ]

    result = await graph.validate_node({
        "bugs": [findings[0]],
        "security": [findings[1]],
        "performance": [],
        "quality": [findings[2]],
        "impact": [],
        "ctx": {"semgrep": [], "ruff": []},
    })

    retained_ids = {finding.id for finding in result["findings"]}
    dropped_ids = {
        finding.id for finding in findings
        if f"finding_dropped id={finding.id}" in caplog.text
    }
    assert retained_ids | dropped_ids == {finding.id for finding in findings}
    assert retained_ids & dropped_ids == set()


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


@pytest.mark.asyncio
async def test_fix_node_runs_behavior_critic(monkeypatch):
    prompts = []
    finding = graph.Finding(
        line=2, category="bug", severity="high", title="Unsafe operation"
    )
    responses = iter([
        {
            "fixes": [{
                "finding_id": finding.id,
                "original": "unsafe(value)",
                "replacement": "safe(value)",
                "note": "Use a behavior-preserving replacement.",
            }]
        },
        {
            "fixes_defect": True,
            "breaks_valid_use": True,
            "reason": "The replacement changes valid-input behavior.",
        },
    ])

    async def fake_ask_json(prompt):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    state = {
        "filename": "sample.py",
        "source": "def f(value):\n    return unsafe(value)\n",
        "ctx": {
            "changed_functions": ["f"],
            "functions": [{"name": "f", "start": 1, "end": 2}],
        },
        "findings": [finding],
    }

    await graph.fix_node(state)

    assert "Preserve imports at the top" in prompts[0]
    assert "Does this fix correct the flagged defect WITHOUT breaking valid uses?" in prompts[1]
    assert "Externalizing a hardcoded secret" in prompts[1]
    assert "adding a required configuration input is allowed" in prompts[1]
    assert finding.id in prompts[1]
    assert finding.fix["critic_safe"] is False
    assert finding.critic_verdict["reason"] == "The replacement changes valid-input behavior."
    assert finding.suggestion == ""


@pytest.mark.asyncio
async def test_run_graph_end_to_end_with_only_llm_mocked(monkeypatch):
    monkeypatch.setattr(
        graph.settings, "review_categories", "security,bug,performance"
    )
    source = (
        "import ast\n"
        "import os\n"
        "\n"
        'API_KEY = "hardcoded"\n'
        "\n"
        "def parse(value):\n"
        "    return eval(value)\n"
        "\n"
        "def run(command):\n"
        "    os.system(command)\n"
        "\n"
        "def adjust(value):\n"
        "    return value + 1\n"
    )
    findings = {
        "security": [
            {"line": 4, "severity": "high", "title": "Hardcoded secret", "explanation": "A credential is embedded in source."},
            {"line": 7, "severity": "high", "title": "Unsafe evaluation", "explanation": "Untrusted input is evaluated."},
            {"line": 10, "severity": "high", "title": "Command injection", "explanation": "Input reaches shell execution."},
        ],
        "bug": [{
            "line": 13, "severity": "high", "title": "Off-by-one arithmetic",
            "explanation": "The result is one less than expected.",
        }],
    }
    prompts = []

    async def fake_ask_json(prompt):
        prompts.append(prompt)
        if prompt.startswith("TASK: SECURITY."):
            return {"findings": findings["security"]}
        if prompt.startswith("TASK: BUGS."):
            return {"bugs": findings["bug"]}
        if prompt.startswith("TASK: PERFORMANCE."):
            return {"findings": []}
        if prompt.startswith("TASK: FIXES."):
            match = re.search(
                r"finding_id=([0-9a-f]+).*title=Off-by-one arithmetic", prompt
            )
            assert match
            return {"fixes": [{
                "finding_id": match.group(1),
                "original": "return value + 1",
                "replacement": "return value + 2",
                "note": "Use the correct offset.",
            }]}
        if prompt.startswith("Finding id:"):
            return {
                "fixes_defect": True,
                "breaks_valid_use": False,
                "reason": "The corrected arithmetic preserves the contract.",
            }
        raise AssertionError(f"Unexpected LLM prompt: {prompt[:80]}")

    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    context = parse_python(source)
    context.update({
        "changed_functions": ["parse", "run", "adjust"],
        "semgrep": [],
        "ruff": [],
        "related_code": {"callers": [], "text": "", "tests_changed": None},
    })

    result = await graph.run_graph("sample.py", source, context)

    assert len(result["findings"]) == 4
    assert result["verify"]["accepted"] == 1, [
        finding.fix for finding in result["findings"]
    ]
    bug = next(item for item in result["findings"] if item.title == "Off-by-one arithmetic")
    assert bug.fix["verified"] is True
    assert bug.critic_verdict["fixes_defect"] is True
    assert sum(prompt.startswith("TASK:") for prompt in prompts) == 4
    assert any(prompt.startswith("Finding id:") for prompt in prompts)
    for finding in result["findings"]:
        publisher._finding_text(finding, include_suggestion=True, source=source)


@pytest.mark.asyncio
async def test_fix_and_critic_verdicts_follow_finding_ids_after_reordering(monkeypatch):
    findings = [
        graph.Finding(line=2, category="bug", severity="high", title="First issue"),
        graph.Finding(line=3, category="bug", severity="high", title="Second issue"),
    ]
    verdicts = {
        findings[0].id: {"fixes_defect": True, "breaks_valid_use": False, "reason": "first"},
        findings[1].id: {"fixes_defect": False, "breaks_valid_use": False, "reason": "second"},
    }

    async def fake_ask_json(prompt):
        if "TASK: FIXES." in prompt:
            return {"fixes": [
                {
                    "finding_id": findings[1].id,
                    "original": "second()",
                    "replacement": "second_safe()",
                },
                {
                    "finding_id": findings[0].id,
                    "original": "first()",
                    "replacement": "first_safe()",
                },
            ]}
        finding_id = next(identity for identity in verdicts if identity in prompt)
        return verdicts[finding_id]

    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    await graph.fix_node({
        "filename": "sample.py",
        "source": "def f():\n    first()\n    second()\n",
        "ctx": {"changed_functions": ["f"], "functions": [{"name": "f", "start": 1, "end": 3}]},
        "findings": findings,
    })

    assert findings[0].fix["original"] == "first()"
    assert findings[1].fix["original"] == "second()"
    assert findings[0].critic_verdict["reason"] == "first"
    assert findings[1].critic_verdict["reason"] == "second"
    assert findings[0].fix["critic_safe"] is True
    assert findings[1].fix["critic_safe"] is False


@pytest.mark.asyncio
async def test_critic_parse_failure_retries_then_uses_deterministic_guards(monkeypatch):
    finding = graph.Finding(
        line=2, category="bug", severity="high", title="Incorrect increment"
    )
    calls = 0

    async def fake_ask_json(prompt):
        nonlocal calls
        calls += 1
        if "TASK: FIXES." in prompt:
            return {"fixes": [{
                "finding_id": finding.id,
                "original": "return value + 1",
                "replacement": "return value + 2",
                "note": "Adjust the increment.",
            }]}
        return {"safe": True, "reason": "legacy malformed verdict"}

    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    monkeypatch.setattr(graph, "CRITIC_TIMEOUT_SECONDS", 1)
    await graph.fix_node({
        "filename": "sample.py",
        "source": "def increment(value):\n    return value + 1\n",
        "ctx": {"changed_functions": ["increment"], "functions": [{"name": "increment", "start": 1, "end": 2}]},
        "findings": [finding],
    })

    monkeypatch.setattr(verify, "run_semgrep", lambda source: [])
    result = await verify.verify_fixes(
        "def increment(value):\n    return value + 1\n", [finding]
    )
    assert calls == 3
    assert finding.critic_verdict is None
    assert finding.fix["critic_failure"] == "critic returned an invalid verdict shape"
    assert result["accepted"] == 1


@pytest.mark.asyncio
async def test_critic_timeout_retries_then_uses_deterministic_guards(monkeypatch):
    finding = graph.Finding(
        line=2, category="bug", severity="high", title="Incorrect increment"
    )
    calls = 0

    async def fake_ask_json(prompt):
        nonlocal calls
        if "TASK: FIXES." in prompt:
            return {"fixes": [{
                "finding_id": finding.id,
                "original": "return value + 1",
                "replacement": "return value + 2",
            }]}
        calls += 1
        await asyncio.sleep(0.02)

    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    monkeypatch.setattr(graph, "CRITIC_TIMEOUT_SECONDS", 0.001)
    await graph.fix_node({
        "filename": "sample.py",
        "source": "def increment(value):\n    return value + 1\n",
        "ctx": {"changed_functions": ["increment"], "functions": [{"name": "increment", "start": 1, "end": 2}]},
        "findings": [finding],
    })

    monkeypatch.setattr(verify, "run_semgrep", lambda source: [])
    result = await verify.verify_fixes(
        "def increment(value):\n    return value + 1\n", [finding]
    )
    assert calls == 2
    assert finding.fix["critic_failure"] == "critic timed out"
    assert result["accepted"] == 1


@pytest.mark.asyncio
async def test_base_version_revert_skips_critic_and_is_verified(monkeypatch):
    calls = []

    async def fake_ask_json(prompt):
        calls.append(prompt)
        return {"fixes": [{
            "finding_id": finding.id,
            "original": "return value - 1",
            "replacement": "return value + 1",
        }]}

    finding = graph.Finding(
        line=2, category="bug", severity="high", title="Incorrect arithmetic"
    )
    monkeypatch.setattr(graph, "ask_json", fake_ask_json)
    source = "def calculate(value):\n    return value - 1\n"
    diff = "@@ -2 +2 @@\n-    return value + 1\n+    return value - 1\n"
    await graph.fix_node({
        "filename": "sample.py",
        "source": source,
        "ctx": {
            "changed_functions": ["calculate"],
            "functions": [{"name": "calculate", "start": 1, "end": 2}],
            "base_source": "def calculate(value):\n    return value + 1\n",
            "unified_diff": diff,
        },
        "findings": [finding],
    })

    monkeypatch.setattr(verify, "run_semgrep", lambda source: [])
    result = await verify.verify_fixes(source, [finding])
    assert len(calls) == 1
    assert finding.fix["base_revert"] is True
    assert finding.critic_verdict["source"] == "base-revert"
    assert result["accepted"] == 1


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
    assert "2: constant = 1" in header


def test_header_includes_module_assignments_capped_at_40_lines():
    assignments = "\n".join(f"VALUE_{number} = {number}" for number in range(45))
    state = {
        "filename": "app.py",
        "source": assignments,
        "ctx": {"changed_functions": []},
    }

    header = graph._header(state)

    assert "1: VALUE_0 = 0" in header
    assert "40: VALUE_39 = 39" in header
    assert "41: VALUE_40 = 40" not in header
    assert "45: VALUE_44 = 44" not in header


@pytest.mark.asyncio
async def test_validate_drops_unverified_undefined_name_claims(monkeypatch, caplog):
    monkeypatch.setattr(
        graph.settings, "review_categories", "security,bug,performance"
    )
    caplog.set_level(logging.INFO, logger="review-agent")
    finding = graph.Finding(
        line=2,
        category="bug",
        title="NameError: name 'missing_name' is not defined",
        explanation="The function references a missing name.",
    )

    result = await graph.validate_node({
        "source": "def f():\n    return value\n",
        "bugs": [finding],
        "ctx": {"semgrep": [], "ruff": []},
    })

    assert result["findings"] == []
    assert "reason=unverified_undefined_name_claim" in caplog.text


@pytest.mark.asyncio
async def test_validate_keeps_undefined_name_claims_reported_by_pyflakes(monkeypatch):
    monkeypatch.setattr(
        graph.settings, "review_categories", "security,bug,performance"
    )
    finding = graph.Finding(
        line=2,
        category="bug",
        title="Undefined name `missing_name`",
    )

    result = await graph.validate_node({
        "source": "def f():\n    return missing_name\n",
        "bugs": [finding],
        "ctx": {"semgrep": [], "ruff": []},
    })

    assert result["findings"] == [finding]


@pytest.mark.asyncio
async def test_validate_drops_low_severity_speculation_without_static_hint(monkeypatch, caplog):
    monkeypatch.setattr(
        graph.settings, "review_categories", "security,bug,performance"
    )
    caplog.set_level(logging.INFO, logger="review-agent")
    speculative = graph.Finding(
        line=2,
        category="bug",
        severity="low",
        title="Assumes the value is valid",
    )
    hinted_speculation = graph.Finding(
        line=8,
        category="bug",
        severity="low",
        title="May raise if the input is empty",
    )

    result = await graph.validate_node({
        "source": (
            "def f():\n    pass\n\n\n\n\n"
            "def g():\n    pass\n"
        ),
        "bugs": [speculative, hinted_speculation],
        "ctx": {
            "semgrep": [],
            "ruff": [
                {
                    "line": 8,
                    "rule": "B023",
                    "message": "Potential late-binding bug",
                    "category": "bug",
                }
            ],
        },
    })

    assert result["findings"] == [hinted_speculation]
    assert "reason=low_severity_speculation" in caplog.text


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
