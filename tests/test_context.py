import json
import subprocess

import pytest

from app import context


def test_semgrep_security_rules_detect_requested_patterns():
    source = """
import hashlib
import os
import pickle
import random
import subprocess
import yaml

AWS_KEY = "AKIA1234567890ABCDEF"
DATABASE_URL = "postgresql://review:secret@db.example/review"
os.system("echo unsafe")
subprocess.run("echo unsafe", shell=True)
yaml.load(payload, Loader=yaml.Loader)
pickle.loads(payload)
hashlib.md5(payload)
token = random.choice(values)
yaml.load(payload, Loader=yaml.SafeLoader)
"""

    hints = context.run_semgrep(source)
    rules = {hint["rule"] for hint in hints}

    assert {
        "aws-access-key",
        "database-uri-password",
        "os-system",
        "subprocess-shell-true",
        "yaml-unsafe-load",
        "pickle-loads",
        "hashlib-weak-hash",
        "random-credential",
    } <= rules
    unsafe_yaml_lines = [
        hint["line"] for hint in hints if hint["rule"] == "yaml-unsafe-load"
    ]
    assert unsafe_yaml_lines == [13]


def test_ruff_rules_are_mapped_by_category(monkeypatch):
    diagnostics = [
        {"code": "S307", "location": {"row": 2}, "message": "Use of eval"},
        {"code": "S110", "location": {"row": 2}, "message": "Try except pass"},
        {"code": "B023", "location": {"row": 3}, "message": "Late binding"},
        {"code": "PERF401", "location": {"row": 4}, "message": "Use a comprehension"},
        {"code": "F401", "location": {"row": 5}, "message": "Unused import"},
    ]
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, json.dumps(diagnostics), "")

    monkeypatch.setattr(context.subprocess, "run", fake_run)

    result = context.run_ruff("eval(value)\n")

    assert [item["category"] for item in result] == [
        "security", "bug", "bug", "performance", "quality",
    ]
    assert result[0] == {
        "rule": "S307",
        "line": 2,
        "message": "Use of eval",
        "category": "security",
    }
    assert result[1]["category"] == "bug"
    command, kwargs = calls[0]
    assert command[:3] == ["ruff", "check", "--isolated"]
    assert "lint.mccabe.max-complexity=10" in command
    assert command[-1] == "-"
    assert kwargs["input"] == "eval(value)\n"
    assert "--select" in command


@pytest.mark.asyncio
async def test_file_context_runs_analyzers_off_thread_and_filters_changed_lines(monkeypatch):
    monkeypatch.setattr(context, "run_semgrep", lambda source: [
        {"rule": "eval-use", "line": 2, "message": "unsafe", "severity": "ERROR"},
        {"rule": "eval-use", "line": 9, "message": "unchanged", "severity": "ERROR"},
    ])
    monkeypatch.setattr(context, "run_ruff", lambda source: [
        {"rule": "S307", "line": 2, "message": "unsafe", "category": "security"},
        {"rule": "B023", "line": 8, "message": "unchanged", "category": "bug"},
    ])
    patch = "@@ -1 +1,2 @@\n-old()\n+new()\n+eval(value)\n"

    result = await context.build_file_context("app.py", "def f():\n    eval(value)\n", patch)

    assert [hint["line"] for hint in result["semgrep"]] == [2]
    assert result["ruff"] == [
        {"rule": "S307", "line": 2, "message": "unsafe", "category": "security"}
    ]


@pytest.mark.asyncio
async def test_file_context_includes_changed_top_level_module_code(monkeypatch):
    monkeypatch.setattr(context, "run_semgrep", lambda source: [])
    monkeypatch.setattr(context, "run_ruff", lambda source: [])
    source = (
        "import os\n"
        "DEFAULT_PATH = '/tmp/data'\n"
        "configured_path = os.getenv('DATA_PATH', DEFAULT_PATH)\n"
        "\n"
        "def load_data():\n"
        "    return configured_path\n"
    )
    patch = "@@ -2 +2 @@\n-DEFAULT_PATH = '/data'\n+DEFAULT_PATH = '/tmp/data'\n"

    result = await context.build_file_context("settings.py", source, patch)

    assert result["changed_functions"] == ["<module>"]
    module_unit = next(item for item in result["functions"] if item["name"] == "<module>")
    assert module_unit["lines"] == result["module_context_lines"]
    assert 2 in result["module_context_lines"]
    assert max(result["module_context_lines"]) <= len(source.splitlines())
    assert len(result["module_context_lines"]) <= 150
