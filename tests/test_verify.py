import pytest

from app import verify


@pytest.mark.asyncio
async def test_behavior_preserving_axis_fix_is_verified(monkeypatch):
    monkeypatch.setattr(verify, "run_semgrep", lambda source: [])
    source = (
        "import numpy as np\n"
        "def compute(predictions):\n"
        "    preds = np.argmax(predictions, axis=0)\n"
        "    return preds\n"
    )
    finding = {
        "line": 3,
        "title": "Incorrect prediction axis",
        "explanation": "The predictions use the wrong axis.",
        "fix": {
            "original": "axis=0",
            "replacement": "axis=1",
            "critic_safe": True,
        },
    }

    result = await verify.verify_fixes(source, [finding])

    assert result["accepted"] == 1
    assert finding["fix"]["verified"] is True
    assert finding["fix"]["problems"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename,source,original,replacement,reason",
    [
        (
            "pickle",
            "import pickle\nimport json\n"
            "def load(stream):\n    return pickle.load(stream)\n",
            "pickle.load(stream)",
            "json.load(stream)",
            "return type or shape",
        ),
        (
            "eval",
            "import ast\n"
            "def parse(value):\n    return eval(value)\n",
            "eval(value)",
            "ast.literal_eval(value)",
            "return type or shape",
        ),
        (
            "signature",
            "def compute(value):\n    return value\n",
            "def compute(value):",
            "def compute(value, other):",
            "signature",
        ),
        (
            "trivial",
            "result = evaluate(value)\n",
            "evaluate(value)",
            "None",
            "trivial value",
        ),
        (
            "import-placement",
            "unsafe_call()\n",
            "unsafe_call()",
            "safe_call()\nimport ast",
            "top-level imports",
        ),
        (
            "docstring",
            '"""module docs"""\nunsafe_call()\n',
            '"""module docs"""',
            'import ast\n"""module docs"""',
            "top-level imports",
        ),
    ],
)
async def test_generic_unsafe_fix_guards_reject_invalid_suggestions(
    monkeypatch, filename, source, original, replacement, reason
):
    monkeypatch.setattr(verify, "run_semgrep", lambda source: [])
    finding = {
        "line": 1,
        "title": "Review finding",
        "explanation": "A proposed replacement requires review.",
        "fix": {
            "original": original,
            "replacement": replacement,
            "critic_safe": True,
        },
    }

    result = await verify.verify_fixes(source, [finding])

    assert result["accepted"] == 0
    assert finding["fix"]["verified"] is False
    assert any(reason in problem for problem in finding["fix"]["problems"])


@pytest.mark.asyncio
async def test_signature_change_is_allowed_for_signature_finding(monkeypatch):
    monkeypatch.setattr(verify, "run_semgrep", lambda source: [])
    finding = {
        "line": 1,
        "title": "Function signature change",
        "explanation": "The caller contract needs an additional argument.",
        "fix": {
            "original": "def compute(value):",
            "replacement": "def compute(value, other):",
            "critic_safe": True,
        },
    }

    result = await verify.verify_fixes(
        "def compute(value):\n    return value\n", [finding]
    )

    assert result["accepted"] == 1
    assert finding["fix"]["verified"] is True


@pytest.mark.asyncio
async def test_fix_with_new_undefined_name_is_rejected(monkeypatch):
    monkeypatch.setattr(verify, "run_semgrep", lambda source: [])
    finding = {
        "line": 2,
        "title": "Undefined name",
        "explanation": "A name is missing.",
        "fix": {
            "original": "return value",
            "replacement": "return missing_name",
            "critic_safe": True,
        },
    }

    result = await verify.verify_fixes(
        "def f():\n    return value\n", [finding]
    )

    assert result["accepted"] == 0
    assert finding["fix"]["verified"] is False
    assert any("undefined names" in problem for problem in finding["fix"]["problems"])
