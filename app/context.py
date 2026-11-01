"""Step 2: turn a changed file into structured context (Tree-sitter + Semgrep)."""
import asyncio
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from app.diffing import classify_findings, semantic_diff

_PARSER = Parser(Language(tspython.language()))
RULES = Path(__file__).resolve().parent.parent / "rules" / "python.yml"
log = logging.getLogger(__name__)
RUFF_SECURITY = "S"
RUFF_SELECT = (
    f"{RUFF_SECURITY},B,F,A001,A002,E722,PLW,PERF,SIM115,C4,C901,"
    "PLR0912,PLR0913,PLR0915,F401,F841,N,E501,I,D1,ANN,ERA"
)


def _ruff_category(rule: str) -> str:
    if rule == "S110":
        return "bug"
    if rule.startswith("S"):
        return "security"
    if rule.startswith(("PERF", "SIM115", "C4")):
        return "performance"
    if rule in {"C901", "PLR0912", "PLR0913", "PLR0915", "F401", "F841"} or rule.startswith(
        ("N", "E501", "I", "D1", "ANN", "ERA")
    ):
        return "quality"
    return "bug"


def changed_lines(patch: str | None) -> set[int]:
    """New-file line numbers added/modified in a unified diff patch."""
    lines, cur = set(), 0
    for ln in (patch or "").splitlines():
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", ln)
        if m:
            cur = int(m.group(1))
        elif ln.startswith("+"):
            lines.add(cur)
            cur += 1
        elif not ln.startswith("-"):
            cur += 1
    return lines


def _module_context_lines(
    source: str, changed: set[int], info: dict, max_lines: int = 150
) -> list[int]:
    lines = source.splitlines()
    enclosed = {
        number
        for unit in (*info["functions"], *info["classes"])
        for number in range(unit["start"], unit["end"] + 1)
    }
    top_level_changes = sorted(
        number for number in changed if number not in enclosed and 1 <= number <= len(lines)
    )
    if not top_level_changes:
        return []

    intervals = []
    for number in top_level_changes:
        start, end = max(1, number - 10), min(len(lines), number + 10)
        if intervals and start <= intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    selected = [
        number
        for start, end in intervals
        for number in range(start, end + 1)
    ]
    return selected[:max_lines]


def parse_python(source: str) -> dict:
    """Functions, classes and imports with 1-based line ranges."""
    src = source.encode()
    out = {"functions": [], "classes": [], "imports": []}

    def text(n):
        return src[n.start_byte:n.end_byte].decode()

    def walk(node, cls=None):
        for c in node.children:
            span = (c.start_point[0] + 1, c.end_point[0] + 1)
            if c.type == "function_definition":
                name = text(c.child_by_field_name("name"))
                out["functions"].append(
                    {"name": f"{cls}.{name}" if cls else name, "start": span[0], "end": span[1]})
                walk(c, cls)
            elif c.type == "class_definition":
                name = text(c.child_by_field_name("name"))
                out["classes"].append({"name": name, "start": span[0], "end": span[1]})
                walk(c, name)
            elif c.type in ("import_statement", "import_from_statement"):
                out["imports"].append(text(c))
            else:
                walk(c, cls)

    walk(_PARSER.parse(src).root_node)
    return out


def run_semgrep(source: str) -> list[dict]:
    """Run local rules. Returns [] if semgrep is unavailable."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "target.py"
        f.write_text(source)
        try:
            r = subprocess.run(
                ["semgrep", "--config", str(RULES), "--json", "--quiet", "--metrics=off", str(f)],
                capture_output=True, text=True, timeout=120)
            res = json.loads(r.stdout or "{}").get("results", [])
        except OSError as error:
            log.error("Semgrep is unavailable: %s", error)
            return []
        except subprocess.TimeoutExpired:
            log.error("Semgrep scan timed out")
            return []
        except json.JSONDecodeError as error:
            log.error("Semgrep returned invalid JSON: %s", error)
            return []
        if r.returncode not in (0, 1):
            log.error("Semgrep failed with exit code %d: %s", r.returncode, r.stderr[:300])
            return []
    return [{"rule": x["check_id"].split(".")[-1], "line": x["start"]["line"],
             "severity": x["extra"]["severity"], "message": x["extra"]["message"]} for x in res]


def run_ruff(source: str) -> list[dict]:
    """Run selected local Ruff rules, returning categorized findings."""
    try:
        result = subprocess.run(
            [
                "ruff", "check", "--isolated", "--config",
                "lint.mccabe.max-complexity=10", "--output-format", "json",
                "--select", RUFF_SELECT, "-",
            ],
            input=source,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode not in (0, 1):
            log.error("Ruff failed with exit code %d: %s", result.returncode, result.stderr[:300])
            return []
        diagnostics = json.loads(result.stdout or "[]")
    except OSError as error:
        log.error("Ruff is unavailable: %s", error)
        return []
    except subprocess.TimeoutExpired:
        log.error("Ruff scan timed out")
        return []
    except json.JSONDecodeError as error:
        log.error("Ruff returned invalid JSON: %s", error)
        return []
    return [
        {
            "rule": item["code"],
            "line": item["location"]["row"],
            "message": item["message"],
            "category": _ruff_category(item["code"]),
        }
        for item in diagnostics
    ]


async def build_file_context(
    filename: str,
    source: str,
    patch: str | None,
    base_source: str | None = None,
) -> dict:
    """Everything the agents need for one changed Python file."""
    ch = changed_lines(patch)
    info = parse_python(source)
    touched = [fn["name"] for fn in info["functions"]
               if any(fn["start"] <= n <= fn["end"] for n in ch)]
    module_lines = _module_context_lines(source, ch, info)
    if module_lines:
        touched.append("<module>")
        info["functions"].append({
            "name": "<module>",
            "start": module_lines[0],
            "end": module_lines[-1],
            "lines": module_lines,
        })
    if base_source is None:
        semgrep_hints, ruff_hints = await asyncio.gather(
            asyncio.to_thread(run_semgrep, source),
            asyncio.to_thread(run_ruff, source),
        )
        hints = [h for h in semgrep_hints if h["line"] in ch]
        ruff = [h for h in ruff_hints if h["line"] in ch]
        baseline = None
    else:
        base_semgrep, head_semgrep, base_ruff, head_ruff = await asyncio.gather(
            asyncio.to_thread(run_semgrep, base_source),
            asyncio.to_thread(run_semgrep, source),
            asyncio.to_thread(run_ruff, base_source),
            asyncio.to_thread(run_ruff, source),
        )
        semgrep_comparison = classify_findings(
            base_semgrep, head_semgrep, base_source, source
        )
        ruff_comparison = classify_findings(
            base_ruff, head_ruff, base_source, source
        )
        hints = [
            {**hint, "category": "security"}
            for hint in semgrep_comparison
            if hint["status"] == "new" and hint["line"] in ch
        ]
        ruff = [
            hint for hint in ruff_comparison
            if hint["status"] == "new" and hint["line"] in ch
        ]
        counts = {"new": 0, "pre-existing": 0, "fixed": 0}
        for finding in semgrep_comparison + ruff_comparison:
            counts[finding["status"]] += 1
        baseline = counts
    result = {
        "file": filename,
        "imports": info["imports"],
        "classes": [c["name"] for c in info["classes"]],
        "functions": info["functions"],
        "module_context_lines": module_lines,
        "changed_functions": touched,
        "semgrep": hints,
        "ruff": ruff,
        "unified_diff": patch or "",
        "semantic_diff": semantic_diff(base_source, source),
        "base_source": base_source,
    }
    if baseline is not None:
        result["baseline_comparison"] = baseline
    return result