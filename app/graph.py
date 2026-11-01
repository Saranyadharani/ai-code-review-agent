"""Step 4: LangGraph workflow.  (bug || security) -> validate -> fix -> verify"""
import ast
import json
import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.config import settings
from app.llm import ask_json
from app.review_rules import (
    category_for_rule,
    correct_category,
    deduplicate_findings,
    find_text_occurrence,
    is_style_only,
)
from app.verify import verify_fixes

class Finding(BaseModel):
    line: int
    category: Literal["bug", "security", "performance", "quality", "dependency", "impact"]
    title: str
    severity: str = "medium"
    explanation: str = ""
    suggestion: str = ""
    source: str = "llm"  # llm | semgrep
    rule_id: str | None = None
    fix: dict | None = None


class State(TypedDict, total=False):
    filename: str
    source: str
    ctx: dict
    bugs: list[Finding]
    security: list[Finding]
    performance: list[Finding]
    quality: list[Finding]
    impact: list[Finding]
    findings: list[Finding]
    verify: dict


def _items(data, key):
    """Tolerate odd model output shapes; return a list of dicts."""
    if isinstance(data, dict):
        v = data.get(key)
        if v is None:
            v = next((x for x in data.values() if isinstance(x, list)), [])
        data = [v] if isinstance(v, dict) else v
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def _build(data, key, category):
    out = []
    for it in _items(data, key):
        try:
            out.append(Finding(category=category, **{k: it[k] for k in
                       ("line", "severity", "title", "explanation", "suggestion",
                        "rule_id", "source") if k in it}))
        except (ValueError, TypeError):
            pass
    return out


def _header(s: State) -> str:
    lines = s["source"].splitlines()
    import_text = set(s["ctx"].get("imports", []))
    selected_lines = {
        number
        for number, line in enumerate(lines, 1)
        if line.strip() in import_text
    }
    try:
        tree = ast.parse(s["source"])
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                selected_lines.update(
                    range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1)
                )
    changed = set(s["ctx"].get("changed_functions", []))
    module_lines = s["ctx"].get("module_context_lines", [])
    if "<module>" in changed:
        selected_lines.update(module_lines)
    for function in s["ctx"].get("functions", []):
        if function["name"] in changed:
            if function["name"] == "<module>":
                selected_lines.update(function.get("lines", module_lines))
            else:
                selected_lines.update(range(function["start"], function["end"] + 1))
    numbered = "\n".join(
        f"{number}: {lines[number - 1]}"
        for number in sorted(selected_lines)
        if 1 <= number <= len(lines)
    )
    semantic = json.dumps(s["ctx"].get("semantic_diff", {}), sort_keys=True)
    diff = s["ctx"].get("unified_diff", "")
    return (f"File: {s['filename']}\nChanged functions: {', '.join(s['ctx']['changed_functions']) or 'none'}\n"
            f"Imports, changed-function and module-level source:\n"
            f"{numbered or '(no changed source found)'}\n"
            f"Unified PR diff:\n{diff or '(unavailable)'}\n"
            f"Semantic diff summary:\n{semantic}\n"
            f"Related code:\n{s['ctx'].get('related_code', {}).get('text') or '(unavailable)'}\n")


KEYS = ("Each item has keys: line (int, from the numbered source), severity (high|medium|low), "
        "title (short), explanation (1-2 sentences), suggestion (the fix).")


async def bug_node(s: State):
    hints = "\n".join(
        f"- line {h['line']}: {h['rule']}: {h['message']}"
        for h in s["ctx"].get("ruff", []) if h["category"] == "bug"
    ) or "none"
    data = await ask_json(
        "TASK: BUGS. You are a senior engineer. Find logic bugs and missed edge cases (off-by-one, "
        "division by zero, mutable defaults, wrong conditions, swallowed errors, late-binding closures, "
        "shadowed builtins, UnboundLocalError risks, `except Exception: pass`, wrong except ordering, "
        "and likely TypeError or AttributeError) in the CHANGED functions only. No security issues. "
        f"Ruff hints:\n{hints}\nReply as JSON object with key \"bugs\" (a list). "
        f"{KEYS}\n\n" + _header(s))
    return {"bugs": _build(data or [], "bugs", "bug")}


async def security_node(s: State):
    hints = "\n".join(
        f"- line {h['line']}: {h['rule']}: {h['message']}"
        for h in s["ctx"]["semgrep"]
    )
    hints += "\n".join(
        f"\n- line {h['line']}: {h['rule']}: {h['message']}"
        for h in s["ctx"].get("ruff", []) if h["category"] == "security"
    )
    hints = hints or "none"
    data = await ask_json(
        "TASK: SECURITY. You are a security reviewer. Report real vulnerabilities in the CHANGED functions "
        "(injection, unsafe deserialization, eval, hardcoded secrets, path traversal, ...). Static-analysis "
        f"hints follow; confirm the real ones and drop false positives.\nHints:\n{hints}\n"
        f"Reply as JSON object with key \"findings\" (a list). {KEYS}\n\n" + _header(s))
    return {"security": _build(data or [], "findings", "security")}


async def performance_node(s: State):
    hints = "\n".join(
        f"- line {h['line']}: {h['rule']}: {h['message']}"
        for h in s["ctx"].get("ruff", []) if h["category"] == "performance"
    ) or "none"
    data = await ask_json(
        "TASK: PERFORMANCE. Find meaningful performance issues in CHANGED functions only: string += "
        "inside loops, `x in list` where a set or dict fits, heavy work or database queries inside loops "
        "that can be hoisted, unclosed files/sockets/connections (suggest `with`), and generator "
        f"opportunities. Ruff hints:\n{hints}\nReply as JSON object with key \"findings\" (a list). "
        f"{KEYS}\n\n" + _header(s))
    return {"performance": _build(data or [], "findings", "performance")}


async def quality_node(s: State):
    hints = "\n".join(
        f"- line {h['line']}: {h['rule']}: {h['message']}"
        for h in s["ctx"].get("ruff", []) if h["category"] == "quality"
    ) or "none"
    data = await ask_json(
        "TASK: CODE QUALITY. Report actionable issues in CHANGED functions only: PEP 8 naming, deeply "
        "nested/long/complex functions, dead code, and missing docstrings or type hints. Ruff hints:\n"
        f"{hints}\nReply as JSON object with key \"findings\" (a list). {KEYS}\n\n" + _header(s))
    return {"quality": _build(data or [], "findings", "quality")}


async def impact_node(s: State):
    related = s["ctx"].get("related_code", {})
    callers = [
        caller for caller in related.get("callers", [])
        if isinstance(caller, dict)
        and isinstance(caller.get("file"), str)
        and isinstance(caller.get("line"), int)
        and isinstance(caller.get("symbol"), str)
    ]
    if not callers:
        return {"impact": []}
    test_status = (
        "tests changed" if related.get("tests_changed") is True
        else "no test changes found" if related.get("tests_changed") is False
        else "test-change status unavailable"
    )
    semantic = s["ctx"].get("semantic_diff", {})
    removed_symbols = {
        symbol.rsplit(".", 1)[-1]
        for symbol in semantic.get("removed_functions", [])
        if isinstance(symbol, str)
    }
    caller_list = "\n".join(
        f"- {caller['symbol']} referenced at {caller['file']}:{caller['line']}"
        for caller in callers
    )
    prompt = (
        "TASK: IMPACT ANALYSIS. Use the semantic diff and Related code to identify removed or renamed "
        "functions still referenced, changed signatures with callers not updated, changed return shapes, "
        "removed imports still used, and behavior changes without test changes. Report a finding only "
        "when it names a retrieved affected caller using the exact path:line shown below, or names a "
        "removed function that the retrieved callers confirm is still referenced. If none apply, return "
        "an empty findings list. Do not report a general lack of test changes by itself. "
        f"Retrieved callers:\n{caller_list}\n"
        f"Related-test status: {test_status}.\n"
        f"Reply as JSON object with key \"findings\" (a list). {KEYS}\n\n" + _header(s)
    )
    data = await ask_json(prompt)
    findings = _build(data or [], "findings", "impact")
    accepted = []
    for finding in findings:
        description = f"{finding.title}\n{finding.explanation}"
        if any(f"{caller['file']}:{caller['line']}" in description for caller in callers):
            accepted.append(finding)
            continue
        mentions_referenced_removed_symbol = any(
            caller["symbol"].rsplit(".", 1)[-1] in removed_symbols
            and re.search(
                rf"\b{re.escape(caller['symbol'].rsplit('.', 1)[-1])}\b",
                description,
            )
            for caller in callers
        )
        if mentions_referenced_removed_symbol:
            accepted.append(finding)
    return {"impact": accepted}


async def validate_node(s: State):
    """Correct categories and merge duplicate findings across agents and analyzers."""
    merged = (
        s.get("bugs", []) + s.get("security", [])
        + s.get("performance", []) + s.get("quality", [])
        + s.get("impact", [])
    )
    static_hints = [
        {
            "line": h["line"],
            "category": category_for_rule(h.get("rule"), "bug"),
            "title": h["message"],
            "severity": "high" if h["severity"] == "ERROR" else "medium",
            "source": "semgrep",
            "rule_id": h.get("rule"),
        }
        for h in s["ctx"].get("semgrep", [])
    ]
    static_hints.extend(
        {
            "line": h["line"],
            "category": h["category"],
            "title": h["message"],
            "severity": "medium",
            "source": "ruff",
            "rule_id": h["rule"],
        }
        for h in s["ctx"].get("ruff", [])
    )
    for hint in static_hints:
        if not any(
            finding.category == hint["category"]
            and abs(finding.line - hint["line"]) <= 2
            for finding in merged
        ):
            merged.append(Finding(**hint))
    allowed_categories = {
        category.strip().lower()
        for category in settings.review_categories.split(",")
        if category.strip()
    }
    filtered = []
    for finding in merged:
        category = correct_category(finding)
        if category in {"performance", "quality"} and finding.severity.lower() == "low":
            if category not in allowed_categories:
                continue
        if is_style_only(finding) and "quality" not in allowed_categories:
            continue
        filtered.append(finding)
    deduplicated = deduplicate_findings(filtered)
    return {"findings": sorted(deduplicated, key=lambda finding: finding.line)}


async def fix_node(s: State):
    """Ask for fixes and an independent behavior-preservation critique."""
    todo = [
        (i, f) for i, f in enumerate(s["findings"])
        if f.severity in ("high", "medium") and f.category != "impact"
    ]
    if not todo:
        return {}
    listing = "\n".join(f"{i}: line {f.line}: {f.title}" for i, f in todo)
    data = await ask_json(
        "TASK: FIXES. For each issue below propose a minimal fix. Preserve imports at the top of the file; "
        "never add imports inside a function or midway through executable module code. If an import cannot "
        "be safely placed with the existing top-level imports, provide advice only. "
        "Reply as JSON object with key \"fixes\" "
        "(a list); each item has index (int), original (the EXACT existing text to replace, copied from the "
        "source without line numbers), replacement (the new text).\nIssues:\n" + listing + "\n\n" + _header(s))
    for it in _items(data, "fixes"):
        try:
            f, orig, new = s["findings"][int(it["index"])], it["original"], it["replacement"]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        if not isinstance(orig, str) or not isinstance(new, str):
            continue
        occurrence = find_text_occurrence(s["source"], orig, f.line)
        if occurrence is None:
            continue
        critique = await ask_json(
            "TASK: FIX CRITIC. Does this fix preserve behaviour for valid inputs? "
            "Reply JSON {safe: bool, reason}. Evaluate the original and proposed code conservatively.\n"
            f"Finding: {f.title}\n{f.explanation}\nOriginal:\n{orig}\nReplacement:\n{new}"
        )
        safe = isinstance(critique, dict) and critique.get("safe") is True
        reason = (
            str(critique.get("reason", "")).strip()
            if isinstance(critique, dict) else ""
        )
        f.fix = {
            "original": orig,
            "replacement": new,
            "start_line": occurrence[1],
            "critic_safe": safe,
            "critic_reason": reason or ("Critic rejected behavior preservation." if not safe else ""),
        }
        if not safe:
            f.suggestion = reason or "The proposed fix could not be confirmed as behavior-preserving."
    return {}


async def verify_node(s: State):
    return {"verify": await verify_fixes(s["source"], s["findings"])}


def _build_graph():
    g = StateGraph(State)
    for name, fn in (("bug", bug_node), ("security", security_node),
                     ("performance", performance_node), ("quality", quality_node),
                     ("impact", impact_node),
                     ("validate", validate_node), ("fix", fix_node), ("verify", verify_node)):
        g.add_node(name, fn)
    g.add_edge(START, "bug")
    g.add_edge(START, "security")
    g.add_edge(START, "performance")
    g.add_edge(START, "quality")
    g.add_edge(START, "impact")
    g.add_edge("bug", "validate")
    g.add_edge("security", "validate")
    g.add_edge("performance", "validate")
    g.add_edge("quality", "validate")
    g.add_edge("impact", "validate")
    g.add_edge("validate", "fix")
    g.add_edge("fix", "verify")
    g.add_edge("verify", END)
    return g.compile()


GRAPH = _build_graph()


async def run_graph(filename: str, source: str, ctx: dict) -> State:
    return await GRAPH.ainvoke({"filename": filename, "source": source, "ctx": ctx})