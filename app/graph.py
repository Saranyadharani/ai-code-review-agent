"""Step 4: LangGraph workflow.  (bug || security) -> validate -> fix -> verify"""
import ast
import asyncio
import json
import logging
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
    ensure_finding_id,
    find_text_occurrence,
    is_style_only,
)
from app.verify import is_base_version_revert, verify_fixes

log = logging.getLogger("review-agent")
CRITIC_TIMEOUT_SECONDS = 20
QUALITY_FINDINGS_PER_FUNCTION = 3


def _configured_categories() -> set[str]:
    return {
        category.strip().lower()
        for category in settings.review_categories.split(",")
        if category.strip()
    }


def _function_for_line(line: int, functions: list[dict]) -> str:
    matches = [
        function
        for function in functions
        if isinstance(function.get("start"), int)
        and isinstance(function.get("end"), int)
        and function["start"] <= line <= function["end"]
    ]
    if not matches:
        return "<module>"
    return min(matches, key=lambda function: function["end"] - function["start"])["name"]

class Finding(BaseModel):
    id: str = ""
    line: int
    category: Literal["bug", "security", "performance", "quality", "dependency", "impact"]
    title: str
    severity: str = "medium"
    explanation: str = ""
    suggestion: str = ""
    source: str = "llm"  # llm | semgrep
    rule_id: str | None = None
    fix: dict | None = None
    critic_verdict: dict | None = None

    def model_post_init(self, __context) -> None:
        ensure_finding_id(self)


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
    if "quality" not in _configured_categories():
        return {"quality": []}
    hints = "\n".join(
        f"- line {h['line']}: {h['rule']}: {h['message']}"
        for h in s["ctx"].get("ruff", []) if h["category"] == "quality"
    ) or "none"
    data = await ask_json(
        "TASK: CODE QUALITY. Report actionable issues in CHANGED functions only: PEP 8 naming, deeply "
        "nested/long/complex functions, dead code, and missing docstrings or type hints. "
        f"Return no more than {QUALITY_FINDINGS_PER_FUNCTION} findings per function. Ruff hints:\n"
        f"{hints}\nReply as JSON object with key \"findings\" (a list). {KEYS}\n\n" + _header(s))
    findings = _build(data or [], "findings", "quality")
    counts: dict[str, int] = {}
    kept = []
    functions = s["ctx"].get("functions", [])
    for finding in sorted(findings, key=lambda item: (
        {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(item.severity.lower(), 4),
        item.line,
    )):
        function = _function_for_line(finding.line, functions)
        if counts.get(function, 0) >= QUALITY_FINDINGS_PER_FUNCTION:
            log.info(
                "finding_dropped id=%s category=quality reason=per_function_quality_cap:%s",
                ensure_finding_id(finding), function,
            )
            continue
        counts[function] = counts.get(function, 0) + 1
        kept.append(finding)
    return {"quality": kept}


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
        finding_id = ensure_finding_id(finding)
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
        else:
            log.info(
                "finding_dropped id=%s category=%s reason=no_retrieved_affected_caller",
                finding_id, finding.category,
            )
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
        candidate = Finding(**hint)
        if not any(
            finding.category == candidate.category
            and abs(finding.line - candidate.line) <= 2
            for finding in merged
        ):
            merged.append(candidate)
        else:
            log.info(
                "finding_dropped id=%s category=%s reason=static_hint_covered",
                ensure_finding_id(candidate), candidate.category,
            )
    allowed_categories = _configured_categories()
    filtered = []
    for finding in merged:
        category = correct_category(finding)
        if is_style_only(finding) and category != "security":
            if "quality" not in allowed_categories:
                log.info(
                    "finding_dropped id=%s category=%s reason=quality_not_enabled",
                    ensure_finding_id(finding), category,
                )
                continue
            finding.category = "quality"
            category = "quality"
        if category == "quality" and "quality" not in allowed_categories:
            log.info(
                "finding_dropped id=%s category=quality reason=quality_not_enabled",
                ensure_finding_id(finding),
            )
            continue
        if category in {"performance", "quality"} and finding.severity.lower() == "low":
            if category not in allowed_categories:
                log.info(
                    "finding_dropped id=%s category=%s reason=low_severity_noise",
                    ensure_finding_id(finding), category,
                )
                continue
        if is_style_only(finding) and "quality" not in allowed_categories:
            log.info(
                "finding_dropped id=%s category=%s reason=style_noise",
                ensure_finding_id(finding), category,
            )
            continue
        filtered.append(finding)
    deduplicated = deduplicate_findings(filtered)
    return {"findings": sorted(deduplicated, key=lambda finding: finding.line)}


async def fix_node(s: State):
    """Ask for fixes and an independent behavior-preservation critique."""
    quality_enabled = "quality" in _configured_categories()
    todo = [
        f for f in s["findings"]
        if f.severity in ("high", "medium")
        and f.category != "impact"
        and (quality_enabled or (
            f.category != "quality" and not is_style_only(f)
        ))
    ]
    eligible_ids = {ensure_finding_id(finding) for finding in todo}
    for finding in s["findings"]:
        if ensure_finding_id(finding) not in eligible_ids:
            log.info(
                "finding_decision id=%s category=%s decision=rejected source=no-fix "
                "reason=not_eligible_for_fix",
                ensure_finding_id(finding), finding.category,
            )
    if not todo:
        return {}
    listing = "\n".join(
        f"- finding_id={ensure_finding_id(f)} line={f.line} "
        f"category={f.category} title={f.title}"
        for f in todo
    )
    data = await ask_json(
        "TASK: FIXES. For each issue below propose a minimal fix. Preserve imports at the top of the file; "
        "never add imports inside a function or midway through executable module code. If an import cannot "
        "be safely placed with the existing top-level imports, provide advice only. "
        "Reply as JSON object with key \"fixes\" (a list); each item has finding_id, original "
        "(the EXACT existing text to replace), replacement, and note (a concise imperative "
        "instruction about this finding only; never mention another finding or its fix). "
        "Never use list positions or indices.\nIssues:\n" + listing + "\n\n" + _header(s))
    findings_by_id = {ensure_finding_id(finding): finding for finding in todo}
    fixes_by_id = {}
    for item in _items(data, "fixes"):
        try:
            finding_id = item["finding_id"]
            original = item["original"]
            replacement = item["replacement"]
            note = item.get("note", "")
        except (KeyError, TypeError):
            continue
        if (
            finding_id not in findings_by_id
            or not isinstance(original, str)
            or not isinstance(replacement, str)
            or not isinstance(note, str)
        ):
            continue
        finding = findings_by_id[finding_id]
        occurrence = find_text_occurrence(s["source"], original, finding.line)
        if occurrence is None:
            finding.fix = {
                "finding_id": finding_id,
                "original": original,
                "replacement": replacement,
                "note": note.strip(),
                "verified": False,
                "critic_safe": False,
                "problems": ["Original text was not found in the source."],
            }
            log.info(
                "finding_decision id=%s category=%s decision=rejected source=fix-agent "
                "reason=original_text_not_found",
                finding_id, finding.category,
            )
            continue
        fix = {
            "finding_id": finding_id,
            "original": original,
            "replacement": replacement,
            "start_line": occurrence[1],
            "note": note.strip(),
        }
        fixes_by_id[finding_id] = fix

    for finding_id, finding in findings_by_id.items():
        fix = fixes_by_id.get(finding_id)
        if fix is None:
            log.info(
                "finding_decision id=%s category=%s decision=rejected source=fix-agent "
                "reason=no_fix_proposed",
                finding_id, finding.category,
            )
            continue
        if is_base_version_revert(
            s["source"],
            s["ctx"].get("base_source"),
            s["ctx"].get("unified_diff", ""),
            finding,
            fix["original"],
            fix["replacement"],
        ):
            finding.critic_verdict = {
                "fixes_defect": True,
                "breaks_valid_use": False,
                "reason": "Restores the base-version line modified by this pull request.",
                "source": "base-revert",
            }
            fix["critic_verdict"] = finding.critic_verdict
            fix["critic_safe"] = True
            fix["base_revert"] = True
            finding.fix = fix
            log.info(
                "finding_decision id=%s category=%s decision=accepted source=base-revert",
                finding_id, finding.category,
            )
            continue

        prompt = (
            f"Finding id: {finding_id}\n"
            "Does this fix correct the flagged defect WITHOUT breaking valid uses? "
            "Changing the buggy behaviour is expected and allowed. Reject only if it changes the "
            "public signature, return type or data/file format for valid inputs, removes the flagged "
            "functionality, swaps the flagged code for a trivial value, or alters unrelated lines. "
            "Externalizing a hardcoded secret to an environment variable or configuration is an "
            "accepted security fix, not a behavior break. For an original security defect, adding a "
            "required configuration input is allowed.\n"
            "Reply strictly as JSON with exactly these keys and types: "
            '{"fixes_defect": bool, "breaks_valid_use": bool, "reason": str}.\n'
            f"Finding: {finding.title}\n{finding.explanation}\n"
            f"Original:\n{fix['original']}\nReplacement:\n{fix['replacement']}"
        )
        verdict = None
        critic_source = "deterministic-fallback"
        critic_error = "critic output unavailable"
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    ask_json(prompt), timeout=CRITIC_TIMEOUT_SECONDS
                )
                if (
                    isinstance(response, dict)
                    and set(response) == {"fixes_defect", "breaks_valid_use", "reason"}
                    and type(response["fixes_defect"]) is bool
                    and type(response["breaks_valid_use"]) is bool
                    and isinstance(response["reason"], str)
                ):
                    verdict = response
                    critic_source = "critic"
                    break
                critic_error = "critic returned an invalid verdict shape"
            except (asyncio.TimeoutError, TimeoutError):
                critic_error = "critic timed out"
            except Exception as error:
                critic_error = f"critic failed: {type(error).__name__}"
            if attempt == 0:
                log.info("critic_retry id=%s reason=%s", finding_id, critic_error)
        if verdict is not None:
            finding.critic_verdict = {**verdict, "source": "critic"}
            fix["critic_verdict"] = finding.critic_verdict
            fix["critic_safe"] = (
                verdict["fixes_defect"] and not verdict["breaks_valid_use"]
            )
            fix["critic_reason"] = verdict["reason"]
        else:
            finding.critic_verdict = None
            fix["critic_safe"] = True
            fix["critic_reason"] = ""
            fix["critic_failure"] = critic_error
        finding.fix = fix
        accepted = (
            verdict["fixes_defect"] and not verdict["breaks_valid_use"]
            if verdict is not None else None
        )
        log.info(
            "finding_decision id=%s category=%s decision=%s source=%s reason=%s",
            finding_id,
            finding.category,
            "accepted" if accepted is True else "rejected" if accepted is False else "pending-guards",
            critic_source,
            verdict["reason"] if verdict is not None else critic_error,
        )
    return {}


async def verify_node(s: State):
    return {
        "verify": await verify_fixes(
            s["source"],
            s["findings"],
            base_source=s["ctx"].get("base_source"),
            unified_diff=s["ctx"].get("unified_diff", ""),
        )
    }


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