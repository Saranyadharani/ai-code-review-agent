"""Verify proposed fixes without executing PR code."""
import ast
import asyncio
import io
import logging
import re
from collections import Counter

from pyflakes.api import check
from pyflakes.reporter import Reporter

from app.context import run_semgrep
from app.review_rules import (
    ensure_finding_id,
    find_text_occurrence,
    finding_concerns,
    finding_value,
)

log = logging.getLogger("review-agent")


def _finding_value(finding, key: str, default=""):
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def undefined_names(code: str) -> set[str]:
    """Pyflakes 'undefined name' messages, without line numbers."""
    out = io.StringIO()
    check(code, "f.py", Reporter(out, out))
    return {
        line.split(":", 3)[-1].strip()
        for line in out.getvalue().splitlines()
        if "undefined name" in line
    }


def _function_signatures(code: str) -> dict[str, str]:
    tree = ast.parse(code)
    signatures = {}

    def walk(body, scope=()):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = ".".join((*scope, node.name))
                signatures[key] = ast.dump(
                    ast.arguments(
                        posonlyargs=node.args.posonlyargs,
                        args=node.args.args,
                        vararg=node.args.vararg,
                        kwonlyargs=node.args.kwonlyargs,
                        kw_defaults=node.args.kw_defaults,
                        kwarg=node.args.kwarg,
                        defaults=node.args.defaults,
                    ),
                    include_attributes=False,
                ) + (
                    ast.dump(node.returns, include_attributes=False)
                    if node.returns is not None else ""
                )
                walk(node.body, (*scope, node.name))
            elif isinstance(node, ast.ClassDef):
                walk(node.body, (*scope, node.name))

    walk(tree.body)
    return signatures


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return type(node).__name__


def _expression_shape(node: ast.AST | None):
    if node is None:
        return None
    if isinstance(node, ast.Call):
        return ("call", _call_name(node.func))
    if isinstance(node, ast.Constant):
        return ("constant", type(node.value).__name__)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return (type(node).__name__, len(node.elts))
    if isinstance(node, ast.Dict):
        return ("dict", len(node.keys))
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Attribute):
        return ("attribute", _call_name(node))
    if isinstance(node, ast.BinOp):
        return (
            "binary",
            type(node.op).__name__,
            _expression_shape(node.left),
            _expression_shape(node.right),
        )
    if isinstance(node, ast.UnaryOp):
        return ("unary", type(node.op).__name__, _expression_shape(node.operand))
    if isinstance(node, ast.IfExp):
        return (
            "conditional",
            _expression_shape(node.body),
            _expression_shape(node.orelse),
        )
    return type(node).__name__


def _return_shapes(code: str) -> dict[str, tuple]:
    tree = ast.parse(code)
    shapes = {}

    def expression_shape(node: ast.AST | None, bindings: dict[str, object]):
        if isinstance(node, ast.Name) and node.id in bindings:
            return ("bound-value", bindings[node.id])
        return _expression_shape(node)

    def inspect_statements(body, collected, bindings):
        for node in body:
            if isinstance(node, ast.Return):
                collected.append(expression_shape(node.value, bindings))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            else:
                if isinstance(node, ast.Assign):
                    shape = _expression_shape(node.value)
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            bindings[target.id] = shape
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    bindings[node.target.id] = _expression_shape(node.value)
                inspect_statements(ast.iter_child_nodes(node), collected, bindings)

    def walk(body, scope=()):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = ".".join((*scope, node.name))
                found = []
                inspect_statements(node.body, found, {})
                shapes[key] = tuple(found)
                walk(node.body, (*scope, node.name))
            elif isinstance(node, ast.ClassDef):
                walk(node.body, (*scope, node.name))

    walk(tree.body)
    return shapes


def _imports(code: str) -> tuple[Counter, Counter]:
    tree = ast.parse(code)
    all_imports = Counter(
        ast.dump(node, include_attributes=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    leading_imports = Counter()
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            leading_imports[ast.dump(node, include_attributes=False)] += 1
        else:
            break
    return all_imports, leading_imports


def _module_docstring(code: str) -> str | None:
    body = ast.parse(code).body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value.value
    return None


def _imports_are_at_top(before: str, after: str) -> bool:
    if _module_docstring(before) != _module_docstring(after):
        return False
    before_all, before_top = _imports(before)
    after_all, after_top = _imports(after)
    new_imports = after_all - before_all
    new_top_imports = after_top - before_top
    return new_imports == new_top_imports


def _fragment_expression(source: str) -> ast.AST | None:
    try:
        return ast.parse(source, mode="eval").body
    except SyntaxError:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
    if len(tree.body) != 1:
        return None
    statement = tree.body[0]
    if isinstance(statement, ast.Return):
        return statement.value
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return statement.value
    if isinstance(statement, ast.Expr):
        return statement.value
    return None


def _fix_problems(source: str, patched: str, finding, fix: dict) -> list[str]:
    problems = []
    try:
        ast.parse(source)
        ast.parse(patched)
    except SyntaxError as error:
        return [f"Patched code does not parse: {error.msg}"]

    new_names = undefined_names(patched) - undefined_names(source)
    if new_names:
        problems.append("Introduces undefined names: " + ", ".join(sorted(new_names)))

    is_security_defect = finding_value(finding, "category", "") == "security"
    if not fix.get("base_revert") and not is_security_defect and (
        not finding_concerns(finding, "signature")
        and _function_signatures(source) != _function_signatures(patched)
    ):
        problems.append("Changes a function signature.")

    if not fix.get("base_revert") and (
        not finding_concerns(finding, "return_shape")
        and _return_shapes(source) != _return_shapes(patched)
    ):
        problems.append("Changes a function return type or shape.")

    original_expr = _fragment_expression(fix["original"])
    replacement_expr = _fragment_expression(fix["replacement"])
    original_calls = (
        any(isinstance(node, ast.Call) for node in ast.walk(original_expr))
        if original_expr is not None else False
    )
    replacement_calls = (
        any(isinstance(node, ast.Call) for node in ast.walk(replacement_expr))
        if replacement_expr is not None else False
    )
    if original_calls and not replacement_calls:
        problems.append("Removes the flagged call instead of replacing it safely.")
    if original_expr is not None and replacement_expr is not None:
        trivial = (
            isinstance(replacement_expr, ast.Constant)
            or isinstance(replacement_expr, (ast.List, ast.Tuple, ast.Dict, ast.Set))
            and not getattr(replacement_expr, "elts", getattr(replacement_expr, "keys", []))
        )
        nontrivial_original = not isinstance(
            original_expr, (ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Dict, ast.Set)
        )
        if trivial and nontrivial_original:
            problems.append("Replaces a nontrivial expression with a trivial value.")

    if not _imports_are_at_top(source, patched):
        problems.append("New imports must be placed with the top-level imports.")
    if fix.get("critic_safe") is False:
        problems.append("The critic rejected this fix as unsafe for valid uses.")
    return problems


def is_base_version_revert(
    source: str,
    base_source: str | None,
    unified_diff: str,
    finding,
    original: str,
    replacement: str,
) -> bool:
    """Return true when a fix restores the base text for a PR-modified line."""
    if base_source is None or not unified_diff or not replacement:
        return False
    occurrence = find_text_occurrence(
        source, original, _finding_value(finding, "line", None)
    )
    if occurrence is None:
        return False
    offset, start_line = occurrence
    patched = source[:offset] + replacement + source[offset + len(original):]
    patched_lines = patched.splitlines()
    base_lines = base_source.splitlines()

    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    old_line = new_line = 0
    block_old_start = block_new_start = 0
    deleted: list[str] = []
    added: list[str] = []

    def matches_block() -> bool:
        if not added or not deleted or not (
            block_new_start <= start_line < block_new_start + len(added)
        ):
            return False
        head_index, base_index = block_new_start - 1, block_old_start - 1
        return (
            patched_lines[head_index:head_index + len(added)] == deleted
            and base_lines[base_index:base_index + len(deleted)] == deleted
        )

    for line in unified_diff.splitlines():
        match = hunk_pattern.match(line)
        if match:
            if matches_block():
                return True
            old_line = int(match.group(1))
            new_line = int(match.group(3))
            deleted = []
            added = []
            block_new_start = new_line
        elif line.startswith("\\"):
            continue
        elif line.startswith("-"):
            if not deleted and not added:
                block_old_start, block_new_start = old_line, new_line
            deleted.append(line[1:])
            old_line += 1
        elif line.startswith("+"):
            if not deleted and not added:
                block_old_start, block_new_start = old_line, new_line
            added.append(line[1:])
            new_line += 1
        else:
            if matches_block():
                return True
            deleted = []
            added = []
            old_line += 1
            new_line += 1
    return matches_block()


def _finding_fix(finding):
    return finding.get("fix") if isinstance(finding, dict) else getattr(finding, "fix", None)


def _set_verified(finding, fix: dict, verified: bool, problems: list[str]) -> None:
    fix["verified"] = verified
    fix["problems"] = problems
    if isinstance(finding, dict):
        finding["fix"] = fix


async def verify_fixes(
    source: str,
    findings: list,
    base_source: str | None = None,
    unified_diff: str = "",
) -> dict:
    """Verify each proposed fix, then check all accepted fixes together."""
    accepted = []
    for finding in findings:
        finding_id = ensure_finding_id(finding)
        fix = _finding_fix(finding)
        if not isinstance(fix, dict):
            log.info(
                "finding_decision id=%s category=%s decision=rejected source=verify "
                "reason=no-fix",
                finding_id, _finding_value(finding, "category", ""),
            )
            continue
        original, replacement = fix.get("original"), fix.get("replacement")
        if not isinstance(original, str) or not original or not isinstance(replacement, str):
            _set_verified(finding, fix, False, ["Fix must include exact original and replacement text."])
            log.info(
                "finding_decision id=%s category=%s decision=rejected source=verify "
                "reason=invalid-fix-shape",
                finding_id, _finding_value(finding, "category", ""),
            )
            continue
        occurrence = find_text_occurrence(
            source, original, fix.get("start_line", _finding_value(finding, "line", None))
        )
        if occurrence is None:
            _set_verified(finding, fix, False, ["Original text was not found in the source."])
            log.info(
                "finding_decision id=%s category=%s decision=rejected source=verify "
                "reason=original-text-not-found",
                finding_id, _finding_value(finding, "category", ""),
            )
            continue
        offset, start_line = occurrence
        fix["start_line"] = start_line
        if base_source is not None and not fix.get("base_revert"):
            fix["base_revert"] = is_base_version_revert(
                source, base_source, unified_diff, finding, original, replacement
            )
            if fix["base_revert"]:
                fix["critic_safe"] = True
        patched = source[:offset] + replacement + source[offset + len(original):]
        problems = _fix_problems(source, patched, finding, fix)
        _set_verified(finding, fix, not problems, problems)
        if not problems:
            accepted.append((finding, fix))
        log.info(
            "finding_decision id=%s category=%s decision=%s source=deterministic-guards%s",
            finding_id,
            _finding_value(finding, "category", ""),
            "accepted" if not problems else "rejected",
            "" if not problems else " reason=" + "; ".join(problems),
        )

    if not accepted:
        return {"semgrep_before": 0, "semgrep_after": 0, "accepted": 0}

    combined = source
    for _, fix in accepted:
        occurrence = find_text_occurrence(
            combined, fix["original"], fix.get("start_line")
        )
        if occurrence is not None:
            offset, _ = occurrence
            combined = (
                combined[:offset] + fix["replacement"]
                + combined[offset + len(fix["original"]):]
            )
    try:
        ast.parse(combined)
    except SyntaxError as error:
        message = f"Combined fixes do not parse: {error.msg}"
        for finding, fix in accepted:
            _set_verified(finding, fix, False, [message])
            log.info(
                "finding_dropped id=%s category=%s reason=%s",
                ensure_finding_id(finding),
                _finding_value(finding, "category", ""),
                message,
            )
        return {"semgrep_before": 0, "semgrep_after": 0, "accepted": 0}
    combined_names = undefined_names(combined) - undefined_names(source)
    if combined_names:
        message = "Combined fixes introduce undefined names: " + ", ".join(sorted(combined_names))
        for finding, fix in accepted:
            _set_verified(finding, fix, False, [message])
            log.info(
                "finding_dropped id=%s category=%s reason=%s",
                ensure_finding_id(finding),
                _finding_value(finding, "category", ""),
                message,
            )
        return {"semgrep_before": 0, "semgrep_after": 0, "accepted": 0}

    before, after = await asyncio.gather(
        asyncio.to_thread(run_semgrep, source),
        asyncio.to_thread(run_semgrep, combined),
    )
    return {
        "semgrep_before": len(before),
        "semgrep_after": len(after),
        "accepted": len(accepted),
    }
