"""Semantic Python and static-finding comparison between PR base and head."""
import re
from collections import Counter

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

_PARSER = Parser(Language(tspython.language()))


def _text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _definitions(source: str) -> tuple[dict, dict, list[str]]:
    encoded = source.encode()
    tree = _PARSER.parse(encoded)
    functions = {}
    classes = {}
    imports = []

    def walk(node, scope=()):
        for child in node.children:
            if child.type == "function_definition":
                name = _text(encoded, child.child_by_field_name("name"))
                qualified = ".".join((*scope, name))
                parameters = child.child_by_field_name("parameters")
                returns = child.child_by_field_name("return_type")
                signature = (
                    _text(encoded, parameters) if parameters else "()",
                    _text(encoded, returns) if returns else "",
                )
                functions[qualified] = {
                    "start": child.start_point[0] + 1,
                    "signature": signature,
                    "source": _text(encoded, child),
                    "public": not name.startswith("_"),
                }
                walk(child, (*scope, name))
            elif child.type == "class_definition":
                name = _text(encoded, child.child_by_field_name("name"))
                qualified = ".".join((*scope, name))
                bases = child.child_by_field_name("superclasses")
                classes[qualified] = {
                    "start": child.start_point[0] + 1,
                    "signature": _text(encoded, bases) if bases else "",
                    "source": _text(encoded, child),
                }
                walk(child, (*scope, name))
            elif child.type in ("import_statement", "import_from_statement"):
                imports.append(_text(encoded, child))
            else:
                walk(child, scope)

    walk(tree.root_node)
    return functions, classes, imports


def semantic_diff(base_source: str | None, head_source: str) -> dict:
    """Describe changed definitions and imports and flag likely breaking API edits."""
    base_functions, base_classes, base_imports = _definitions(base_source or "")
    head_functions, head_classes, head_imports = _definitions(head_source)

    added_functions = sorted(head_functions.keys() - base_functions.keys())
    removed_functions = sorted(base_functions.keys() - head_functions.keys())
    signature_changes = [
        {
            "name": name,
            "base": base_functions[name]["signature"],
            "head": head_functions[name]["signature"],
        }
        for name in sorted(base_functions.keys() & head_functions.keys())
        if base_functions[name]["signature"] != head_functions[name]["signature"]
    ]
    modified_functions = sorted(
        name for name in base_functions.keys() & head_functions.keys()
        if base_functions[name]["source"] != head_functions[name]["source"]
    )
    added_classes = sorted(head_classes.keys() - base_classes.keys())
    removed_classes = sorted(base_classes.keys() - head_classes.keys())
    modified_classes = sorted(
        name for name in base_classes.keys() & head_classes.keys()
        if (
            base_classes[name]["signature"] != head_classes[name]["signature"]
            or base_classes[name]["source"] != head_classes[name]["source"]
        )
    )

    removed_imports = sorted(set(base_imports) - set(head_imports))
    added_imports = sorted(set(head_imports) - set(base_imports))
    changed_imports = []
    remaining_added = set(added_imports)
    for removed in removed_imports:
        old_module = re.match(r"\s*from\s+([\w.]+)\s+import", removed)
        if not old_module:
            continue
        replacement = next(
            (item for item in remaining_added
             if (match := re.match(r"\s*from\s+([\w.]+)\s+import", item))
             and match.group(1) == old_module.group(1)),
            None,
        )
        if replacement:
            changed_imports.append({"base": removed, "head": replacement})
            remaining_added.remove(replacement)

    removed_public_functions = [
        name for name in removed_functions if base_functions[name]["public"]
    ]
    breaking_candidates = [
        *[f"Removed public function: {name}" for name in removed_public_functions],
        *[f"Changed function signature: {item['name']}" for item in signature_changes],
        *[f"Removed class: {name}" for name in removed_classes],
    ]
    return {
        "added_functions": added_functions,
        "removed_functions": removed_functions,
        "modified_functions": modified_functions,
        "signature_changes": signature_changes,
        "added_classes": added_classes,
        "removed_classes": removed_classes,
        "modified_classes": modified_classes,
        "added_imports": added_imports,
        "removed_imports": removed_imports,
        "changed_imports": changed_imports,
        "removed_public_functions": removed_public_functions,
        "breaking_change_candidates": breaking_candidates,
    }


def _normalized_line(source: str, line: int) -> str:
    lines = source.splitlines()
    if not 1 <= line <= len(lines):
        return ""
    return re.sub(r"\s+", " ", lines[line - 1]).strip().casefold()


def classify_findings(
    base_findings: list[dict],
    head_findings: list[dict],
    base_source: str,
    head_source: str,
) -> list[dict]:
    """Classify analyzer findings by rule and normalized source line, not line number."""
    base_counts = Counter(
        (finding["rule"], _normalized_line(base_source, finding["line"]))
        for finding in base_findings
    )
    pre_existing_counts = Counter()
    classified = []
    for finding in head_findings:
        key = (finding["rule"], _normalized_line(head_source, finding["line"]))
        status = "pre-existing" if pre_existing_counts[key] < base_counts[key] else "new"
        if status == "pre-existing":
            pre_existing_counts[key] += 1
        classified.append({**finding, "status": status})

    remaining_pre_existing = pre_existing_counts.copy()
    for finding in base_findings:
        key = (finding["rule"], _normalized_line(base_source, finding["line"]))
        if remaining_pre_existing[key]:
            remaining_pre_existing[key] -= 1
        else:
            classified.append({**finding, "status": "fixed"})
    return classified
