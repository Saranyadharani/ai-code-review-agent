"""Lazy, commit-scoped repository symbol index for impact analysis."""
import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from app.config import settings
from app.github_client import GitHubClient

log = logging.getLogger(__name__)
_PARSER = Parser(Language(tspython.language()))
_TREE_CACHE: dict[tuple[str, str], "_IndexCache"] = {}
_TREE_LOCK = asyncio.Lock()
_MAX_FILE_BYTES = 100_000


def _is_test_path(path: str) -> bool:
    parts = PurePosixPath(path.lower()).parts
    name = parts[-1] if parts else ""
    return (
        any(part in {"test", "tests"} for part in parts)
        or name.startswith("test_")
        or name.endswith(("_test.py", "_spec.py"))
    )


@dataclass
class _IndexCache:
    paths: list[str]
    contents: dict[str, str] = field(default_factory=dict)
    parsed: dict[str, dict] = field(default_factory=dict)


class RepositoryIndex:
    """Tree-sitter symbol index backed by lazily fetched source files."""

    def __init__(
        self, owner: str, repo: str, sha: str, client: GitHubClient, cache: _IndexCache
    ):
        self.owner = owner
        self.repo = repo
        self.sha = sha
        self.client = client
        self.cache = cache
        self.key = f"{owner}/{repo}"
        self._definitions: dict[str, list[dict]] = {}
        self._callers: dict[str, list[dict]] = {}
        self._indexed = False

    @classmethod
    async def create(
        cls, owner: str, repo: str, sha: str, client: GitHubClient
    ) -> "RepositoryIndex":
        key = (f"{owner}/{repo}", sha)
        async with _TREE_LOCK:
            cache = _TREE_CACHE.get(key)
            if cache is None:
                response = await client.get_repository_tree(owner, repo, sha)
                tree = response.get("tree", [])
                if response.get("truncated"):
                    log.warning("GitHub repository tree was truncated for %s@%s", key[0], sha)
                limit = max(0, settings.repo_context_max_files)
                paths = [
                    item["path"]
                    for item in tree
                    if item.get("type") == "blob"
                    and isinstance(item.get("path"), str)
                    and item["path"].endswith(".py")
                    and item.get("size", 0) <= _MAX_FILE_BYTES
                ][:limit]
                cache = _IndexCache(paths=paths)
                _TREE_CACHE[key] = cache
        return cls(owner, repo, sha, client, cache)

    async def _content(self, path: str) -> str:
        if path not in self.cache.contents:
            self.cache.contents[path] = await self.client.get_file_content(
                self.owner, self.repo, path, self.sha
            )
        return self.cache.contents[path]

    @staticmethod
    def _node_text(source: bytes, node: Node) -> str:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _parse(self, path: str, text: str) -> dict:
        source = text.encode("utf-8")
        root = _PARSER.parse(source).root_node
        functions, classes, calls, import_nodes = [], [], [], []

        def walk(
            node: Node, function: dict | None = None, class_name: str | None = None
        ) -> None:
            for child in node.children:
                if child.type == "class_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        continue
                    name = self._node_text(source, name_node)
                    classes.append({
                        "name": name,
                        "start": child.start_point[0] + 1,
                        "end": child.end_point[0] + 1,
                    })
                    walk(child, function, name)
                elif child.type == "function_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        continue
                    name = self._node_text(source, name_node)
                    item = {
                        "name": name,
                        "qualified_name": f"{class_name}.{name}" if class_name else name,
                        "class": class_name,
                        "file": path,
                        "start": child.start_point[0] + 1,
                        "end": child.end_point[0] + 1,
                    }
                    functions.append(item)
                    walk(child, item, class_name)
                elif child.type in ("import_statement", "import_from_statement"):
                    import_nodes.append(self._node_text(source, child))
                    walk(child, function, class_name)
                elif child.type == "call":
                    target = child.child_by_field_name("function")
                    if target is not None:
                        calls.append({
                            "name": self._node_text(source, target).rsplit(".", 1)[-1],
                            "line": child.start_point[0] + 1,
                            "function": function["name"] if function else "<module>",
                            "function_start": function["start"] if function else child.start_point[0] + 1,
                            "function_end": function["end"] if function else child.end_point[0] + 1,
                            "file": path,
                        })
                    walk(child, function, class_name)
                else:
                    walk(child, function, class_name)

        walk(root)
        return {
            "functions": functions,
            "classes": classes,
            "calls": calls,
            "imports": import_nodes,
            "resolved_imports": self._resolve_imports(path, import_nodes),
        }

    def _resolve_imports(self, path: str, imports: list[str]) -> list[str]:
        known_paths = set(self.cache.paths)
        resolved = set()

        def candidates(module: str):
            module_path = module.replace(".", "/").strip("/")
            if not module_path:
                return []
            return [
                candidate
                for prefix in ("", "src/")
                for candidate in (
                    f"{prefix}{module_path}.py",
                    f"{prefix}{module_path}/__init__.py",
                )
            ]

        for statement in imports:
            match = re.match(r"\s*from\s+([.\w]+)\s+import\s+(.+)", statement, re.S)
            if match:
                module, imported = match.groups()
                relative_depth = len(module) - len(module.lstrip("."))
                module_name = module.lstrip(".")
                if relative_depth:
                    parent = list(PurePosixPath(path).parent.parts)
                    parent = parent[:max(0, len(parent) - relative_depth + 1)]
                    module_name = ".".join([*parent, module_name]).strip(".")
                options = candidates(module_name)
                for option in options:
                    if option in known_paths:
                        resolved.add(option)
                for imported_name in imported.replace("(", "").replace(")", "").split(","):
                    symbol = imported_name.strip().split()[0] if imported_name.strip() else ""
                    if symbol and symbol != "*":
                        resolved.update(
                            option for option in candidates(f"{module_name}.{symbol}")
                            if option in known_paths
                        )
            else:
                match = re.match(r"\s*import\s+(.+)", statement, re.S)
                if match:
                    for item in match.group(1).split(","):
                        module = item.strip().split()[0]
                        resolved.update(option for option in candidates(module) if option in known_paths)
        return sorted(resolved)

    async def _ensure_indexed(self) -> None:
        if self._indexed:
            return
        for path in self.cache.paths:
            text = await self._content(path)
            parsed = self.cache.parsed.get(path)
            if parsed is None:
                parsed = self._parse(path, text)
                self.cache.parsed[path] = parsed
            for definition in parsed["functions"]:
                self._definitions.setdefault(definition["name"], []).append(definition)
                if definition["qualified_name"] != definition["name"]:
                    self._definitions.setdefault(definition["qualified_name"], []).append(definition)
            for call in parsed["calls"]:
                self._callers.setdefault(call["name"], []).append(call)
        self._indexed = True

    async def callers_of(self, function_name: str) -> list[dict]:
        await self._ensure_indexed()
        return list(self._callers.get(function_name.rsplit(".", 1)[-1], []))

    async def callees_of(self, function_name: str) -> list[dict]:
        await self._ensure_indexed()
        definitions = self._definitions.get(function_name, [])
        if not definitions:
            definitions = self._definitions.get(function_name.rsplit(".", 1)[-1], [])
        if not definitions:
            return []
        definition = definitions[0]
        parsed = self.cache.parsed[definition["file"]]
        names = {
            call["name"] for call in parsed["calls"]
            if definition["start"] <= call["line"] <= definition["end"]
        }
        return [
            item
            for name in sorted(names)
            for item in self._definitions.get(name, [])
            if item is not definition
        ]

    async def tests_for(self, function_name: str) -> list[dict]:
        callers = await self.callers_of(function_name)
        return [call for call in callers if "test" in call["file"].lower()]

    async def imports_of(self, filename: str) -> list[str]:
        if filename not in self.cache.parsed:
            text = await self._content(filename)
            self.cache.parsed[filename] = self._parse(filename, text)
        return list(self.cache.parsed[filename]["resolved_imports"])

    def _snippet(self, path: str, start: int, end: int, label: str, budget: int) -> str:
        lines = self.cache.contents[path].splitlines()
        snippet = "\n".join(lines[max(0, start - 1):min(len(lines), end)])
        text = f"{label} ({path}:{start}):\n{snippet}"
        return text[:budget]

    async def related_context(
        self,
        filename: str,
        changed_functions: list[str],
        changed_files: list[str],
        max_tokens: int | None = None,
    ) -> dict:
        """Return related snippets capped using a conservative four-characters/token estimate."""
        token_limit = settings.max_context_tokens if max_tokens is None else max_tokens
        budget = max(0, token_limit * 4)
        tests_changed = any(_is_test_path(path) for path in changed_files)
        if not changed_functions or budget == 0:
            return {"text": "", "tests_changed": tests_changed}

        await self._ensure_indexed()
        candidates: list[str] = []
        related_callers: list[dict] = []
        parsed = self.cache.parsed.get(filename)
        if parsed is None and filename in self.cache.paths:
            source = await self._content(filename)
            parsed = self._parse(filename, source)
            self.cache.parsed[filename] = parsed
        if parsed is None:
            parsed = {"functions": [], "classes": [], "resolved_imports": []}
        related_imports = (
            await self.imports_of(filename) if filename in self.cache.paths else []
        )
        if related_imports:
            candidates.append(f"Repository imports: {', '.join(related_imports)}")
        for changed in changed_functions:
            if changed == "<module>":
                continue
            base_name = changed.rsplit(".", 1)[-1]
            definitions = [
                item for item in parsed["functions"]
                if item["name"] == base_name or item["qualified_name"] == changed
            ]
            callers = await self.callers_of(base_name)
            for caller in callers:
                if caller["file"] != filename:
                    item = {
                        "symbol": base_name,
                        "file": caller["file"],
                        "line": caller["line"],
                        "function": caller["function"],
                    }
                    if item not in related_callers:
                        related_callers.append(item)
            tests = await self.tests_for(base_name)
            for call in callers[:20]:
                if call["file"] == filename:
                    continue
                candidates.append(self._snippet(
                    call["file"], call["function_start"], call["function_end"],
                    f"Caller of {base_name}", budget,
                ))
            for test in tests[:20]:
                candidates.append(self._snippet(
                    test["file"], test["function_start"], test["function_end"],
                    f"Related test for {base_name}", budget,
                ))
            for definition in definitions:
                if definition["class"]:
                    cls = next(
                        (item for item in parsed["classes"] if item["name"] == definition["class"]),
                        None,
                    )
                    if cls:
                        candidates.append(self._snippet(
                            filename, cls["start"], cls["end"],
                            f"Enclosing class {cls['name']}", budget,
                        ))
                callees = await self.callees_of(base_name)
                for callee in callees[:20]:
                    candidates.append(self._snippet(
                        callee["file"], callee["start"], callee["end"],
                        f"Called function {callee['name']}", budget,
                    ))
        output = []
        used = 0
        for candidate in candidates:
            separator = "\n\n" if output else ""
            remaining = budget - used - len(separator)
            if remaining <= 0:
                break
            clipped = candidate[:remaining]
            output.append(clipped)
            used += len(separator) + len(clipped)
        return {
            "text": "\n\n".join(output),
            "callers": related_callers,
            "tests_changed": tests_changed,
            "changed_test_files": [path for path in changed_files if _is_test_path(path)],
        }


async def create_repository_index(
    owner: str, repo: str, sha: str, client: GitHubClient
) -> RepositoryIndex:
    return await RepositoryIndex.create(owner, repo, sha, client)
