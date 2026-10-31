"""Pinned Python dependency vulnerability checks using the OSV API."""
import logging
import re
import tomllib
from dataclasses import dataclass
from typing import Any

import httpx

from app.graph import Finding

log = logging.getLogger(__name__)
OSV_API = "https://api.osv.dev"
_REQUIREMENT = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?\s*(.*)$"
)
_PIN = re.compile(r"(?:^|,)\s*==\s*([A-Za-z0-9][A-Za-z0-9.+!_-]*)")


@dataclass(frozen=True)
class Dependency:
    name: str
    version: str | None
    line: int


def _dependency(name: str, specifier: Any, line: int) -> Dependency | None:
    if name.casefold() == "python":
        return None
    if isinstance(specifier, dict):
        specifier = specifier.get("version", "")
    if not isinstance(specifier, str):
        return Dependency(name, None, line)
    pin = _PIN.search(specifier)
    return Dependency(name, pin.group(1) if pin else None, line)


def _requirements_dependencies(content: str) -> list[Dependency]:
    dependencies = []
    for line_number, raw in enumerate(content.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http:", "https:")):
            continue
        match = _REQUIREMENT.match(line)
        if match:
            dependency = _dependency(match.group(1), match.group(2), line_number)
            if dependency is not None:
                dependencies.append(dependency)
    return dependencies


def _toml_dependencies(filename: str, content: str) -> list[Dependency]:
    try:
        document = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        log.warning("Could not parse dependency file %s as TOML", filename)
        return []

    declarations: list[tuple[str, Any]] = []
    project = document.get("project", {})
    declarations.extend(
        ("", item) for item in project.get("dependencies", [])
    )
    for optional in project.get("optional-dependencies", {}).values():
        declarations.extend(("", item) for item in optional)

    poetry = document.get("tool", {}).get("poetry", {})
    for name, spec in poetry.get("dependencies", {}).items():
        declarations.append((name, spec))
    for group in poetry.get("group", {}).values():
        for name, spec in group.get("dependencies", {}).items():
            declarations.append((name, spec))

    lines = content.splitlines()
    parsed = []
    for name, spec in declarations:
        if not name:
            if not isinstance(spec, str):
                continue
            match = _REQUIREMENT.match(spec)
            if not match:
                continue
            name, requirement_spec = match.groups()
        else:
            requirement_spec = spec.get("version", "") if isinstance(spec, dict) else spec

        dependency = _dependency(name, requirement_spec, 1)
        if dependency is None:
            continue
        line_number = _toml_line_for_dependency(lines, name, requirement_spec)
        parsed.append(Dependency(dependency.name, dependency.version, line_number))
    return parsed


def _toml_line_for_dependency(lines: list[str], name: str, specifier: Any) -> int:
    quoted_name = re.escape(name)
    assignment = re.compile(rf"^\s*['\"]?{quoted_name}['\"]?\s*=", re.IGNORECASE)
    requirement = re.compile(
        rf"['\"]{quoted_name}(?:\[[^\]]+\])?\s*{re.escape(str(specifier))}['\"]",
        re.IGNORECASE,
    )
    for line_number, line in enumerate(lines, 1):
        if assignment.search(line) or requirement.search(line):
            return line_number
    return 1


def parse_dependencies(filename: str, content: str) -> list[Dependency]:
    if filename.endswith(".toml"):
        return _toml_dependencies(filename, content)
    return _requirements_dependencies(content)


def _severity(vulnerability: dict) -> str:
    scores = []
    for item in vulnerability.get("severity", []):
        score = item.get("score")
        if isinstance(score, str):
            match = re.fullmatch(r"\d+(?:\.\d+)?", score.strip())
            if match:
                scores.append(float(match.group()))
    if not scores:
        return "medium"
    score = max(scores)
    return "high" if score >= 7 else "medium" if score >= 4 else "low"


def _fixed_version(vulnerability: dict, package_name: str) -> str | None:
    for affected in vulnerability.get("affected", []):
        package = affected.get("package", {})
        if package.get("ecosystem", "").casefold() != "pypi":
            continue
        if package.get("name", "").casefold() != package_name.casefold():
            continue
        for version_range in affected.get("ranges", []):
            for event in version_range.get("events", []):
                fixed = event.get("fixed")
                if fixed:
                    return fixed
    return None


async def scan_dependencies(
    filename: str,
    content: str,
    client: httpx.AsyncClient | None = None,
) -> list[Finding]:
    """Report unpinned dependencies and OSV vulnerabilities for exact pins."""
    dependencies = parse_dependencies(filename, content)
    findings = [
        Finding(
            line=dependency.line,
            category="dependency",
            title=f"Cannot check unpinned dependency: {dependency.name}",
            severity="low",
            explanation="Pin this dependency with an exact == version to check it against OSV.",
        )
        for dependency in dependencies
        if dependency.version is None
    ]
    pinned = [dependency for dependency in dependencies if dependency.version is not None]
    if not pinned:
        return findings

    own_client = client is None
    http = client or httpx.AsyncClient(base_url=OSV_API, timeout=20)
    try:
        try:
            response = await http.post(
                "/v1/querybatch",
                json={
                    "queries": [
                        {
                            "package": {"name": dependency.name, "ecosystem": "PyPI"},
                            "version": dependency.version,
                        }
                        for dependency in pinned
                    ]
                },
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            log.warning("OSV dependency batch query failed for %s: %s", filename, error)
            return findings

        for dependency, result in zip(pinned, results):
            for reference in result.get("vulns", []):
                vulnerability_id = reference.get("id")
                if not vulnerability_id:
                    continue
                try:
                    detail_response = await http.get(f"/v1/vulns/{vulnerability_id}")
                    detail_response.raise_for_status()
                    vulnerability = detail_response.json()
                except (httpx.HTTPError, ValueError, AttributeError) as error:
                    log.warning("OSV detail query failed for %s: %s", vulnerability_id, error)
                    continue

                fixed = _fixed_version(vulnerability, dependency.name)
                summary = vulnerability.get("summary") or vulnerability.get("details") or ""
                suggestion = (
                    f"Update {dependency.name} from =={dependency.version} to =={fixed}."
                    if fixed else ""
                )
                explanation = f"{vulnerability_id}: {summary}".strip()
                if fixed:
                    explanation += f" A fixed version is available: {fixed}."
                findings.append(Finding(
                    line=dependency.line,
                    category="dependency",
                    title=f"Vulnerable dependency: {dependency.name}=={dependency.version}",
                    severity=_severity(vulnerability),
                    explanation=explanation,
                    suggestion=suggestion,
                    source="osv",
                ))
    finally:
        if own_client:
            await http.aclose()
    return findings
