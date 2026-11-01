"""Configurable finding classification, ranking, deduplication, and noise rules."""
import logging
import hashlib
import re

from pydantic import BaseModel

CATEGORY_PRIORITY = {
    "security": 0,
    "bug": 1,
    "performance": 2,
    "quality": 3,
    "dependency": 4,
    "impact": 5,
}
SEVERITY_PRIORITY = {"critical": 0, "high": 1, "medium": 2, "low": 3}

SECURITY_RULE_IDS = {
    "aws-access-key",
    "database-uri-password",
    "eval-use",
    "hardcoded-secret",
    "hashlib-weak-hash",
    "os-system",
    "pickle-load",
    "pickle-loads",
    "random-credential",
    "sql-fstring",
    "subprocess-shell-true",
    "yaml-unsafe-load",
}
SECURITY_RULE_PATTERNS = (r"^S\d{3}$",)
NON_SECURITY_RULE_CATEGORIES = {"S110": "bug"}
OTHER_RULE_CATEGORIES = {
    "BARE-EXCEPT": "bug",
}
SECURITY_KEYWORDS = (
    r"\binjection\b",
    r"\bpath traversal\b",
    r"\bssrf\b",
    r"\bxxe\b",
    r"\bunsafe deserializ\w*\b",
    r"\bdeserializ\w*\b",
    r"\bpickle(?:\.load|\.loads)?\b",
    r"\beval\s*\(",
    r"\bhardcoded (?:secret|credential|password|token)\b",
    r"\bweak (?:hash|crypto(?:graphy)?)\b",
    r"\bunsafe yaml\b",
    r"\bshell execution\b",
    r"\bcommand execution\b",
    r"\bsql\b.{0,30}\binjection\b",
    r"\b(?:hardcoded|embedded|exposed|leaked|plaintext)\s+"
    r"(?:secret|credential|password|token)\b",
)
STYLE_ONLY_KEYWORDS = (
    r"\bstyle(?:-only)?\b",
    r"\bpep\s*8\b",
    r"\bnaming convention\b",
    r"\bformatting\b",
    r"\bline length\b",
    r"\bdocstring\b",
    r"\btype hint\b",
    r"\bwhitespace\b",
    r"\bimport order\b",
)
DEDUPLICATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "does",
    "for", "from", "in", "is", "it", "may", "might", "of", "on", "or", "should",
    "that", "the", "this", "to", "use", "with", "would", "covered", "finding",
    "hint", "issue", "llm", "potential", "uncovered",
}

_SECURITY_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in SECURITY_KEYWORDS)
_SECURITY_RULE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in SECURITY_RULE_PATTERNS
)
_STYLE_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in STYLE_ONLY_KEYWORDS)
_WORD_PATTERN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_SECURITY_RULE_IDS_NORMALIZED = {rule.upper() for rule in SECURITY_RULE_IDS}
_CONTRACT_CONCERNS = {
    "signature": re.compile(
        r"\b(?:public )?signature\b|\bparameter list\b|\bargument list\b", re.I
    ),
    "return_shape": re.compile(
        r"\b(?:return|result|output)\b.{0,30}\b(?:type|shape|format|semantics?)\b"
        r"|\b(?:type|shape|format|semantics?)\b.{0,30}\b(?:return|result|output)\b",
        re.I,
    ),
}
log = logging.getLogger("review-agent")


def category_for_rule(rule_id: str | None, fallback: str = "bug") -> str:
    normalized = str(rule_id or "").strip().upper()
    if normalized in NON_SECURITY_RULE_CATEGORIES:
        return NON_SECURITY_RULE_CATEGORIES[normalized]
    if normalized in _SECURITY_RULE_IDS_NORMALIZED or any(
        pattern.fullmatch(normalized) for pattern in _SECURITY_RULE_PATTERNS
    ):
        return "security"
    return OTHER_RULE_CATEGORIES.get(normalized, fallback)


def finding_value(finding, key: str, default=None):
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def ensure_finding_id(finding) -> str:
    value = finding_value(finding, "id")
    if isinstance(value, str) and value.strip():
        return value
    identity_data = "\0".join(
        str(finding_value(finding, key, "") or "")
        for key in ("category", "line", "rule_id", "title", "explanation")
    )
    identity = hashlib.sha256(identity_data.encode("utf-8")).hexdigest()[:24]
    if isinstance(finding, dict):
        finding["id"] = identity
    elif isinstance(finding, BaseModel):
        object.__setattr__(finding, "id", identity)
    else:
        setattr(finding, "id", identity)
    return identity


def finding_concerns(finding, concern: str) -> bool:
    pattern = _CONTRACT_CONCERNS.get(concern)
    if pattern is None:
        return False
    text = " ".join(
        str(finding_value(finding, key, "") or "")
        for key in ("title", "explanation")
    )
    return pattern.search(text) is not None


def finding_rule_id(finding) -> str:
    value = finding_value(finding, "rule_id") or finding_value(finding, "rule")
    return str(value or "").strip().upper()


def find_text_occurrence(source: str, text: str, target_line: int | None = None) -> tuple[int, int] | None:
    """Find the matching source occurrence nearest a 1-based finding line."""
    if not text:
        return None
    matches = []
    offset = source.find(text)
    while offset >= 0:
        line = source.count("\n", 0, offset) + 1
        matches.append((offset, line))
        offset = source.find(text, offset + 1)
    if not matches:
        return None
    if target_line is None:
        return matches[0]
    return min(matches, key=lambda match: abs(match[1] - target_line))


def is_security_finding(finding) -> bool:
    rule_id = finding_rule_id(finding)
    if rule_id in NON_SECURITY_RULE_CATEGORIES:
        return False
    if rule_id in _SECURITY_RULE_IDS_NORMALIZED:
        return True
    if any(pattern.fullmatch(rule_id) for pattern in _SECURITY_RULE_PATTERNS):
        return True
    text = " ".join(
        str(finding_value(finding, key, "") or "")
        for key in ("title", "explanation", "suggestion", "message")
    )
    return any(pattern.search(text) for pattern in _SECURITY_PATTERNS)


def correct_category(finding) -> str:
    rule_id = finding_rule_id(finding)
    configured_category = category_for_rule(rule_id, "")
    if configured_category:
        if isinstance(finding, dict):
            finding["category"] = configured_category
        else:
            finding.category = configured_category
        return configured_category
    if is_security_finding(finding):
        category = "security"
        if isinstance(finding, dict):
            finding["category"] = category
        else:
            finding.category = category
        return category
    return str(finding_value(finding, "category", "bug"))


def is_style_only(finding) -> bool:
    if str(finding_value(finding, "category", "")).lower() == "security":
        return False
    text = " ".join(
        str(finding_value(finding, key, "") or "")
        for key in ("title", "explanation", "suggestion")
    )
    return any(pattern.search(text) for pattern in _STYLE_PATTERNS)


def _tokens(finding) -> set[str]:
    text = " ".join(
        str(finding_value(finding, key, "") or "")
        for key in ("title", "explanation")
    ).casefold()
    return {
        token for token in _WORD_PATTERN.findall(text)
        if len(token) > 1 and token not in DEDUPLICATION_STOPWORDS
    }


def findings_are_duplicates(left, right) -> bool:
    left_line = finding_value(left, "line")
    right_line = finding_value(right, "line")
    if isinstance(left_line, int) and isinstance(right_line, int):
        if abs(left_line - right_line) <= 1:
            return True
    left_rule, right_rule = finding_rule_id(left), finding_rule_id(right)
    if left_rule and left_rule == right_rule:
        return True
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    overlap = left_tokens & right_tokens
    return len(overlap) >= 2 and len(overlap) / min(len(left_tokens), len(right_tokens)) >= 0.6


def _category_rank(finding) -> int:
    return CATEGORY_PRIORITY.get(str(finding_value(finding, "category", "")), 99)


def _severity_rank(finding) -> int:
    return SEVERITY_PRIORITY.get(
        str(finding_value(finding, "severity", "medium")).lower(), 99
    )


def _fix_rank(finding) -> tuple[int, int, int]:
    fix = finding_value(finding, "fix")
    if not isinstance(fix, dict):
        return (0, 0, 0)
    return (
        int(fix.get("verified") is True),
        int(fix.get("critic_safe") is True),
        int(bool(fix.get("original") and fix.get("replacement"))),
    )


def _rank(finding) -> tuple[int, int, tuple[int, int, int]]:
    return (_category_rank(finding), _severity_rank(finding), _fix_rank(finding))


def _set_fix(finding, fix) -> None:
    if isinstance(finding, dict):
        finding["fix"] = fix
    else:
        finding.fix = fix


def _merge_best_fix(winner, duplicate) -> None:
    if _fix_rank(duplicate) > _fix_rank(winner):
        fix = dict(finding_value(duplicate, "fix"))
        fix["finding_id"] = ensure_finding_id(winner)
        _set_fix(winner, fix)
        if isinstance(winner, BaseModel):
            winner.critic_verdict = finding_value(duplicate, "critic_verdict")
        elif isinstance(winner, dict):
            winner["critic_verdict"] = finding_value(duplicate, "critic_verdict")


def deduplicate_findings(findings):
    """Keep the highest-ranked finding for each issue while retaining its best fix."""
    for finding in findings:
        correct_category(finding)
        ensure_finding_id(finding)
    parents = list(range(len(findings)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(findings)):
        for right in range(left + 1, len(findings)):
            if findings_are_duplicates(findings[left], findings[right]):
                left_root, right_root = root(left), root(right)
                if left_root != right_root:
                    parents[right_root] = left_root

    groups = {}
    for index, finding in enumerate(findings):
        groups.setdefault(root(index), []).append(finding)

    winners = []
    for group in groups.values():
        ranked = sorted(group, key=_rank)
        winner = ranked[0]
        for duplicate in ranked[1:]:
            _merge_best_fix(winner, duplicate)
            log.info(
                "finding_dropped id=%s category=%s reason=deduplicated_into:%s",
                ensure_finding_id(duplicate),
                finding_value(duplicate, "category", ""),
                ensure_finding_id(winner),
            )
        winners.append(winner)
    return sorted(winners, key=_rank)
