"""Publish verified review findings as a GitHub pull request review."""
import asyncio
import hashlib
import logging
import re
from collections.abc import Sequence

from app.config import settings
from app.context import changed_lines
from app.github_client import GitHubClient

log = logging.getLogger(__name__)

_publish_lock = asyncio.Lock()
_MARKER_PATTERN = re.compile(r"<!-- review-agent:([a-f0-9]{40}) -->")


def _value(finding, key: str, default=None):
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def _finding_marker(filename: str, finding) -> str:
    identity = f"{filename}{_value(finding, 'line', '')}{_value(finding, 'title', '')}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return f"<!-- review-agent:{digest} -->"


def _suggestion_details(finding, source: str) -> tuple[str | None, int | None, int | None]:
    fix = _value(finding, "fix")
    if not isinstance(fix, dict) or fix.get("verified") is not True:
        return None, None, None

    original = fix.get("original")
    replacement = fix.get("replacement")
    if not isinstance(original, str) or not original or not isinstance(replacement, str):
        return None, None, None

    offset = source.find(original)
    if offset < 0:
        return None, None, None

    start = source[:offset].count("\n") + 1
    n_old = original.count("\n") + 1
    patched = source.replace(original, replacement, 1)
    n_new = n_old + replacement.count("\n") - original.count("\n")
    suggestion = "\n".join(patched.splitlines()[start - 1:start - 1 + n_new])
    return suggestion, start, start + n_old - 1


def _finding_text(finding, include_suggestion: bool, source: str) -> str:
    severity = str(_value(finding, "severity", "medium")).upper()
    category = str(_value(finding, "category", "finding"))
    title = str(_value(finding, "title", "Review finding"))
    explanation = str(_value(finding, "explanation", "")).strip()
    text = f"**{severity} / {category}: {title}**"
    if explanation:
        text += f"\n\n{explanation}"

    fix = _value(finding, "fix")
    if isinstance(fix, dict):
        replacement = fix.get("replacement")
        suggestion, _, _ = _suggestion_details(finding, source)
        if suggestion and include_suggestion:
            text += f"\n\n```suggestion\n{suggestion}\n```"
        elif replacement:
            text += f"\n\nProposed fix (not applied as an inline suggestion):\n{replacement.rstrip()}"
    elif _value(finding, "suggestion"):
        text += f"\n\nSuggested fix: {_value(finding, 'suggestion')}"
    return text


async def post_review(
    owner: str,
    repo: str,
    number: int,
    head_sha: str,
    filename: str,
    findings: Sequence[object],
    patch: str | None = None,
    source: str = "",
    max_inline_comments: int | None = None,
    categories: set[str] | None = None,
) -> dict:
    """Compatibility wrapper for publishing findings from one file."""
    return await post_reviews(
        owner,
        repo,
        number,
        head_sha,
        [{"filename": filename, "findings": findings, "patch": patch, "source": source}],
        max_inline_comments=max_inline_comments,
        categories=categories,
    )


async def post_reviews(
    owner: str,
    repo: str,
    number: int,
    head_sha: str,
    files: Sequence[dict],
    max_inline_comments: int | None = None,
    categories: set[str] | None = None,
) -> dict:
    """Publish one PR review with a global inline-comment cap."""
    scope = f"{owner}/{repo}#{number}:"
    allowed_categories = categories
    if allowed_categories is None:
        allowed_categories = {
            item.strip().lower()
            for item in settings.review_categories.split(",")
            if item.strip()
        }
    else:
        allowed_categories = {category.lower() for category in allowed_categories}
    limit = settings.max_inline_comments if max_inline_comments is None else max_inline_comments
    limit = max(0, limit)

    async with _publish_lock:
        gh = GitHubClient()
        try:
            existing_comments = await gh.list_pull_request_comments(owner, repo, number)
        except Exception:
            await gh.aclose()
            raise
        existing_markers = {
            marker
            for comment in existing_comments
            for marker in _MARKER_PATTERN.findall(comment.get("body", ""))
        }
        pending = []
        seen = set(existing_markers)
        for file in files:
            filename = file["filename"]
            for finding in file["findings"]:
                if _value(finding, "category", "") not in allowed_categories:
                    continue
                marker = _finding_marker(filename, finding)
                marker_value = marker.removeprefix("<!-- review-agent:").removesuffix(" -->")
                if marker_value in seen:
                    continue
                pending.append((
                    filename, file.get("patch"), file.get("source", ""),
                    finding, marker,
                ))
                seen.add(marker_value)

        if not pending:
            await gh.aclose()
            return {"published": False, "comments": 0, "summary_findings": 0}

        inline = []
        summary = []
        for filename, patch, source, finding, marker in pending:
            diff_lines = changed_lines(patch)
            suggestion, start_line, end_line = _suggestion_details(finding, source)
            finding_line = _value(finding, "line")
            comment_start = start_line if suggestion is not None else finding_line
            comment_end = end_line if suggestion is not None else finding_line
            valid_range = (
                isinstance(comment_start, int)
                and isinstance(comment_end, int)
                and all(line in diff_lines for line in range(comment_start, comment_end + 1))
            )
            severity = str(_value(finding, "severity", "medium")).lower()
            category = _value(finding, "category", "")
            if valid_range and severity in {"high", "medium"} and category != "quality":
                inline.append((
                    {"filename": filename, "source": source, "finding": finding,
                     "start_line": comment_start, "end_line": comment_end,
                     "marker": marker},
                    severity,
                ))
            else:
                summary.append((filename, finding, marker, source))

        inline.sort(key=lambda item: 0 if item[1] == "high" else 1)
        selected = [item[0] for item in inline[:limit]]
        overflow = [item[0] for item in inline[limit:]]
        summary.extend(
            (item["filename"], item["finding"], item["marker"], item["source"])
            for item in overflow
        )

        comments = []
        for item in selected:
            finding = item["finding"]
            comment = {
                "path": item["filename"],
                "line": item["end_line"],
                "side": "RIGHT",
                "body": (
                    _finding_text(finding, include_suggestion=True, source=item["source"])
                    + "\n\n" + item["marker"]
                ),
            }
            if item["start_line"] != item["end_line"]:
                comment["start_line"] = item["start_line"]
                comment["start_side"] = "RIGHT"
            comments.append(comment)

        body = ""
        if summary:
            body = "Findings in the review summary:\n\n"
            body += "\n\n".join(
                f"- `{filename}:{_value(finding, 'line')}` — "
                f"{_finding_text(finding, include_suggestion=False, source=source)} "
                f"{marker}"
                for filename, finding, marker, source in summary
            )

        try:
            await gh.create_pull_request_review(
                owner, repo, number, head_sha, body, comments
            )
        finally:
            await gh.aclose()

    log.info(
        "Published review for %s/%s#%d (%d inline, %d in summary)",
        owner, repo, number, len(comments), len(summary),
    )
    return {
        "published": True,
        "comments": len(comments),
        "summary_findings": len(summary),
    }
