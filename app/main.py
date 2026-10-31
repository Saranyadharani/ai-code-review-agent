import logging
from pathlib import PurePosixPath

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from app.config import settings
from app.context import build_file_context
from app.deps import scan_dependencies
from app.github_client import GitHubClient
from app.graph import run_graph
from app.publisher import post_reviews
from app.repo_index import create_repository_index
from app.security import verify_signature

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("review-agent")

app = FastAPI(title="AI Code Review Agent")

# PR actions worth reviewing. "synchronize" = new commits pushed to the PR.
REVIEWABLE_ACTIONS = {"opened", "synchronize", "reopened"}


async def collect_pr_data(owner: str, repo: str, number: int) -> dict:
    """Step 1 milestone: fetch the PR and every changed file with its diff."""
    gh = GitHubClient()
    try:
        pr = await gh.get_pr(owner, repo, number)
        files = await gh.get_changed_files(owner, repo, number)
    finally:
        await gh.aclose()
    return {
        "repo": f"{owner}/{repo}",
        "number": number,
        "title": pr["title"],
        "author": pr["user"]["login"],
        "head_sha": pr["head"]["sha"],
        "base_sha": pr["base"]["sha"],
        "base_branch": pr["base"]["ref"],
        "files": [f.__dict__ for f in files],
    }


async def run_review(owner: str, repo: str, number: int) -> None:
    """Run analysis and, when enabled, publish the results to GitHub."""
    try:
        data = await collect_pr_data(owner, repo, number)
    except (httpx.HTTPError, KeyError, ValueError):
        log.exception("Could not collect PR data for %s/%s#%d", owner, repo, number)
        return
    log.info(
        "PR %s#%d '%s': %d changed files",
        data["repo"], number, data["title"], len(data["files"]),
    )
    gh = GitHubClient()
    review_items = []
    repo_index = None
    if len(data["files"]) > 50:
        log.info("Skipping repository indexing for %s: %d changed files exceeds limit 50",
                 data["repo"], len(data["files"]))
    elif settings.groq_api_key:
        try:
            repo_index = await create_repository_index(
                owner, repo, data["head_sha"], gh
            )
        except (httpx.HTTPError, AttributeError, KeyError, TypeError, ValueError):
            log.exception("Could not initialize repository index for %s", data["repo"])
    try:
        for f in data["files"]:
            log.info("  %s %s (+%d/-%d)", f["status"], f["filename"], f["additions"], f["deletions"])
            is_python = f["filename"].endswith(".py")
            is_dependency_file = (
                PurePosixPath(f["filename"]).name.startswith("requirements")
                and f["filename"].endswith(".txt")
            ) or PurePosixPath(f["filename"]).name == "pyproject.toml"
            if f["status"] == "removed" and (not is_python or is_dependency_file):
                continue
            if not is_python and not is_dependency_file:
                continue
            try:
                src = (
                    ""
                    if f["status"] == "removed"
                    else await gh.get_file_content(owner, repo, f["filename"], data["head_sha"])
                )
                if is_dependency_file:
                    findings = await scan_dependencies(f["filename"], src)
                    for finding in findings:
                        log.info(
                            "    [%s/%s] line %d: %s (%s)",
                            finding.category, finding.severity, finding.line,
                            finding.title, finding.source,
                        )
                    if settings.publish_reviews and findings:
                        review_items.append({
                            "filename": f["filename"], "findings": findings,
                            "patch": f["patch"], "source": src,
                        })
                    continue

                base_source = None
                if f["status"] != "added" and data.get("base_sha"):
                    base_path = (
                        f.get("previous_filename")
                        if f["status"] == "renamed" and f.get("previous_filename")
                        else f["filename"]
                    )
                    try:
                        base_source = await gh.get_file_content(
                            owner, repo, base_path, data["base_sha"]
                        )
                    except httpx.HTTPError:
                        log.exception(
                            "Could not fetch base source for %s; analyzing head without baseline",
                            base_path,
                        )
                ctx = await build_file_context(
                    f["filename"], src, f["patch"], base_source=base_source
                )
                if repo_index is not None:
                    try:
                        impact_symbols = list(ctx["changed_functions"])
                        semantic = ctx.get("semantic_diff", {})
                        impact_symbols.extend(semantic.get("removed_functions", []))
                        impact_symbols.extend(
                            item["name"]
                            for item in semantic.get("signature_changes", [])
                        )
                        ctx["related_code"] = await repo_index.related_context(
                            f["filename"],
                            list(dict.fromkeys(impact_symbols)),
                            [item["filename"] for item in data["files"]],
                        )
                    except (httpx.HTTPError, KeyError, ValueError):
                        log.exception("Could not build related context for %s", f["filename"])
                        ctx["related_code"] = {"text": "", "tests_changed": None}
                log.info("  context: %s", ctx["changed_functions"])
                for h in ctx["semgrep"]:
                    log.info("    semgrep %s line %d: %s", h["rule"], h["line"], h["message"])
                if settings.groq_api_key:
                    try:
                        res = await run_graph(f["filename"], src, ctx)
                    except Exception:
                        log.exception("Review graph failed for %s", f["filename"])
                        continue
                    for x in res["findings"]:
                        log.info("    [%s/%s] line %d: %s (%s)", x.category, x.severity, x.line, x.title, x.source)
                        if x.fix:
                            ok = "OK" if x.fix["verified"] else "REJECTED " + ", ".join(x.fix["problems"])
                            log.info("      fix [%s]: %r -> %r", ok, x.fix["original"][:60], x.fix["replacement"][:60])
                    v = res.get("verify") or {}
                    if v.get("accepted"):
                        log.info("  verified %d fixes: static issues %d -> %d",
                                 v["accepted"], v["semgrep_before"], v["semgrep_after"])
                    if settings.publish_reviews and res["findings"]:
                        review_items.append({
                            "filename": f["filename"], "findings": res["findings"],
                            "patch": f["patch"], "source": src,
                        })
            except (httpx.HTTPError, KeyError, ValueError):
                log.exception("Could not review file %s", f["filename"])
                continue
        if settings.publish_reviews and review_items:
            try:
                await post_reviews(owner, repo, number, data["head_sha"], review_items)
            except (httpx.HTTPError, KeyError, ValueError):
                log.exception("Could not publish review for %s/%s#%d", owner, repo, number)
    finally:
        await gh.aclose()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/repos/{owner}/{repo}/pulls")
async def list_prs(owner: str, repo: str) -> list[dict]:
    """Manual check that your token works: lists open PRs."""
    gh = GitHubClient()
    try:
        prs = await gh.list_open_prs(owner, repo)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, "GitHub API error") from e
    finally:
        await gh.aclose()
    return [{"number": p["number"], "title": p["title"], "author": p["user"]["login"]} for p in prs]


@app.post("/review/{owner}/{repo}/{number}")
async def review_manually(owner: str, repo: str, number: int) -> dict:
    """Run the collection step on demand, no webhook needed. Returns diffs."""
    try:
        return await collect_pr_data(owner, repo, number)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, "GitHub API error") from e


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    background: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
) -> dict:
    body = await request.body()  # raw bytes: the signature covers the exact payload
    if not verify_signature(settings.github_webhook_secret, body, x_hub_signature_256):
        raise HTTPException(401, "Invalid signature")

    if x_github_event == "ping":
        return {"msg": "pong"}
    if x_github_event != "pull_request":
        return {"msg": f"ignored event {x_github_event}"}

    payload = await request.json()
    action = payload.get("action")
    if action not in REVIEWABLE_ACTIONS:
        return {"msg": f"ignored action {action}"}

    owner, repo = payload["repository"]["full_name"].split("/", 1)
    number = payload["pull_request"]["number"]

    # Respond fast (GitHub times out at 10s); do the real work in the background.
    background.add_task(run_review, owner, repo, number)
    return {"msg": "review queued", "pr": number}