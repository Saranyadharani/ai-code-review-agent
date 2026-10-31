"""Thin async client for the GitHub REST API.

Supports PR listing and retrieval, changed-file and source retrieval, and
submitting pull request reviews.
"""
import base64
from dataclasses import dataclass

import httpx

from app.config import settings


@dataclass
class ChangedFile:
    filename: str
    status: str  # added | modified | removed | renamed
    additions: int
    deletions: int
    patch: str | None  # unified diff hunk; None for binary/very large files
    previous_filename: str | None = None


class GitHubClient:
    def __init__(self, token: str | None = None, base_url: str | None = None):
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.github_api_url,
            headers={
                "Authorization": f"Bearer {token or settings.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_open_prs(self, owner: str, repo: str) -> list[dict]:
        r = await self._client.get(
            f"/repos/{owner}/{repo}/pulls", params={"state": "open", "per_page": 50}
        )
        r.raise_for_status()
        return r.json()

    async def get_pr(self, owner: str, repo: str, number: int) -> dict:
        r = await self._client.get(f"/repos/{owner}/{repo}/pulls/{number}")
        r.raise_for_status()
        return r.json()

    async def get_changed_files(
        self, owner: str, repo: str, number: int
    ) -> list[ChangedFile]:
        """All files changed in a PR. GitHub paginates at 100 per page."""
        files: list[ChangedFile] = []
        page = 1
        while True:
            r = await self._client.get(
                f"/repos/{owner}/{repo}/pulls/{number}/files",
                params={"per_page": 100, "page": page},
            )
            r.raise_for_status()
            batch = r.json()
            files.extend(
                ChangedFile(
                    filename=f["filename"],
                    status=f["status"],
                    additions=f["additions"],
                    deletions=f["deletions"],
                    patch=f.get("patch"),
                    previous_filename=f.get("previous_filename"),
                )
                for f in batch
            )
            if len(batch) < 100:
                return files
            page += 1

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str
    ) -> str:
        """Full text of a file at a commit/branch (used later for repo context)."""
        r = await self._client.get(
            f"/repos/{owner}/{repo}/contents/{path}", params={"ref": ref}
        )
        r.raise_for_status()
        data = r.json()
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    async def get_repository_tree(self, owner: str, repo: str, sha: str) -> dict:
        """Fetch the recursive file tree for a commit."""
        response = await self._client.get(
            f"/repos/{owner}/{repo}/git/trees/{sha}",
            params={"recursive": "1"},
        )
        response.raise_for_status()
        return response.json()

    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        number: int,
        head_sha: str,
        body: str,
        comments: list[dict],
    ) -> dict:
        """Create a review with inline comments against the reviewed head commit."""
        r = await self._client.post(
            f"/repos/{owner}/{repo}/pulls/{number}/reviews",
            json={
                "commit_id": head_sha,
                "body": body,
                "event": "COMMENT",
                "comments": comments,
            },
        )
        r.raise_for_status()
        return r.json()

    async def list_pull_request_comments(
        self, owner: str, repo: str, number: int
    ) -> list[dict]:
        """List every inline comment on a pull request."""
        comments = []
        page = 1
        while True:
            response = await self._client.get(
                f"/repos/{owner}/{repo}/pulls/{number}/comments",
                params={"per_page": 100, "page": page},
            )
            response.raise_for_status()
            batch = response.json()
            comments.extend(batch)
            if len(batch) < 100:
                return comments
            page += 1
