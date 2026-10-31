# AI Code Review Agent

GitHub → FastAPI → analyze changed Python and dependency files → optionally publish a PR review.

## Run it

```bash
cp .env.example .env        # fill in GITHUB_TOKEN and GITHUB_WEBHOOK_SECRET
docker compose up --build
```

API at http://localhost:8010 (docs at /docs).

## Milestone checks

1. **Token works**: `curl localhost:8010/repos/OWNER/REPO/pulls` lists open PRs.
2. **Read changed files**: `curl -X POST localhost:8010/review/OWNER/REPO/PR_NUMBER`
   returns the PR metadata and every changed file with its diff patch.
3. **Webhook**: expose port 8010 (e.g. `ngrok http 8010`), then in the repo go to
   Settings → Webhooks → Add webhook:
   - Payload URL: `https://<ngrok-id>.ngrok.app/webhook/github`
   - Content type: `application/json`
   - Secret: same as `GITHUB_WEBHOOK_SECRET`
   - Events: Pull requests
   Open a PR and watch the API logs for the review results.

## Token scopes
Fine-grained PAT with Pull requests: read and Contents: read. Set
`PUBLISH_REVIEWS=true` and grant Pull requests: write to publish reviews. Reviews
are disabled by default. Findings on changed lines become inline comments;
findings outside the diff go in the review summary. Verified fixes are formatted
as GitHub suggestions. Identical findings are deduplicated in process memory,
so the cache resets when the API restarts.

## Review coverage and noise controls
Python files are scanned with local Semgrep rules and Ruff. Ruff findings are
categorized as security, bug, performance, or quality; the LangGraph review
includes bug, security, performance, and quality analysis. Changed
`requirements*.txt` and `pyproject.toml` files are checked against OSV for
exactly pinned PyPI dependencies; unpinned dependencies are reported as
unchecked.

`REVIEW_CATEGORIES` controls which categories are published and defaults to
`security,bug,performance,dependency`. Add `quality` to opt in. `MAX_INLINE_COMMENTS`
defaults to `15`; higher-severity findings are prioritized, and overflow is
included in the review summary. Low-severity and quality findings are summary
only.

## Tests
`pytest` (external GitHub, OSV, and Groq calls are mocked).

The optional n8n notification/approval workflow remains optional and was not implemented.
