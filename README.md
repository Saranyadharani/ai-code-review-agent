# AI Code Review Agent

An automated reviewer for changed Python code and dependency files in GitHub pull requests. It combines static analysis with focused AI review tasks, checks proposed fixes, and can optionally publish findings to GitHub.

## Architecture Diagram

<img width="6225" height="2913" alt="image" src="https://github.com/user-attachments/assets/bbe98589-ae8e-4d11-b64c-34f0e5d841c2" />

## How a review works

1. **Start:** GitHub Actions runs `python -m app.cli`, or GitHub sends a signed webhook to the FastAPI app.
2. **Collect PR data:** `run_review()` fetches pull-request details, changed files, diffs, and file contents.
3. **Build Python context:** Tree-sitter identifies imports, classes, functions, and changed code. Semgrep and Ruff scan the Python source and provide hints. When the base version is available, analyzer results are compared with the PR version.
4. **Review Python changes:** LangGraph runs focused tasks for bugs, security, and performance. Quality review is opt-in. Impact review uses related callers when available.
5. **Check dependencies:** Changed `requirements*.txt` and `pyproject.toml` files are checked against OSV for pinned PyPI versions.
6. **Validate and verify:** Findings are categorized and deduplicated. A fix may be proposed and critiqued; deterministic checks inspect the patched source.
7. **Publish:** If publishing is enabled, findings are posted as inline comments or in a review summary. Verified fixes can be included as GitHub suggestion blocks for the repository owner to review and apply.

The review tasks are separate prompts in one LangGraph workflow, not separate programs or different AI models. Semgrep and Ruff are static-analysis tools, not agents.

## Review coverage

### Python code

- **Bug review:** Logic errors and missed edge cases, such as off-by-one errors, mutable defaults, swallowed exceptions, and likely type errors.
- **Security review:** Possible risks such as injection, unsafe deserialization, `eval`, hardcoded secrets, and weak cryptography.
- **Performance review:** Meaningful inefficiencies, such as repeated expensive work in loops or inefficient data structures.
- **Quality review:** Style and maintainability issues such as naming, complexity, missing docstrings, or missing type hints. Enable `quality` in `REVIEW_CATEGORIES` to use this review.
- **Impact review:** Potential effects on code that calls changed functions. This uses a lightweight, name-based repository index when available.

### Static analysis

- **Semgrep** checks patterns defined in [`rules/python.yml`](rules/python.yml), such as `eval`, unsafe pickle/YAML loading, SQL f-strings, and shell-command risks.
- **Ruff** checks selected built-in Python rules for security, bugs, performance, and quality. Ruff does not use `rules/python.yml`.
- Both tools provide hints; a match is not by itself proof of a defect. The review tasks assess the code and context.

### Dependency review

When a dependency file changes, the app parses dependencies and checks exact-pinned PyPI versions against the OSV vulnerability database. If a known vulnerability has a fixed version, the finding can recommend that version. Unpinned dependencies are reported as not checkable.

## Publishing and configuration

Reviews are disabled by default. To publish comments, set:

```env
PUBLISH_REVIEWS=true
```

The GitHub token needs permission to read repository contents and pull requests, and to write pull-request reviews when publishing is enabled.

Relevant settings:

| Setting | Purpose |
|---|---|
| `GITHUB_TOKEN` | Accesses GitHub repositories and pull requests |
| `GITHUB_WEBHOOK_SECRET` | Verifies webhook requests when using the API webhook |
| `GROQ_API_KEY` | Enables AI review tasks |
| `PUBLISH_REVIEWS` | Enables or disables posting reviews |
| `REVIEW_CATEGORIES` | Selects findings to publish; defaults to `security,bug,performance,dependency,impact`. Add `quality` to opt in |
| `MAX_INLINE_COMMENTS` | Limits inline comments; defaults to `15`. Overflow findings go in the review summary |

See [`app/config.py`](app/config.py) for application defaults.

## GitHub Actions

The workflow runs when a pull request is opened, reopened, or updated with new commits. It skips draft pull requests and pull requests from forks. Configure the `GROQ_API_KEY` repository secret; the workflow uses GitHub's built-in `GITHUB_TOKEN` to access the pull request and publish reviews. Set **Settings → Actions → General → Workflow permissions** to **Read and write**.


## Limitations and security notes

- AI and static analysis can miss issues or report false positives. Review findings before applying them.
- Proposed fixes are checked in memory; pull-request code is not executed, and the verifier does not run the repository's test suite.
- A verified fix may be offered as a GitHub suggestion, but the agent does not apply or merge it automatically.
- Deterministic fix checks are conservative: a correct fix may be shown as advice only, without a one-click suggestion.
- Impact analysis is lightweight and name-based, so it may miss dynamic calls or confuse functions with the same name.
- Dependency checks require exact pinned versions and depend on OSV's available data.
- Only relevant changed code and related context are sent for review; the tool does not fully understand the entire repository.
- The configured LLM provider's data policies apply to code sent for analysis. Do not send private code unless that use is approved.
- A PostgreSQL service is present in Docker Compose, but the current review flow does not persist review results to it.
- The optional n8n notification/approval workflow has not been implemented.
