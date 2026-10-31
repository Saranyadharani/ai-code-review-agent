import pytest

from app.repo_index import create_repository_index


class FakeGitHub:
    def __init__(self, sources):
        self.sources = sources

    async def get_repository_tree(self, owner, repo, sha):
        return {
            "tree": [
                {"path": path, "type": "blob", "size": len(source)}
                for path, source in self.sources.items()
            ]
        }

    async def get_file_content(self, owner, repo, path, ref):
        return self.sources[path]


@pytest.mark.asyncio
async def test_callers_are_found_across_repository_files():
    client = FakeGitHub({
        "service.py": "def target(value):\n    return value\n",
        "handlers.py": "from service import target\n\ndef handle():\n    return target(1)\n",
    })
    index = await create_repository_index("acme", "repo", "caller-test-sha", client)

    callers = await index.callers_of("target")

    assert [(item["file"], item["line"], item["function"]) for item in callers] == [
        ("handlers.py", 4, "handle")
    ]


@pytest.mark.asyncio
async def test_related_context_finds_callers_of_a_removed_function():
    client = FakeGitHub({
        "handlers.py": "def handle():\n    return removed_function()\n",
    })
    index = await create_repository_index("acme", "repo", "removed-test-sha", client)

    related = await index.related_context(
        "deleted.py", ["removed_function"], ["deleted.py"], max_tokens=100
    )

    assert "Caller of removed_function (handlers.py:1)" in related["text"]
    assert "return removed_function()" in related["text"]


@pytest.mark.asyncio
async def test_related_context_respects_character_budget():
    client = FakeGitHub({
        "service.py": (
            "def target(value):\n"
            "    return value\n"
            "def long_caller():\n"
            + "".join(f"    target({number})\n" for number in range(80))
        ),
        "tests/test_service.py": "def test_target():\n    assert target(1) == 1\n",
    })
    index = await create_repository_index("acme", "repo", "budget-test-sha", client)

    related = await index.related_context(
        "service.py", ["target"], ["service.py"], max_tokens=20
    )

    assert len(related["text"]) <= 20 * 4
    assert not related["tests_changed"]
