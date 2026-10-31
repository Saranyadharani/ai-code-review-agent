"""Step 6a: verify proposed fixes without executing PR code."""
import asyncio
import io

from pyflakes.api import check
from pyflakes.reporter import Reporter

from app.context import run_semgrep


def undefined_names(code: str) -> set[str]:
    """Pyflakes 'undefined name' messages, without line numbers."""
    out = io.StringIO()
    check(code, "f.py", Reporter(out, out))
    return {l.split(":", 3)[-1].strip() for l in out.getvalue().splitlines() if "undefined name" in l}


async def verify_fixes(source: str, findings: list) -> dict:
    """Mark each fix verified/rejected, then re-scan all accepted fixes together with Semgrep."""
    base = undefined_names(source)
    accepted = []
    for f in findings:
        if not f.fix:
            continue
        patched = source.replace(f.fix["original"], f.fix["replacement"], 1)
        new = undefined_names(patched) - base
        f.fix["verified"] = not new
        f.fix["problems"] = sorted(new)
        if not new:
            accepted.append(f)
    if not accepted:
        return {"semgrep_before": 0, "semgrep_after": 0, "accepted": 0}
    combined = source
    for f in accepted:
        if f.fix["original"] in combined:
            combined = combined.replace(f.fix["original"], f.fix["replacement"], 1)
    before = await asyncio.to_thread(run_semgrep, source)
    after = await asyncio.to_thread(run_semgrep, combined)
    return {"semgrep_before": len(before), "semgrep_after": len(after), "accepted": len(accepted)}