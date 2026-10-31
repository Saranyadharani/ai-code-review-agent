"""Webhook signature verification.

GitHub signs every webhook body with HMAC-SHA256 using your shared secret and
sends it in the X-Hub-Signature-256 header as "sha256=<hexdigest>". We recompute
it and compare in constant time so nobody else can trigger reviews.
"""
import hashlib
import hmac


def compute_signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    if not secret or not header:
        return False
    return hmac.compare_digest(compute_signature(secret, body), header)
