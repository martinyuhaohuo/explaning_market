"""Optional Langfuse tracing.

Tracing is on only when the three LANGFUSE_* settings are present and the host
accepts a connection; otherwise `langfuse_handler` is a no-op callback, so a
missing or stopped Langfuse server (e.g. a local one on localhost:3000) never
fills the output with export retries.
"""

import os
import socket
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()
_LANGFUSE_KEYS = (
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


def host_reachable(url: str, timeout: float = 1.0) -> bool:
    """Whether a TCP connection to the URL's host and port opens within `timeout` seconds."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


langfuse = None
# Keep callers' callback lists valid while making tracing optional.
langfuse_handler = BaseCallbackHandler()

if all(os.environ.get(key) for key in _LANGFUSE_KEYS):
    if host_reachable(os.environ["LANGFUSE_HOST"]):
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        langfuse = get_client()
        langfuse_handler = CallbackHandler()
    else:
        print(f"[WARN] Langfuse host {os.environ['LANGFUSE_HOST']} is unreachable; tracing is off.")
