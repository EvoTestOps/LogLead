"""MCP server exposing LogLead's run-comparison analyses over a live session.

Install with ``pip install loglead[mcp]``, then run ``loglead-mcp``.

Importing this package requires the optional ``mcp`` dependency, so the server
symbols are loaded lazily -- :mod:`loglead.mcp.session` can be used on its own
without it, which is what the demos and tests do.
"""

from .session import Session, SessionStore, default_cache_dir

__all__ = ["Session", "SessionStore", "default_cache_dir", "mcp", "main"]


def __getattr__(name):
    if name in ("mcp", "main"):
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
