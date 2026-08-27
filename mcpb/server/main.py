"""Entry point for the LogLead MCPB bundle.

The manifest launches this file with `uv run`, which resolves and installs
LogLead (with its `mcp` extra) from GitHub's `main` branch per pyproject.toml
in this directory, then hands off to LogLead's own MCP server main().
"""

from __future__ import annotations

from loglead.mcp.server import main

if __name__ == "__main__":
    main()
