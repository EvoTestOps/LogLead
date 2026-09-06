"""Allow ``python -m loglead.mcp`` as an alternative to the ``loglead-mcp`` script."""

from .server import main

if __name__ == "__main__":
    main()
