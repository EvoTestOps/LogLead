"""Opt-in console logging for scripts, demos and notebooks (see docs/logging.md §5).

Programs that configure logging themselves should not call this: LogLead's records would then be
printed twice, once by this handler and once by the program's own root handler.
"""

import logging

_handler = None


def enable_console_logging(level="INFO"):
    global _handler

    logger = logging.getLogger("loglead")
    if _handler is None:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(_handler)
    logger.setLevel(level)
