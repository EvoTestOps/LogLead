"""Named regex masking patterns for :meth:`EventLogEnhancer.normalize`.

Ported verbatim from LogDelta's ``logdelta/regex_masking.py``.

Each pattern is a list of ``(replacement, regex)`` tuples. ``normalize()`` applies
every pair twice by default so that adjacent tokens both get masked.

Only patterns registered in :data:`PATTERNS` should ever be handed to
``normalize()`` from untrusted input: ``EventLogEnhancer.normalize`` builds a
Python expression from the supplied list and ``eval()``s it
(``loglead/enhancers/eventlog.py``), so an arbitrary caller-supplied list is an
arbitrary-code-execution vector. Use :func:`get_pattern`.
"""

myllari = [
    ("${start}<QUOTED_ALPHANUMERIC>${end}", r"(?P<start>[^A-Za-z0-9-_]|^)'[a-zA-Z0-9-_]{16,}'(?P<end>[^A-Za-z0-9-_]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9/]|^)\d{2}/\d{2}/\d{4}(?P<end>[^0-9/]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9-]|^)\d{4}-\d{2}-\d{2}(?P<end>[^0-9-]|$)"),
    ("${start}<DATE_XX>${end}", r"(?P<start>[^A-Za-z0-9_]|^)DATE_\d{2}(?P<end>[^A-Za-z0-9_]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^A-Za-z]|^)\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b(?P<end>[^A-Za-z]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:.]|^)\d{2}:\d{2}(?::\d{2}(?:\.\d{3})?)?(?P<end>[^0-9:.]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{2}:\d{2}(?P<end>[^0-9:]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9.]|^)\d{1,2}\.\d{1,2}\.\d{4} \d{1,2}\.\d{1,2}\.\d{2}(?P<end>[^0-9.]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^0-9.]|^)\d{1,5}(?:\.\d{1,3}){1,4}(?P<end>[^0-9.]|$)"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9:/]|^)(https?://[^\s]+)(?P<end>[^A-Za-z0-9:/]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^A-Za-z]|^)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{2}| \d) \d{4}\b(?P<end>[^A-Za-z]|$)"),
    ("${start}<TXID>${end}", r"(?P<start>[^0-9A-Fa-f-]|^)\d{4}-[0-9A-Fa-f]{16}(?P<end>[^0-9A-Fa-f-]|$)"),
    ("${start}<FILEPATH>${end}", r"(?P<start>[^A-Za-z0-9:\\]|^)[A-Za-z]:\\(?:[^\\\n]+\\)*[^\\\n]+(?P<end>[^A-Za-z0-9:\\]|$)"),
    ("${start}<APIKEY>${end}", r"(?P<start>[^A-Za-z0-9\"]|^)\"x-apikey\":\s\"[^\"]+\"(?P<end>[^A-Za-z0-9\"]|$)"),
    ("${start}<TIMEMS>${end}", r"(?P<start>[^0-9ms]|^)\b\d+\s+ms\b(?P<end>[^0-9ms]|$)"),
    ("${start}<SECONDS>${end}", r"(?P<start>[^0-9s-]|^)-?\d{1,4}s(?P<end>[^0-9s-]|$)"),
    ("${start}<HEXBLOCKS>${end}", r"(?P<start>[^0-9A-Fa-f-]|^)(?:[0-9A-Fa-f]{4,}-)+[0-9A-Fa-f]{4,}(?P<end>[^0-9A-Fa-f-]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^0-9A-Fa-f]|^)0x[0-9A-Fa-f]+(?P<end>[^0-9A-Fa-f]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^0-9A-Fa-f]|^)([0-9A-Fa-f]{6,})(?P<end>[^0-9A-Fa-f]|$)"),
    ("${start}<LARGEINT>${end}", r"(?P<start>[^0-9]|^)\d{4,}(?P<end>[^0-9]|$)")
]

#As above but improved see #New
myllari_extended = [
    ("${start}<QUOTED_ALPHANUMERIC>${end}", r"(?P<start>[^A-Za-z0-9-_]|^)'[a-zA-Z0-9-_]{16,}'(?P<end>[^A-Za-z0-9-_]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9/]|^)\d{2}/\d{2}/\d{4}(?P<end>[^0-9/]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9-]|^)\d{4}-\d{2}-\d{2}(?P<end>[^0-9-]|$)"),
    ("${start}<DATE_XX>${end}", r"(?P<start>[^A-Za-z0-9_]|^)DATE_\d{2}(?P<end>[^A-Za-z0-9_]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^A-Za-z]|^)\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b(?P<end>[^A-Za-z]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{2}:\d{2}:\d{2},\d{3}(?P<end>[^0-9:]|$)"),  #New pattern for HH:MM:SS,MMM format
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:.]|^)\d{2}:\d{2}(?::\d{2}(?:\.\d{3})?)?(?P<end>[^0-9:.]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{2}:\d{2}(?P<end>[^0-9:]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9.]|^)\d{1,2}\.\d{1,2}\.\d{4} \d{1,2}\.\d{1,2}\.\d{2}(?P<end>[^0-9.]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^0-9.]|^)\d{1,5}(?:\.\d{1,3}){1,4}(?P<end>[^0-9.]|$)"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9:/]|^)(https?://[^\s]+)(?P<end>[^A-Za-z0-9:/]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^A-Za-z]|^)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{2}| \d) \d{4}\b(?P<end>[^A-Za-z]|$)"),
    ("${start}<TXID>${end}", r"(?P<start>[^0-9A-Fa-f-]|^)\d{4}-[0-9A-Fa-f]{16}(?P<end>[^0-9A-Fa-f-]|$)"),
    ("${start}<FILEPATH>${end}", r"(?P<start>[^A-Za-z0-9:\\]|^)[A-Za-z]:\\(?:[^\\\n]+\\)*[^\\\n]+(?P<end>[^A-Za-z0-9:\\]|$)"),
    ("${start}<APIKEY>${end}", r"(?P<start>[^A-Za-z0-9\"]|^)\"x-apikey\":\s\"[^\"]+\"(?P<end>[^A-Za-z0-9\"]|$)"),
    ("${start}<TIMEMS>${end}", r"(?P<start>[^0-9ms]|^)\b\d+\s+ms\b(?P<end>[^0-9ms]|$)"),
    ("${start}<SECONDS>${end}", r"(?P<start>[^0-9s-]|^)-?\d{1,4}s(?P<end>[^0-9s-]|$)"),
    ("${start}<HEXBLOCKS>${end}", r"(?P<start>[^0-9A-Fa-f-]|^)(?:[0-9A-Fa-f]{4,}-)+[0-9A-Fa-f]{4,}(?P<end>[^0-9A-Fa-f-]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^0-9A-Fa-f]|^)0x[0-9A-Fa-f]+(?P<end>[^0-9A-Fa-f]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^0-9A-Fa-f]|^)([0-9A-Fa-f]{6,})(?P<end>[^0-9A-Fa-f]|$)"),
    ("${start}<LARGEINT>${end}", r"(?P<start>[^0-9]|^)\d{4,}(?P<end>[^0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?P<end>[^A-Za-z0-9]|$)"), #New
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)([\-\+]?[1-9]\d+)(?P<end>[^A-Za-z0-9]|$)") #New
]

# Drain.ini default regexes as in LogLead -> 3 HEX pattern were too greedy and are commented out
drain_loglead = [
    ("${start}<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9a-f]{6,} ?){3,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9A-F]{4} ?){4,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
#   ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)([a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
#   ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]+|[a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
#   ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]{2,}(?:[a-f0-9A-F]{2})*|[a-f0-9A-F]{2}(?:[a-f0-9A-F]{2})*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)([\-\+]?\d+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${cmd}<CMD>", r"(?P<cmd>executed cmd )(\".+?\")")
]
# NOTE: name kept for LogDelta compatibility.

# Drain.ini default regexes original
drain_orig = [
    ("${start}<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9a-f]{6,} ?){3,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9A-F]{4} ?){4,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)([a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]+|[a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]{2,}(?:[a-f0-9A-F]{2})*|[a-f0-9A-F]{2}(?:[a-f0-9A-F]{2})*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)([\-\+]?\d+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${cmd}<CMD>", r"(?P<cmd>executed cmd )(\".+?\")")
]

#: Allowlist of maskings that may be selected by name from untrusted input.
PATTERNS = {
    "myllari": myllari,
    "myllari_extended": myllari_extended,
    "drain_loglead": drain_loglead,
    "drain_orig": drain_orig,
}


def get_pattern(name):
    """Resolve a masking pattern by name, rejecting anything not allowlisted.

    :param name: key of :data:`PATTERNS`.
    :raises ValueError: if ``name`` is not a known pattern.
    """
    try:
        return PATTERNS[name]
    except KeyError:
        raise ValueError(
            f"Unknown masking pattern {name!r}. Valid options: {sorted(PATTERNS)}"
        ) from None
