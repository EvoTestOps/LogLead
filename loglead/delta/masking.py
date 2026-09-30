"""Named regex masking patterns for :meth:`EventLogEnhancer.mask`.

``myllari``, ``myllari_extended``, ``drain_loglead`` and ``drain_orig`` are ported
verbatim from LogDelta's ``logdelta/regex_masking.py``. The per-dataset masks below
them were written against the datasets LogLead loads, and ``merged`` is their union
and the default.

Each pattern is a list of ``(replacement, regex)`` tuples. ``mask()`` applies
every pair twice by default so that adjacent tokens both get masked.

These are the built-in patterns; resolve them by name via :func:`get_pattern`
rather than reaching into :data:`PATTERNS` directly, so a typo'd or unknown
name fails with a clear error instead of a ``KeyError``. A caller that wants a
pattern beyond this fixed set -- their own, or one of these with additions --
registers it under its own name via
:class:`loglead.mcp.mask_registry.MaskPatternRegistry`, whose ``resolve()``
checks this module first and only then its own registered patterns, so a
custom name can never shadow a built-in one.
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


# Per-dataset masks and their merge
# ---------------------------------
# One mask per dataset family LogLead loads, and ``merged``, their union and the
# default. Every mask lists its pairs in the same relative order as ``merged``:
# context-anchored ids first, then whole structured tokens (URLs, timestamps, UUIDs,
# addresses), then numbers with a unit or shape, and bare numbers last. Keep it that
# way when editing -- a general pattern that runs early eats what a specific one
# needed, as myllari's <VERSION> does to IPv4 addresses. A pattern added to a
# dataset mask goes into ``merged`` too, at the same position.
#
# The timestamp patterns in each mask are the ones its raw files carry, since
# RawLoader leaves the whole line, timestamp included, in m_message.

bgl = [
    ("${start}<NODE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:R\d{2}-M\d(?:-[A-Za-z0-9]{1,3}(?::J\d{2})?)*|bglio\d+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6}(?P<end>[^0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<MAC>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

hdfs = [
    ("${start}<BLK>${end}", r"(?P<start>[^A-Za-z0-9]|^)blk_-?\d+(?P<end>[^0-9]|$)"),
    ("${start}${kind}_<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?P<kind>attempt|task|job|jvm|container|appattempt|application)_\d+(?:_[a-z]?_?\d+)*(?P<end>[^0-9_]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

hadoop = [
    ("${start}${kind}_<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?P<kind>attempt|task|job|jvm|container|appattempt|application)_\d+(?:_[a-z]?_?\d+)*(?P<end>[^0-9_]|$)"),
    ("${start}<HOST>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:MININT|minint|MSRA-SA|msra-sa)-[A-Za-z0-9]+(?:\.fareast\.corp\.microsoft\.com)?(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[A-Za-z_$;\]]@)[0-9a-f]{6,8}(?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<NUM>", r"(?P<start>\$Proxy|Generated(?:Serialization)?(?:Method|Constructor)Accessor)\d+"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<DURATION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)(?:\d+(?:\.\d+)?(?:ms|us|µs|ns|s)|\d{1,3}[hm]){1,4}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

thunderbird = [
    ("${start}<QID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[a-z][0-9AB][0-9A-Za-z]{6}\d{6}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<MAC>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<PCI>${end}", r"(?P<start>[^0-9A-Za-z:]|^)(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7](?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NODE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[a-d]n|ln|sn|tn)\d{1,4}|en\d{2,4})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

spirit = [
    ("${start}<QID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[a-z][0-9AB][0-9A-Za-z]{6}\d{6}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<MAC>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<PCI>${end}", r"(?P<start>[^0-9A-Za-z:]|^)(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7](?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NODE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[a-d]n|ln|sn|tn)\d{1,4}|en\d{2,4})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

liberty = [
    ("${start}<QID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[a-z][0-9AB][0-9A-Za-z]{6}\d{6}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<MAC>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NODE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[a-d]n|ln|sn|tn)\d{1,4}|en\d{2,4})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

openstack = [
    ("${start}<IFACE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:qbr|qvb|qvo|tap)[0-9a-f]{8}-[0-9a-f]{0,2}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<MAC>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

# light-oauth2: the LO2 and LO2v2 loaders and the MuFaNo runs.
lo2 = [
    ("${start}<ID>${end}", r"(?P<start>\]  |[Cc]ode ?= ?'?)[A-Za-z0-9_-]{22}(?P<end>[^A-Za-z0-9_-]|$)"),
    ("${start}<AUTH>", r"(?P<start>(?:Basic|Bearer) )[A-Za-z0-9+/._~-]*[0-9+/][A-Za-z0-9+/._~-]*=*"),
    ("${start}<HEX>${end}", r"(?P<start>[A-Za-z_$;\]]@)[0-9a-f]{6,8}(?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<JWT>${end}", r"(?P<start>[^A-Za-z0-9_-]|^)eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?(?P<end>[^A-Za-z0-9_-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<B64>${end}", r"(?P<start>[\"'=]|^)[A-Za-z0-9_-]{43}(?P<end>[\"'&\s,]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

nezha = [
    ("${start}<POD>${end}", r"(?P<start>-)[0-9a-f]{6,10}-[bcdfghjklmnpqrstvwxz2456789]{5}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<JWT>${end}", r"(?P<start>[^A-Za-z0-9_-]|^)eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?(?P<end>[^A-Za-z0-9_-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

# Also IoT-23, whose conn.log is a Zeek log.
zeek = [
    ("${start}<UID>", r"(?P<start>\b(?:[a-z_]*uids?|id)=)[CF][A-Za-z0-9]{14,17}(?:,[CF][A-Za-z0-9]{14,17})*"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<EMAIL>${end}", r"(?P<start>[^A-Za-z0-9._%+-]|^)[A-Za-z0-9._%+-]+@[A-Za-z0-9-]*[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?P<end>[^A-Za-z0-9-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})(?P<end>[^0-9A-Za-z:]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

access_log = [
    ("${start}<FILTER>", r"(?P<start>/filter/)[^\s?]+"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{2}/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{4}:\d{2}:\d{2}:\d{2}(?: [+-]\d{4})?(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("<ENC>", r"(?:%[0-9A-Fa-f]{2}){2,}"),
    ("${start}<RES>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d{2,}x\d{2,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

iis = [
    ("${start}<SESSION>", r"(?P<start>(?i:sessionid|x-owa-canary|canary)=|session=<)[^&;\s>]+"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<EMAIL>${end}", r"(?P<start>[^A-Za-z0-9._%+-]|^)[A-Za-z0-9._%+-]+@[A-Za-z0-9-]*[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?P<end>[^A-Za-z0-9-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})(?P<end>[^0-9A-Za-z:]|$)"),
    ("<ENC>", r"(?:%[0-9A-Fa-f]{2}){2,}"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

syslog = [
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<EMAIL>${end}", r"(?P<start>[^A-Za-z0-9._%+-]|^)[A-Za-z0-9._%+-]+@[A-Za-z0-9-]*[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?P<end>[^A-Za-z0-9-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<MAC>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9-]|^)\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}(?P<end>[^0-9-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})(?P<end>[^0-9A-Za-z:]|$)"),
    ("${start}<PCI>${end}", r"(?P<start>[^0-9A-Za-z:]|^)(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7](?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DURATION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)(?:\d+(?:\.\d+)?(?:ms|us|µs|ns|s)|\d{1,3}[hm]){1,4}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

logfmt = [
    ("${start}<POD>${end}", r"(?P<start>-)[0-9a-f]{6,10}-[bcdfghjklmnpqrstvwxz2456789]{5}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<POD>${end}", r"(?P<start>-)[bcdfghjklmnpqrstvwxz2456789]{5}(?P<end>_[0-9a-f]{8}-)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<DURATION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)(?:\d+(?:\.\d+)?(?:ms|us|µs|ns|s)|\d{1,3}[hm]){1,4}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

# Loghub's 2k samples mix sixteen systems.
loghub = [
    ("${start}<BLK>${end}", r"(?P<start>[^A-Za-z0-9]|^)blk_-?\d+(?P<end>[^0-9]|$)"),
    ("${start}${kind}_<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?P<kind>attempt|task|job|jvm|container|appattempt|application)_\d+(?:_[a-z]?_?\d+)*(?P<end>[^0-9_]|$)"),
    ("${start}<HOST>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:MININT|minint|MSRA-SA|msra-sa)-[A-Za-z0-9]+(?:\.fareast\.corp\.microsoft\.com)?(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NODE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:R\d{2}-M\d(?:-[A-Za-z0-9]{1,3}(?::J\d{2})?)*|bglio\d+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<QID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[a-z][0-9AB][0-9A-Za-z]{6}\d{6}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[A-Za-z_$;\]]@)[0-9a-f]{6,8}(?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<EMAIL>${end}", r"(?P<start>[^A-Za-z0-9._%+-]|^)[A-Za-z0-9._%+-]+@[A-Za-z0-9-]*[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?P<end>[^A-Za-z0-9-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<MAC>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9-]|^)\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}(?P<end>[^0-9-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})(?P<end>[^0-9A-Za-z:]|$)"),
    ("${start}<PCI>${end}", r"(?P<start>[^0-9A-Za-z:]|^)(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7](?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9A-Za-z:.+-]|^)[0-2]?\d:[0-5]\d(?P<end>[^0-9A-Za-z:.]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NODE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[a-d]n|ln|sn|tn)\d{1,4}|en\d{2,4})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DURATION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)(?:\d+(?:\.\d+)?(?:ms|us|µs|ns|s)|\d{1,3}[hm]){1,4}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

gha = [
    ("", r"\x1b\[[0-9;]*[A-Za-z]"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9A-Za-z:.+-]|^)[0-2]?\d:[0-5]\d(?P<end>[^0-9A-Za-z:.]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DURATION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)(?:\d+(?:\.\d+)?(?:ms|us|µs|ns|s)|\d{1,3}[hm]){1,4}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

pro_android = [
    ("${start}<HEX>${end}", r"(?P<start>[A-Za-z_$;\]]@)[0-9a-f]{6,8}(?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9.]|^)\d{1,2}\.\d{1,2}\.\d{4}[ T]\d{1,2}[:.]\d{2}[:.]\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<DURATION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)(?:\d+(?:\.\d+)?(?:ms|us|µs|ns|s)|\d{1,3}[hm]){1,4}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

comp_ws = [
    ("${start}<THREAD>${end}", r"(?P<start>\[)(?:[0-9a-f]{4}\.[0-9a-f]{4}|T\d+)(?P<end>\])"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<JWT>${end}", r"(?P<start>[^A-Za-z0-9_-]|^)eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?(?P<end>[^A-Za-z0-9_-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<SID>${end}", r"(?P<start>[^A-Za-z0-9-]|^)S-1-\d+(?:-\d+)+(?P<end>[^0-9-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9-]|^)\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}(?P<end>[^0-9-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})(?P<end>[^0-9A-Za-z:]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<B64>${end}", r"(?P<start>[\"'\s=:(,]|^)[A-Za-z0-9+/]*\d[A-Za-z0-9+/]*={1,2}(?P<end>[\"'\s,;)&]|$)"),
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

ait_ads = [
    ("${start}<SESSION>", r"(?P<start>(?i:sessionid|x-owa-canary|canary)=|session=<)[^&;\s>]+"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<EMAIL>${end}", r"(?P<start>[^A-Za-z0-9._%+-]|^)[A-Za-z0-9._%+-]+@[A-Za-z0-9-]*[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?P<end>[^A-Za-z0-9-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})(?P<end>[^0-9A-Za-z:]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

security_datasets = [
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<SID>${end}", r"(?P<start>[^A-Za-z0-9-]|^)S-1-\d+(?:-\d+)+(?P<end>[^0-9-]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),
]

# The default: every pattern of the masks above, in their shared order.
merged = [
    ("", r"\x1b\[[0-9;]*[A-Za-z]"),  # colour codes in CI output would read as numbers ([32m)
    ("${start}<ID>${end}", r"(?P<start>\]  |[Cc]ode ?= ?'?)[A-Za-z0-9_-]{22}(?P<end>[^A-Za-z0-9_-]|$)"),  # light-oauth2 correlation id (two spaces after the thread) or auth code
    ("${start}<AUTH>", r"(?P<start>(?:Basic|Bearer) )[A-Za-z0-9+/._~-]*[0-9+/][A-Za-z0-9+/._~-]*=*"),  # needs a digit or + /, so "Basic information" stays
    ("${start}<UID>", r"(?P<start>\b(?:[a-z_]*uids?|id)=)[CF][A-Za-z0-9]{14,17}(?:,[CF][A-Za-z0-9]{14,17})*"),  # Zeek connection and file uids
    ("${start}<SESSION>", r"(?P<start>(?i:sessionid|x-owa-canary|canary)=|session=<)[^&;\s>]+"),  # IIS/Exchange session and canary tokens, dovecot session=<...>
    ("${start}<FILTER>", r"(?P<start>/filter/)[^\s?]+"),
    ("${start}<BLK>${end}", r"(?P<start>[^A-Za-z0-9]|^)blk_-?\d+(?P<end>[^0-9]|$)"),
    ("${start}${kind}_<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?P<kind>attempt|task|job|jvm|container|appattempt|application)_\d+(?:_[a-z]?_?\d+)*(?P<end>[^0-9_]|$)"),  # keeps the id kind: attempt_<ID>, job_<ID>
    ("${start}<HOST>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:MININT|minint|MSRA-SA|msra-sa)-[A-Za-z0-9]+(?:\.fareast\.corp\.microsoft\.com)?(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NODE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:R\d{2}-M\d(?:-[A-Za-z0-9]{1,3}(?::J\d{2})?)*|bglio\d+)(?P<end>[^A-Za-z0-9]|$)"),  # BGL rack/midplane/node locations and I/O nodes
    ("${start}<QID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[a-z][0-9AB][0-9A-Za-z]{6}\d{6}(?P<end>[^A-Za-z0-9]|$)"),  # sendmail queue id: year, month (0-9AB), 6 time chars, pid
    ("${start}<IFACE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:qbr|qvb|qvo|tap)[0-9a-f]{8}-[0-9a-f]{0,2}(?P<end>[^A-Za-z0-9]|$)"),  # neutron port devices: prefix + first 11 chars of the port uuid
    ("${start}<POD>${end}", r"(?P<start>-)[0-9a-f]{6,10}-[bcdfghjklmnpqrstvwxz2456789]{5}(?P<end>[^A-Za-z0-9]|$)"),  # ReplicaSet hash + pod suffix; the suffix alphabet has no vowels
    ("${start}<POD>${end}", r"(?P<start>-)[bcdfghjklmnpqrstvwxz2456789]{5}(?P<end>_[0-9a-f]{8}-)"),  # DaemonSet/StatefulSet suffix, only inside /var/log/pods/<ns>_<pod>_<uid>
    ("${start}<THREAD>${end}", r"(?P<start>\[)(?:[0-9a-f]{4}\.[0-9a-f]{4}|T\d+)(?P<end>\])"),  # [pid.tid] and [T1234] thread tags
    ("${start}<HEX>${end}", r"(?P<start>[A-Za-z_$;\]]@)[0-9a-f]{6,8}(?P<end>[^0-9A-Za-z.]|$)"),  # Object.toString() identity hash, Shuffle@6ad3381f
    ("${start}<NUM>", r"(?P<start>\$Proxy|Generated(?:Serialization)?(?:Method|Constructor)Accessor)\d+"),  # $Proxy15, GeneratedMethodAccessor8
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9]|^)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>()\[\]{},;|]+(?P<end>[\s\"'<>()\[\]{},;|]|$)"),
    ("${start}<EMAIL>${end}", r"(?P<start>[^A-Za-z0-9._%+-]|^)[A-Za-z0-9._%+-]+@[A-Za-z0-9-]*[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?P<end>[^A-Za-z0-9-]|$)"),  # first domain label needs a letter: not health@2.0-service.pixel
    ("${start}<JWT>${end}", r"(?P<start>[^A-Za-z0-9_-]|^)eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?(?P<end>[^A-Za-z0-9_-]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6}(?P<end>[^0-9]|$)"),  # BGL's own 2005-06-03-15.42.50.675872
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?(?P<end>[^0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9]|^)\d{2}/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{4}:\d{2}:\d{2}:\d{2}(?: [+-]\d{4})?(?P<end>[^0-9]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^A-Za-z]|^)(?:(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:(?:\s+[A-Z]{3,4})?\s+\d{4})?|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}(?: \d{2}:\d{2}:\d{2})?)(?P<end>[^0-9]|$)"),  # syslog 'Jun  9 06:06:20', ctime 'Sat May 04 07:16:00 CST 2013'
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9.]|^)\d{1,2}\.\d{1,2}\.\d{4}[ T]\d{1,2}[:.]\d{2}[:.]\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9]|$)"),  # 10.07.2020 12:16:19.900
    ("${start}<DATE>${end}", r"(?P<start>[^0-9.]|^)(?:(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}|\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ -](?:19|20)\d{2})(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),  # year pinned to 19xx/20xx so that build numbers such as 15.1.1713 stay versions
    ("${start}<UUID>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{1,12}(?P<end>[^0-9A-Za-z-]|$)"),
    ("${start}<SID>${end}", r"(?P<start>[^A-Za-z0-9-]|^)S-1-\d+(?:-\d+)+(?P<end>[^0-9-]|$)"),
    ("${start}<MAC>${end}", r"(?P<start>[^A-Za-z0-9]|^)[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5,}(?P<end>[^A-Za-z0-9]|$)"),  # 6 groups is a MAC, 8 an InfiniBand GUID; before IPv6 and TIME
    ("${start}<IP>${end}", r"(?P<start>[^0-9.]|^)(?:::ffff:)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),  # before VERSION, which would otherwise take it
    ("${start}<IP>${end}", r"(?P<start>[^0-9-]|^)\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}(?P<end>[^0-9-]|$)"),  # 10-1-2-3 inside reverse-DNS hostnames
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})(?P<end>[^0-9A-Za-z:]|$)"),
    ("${start}<PCI>${end}", r"(?P<start>[^0-9A-Za-z:]|^)(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7](?P<end>[^0-9A-Za-z.]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?P<end>[^0-9:]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9A-Za-z:.+-]|^)[0-2]?\d:[0-5]\d(?P<end>[^0-9A-Za-z:.]|$)"),  # not after + or -, where it is a UTC offset
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)0[xX][0-9a-fA-F]+(?P<end>[^A-Za-z0-9]|$)"),
    ("<ENC>", r"(?:%[0-9A-Fa-f]{2}){2,}"),  # two or more escaped bytes (encoded non-ASCII); a lone %2C stays
    ("${start}<B64>${end}", r"(?P<start>[\"'=]|^)[A-Za-z0-9_-]{43}(?P<end>[\"'&\s,]|$)"),  # unpadded SHA-256 (PKCE challenge); quoted only, 43 chars is also a class name
    ("${start}<B64>${end}", r"(?P<start>[\"'\s=:(,]|^)[A-Za-z0-9+/]*\d[A-Za-z0-9+/]*={1,2}(?P<end>[\"'\s,;)&]|$)"),  # padded base64 with a digit, as a whole value only: KEY= also ends in =
    ("${start}<RES>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d{2,}x\d{2,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NODE>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:(?:[a-d]n|ln|sn|tn)\d{1,4}|en\d{2,4})(?P<end>[^A-Za-z0-9]|$)"),  # HPC node names; en needs 2 digits so macOS en0..en9 stay
    ("${start}<SIZE>${end}", r"(?P<start>[^A-Za-z0-9.%]|^)(?:\d+(?:\.\d+)?(?:\s?[kKMGTP]i?B(?:ytes|s)?|[KMGT]bytes|[kKMG])|\d{1,4}(?:\.\d+)?B)(?P<end>[^A-Za-z0-9]|$)"),  # bare B only on short numbers: 426578925B is a postfix queue id
    ("${start}<DURATION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)(?:\d+(?:\.\d+)?(?:ms|us|µs|ns|s)|\d{1,3}[hm]){1,4}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^A-Za-z0-9.]|^)v?\d+(?:\.\d+){2,}(?P<end>[^A-Za-z0-9.]|\.[^0-9]|\.$|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(?:[a-fA-F][0-9a-fA-F]{6,}|\d[a-fA-F][0-9a-fA-F]{5,}|\d{2}[a-fA-F][0-9a-fA-F]{4,}|\d{3}[a-fA-F][0-9a-fA-F]{3,}|\d{4}[a-fA-F][0-9a-fA-F]{2,}|\d{5}[a-fA-F][0-9a-fA-F]+|\d{6,}[a-fA-F][0-9a-fA-F]*)(?P<end>[^A-Za-z0-9]|$)"),  # 7+ chars with an a-f letter, so an all-digit run stays <NUM>
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?P<end>[^A-Za-z0-9]|$)"),  # unsigned: a sign would glue onto the token before it (3.2.0-16)
]

DATASET_MASKS = {
    "bgl": bgl, "hdfs": hdfs, "hadoop": hadoop, "thunderbird": thunderbird, "spirit": spirit,
    "liberty": liberty, "openstack": openstack, "lo2": lo2, "nezha": nezha, "zeek": zeek,
    "access_log": access_log, "iis": iis, "syslog": syslog, "logfmt": logfmt, "loghub": loghub,
    "gha": gha, "pro_android": pro_android, "comp_ws": comp_ws, "ait_ads": ait_ads,
    "security_datasets": security_datasets,
}

DEFAULT_PATTERN = "merged"

#: Allowlist of maskings that may be selected by name from untrusted input.
PATTERNS = {
    "myllari": myllari,
    "myllari_extended": myllari_extended,
    "drain_loglead": drain_loglead,
    "drain_orig": drain_orig,
    "merged": merged,
    **DATASET_MASKS,
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
