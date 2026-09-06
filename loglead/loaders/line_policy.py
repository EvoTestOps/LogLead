import polars as pl

__all__ = ['event_start', 'starts_event', 'to_events', 'POLICIES', 'normalize_policy']

"""
line_policy

One log event is not always one line of text. A Java stack trace, a Python traceback, a printed
SQL statement or an ASCII banner all put a second, third and fiftieth line under the line that
started the event, and none of those lines carry a timestamp of their own:

    2015-10-19 15:49:52,128 ERROR No route to host from MININT-FNANLIN to msra-sa-2
            at org.apache.hadoop.ipc.Client.call(Client.java:1476)
            ... 38 more

Every loader that reads text therefore has to answer two questions, and they are separate
questions with separate answers:

1. **Which lines start an event?** LogLead's loaders disagree on this by design - RawLoader knows
   a line started an event because it got a timestamp out of it, HadoopLoader because the line
   matched a date regex, SyslogLoader because its RFC pattern matched. `event_start()` names those
   tests so they can be picked, and combined, rather than rewritten.
2. **What happens to the lines that did not?** There are six defensible answers and every one of
   them is in use somewhere: throw the line away, keep it as its own row, keep it as its own row
   but give it the time of the event above, fold it into that event's message, fold it into a
   separate 'trace' column, or refuse to guess and raise. `to_events()` implements all six once.

Keeping the two apart is the point. Before this module they were welded together - one keyword per
loader that picked a test and a policy at the same time - so a loader could only ever have the one
combination its author needed, and 'merge' meant 'into a trace column' in RawLoader and 'into the
message' in HadoopLoader.

**Everything here groups per file.** A running count over a whole multi-file frame lets a file
whose first line is a continuation attach it to the last event of the *previous, unrelated* file,
and the same mistake makes a forward-filled timestamp cross a file boundary. That is the one thing
a multi-file merge has to get right, so it is done here rather than trusted to each caller.

Nothing in this module runs Python per row - it is Polars expressions and one group_by/agg, the
same as the hand-written versions it replaces.
"""

# The canonical policy names. SyslogLoader's vocabulary, because four of the six were already
# spelled this way there and it is the only one of the three that distinguishes the two merges.
POLICIES = ("drop", "keep", "fill-lastseen", "merge-message", "merge-add-column", "raise")

# RawLoader shipped 'merge' for what is spelled 'merge-add-column' here - it folds the continuation
# lines into a 'trace' column. Its keyword keeps accepting the old name, so this is a rename in the
# implementation only and no caller has to change.
_ALIASES = {"merge": "merge-add-column"}

# Names used only inside to_events(), long enough not to collide with a real column.
_GROUP = "_line_policy_event"
_TEXT = "_line_policy_text"

# A continuation line in every format that indents them: leading space or tab. A blank line counts
# as one too - a traceback routinely has one in the middle, and starting a new event on it would
# split the traceback in half.
_INDENTED = r'^[ \t]'


def normalize_policy(policy, keyword="policy"):
    """Resolve an alias and reject anything that is not a policy.

    Raising here rather than falling through to a silent no-op matters: the loaders used to compare
    their keyword against a chain of elif branches, so a typo meant 'keep' without saying so.
    """
    resolved = _ALIASES.get(policy, policy)
    if resolved not in POLICIES:
        raise ValueError(f"{keyword} must be one of {', '.join(POLICIES)}, got {policy!r}")
    return resolved


def event_start(kind, column=None, pattern=None, text_column=None):
    """A boolean expression answering 'does this line start a new event?'.

    kind:
      'parsed'  - the loader got a value out of this line, so the line was an event. `column` is
                  the capture that proves it, usually 'm_timestamp'.
      'pattern' - `text_column` matches `pattern`. What a loader written against one known dataset
                  uses, HadoopLoader's date regex being the example.
      'indent'  - `text_column` does not begin with whitespace and is not blank. The rule that
                  needs no knowledge of the format at all, which is what makes it the one to reach
                  for when a format was guessed rather than known.

    They are ordinary Polars expressions, so they combine: 'parsed' | 'indent' reads as "a line
    that parsed, or failing that at least one that is not indented" - the useful rule for a log
    whose stack traces outnumber its events.

    A null line is never an event start, so a ragged read cannot manufacture one.
    """
    if kind == "parsed":
        if not column:
            raise ValueError("event_start('parsed') needs column=")
        return pl.col(column).is_not_null()
    if kind == "pattern":
        if not (pattern and text_column):
            raise ValueError("event_start('pattern') needs pattern= and text_column=")
        return pl.col(text_column).str.contains(pattern).fill_null(False)
    if kind == "indent":
        if not text_column:
            raise ValueError("event_start('indent') needs text_column=")
        indented = pl.col(text_column).str.contains(_INDENTED)
        blank = pl.col(text_column).str.strip_chars() == ""
        return (~(indented | blank)).fill_null(False)
    raise ValueError(f"kind must be 'parsed', 'pattern' or 'indent', got {kind!r}")


def starts_event(lines):
    """event_start('indent') for a Series of lines rather than for a column of a frame.

    AutoLoader scores a *sample* of a file before it has a frame at all, and needs the same answer
    the loader it is about to build will get. Two definitions of "this line starts an event" that
    could drift apart is exactly the bug this module exists to prevent, so there is one test with
    two entry points instead.
    """
    return ~(lines.str.contains(_INDENTED) | (lines.str.strip_chars() == ""))


def to_events(df, starts, policy="merge-message", text_column="m_message", text=None,
              timestamp_column="m_timestamp", partition_by="file_name", trace_column="trace"):
    """Turn a frame of lines into a frame of events, by the given policy.

    - starts: the expression from event_start(), or any boolean expression over df.
    - policy: one of POLICIES, or RawLoader's 'merge' alias.
    - text_column: the column holding the line's text, and where merged text is written back.
    - text: where to *read* the text from, if that is not text_column itself - SyslogLoader keeps
      the raw line beside the parsed message and wants whichever it has.
    - partition_by: the column that separates files. Ignored when the frame does not have it,
      which is the single-file case.

    'keep' returns the frame untouched, and is not a no-op worth optimizing away: it is what a
    caller asks for when it wants every line countable.
    """
    policy = normalize_policy(policy)
    if policy == "keep" or df.is_empty():
        return df

    partition = [partition_by] if partition_by and partition_by in df.columns else []
    is_start = starts.cast(pl.Boolean).fill_null(False)

    if policy == "raise":
        stray = df.filter(~is_start)
        if stray.is_empty():
            return df
        example = str(stray.select(text_column).item(0, 0))[:200]
        raise ValueError(
            f"{len(stray)} of {len(df)} lines do not start an event. This is normal for "
            f"multi-line messages - use policy 'merge-message', 'merge-add-column', 'keep', "
            f"'drop' or 'fill-lastseen' to tolerate them. First one: {example}")

    if policy == "drop":
        return df.filter(is_start)

    if policy == "fill-lastseen":
        # The line stays its own row and borrows the event's clock, which is the opposite of
        # merging rather than a variant of it: nothing is joined, nothing disappears.
        filled = pl.col(timestamp_column).fill_null(strategy="forward")
        return df.with_columns(filled.over(partition) if partition else filled)

    into_message = policy == "merge-message"
    source = text if text is not None else pl.col(text_column)
    # The group is a running count of event starts, so every line that did not start one lands in
    # the event above it. Per file, or a file opening with a continuation would join the previous
    # file's last event.
    event = is_start.cum_sum()
    frame = df.with_columns(
        source.alias(_TEXT), (event.over(partition) if partition else event).alias(_GROUP))

    carried = [c for c in df.columns if c not in partition and c != text_column]
    aggregations = [pl.col(c).first().alias(c) for c in carried]
    if into_message:
        # The event's text does not end until the next event begins, so all of it is the message.
        aggregations.append(pl.col(_TEXT).str.join("\n").alias(text_column))
    else:
        # first()/slice(1) rather than filtering on "did it start an event", so a continuation with
        # no event above it in its own file keeps its first line as the message instead of vanishing.
        aggregations.append(pl.col(_TEXT).first().alias(text_column))
        aggregations.append(pl.col(_TEXT).slice(1).str.join("\n").alias(trace_column))

    merged = frame.group_by(partition + [_GROUP], maintain_order=True).agg(aggregations)
    if not into_message:
        # An event with no continuation lines has no trace, rather than an empty one.
        merged = merged.with_columns(
            pl.when(pl.col(trace_column) == "").then(None)
            .otherwise(pl.col(trace_column)).alias(trace_column))

    # Every column the caller came in with, in the order it had them - a merge changes how many
    # rows there are, not what the frame is.
    columns = list(df.columns)
    if text_column not in columns:
        columns.append(text_column)
    return merged.select(columns + ([] if into_message else [trace_column]))
