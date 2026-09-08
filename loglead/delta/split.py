"""Cutting one big log file into a log root of slices.

Everything else in this package *consumes* a log root: a directory whose files
and subdirectories are log folders to compare against each other. A single log
file -- ``BGL.log``, one 743 MB stream of 4.7 million lines -- is not that, and
no comparison can be made of it, because there is nothing to compare it to.

This module produces one. It cuts the file into ``n_slices`` pieces and writes
them side by side::

    out_dir/BGL_slice_000.log
    out_dir/BGL_slice_001.log
    ...

A log file sitting directly in a log root is a log folder of its own, so that
flat shape *is* a log root and :func:`log_root.read_log_root` reads it with no
further help. It is the same shape as the ``hdfs_balanced_5k`` test corpus.

.. note::
   This is the one analysis-side module that writes files, alongside
   :mod:`loglead.delta.export`. The invariant the rest of the package keeps --
   return frames, never touch the disk -- cannot apply to something whose whole
   job is producing files. It stays honest in the other two respects: no module
   state, and no ``os.chdir``.

Neither mode reads the file into memory. Both stream it through a fixed buffer,
so the cost is bytes moved and the memory is constant whether the log is 700 MB
or 70 GB.
"""

import gzip
import os
import shutil
import time

#: Bytes moved per read/write while streaming. Large enough that the syscall
#: overhead disappears, small enough to stay irrelevant next to a log frame.
_BUFFER = 8 << 20

#: How a file may be cut. Both are constant-memory streaming passes over it.
SPLIT_MODES = ("lines", "bytes")

#: Extensions meaning "this is compressed", stripped from the slice name so a
#: slice of BGL.log.gz is BGL_slice_000.log rather than BGL_slice_000.log.gz --
#: the slices themselves are written as plain text.
_COMPRESSED = (".gz",)


def _open_maybe_gzip(path):
    """Open for binary reading, transparently decompressing a ``.gz``.

    Thunderbird, Spirit and Liberty are usually left packed, and Polars reads
    them that way, so a splitter that refused them would be the one part of the
    pipeline that could not.
    """
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _slice_names(path, n_slices, stem=None):
    """``<stem>_slice_<NNN><ext>`` per slice, zero-padded to fit ``n_slices``."""
    base = os.path.basename(path)
    for suffix in _COMPRESSED:
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    default_stem, ext = os.path.splitext(base)
    if not ext:
        # A log with no extension still deserves one, since filename_pattern
        # defaults to "*.log" and would otherwise match none of the slices.
        ext = ".log"
    stem = stem or default_stem
    width = max(3, len(str(n_slices - 1)))
    return [f"{stem}_slice_{index:0{width}d}{ext}" for index in range(n_slices)]


def _count_lines(path):
    """Line count, streamed. One pass, no line ever held beyond the buffer."""
    total = 0
    tail = b"\n"
    with _open_maybe_gzip(path) as handle:
        for chunk in iter(lambda: handle.read(_BUFFER), b""):
            total += chunk.count(b"\n")
            tail = chunk[-1:]
    # A final line with no trailing newline is still a line.
    if tail and tail != b"\n":
        total += 1
    return total


def _uncompressed_size(path):
    """Bytes the slices will hold in total, which is what has to fit on disk."""
    if not path.lower().endswith(".gz"):
        return os.path.getsize(path)
    total = 0
    with _open_maybe_gzip(path) as handle:
        for chunk in iter(lambda: handle.read(_BUFFER), b""):
            total += len(chunk)
    return total


def _check_out_dir(out_dir, overwrite):
    """Refuse to write into a directory that already holds something."""
    if not os.path.isdir(out_dir):
        return
    existing = os.listdir(out_dir)
    if existing and not overwrite:
        raise FileExistsError(
            f"{out_dir} already holds {len(existing)} entry/entries, e.g. {existing[:3]}. "
            f"Pass overwrite=True to replace them, or name an empty directory."
        )


def _check_disk_space(out_dir, needed):
    """Refuse before writing rather than half way through a 743 MB copy."""
    target = out_dir
    while not os.path.isdir(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    free = shutil.disk_usage(target).free
    if free < needed:
        raise OSError(
            f"Splitting needs {needed / 1e9:.1f} GB for the slices -- they are a second copy of "
            f"the log -- but {target} has {free / 1e9:.1f} GB free."
        )


def _split_by_bytes(source, targets, total_bytes):
    """Equal byte spans, each cut forward to the next newline. One pass.

    What ``split -n l/K`` does. Slices come out equal in bytes, so their line
    counts differ by however much line length varies -- on BGL that is a
    306k-390k spread across ten slices.
    """
    span = total_bytes // len(targets)
    written = []
    with _open_maybe_gzip(source) as reader:
        for index, target in enumerate(targets):
            last = index == len(targets) - 1
            size = lines = 0
            tail = b"\n"
            with open(target, "wb") as writer:
                while not last and size < span:
                    chunk = reader.read(min(_BUFFER, span - size))
                    if not chunk:
                        break
                    writer.write(chunk)
                    size += len(chunk)
                    lines += chunk.count(b"\n")
                    tail = chunk[-1:]
                if last:
                    # The last slice takes the remainder, so rounding never
                    # leaves a tail of the log unwritten.
                    for chunk in iter(lambda: reader.read(_BUFFER), b""):
                        writer.write(chunk)
                        size += len(chunk)
                        lines += chunk.count(b"\n")
                        tail = chunk[-1:]
                elif tail != b"\n":
                    # The span landed mid-line; finish that line, so no slice
                    # begins or ends with half a log line. Skipped when it
                    # happened to land on a boundary, which would otherwise
                    # pull a whole extra line into this slice.
                    rest = reader.readline()
                    if rest:
                        writer.write(rest)
                        size += len(rest)
                        lines += 1
                        tail = rest[-1:]
            if tail and tail != b"\n":
                lines += 1  # last line of the file, with no trailing newline
            written.append((target, size, lines))
    return written


def _split_by_lines(source, targets, total_lines):
    """Equal line counts. The second pass; the first one counted the lines.

    Preferred over equal bytes because slices are only comparable to each other
    if they hold a comparable number of events.
    """
    n = len(targets)
    base, remainder = divmod(total_lines, n)
    # The first `remainder` slices take one extra line, so the quotas sum to
    # exactly total_lines and the last slice is not left holding the rounding.
    quotas = [base + (1 if index < remainder else 0) for index in range(n)]
    written = []
    with _open_maybe_gzip(source) as reader:
        for target, quota in zip(targets, quotas):
            size = lines = 0
            with open(target, "wb") as writer:
                while lines < quota:
                    line = reader.readline()
                    if not line:
                        break
                    writer.write(line)
                    size += len(line)
                    lines += 1
            written.append((target, size, lines))
    return written


def split_log_file(path, out_dir, n_slices=10, by="lines", stem=None, overwrite=False):
    """Cut one log file into ``n_slices`` slices, written side by side as a log root.

    The slices are what the comparison tools then treat as log folders, so
    ``n_slices`` is the question being asked: ten slices of one long log answers
    "did something change part way through?".

    :param path: the log file to split. A ``.gz`` is decompressed on the way in
        and the slices are written as plain text.
    :param out_dir: where the slices go. Must be empty unless ``overwrite``.
    :param n_slices: how many slices to cut.
    :param by: ``"lines"`` for equal line counts (two streaming passes: count,
        then write), or ``"bytes"`` for equal byte spans cut forward to the next
        newline (one pass, the faster of the two, but line counts drift).
    :param stem: name the slices after this instead of the source file's stem.
    :param overwrite: replace whatever ``out_dir`` already holds.
    :returns: a manifest dict -- ``out_dir``, ``by``, ``n_slices``,
        ``total_lines``, ``total_bytes``, ``elapsed_seconds``, and ``slices``,
        one ``{file, bytes, lines}`` per slice.

    Neither mode knows where an event begins, so a multi-line event -- a stack
    trace -- can be cut in two at a slice boundary. There is one boundary per
    cut, so a ten-way split risks nine such events however long the log is; it
    is negligible against millions of lines, but it is not zero.
    """
    started = time.perf_counter()
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Not a file: {path}. split_log_file cuts up one log file; a directory of logs is "
            f"already a log root, so open it with read_log_root instead."
        )
    if by not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode {by!r}. Valid modes: {', '.join(SPLIT_MODES)}.")
    n_slices = int(n_slices)
    if n_slices < 2:
        raise ValueError(
            f"n_slices={n_slices} would produce nothing to compare. Every analysis here judges a "
            f"log folder against the others, so a split needs at least 2 slices."
        )

    out_dir = os.path.abspath(os.path.expanduser(str(out_dir)))
    _check_out_dir(out_dir, overwrite)

    total_bytes = _uncompressed_size(path)
    if total_bytes == 0:
        raise ValueError(f"{path} is empty, so there is nothing to split.")
    _check_disk_space(out_dir, total_bytes)

    total_lines = _count_lines(path) if by == "lines" else None
    if by == "lines" and total_lines < n_slices:
        raise ValueError(
            f"{path} has {total_lines} line(s), which is fewer than the {n_slices} slices asked "
            f"for. Ask for fewer slices."
        )

    os.makedirs(out_dir, exist_ok=True)
    targets = [os.path.join(out_dir, name) for name in _slice_names(path, n_slices, stem)]
    if overwrite:
        for existing in os.listdir(out_dir):
            victim = os.path.join(out_dir, existing)
            if os.path.isdir(victim):
                shutil.rmtree(victim)
            else:
                os.remove(victim)

    if by == "lines":
        written = _split_by_lines(path, targets, total_lines)
    else:
        written = _split_by_bytes(path, targets, total_bytes)

    slices = [
        {"file": os.path.basename(target), "bytes": size, "lines": lines}
        for target, size, lines in written
    ]
    return {
        "source": path,
        "out_dir": out_dir,
        "by": by,
        "n_slices": len(slices),
        "total_lines": sum(entry["lines"] for entry in slices),
        "total_bytes": sum(entry["bytes"] for entry in slices),
        "slices": slices,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
