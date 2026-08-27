"""Long-lived sessions: load, mask, and parse a log root exactly once.

This is the point of the MCP server. Loading logs is cheap; masking them and
running a template parser over it is not, and a real investigation runs dozens
of analyses over the same data. LogDelta re-derived every enhanced column on
every step because its analysis functions discarded the frame they enhanced.

A :class:`Session` holds the enhanced frame and grows it in place:

* opening a log root reads it, masks it, and pre-parses whatever was asked for;
* every later analysis calls :meth:`Session.ensure_content`, which adds only
  the column that is actually missing and keeps it;
* the frame is mirrored to a parquet cache, so restarting the server re-attaches
  in seconds instead of re-parsing.
"""

from __future__ import annotations  # `X | None` annotations on Python 3.9

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from ..delta import export, log_root, masking
from ..enhancers import EventLogEnhancer

#: Columns present straight from the loader, before any enhancement. Which
#: columns a session *requires* is :data:`log_root.REQUIRED_COLUMNS`, checked at
#: load time where the loader can still be named in the error.
BASE_COLUMNS = ("m_message", "file_name", "orig_file_name", "folder")


def default_cache_dir():
    """Cache root: ``$LOGLEAD_MCP_CACHE``, else ``$XDG_CACHE_HOME/loglead-mcp``."""
    explicit = os.environ.get("LOGLEAD_MCP_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "loglead-mcp"


@dataclass
class Session:
    """One loaded log root, plus every enhanced column derived from it so far."""

    session_id: str
    root: Path
    filename_pattern: str
    df: pl.DataFrame
    masked: bool
    mask_pattern: str | None
    file_name_normalizer: str
    output_dir: Path
    cache_path: Path | None = None
    #: Format name the log root was read with. See log_root.available_formats().
    format: str = "auto"
    #: {detected format: n files} when "auto" chose per file. Empty on a cache
    #: hit -- nothing was read, so there was nothing to detect.
    detected_formats: dict = dataclass_field(default_factory=dict)
    #: {directory name -> meaningful name} applied to ``folder``. See set_folder_names.
    folder_names: dict = dataclass_field(default_factory=dict)
    #: Whether folder_names kept the folder name as a suffix.
    keep_original_folder_name: bool = True
    #: Kept so the cache key can be recomputed when the names change.
    min_file_size: int = 0
    #: "csv" (tab-separated, drops nested columns) or "xlsx" (keeps them).
    table_format: str = "csv"
    created_at: datetime = dataclass_field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_used_at: datetime = dataclass_field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    #: derived column -> source column it was computed from. See ensure_content.
    content_source: dict = dataclass_field(default_factory=dict)
    _dirty: bool = False

    # -- introspection ----------------------------------------------------- #

    @property
    def folders(self):
        return self.df.select("folder").unique().sort("folder").to_series().to_list()

    @property
    def enhanced_columns(self):
        """The ``e_*`` columns computed so far, i.e. what is cached and free."""
        return sorted(col for col in self.df.columns if col.startswith("e_"))

    @property
    def parsers(self):
        """Parser algorithms already applied, e.g. ``{"tip", "drain"}``."""
        return sorted(
            col[len("e_event_"):-len("_id")]
            for col in self.df.columns
            if col.startswith("e_event_") and col.endswith("_id")
        )

    def summary(self):
        return {
            "session_id": self.session_id,
            "root": str(self.root),
            "filename_pattern": self.filename_pattern,
            "format": self.format,
            "detected_formats": self.detected_formats,
            "n_folders": self.df.select("folder").n_unique(),
            "n_files": self.df.select("file_name").n_unique(),
            "n_rows": self.df.height,
            "masked": self.masked,
            "mask_pattern": self.mask_pattern,
            "file_name_normalizer": self.file_name_normalizer,
            "n_named_folders": len(self.folder_names),
            "table_format": self.table_format,
            "parsers": self.parsers,
            "enhanced_columns": self.enhanced_columns,
            "output_dir": str(self.output_dir),
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "last_used_at": self.last_used_at.isoformat(timespec="seconds"),
        }

    # -- the incremental-enhancement contract ------------------------------ #

    def touch(self):
        self.last_used_at = datetime.now(timezone.utc)

    def ensure_content(self, mask, content_format):
        """Guarantee the column for ``content_format`` exists, and keep it.

        :returns: ``(df, field)``. The frame is this session's own, so the
            column survives for every later call -- the whole reason sessions
            exist.

        Caching across calls needs one guard the batch tools never needed.
        ``EventLogEnhancer`` short-circuits on the *output* column ("e_words
        already found") without checking which *input* it was asked for, so a
        ``mask=True`` call followed by ``mask=False`` would silently hand back
        the masked tokens. We record what each derived column came from and
        recompute it when the source changes.
        """
        self.touch()
        if mask and not self.masked:
            raise ValueError(
                f"Session {self.session_id!r} was opened with mask=False, so there is "
                "no 'e_message_normalized' column. Re-open the log root with mask=True, "
                "or pass mask=False to this analysis."
            )

        field = "e_message_normalized" if mask else "m_message"
        target = log_root.content_column(mask, content_format)
        derived = log_root.derived_columns(content_format)
        if derived and self.content_source.get(target, field) != field:
            stale = [col for col in derived if col in self.df.columns]
            self.df = self.df.drop(stale)
            self._dirty = True

        before = set(self.df.columns)
        df, resolved_field = log_root.prepare_content(self.df, mask, content_format)
        if set(df.columns) != before:
            self.df = df
            self._dirty = True
        if derived:
            self.content_source[target] = field
        return self.df, resolved_field

    def flush(self):
        """Rewrite the parquet cache if new columns were computed."""
        if self._dirty and self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.write_parquet(self.cache_path)
            # Which source each derived column came from is not recoverable
            # from the parquet itself, so it rides alongside it.
            self.cache_path.with_suffix(".json").write_text(
                json.dumps(self.content_source)
            )
            self._dirty = False


class SessionStore:
    """Open sessions, backed by a parquet cache keyed on the logs + preprocessing."""

    def __init__(self, cache_dir=None, output_root=None):
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.output_root = (
            Path(output_root) if output_root else self.cache_dir / "output"
        )
        self._sessions = {}

    # -- cache keying ------------------------------------------------------ #

    # Bumped whenever a change to LogLead itself makes the same inputs produce a different frame,
    # since nothing else in the key would notice. 2: AutoLoader's generic text branch began folding
    # continuation lines into the event that printed them, so a log with stack traces in it now has
    # fewer rows than the cached copy of it does.
    _PREPROCESSING_VERSION = "2"

    def _cache_key(self, root, filename_pattern, mask_pattern, file_name_normalizer,
                   min_file_size, folder_names=None, keep_original_folder_name=True,
                   format="auto"):
        n_files, total_bytes, max_mtime = log_root.count_log_root_files(
            root, filename_pattern, min_file_size
        )
        if n_files == 0:
            raise FileNotFoundError(
                f"No files matching {filename_pattern!r} under {root}"
            )
        # The on-disk fingerprint is cheap (stat only) but catches added, removed,
        # and rewritten files, so a stale parquet cannot be served.
        #
        # Folder names rewrite the folder column and are persisted into the
        # parquet, so they have to be in the key too: a cache hit skips
        # preprocessing entirely, and would otherwise serve a misnamed frame.
        # The format is the most load-bearing part of the key, because it
        # decides which loader read the files and so every column in the frame:
        # the same logs read as "raw" and as "json" agree on nothing but paths.
        # sort_keys because dict order is insertion order, and two equal
        # mappings must hash the same.
        payload = "|".join([
            str(root), filename_pattern, mask_pattern or "", file_name_normalizer,
            str(min_file_size), str(n_files), str(total_bytes), f"{max_mtime:.0f}",
            json.dumps(folder_names or {}, sort_keys=True), str(keep_original_folder_name),
            format, self._PREPROCESSING_VERSION,
        ])
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return digest, n_files

    # -- lifecycle --------------------------------------------------------- #

    def open(self, path, filename_pattern="*.log", mask=True,
             mask_pattern="myllari_extended", parsers=(), file_name_normalizer="none",
             min_file_size=0, output_dir=None, table_format="csv", session_id=None,
             refresh=False, folder_names=None, keep_original_folder_name=True,
             format="auto"):
        """Load a log root into a session, reusing the parquet cache when possible.

        :param mask: run :meth:`EventLogEnhancer.normalize` at open time.
            Required by every ``mask=True`` analysis and by all pre-parsing.
        :param mask_pattern: name from :data:`loglead.delta.masking.PATTERNS`.
            Never a raw regex list -- ``normalize()`` ``eval()``s what it is given.
        :param parsers: algorithms to pre-parse, e.g. ``["tip"]``. Parsing at
            open time is optional; analyses parse on demand either way.
        :param table_format: ``"csv"`` or ``"xlsx"`` for result tables.
        :param refresh: ignore any cached parquet and re-read from disk.
        :param folder_names: ``{folder name: meaningful name}`` so output is
            readable. See :func:`log_root.apply_folder_names`.
        :param keep_original_folder_name: keep the folder name as a suffix.
        :param format: which loader reads the files -- a name from
            :func:`log_root.available_formats`. ``"auto"`` detects per file.
        :returns: ``(session, info)`` where ``info`` records the cache outcome.
        """
        if table_format not in export.TABLE_FORMATS:
            raise ValueError(
                f"Unknown table_format {table_format!r}. "
                f"Valid options: {list(export.TABLE_FORMATS)}"
            )
        root = Path(os.path.abspath(os.path.expanduser(str(path))))
        if not root.is_dir():
            raise FileNotFoundError(f"Log root not found: {root}")
        # Validate everything cheap before reading a single log file.
        if session_id and session_id in self._sessions:
            raise ValueError(f"Session id {session_id!r} is already in use.")
        session_id = session_id or f"{root.name}-{uuid.uuid4().hex[:8]}"

        effective_mask_pattern = mask_pattern if mask else None
        if mask:
            masking.get_pattern(mask_pattern)  # validate before doing any work
        if file_name_normalizer not in log_root.FILE_NAME_NORMALIZERS:
            raise ValueError(
                f"Unknown file_name_normalizer {file_name_normalizer!r}. "
                f"Valid options: {sorted(log_root.FILE_NAME_NORMALIZERS)}"
            )
        format = str(format or "auto")
        log_root.resolve_format(format)  # validate before doing any work
        parsers = [str(p).lower().replace("parse-", "") for p in (parsers or [])]
        folder_names = log_root.validate_folder_names(folder_names)

        digest, n_files = self._cache_key(
            root, filename_pattern, effective_mask_pattern, file_name_normalizer,
            min_file_size, folder_names, keep_original_folder_name, format,
        )
        cache_path = self.cache_dir / f"{root.name}-{digest}.parquet"

        started = time.time()
        cache_hit = cache_path.exists() and not refresh
        content_source = {}
        read_info = {}
        if cache_hit:
            df = pl.read_parquet(cache_path)
            sidecar = cache_path.with_suffix(".json")
            if sidecar.exists():
                content_source = json.loads(sidecar.read_text())
        else:
            df, read_info = log_root.read_log_root(root, filename_pattern, min_file_size, format)
            if mask:
                df = EventLogEnhancer(df).normalize(
                    regexs=masking.get_pattern(mask_pattern)
                )
            df = log_root.normalize_file_names(df, file_name_normalizer)
            # Strictly after file-name normalization: strip_folder_id derives the
            # id to strip from the raw directory name, so renaming first would leave
            # file names unmatched across log folders, emptying every L3/L4 result.
            df, _ = log_root.apply_folder_names(df, folder_names, keep_original_folder_name)

        session = Session(
            session_id=session_id,
            root=root,
            filename_pattern=filename_pattern,
            df=df,
            masked=mask,
            mask_pattern=effective_mask_pattern,
            file_name_normalizer=file_name_normalizer,
            output_dir=Path(output_dir) if output_dir else self.output_root / session_id,
            cache_path=cache_path,
            format=format,
            detected_formats=read_info.get("detected_formats", {}),
            folder_names=folder_names,
            keep_original_folder_name=keep_original_folder_name,
            min_file_size=min_file_size,
            table_format=table_format,
            content_source=content_source,
            _dirty=not cache_hit,
        )

        for parser in parsers:
            session.ensure_content(mask, f"Parse-{parser}")
        session.flush()
        self._sessions[session_id] = session

        return session, {
            "cache_hit": cache_hit,
            "cache_path": str(cache_path),
            "n_files_on_disk": n_files,
            "dropped_rows": read_info.get("dropped_rows", 0),
            "elapsed_seconds": round(time.time() - started, 2),
        }

    def set_folder_names(self, session_id, folder_names, keep_original=True):
        """Rename an open session's log folders in place, without re-reading.

        The mapping replaces any previous one -- it is always applied to the
        original folder names, so names never stack. Every enhanced column
        computed so far is kept; only the ``folder`` column changes.

        :returns: ``(session, info)`` with the naming counts.
        """
        session = self.get(session_id)
        folder_names = log_root.validate_folder_names(folder_names)
        df, info = log_root.apply_folder_names(session.df, folder_names, keep_original)

        # The names are part of the cache key, so a new mapping belongs in a
        # different parquet. Repoint before flushing or the old one is clobbered.
        if session.cache_path is not None:
            digest, _ = self._cache_key(
                session.root, session.filename_pattern, session.mask_pattern,
                session.file_name_normalizer, session.min_file_size, folder_names,
                keep_original, session.format,
            )
            session.cache_path = self.cache_dir / f"{session.root.name}-{digest}.parquet"

        session.df = df
        session.folder_names = folder_names
        session.keep_original_folder_name = keep_original
        session._dirty = True
        session.flush()
        return session, info

    def get(self, session_id):
        try:
            session = self._sessions[session_id]
        except KeyError:
            known = sorted(self._sessions)
            raise ValueError(
                f"No open session {session_id!r}. "
                + (f"Open sessions: {known}" if known else "Call open_log_root first.")
            ) from None
        session.touch()
        return session

    def list(self):
        return [s.summary() for s in self._sessions.values()]

    def close(self, session_id):
        session = self.get(session_id)
        session.flush()
        del self._sessions[session_id]
        return session.summary()

    def clear_cache(self):
        """Delete the whole parquet cache. Open sessions keep working in memory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        return str(self.cache_dir)
