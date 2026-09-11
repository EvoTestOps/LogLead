"""Caller-defined named mask patterns, layered on top of :mod:`loglead.delta.masking`.

``masking.PATTERNS`` is a fixed allowlist: a name a caller cannot extend,
because ``EventLogEnhancer.normalize()`` used to build its query by
interpolating pattern text into a string and ``eval()``-ing it, so raw
caller-supplied regex text was a code-injection vector (see the removed
warning in :mod:`loglead.delta.masking`). ``normalize()`` now builds a Polars
expression chain directly -- no eval -- so a caller-supplied ``(replacement,
regex)`` pair is just a Polars regex replace, the same risk as any other
caller-supplied string handed to ``str.replace_all``.

This registry is what lets a caller actually supply one: :meth:`register`
validates a named pattern set (each pattern is test-run through Polars, so a
regex Polars cannot use is rejected at registration time rather than
mid-analysis) and persists it as one JSON file per name under the session
store's cache directory, next to the parquet cache. :meth:`resolve` is the
counterpart to :func:`masking.get_pattern` used everywhere a mask pattern name
is accepted: built-ins first, then the registry, so a custom name can never
shadow a built-in one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from ..delta import masking

#: Custom pattern names are filesystem-safe and visually distinct from paths.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Keeps one bad registration from writing an unbounded file or making
#: `normalize()` chain thousands of `.str.replace_all()` calls.
MAX_PATTERNS = 200
MAX_REGEX_LENGTH = 500
MAX_REPLACEMENT_LENGTH = 200


class MaskPatternRegistry:
    """Named, persisted, caller-defined mask patterns for one cache directory."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}.json"

    def register(
        self,
        name: str,
        patterns: Sequence[dict],
        base: Optional[str] = None,
        description: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict:
        """Validate and persist a named custom mask pattern.

        :param name: how the pattern is referenced later, via the same
            ``mask_pattern`` argument that takes a built-in name.
        :param patterns: ``[{"replacement": str, "regex": str}, ...]``, applied
            in order, after ``base``'s patterns if one is given.
        :param base: an existing pattern name (built-in or custom) whose
            patterns run first; ``patterns`` is appended to it. ``None`` means
            ``patterns`` is the whole mask, replacing nothing.
        :param description: free text, returned by :meth:`list_records` so a
            caller can tell registered patterns apart later.
        :param overwrite: replace an existing registration with this name.
        :raises ValueError: on a bad name, an unknown ``base``, a pattern
            Polars cannot compile, or a name collision without ``overwrite``.
        """
        if not _NAME_RE.match(name):
            raise ValueError(
                f"Invalid mask pattern name {name!r}: use 1-64 characters of "
                "letters, digits, '_' or '-'."
            )
        if name in masking.PATTERNS:
            raise ValueError(
                f"{name!r} is a built-in mask pattern name and cannot be "
                "overridden; choose a different name."
            )
        path = self._path(name)
        if path.exists() and not overwrite:
            raise ValueError(
                f"Mask pattern {name!r} already exists. Pass overwrite=True to replace it."
            )
        if not patterns:
            raise ValueError("patterns must be a non-empty list.")

        resolved = list(self.resolve(base)) if base else []
        if len(resolved) + len(patterns) > MAX_PATTERNS:
            raise ValueError(
                f"{len(resolved) + len(patterns)} patterns exceeds the limit of {MAX_PATTERNS}."
            )
        for i, item in enumerate(patterns):
            if not isinstance(item, dict) or "replacement" not in item or "regex" not in item:
                raise ValueError(
                    f"patterns[{i}] must be a {{'replacement': str, 'regex': str}} object, "
                    f"got {item!r}."
                )
            replacement, regex = str(item["replacement"]), str(item["regex"])
            if len(regex) > MAX_REGEX_LENGTH:
                raise ValueError(f"patterns[{i}].regex exceeds {MAX_REGEX_LENGTH} characters.")
            if len(replacement) > MAX_REPLACEMENT_LENGTH:
                raise ValueError(
                    f"patterns[{i}].replacement exceeds {MAX_REPLACEMENT_LENGTH} characters."
                )
            _check_pattern_compiles(regex, replacement, i)
            resolved.append((replacement, regex))

        record = {
            "name": name,
            "base": base,
            "description": description,
            "patterns": [{"replacement": r, "regex": p} for r, p in resolved],
        }
        path.write_text(json.dumps(record, indent=2))
        return record

    def get(self, name: str) -> list[tuple[str, str]]:
        """The custom pattern ``name``'s resolved ``(replacement, regex)`` pairs.

        :raises KeyError: if ``name`` is not a registered custom pattern.
        """
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        record = json.loads(path.read_text())
        return [(item["replacement"], item["regex"]) for item in record["patterns"]]

    def resolve(self, name: str) -> list[tuple[str, str]]:
        """A pattern's ``(replacement, regex)`` pairs, built-in or custom.

        Same contract as :func:`masking.get_pattern`, extended to also check
        this registry -- the counterpart everywhere a mask pattern name is
        accepted.

        :raises ValueError: if ``name`` is neither a built-in nor a registered
            custom pattern.
        """
        try:
            return masking.get_pattern(name)
        except ValueError:
            pass
        try:
            return self.get(name)
        except KeyError:
            raise ValueError(
                f"Unknown masking pattern {name!r}. Valid options: "
                f"{sorted(masking.PATTERNS)} (built-in), {self.list_names()} (custom)"
            ) from None

    def list_names(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def list_records(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.directory.glob("*.json"))]

    def delete(self, name: str) -> None:
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        path.unlink()


def _check_pattern_compiles(regex: str, replacement: str, index: int) -> None:
    """Run one ``(replacement, regex)`` pair through Polars on a throwaway frame.

    Polars uses the Rust ``regex`` crate, not Python's ``re`` -- syntax
    accepted by one can be rejected by the other -- so this is the same call
    :meth:`EventLogEnhancer.normalize` will make, run early enough that a bad
    pattern fails at registration instead of partway through masking a log root.
    """
    try:
        pl.DataFrame({"x": ["sample text 123"]}).select(
            pl.col("x").str.replace_all(regex, replacement)
        )
    except Exception as exc:
        raise ValueError(f"patterns[{index}] is not a valid Polars regex replacement: {exc}") from None
