"""Writing analysis results to disk.

Replaces LogDelta's ``_write_output``, which read the destination from module
globals set by a separate call. Here the destination is an argument, so two
concurrent analyses cannot write over each other.

The file-name convention is kept identical to LogDelta's so existing output
folders stay readable::

    [<prefix>_]<analysis>_L<level>[_<target>][_vs_<comparison>]_mask=<bool>
        [_format=<X>][_vec=<Y>][_<file>]_<YYYYMMDDHHMMSS>.<ext>
"""

import datetime
import os

import polars as pl

TABLE_FORMATS = ("csv", "xlsx")


def _path_safe(value):
    """Flatten anything that would break out of, or mangle, a file name."""
    for char in "/\\\n\r\t":
        value = value.replace(char, "_")
    return value.strip()


def build_file_name(
    analysis, level=0, target_folder="", comparison_folder="", file="", mask=False,
    content_format="", vectorizer="", prefix="", timestamp=None,
):
    """Assemble the LogDelta-style stem for an output file (no extension)."""
    name = f"{prefix}_{analysis}" if prefix else analysis
    name += f"_L{level}"
    # Log folder names come from directory names and may carry a caller-supplied
    # name on top, so neither is trusted to be path-safe.
    if target_folder:
        name += f"_{_path_safe(target_folder)}"
    if comparison_folder:
        name += f"_vs_{_path_safe(comparison_folder)}"
    name += f"_mask={mask}"
    if content_format:
        name += f"_format={content_format}"
    if vectorizer:
        name += f"_vec={vectorizer}"
    if file:
        name += "_" + _path_safe(file)
    stamp = timestamp or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{name}_{stamp}"


def write_table(df, output_dir, file_stem, table_format="csv"):
    """Write a DataFrame as tab-separated CSV or as XLSX.

    CSV cannot represent nested columns, so List/Struct/Array columns are
    dropped -- use ``xlsx`` when you need to keep e.g. ``e_words``.

    :returns: absolute path written.
    """
    if table_format not in TABLE_FORMATS:
        raise ValueError(
            f"Unknown table_format {table_format!r}. Valid options: {list(TABLE_FORMATS)}"
        )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{file_stem}.{table_format}")

    if table_format == "xlsx":
        df.write_excel(path)
    else:
        flat = [
            col for col, dtype in df.schema.items()
            if not isinstance(dtype, (pl.List, pl.Struct, pl.Array))
        ]
        df.select(flat).write_csv(path, separator="\t")
    return os.path.abspath(path)


def write_figure(fig, output_dir, file_stem):
    """Write a plotly figure as a standalone interactive HTML page.

    :returns: absolute path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{file_stem}.html")
    fig.write_html(path)
    return os.path.abspath(path)
