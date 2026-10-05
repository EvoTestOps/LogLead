import glob
import logging
import os
import re

import polars as pl

from . import line_policy
from .base import BaseLoader

logger = logging.getLogger(__name__)

__all__ = ['LO2LoaderFixed']

SERVICES = ("client", "code", "key", "refresh-token", "service", "token", "user")

# light-oauth2-oauth2-client-1.log: docker compose names the file project-service-replica.
_SERVICE_FILE = r"oauth2-oauth2-(.+)-\d+\.log$"
_RUN_EPOCH = r"_(\d+)$"
# 10:22:07.600 [XNIO-1 task-1]  dr-v-4l0RxK8A5N23TO_NA DEBUG c.n.openapi.ApiNormalisedPath <init> - path = ...
# Hazelcast's own threads log no correlation id, which leaves that field empty.
_EVENT = (r"^(?<time>\d{2}:\d{2}:\d{2}\.\d{3}) \[(?<thread>[^\]]*)\] +(?<correlation_id>\S*) +"
          r"(?<level>TRACE|DEBUG|INFO|WARN|ERROR|FATAL) +(?<logger>\S+) +(?<method>\S+) - (?<message>.*)$")


class LO2LoaderFixed(BaseLoader):
    """Reads the LO2v2 Light-OAuth2 logs: run directories, one directory per test case in each, one
    log file per microservice in each test case. A test case named 'correct' is normal and every
    other name is the error that was injected, so test_case is the label - keep it out of features.

    Unlike LO2Loader it does no sampling. It reads every run, test case and service it is given,
    in sorted order, so the same arguments give the same rows on any machine. Choosing which error
    types go into an experiment belongs to the experiment: pass test_cases, or filter df_seq.

    runs: None for all, an int for the first n in sorted order, or a list of run directory names.
    test_cases: None for all, or a list of test-case names ('correct' is not added implicitly).
    services: None for all seven, or a list of short names such as ['client', 'token'].
    continuation_lines: what to do with lines that carry no timestamp - the stack-trace frames and
      exception headers. One of line_policy.POLICIES. The default 'fill-lastseen' keeps each as its
      own row with the time of the line above. 'merge-message' is offered but misleading on the
      published archive: its logs were thinned to every 20th line, so the line above a frame is
      almost never the event that threw it.
    seq_by: 'service' makes each service log of a test case a sequence (what LO2Loader did);
      'test_case' makes all seven services of a test case one sequence.

    Log times are UTC and carry no date. The date comes from the epoch the run directory is named
    after, which is when the run started; a time earlier than that start means the run passed midnight.
    """

    def __init__(self, filename, df=None, df_seq=None, runs=None, test_cases=None, services=None,
                 continuation_lines="fill-lastseen", seq_by="service"):
        if services is not None:
            unknown = sorted(set(services) - set(SERVICES))
            if unknown:
                raise ValueError(f"Unknown service(s) {unknown}; choose from {', '.join(SERVICES)}")
        if seq_by not in ("service", "test_case"):
            raise ValueError(f"seq_by must be 'service' or 'test_case', got {seq_by!r}")
        self.runs = runs
        self.test_cases = test_cases
        self.services = services
        self.continuation_lines = line_policy.normalize_policy(continuation_lines, "continuation_lines")
        self.seq_by = seq_by
        super().__init__(filename, df, df_seq)

    def _run_dirs(self):
        # A run is any directory holding test-case directories of service logs, which leaves out
        # the 'prerun results' folder the archive ships beside the runs.
        runs = sorted(d for d in os.listdir(self.filename)
                      if next(glob.iglob(os.path.join(glob.escape(os.path.join(self.filename, d)), "*", "*.log")), None))
        if isinstance(self.runs, int):
            return runs[:self.runs]
        if self.runs is not None:
            missing = sorted(set(self.runs) - set(runs))
            if missing:
                raise ValueError(f"Run(s) not found in {self.filename}: {missing}")
            return [r for r in runs if r in set(self.runs)]
        return runs

    def _files(self):
        files = []
        wanted_cases = set(self.test_cases) if self.test_cases is not None else None
        seen_cases = set()
        for run in self._run_dirs():
            run_path = os.path.join(self.filename, run)
            for case in sorted(os.listdir(run_path)):
                if wanted_cases is not None and case not in wanted_cases:
                    continue
                seen_cases.add(case)
                for path in sorted(glob.glob(os.path.join(glob.escape(os.path.join(run_path, case)), "*.log"))):
                    service = re.search(_SERVICE_FILE, os.path.basename(path))
                    if self.services is None or (service and service.group(1) in self.services):
                        files.append(path)
        if wanted_cases is not None and wanted_cases - seen_cases:
            logger.warning("LO2LoaderFixed: test case(s) found in no run read: %s",
                           ", ".join(sorted(wanted_cases - seen_cases)))
        if not files:
            raise ValueError(f"No LO2 log files found under {os.path.abspath(self.filename)} "
                             f"for the given runs, test_cases and services.")
        return files

    def load(self):
        files = self._files()
        # Lazy, so preprocess() runs as one query with the read: Polars then reads and parses the
        # files in parallel, several times faster than parsing a collected frame of this many chunks.
        self.df = pl.scan_csv(
            files, has_header=False, schema={"m_message": pl.String}, infer_schema=False,
            quote_char=None, separator=self._csv_separator, encoding="utf8-lossy",
            truncate_ragged_lines=True, include_file_paths="file_name")
        self._paths = self._path_info(pl.DataFrame({"file_name": files}))

    def preprocess(self):
        lf = self.df.lazy().filter(pl.col("m_message").str.strip_chars() != "")
        paths = getattr(self, "_paths", None)
        if paths is None:
            paths = self._path_info(lf.select(pl.col("file_name").unique(maintain_order=True)).collect())
        lf = self._parse_lines(lf.join(paths.lazy(), on="file_name", how="left", maintain_order="left"))
        self.df = line_policy.to_events(
            lf.collect(), line_policy.event_start("parsed", column="m_timestamp"),
            policy=self.continuation_lines, partition_by="file_name")
        self.df = self.df.drop("file_name").with_columns(normal=pl.col("test_case") == "correct")
        logger.info("LO2LoaderFixed: %d event(s) from %d file(s).", len(self.df), len(paths))
        self.df_seq = self.sequences()

    def _path_info(self, paths):
        run_dir = pl.col("file_name").str.extract(r"([^/\\]+)[/\\][^/\\]+[/\\][^/\\]+$")
        paths = paths.with_columns(
            run=run_dir,
            test_case=pl.col("file_name").str.extract(r"([^/\\]+)[/\\][^/\\]+$"),
            service=pl.col("file_name").str.extract(_SERVICE_FILE),
            run_start=pl.from_epoch(run_dir.str.extract(_RUN_EPOCH).cast(pl.Int64), time_unit="s"),
        )
        if paths["run_start"].null_count():
            logger.warning("LO2LoaderFixed: %d run directory name(s) end in no epoch, so their lines "
                           "get no date and m_timestamp stays null.", paths["run_start"].null_count())
        key = [pl.col("run"), pl.col("test_case")]
        if self.seq_by == "service":
            key.append(pl.col("service"))
        return paths.with_columns(seq_id=pl.concat_str(key, separator="__"))

    @staticmethod
    def _parse_lines(lf):
        lf = lf.with_columns(pl.col("m_message").str.extract_groups(_EVENT).alias("_fields")).unnest("_fields")
        time = pl.col("time").str.to_time("%H:%M:%S%.3f")
        stamp = pl.col("run_start").dt.date().dt.combine(time)
        passed_midnight = time < pl.col("run_start").dt.time()
        return lf.select(
            pl.when(passed_midnight).then(stamp.dt.offset_by("1d")).otherwise(stamp).alias("m_timestamp"),
            # A continuation line has no fields to split off, so its whole text is the message.
            pl.coalesce("message", "m_message").alias("m_message"),
            "thread", pl.col("correlation_id").replace("", None), "level", "logger", "method",
            "run", "test_case", "service", "seq_id", "file_name")

    def sequences(self):
        if self.df is None:
            return None
        key = ["seq_id", "run", "test_case"] + (["service"] if self.seq_by == "service" else [])
        return (self.df.lazy().select(key).unique(maintain_order=True)
                .with_columns(normal=pl.col("test_case") == "correct").collect())

    def check_for_nulls_and_non_utf8(self, source=None):
        # Continuation lines have no thread, level or logger, so nulls there are the expected shape.
        self._log_nulls_and_non_utf8(
            "LO2LoaderFixed", "stack-trace lines carry no thread, level, logger or method.")
