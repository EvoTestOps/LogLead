# This file demonstrates AutoLoader: the loader for when you do not know which loader you need.
# It looks at a file, works out what format it is in, and builds the loader that reads that format.
#
# The first two sections need no download at all - they write a handful of real log lines to a
# temporary folder and detect them - so this runs anywhere. The third section needs the public
# datasets and is skipped unless LOG_DATA_PATH points at them.

import os
import shutil
import tempfile

from dotenv import load_dotenv, find_dotenv

from loglead.loaders import AutoLoader, detect_format

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

load_dotenv(find_dotenv())
full_data = os.getenv("LOG_DATA_PATH")

# One real line per format, taken from the corpora LogLead already tests against. Enough for
# detection: it scores a sample of the file, and these files are all sample.
SAMPLES = {
    "messages.log": [
        "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 "
        "euid=0 tty=NODEVssh ruser= rhost=218.188.2.4",
        "Jun 14 15:16:02 combo sshd(pam_unix)[19940]: check pass; user unknown",
        "Jun 14 15:16:02 combo sshd(pam_unix)[19941]: session opened for user cyrus",
    ],
    "agent.log": [
        'ts=2024-05-29T13:44:15.803Z caller=main.go:120 level=info msg="starting Grafana Agent"',
        'ts=2024-05-29T13:44:15.912Z caller=server.go:191 level=info msg="server listening" addr=:80',
        'ts=2024-05-29T13:44:16.004Z caller=reporter.go:75 level=warn msg="failed to report" err=eof',
    ],
    "access.log": [
        '31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "GET /image/60844 HTTP/1.1" 200 5667 '
        '"https://www.zanbil.ir/" "Mozilla/5.0 (Linux; Android 6.0)"',
        '54.36.149.41 - - [22/Jan/2019:03:56:17 +0330] "GET /filter/27 HTTP/1.1" 200 30577 '
        '"-" "Mozilla/5.0 (compatible; AhrefsBot/6.1)"',
    ],
    "events.json": [
        '{"time": "17/May/2015:08:05:32 +0000", "remote_ip": "93.180.71.3", "remote_user": "-", '
        '"request": "GET /downloads/product_1 HTTP/1.1", "response": 304, "bytes": 0, '
        '"referrer": "-", "agent": "Debian APT-HTTP/1.3"}',
        '{"time": "17/May/2015:08:05:33 +0000", "remote_ip": "93.180.71.3", "remote_user": "-", '
        '"request": "GET /downloads/product_2 HTTP/1.1", "response": 200, "bytes": 3316, '
        '"referrer": "-", "agent": "Debian APT-HTTP/1.3"}',
    ],
    "app.log": [
        "2015-10-18 18:01:47,978 INFO [main] o.a.h.mapreduce.v2.app.MRAppMaster: Created MRAppMaster",
        "2015-10-18 18:01:48,116 INFO [main] o.a.h.mapreduce.v2.app.MRAppMaster: OutputCommitter set",
        "2015-10-18 18:01:48,342 WARN [main] o.a.h.util.NativeCodeLoader: Unable to load library",
    ],
}

folder = tempfile.mkdtemp(prefix="loglead_auto_demo_")
for name, lines in SAMPLES.items():
    with open(os.path.join(folder, name), "w") as handle:
        handle.write("\n".join(lines) + "\n")

try:
    print("=" * 78)
    print("1. Detecting one file at a time - the decision, without loading anything")
    print("=" * 78)
    # detect_format() is importable on its own: it answers "which loader, with what arguments",
    # which is useful when you want to see the decision before acting on it.
    for name in SAMPLES:
        detection = detect_format(os.path.join(folder, name))
        print(f"  {name:14} -> {detection.loader.__name__:16} {detection.format:26} "
              f"matched {detection.rate:.0%} of the sample")

    print()
    print("=" * 78)
    print("2. Loading a folder that holds all five at once")
    print("=" * 78)
    # A log folder holding several formats is the normal case rather than the exception, so
    # detection happens per file and the results are stacked into one frame.
    loader = AutoLoader(filename=folder, filename_pattern="*")
    df = loader.execute()
    print(f"\n  {df.height} rows and {df.width} columns from {len(SAMPLES)} files in "
          f"{len(set(loader.detections()['format']))} formats")
    print(f"  m_timestamp is {df.schema['m_timestamp']} with {df['m_timestamp'].null_count()} "
          f"nulls - normalized, so this frame stacks with any other loader's output")
    print()
    print(loader.detections().select("format", "loader", "rate", "sampled_lines"))
    print()
    print("  Columns each format contributed, e.g. status/method from the access log and "
          "app_name from syslog:")
    print(f"  {sorted(df.columns)}")

finally:
    shutil.rmtree(folder, ignore_errors=True)

print()
print("=" * 78)
print("3. Recognizing a public dataset, labels and all")
print("=" * 78)
# Datasets that have their own loader are recognized from what sits *next to* the log, not only
# from the log: HDFSLoader needs its anomaly_label.csv and HadoopLoader its abnormal_label.txt, and
# those live at a known place relative to the data. That is what lets AutoLoader hand back a
# sequence-level frame with real labels rather than just the raw lines.
if not full_data:
    print("  Skipped: set LOG_DATA_PATH in a .env file to point at your dataset folder.")
    print("  See .env.sample. With it set, this section loads the Hadoop dataset by detection")
    print("  alone and shows that its anomaly labels survived.")
else:
    hadoop_path = os.path.join(full_data, "hadoop")
    if not os.path.isdir(hadoop_path):
        print(f"  Skipped: {hadoop_path} not found. Download it with")
        print("  uv run downloader/download_data.py --config tests/datasets.yml")
    else:
        loader = AutoLoader(filename=hadoop_path)
        df = loader.execute()
        print(f"\n  Event level:    {df.height} rows")
        print(f"  Sequence level: {loader.df_seq.height} rows, "
              f"{loader.df_seq['anomaly'].sum()} of them anomalous")
        print("  The labels came from abnormal_label.txt beside the logs - nothing was passed in.")
