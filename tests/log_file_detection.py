"""Check that AutoLoader picks the same loader tests/loaders.py picks by name.

    uv run tests/log_file_detection.py                          # tests/datasets_mid_labels.yml
    uv run tests/log_file_detection.py --config tests/datasets_json.yml

Nothing is loaded - only the first ~1000 lines of each file are sampled - so this runs in seconds
and needs no memory, which is the point: it covers the datasets that cannot be loaded on an
ordinary machine at all. Thunderbird, Spirit and Liberty are 30-38 GB unzipped and AWSCTD expands
to 174 million rows, so tests/loaders.py skips or OOMs on them, and detection is the only part of
AutoLoader that can be checked against them here.

That AutoLoader *delegates* faithfully once it has decided is checked separately, by
tests/datasets_auto.yml, which re-loads whole corpora through it and compares row counts.
"""
import argparse
import os

import yaml

from loglead.loaders.auto import detect_dataset, detect_format

parser = argparse.ArgumentParser(description='AutoLoader detection check')
parser.add_argument('--config', type=str, default='datasets_mid_labels.yml',
                    help='Path to the YAML file containing dataset information. '
                         'Default is datasets_mid_labels.yml.')
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
full_data_path = os.path.expanduser(config['root_folder'])

# Which loader tests/loaders.py builds for each dataset name. Kept in the same shape as its
# create_correct_loader() if/elif chain, and it has to stay in step with it - that chain is the
# reference answer this file checks detection against.
BY_NAME = {
    "hdfs": "HDFSLoader",
    "hadoop": "HadoopLoader",
    "bgl": "BGLLoader",
    "thunderbird": "ThuSpiLibLoader",
    "spirit": "ThuSpiLibLoader",
    "liberty": "ThuSpiLibLoader",
    "profilence": "ProLoader",
    "nezha": "NezhaLoader",
    "adfa": "ADFALoader",
    "awsctd": "AWSCTDLoader",
    "lo2": "LO2Loader",
}
BY_LOADER_KEY = {
    "access_log": "AccessLogLoader",
    "delimited": "DelimitedLoader",
    "logfmt": "LogfmtLoader",
    "syslog": "SyslogLoader",
}


def expected_loader(name, data):
    """The loader tests/loaders.py would build, or None when there is nothing to compare against."""
    if name in BY_NAME:
        return BY_NAME[name]
    if data.get('loader') in BY_LOADER_KEY:
        return BY_LOADER_KEY[data['loader']]
    if data.get('loader') == 'auto':
        return None  # the entry already asks for detection; there is no second opinion to check
    if 'format' in data:
        return "JsonLoader"
    return None


def dataset_path(name, data):
    """Where AutoLoader would be pointed. A dataset that names one log file gets that file; one
    that is a directory of logs gets the directory, which is also what the dataset probe wants.

    A named log file that is not there is looked for compressed as well: the supercomputer logs
    ship as .gz and are 30-38 GB unpacked, so leaving them packed is the normal state of a machine
    that has downloaded but not extracted them - and Polars reads .gz transparently, so detection
    works on them exactly as it would on the extracted file.
    """
    folder = os.path.join(full_data_path, name)
    if 'log_file' in data and '*' not in data['log_file']:
        named = os.path.join(folder, data['log_file'])
        if os.path.exists(named):
            return named
        packed = f"{os.path.splitext(named)[0]}.gz"
        return packed if os.path.exists(packed) else named
    return folder


print(f"Detection test starting. Data folder: {full_data_path}")
print(f"{'dataset':14} {'expected':17} {'detected':17} {'format':26} {'rate':>6}  result")
print("-" * 94)

failures = missing = 0
for dataset in config['datasets']:
    name = dataset['name']
    expected = expected_loader(name, dataset)
    path = dataset_path(name, dataset)

    if not os.path.exists(path):
        print(f"{name:14} {expected or '(auto)':17} {'-':17} {'path not downloaded':26} "
              f"{'':>6}  SKIP")
        missing += 1
        continue

    try:
        if os.path.isdir(path):
            # Nezha holds two systems in one directory and cannot be detected without being told
            # which; any of its systems will do to confirm the loader choice.
            system = (dataset.get('systems') or [None])[0]
            detection = detect_dataset(path, system=system) or None
            if detection is None:
                # Not a public dataset - fall through to the per-file format probe, using the
                # first file the entry's pattern matches.
                import glob
                pattern = dataset.get('filename_pattern') or '*'
                found = sorted(glob.glob(os.path.join(path, '**', pattern), recursive=True))
                found = [f for f in found if os.path.isfile(f) and os.path.getsize(f) > 0]
                if not found:
                    print(f"{name:14} {expected or '(auto)':17} {'-':17} "
                          f"{'no files match pattern':26} {'':>6}  SKIP")
                    missing += 1
                    continue
                detection = detect_format(found[0])
        else:
            detection = detect_format(path)
    except Exception as error:
        print(f"{name:14} {expected or '(auto)':17} {'RAISED':17} "
              f"{type(error).__name__:26} {'':>6}  FAIL")
        failures += 1
        continue

    got = detection.loader.__name__
    if expected is None:
        result = "(no reference)"
    elif got == expected:
        result = "OK"
    else:
        result = "FAIL"
        failures += 1
    print(f"{name:14} {expected or '(auto)':17} {got:17} {detection.format:26} "
          f"{detection.rate:>6.3f}  {result}")

print("-" * 94)
summary = "All detections match the loader tests/loaders.py would build." if not failures \
    else f"{failures} dataset(s) detected as the WRONG loader."
if missing:
    summary += f" {missing} skipped (not downloaded)."
print(summary)
print("Detection test complete.")
