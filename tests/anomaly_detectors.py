import glob
import os
import yaml
import polars as pl
import argparse

from loglead import AnomalyDetector, select_predictors, print_predictor_report

# Set up argument parser
parser = argparse.ArgumentParser(description='Dataset Loader Configuration')
parser.add_argument('--config', type=str, default='datasets.yml', help='Path to the YAML file containing dataset information. Default is datasets.yml.')
args = parser.parse_args()

# Read the configuration file
config_file = args.config
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

test_data_path = os.path.expanduser(config['root_folder'])
test_data_path = os.path.join(test_data_path, "test_data") 
 
# Get all .parquet files in the directory
all_files = glob.glob(os.path.join(test_data_path, "*.parquet"))

print("Anomaly detectors test starting.")
print(f"Found files {all_files}")

# Extract unique dataset names with '_eh_' , excluding '_seq' files
datasets = set()
for f in all_files:
    basename = os.path.basename(f)
    if "_lo" in basename and "_eh" in basename and "_seq" not in basename:
        dataset_name = basename.replace(".parquet", "")
        datasets.add(dataset_name)

cols_event = ["m_message", "e_words", "e_event_drain_id", "e_trigrams", "e_event_tip_id"] 
numeric_cols = ["seq_len", "eve_len_max", "duration_sec", "eve_len_over1", "nep_prob_nmax_avg", "nep_prob_nmax_min"]

# Detectors that never look at the labels, so they also run on unlabeled data. LOF and OneClassSVM
# are unsupervised too, but too slow to keep in the test suite.
unsupervised_methods = ["train_IsolationForest", "train_KMeans", "train_RarityModel", "train_OOVDetector"]

# RarityModel scores how rare a row's terms are and OOVDetector counts terms missing from the
# training vocabulary; both need the term matrix that only item_list_col builds, so neither means
# anything over structured columns.
_NEEDS_ITEM_LIST = {"train_RarityModel", "train_OOVDetector"}


def run_anomaly_scoring(df, cols_event, numeric_cols, test_frac):
    """Unsupervised anomaly scoring for data that carries no labels.

    With no ground truth there is nothing to be accurate against, so accuracy/F1/AUC-ROC are not
    computed here - print_scores and store_scores both route through accuracy_score and so must
    stay off. What is tested instead is that the split, the fit and the scoring all run and yield
    one binary prediction plus one continuous score per test row. auc_roc=True is what makes
    AnomalyDetector emit that continuous score ('pred_ano_proba'); despite the name it computes no
    AUC-ROC by itself.
    """
    predictor_sets = [{"item_list_col": col} for col in cols_event if col in df.columns]
    if all(column in df.columns for column in numeric_cols):
        predictor_sets.append({"numeric_cols": numeric_cols})
    if not predictor_sets:
        print(f"Skipped anomaly scoring. None of the predictor columns {cols_event} are present.")
        return

    for predictors in predictor_sets:
        col = predictors.get("item_list_col", "numeric columns")
        print(f"Running unsupervised anomaly scoring with {col}")
        sad = AnomalyDetector(print_scores=False, store_scores=False, auc_roc=True, **predictors)
        sad.test_train_split(df, test_frac=test_frac)
        expected_rows = len(sad.test_df)
        for method in unsupervised_methods:
            # RarityModel and OOVDetector need the sparse term matrix that only item_list_col builds.
            if "numeric_cols" in predictors and method in ("train_RarityModel", "train_OOVDetector"):
                continue
            if col == "m_message" and method == "train_OOVDetector":
                continue  # Skipped in the labeled path too
            getattr(sad, method)()
            scored = sad.predict()
            scores = scored["pred_ano_proba"]
            if len(scored) != expected_rows or scores.null_count():
                print(f"  MISMATCH! {method} returned {len(scored)} rows and {scores.null_count()} "
                      f"null scores for {expected_rows} test rows.")
                continue
            # KMeans always reports 0 flagged: AnomalyDetector.predict() derives the flag with
            # "< 0", but KMeans returns cluster indices, which are never negative. Its distance
            # score is the informative output.
            print(f"  {method}: flagged {scored['pred_ano'].sum()}/{expected_rows}, "
                  f"score range [{scores.min():.4f}, {scores.max():.4f}]")


def _predictor_columns(df, dataset_config):
    """The structured columns to predict from, either pinned in the YAML or chosen by profiling.

    Pinning (`predictor_cols: {numeric: [...], categorical: [...]}` on the dataset entry) makes a
    run reproducible; profiling keeps working when a dataset gains columns or a new dataset is
    added. Either way the columns in cols_event are excluded, since those are already tested as
    item_list_col and would otherwise be counted twice.
    """
    pinned = (dataset_config or {}).get("predictor_cols") or {}
    if pinned:
        numeric = [c for c in pinned.get("numeric", []) if c in df.columns]
        categorical = [c for c in pinned.get("categorical", []) if c in df.columns]
        missing = [c for c in (pinned.get("numeric", []) + pinned.get("categorical", []))
                   if c not in df.columns]
        if missing:
            print(f"  WARNING! pinned predictor column(s) not in data, ignored: {missing}")
        return numeric, categorical
    # file_name is where a row came from, not something the log said. Datasets that put one
    # scenario per file therefore leak the label through it: on Nezha WebShop it sorts 78.9% of
    # rows into groups that are entirely normal or entirely anomalous, which is what took that
    # dataset to F1 0.999 - a filename lookup, not log analysis.
    numeric, categorical, profile = select_predictors(df, exclude=set(cols_event) | {"file_name"})
    print_predictor_report(profile)
    return numeric, categorical


def run_structured_predictors(df, test_frac, dataset_config=None):
    """Detect anomalies from the fields the log arrived with, rather than from its message text.

    Thunderbird lands component/userid/location, BGL lands type/level, a JSON log lands its keys -
    fixed schemas that carry signal on their own. Categorical ones cannot go through numeric_cols
    (sklearn gets an object array and the fit dies), so they are passed as categorical_cols and
    one-hot encoded.

    Labeled data gets the full supervised sweep and a scored table; unlabeled data gets the same
    unsupervised scoring as run_anomaly_scoring, since there is nothing to be accurate against.
    """
    numeric, categorical = _predictor_columns(df, dataset_config)
    # A null in a numeric column reaches sklearn as NaN, which most models refuse to fit. Choosing
    # a fill value would be a modelling decision this harness has no basis to make, so drop those
    # columns instead - categorical nulls are fine, the encoder gives them their own category.
    null_numeric = [c for c in numeric if df[c].null_count()]
    if null_numeric:
        print(f"  Dropping numeric column(s) containing nulls (would be NaN to sklearn): {null_numeric}")
        numeric = [c for c in numeric if c not in null_numeric]
    if not numeric and not categorical:
        print("  Skipped structured-column detection: no usable predictor columns.")
        return
    print(f"  Predicting from numeric={numeric} categorical={categorical}")

    labeled = "anomaly" in df.columns
    normal_count = df["normal"].sum() if "normal" in df.columns else 0
    anomaly_count = df["anomaly"].sum() if "anomaly" in df.columns else 0
    if labeled and not (normal_count > 9 and anomaly_count > 9):
        print(f"  Skipped: too few normal ({normal_count}) or anomaly ({anomaly_count}) instances.")
        return

    # Narrow before splitting. test_train_split() shuffles the whole frame, and on a JSON dataset
    # that means dragging every unused column through the permutation - ait_ads carries 157 columns,
    # 37 of them nested List(Struct), and shuffling those to build a 20-feature matrix peaked at
    # 11.9 GB and got the process OOM-killed. Selecting first costs nothing and drops it to ~1 GB.
    keep = numeric + categorical + [c for c in ("anomaly", "normal") if c in df.columns]
    sad = AnomalyDetector(numeric_cols=numeric, categorical_cols=categorical,
                          print_scores=False, store_scores=labeled, auc_roc=True)
    sad.test_train_split(df.select(keep), test_frac=test_frac)
    print(f"  Encoded {len(numeric)} numeric + {len(categorical)} categorical columns "
          f"into {sad.X_train.shape[1]} features.")

    if labeled:
        sad.evaluate_all_ads(disabled_methods={"train_LOF", "train_OneClassSVM"} | _NEEDS_ITEM_LIST)
        print(sad.storage.calculate_average_scores(score_type="f1", metric="median"))
    else:
        for method in unsupervised_methods:
            if method in _NEEDS_ITEM_LIST:
                continue
            getattr(sad, method)()
            scored = sad.predict()
            scores = scored["pred_ano_proba"]
            print(f"    {method}: flagged {scored['pred_ano'].sum()}/{len(scored)}, "
                  f"score range [{scores.min():.4f}, {scores.max():.4f}]")


def run_anomaly_detectors(df, cols_event, numeric_cols, test_frac):
    # Unlabeled data cannot be evaluated, only scored - every supervised model and every score
    # here needs a ground truth to compare against.
    if "anomaly" not in df.columns:
        print("No 'anomaly' column: data is unlabeled, running unsupervised anomaly scoring instead.")
        run_anomaly_scoring(df, cols_event, numeric_cols, test_frac)
        return

    for col in cols_event:
        disabled_methods = set()
        disabled_methods.add("train_LOF")  # Too slow. Disabled to make tests run faster
        disabled_methods.add("train_OneClassSVM")  # Too slow. Disabled to make tests run faster
        if col == "m_message":
            disabled_methods.add("train_OOVDetector")
        if ((col == "e_trigrams" or col == "m_message") and "nezha_tt" in df.columns):
            disabled_methods.add("train_RF")  # Disable this slow combo.
        normal_count = df["normal"].sum() if "normal" in df.columns else 0
        anomaly_count = df["anomaly"].sum() if "anomaly" in df.columns else 0
        if col in df.columns and "anomaly" in df.columns and normal_count > 9 and anomaly_count > 9:
            print(f"Running anomaly detectors with {col}")
            sad = AnomalyDetector(item_list_col=col, print_scores=False, store_scores=True)
            sad.test_train_split(df, test_frac=test_frac)
            sad.evaluate_all_ads(disabled_methods=disabled_methods)
        else:
            print (f"Skipped column {col}. Missing, no anomaly label or too low number (<10) of normal ({normal_count}) or anomaly ({anomaly_count}) instances")


    # Because this requires all the determined columns, it inherently only runs with sequenced dfs
    all_columns_exist = all(column in df.columns for column in numeric_cols)
    if "anomaly" in df.columns and df["normal"].sum() > 9 and df["anomaly"].sum() > 9 and all_columns_exist:
        print(f"Running seqeuence anomaly detectors with numeric columns {numeric_cols}")
        disabled_methods = {"train_RarityModel", "train_OOVDetector"}
        sad = AnomalyDetector(numeric_cols=numeric_cols, print_scores=False, store_scores=True)
        sad.test_train_split(df, test_frac=test_frac) 
        sad.evaluate_all_ads(disabled_methods=disabled_methods)

for dataset in datasets:
    dataset_config = next((d for d in config['datasets'] if d['name'] in dataset), None)

    if dataset_config and not dataset_config.get('anomaly_detection', True):
        print(f"Skipping anomaly detection for dataset: {dataset}")
        continue

    # Load the event level data
    primary_file = os.path.join(test_data_path, f"{dataset}.parquet")
    seq_file = primary_file.replace(f"{dataset}.parquet", f"{dataset}_seq.parquet")

    if os.path.exists(seq_file):
        df_seq = pl.read_parquet(seq_file)
        print(f"Running sequence anomaly detectors with {seq_file}")
        run_anomaly_detectors(df_seq, cols_event, numeric_cols, test_frac=0.2)
        print(f"Running structured-column detectors with {seq_file}")
        run_structured_predictors(df_seq, test_frac=0.2, dataset_config=dataset_config)
    else:
        df = pl.read_parquet(primary_file)
        print(f"Running event anomaly detectors with {primary_file}")
        run_anomaly_detectors(df, cols_event, numeric_cols, test_frac=0.5)
        print(f"Running structured-column detectors with {primary_file}")
        run_structured_predictors(df, test_frac=0.5, dataset_config=dataset_config)

print("Anomaly detectors test complete.")