import numpy as np
import polars as pl

from .sequence_modelling import LookaheadPairs

__all__ = ['LAPDetector']

#: score name -> function of one sequence's per-event scores. Higher is more anomalous.
SCORES = {
    "max": lambda scores: max(scores),
    "mean": lambda scores: float(np.mean(scores)),
}


class LAPDetector:
    """Lookahead pairs as an anomaly detector over ordered event lists.

    Trains on the sequences of ``train_df[item_list_col]`` and scores each sequence of
    ``test_df[item_list_col]``. Like NEPDetector it ignores the X matrices AnomalyDetector passes,
    as a bag of events has lost the order the model needs.

    ``scores`` holds the ``score`` (see :data:`SCORES`) of the per-event scores of
    :class:`LookaheadPairs`. A sequence is predicted anomalous when at least ``threshold`` of its
    pairs were never seen in training.
    """

    def __init__(self, item_list_col, train_df, test_df, window=10, mode="set", score="max",
                 threshold=1):
        if score not in SCORES:
            raise ValueError(f"Unknown score {score!r}. Valid options: {list(SCORES)}")
        self.item_list_col = item_list_col
        self.train_df = train_df
        self.test_df = test_df
        self.window = window
        self.mode = mode
        self.score = score
        self.threshold = threshold
        self.model = None
        self.scores = None
        self.is_ano = None

    def fit(self, X_train=None, labels=None):
        self.model = LookaheadPairs(window=self.window, mode=self.mode)
        self.model.create_model(self.train_df[self.item_list_col].to_list())

    def predict(self, X_test=None):
        seqs = self.test_df[self.item_list_col].to_list()
        mismatches, event_scores = self.model.predict_list(seqs)
        score_fn = SCORES[self.score]
        self.scores = np.array([score_fn(s) for s in event_scores], dtype=np.float64)
        self.is_ano = (np.array([sum(m) for m in mismatches]) >= self.threshold).astype(int)
        return self.is_ano

    @staticmethod
    def supports(df, item_list_col):
        """True when ``item_list_col`` is an ordered list of string events LAP can model."""
        return bool(item_list_col) and df.schema.get(item_list_col) == pl.List(pl.Utf8)
