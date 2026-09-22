import numpy as np
import polars as pl

from .sequence_modelling import NextEventPredictionNgram

__all__ = ['NEPDetector']

#: score name -> function of one sequence's (nep_abs, nep_prob_nmax) lists. Higher is more anomalous.
SCORES = {
    "nmax_min": lambda s_abs, s_nmax, min_prob: 1 - min(s_nmax),
    "nmax_avg": lambda s_abs, s_nmax, min_prob: 1 - float(np.mean(s_nmax)),
    # Same formula as SequenceEnhancer._perplexity in enhancers/sequence.py, kept separate because
    # that one is vectorized in Polars over a whole list-column instead of one sequence's list.
    "perplexity": lambda s_abs, s_nmax, min_prob: float(
        np.exp(-np.mean(np.log(np.clip(s_nmax, min_prob, None))))),
}


class NEPDetector:
    """Next event prediction as an anomaly detector over ordered event lists.

    Trains an n-gram model on the sequences of ``train_df[item_list_col]`` and scores each
    sequence of ``test_df[item_list_col]``. The X matrices AnomalyDetector passes to ``fit`` and
    ``predict`` are ignored, as a bag of events has lost the order the model needs.

    ``scores`` holds the continuous score chosen with ``score`` (see :data:`SCORES`). A sequence
    is predicted anomalous when it contains at least ``threshold`` n-grams never seen in training.
    """

    def __init__(self, item_list_col, train_df, test_df, ngrams=5, score="nmax_min", threshold=1,
                 min_prob=1e-6):
        if score not in SCORES:
            raise ValueError(f"Unknown score {score!r}. Valid options: {list(SCORES)}")
        self.item_list_col = item_list_col
        self.train_df = train_df
        self.test_df = test_df
        self.ngrams = ngrams
        self.score = score
        self.threshold = threshold
        self.min_prob = min_prob
        self.model = None
        self.scores = None
        self.is_ano = None

    def fit(self, X_train=None, labels=None):
        self.model = NextEventPredictionNgram(ngrams=self.ngrams)
        self.model.create_ngram_model(self.train_df[self.item_list_col].to_list())

    def predict(self, X_test=None):
        seqs = self.test_df[self.item_list_col].to_list()
        _, _, scores_abs, _, scores_nmax = self.model.predict_list(seqs)
        score_fn = SCORES[self.score]
        self.scores = np.array([score_fn(s_abs, s_nmax, self.min_prob)
                                for s_abs, s_nmax in zip(scores_abs, scores_nmax)], dtype=np.float64)
        unseen = np.array([s_abs.count(0) for s_abs in scores_abs])
        self.is_ano = (unseen >= self.threshold).astype(int)
        return self.is_ano

    @staticmethod
    def supports(df, item_list_col):
        """True when ``item_list_col`` is an ordered list of string events NEP can model."""
        return bool(item_list_col) and df.schema.get(item_list_col) == pl.List(pl.Utf8)
