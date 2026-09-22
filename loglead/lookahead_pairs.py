# This file implements lookahead pairs, following
# Inoue H., Somayaji A. "Lookahead pairs and full sequences: a tale of two anomaly detection methods",
# 2nd Annual Symposium on Information Assurance, pp. 9-19, 2007,
# and its frequency-based variant from
# Hubballi N. "Pairgram: Modeling frequency information of lookahead pairs for system call based
# anomaly detection", COMSNETS 2012, pp. 1-10.
# See https://github.com/EvoTestOps/LogLead/issues/9


from collections import Counter

import numpy as np

__all__ = ['LookaheadPairs']


class LookaheadPairs:
    """Lookahead pairs: which event may follow which, at which distance.

    A window of ``window + 1`` events slides over each sequence, as in
    NextEventPredictionNgram. The last event of the window pairs with each earlier one, and the
    position in the window is the distance, so each distance ``k = 1..window`` has its own pair
    counter. An event whose pair at some distance was never seen in training has a mismatch there.
    Runs in O(n) time, O(n * window) pair lookups.

    :param window: how far back each event looks.
    :param mode: ``"set"`` scores the share of an event's pairs never seen in training (Inoue &
        Somayaji). ``"freq"`` scores ``1 - P(b at k | a at k)`` averaged over the pairs, so a pair
        seen rarely scores near a mismatch (Hubballi's pairgram).
    """

    _start_ = "SoS"  # Start of Sequence used in padding the sequence
    _end_ = "EoS"  # End of Sequence, lets a sequence that stops early be judged by its end

    def __init__(self, window=10, mode="set"):
        if mode not in ("set", "freq"):
            raise ValueError(f"Unknown mode {mode!r}. Valid options: 'set', 'freq'")
        self.window = window
        self.mode = mode
        # Index k-1 holds distance k
        self.pair_counters = [Counter() for _ in range(window)]  # (a, b) -> times b followed a
        self.anchor_counters = [Counter() for _ in range(window)]  # a -> times a had a follower

    # With window 2 it is SoS SoS E1 E2 E3 EoS: every event, and EoS, has window events before it
    def _pad(self, seq):
        return [self._start_] * self.window + list(seq) + [self._end_]

    # The events k positions before each of seq[window:]
    def _anchors(self, seq, k):
        return seq[self.window - k:len(seq) - k]

    def create_model(self, train_data):
        for seq in train_data:
            seq = self._pad(seq)
            for k in range(1, self.window + 1):
                anchors = self._anchors(seq, k)
                self.pair_counters[k - 1].update(zip(anchors, seq[self.window:]))
                self.anchor_counters[k - 1].update(anchors)

    def predict_list(self, test_data):
        mismatches_list = []
        scores_list = []
        for seq in test_data:
            mismatches, scores = self.predict_and_score(seq)
            mismatches_list.append(mismatches)
            scores_list.append(scores)
        return mismatches_list, scores_list

    # Return two lists with one element per event, plus a last one for the end of sequence
    # 1. mismatches = How many of the event's pairs were never seen in training 0 0 3
    # 2. scores = Share of unseen pairs (set) or 1 - mean pair probability (freq), 0..1
    def predict_and_score(self, seq):
        seq = self._pad(seq)
        followers = seq[self.window:]
        counts = np.array([[counter.get(pair, 0) for pair in zip(self._anchors(seq, k), followers)]
                           for k, counter in enumerate(self.pair_counters, start=1)],
                          dtype=np.float64).reshape(self.window, len(followers))
        mismatches = (counts == 0).sum(axis=0)
        if self.mode == "set":
            scores = mismatches / self.window
        else:
            anchors = np.array([[counter.get(a, 0) for a in self._anchors(seq, k)]
                                for k, counter in enumerate(self.anchor_counters, start=1)],
                               dtype=np.float64).reshape(self.window, len(followers))
            scores = (1.0 - counts / np.maximum(anchors, 1)).mean(axis=0)
        return mismatches.tolist(), scores.tolist()
