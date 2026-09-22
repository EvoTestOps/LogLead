# This file implements lookahead pairs, following
# Inoue H., Somayaji A. "Lookahead pairs and full sequences: a tale of two anomaly detection methods",
# 2nd Annual Symposium on Information Assurance, pp. 9-19, 2007,
# and its frequency-based variant from
# Hubballi N. "Pairgram: Modeling frequency information of lookahead pairs for system call based
# anomaly detection", COMSNETS 2012, pp. 1-10.
# See https://github.com/EvoTestOps/LogLead/issues/9


from collections import Counter

__all__ = ['LookaheadPairs']


class LookaheadPairs:
    """Lookahead pairs: which event may follow which, at which distance.

    Training records every pair ``(a, b, k)`` where event ``b`` occurs ``k`` positions after
    event ``a`` in a sequence, for ``k = 1..window``. An event is then judged by its backward
    pairs, ``(e[j-k], e[j], k)`` for each ``k``: one never seen in training is a mismatch.

    :param window: how far back each event looks. ``None`` pairs every event with every earlier
        one, which costs O(n^2) per sequence rather than O(n * window).
    :param offset: keep the distance ``k`` in the pair. ``False`` only asks whether ``b`` ever
        followed ``a`` within the window.
    :param mode: ``"set"`` scores the share of an event's pairs never seen in training (Inoue &
        Somayaji). ``"freq"`` scores ``1 - P(b at k | a at k)`` averaged over the pairs, so a pair
        seen rarely scores near a mismatch (Hubballi's pairgram).
    """

    _start_ = "SoS"  # Start of Sequence, lets the first events pair with the start
    _end_ = "EoS"  # End of Sequence, lets a sequence that stops early be judged by its end

    def __init__(self, window=10, offset=True, mode="set"):
        if mode not in ("set", "freq"):
            raise ValueError(f"Unknown mode {mode!r}. Valid options: 'set', 'freq'")
        self.window = window
        self.offset = offset
        self.mode = mode
        self.pair_counter = Counter()  # (a, b, k) -> times b followed a at distance k
        self.anchor_counter = Counter()  # (a, k) -> times a had any event at distance k

    def _pad(self, seq):
        return [self._start_] + list(seq) + [self._end_]

    def _backward_pairs(self, seq, j):
        """The pairs ending at position ``j`` of a padded sequence."""
        reach = j if self.window is None else min(self.window, j)
        for k in range(1, reach + 1):
            yield seq[j - k], seq[j], (k if self.offset else 0)

    def create_model(self, train_data):
        for seq in train_data:
            seq = self._pad(seq)
            for j in range(1, len(seq)):
                for a, b, k in self._backward_pairs(seq, j):
                    self.pair_counter[(a, b, k)] += 1
                    self.anchor_counter[(a, k)] += 1

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
        mismatches = []
        scores = []
        for j in range(1, len(seq)):
            unseen = 0
            miss = 0.0
            n = 0
            for a, b, k in self._backward_pairs(seq, j):
                n += 1
                count = self.pair_counter.get((a, b, k), 0)
                if count == 0:
                    unseen += 1
                    miss += 1.0
                elif self.mode == "freq":
                    miss += 1.0 - count / self.anchor_counter[(a, k)]
            mismatches.append(unseen)
            scores.append(miss / n if n else 0.0)
        return mismatches, scores
