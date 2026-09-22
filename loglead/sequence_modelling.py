# This file implements two ways of modelling the order of events in a sequence: next event
# prediction and lookahead pairs.
#
# NextEventPredictionNgram is a slightly modified version of
# https://github.com/EvoTestOps/next_event_prediction, presented in
# Mäntylä M., Varela M., Hashemi S.
# "Pinpointing Anomaly Events in Logs from Stability Testing - N-Grams vs. Deep-Learning",
# 5th International Workshop on the Next Level of Test Automation (NEXTA), ICST Workshops, pp. 1-8, 2022,
# Preprint: https://arxiv.org/abs/2202.09214
# Only N-gram version implemented here. For LSTM see link above.
#
# LookaheadPairs follows
# Inoue H., Somayaji A. "Lookahead pairs and full sequences: a tale of two anomaly detection methods",
# 2nd Annual Symposium on Information Assurance, pp. 9-19, 2007,
# and its frequency-based variant from
# Hubballi N. "Pairgram: Modeling frequency information of lookahead pairs for system call based
# anomaly detection", COMSNETS 2012, pp. 1-10.
# See https://github.com/EvoTestOps/LogLead/issues/9


from collections import Counter

import numpy as np

__all__ = ['NextEventPredictionNgram', 'LookaheadPairs']


class NextEventPredictionNgram:
    """ N-gram based next event prediction
        Assumes parsed input e.g. via Drain parser
        Implemented on top of Python dictionaries with O(1) lookup time.
        Runs in O(n) time
    """

    _start_ = "SoS"  # Start of Sequence used in padding the sequence
    _end_ = "EoS"  # End of Sequence used in padding the sequence

    def __init__(self, ngrams=5):
        self.n_gram_window_len = ngrams  # Length of window. Default is 5 (4+1: use 4 to predict 1)
        self.n_gram_counter = Counter()  # Counter for sequences length n_gram_window_len
        self.n_gram_counter_1 = Counter()  # Counter for sequences length n_gram_window_len-1
        self.n1_gram_winner = dict()  # The event n following n-1 gram (the prediction)

    def create_ngram_model(self, train_data):
        ngrams = list()
        ngrams_minus_1 = list()
        for seq in train_data:
            n, n_minus_1 = self.slice_ngrams(seq)
            ngrams.extend(n)
            ngrams_minus_1.extend(n_minus_1)
        self.n_gram_counter += Counter(ngrams)
        self.n_gram_counter_1 += Counter(ngrams_minus_1)

        for idx, s in enumerate(ngrams):
            #precompute the most likely sequence following n-1gram. Needed to keep prediction times fast
            if ngrams_minus_1[idx] in self.n1_gram_winner: #is there existing winner
                n_gram = self.n1_gram_winner[ngrams_minus_1[idx]]
                if self.n_gram_counter[n_gram] < self.n_gram_counter[s]: #there is but we are bigger replace
                    self.n1_gram_winner[ngrams_minus_1[idx]] = s
            else:
                self.n1_gram_winner[ngrams_minus_1[idx]] = s #no n-1-gram key or winner add a new one...

    #Produce required n-grams. E.g. With sequence [e1 e2 e3 e4 e5] and n_gram_window_len=3 we produce [e1 e2 e3], [e2 e3 e4], and [e3 e4 5]
    def slice_ngrams(self, seq):
        #Add SoS and EoS
        #with n-gram 3 it is SoS SoS E1 E2 E3 EoS
        #No need to pad more than one EoS as the final event to be predicted is EoS
        seq = [self._start_]*(self.n_gram_window_len-1) + seq + [self._end_]
        ngrams = list()
        ngrams_minus_1 = list()
        for i in range(self.n_gram_window_len, len(seq)+1):#len +1 because [0:i] leaves out the last element
            ngram_s = seq[i-self.n_gram_window_len:i]
            # convert into a line
            line = ' '.join(ngram_s)
            # store
            ngrams.append(line)
            ngram_s_1= seq[i-self.n_gram_window_len:i-1]
            line2 = ' '.join(ngram_s_1)
            # store
            ngrams_minus_1.append(line2)
        return ngrams, ngrams_minus_1

    def predict_list(self, test_data):
        ngram_preds = []
        ngram_preds_correct = []
        scores_abs_list = []
        scores_prop_norm_sum_list = []
        scores_prop_norm_max_list = []

        # Process sequences
        for normal_s in test_data:
            preds, correct, s_abs, s_prob_norm_sum, s_prob_norm_max = self.predict_and_score(normal_s)
            ngram_preds.append(preds)
            ngram_preds_correct.append(correct)
            scores_abs_list.append(s_abs)
            scores_prop_norm_sum_list.append(s_prob_norm_sum)
            scores_prop_norm_max_list.append(s_prob_norm_max)

        # Return all collected data
        return ngram_preds, ngram_preds_correct, scores_abs_list, scores_prop_norm_sum_list, scores_prop_norm_max_list

    # Return five nep results
    # 1. prediction = What was the predicted event (k=1): E1 E2 E3
    # 2. correct_preds = Binary 1 or 0 if the prediction is correct 1 1 0
    # 3. scores_abs = How many times the given n_gram is seen 323 32 3
    # 4. scores_prob_norm_sum = How probable the ngram is among all ngrams (normalized to sum)
    # 5. scores_prob_norm_max = As above but normalized to the most frequent n-gram
    def predict_and_score(self, seq):
        ngrams, ngrams_minus_1 = self.slice_ngrams(seq)

        correct_preds = []
        predictions = []

        scores_abs = []  # Store absolute scores
        scores_sum = []  # Store absolutes scores for n-1 grams
        scores_max = []  # Store max scores

        for n, n_1 in zip(ngrams, ngrams_minus_1):
            # Predict_seq operations
            to_be_matched_s = n_1
            if to_be_matched_s in self.n1_gram_winner:
                winner = self.n1_gram_winner[to_be_matched_s]
                prediction = winner.rpartition(' ')[2]
                predictions.append(prediction)
                correct_preds.append(1 if winner == n else 0) #1 for correct prediction 0 for incorrect
            else:
                correct_preds.append(0)
                predictions.append("<UNSEEN>") #We have not seen n_1 ever before

            #score sequence operations
            #how many times we see E1 E2 E3
            scores_abs.append(self.n_gram_counter [n])
            #how many times we see E1 E2
            scores_sum.append(self.n_gram_counter_1 [n_1]) #How many times is n_1 seen. This is the sum normalization
            if to_be_matched_s in self.n1_gram_winner:
                winner = self.n1_gram_winner[to_be_matched_s]
                scores_max.append(self.n_gram_counter [winner]) #How many times is the winner (most likely ngram) seen. This is the max normalization
            else:
                scores_max.append(0)

        #Remove 0s from n1 gram list to get rid of division by zero.
        # If n-1 gram is zero following n-gram must be zero as well so it does not effect the results
        scores_sum = [1 if i ==0 else i for i in scores_sum]
        scores_max = [1 if i ==0 else i for i in scores_max]
        #Convert n-gram freq counts to probs of n-gram given n-gram-minus-1 and most likely n-gram
        scores_prop_norm_sum = np.divide(np.array(scores_abs), np.array(scores_sum)).tolist()
        scores_prop_norm_max = np.divide(np.array(scores_abs), np.array(scores_max)).tolist()
        #scores_abs = np.array(scores_abs)
        return predictions, correct_preds, scores_abs, scores_prop_norm_sum, scores_prop_norm_max


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
