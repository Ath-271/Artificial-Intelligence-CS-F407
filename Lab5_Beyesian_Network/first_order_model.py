"""
first_order_model.py — Part V: first-order autoregressive language model,
generated from the lab's suggested prompt and then inspected/corrected
(Part VI) before use.

CORRECTIONS MADE to the first draft before accepting it:
1. The first draft's `predict_most_probable` broke probability ties by
   whichever key Python's dict iteration returned first (insertion
   order) rather than deterministically; this was changed to break
   ties alphabetically so results are reproducible and explainable.
2. The first draft did not raise/handle the case of an unseen previous
   token (KeyError risk) — Question 7 asks what should happen here, so
   `next_token_distribution` was corrected to return an explicit empty
   dict in that case rather than crashing, and callers check for it.
3. A `random.seed(...)` parameter was added to `generate_sentence` for
   reproducibility of the generated-text examples in this report.
"""
import random
from collections import defaultdict, Counter
from dataset import build_dataset, START, END


class FirstOrderModel:
    def __init__(self, sentences):
        """
        sentences: list of token lists, each already wrapped with
                   <START> ... <END> (see dataset.py).
        """
        self.transition_counts = defaultdict(Counter)   # WHERE COUNTS ARE STORED (Q4)
        self._count_transitions(sentences)
        self.probabilities = self._build_cpt()           # WHERE P(X_t | X_t-1) IS COMPUTED (Q5)

    def _count_transitions(self, sentences):
        for sent in sentences:
            for i in range(len(sent) - 1):
                w_i, w_j = sent[i], sent[i + 1]
                self.transition_counts[w_i][w_j] += 1

    def _build_cpt(self):
        """P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)"""
        probabilities = {}
        for w_i, counter in self.transition_counts.items():
            total = sum(counter.values())
            probabilities[w_i] = {w_j: c / total for w_j, c in counter.items()}
        return probabilities

    def show_distribution(self, prev_token):
        """Display P(next | prev_token)."""
        dist = self.probabilities.get(prev_token, {})
        if not dist:
            print(f"No observed transitions from {prev_token!r}.")
            return
        print(f"P(next | {prev_token!r}):")
        for w, p in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"  {w:8s} {p:.4f}")

    def next_token_distribution(self, prev_token):
        """Returns {} if prev_token was never observed (Question 7)."""
        return self.probabilities.get(prev_token, {})

    def predict_most_probable(self, prev_token):
        """argmax_w P(w | prev_token). Ties broken alphabetically."""
        dist = self.next_token_distribution(prev_token)
        if not dist:
            return None
        max_p = max(dist.values())
        candidates = sorted(w for w, p in dist.items() if p == max_p)
        return candidates[0]

    def sample_next(self, prev_token, rng):
        """Sample the next token from P(next | prev_token)."""
        dist = self.next_token_distribution(prev_token)
        if not dist:
            return None   # unseen context: nothing to sample (Question 7)
        words, probs = zip(*dist.items())
        return rng.choices(words, weights=probs, k=1)[0]

    def generate_sentence(self, mode="sample", rng=None, max_len=20):
        """
        mode: 'sample'  -> Mode B, sample from P(w | prev)
              'greedy'  -> Mode A, always take argmax_w P(w | prev)
        """
        if rng is None:
            rng = random.Random()
        tokens = [START]
        for _ in range(max_len):
            prev = tokens[-1]
            nxt = (self.predict_most_probable(prev) if mode == "greedy"
                   else self.sample_next(prev, rng))
            if nxt is None:
                break
            tokens.append(nxt)
            if nxt == END:
                break
        return tokens


if __name__ == "__main__":
    data = build_dataset()
    model = FirstOrderModel(data)
    for w in ["the", "cat", "dog", "sat", "ran"]:
        model.show_distribution(w)
        print()
