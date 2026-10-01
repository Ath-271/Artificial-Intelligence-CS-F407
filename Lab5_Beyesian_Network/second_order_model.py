"""
second_order_model.py — Part XII: second-order autoregressive model,
generated from the Part XII prompt and then inspected/corrected.

CORRECTIONS MADE to the first draft before accepting it:
1. The first draft keyed contexts by a Python list ([w_{t-2}, w_{t-1}]),
   which is unhashable and cannot be a dict key; changed every context
   to a tuple (w_{t-2}, w_{t-1}), consistent with the lab's notation
   P(X_t | X_{t-2}, X_{t-1}).
2. The first draft did not handle generation at t=2 (only one previous
   token exists right after <START>); corrected generate_sentence to
   fall back to the first-order distribution P(X_2 | X_1) for exactly
   that first step, then switch to the second-order model from t=3
   onward -- matching the lab's own chain-rule example on page 2,
   where P(X_2 | X_1) is a first-order term even in the fuller model.
3. Same tie-breaking and "unseen context" handling as first_order_model.py,
   for consistency and to avoid crashing on unseen big("prev2,prev1") pairs.
"""
import random
from collections import defaultdict, Counter
from dataset import build_dataset, START, END
from first_order_model import FirstOrderModel


class SecondOrderModel:
    def __init__(self, sentences):
        self.triple_counts = defaultdict(Counter)   # keyed by (w_{t-2}, w_{t-1})
        self._count_triples(sentences)
        self.probabilities = self._build_cpt()
        # fallback for the very first transition after <START>, where
        # only one previous token exists
        self.first_order_fallback = FirstOrderModel(sentences)

    def _count_triples(self, sentences):
        for sent in sentences:
            for i in range(len(sent) - 2):
                ctx = (sent[i], sent[i + 1])
                nxt = sent[i + 2]
                self.triple_counts[ctx][nxt] += 1

    def _build_cpt(self):
        probabilities = {}
        for ctx, counter in self.triple_counts.items():
            total = sum(counter.values())
            probabilities[ctx] = {w: c / total for w, c in counter.items()}
        return probabilities

    def next_token_distribution(self, ctx):
        """ctx: tuple (w_{t-2}, w_{t-1}). Returns {} if unseen."""
        return self.probabilities.get(ctx, {})

    def show_distribution(self, ctx):
        dist = self.next_token_distribution(ctx)
        if not dist:
            print(f"No observed transitions from context {ctx}.")
            return
        print(f"P(next | {ctx}):")
        for w, p in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"  {w:8s} {p:.4f}")

    def predict_most_probable(self, ctx):
        dist = self.next_token_distribution(ctx)
        if not dist:
            return None
        max_p = max(dist.values())
        return sorted(w for w, p in dist.items() if p == max_p)[0]

    def sample_next(self, ctx, rng):
        dist = self.next_token_distribution(ctx)
        if not dist:
            return None
        words, probs = zip(*dist.items())
        return rng.choices(words, weights=probs, k=1)[0]

    def generate_sentence(self, mode="sample", rng=None, max_len=20):
        if rng is None:
            rng = random.Random()
        tokens = [START]

        # t=2 step: only one previous token (START) exists -> first-order fallback
        prev = tokens[-1]
        nxt = (self.first_order_fallback.predict_most_probable(prev) if mode == "greedy"
               else self.first_order_fallback.sample_next(prev, rng))
        if nxt is None:
            return tokens
        tokens.append(nxt)
        if nxt == END:
            return tokens

        # t>=3: genuine second-order context (w_{t-2}, w_{t-1})
        for _ in range(max_len - 1):
            ctx = (tokens[-2], tokens[-1])
            nxt = (self.predict_most_probable(ctx) if mode == "greedy"
                   else self.sample_next(ctx, rng))
            if nxt is None:
                break
            tokens.append(nxt)
            if nxt == END:
                break
        return tokens


if __name__ == "__main__":
    data = build_dataset()
    model = SecondOrderModel(data)
    for ctx in [("<START>", "the"), ("the", "cat"), ("the", "dog"),
                ("cat", "sat"), ("dog", "sat"), ("sat", "on")]:
        model.show_distribution(ctx)
        print()
