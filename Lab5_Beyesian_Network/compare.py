"""
compare.py — Part XIII: Comparing the two models

Measures: number of distinct parameters, number of zero-probability
contexts (observed contexts with only one possible continuation --
i.e. deterministic / "zero entropy", and also genuinely unseen
contexts among all contexts that COULD occur), diversity of generated
sentences, and prints examples for qualitative coherence comparison.
"""
import random
import itertools
from dataset import build_dataset, START, END
from first_order_model import FirstOrderModel
from second_order_model import SecondOrderModel
from generate import render


def count_parameters(cpt):
    """Each context contributes (num_continuations - 1) free parameters
    (probabilities for a discrete distribution sum to 1, so the last
    one is determined), but we also report the raw cell count since
    that's the more intuitive 'table size' for this lab."""
    free_params = sum(len(dist) - 1 for dist in cpt.values())
    raw_cells = sum(len(dist) for dist in cpt.values())
    return free_params, raw_cells


def count_deterministic_contexts(cpt):
    """Contexts where only one continuation was ever observed --
    these are the 'zero-probability-elsewhere' contexts: every other
    vocabulary word has probability exactly 0 given this context."""
    return sum(1 for dist in cpt.values() if len(dist) == 1)


def possible_contexts_coverage(cpt, vocab, order):
    """Of all contexts that COULD exist (vocab^order combinations),
    how many were actually observed (non-zero row) vs unseen?"""
    total_possible = len(vocab) ** order
    observed = len(cpt)
    return observed, total_possible


if __name__ == "__main__":
    data = build_dataset()
    vocab = sorted(set(tok for sent in data for tok in sent))

    m1 = FirstOrderModel(data)
    m2 = SecondOrderModel(data)

    print("=" * 70)
    print("Parameter count")
    print("=" * 70)
    fp1, rc1 = count_parameters(m1.probabilities)
    fp2, rc2 = count_parameters(m2.probabilities)
    print(f"First-order:  {len(m1.probabilities):2d} observed contexts "
          f"(out of {len(vocab)} possible single-word contexts), "
          f"{rc1} table cells, {fp1} free parameters")
    print(f"Second-order: {len(m2.probabilities):2d} observed contexts "
          f"(out of {len(vocab)**2} possible 2-word contexts), "
          f"{rc2} table cells, {fp2} free parameters")

    print()
    print("=" * 70)
    print("Zero-probability / deterministic contexts")
    print("=" * 70)
    d1 = count_deterministic_contexts(m1.probabilities)
    d2 = count_deterministic_contexts(m2.probabilities)
    print(f"First-order:  {d1}/{len(m1.probabilities)} observed contexts are "
          f"deterministic (single continuation, all other words prob 0)")
    print(f"Second-order: {d2}/{len(m2.probabilities)} observed contexts are "
          f"deterministic")
    obs1, poss1 = possible_contexts_coverage(m1.probabilities, vocab, 1)
    obs2, poss2 = possible_contexts_coverage(m2.probabilities, vocab, 2)
    print(f"\nFirst-order:  {obs1}/{poss1} possible 1-word contexts observed "
          f"({poss1-obs1} totally unseen -> underlying distribution undefined there)")
    print(f"Second-order: {obs2}/{poss2} possible 2-word contexts observed "
          f"({poss2-obs2} totally unseen)")

    print()
    print("=" * 70)
    print("Diversity of generated sentences (50 samples each, sampling mode)")
    print("=" * 70)
    rng1 = random.Random(0)
    gen1 = [render(m1.generate_sentence(mode="sample", rng=rng1)) for _ in range(50)]
    rng2 = random.Random(0)
    gen2 = [render(m2.generate_sentence(mode="sample", rng=rng2)) for _ in range(50)]
    print(f"First-order:  {len(set(gen1))} distinct sentences out of 50 samples")
    print(f"Second-order: {len(set(gen2))} distinct sentences out of 50 samples")

    print()
    print("=" * 70)
    print("Qualitative examples (5 each, sampling mode)")
    print("=" * 70)
    rng1b = random.Random(1)
    rng2b = random.Random(1)
    print("First-order samples:")
    for i in range(5):
        print(f"  {i+1}. {render(m1.generate_sentence(mode='sample', rng=rng1b))}")
    print("\nSecond-order samples:")
    for i in range(5):
        print(f"  {i+1}. {render(m2.generate_sentence(mode='sample', rng=rng2b))}")
