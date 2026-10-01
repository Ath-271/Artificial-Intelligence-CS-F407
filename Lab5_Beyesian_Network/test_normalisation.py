"""
test_normalisation.py — Part VII: Test the Probability Model

Checks the probabilistic invariant that must hold if the model is
correct: for every context word w,  sum_v P(v | w) = 1.
"""
from dataset import build_dataset
from first_order_model import FirstOrderModel


def test_normalisation(model, tol=1e-9):
    print(f"{'word':10s} {'sum P(v|word)':>15s}   ok?")
    all_ok = True
    for word, dist in model.probabilities.items():
        total = sum(dist.values())
        ok = abs(total - 1.0) < tol
        all_ok = all_ok and ok
        print(f"{word!r:10s} {total:15.10f}   {'OK' if ok else 'FAIL'}")
    print(f"\nAll contexts normalise to 1.0: {all_ok}")
    return all_ok


if __name__ == "__main__":
    data = build_dataset()
    model = FirstOrderModel(data)
    test_normalisation(model)
