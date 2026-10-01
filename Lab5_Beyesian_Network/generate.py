"""
generate.py — Part IX (generate text) and Part X (greedy vs sampling)
"""
import random
from dataset import build_dataset
from first_order_model import FirstOrderModel


def render(tokens):
    # drop <START>/<END> for readability, keep the raw form available too
    inner = [t for t in tokens if t not in ("<START>", "<END>")]
    return " ".join(inner)


if __name__ == "__main__":
    data = build_dataset()
    model = FirstOrderModel(data)

    print("=" * 70)
    print("Part IX: Generate at least 20 sentences (sampling mode)")
    print("=" * 70)
    rng = random.Random(42)
    generated = []
    for i in range(20):
        toks = model.generate_sentence(mode="sample", rng=rng)
        generated.append(toks)
        print(f"{i+1:2d}. {render(toks)}    (raw: {' '.join(toks)})")

    with open("generated_sentences.txt", "w") as f:
        for toks in generated:
            f.write(" ".join(toks) + "\n")
    print("\nSaved to generated_sentences.txt")

    print()
    print("=" * 70)
    print("Part X: Mode A (greedy) vs Mode B (sampling) -- 5 sentences each")
    print("=" * 70)
    print("\nMode A -- greedy (always argmax):")
    greedy_sentences = []
    for i in range(5):
        toks = model.generate_sentence(mode="greedy")
        greedy_sentences.append(render(toks))
        print(f"  {i+1}. {render(toks)}")

    print("\nMode B -- sampling:")
    rng2 = random.Random(7)
    sampling_sentences = []
    for i in range(5):
        toks = model.generate_sentence(mode="sample", rng=rng2)
        sampling_sentences.append(render(toks))
        print(f"  {i+1}. {render(toks)}")

    print(f"\nDistinct greedy sentences:   {len(set(greedy_sentences))} / 5")
    print(f"Distinct sampled sentences:  {len(set(sampling_sentences))} / 5")
