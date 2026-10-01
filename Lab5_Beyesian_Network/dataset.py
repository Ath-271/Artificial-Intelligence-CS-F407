"""
dataset.py — Part III: Build a Small Language Dataset

Lowercases the text, tokenises on whitespace, and wraps each sentence
with <START> and <END> tokens, as specified in the lab.
"""

RAW_SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

START = "<START>"
END = "<END>"


def tokenise(sentence):
    """Lowercase + whitespace tokenisation, wrapped with START/END."""
    tokens = sentence.lower().strip().split()
    return [START] + tokens + [END]


def build_dataset(raw_sentences=RAW_SENTENCES):
    return [tokenise(s) for s in raw_sentences]


if __name__ == "__main__":
    data = build_dataset()
    for sent in data:
        print(" ".join(sent))
