# Last edited: 2026-10-22 11:05
"""Train the 10K-vocab byte-level BPE tokenizer on TinyStories and save it to data/.

Run from the repo root:  uv run python scripts/train_bpe_tinystories.py
"""
import resource
import sys
import time

from cs336_basics.bpe_io import save_bpe
from cs336_basics.train_bpe import train_bpe

INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_PATH = "data/tinystories_vocab.json"
MERGES_PATH = "data/tinystories_merges.txt"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]


def peak_memory_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes. This only counts this process, not the workers.
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def main() -> None:
    start = time.time()
    vocab, merges = train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)
    elapsed = time.time() - start

    save_bpe(vocab, merges, VOCAB_PATH, MERGES_PATH)

    longest = max(vocab.values(), key=len)
    print(f"Training took {elapsed / 60:.1f} min, peak memory of main process {peak_memory_gb():.1f} GB")
    print(f"Vocab size: {len(vocab)}, merges: {len(merges)}")
    print(f"Longest token ({len(longest)} bytes): {longest!r}")
    print(f"Saved to {VOCAB_PATH} and {MERGES_PATH}")


if __name__ == "__main__":  # required: multiprocessing re-imports this file on macOS
    main()
