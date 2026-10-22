# Last edited: 2026-10-22 11:05
"""Encode a text file into token ids and save them as a flat uint16 array.

Run from the repo root, for example:
  uv run python scripts/encode_dataset.py data/TinyStoriesV2-GPT4-valid.txt data/tinystories_valid.bin
  uv run python scripts/encode_dataset.py data/TinyStoriesV2-GPT4-train.txt data/tinystories_train.bin

Load the result later with:  np.memmap(path, dtype=np.uint16, mode="r")
"""
import sys
import time
from itertools import islice

import numpy as np

from cs336_basics.tokenizer import Tokenizer

VOCAB_PATH = "data/tinystories_vocab.json"
MERGES_PATH = "data/tinystories_merges.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]
BLOCK_SIZE = 1_000_000  # ids held in memory at once


def main(input_path: str, output_path: str) -> None:
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)
    # uint16 holds 0..65535, so every id must fit. Our 10K vocab does.
    assert len(tokenizer.vocab) <= 2**16, "vocab too large for uint16"

    start = time.time()
    num_ids = 0
    num_bytes = 0
    with open(input_path, encoding="utf-8") as text_file, open(output_path, "wb") as out_file:
        ids = tokenizer.encode_iterable(text_file)  # lazy: the text file is never fully loaded
        while block := list(islice(ids, BLOCK_SIZE)):
            np.array(block, dtype=np.uint16).tofile(out_file)
            num_ids += len(block)
    with open(input_path, "rb") as f:
        f.seek(0, 2)
        num_bytes = f.tell()

    elapsed = time.time() - start
    print(f"Wrote {num_ids:,} ids to {output_path} in {elapsed / 60:.1f} min")
    print(f"Compression: {num_bytes / num_ids:.2f} bytes/token, throughput: {num_bytes / elapsed / 1e6:.2f} MB/s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
