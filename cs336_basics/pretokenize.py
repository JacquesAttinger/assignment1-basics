# Last edited: 2026-10-22 10:52
import os
import re
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

import regex

# GPT-2 pre-tokenization pattern (from the assignment handout, section 2.4).
# It needs the third-party `regex` package because it uses \p{L} and \p{N}.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = regex.compile(PAT)


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Cut `text` at every special token and drop the special tokens themselves.

    No merge may cross a document boundary, and special tokens must not
    contribute to merge counts, so we only keep the text between them.
    """
    if not special_tokens:
        return [text]
    # re.escape: a token such as "<|endoftext|>" contains "|", which is regex syntax.
    delimiter = "|".join(re.escape(token) for token in special_tokens)
    return re.split(delimiter, text)


def pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """Count each pre-token in `text`, with every pre-token written as a tuple of single bytes.

    The GPT-2 regex keeps the leading space, so the first word differs from later ones.
    Example: "low low lower" ->
        {(b'l', b'o', b'w'): 1,
         (b' ', b'l', b'o', b'w'): 1,
         (b' ', b'l', b'o', b'w', b'e', b'r'): 1}
    """
    counts: Counter[tuple[bytes, ...]] = Counter()
    for piece in split_on_special_tokens(text, special_tokens):
        # finditer yields one match at a time, so we never store the full list of words.
        for match in COMPILED_PAT.finditer(piece):
            pretoken_bytes = match.group().encode("utf-8")
            # bytes([b]) makes a length-1 bytes object; Python has no single-byte type.
            counts[tuple(bytes([b]) for b in pretoken_bytes)] += 1
    return counts


# Copied verbatim from cs336_basics/pretokenization_example.py (the handout allows this).
# We cannot import it: that file runs example code at import time.
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _count_chunk(args: tuple[str | os.PathLike, int, int, list[str]]) -> Counter[tuple[bytes, ...]]:
    """Worker: read bytes [start, end) of the file and count its pre-tokens.

    Defined at module top level so `multiprocessing` can send it to other processes.
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretoken_counts(chunk, special_tokens)


def pretoken_counts_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[tuple[bytes, ...]]:
    """Same result as `pretoken_counts` on the whole file, but spread over several processes.

    The file is cut right before a special token, so no document (and no pre-token)
    is split between two chunks. Each process counts its own chunk, then we add the counters.
    """
    num_processes = num_processes or os.cpu_count() or 1
    with open(input_path, "rb") as f:
        if special_tokens:
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        else:
            # No safe cut points exist, so count the whole file as one chunk.
            f.seek(0, os.SEEK_END)
            boundaries = [0, f.tell()]

    jobs = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    total: Counter[tuple[bytes, ...]] = Counter()
    with Pool(len(jobs)) as pool:
        for partial in pool.imap_unordered(_count_chunk, jobs):
            total.update(partial)  # Counter.update adds counts instead of overwriting them
    return total
