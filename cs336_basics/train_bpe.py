# Last edited: 2026-10-22 10:47
import os
from collections import Counter, defaultdict

from cs336_basics.pretokenize import pretoken_counts_parallel

Pair = tuple[int, int]


def _count_pairs(
    words: list[list[int]], freqs: list[int]
) -> tuple[Counter[Pair], defaultdict[Pair, set[int]]]:
    """Count every neighbouring token pair, weighted by how often its word occurs.

    Also records which words contain each pair, so a merge only revisits those words.
    """
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: defaultdict[Pair, set[int]] = defaultdict(set)
    for word_index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freqs[word_index]
            pair_to_words[pair].add(word_index)
    return pair_counts, pair_to_words


def _merge_word(word: list[int], pair: Pair, new_id: int) -> list[int]:
    """Return `word` with every non-overlapping left-to-right (a, b) replaced by `new_id`."""
    merged: list[int] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return merged


def _apply_merge(
    pair: Pair,
    new_id: int,
    words: list[list[int]],
    freqs: list[int],
    pair_counts: Counter[Pair],
    pair_to_words: defaultdict[Pair, set[int]],
) -> None:
    """Merge `pair` in every word that has it, and fix the pair counts for just those words."""
    for word_index in list(pair_to_words[pair]):  # copy: the set changes while we loop
        word, freq = words[word_index], freqs[word_index]
        # Take this word's old pairs out of the tables...
        for old_pair in zip(word, word[1:]):
            pair_counts[old_pair] -= freq
            if pair_counts[old_pair] <= 0:
                del pair_counts[old_pair]
            pair_to_words[old_pair].discard(word_index)
        # ...rewrite the word, then put its new pairs back in.
        words[word_index] = word = _merge_word(word, pair, new_id)
        for new_pair in zip(word, word[1:]):
            pair_counts[new_pair] += freq
            pair_to_words[new_pair].add(word_index)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return (vocab, merges).

    vocab maps token id -> bytes: ids 0-255 are the single bytes, then one id per merge,
    then the special tokens. merges lists (token1, token2) in the order they were learned.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # Each distinct pre-token becomes one "word": a list of token ids plus its frequency.
    counts = pretoken_counts_parallel(input_path, special_tokens)
    words = [[byte[0] for byte in pretoken] for pretoken in counts]
    freqs = list(counts.values())
    pair_counts, pair_to_words = _count_pairs(words, freqs)

    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(max(num_merges, 0)):
        if not pair_counts:
            break  # every word is already a single token; nothing left to merge
        # Most frequent pair; ties go to the lexicographically greater pair of byte strings.
        best = max(pair_counts, key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merges.append((vocab[best[0]], vocab[best[1]]))
        _apply_merge(best, new_id, words, freqs, pair_counts, pair_to_words)

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab, merges
