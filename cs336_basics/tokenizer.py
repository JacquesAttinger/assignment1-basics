# Last edited: 2026-10-22 10:52
import re

from cs336_basics.pretokenize import COMPILED_PAT


class Tokenizer:
    """Byte-level BPE tokenizer built from a trained vocab, merges and special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)  # copy, so adding special tokens never changes the caller's dict
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Merge order: the earlier a merge was learned, the lower its rank (and the sooner it applies).
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(merges)}
        # Remember the ids of pre-tokens we already encoded (words like " the" repeat a lot).
        self._pretoken_cache: dict[bytes, list[int]] = {}

        # Reverse lookup (bytes -> id), used when encoding.
        self.bytes_to_id: dict[bytes, int] = {token: token_id for token_id, token in self.vocab.items()}

        # Special tokens missing from the vocab get fresh ids after the largest existing one.
        for special in self.special_tokens:
            special_bytes = special.encode("utf-8")
            if special_bytes not in self.bytes_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = special_bytes
                self.bytes_to_id[special_bytes] = new_id

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text.

        Look up each id's bytes, join them, then decode as UTF-8. Ids can form byte
        sequences that are not valid UTF-8, so bad bytes become U+FFFD instead of raising.
        """
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        """Turn text into token ids.

        1. Cut out special tokens; each one becomes a single id and is never split.
        2. Pre-tokenize the remaining text with the GPT-2 regex.
        3. Merge the bytes of each pre-token, following the merge order from training.
        """
        ids: list[int] = []
        for piece in self._split_keeping_special_tokens(text):
            if piece in self.special_tokens:
                ids.append(self.bytes_to_id[piece.encode("utf-8")])
                continue
            for match in COMPILED_PAT.finditer(piece):
                ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def _split_keeping_special_tokens(self, text: str) -> list[str]:
        """Split `text` around special tokens, keeping the special tokens in the result."""
        if not self.special_tokens:
            return [text]
        # Longest first, so "<|a|><|a|>" wins over "<|a|>" when both are special tokens.
        longest_first = sorted(self.special_tokens, key=len, reverse=True)
        # The parentheses make re.split keep the matched special tokens in the output.
        pattern = "(" + "|".join(re.escape(token) for token in longest_first) + ")"
        return [piece for piece in re.split(pattern, text) if piece]

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        """Apply BPE merges to one pre-token and return its ids."""
        cached = self._pretoken_cache.get(pretoken)
        if cached is not None:
            return cached

        word = [bytes([b]) for b in pretoken]
        while len(word) > 1:
            # Of all neighbouring pairs in the word, find the one learned earliest in training.
            best = min(zip(word, word[1:]), key=lambda pair: self.merge_ranks.get(pair, float("inf")))
            if best not in self.merge_ranks:
                break  # no pair in this word was ever merged
            word = self._merge_pair(word, best)

        ids = [self.bytes_to_id[token] for token in word]
        self._pretoken_cache[pretoken] = ids
        return ids

    @staticmethod
    def _merge_pair(word: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        """Glue every left-to-right occurrence of `pair` in `word` into one token."""
        merged: list[bytes] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                merged.append(word[i] + word[i + 1])
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return merged
