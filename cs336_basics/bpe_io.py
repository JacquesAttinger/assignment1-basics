# Last edited: 2026-10-22 10:54
import json
import os


def save_bpe(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """Write a trained vocab and merges to disk.

    Tokens are stored as hex text, because raw bytes can be invalid UTF-8 or contain
    spaces and newlines. vocab file: JSON {"<id>": "<hex>"}. merges file: one "<hex> <hex>" per line.
    """
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(token_id): token.hex() for token_id, token in vocab.items()}, f)
    with open(merges_path, "w", encoding="utf-8") as f:
        for first, second in merges:
            f.write(f"{first.hex()} {second.hex()}\n")


def load_bpe(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Read files written by `save_bpe` and return (vocab, merges)."""
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {int(token_id): bytes.fromhex(hex_token) for token_id, hex_token in json.load(f).items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            first_hex, second_hex = line.split()
            merges.append((bytes.fromhex(first_hex), bytes.fromhex(second_hex)))
    return vocab, merges
