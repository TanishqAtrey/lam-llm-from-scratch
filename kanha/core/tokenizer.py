"""
kanha/core/tokenizer.py
SentencePiece tokenizer wrapper with special tokens.

Handles:
  - Encoding text → token ids
  - Decoding token ids → text
  - Special tokens: <pad>, <unk>, <s>, </s>
  - Training a new tokenizer from raw text
"""

import os
import sentencepiece as spm
from typing import List, Union
from kanha.utils.logging import get_logger
from kanha.utils.config import cfg

log = get_logger(__name__)


class KanhaTokenizer:
    """
    Thin wrapper around SentencePiece.

    Usage:
        tok = KanhaTokenizer("models/tokenizer/tokenizer.model")
        ids = tok.encode("Hello world")
        txt = tok.decode(ids)
    """

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = cfg.paths.tokenizer_model

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found at: {model_path}\n"
                f"Run: python scripts/train_tokenizer.py"
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.GetPieceSize()
        log.info(f"Tokenizer loaded | vocab: {self.vocab_size:,}")

    # ─── Encoding ──────────────────────────────────────────────────────────
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int = None,
    ) -> List[int]:
        """
        text → list of integer token ids.

        Args:
            text       : input string
            add_bos    : prepend <s> token
            add_eos    : append </s> token
            max_length : truncate to this length (including special tokens)
        """
        ids: List[int] = self.sp.Encode(text)

        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Encodes a list of strings."""
        return [self.encode(t, **kwargs) for t in texts]

    # ─── Decoding ──────────────────────────────────────────────────────────
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        List of token ids → string.

        Args:
            ids           : list of integer token ids
            skip_special  : remove <s>, </s>, <pad> tokens
        """
        if skip_special:
            ids = [i for i in ids if i not in (self.BOS_ID, self.EOS_ID, self.PAD_ID)]
        return self.sp.Decode(ids)

    def decode_batch(self, batch: List[List[int]], **kwargs) -> List[str]:
        """Decodes a batch of token id lists."""
        return [self.decode(ids, **kwargs) for ids in batch]

    # ─── Padding / Truncation ──────────────────────────────────────────────
    def pad(self, batch: List[List[int]], max_len: int = None) -> List[List[int]]:
        """
        Pads a batch of token id lists to the same length.
        Truncates from the right if max_len given.
        """
        if max_len is None:
            max_len = max(len(s) for s in batch)
        return [
            s[:max_len] + [self.PAD_ID] * max(0, max_len - len(s))
            for s in batch
        ]

    # ─── Vocabulary helpers ────────────────────────────────────────────────
    def token_to_id(self, token: str) -> int:
        return self.sp.PieceToId(token)

    def id_to_token(self, idx: int) -> str:
        return self.sp.IdToPiece(idx)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"KanhaTokenizer(vocab_size={self.vocab_size})"


# ─── Training a new tokenizer from scratch ────────────────────────────────────
def train_tokenizer(
    input_files: Union[str, List[str]],
    model_prefix: str = "models/tokenizer/tokenizer",
    vocab_size: int = None,
    sample_size: int = 5_000_000,
    num_threads: int = 8,
):
    """
    Trains a BPE SentencePiece tokenizer from raw text files.

    Args:
        input_files  : path or list of paths to raw .txt files
        model_prefix : output path (will create .model and .vocab)
        vocab_size   : vocabulary size (defaults to config value)
    """
    from kanha.utils.helpers import ensure_dir
    import os

    if vocab_size is None:
        vocab_size = cfg.tokenizer.vocab_size

    if isinstance(input_files, list):
        input_files = ",".join(input_files)

    ensure_dir(os.path.dirname(model_prefix))

    log.info(f"Training tokenizer | vocab_size={vocab_size} | input={input_files}")

    spm.SentencePieceTrainer.Train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        character_coverage=0.9995,
        num_threads=num_threads,
        shuffle_input_sentence=True,
        input_sentence_size=sample_size,   # Fix: caps RAM — samples N sentences instead of loading full file
    )
    log.info(f"Tokenizer saved to {model_prefix}.model")