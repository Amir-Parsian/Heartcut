"""Tiny randomly initialised transformer checkpoints for hermetic tests.

The real students download pre-trained models from the Hugging Face Hub. Tests build
2-layer, 32-dimensional BERT models with a vocabulary made from the synthetic dataset,
so they run offline in seconds while exercising the same code paths.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from tests.synthetic import generate

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _vocab() -> list[str]:
    words: set[str] = set()
    for row in generate(30, seed=0) + generate(30, seed=1):
        words.update(re.findall(r"\w+|[^\w\s]", row["text"].lower()))
    return SPECIAL + sorted(words)


def build_tiny_bert(directory: Path) -> Path:
    from transformers import BertConfig, BertModel, BertTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(_vocab()) + "\n", encoding="utf-8")
    # transformers 5 takes ``vocab`` and silently ignores the old ``vocab_file`` argument.
    params = inspect.signature(BertTokenizerFast.__init__).parameters
    key = "vocab" if "vocab" in params else "vocab_file"
    tokenizer = BertTokenizerFast(**{key: str(vocab_file)}, do_lower_case=True)
    if "[UNK]" in tokenizer.tokenize("where is my card"):
        raise RuntimeError("tiny tokenizer did not load its vocabulary")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


def build_tiny_sentence_transformer(directory: Path) -> Path:
    from sentence_transformers import SentenceTransformer

    try:  # sentence-transformers >= 5.x
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    except ImportError:  # pragma: no cover - older releases
        from sentence_transformers.models import Pooling, Transformer

    bert = build_tiny_bert(directory.parent / f"{directory.name}-bert")
    transformer = Transformer(str(bert), max_seq_length=32)
    pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="mean")
    SentenceTransformer(modules=[transformer, pooling], device="cpu").save(str(directory))
    return directory
