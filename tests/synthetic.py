"""A small, deterministic synthetic support-intent dataset for tests.

Real benchmark data is never committed; this generator gives the tests a dataset with
the same shape (short customer messages, several intents) in a few milliseconds.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

INTENTS: dict[str, list[str]] = {
    "card_arrival": [
        "my {card} still hasn't arrived",
        "when will my new {card} arrive",
        "i ordered a {card} {when} and it is not here yet",
        "how long does {card} delivery take",
    ],
    "lost_card": [
        "i lost my {card}",
        "my {card} was stolen {when}",
        "someone took my {card}, please block it",
        "i can't find my {card} anywhere",
    ],
    "refund_status": [
        "where is my refund",
        "the refund from {when} is not showing",
        "i was promised a refund {when} but nothing came",
        "how long do refunds take to appear",
    ],
    "top_up_failed": [
        "my top up failed",
        "top up did not go through {when}",
        "why was my top up declined",
        "i tried to top up with my {card} and it failed",
    ],
    "change_pin": [
        "how do i change my pin",
        "i want a new pin for my {card}",
        "can i reset the pin on my {card}",
        "where can i update my pin",
    ],
    "exchange_rate": [
        "what exchange rate do you use",
        "how much is one euro in dollars",
        "is the exchange rate fixed {when}",
        "why did i get a bad exchange rate",
    ],
    "cancel_transfer": [
        "cancel my transfer please",
        "i sent money {when} by mistake, can i cancel it",
        "how do i stop a transfer",
        "please reverse the payment i just made",
    ],
    "close_account": [
        "i want to close my account",
        "how do i delete my account",
        "please terminate my account {when}",
        "close everything, i am leaving",
    ],
}

_FILLERS = {
    "card": ["card", "debit card", "visa card", "new card", "replacement card"],
    "when": ["yesterday", "last week", "two days ago", "this morning", "on monday"],
}
_PREFIXES = ["", "hi, ", "hello ", "help: ", "urgent - "]
_SUFFIXES = ["", "?", "!", " thanks", " asap"]


def generate(n_per_intent: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for intent, templates in INTENTS.items():
        for _ in range(n_per_intent):
            text = rng.choice(templates).format(
                card=rng.choice(_FILLERS["card"]), when=rng.choice(_FILLERS["when"])
            )
            rows.append(
                {"text": rng.choice(_PREFIXES) + text + rng.choice(_SUFFIXES), "intent": intent}
            )
    rng.shuffle(rows)
    return rows


def write_dataset(directory: Path, n_train: int = 25, n_test: int = 10) -> dict[str, Path]:
    """Write train/test JSONL files (200 training examples by default)."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, n, seed in (("train", n_train, 1), ("test", n_test, 2)):
        path = directory / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in generate(n, seed):
                fh.write(json.dumps(row) + "\n")
        paths[split] = path
    return paths
