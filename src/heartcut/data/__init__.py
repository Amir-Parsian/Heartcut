"""Dataset loading into the common ``id, text, label`` schema."""

from heartcut.data.loaders import available_loaders, register_loader, resolve_loader
from heartcut.data.prepare import PreparedData, load_hidden_gold, load_prepared, prepare_data

__all__ = [
    "PreparedData",
    "available_loaders",
    "load_hidden_gold",
    "load_prepared",
    "prepare_data",
    "register_loader",
    "resolve_loader",
]
