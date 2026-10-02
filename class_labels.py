"""Canonical class names and source-label aliases."""

CLASS_NAMES = (
    "healthy",
    "blast",
    "brown_spot",
    "bacterial_leaf_blight",
    "leaf_scald",
    "narrow_brown_spot",
)

_NORMALIZED_LABELS = {
    "healthy": "healthy",
    "normal": "healthy",
    "blast": "blast",
    "riceblast": "blast",
    "leafblast": "blast",
    "brownspot": "brown_spot",
    "blb": "bacterial_leaf_blight",
    "bacterialblight": "bacterial_leaf_blight",
    "bacterialleafblight": "bacterial_leaf_blight",
    "leafscald": "leaf_scald",
    "narrowbrownspot": "narrow_brown_spot",
}


def canonical_class_name(label: str) -> str | None:
    """Map a dataset folder label to a supported canonical class."""
    normalized = "".join(character for character in label.casefold() if character.isalnum())
    return _NORMALIZED_LABELS.get(normalized)
