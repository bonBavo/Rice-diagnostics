from organize_dataset import map_label


def test_supported_classes_are_mapped_without_including_other_categories():
    assert map_label("Healthy") == "healthy"
    assert map_label("Normal") == "healthy"
    assert map_label("Blast") == "blast"
    assert map_label("Leaf Blast") == "blast"
    assert map_label("Brownspot") == "brown_spot"
    assert map_label("Bacterialblight") == "bacterial_leaf_blight"
    assert map_label("BLB") == "bacterial_leaf_blight"
    assert map_label("Leaf Scald") == "leaf_scald"
    assert map_label("Narrow Brown Spot") == "narrow_brown_spot"
    assert map_label("Tungro") is None
    assert map_label("Hispa") is None
