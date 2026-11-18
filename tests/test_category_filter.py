import pytest
from astravisionx.models.category_map import (
    classify_detection,
    link_parent_child,
    enrich_detections,
    get_demographic_estimate,
    CATEGORY_PERSONS,
    CATEGORY_VEHICLES,
    CATEGORY_LICENSE_PLATES,
    CATEGORY_CLOTHING,
    CATEGORY_ANIMALS,
    CATEGORY_OTHER
)

def test_classify_detection_categories():
    assert classify_detection("Person")[0] == CATEGORY_PERSONS
    assert classify_detection("Man")[0] == CATEGORY_PERSONS
    assert classify_detection("Woman")[0] == CATEGORY_PERSONS
    assert classify_detection("Human face")[0] == CATEGORY_PERSONS
    assert classify_detection("Car")[0] == CATEGORY_VEHICLES
    assert classify_detection("Bus")[0] == CATEGORY_VEHICLES
    assert classify_detection("Vehicle registration plate")[0] == CATEGORY_LICENSE_PLATES
    assert classify_detection("Jacket")[0] == CATEGORY_CLOTHING
    assert classify_detection("Jeans")[0] == CATEGORY_CLOTHING
    assert classify_detection("Hat")[0] == CATEGORY_CLOTHING
    assert classify_detection("Dog")[0] == CATEGORY_ANIMALS
    assert classify_detection("Cat")[0] == CATEGORY_ANIMALS
    assert classify_detection("Coffee table")[0] == CATEGORY_OTHER

def test_subcategories():
    assert classify_detection("Car")[1] == "car"
    assert classify_detection("Bus")[1] == "bus"
    assert classify_detection("Shirt")[1] == "upper_body"
    assert classify_detection("Jeans")[1] == "lower_body"
    assert classify_detection("Hat")[1] == "headwear"

def test_demographic_estimate():
    assert get_demographic_estimate("Man") == "male"
    assert get_demographic_estimate("Woman") == "female"
    assert get_demographic_estimate("Boy") == "male"
    assert get_demographic_estimate("Girl") == "female"
    assert get_demographic_estimate("Person") == "unclear"

def test_parent_child_containment():
    detections = [
        # 0: Person box (large)
        {"bbox": [10, 10, 300, 500], "class_name": "Person"},
        # 1: Jacket box (contained inside person)
        {"bbox": [30, 80, 250, 300], "class_name": "Jacket"},
        # 2: Car box
        {"bbox": [400, 100, 800, 500], "class_name": "Car"},
        # 3: License plate box (contained inside car)
        {"bbox": [550, 420, 650, 470], "class_name": "Vehicle registration plate"},
        # 4: Stray Hat (outside person)
        {"bbox": [850, 50, 900, 100], "class_name": "Hat"},
    ]
    meta = enrich_detections(detections, enable_demographics=True)
    
    # Assert hierarchy links
    assert detections[1]["parent_index"] == 0
    assert detections[1]["parent_type"] == "person"
    assert detections[3]["parent_index"] == 2
    assert detections[3]["parent_type"] == "vehicle"
    assert detections[4]["parent_index"] is None

    # Assert category counts
    counts = meta["category_counts"]
    assert counts[CATEGORY_PERSONS] == 1
    assert counts[CATEGORY_CLOTHING] == 2
    assert counts[CATEGORY_VEHICLES] == 1
    assert counts[CATEGORY_LICENSE_PLATES] == 1
