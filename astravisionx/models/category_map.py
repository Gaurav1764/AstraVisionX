"""
Category mapping for OIV7 YOLO detections.

Maps all 601 Open Images V7 class names into filterable top-level categories,
sub-categories, and handles parent-child linking via bounding-box containment.
"""

from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────
#  Top-level categories
# ──────────────────────────────────────────────────────────────────────
CATEGORY_PERSONS = "persons"
CATEGORY_VEHICLES = "vehicles"
CATEGORY_LICENSE_PLATES = "license_plates"
CATEGORY_CLOTHING = "clothing"
CATEGORY_ANIMALS = "animals"
CATEGORY_OTHER = "other"

ALL_CATEGORIES = [
    CATEGORY_PERSONS, CATEGORY_VEHICLES, CATEGORY_LICENSE_PLATES,
    CATEGORY_CLOTHING, CATEGORY_ANIMALS, CATEGORY_OTHER,
]

CATEGORY_ICONS = {
    CATEGORY_PERSONS: "👤",
    CATEGORY_VEHICLES: "🚗",
    CATEGORY_LICENSE_PLATES: "🔢",
    CATEGORY_CLOTHING: "👕",
    CATEGORY_ANIMALS: "🐾",
    CATEGORY_OTHER: "📦",
}

# ──────────────────────────────────────────────────────────────────────
#  Person classes (full persons + body parts)
# ──────────────────────────────────────────────────────────────────────
_PERSON_FULL = {
    "Person", "Man", "Woman", "Boy", "Girl",
    "Human body",
}
_PERSON_BODY_PARTS = {
    "Human arm", "Human beard", "Human ear", "Human eye",
    "Human face", "Human foot", "Human hair", "Human hand",
    "Human head", "Human leg", "Human mouth", "Human nose",
}
_PERSON_ALL = _PERSON_FULL | _PERSON_BODY_PARTS

DEMOGRAPHIC_MAP = {
    "Man": "male",
    "Boy": "male",
    "Woman": "female",
    "Girl": "female",
    "Person": "unclear",
    "Human body": "unclear",
}

# ──────────────────────────────────────────────────────────────────────
#  Vehicle classes + sub-types
# ──────────────────────────────────────────────────────────────────────
_VEHICLE_CAR = {"Car", "Taxi", "Limousine", "Golf cart"}
_VEHICLE_BUS = {"Bus"}
_VEHICLE_TRUCK = {"Truck", "Ambulance", "Snowplow"}
_VEHICLE_MOTORCYCLE = {"Motorcycle"}
_VEHICLE_BICYCLE = {"Bicycle", "Unicycle", "Segway", "Bicycle wheel"}
_VEHICLE_AIRCRAFT = {"Aircraft", "Airplane", "Helicopter", "Rocket"}
_VEHICLE_WATERCRAFT = {"Boat", "Canoe", "Gondola", "Jet ski", "Barge",
                        "Submarine", "Watercraft"}
_VEHICLE_OTHER = {"Van", "Cart", "Tank", "Train", "Snowmobile",
                  "Land vehicle", "Vehicle", "Auto part", "Tire"}

_VEHICLE_ALL = (_VEHICLE_CAR | _VEHICLE_BUS | _VEHICLE_TRUCK |
                _VEHICLE_MOTORCYCLE | _VEHICLE_BICYCLE | _VEHICLE_AIRCRAFT |
                _VEHICLE_WATERCRAFT | _VEHICLE_OTHER)

_VEHICLE_SUBTYPE = {}
for _cls in _VEHICLE_CAR:        _VEHICLE_SUBTYPE[_cls] = "car"
for _cls in _VEHICLE_BUS:        _VEHICLE_SUBTYPE[_cls] = "bus"
for _cls in _VEHICLE_TRUCK:      _VEHICLE_SUBTYPE[_cls] = "truck"
for _cls in _VEHICLE_MOTORCYCLE: _VEHICLE_SUBTYPE[_cls] = "motorcycle"
for _cls in _VEHICLE_BICYCLE:    _VEHICLE_SUBTYPE[_cls] = "bicycle"
for _cls in _VEHICLE_AIRCRAFT:   _VEHICLE_SUBTYPE[_cls] = "aircraft"
for _cls in _VEHICLE_WATERCRAFT: _VEHICLE_SUBTYPE[_cls] = "watercraft"
for _cls in _VEHICLE_OTHER:      _VEHICLE_SUBTYPE[_cls] = "other_vehicle"

# ──────────────────────────────────────────────────────────────────────
#  License plate class
# ──────────────────────────────────────────────────────────────────────
_LICENSE_PLATE_ALL = {"Vehicle registration plate"}

# ──────────────────────────────────────────────────────────────────────
#  Clothing classes + sub-types
# ──────────────────────────────────────────────────────────────────────
_CLOTHING_UPPER = {"Shirt", "Jacket", "Coat", "Brassiere", "Sports uniform",
                   "Suit", "Clothing"}
_CLOTHING_LOWER = {"Jeans", "Trousers", "Shorts", "Skirt", "Miniskirt"}
_CLOTHING_FULL_BODY = {"Dress", "Swimwear"}
_CLOTHING_HEADWEAR = {"Hat", "Helmet", "Cowboy hat", "Fedora", "Sombrero",
                      "Sun hat", "Bicycle helmet", "Football helmet",
                      "Tiara", "Swim cap", "Crown"}
_CLOTHING_FOOTWEAR = {"Boot", "High heels", "Footwear", "Sandal",
                      "Roller skates"}
_CLOTHING_ACCESSORIES = {
    "Glasses", "Sunglasses", "Goggles", "Belt", "Tie", "Scarf",
    "Glove", "Necklace", "Earrings", "Watch", "Handbag", "Backpack",
    "Suitcase", "Briefcase", "Sock", "Fashion accessory",
    "Luggage and bags", "Perfume", "Cosmetics", "Lipstick",
    "Face powder", "Hair spray", "Personal care",
}

_CLOTHING_ALL = (_CLOTHING_UPPER | _CLOTHING_LOWER | _CLOTHING_FULL_BODY |
                 _CLOTHING_HEADWEAR | _CLOTHING_FOOTWEAR | _CLOTHING_ACCESSORIES)

_CLOTHING_SUBTYPE = {}
for _cls in _CLOTHING_UPPER:       _CLOTHING_SUBTYPE[_cls] = "upper_body"
for _cls in _CLOTHING_LOWER:       _CLOTHING_SUBTYPE[_cls] = "lower_body"
for _cls in _CLOTHING_FULL_BODY:   _CLOTHING_SUBTYPE[_cls] = "full_body"
for _cls in _CLOTHING_HEADWEAR:    _CLOTHING_SUBTYPE[_cls] = "headwear"
for _cls in _CLOTHING_FOOTWEAR:    _CLOTHING_SUBTYPE[_cls] = "footwear"
for _cls in _CLOTHING_ACCESSORIES: _CLOTHING_SUBTYPE[_cls] = "accessories"

# ──────────────────────────────────────────────────────────────────────
#  Animal classes
# ──────────────────────────────────────────────────────────────────────
_ANIMAL_ALL = {
    "Alpaca", "Animal", "Ant", "Antelope", "Armadillo",
    "Bat (Animal)", "Bear", "Bee", "Beetle", "Bird", "Blue jay",
    "Brown bear", "Bull", "Butterfly", "Camel", "Canary", "Carnivore",
    "Cat", "Caterpillar", "Cattle", "Centipede", "Cheetah", "Chicken",
    "Crab", "Crocodile", "Deer", "Dinosaur", "Dog", "Dolphin",
    "Dragonfly", "Duck", "Eagle", "Elephant", "Falcon", "Fish", "Fox",
    "Frog", "Giraffe", "Goat", "Goldfish", "Goose", "Hamster",
    "Harbor seal", "Hedgehog", "Hippopotamus", "Horse", "Insect",
    "Invertebrate", "Isopod", "Jaguar (Animal)", "Jellyfish",
    "Kangaroo", "Koala", "Ladybug", "Leopard", "Lion", "Lizard",
    "Lobster", "Lynx", "Magpie", "Mammal", "Marine invertebrates",
    "Marine mammal", "Monkey", "Moths and butterflies", "Mouse",
    "Mule", "Ostrich", "Otter", "Owl", "Oyster", "Panda", "Parrot",
    "Penguin", "Pig", "Polar bear", "Porcupine", "Rabbit", "Raccoon",
    "Raven", "Rays and skates", "Red panda", "Reptile", "Rhinoceros",
    "Scorpion", "Sea lion", "Sea turtle", "Seahorse", "Shark", "Sheep",
    "Shellfish", "Shrimp", "Skunk", "Snail", "Snake", "Sparrow",
    "Spider", "Squid", "Squirrel", "Starfish", "Swan", "Tick",
    "Tiger", "Tortoise", "Turkey", "Turtle", "Whale", "Woodpecker",
    "Worm", "Zebra",
    "Cat furniture", "Dog bed", "Teddy bear",
}

# ──────────────────────────────────────────────────────────────────────
#  Master lookup functions
# ──────────────────────────────────────────────────────────────────────

def classify_detection(class_name: str) -> Tuple[str, Optional[str]]:
    """Return (category, sub_category) for an OIV7 class name."""
    clean = class_name.replace(" (⚠️ Uncertain match)", "").strip()

    if clean in _PERSON_ALL:
        sub = "body_part" if clean in _PERSON_BODY_PARTS else "full_person"
        return CATEGORY_PERSONS, sub

    if clean in _LICENSE_PLATE_ALL:
        return CATEGORY_LICENSE_PLATES, None

    if clean in _VEHICLE_ALL:
        return CATEGORY_VEHICLES, _VEHICLE_SUBTYPE.get(clean, "other_vehicle")

    if clean in _CLOTHING_ALL:
        return CATEGORY_CLOTHING, _CLOTHING_SUBTYPE.get(clean, None)

    if clean in _ANIMAL_ALL:
        return CATEGORY_ANIMALS, None

    return CATEGORY_OTHER, None


def get_demographic_estimate(class_name: str) -> Optional[str]:
    """Return 'male', 'female', or 'unclear' for person classes, or None."""
    clean = class_name.replace(" (⚠️ Uncertain match)", "").strip()
    return DEMOGRAPHIC_MAP.get(clean)


# ──────────────────────────────────────────────────────────────────────
#  Parent-child linking via bounding-box containment
# ──────────────────────────────────────────────────────────────────────

def _ioa(child_bbox: list, parent_bbox: list) -> float:
    """
    Intersection-over-Area of the child box.
    Returns what fraction of the child box is inside the parent box.
    """
    cx1, cy1, cx2, cy2 = child_bbox
    px1, py1, px2, py2 = parent_bbox

    ix1 = max(cx1, px1)
    iy1 = max(cy1, py1)
    ix2 = min(cx2, px2)
    iy2 = min(cy2, py2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    child_area = max((cx2 - cx1) * (cy2 - cy1), 1e-6)
    return inter / child_area


def link_parent_child(detections: List[Dict[str, Any]], ioa_threshold: float = 0.5) -> None:
    """
    Set parent_index and parent_type on detections in-place.
    - Clothing items inside a person box -> parent is that person
    - License plates inside a vehicle box -> parent is that vehicle
    - Body parts inside a person box -> parent is that person
    """
    person_indices = []
    vehicle_indices = []

    for i, det in enumerate(detections):
        cat = det.get("category")
        sub = det.get("sub_category")
        if cat == CATEGORY_PERSONS and sub == "full_person":
            person_indices.append(i)
        elif cat == CATEGORY_VEHICLES:
            vehicle_indices.append(i)

    for i, det in enumerate(detections):
        cat = det.get("category")
        sub = det.get("sub_category")

        # Clothing -> nearest enclosing person
        if cat == CATEGORY_CLOTHING:
            best_idx, best_ioa = None, 0.0
            for pi in person_indices:
                score = _ioa(det["bbox"], detections[pi]["bbox"])
                if score > best_ioa and score >= ioa_threshold:
                    best_ioa = score
                    best_idx = pi
            if best_idx is not None:
                det["parent_index"] = best_idx
                det["parent_type"] = "person"

        # License plate -> nearest enclosing vehicle
        elif cat == CATEGORY_LICENSE_PLATES:
            best_idx, best_ioa = None, 0.0
            for vi in vehicle_indices:
                score = _ioa(det["bbox"], detections[vi]["bbox"])
                if score > best_ioa and score >= ioa_threshold:
                    best_ioa = score
                    best_idx = vi
            if best_idx is not None:
                det["parent_index"] = best_idx
                det["parent_type"] = "vehicle"

        # Body parts -> nearest enclosing full person
        elif cat == CATEGORY_PERSONS and sub == "body_part":
            best_idx, best_ioa = None, 0.0
            for pi in person_indices:
                if pi == i:
                    continue
                score = _ioa(det["bbox"], detections[pi]["bbox"])
                if score > best_ioa and score >= ioa_threshold:
                    best_ioa = score
                    best_idx = pi
            if best_idx is not None:
                det["parent_index"] = best_idx
                det["parent_type"] = "person"


def enrich_detections(
    detections: List[Dict[str, Any]],
    enable_demographics: bool = False,
) -> Dict[str, Any]:
    """
    Enrich each detection with category, sub_category, parent_index,
    detection_id, and optional demographic_estimate. Returns summary dict.
    """
    for i, det in enumerate(detections):
        cat, sub = classify_detection(det.get("class_name", ""))
        det["category"] = cat
        det["sub_category"] = sub
        det["parent_index"] = None
        det["parent_type"] = None
        det["detection_id"] = f"det_{i}"

        # Demographics (only for person classes)
        if enable_demographics and cat == CATEGORY_PERSONS:
            pres = get_demographic_estimate(det.get("class_name", ""))
            if pres:
                det["demographic_estimate"] = {
                    "presentation": pres,
                    "source": "yolo_class",
                    "disclaimer": (
                        "Approximate visual estimate of apparent gender "
                        "presentation. May be inaccurate or biased."
                    ),
                }
            else:
                det["demographic_estimate"] = None
        else:
            det["demographic_estimate"] = None

    link_parent_child(detections)

    category_counts = {cat: 0 for cat in ALL_CATEGORIES}
    demographic_counts = {"male": 0, "female": 0, "unclear": 0}
    vehicle_subtypes = set()
    clothing_subtypes = set()

    for det in detections:
        cat = det.get("category", CATEGORY_OTHER)
        category_counts[cat] = category_counts.get(cat, 0) + 1

        if cat == CATEGORY_VEHICLES and det.get("sub_category"):
            vehicle_subtypes.add(det["sub_category"])
        if cat == CATEGORY_CLOTHING and det.get("sub_category"):
            clothing_subtypes.add(det["sub_category"])

        if enable_demographics and det.get("demographic_estimate"):
            pres = det["demographic_estimate"]["presentation"]
            demographic_counts[pres] = demographic_counts.get(pres, 0) + 1

    return {
        "category_counts": category_counts,
        "demographic_enabled": enable_demographics,
        "demographic_counts": demographic_counts if enable_demographics else None,
        "filter_config": {
            "available_vehicle_subtypes": sorted(list(vehicle_subtypes)),
            "available_clothing_subtypes": sorted(list(clothing_subtypes)),
        },
    }
