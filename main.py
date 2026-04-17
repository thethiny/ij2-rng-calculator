"""
Usage example for the IJ2 gear stat generator.

Loads a canonical IJ2 catalog JSON from the data/ folder and validates against
4 known items with confirmed in-game stats. The bundled
``geardefinitionlist.json`` is compatible with the offline ``TweakVars``
decode path used elsewhere in the repo.

IJ2-specific constants are imported from consts.py and passed to
the StatGenerator — the class itself is game-agnostic.
"""

import os

from .main_utils import (
    build_generator,
    get_assets_for_item,
    load_catalog_data,
    load_hashmap,
)

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "geardefinitionlist.json")
HASHMAP_PATH = os.path.join(DATA_DIR, "HashMap.json")

# Validated test cases: (name, item_index, seed, power, level, expected)
VALIDATED_TEST_CASES = [
    (
        "AM_AccessoryEpic2",
        20831,
        52480,
        61,
        10,
        {"Strength": 32, "Ability": 98, "Health": 17},
    ),
    (
        "AM_AccessoryCommon21",
        20689,
        17590,
        51,
        10,
        {"Ability": 37, "Defense": 41},
    ),
    (
        "BM_LegsRare15",
        2064,
        42526,
        5,
        2,
        {"Defense": 14},
    ),
    (
        "BM_LegsRare17",
        2066,
        47454,
        85,
        10,
        {"Strength": 36, "Ability": 46},
    ),
]


def main():
    catalog_data = load_catalog_data(CATALOG_PATH)
    gen = build_generator(catalog_data)

    hashmap = load_hashmap(HASHMAP_PATH)
    if hashmap:
        print(f"HashMap loaded ({len(hashmap['mItems'])} items, {len(hashmap['mGroups'])} groups)")
    else:
        print(f"HashMap not found at {HASHMAP_PATH}, using manual asset flags")

    all_pass = True
    for name, idx, seed, power, level, expected in VALIDATED_TEST_CASES:
        item_def = gen.get_item(idx)

        # Resolve assets from HashMap (or fallback to None)
        assets = get_assets_for_item(hashmap, idx) if hashmap else None

        result = gen.generate(seed, level, idx, assets=assets)

        match = all(
            result["stats"].get(k, 0) == v for k, v in expected.items()
        )
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False

        assets_desc = (
            f"group[{len(assets)}]" if isinstance(assets, list)
            else f"direct" if isinstance(assets, str)
            else "none"
        )
        print(f"[{status}] {name} (seed={seed}, level={level}, assets={assets_desc})")
        print(f"  Item: {item_def.get('i')} | dL={item_def.get('dL')} | p={item_def.get('p', False)}")
        print(f"  Picks: {result['picks']}")
        print(f"  Scale: {result['scale_factor']:.6f}")
        print(f"  Stats: {result['stats']}")
        if result['visual_hash']:
            print(f"  Visual: {result['visual_hash']}")
        print(f"  Expected: {expected}")
        print()

    if all_pass:
        print("All test cases passed.")
    else:
        print("Some test cases FAILED.")


if __name__ == "__main__":
    main()
