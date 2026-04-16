"""
Usage example for the IJ2 gear stat generator.

Loads the geardefinitionlist from the data/ folder and validates
against 4 known items with confirmed in-game stats.

IJ2-specific constants are imported from consts.py and passed to
the StatGenerator — the class itself is game-agnostic.
"""

import json
import os

from ij2_rng import StatGenerator
from ij2_rng.consts import (
    ATTRIBUTES_MAP,
    BASE_STAT_IDS,
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    SCALE_BASE,
    SCALE_BOOST_AT_MAX,
    SCALE_BOOST_BELOW_MAX,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "geardefinitionlist.json")


def main():
    # Load the geardefinitionlist (server's gear catalog)
    with open(DATA_PATH, "r") as f:
        raw = json.load(f)
    catalog_data = raw["data"]

    gen = StatGenerator(
        geardefinitionlist_data=catalog_data,
        lcg_multiplier=LCG_MULTIPLIER,
        lcg_increment=LCG_INCREMENT,
        attributes_map=ATTRIBUTES_MAP,
        base_stat_ids=BASE_STAT_IDS,
        scale_boost_at_max=SCALE_BOOST_AT_MAX,
        scale_boost_below_max=SCALE_BOOST_BELOW_MAX,
        scale_base=SCALE_BASE,
    )

    # Validated test cases: (name, item_index, seed, power, level, has_multi_assets, expected)
    test_cases = [
        (
            "AM_AccessoryEpic2",
            20831,
            52480,
            61,
            10,
            False,
            {"Strength": 32, "Ability": 98, "Health": 17},
        ),
        (
            "AM_AccessoryCommon21",
            20689,
            17590,
            51,
            10,
            False,  # procedural but single-asset group
            {"Ability": 37, "Defense": 41},
        ),
        (
            "BM_LegsRare15",
            2064,
            42526,
            5,
            2,
            True,  # procedural with multiple assets
            {"Defense": 14},
        ),
        (
            "BM_LegsRare17",
            2066,
            47454,
            85,
            10,
            True,  # procedural with multiple assets
            {"Strength": 36, "Ability": 46},
        ),
    ]

    all_pass = True
    for name, idx, seed, power, level, has_multi, expected in test_cases:
        item_def = gen.get_item(idx)
        result = gen.generate(seed, level, idx, has_multiple_assets=has_multi)

        match = all(
            result["stats"].get(k, 0) == v for k, v in expected.items()
        )
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False

        print(f"[{status}] {name} (seed={seed}, level={level})")
        print(f"  Item: {item_def.get('i')} | dL={item_def.get('dL')} | p={item_def.get('p', False)}")
        print(f"  Picks: {result['picks']}")
        print(f"  Scale: {result['scale_factor']:.6f}")
        print(f"  Stats: {result['stats']}")
        print(f"  Expected: {expected}")
        print()

    if all_pass:
        print("All test cases passed.")
    else:
        print("Some test cases FAILED.")


if __name__ == "__main__":
    main()
