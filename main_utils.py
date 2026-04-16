import json
import os
from typing import Optional, Union

try:
    from .ij2_rng import FastStatGenerator, StatGenerator
    from .ij2_rng.consts import (
        ATTRIBUTES_MAP,
        BASE_STAT_IDS,
        LCG_INCREMENT,
        LCG_MULTIPLIER,
        SCALE_BASE,
        SCALE_BOOST_AT_MAX,
        SCALE_BOOST_BELOW_MAX,
    )
except ImportError:
    from ij2_rng import FastStatGenerator, StatGenerator
    from ij2_rng.consts import (
        ATTRIBUTES_MAP,
        BASE_STAT_IDS,
        LCG_INCREMENT,
        LCG_MULTIPLIER,
        SCALE_BASE,
        SCALE_BOOST_AT_MAX,
        SCALE_BOOST_BELOW_MAX,
    )

def get_assets_for_item(hashmap: dict, item_index: int):
    """
    Look up the assets parameter for an item from the HashMap.

    Returns:
      - list[str] for group items (from mGroups)
      - str for direct items (single hash)
      - None for augments / items without visuals
    """
    entry = hashmap["mItems"][item_index]
    item_type = entry.get("type")

    if item_type == "group":
        group_id = entry["group"]
        group = hashmap["mGroups"][group_id]
        return group if group else None
    elif item_type == "direct":
        return entry.get("hash")
    return None


def build_generator(
    catalog_data: dict,
    mode="default",
    cache_dir: Optional[str] = None,
) -> Union[StatGenerator, FastStatGenerator]:
    cache_path = None
    float_lut_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "stat_fast_cache.pkl")
        float_lut_path = os.path.join(cache_dir, "lcg_float_lut.npy")

    if mode == "default":
        return StatGenerator(
            geardefinitionlist_data=catalog_data,
            lcg_multiplier=LCG_MULTIPLIER,
            lcg_increment=LCG_INCREMENT,
            attributes_map=ATTRIBUTES_MAP,
            base_stat_ids=BASE_STAT_IDS,
            scale_boost_at_max=SCALE_BOOST_AT_MAX,
            scale_boost_below_max=SCALE_BOOST_BELOW_MAX,
            scale_base=SCALE_BASE,
        )
    if mode == "fast":
        return FastStatGenerator(
            geardefinitionlist_data=catalog_data,
            lcg_multiplier=LCG_MULTIPLIER,
            lcg_increment=LCG_INCREMENT,
            attributes_map=ATTRIBUTES_MAP,
            base_stat_ids=BASE_STAT_IDS,
            scale_boost_at_max=SCALE_BOOST_AT_MAX,
            scale_boost_below_max=SCALE_BOOST_BELOW_MAX,
            scale_base=SCALE_BASE,
            cache_path=cache_path,
            float_lut_path=float_lut_path,
        )
    raise ValueError(f"Unknown mode: {mode}")


def load_hashmap(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_catalog_payload(catalog_path) -> dict:
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_catalog_data(catalog_path) -> dict:
    raw = load_catalog_payload(catalog_path)
    return raw["data"]
