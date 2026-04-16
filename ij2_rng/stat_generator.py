"""
Gear stat generator — game-agnostic.

All game-specific values (LCG constants, attribute mappings, scale factors)
are passed to the constructor. The class itself has no knowledge of IJ2/UE3
specifics beyond the algorithm structure.

Key code references (ij2_decompile/):
  - CacheGeneratedData (entry point)
  - CalculateScaleFactor
  - AttributeList::Add (non-power path)
  - AttributeList::Add (selector processing)
  - SerializeFrom (int value generation)
  - GetRandomAsset (LCG advance for group items)
  - Defense::Data::Multiply (per-pick scaling)
"""

import ctypes
from typing import List, Optional, Union

from .consts import (
    ATTRIBUTES_MAP,
    BASE_STAT_IDS,
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    SCALE_BASE,
    SCALE_BOOST_AT_MAX,
    SCALE_BOOST_BELOW_MAX,
)
from .lcg import LCGStream, f32, lcg_to_int_range

# Type alias for the assets parameter.
# - list[str]: asset group hashes (from HashMap mGroups) — LCG picks one if len>1
# - str:       single known hash — returned as-is, no LCG advance
# - None:      no asset info — no LCG advance, no visual hash returned
AssetsParam = Optional[Union[List[str], str]]


class StatGenerator:
    """
    Generates gear stats from a RandomSeed, ItemLevel, and item definition.

    The geardefinitionlist data must be loaded from the server's object store
    (endpoint: /objects/geardefinitionlist/{id}) and passed to this class.

    The data contains:
      - items[]: array where index == ItemIndex, with selectors (a.s), level
        bounds (l, h, dL), and procedural flag (p).
      - attributeSets[]: array of attribute set definitions referenced by
        selectors, containing the stat parameter ranges.
    """

    def __init__(
        self,
        geardefinitionlist_data: dict,
        lcg_multiplier: int = LCG_MULTIPLIER,
        lcg_increment: int = LCG_INCREMENT,
        attributes_map: dict = ATTRIBUTES_MAP,
        base_stat_ids: set = BASE_STAT_IDS,
        scale_boost_at_max: float = SCALE_BOOST_AT_MAX,
        scale_boost_below_max: float = SCALE_BOOST_BELOW_MAX,
        scale_base: float = SCALE_BASE,
    ):
        """
        Args:
            geardefinitionlist_data: The 'data' dict from the geardefinitionlist
                JSON object. Must contain 'items' and 'attributeSets' keys.
            lcg_multiplier: LCG multiplier (a in: next = a*seed + c).
            lcg_increment: LCG increment (c in: next = a*seed + c).
            attributes_map: Mapping of {attribute_id: display_name} for stats.
            base_stat_ids: Set of attribute IDs that are base stats (scaled by level).
            scale_boost_at_max: Boost multiplier when level == maxLevel.
            scale_boost_below_max: Boost multiplier when level < maxLevel.
            scale_base: Additive base after boost multiplication.
        """
        self.items = geardefinitionlist_data["items"]
        self.attribute_sets = geardefinitionlist_data["attributeSets"]

        self._lcg_multiplier = lcg_multiplier
        self._lcg_increment = lcg_increment
        self._attributes_map = attributes_map
        self._base_stat_ids = base_stat_ids
        self._scale_boost_at_max = scale_boost_at_max
        self._scale_boost_below_max = scale_boost_below_max
        self._scale_base = scale_base

        # Reverse lookup: stat name -> attribute id (for _apply_scale_per_pick)
        self._name_to_id = {v: k for k, v in attributes_map.items()}

    def _make_rng(self, seed: int) -> LCGStream:
        return LCGStream(seed, self._lcg_multiplier, self._lcg_increment)

    def get_item(self, item_index: int) -> dict:
        return self.items[item_index]

    def generate(
        self,
        seed: int,
        item_level: int,
        item_index: int,
        assets: AssetsParam = None,
    ) -> dict:
        """
        Generate gear stats for an item.

        This replicates the full CacheGeneratedData -> AttributeList::Add ->
        CalculateScaleFactor -> Multiply pipeline from the game binary.

        Args:
            seed: RandomSeed from the inventory item (uint16, 0-65535).
            item_level: ItemLevel from the inventory item (1-30).
            item_index: ItemIndex -- position in the geardefinitionlist items array.
            assets: Visual asset information. Controls GetRandomAsset behavior:
                - list[str]: Asset group hashes (e.g. from HashMap mGroups).
                  If len > 1, LCG advances once to pick the visual variant.
                  If len == 1, no advance; returns that single hash.
                - str: A single known hash. No LCG advance; returned as-is.
                - None: No asset info. No LCG advance, no visual hash in result.

        Returns:
            Dict with keys:
              - picks: list of (stat_name, raw_value) per pick
              - scale_factor: the CalculateScaleFactor result
              - stats: dict of {stat_name: final_displayed_value}
              - visual_hash: str or None — the selected visual asset hash
        """
        item_def = self.items[item_index]
        rng = self._make_rng(seed & 0xFFFF)

        # GetRandomAsset: select visual variant, may advance LCG
        visual_hash = self._get_random_asset(rng, assets)

        # Process all selectors (non-power path: AttributeList::Add)
        picks = self._process_selectors(rng, item_def)

        # CalculateScaleFactor
        scale = self._calculate_scale_factor(
            item_level,
            min_level=item_def.get("l", 1),
            max_level=item_def.get("h", 30),
            design_level=item_def.get("dL", 20),
        )

        # Apply Multiply per-pick, then sum
        stats = self._apply_scale_per_pick(picks, scale)

        return {
            "picks": picks,
            "scale_factor": scale,
            "stats": stats,
            "visual_hash": visual_hash,
        }

    def find_best_seed(
        self,
        item_index: int,
        item_level: int,
        stat_name: str,
        assets: AssetsParam = None,
        seed_range: int = 0xFFFF,
    ) -> dict:
        """
        Brute-force search for the seed that maximizes a given stat.

        Args:
            item_index: ItemIndex in the catalog.
            item_level: ItemLevel to evaluate at.
            stat_name: One of the stat display names from attributes_map.
            assets: Visual asset info (see generate()).
            seed_range: Maximum seed to try (default 0xFFFF = 65535).

        Returns:
            Dict with best_seed, best_value, and full stats at that seed.
        """
        best_seed = 0
        best_value = 0
        best_result = None

        for s in range(seed_range + 1):
            result = self.generate(s, item_level, item_index, assets)
            val = result["stats"].get(stat_name, 0)
            if val > best_value:
                best_value = val
                best_seed = s
                best_result = result

        return {
            "best_seed": best_seed,
            "best_value": best_value,
            "stats": best_result["stats"] if best_result else {},
            "picks": best_result["picks"] if best_result else [],
            "scale_factor": best_result["scale_factor"] if best_result else 0,
        }

    def find_best_seed_total(
        self,
        item_index: int,
        item_level: int,
        assets: AssetsParam = None,
        seed_range: int = 0xFFFF,
        stat_weights: Optional[dict] = None,
    ) -> dict:
        """
        Find the seed that maximizes total base stats (or weighted sum).

        Args:
            item_index: ItemIndex in the catalog.
            item_level: ItemLevel to evaluate at.
            assets: Visual asset info (see generate()).
            seed_range: Maximum seed to try.
            stat_weights: Optional dict of {stat_name: weight} for weighted sum.
                          Defaults to equal weights (1.0 each).

        Returns:
            Dict with best_seed, best_total, and full stats at that seed.
        """
        weights = stat_weights or {n: 1.0 for n in self._attributes_map.values()}
        best_seed = 0
        best_total = 0
        best_result = None

        for s in range(seed_range + 1):
            result = self.generate(s, item_level, item_index, assets)
            total = sum(
                result["stats"].get(name, 0) * w for name, w in weights.items()
            )
            if total > best_total:
                best_total = total
                best_seed = s
                best_result = result

        return {
            "best_seed": best_seed,
            "best_total": best_total,
            "stats": best_result["stats"] if best_result else {},
            "picks": best_result["picks"] if best_result else [],
            "scale_factor": best_result["scale_factor"] if best_result else 0,
        }

    # ----------------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------------

    @staticmethod
    def _get_random_asset(rng: LCGStream, assets: AssetsParam) -> Optional[str]:
        """
        Replicate GetRandomAsset: select a visual variant from the asset group.

        - list with >1 entries: advance LCG, pick by index -> return selected hash
        - list with 1 entry:   no advance -> return that hash
        - str:                  no advance -> return it directly (caller knows the visual)
        - None:                 no advance -> return None
        """
        if assets is None:
            return None

        if isinstance(assets, str):
            return assets

        # list of hashes
        count = len(assets)
        if count == 0:
            return None
        if count == 1:
            return assets[0]

        # Multiple assets: advance LCG to pick one
        seed = rng.advance()
        idx = lcg_to_int_range(seed, 0, count - 1)
        return assets[idx]

    def _process_selectors(self, rng: LCGStream, item_def: dict) -> list:
        """
        Process all attribute selectors for an item.

        Implements the non-power path of AttributeList::Add.
        The item's a.s array contains selectors, each referencing an attributeSet
        by index (s), with a pick count (ct) and chance (ch).

        Returns list of (stat_name, raw_value) tuples.
        """
        picks = []
        selectors = item_def.get("a", {}).get("s", [])

        for selector in selectors:
            ct = selector["ct"]
            set_index = selector["s"]
            chance = selector["ch"]

            # Chance check: advance LCG, compare float
            rand = rng.next_float()
            threshold = ctypes.c_float(chance / 100.0).value
            if threshold < rand:
                continue

            attr_set = self.attribute_sets[set_index]
            direct_attrs = attr_set.get("a", {}).get("a", [])
            sub_selectors = attr_set.get("a", {}).get("s", [])
            direct_count = len(direct_attrs)
            total_count = direct_count + len(sub_selectors)

            if total_count <= 0:
                continue

            step = ctypes.c_float(1.0 / float(total_count)).value

            # Pick ct attributes from the set
            for _ in range(ct):
                rand = rng.next_float()
                idx = 0
                r = rand
                while step < r and idx < total_count - 1:
                    idx += 1
                    r = ctypes.c_float(r - step).value

                if idx < direct_count:
                    attr = direct_attrs[idx]
                    attr_picks = self._create_attribute(rng, attr)
                    picks.extend(attr_picks)

        return picks

    def _create_attribute(self, rng: LCGStream, attr_def: dict) -> list:
        """
        Create an attribute from its definition by reading parameters.

        Implements the Create -> SerializeFrom chain for base stats.

        Returns list of (stat_name, raw_value) tuples (usually 1 entry).
        """
        attr_id = attr_def["i"]
        attr_name = self._attributes_map.get(attr_id, f"attr_{attr_id}")
        picks = []

        for param in attr_def.get("p", []):
            param_id = param["id"]

            if param_id == "value" and "i" in param:
                lo = param["i"]["l"]
                hi = param["i"]["h"]
                if lo < hi:
                    val = rng.next_int_range(lo, hi)
                    picks.append((attr_name, val))

            elif param_id in ("base", "amount", "damage") and "f" in param:
                lo_f = param["f"]["l"]
                hi_f = param["f"]["h"]
                if lo_f < hi_f:
                    rng.advance()

            elif param_id == "set" and "i" in param:
                lo = param["i"]["l"]
                hi = param["i"]["h"]
                if lo < hi:
                    rng.next_int_range(lo, hi)

        return picks

    def _calculate_scale_factor(
        self,
        level: int,
        min_level: int,
        max_level: int,
        design_level: int,
    ) -> float:
        """
        Calculate the level-based scale factor.

        Source: EquipmentSaveData::CalculateScaleFactor

        All arithmetic uses float32 to match the game binary exactly.
        """
        effective = min(max(level, min_level), max_level)

        if effective == max_level:
            boost = self._scale_boost_at_max
        else:
            boost = self._scale_boost_below_max

        if design_level - 1 > 0:
            ratio = f32(
                f32(float(design_level - effective)) / f32(float(design_level - 1))
            )
            base = f32(1.0 - ratio)
            return f32(f32(base * f32(boost)) + self._scale_base)
        return 1.0

    def _apply_scale_per_pick(self, picks: list, scale_factor: float) -> dict:
        """
        Apply the scale factor to each pick independently, then sum by stat.

        Source: Defense::Data::Multiply — value = max(1, int(float(value) * scaleFactor))
        """
        stats = {}
        for stat_name, raw_value in picks:
            attr_id = self._name_to_id.get(stat_name)
            if attr_id is not None and attr_id in self._base_stat_ids:
                scaled = max(
                    1,
                    int(f32(f32(float(raw_value)) * f32(scale_factor))),
                )
            else:
                scaled = raw_value
            stats[stat_name] = stats.get(stat_name, 0) + scaled
        return stats
