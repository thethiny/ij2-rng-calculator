"""
Fast stat generator with compiled item metadata.

This keeps the original algorithm and output shape, but replaces repeated
JSON/dict parsing with precompiled immutable tuples that can be cached to disk.
"""

import os
import pickle
from typing import Any, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    from numba import njit
except ImportError:
    njit = None

from .consts import (
    ATTRIBUTES_MAP,
    BASE_STAT_IDS,
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    SCALE_BASE,
    SCALE_BOOST_AT_MAX,
    SCALE_BOOST_BELOW_MAX,
)
from .lcg import LCGStream, f32, lcg_to_float, lcg_to_int_range
from .seed_archive import IJ2SeedArchive

AssetsParam = Optional[Union[List[str], str]]

_CACHE_VERSION = 2
_OP_ADVANCE = 1
_OP_INT_RANGE = 2
_UINT32_MASK = 0xFFFFFFFF
_INT31_MASK = 0x7FFFFFFF
_ATTRIBUTE_NAMES = ("Health", "Defense", "Strength", "Ability")
_FLOAT_LUT = None
_FLOAT_LUT_PATH = os.path.join(os.path.dirname(__file__), "lcg_float_lut.npy")
_NUMBA_ENABLED = np is not None and njit is not None


def _build_float_lut():
    bits = np.arange(1 << 23, dtype=np.uint32)
    bits |= np.uint32(0x3F800000)
    return bits.view(np.float32) - np.float32(1.0)


def _get_float_lut():
    global _FLOAT_LUT
    if _FLOAT_LUT is not None:
        return _FLOAT_LUT

    if np is None:
        raise RuntimeError("NumPy is required to build the float lookup table")

    if os.path.exists(_FLOAT_LUT_PATH):
        _FLOAT_LUT = np.load(_FLOAT_LUT_PATH, mmap_mode="r")
        return _FLOAT_LUT

    lut = _build_float_lut()
    temp_path = f"{_FLOAT_LUT_PATH}.{os.getpid()}.tmp.npy"
    np.save(temp_path, lut)
    os.replace(temp_path, _FLOAT_LUT_PATH)
    _FLOAT_LUT = np.load(_FLOAT_LUT_PATH, mmap_mode="r")
    return _FLOAT_LUT


def _flatten_numeric_item(compiled_item: Tuple) -> Tuple["np.ndarray", ...]:
    selector_cts = []
    selector_thresholds = []
    selector_direct_counts = []
    selector_total_counts = []
    selector_steps = []
    selector_attr_starts = []
    attr_stat_slots = []
    attr_value_lo = []
    attr_value_hi = []
    attr_advance_counts = []

    for ct, threshold, compiled_set in compiled_item[3]:
        direct_attrs, direct_count, total_count, step = compiled_set
        selector_cts.append(ct)
        selector_thresholds.append(threshold)
        selector_direct_counts.append(direct_count)
        selector_total_counts.append(total_count)
        selector_steps.append(step)
        selector_attr_starts.append(len(attr_stat_slots))

        for _, _, stat_slot, value_lo, value_hi, extra_ops in direct_attrs:
            attr_stat_slots.append(stat_slot)
            attr_value_lo.append(value_lo)
            attr_value_hi.append(value_hi)
            attr_advance_counts.append(len(extra_ops))

    return (
        np.asarray(selector_cts, dtype=np.int32),
        np.asarray(selector_thresholds, dtype=np.float32),
        np.asarray(selector_direct_counts, dtype=np.int32),
        np.asarray(selector_total_counts, dtype=np.int32),
        np.asarray(selector_steps, dtype=np.float32),
        np.asarray(selector_attr_starts, dtype=np.int32),
        np.asarray(attr_stat_slots, dtype=np.int32),
        np.asarray(attr_value_lo, dtype=np.int32),
        np.asarray(attr_value_hi, dtype=np.int32),
        np.asarray(attr_advance_counts, dtype=np.int32),
    )


if _NUMBA_ENABLED:

    @njit(cache=True, nogil=True)
    def _find_best_stats_numba(
        selector_cts,
        selector_thresholds,
        selector_direct_counts,
        selector_total_counts,
        selector_steps,
        selector_attr_starts,
        attr_stat_slots,
        attr_value_lo,
        attr_value_hi,
        attr_advance_counts,
        scale_lookup,
        advance_for_assets,
        float_lut,
        multiplier,
        increment,
    ):
        best_values = np.full(4, -1, dtype=np.int32)
        best_totals = np.full(4, -1, dtype=np.int32)
        best_seeds = np.zeros(4, dtype=np.int32)
        best_stats = np.zeros((4, 4), dtype=np.int32)
        best_total_value = -1
        best_total_seed = 0
        best_total_stats = np.zeros(4, dtype=np.int32)

        mask = np.int64(_UINT32_MASK)
        int31_mask = np.int64(_INT31_MASK)
        mantissa_mask = np.int64(0x7FFFFF)

        for seed in range(0x10000):
            state = np.int64(seed)
            if advance_for_assets:
                state = (multiplier * state + increment) & mask

            stats0 = 0
            stats1 = 0
            stats2 = 0
            stats3 = 0

            for selector_index in range(selector_cts.shape[0]):
                state = (multiplier * state + increment) & mask
                if selector_thresholds[selector_index] < float_lut[state & mantissa_mask]:
                    continue

                direct_count = selector_direct_counts[selector_index]
                total_count = selector_total_counts[selector_index]
                if total_count <= 0:
                    continue

                last_index = total_count - 1
                step = selector_steps[selector_index]
                attr_start = selector_attr_starts[selector_index]

                for _ in range(selector_cts[selector_index]):
                    state = (multiplier * state + increment) & mask
                    r = float_lut[state & mantissa_mask]
                    idx = 0
                    while step < r and idx < last_index:
                        idx += 1
                        r = np.float32(r - step)

                    if idx >= direct_count:
                        continue

                    attr_index = attr_start + idx
                    stat_slot = attr_stat_slots[attr_index]
                    value_lo = attr_value_lo[attr_index]
                    value_hi = attr_value_hi[attr_index]

                    if value_lo < value_hi and stat_slot >= 0:
                        state = (multiplier * state + increment) & mask
                        raw_value = value_lo + int((state & int31_mask) % (value_hi - value_lo + 1))
                        scaled_value = scale_lookup[raw_value]

                        if stat_slot == 0:
                            stats0 += scaled_value
                        elif stat_slot == 1:
                            stats1 += scaled_value
                        elif stat_slot == 2:
                            stats2 += scaled_value
                        elif stat_slot == 3:
                            stats3 += scaled_value

                    for _ in range(attr_advance_counts[attr_index]):
                        state = (multiplier * state + increment) & mask

            total = stats0 + stats1 + stats2 + stats3

            if stats0 > best_values[0] or (stats0 == best_values[0] and total > best_totals[0]):
                best_values[0] = stats0
                best_totals[0] = total
                best_seeds[0] = seed
                best_stats[0, 0] = stats0
                best_stats[0, 1] = stats1
                best_stats[0, 2] = stats2
                best_stats[0, 3] = stats3

            if stats1 > best_values[1] or (stats1 == best_values[1] and total > best_totals[1]):
                best_values[1] = stats1
                best_totals[1] = total
                best_seeds[1] = seed
                best_stats[1, 0] = stats0
                best_stats[1, 1] = stats1
                best_stats[1, 2] = stats2
                best_stats[1, 3] = stats3

            if stats2 > best_values[2] or (stats2 == best_values[2] and total > best_totals[2]):
                best_values[2] = stats2
                best_totals[2] = total
                best_seeds[2] = seed
                best_stats[2, 0] = stats0
                best_stats[2, 1] = stats1
                best_stats[2, 2] = stats2
                best_stats[2, 3] = stats3

            if stats3 > best_values[3] or (stats3 == best_values[3] and total > best_totals[3]):
                best_values[3] = stats3
                best_totals[3] = total
                best_seeds[3] = seed
                best_stats[3, 0] = stats0
                best_stats[3, 1] = stats1
                best_stats[3, 2] = stats2
                best_stats[3, 3] = stats3

            if total > best_total_value:
                best_total_value = total
                best_total_seed = seed
                best_total_stats[0] = stats0
                best_total_stats[1] = stats1
                best_total_stats[2] = stats2
                best_total_stats[3] = stats3

        return (
            best_values,
            best_totals,
            best_seeds,
            best_stats,
            best_total_value,
            best_total_seed,
            best_total_stats,
        )

    @njit(cache=True, nogil=True)
    def _build_seed_archive_numba(
        selector_cts,
        selector_thresholds,
        selector_direct_counts,
        selector_total_counts,
        selector_steps,
        selector_attr_starts,
        attr_stat_slots,
        attr_value_lo,
        attr_value_hi,
        attr_advance_counts,
        scale_lookup,
        asset_count,
        float_lut,
        multiplier,
        increment,
        max_pick_count,
    ):
        seed_count = 0x10000
        visual_indices = np.zeros(seed_count, dtype=np.uint16)
        pick_counts = np.zeros(seed_count, dtype=np.uint8)
        stat_ids = np.zeros((seed_count, max_pick_count), dtype=np.uint8)
        raw_values = np.zeros((seed_count, max_pick_count), dtype=np.uint16)

        best_values = np.full(4, -1, dtype=np.int32)
        best_totals = np.full(4, -1, dtype=np.int32)
        best_seeds = np.zeros(4, dtype=np.int32)
        best_stats = np.zeros((4, 4), dtype=np.int32)
        best_visual_indices = np.zeros(4, dtype=np.int32)
        best_total_value = -1
        best_total_seed = 0
        best_total_stats = np.zeros(4, dtype=np.int32)
        best_total_visual_index = 0

        mask = np.int64(_UINT32_MASK)
        int31_mask = np.int64(_INT31_MASK)
        mantissa_mask = np.int64(0x7FFFFF)

        for seed in range(seed_count):
            state = np.int64(seed)
            visual_index = 0
            if asset_count > 1:
                state = (multiplier * state + increment) & mask
                visual_index = int((state & int31_mask) % asset_count)

            stats0 = 0
            stats1 = 0
            stats2 = 0
            stats3 = 0
            pick_write = 0

            for selector_index in range(selector_cts.shape[0]):
                state = (multiplier * state + increment) & mask
                if selector_thresholds[selector_index] < float_lut[state & mantissa_mask]:
                    continue

                direct_count = selector_direct_counts[selector_index]
                total_count = selector_total_counts[selector_index]
                if total_count <= 0:
                    continue

                last_index = total_count - 1
                step = selector_steps[selector_index]
                attr_start = selector_attr_starts[selector_index]

                for _ in range(selector_cts[selector_index]):
                    state = (multiplier * state + increment) & mask
                    r = float_lut[state & mantissa_mask]
                    idx = 0
                    while step < r and idx < last_index:
                        idx += 1
                        r = np.float32(r - step)

                    if idx >= direct_count:
                        continue

                    attr_index = attr_start + idx
                    stat_slot = attr_stat_slots[attr_index]
                    value_lo = attr_value_lo[attr_index]
                    value_hi = attr_value_hi[attr_index]

                    if value_lo < value_hi and stat_slot >= 0:
                        state = (multiplier * state + increment) & mask
                        raw_value = value_lo + int((state & int31_mask) % (value_hi - value_lo + 1))
                        if pick_write < max_pick_count:
                            stat_ids[seed, pick_write] = stat_slot
                            raw_values[seed, pick_write] = raw_value
                            pick_write += 1

                        scaled_value = scale_lookup[raw_value]
                        if stat_slot == 0:
                            stats0 += scaled_value
                        elif stat_slot == 1:
                            stats1 += scaled_value
                        elif stat_slot == 2:
                            stats2 += scaled_value
                        elif stat_slot == 3:
                            stats3 += scaled_value

                    for _ in range(attr_advance_counts[attr_index]):
                        state = (multiplier * state + increment) & mask

            visual_indices[seed] = visual_index
            pick_counts[seed] = pick_write
            total = stats0 + stats1 + stats2 + stats3

            if stats0 > best_values[0] or (stats0 == best_values[0] and total > best_totals[0]):
                best_values[0] = stats0
                best_totals[0] = total
                best_seeds[0] = seed
                best_stats[0, 0] = stats0
                best_stats[0, 1] = stats1
                best_stats[0, 2] = stats2
                best_stats[0, 3] = stats3
                best_visual_indices[0] = visual_index

            if stats1 > best_values[1] or (stats1 == best_values[1] and total > best_totals[1]):
                best_values[1] = stats1
                best_totals[1] = total
                best_seeds[1] = seed
                best_stats[1, 0] = stats0
                best_stats[1, 1] = stats1
                best_stats[1, 2] = stats2
                best_stats[1, 3] = stats3
                best_visual_indices[1] = visual_index

            if stats2 > best_values[2] or (stats2 == best_values[2] and total > best_totals[2]):
                best_values[2] = stats2
                best_totals[2] = total
                best_seeds[2] = seed
                best_stats[2, 0] = stats0
                best_stats[2, 1] = stats1
                best_stats[2, 2] = stats2
                best_stats[2, 3] = stats3
                best_visual_indices[2] = visual_index

            if stats3 > best_values[3] or (stats3 == best_values[3] and total > best_totals[3]):
                best_values[3] = stats3
                best_totals[3] = total
                best_seeds[3] = seed
                best_stats[3, 0] = stats0
                best_stats[3, 1] = stats1
                best_stats[3, 2] = stats2
                best_stats[3, 3] = stats3
                best_visual_indices[3] = visual_index

            if total > best_total_value:
                best_total_value = total
                best_total_seed = seed
                best_total_stats[0] = stats0
                best_total_stats[1] = stats1
                best_total_stats[2] = stats2
                best_total_stats[3] = stats3
                best_total_visual_index = visual_index

        return (
            visual_indices,
            pick_counts,
            stat_ids,
            raw_values,
            best_values,
            best_totals,
            best_seeds,
            best_stats,
            best_visual_indices,
            best_total_value,
            best_total_seed,
            best_total_stats,
            best_total_visual_index,
        )


class FastStatGenerator:
    """
    Stat generator with precompiled selector / attribute metadata.

    The compile step is cached to disk so subsequent starts can reuse the
    compiled representation instead of rebuilding it from raw catalog JSON.
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
        cache_path: Optional[str] = None,
    ):
        self.items = geardefinitionlist_data["items"]
        self.attribute_sets = geardefinitionlist_data["attributeSets"]

        self._lcg_multiplier = lcg_multiplier
        self._lcg_increment = lcg_increment
        self._attributes_map = attributes_map
        self._base_stat_ids = base_stat_ids
        self._scale_boost_at_max = scale_boost_at_max
        self._scale_boost_below_max = scale_boost_below_max
        self._scale_base = scale_base

        self._cache_path = cache_path or os.path.join(
            os.path.dirname(__file__),
            "stat_fast_cache.pkl",
        )
        self._compiled_items = self._load_or_build_cache()
        self._scale_cache: dict[tuple[int, int], tuple[float, Tuple[int, ...], Optional["np.ndarray"]]] = {}
        self._numeric_item_cache: dict[int, Tuple["np.ndarray", ...]] = {}
        self._use_numba = _NUMBA_ENABLED

    def _make_rng(self, seed: int) -> LCGStream:
        return LCGStream(seed, self._lcg_multiplier, self._lcg_increment)

    def get_item(self, item_index: int) -> dict:
        return self.items[item_index]

    @staticmethod
    def _empty_best_stats_result() -> dict[str, Any]:
        empty_stats = {name: 0 for name in _ATTRIBUTE_NAMES}
        return {
            "best_by_stat": [
                {
                    "stat": stat_name,
                    "seed": 0,
                    "value": 0,
                    "total_stats": 0,
                    "stats": empty_stats.copy(),
                }
                for stat_name in _ATTRIBUTE_NAMES
            ],
            "best_total": {
                "seed": 0,
                "value": 0,
                "stats": empty_stats.copy(),
            },
        }

    def generate(
        self,
        seed: int,
        item_level: int,
        item_index: int,
        assets: AssetsParam = None,
    ) -> dict:
        compiled_item = self._compiled_items[item_index]
        rng = self._make_rng(seed & 0xFFFF)

        visual_hash = self._get_random_asset(rng, assets)
        picks = self._process_selectors_compiled(rng, compiled_item[3])
        scale, _, _ = self._get_scale_data(item_index, item_level)
        stats = self._apply_scale_per_pick(picks, scale)

        return {
            "picks": [(attr_name, raw_value) for _, attr_name, raw_value in picks],
            "scale_factor": scale,
            "stats": stats,
            "visual_hash": visual_hash,
        }

    def find_best_stats(
        self,
        item_index: int,
        item_level: int,
        assets: AssetsParam = None,
    ) -> dict[str, Any]:
        if not self._compiled_items[item_index][3]:
            return self._empty_best_stats_result()
        if self._use_numba:
            return self._find_best_stats_numba(item_index, item_level, assets)
        return self._find_best_stats_python(item_index, item_level, assets)

    def build_seed_archive_item_bytes(
        self,
        item_index: int,
        item_level: int,
        assets: AssetsParam = None,
    ) -> bytes:
        if self._use_numba:
            return self._build_seed_archive_item_bytes_numba(item_index, item_level, assets)
        return self._build_seed_archive_item_bytes_python(item_index, item_level, assets)

    def _find_best_stats_python(
        self,
        item_index: int,
        item_level: int,
        assets: AssetsParam = None,
    ) -> dict[str, Any]:
        compiled_item = self._compiled_items[item_index]
        compiled_selectors = compiled_item[3]
        _, scale_lookup, _ = self._get_scale_data(item_index, item_level)
        advance_for_assets = isinstance(assets, list) and len(assets) > 1

        multiplier = self._lcg_multiplier
        increment = self._lcg_increment

        best_values = [-1, -1, -1, -1]
        best_totals = [-1, -1, -1, -1]
        best_seeds = [0, 0, 0, 0]
        best_stats = [[0, 0, 0, 0] for _ in range(4)]
        best_total_value = -1
        best_total_seed = 0
        best_total_stats = [0, 0, 0, 0]

        for seed in range(0x10000):
            state = seed
            if advance_for_assets:
                state = (multiplier * state + increment) & _UINT32_MASK

            stats = [0, 0, 0, 0]

            for ct, threshold, compiled_set in compiled_selectors:
                state = (multiplier * state + increment) & _UINT32_MASK
                if threshold < self._lcg_state_to_float(state):
                    continue

                direct_attrs, direct_count, total_count, step = compiled_set
                if total_count <= 0:
                    continue

                last_index = total_count - 1

                for _ in range(ct):
                    state = (multiplier * state + increment) & _UINT32_MASK
                    rand = self._lcg_state_to_float(state)
                    idx = 0
                    r = rand
                    while step < r and idx < last_index:
                        idx += 1
                        r = f32(r - step)

                    if idx >= direct_count:
                        continue

                    _, _, stat_slot, value_lo, value_hi, extra_ops = direct_attrs[idx]

                    if value_lo < value_hi and stat_slot >= 0:
                        state = (multiplier * state + increment) & _UINT32_MASK
                        raw_value = value_lo + ((state & _INT31_MASK) % (value_hi - value_lo + 1))
                        stats[stat_slot] += scale_lookup[raw_value]

                    for op_kind, lo, hi in extra_ops:
                        if op_kind == _OP_ADVANCE:
                            state = (multiplier * state + increment) & _UINT32_MASK
                        elif lo < hi:
                            state = (multiplier * state + increment) & _UINT32_MASK

            total = stats[0] + stats[1] + stats[2] + stats[3]

            for stat_index, stat_value in enumerate(stats):
                if stat_value > best_values[stat_index] or (
                    stat_value == best_values[stat_index] and total > best_totals[stat_index]
                ):
                    best_values[stat_index] = stat_value
                    best_totals[stat_index] = total
                    best_seeds[stat_index] = seed
                    best_stats[stat_index] = stats.copy()

            if total > best_total_value:
                best_total_value = total
                best_total_seed = seed
                best_total_stats = stats.copy()

        return {
            "best_by_stat": [
                {
                    "stat": stat_name,
                    "seed": best_seeds[index],
                    "value": best_values[index],
                    "total_stats": best_totals[index],
                    "stats": {
                        name: best_stats[index][slot]
                        for slot, name in enumerate(_ATTRIBUTE_NAMES)
                    },
                }
                for index, stat_name in enumerate(_ATTRIBUTE_NAMES)
            ],
            "best_total": {
                "seed": best_total_seed,
                "value": best_total_value,
                "stats": {
                    name: best_total_stats[index]
                    for index, name in enumerate(_ATTRIBUTE_NAMES)
                },
            },
        }

    def _find_best_stats_numba(
        self,
        item_index: int,
        item_level: int,
        assets: AssetsParam = None,
    ) -> dict[str, Any]:
        numeric_item = self._get_numeric_item(item_index)
        _, _, scale_lookup_array = self._get_scale_data(item_index, item_level)
        advance_for_assets = isinstance(assets, list) and len(assets) > 1

        result = _find_best_stats_numba(
            *numeric_item,
            scale_lookup_array,
            advance_for_assets,
            _get_float_lut(),
            np.int64(self._lcg_multiplier),
            np.int64(self._lcg_increment),
        )

        best_values, best_totals, best_seeds, best_stats, best_total_value, best_total_seed, best_total_stats = result

        return {
            "best_by_stat": [
                {
                    "stat": stat_name,
                    "seed": int(best_seeds[index]),
                    "value": int(best_values[index]),
                    "total_stats": int(best_totals[index]),
                    "stats": {
                        name: int(best_stats[index, slot])
                        for slot, name in enumerate(_ATTRIBUTE_NAMES)
                    },
                }
                for index, stat_name in enumerate(_ATTRIBUTE_NAMES)
            ],
            "best_total": {
                "seed": int(best_total_seed),
                "value": int(best_total_value),
                "stats": {
                    name: int(best_total_stats[index])
                    for index, name in enumerate(_ATTRIBUTE_NAMES)
                },
            },
        }

    @staticmethod
    def _build_best_entries(
        best_values,
        best_totals,
        best_seeds,
        best_stats,
        best_visual_indices,
        best_total_value,
        best_total_seed,
        best_total_stats,
        best_total_visual_index,
    ) -> list[tuple[int, int, int, int, int, int, int, int]]:
        entries = []
        for index in range(4):
            entries.append(
                (
                    int(best_seeds[index]),
                    int(best_values[index]),
                    int(best_totals[index]),
                    int(best_visual_indices[index]),
                    int(best_stats[index][0]),
                    int(best_stats[index][1]),
                    int(best_stats[index][2]),
                    int(best_stats[index][3]),
                )
            )

        entries.append(
            (
                int(best_total_seed),
                int(best_total_value),
                int(best_total_value),
                int(best_total_visual_index),
                int(best_total_stats[0]),
                int(best_total_stats[1]),
                int(best_total_stats[2]),
                int(best_total_stats[3]),
            )
        )
        return entries

    @staticmethod
    def _records_blob_from_arrays(
        max_pick_count: int,
        visual_indices,
        pick_counts,
        stat_ids,
        raw_values,
    ) -> bytes:
        seed_count = IJ2SeedArchive.SEED_COUNT
        record_size = 4 + max_pick_count + (max_pick_count * 2)

        if np is not None and hasattr(visual_indices, "dtype"):
            records = np.zeros((seed_count, record_size), dtype=np.uint8)
            records[:, 0:2] = visual_indices.view(np.uint8).reshape(seed_count, 2)
            records[:, 2] = pick_counts
            if max_pick_count > 0:
                records[:, 4 : 4 + max_pick_count] = stat_ids
                records[:, 4 + max_pick_count :] = raw_values.view(np.uint8).reshape(seed_count, max_pick_count * 2)
            return records.tobytes()

        blob = bytearray(seed_count * record_size)
        for seed in range(seed_count):
            offset = seed * record_size
            visual_index = int(visual_indices[seed])
            blob[offset] = visual_index & 0xFF
            blob[offset + 1] = (visual_index >> 8) & 0xFF
            blob[offset + 2] = int(pick_counts[seed]) & 0xFF
            for pick_index in range(max_pick_count):
                blob[offset + 4 + pick_index] = int(stat_ids[seed][pick_index]) & 0xFF
                raw_value = int(raw_values[seed][pick_index])
                raw_offset = offset + 4 + max_pick_count + (pick_index * 2)
                blob[raw_offset] = raw_value & 0xFF
                blob[raw_offset + 1] = (raw_value >> 8) & 0xFF
        return bytes(blob)

    def _build_seed_archive_item_bytes_numba(
        self,
        item_index: int,
        item_level: int,
        assets: AssetsParam = None,
    ) -> bytes:
        compiled_item = self._compiled_items[item_index]
        max_pick_count = sum(int(selector[0]) for selector in compiled_item[3])
        numeric_item = self._get_numeric_item(item_index)
        _, _, scale_lookup_array = self._get_scale_data(item_index, item_level)
        asset_count = len(assets) if isinstance(assets, list) else 1 if isinstance(assets, str) else 0

        result = _build_seed_archive_numba(
            *numeric_item,
            scale_lookup_array,
            np.int32(asset_count),
            _get_float_lut(),
            np.int64(self._lcg_multiplier),
            np.int64(self._lcg_increment),
            np.int32(max_pick_count),
        )

        (
            visual_indices,
            pick_counts,
            stat_ids,
            raw_values,
            best_values,
            best_totals,
            best_seeds,
            best_stats,
            best_visual_indices,
            best_total_value,
            best_total_seed,
            best_total_stats,
            best_total_visual_index,
        ) = result

        best_entries = self._build_best_entries(
            best_values,
            best_totals,
            best_seeds,
            best_stats,
            best_visual_indices,
            best_total_value,
            best_total_seed,
            best_total_stats,
            best_total_visual_index,
        )
        records_blob = self._records_blob_from_arrays(
            max_pick_count,
            visual_indices,
            pick_counts,
            stat_ids,
            raw_values,
        )
        return IJ2SeedArchive.build_item_bytes(
            item_index=item_index,
            item_level=item_level,
            max_pick_count=max_pick_count,
            best_entries=best_entries,
            records_blob=records_blob,
        )

    def _build_seed_archive_item_bytes_python(
        self,
        item_index: int,
        item_level: int,
        assets: AssetsParam = None,
    ) -> bytes:
        compiled_item = self._compiled_items[item_index]
        compiled_selectors = compiled_item[3]
        _, scale_lookup, _ = self._get_scale_data(item_index, item_level)
        asset_count = len(assets) if isinstance(assets, list) else 1 if isinstance(assets, str) else 0
        max_pick_count = sum(int(selector[0]) for selector in compiled_selectors)

        seed_count = IJ2SeedArchive.SEED_COUNT
        if np is not None:
            visual_indices = np.zeros(seed_count, dtype=np.uint16)
            pick_counts = np.zeros(seed_count, dtype=np.uint8)
            stat_ids = np.zeros((seed_count, max_pick_count), dtype=np.uint8)
            raw_values = np.zeros((seed_count, max_pick_count), dtype=np.uint16)
        else:
            visual_indices = [0] * seed_count
            pick_counts = [0] * seed_count
            stat_ids = [[0] * max_pick_count for _ in range(seed_count)]
            raw_values = [[0] * max_pick_count for _ in range(seed_count)]

        multiplier = self._lcg_multiplier
        increment = self._lcg_increment

        best_values = [-1, -1, -1, -1]
        best_totals = [-1, -1, -1, -1]
        best_seeds = [0, 0, 0, 0]
        best_stats = [[0, 0, 0, 0] for _ in range(4)]
        best_visual_indices = [0, 0, 0, 0]
        best_total_value = -1
        best_total_seed = 0
        best_total_stats = [0, 0, 0, 0]
        best_total_visual_index = 0

        for seed in range(seed_count):
            state = seed
            visual_index = 0
            if asset_count > 1:
                state = (multiplier * state + increment) & _UINT32_MASK
                visual_index = (state & _INT31_MASK) % asset_count

            stats = [0, 0, 0, 0]
            pick_write = 0

            for ct, threshold, compiled_set in compiled_selectors:
                state = (multiplier * state + increment) & _UINT32_MASK
                if threshold < self._lcg_state_to_float(state):
                    continue

                direct_attrs, direct_count, total_count, step = compiled_set
                if total_count <= 0:
                    continue

                last_index = total_count - 1

                for _ in range(ct):
                    state = (multiplier * state + increment) & _UINT32_MASK
                    rand = self._lcg_state_to_float(state)
                    idx = 0
                    r = rand
                    while step < r and idx < last_index:
                        idx += 1
                        r = f32(r - step)

                    if idx >= direct_count:
                        continue

                    _, _, stat_slot, value_lo, value_hi, extra_ops = direct_attrs[idx]

                    if value_lo < value_hi and stat_slot >= 0:
                        state = (multiplier * state + increment) & _UINT32_MASK
                        raw_value = value_lo + ((state & _INT31_MASK) % (value_hi - value_lo + 1))
                        if pick_write < max_pick_count:
                            stat_ids[seed][pick_write] = stat_slot
                            raw_values[seed][pick_write] = raw_value
                            pick_write += 1
                        stats[stat_slot] += scale_lookup[raw_value]

                    for op_kind, lo, hi in extra_ops:
                        if op_kind == _OP_ADVANCE:
                            state = (multiplier * state + increment) & _UINT32_MASK
                        elif lo < hi:
                            state = (multiplier * state + increment) & _UINT32_MASK

            visual_indices[seed] = visual_index
            pick_counts[seed] = pick_write
            total = sum(stats)
            for stat_index, stat_value in enumerate(stats):
                if stat_value > best_values[stat_index] or (
                    stat_value == best_values[stat_index] and total > best_totals[stat_index]
                ):
                    best_values[stat_index] = stat_value
                    best_totals[stat_index] = total
                    best_seeds[stat_index] = seed
                    best_stats[stat_index] = stats.copy()
                    best_visual_indices[stat_index] = visual_index

            if total > best_total_value:
                best_total_value = total
                best_total_seed = seed
                best_total_stats = stats.copy()
                best_total_visual_index = visual_index

        best_entries = self._build_best_entries(
            best_values,
            best_totals,
            best_seeds,
            best_stats,
            best_visual_indices,
            best_total_value,
            best_total_seed,
            best_total_stats,
            best_total_visual_index,
        )
        records_blob = self._records_blob_from_arrays(
            max_pick_count,
            visual_indices,
            pick_counts,
            stat_ids,
            raw_values,
        )
        return IJ2SeedArchive.build_item_bytes(
            item_index=item_index,
            item_level=item_level,
            max_pick_count=max_pick_count,
            best_entries=best_entries,
            records_blob=records_blob,
        )

    @staticmethod
    def _get_random_asset(rng: LCGStream, assets: AssetsParam) -> Optional[str]:
        if assets is None:
            return None

        if isinstance(assets, str):
            return assets

        count = len(assets)
        if count == 0:
            return None
        if count == 1:
            return assets[0]

        seed = rng.advance()
        idx = lcg_to_int_range(seed, 0, count - 1)
        return assets[idx]

    def _process_selectors_compiled(
        self,
        rng: LCGStream,
        compiled_selectors: Tuple[Tuple, ...],
    ) -> list:
        picks = []
        picks_append = picks.append
        next_float = rng.next_float
        next_int_range = rng.next_int_range
        advance = rng.advance

        for ct, threshold, compiled_set in compiled_selectors:
            if threshold < next_float():
                continue

            direct_attrs, direct_count, total_count, step = compiled_set
            if total_count <= 0:
                continue

            last_index = total_count - 1

            for _ in range(ct):
                rand = next_float()
                idx = 0
                r = rand
                while step < r and idx < last_index:
                    idx += 1
                    r = f32(r - step)

                if idx >= direct_count:
                    continue

                attr_id, attr_name, _, value_lo, value_hi, extra_ops = direct_attrs[idx]

                if value_lo < value_hi:
                    picks_append((attr_id, attr_name, next_int_range(value_lo, value_hi)))

                for op_kind, lo, hi in extra_ops:
                    if op_kind == _OP_ADVANCE:
                        advance()
                    elif lo < hi:
                        next_int_range(lo, hi)

        return picks

    def _calculate_scale_factor(
        self,
        level: int,
        min_level: int,
        max_level: int,
        design_level: int,
    ) -> float:
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
        stats = {}
        scale_factor_f32 = f32(scale_factor)

        for attr_id, stat_name, raw_value in picks:
            if attr_id in self._base_stat_ids:
                scaled = max(
                    1,
                    int(f32(f32(float(raw_value)) * scale_factor_f32)),
                )
            else:
                scaled = raw_value
            stats[stat_name] = stats.get(stat_name, 0) + scaled
        return stats

    def _get_numeric_item(self, item_index: int) -> Tuple["np.ndarray", ...]:
        cached = self._numeric_item_cache.get(item_index)
        if cached is not None:
            return cached

        numeric_item = _flatten_numeric_item(self._compiled_items[item_index])
        self._numeric_item_cache[item_index] = numeric_item
        return numeric_item

    def _get_scale_data(
        self,
        item_index: int,
        item_level: int,
    ) -> tuple[float, Tuple[int, ...], Optional["np.ndarray"]]:
        cache_key = (item_index, item_level)
        cached = self._scale_cache.get(cache_key)
        if cached is not None:
            return cached

        compiled_item = self._compiled_items[item_index]
        scale_factor = self._calculate_scale_factor(
            item_level,
            min_level=compiled_item[0],
            max_level=compiled_item[1],
            design_level=compiled_item[2],
        )
        max_raw = compiled_item[4]
        scale_factor_f32 = f32(scale_factor)
        scale_lookup = tuple(
            max(1, int(f32(f32(float(raw_value)) * scale_factor_f32)))
            for raw_value in range(max_raw + 1)
        )
        scale_lookup_array = (
            np.asarray(scale_lookup, dtype=np.int32)
            if self._use_numba and np is not None
            else None
        )
        cached = (scale_factor, scale_lookup, scale_lookup_array)
        self._scale_cache[cache_key] = cached
        return cached

    def _load_or_build_cache(self) -> Tuple[Tuple, ...]:
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "rb") as f:
                    payload = pickle.load(f)
                if self._is_cache_valid(payload):
                    return payload["compiled_items"]
            except Exception:
                pass

        compiled_items = self._build_compiled_items()
        payload = {
            "version": _CACHE_VERSION,
            "item_count": len(self.items),
            "attribute_set_count": len(self.attribute_sets),
            "first_item_name": self.items[0].get("i") if self.items else None,
            "last_item_name": self.items[-1].get("i") if self.items else None,
            "compiled_items": compiled_items,
        }
        with open(self._cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return compiled_items

    def _is_cache_valid(self, payload: dict) -> bool:
        return (
            payload.get("version") == _CACHE_VERSION
            and payload.get("item_count") == len(self.items)
            and payload.get("attribute_set_count") == len(self.attribute_sets)
            and payload.get("first_item_name")
            == (self.items[0].get("i") if self.items else None)
            and payload.get("last_item_name")
            == (self.items[-1].get("i") if self.items else None)
        )

    def _build_compiled_items(self) -> Tuple[Tuple, ...]:
        compiled_sets = tuple(
            self._compile_attr_set(attr_set)
            for attr_set in self.attribute_sets
        )

        compiled_items = []
        for item_def in self.items:
            selectors = item_def.get("a", {}).get("s", [])
            compiled_selectors = []
            for selector in selectors:
                set_index = selector["s"]
                compiled_selectors.append(
                    (
                        selector["ct"],
                        f32(selector["ch"] / 100.0),
                        compiled_sets[set_index],
                    )
                )

            max_raw_value = 0
            for _, _, compiled_set in compiled_selectors:
                for _, _, _, value_lo, value_hi, _ in compiled_set[0]:
                    if value_hi > max_raw_value:
                        max_raw_value = value_hi

            compiled_items.append(
                (
                    item_def.get("l", 1),
                    item_def.get("h", 30),
                    item_def.get("dL", 20),
                    tuple(compiled_selectors),
                    max_raw_value,
                )
            )

        return tuple(compiled_items)

    def _compile_attr_set(self, attr_set: dict) -> Tuple:
        direct_attrs = tuple(
            self._compile_attr(attr_def)
            for attr_def in attr_set.get("a", {}).get("a", [])
        )
        direct_count = len(direct_attrs)
        total_count = direct_count + len(attr_set.get("a", {}).get("s", []))
        step = f32(1.0 / float(total_count)) if total_count > 0 else 0.0
        return (direct_attrs, direct_count, total_count, step)

    def _compile_attr(self, attr_def: dict) -> Tuple:
        attr_id = attr_def["i"]
        stat_slot = attr_id if attr_id in self._base_stat_ids else -1
        attr_name = self._attributes_map.get(attr_id, f"attr_{attr_id}")
        value_lo = 0
        value_hi = 0
        extra_ops = []

        for param in attr_def.get("p", []):
            param_id = param["id"]

            if param_id == "value" and "i" in param:
                value_lo = param["i"]["l"]
                value_hi = param["i"]["h"]

            elif param_id in ("base", "amount", "damage") and "f" in param:
                lo_f = param["f"]["l"]
                hi_f = param["f"]["h"]
                if lo_f < hi_f:
                    extra_ops.append((_OP_ADVANCE, 0, 0))

            elif param_id == "set" and "i" in param:
                lo = param["i"]["l"]
                hi = param["i"]["h"]
                if lo < hi:
                    extra_ops.append((_OP_INT_RANGE, lo, hi))

        return (attr_id, attr_name, stat_slot, value_lo, value_hi, tuple(extra_ops))

    @staticmethod
    def _lcg_state_to_float(state: int) -> float:
        return lcg_to_float(state)
