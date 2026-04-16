# Injustice 2 Gear Stat RNG Calculator

Reverse-engineered algorithm for Injustice 2's deterministic gear stat generation. Given a gear item's `RandomSeed`, `ItemLevel`, and `ItemIndex`, this tool reproduces the exact stat values shown in-game.

For a fully deailed generation sequence refer to [the docs](/docs/generation_sequence.md).

## Data Source

All item definitions, attribute sets, and generation parameters come from the server's **geardefinitionlist** object, fetched via:

```
GET /objects/geardefinitionlist/{id}
```

This is an unauthenticated endpoint that returns the full gear catalog. The latest version used is `0.294@938` (object ID `5ada00cae9f89f37ec89c268`, updated 2018-05-22). A copy is stored in `data/geardefinitionlist.json`.

The `data` field of this object contains:
| Key | Count | Purpose |
|-----|-------|---------|
| `items` | 24,366 | Item definitions. Array index == `ItemIndex`. |
| `attributeSets` | 118 | Stat pools referenced by item selectors. |
| `attributeSettings` | 638 | Per-attribute metadata (name, display range). |
| `usePowerWhenGeneratingItems` | `false` | Controls which code path generates stats. |

## Algorithm Overview

The game client generates gear stats deterministically from three inputs stored in the server inventory:

```json
{
  "RandomSeed": 47454,
  "ItemLevel": 10,
  "ItemIndex": 2066
}
```

The full pipeline is implemented in `EquipmentSaveData::CacheGeneratedData` (ij2_all_0166.c:5160):

### Step 1: Initialize LCG

```c
GInventoryRandomStream.mStream.Seed = this->mSeed;  // uint16 from RandomSeed
```

The game uses a **Linear Congruential Generator** (UE3 standard):

```
next_seed = (196314165 * seed + 907633515) mod 2^32
```

Constants: multiplier `0x0BB40E65`, increment `0x36156A3B`, modulus `2^32`.

### Step 2: GetRandomAsset (Visual Variant)

Before stat generation, `ItemDefinition::GetRandomAsset` (ij2_all_0168.c:14680) selects the visual variant for group items:

```c
if (mAssetGroupIndex != -1 && group.mItems.ArrayNum > 1) {
    GInventoryRandomStream.Seed = LCG(Seed);  // advances LCG once
    selectedAsset = (Seed & 0x7FFFFFFF) % ArrayNum;
}
```

- Items with `"p": true` in the catalog are **procedural** (group items with potentially multiple visual variants).
- If the group has **>1 assets**, the LCG advances once before stat generation.
- If the group has exactly **1 asset** (or the item is non-procedural), no LCG advance occurs.

Whether a group has >1 assets is determined by `ITEMDEFINITIONSAUX.xxx`, not the geardefinitionlist.

### Step 3: Attribute Selection (Non-Power Path)

Since `usePowerWhenGeneratingItems = false`, the game uses `AttributeList::Add` (ij2_all_0165.c:9822) instead of `GenerateAttributes`. This path processes the item's selectors sequentially using the global LCG stream.

Each item's `a.s` array defines selectors:
```json
{
  "a": {
    "s": [
      {"ct": 2, "s": 7, "ch": 100}
    ]
  }
}
```

| Field | Meaning |
|-------|---------|
| `ct` | Count — how many attributes to pick from the set |
| `s` | Set index — index into the `attributeSets` array |
| `ch` | Chance — probability (0-100) that this selector activates |

#### Chance Check

```c
Seed = LCG(Seed);
float rand = bits_to_float(Seed & 0x7FFFFF | 0x3F800000) - 1.0;  // [0, 1)
if ((chance / 100.0f) >= rand) { /* proceed */ }
```

The float conversion takes the low 23 bits as an IEEE 754 mantissa with exponent for [1.0, 2.0), then subtracts 1.0 to produce [0.0, 1.0).

#### Uniform Pick

For each pick (up to `ct` times):

```c
Seed = LCG(Seed);
float rand = bits_to_float(Seed);
float step = 1.0f / total_count;
int index = 0;
while (step < rand) { index++; rand -= step; }
```

This uniformly selects from the attribute set's direct attributes (and sub-selectors, if any). Picks are **with replacement** — the same attribute can be selected multiple times.

#### Value Generation

For each picked base stat attribute (Health, Defense, Strength, Ability), the `"value"` parameter is read via `SerializeFrom` (ij2_all_0169.c:17607):

```c
// Non-power path: random value in [min, max]
Seed = LCG(Seed);
value = min + ((Seed & 0x7FFFFFFF) % (max - min + 1));
```

The `"set"` parameter (always min=max=0) does **not** advance the LCG.

### Step 4: Scale Factor

`CalculateScaleFactor` (ij2_all_0166.c:5301) computes a level-based multiplier using the item's **design level** (`dL` field, typically 20):

```c
effectiveLevel = clamp(mLevel, minLevel, maxLevel);
boost = (effectiveLevel == maxLevel) ? 0.9f : 0.7f;

if (designLevel - 1 > 0)
    return (1.0f - (float)(designLevel - effectiveLevel) / (float)(designLevel - 1)) * boost + 0.1f;
return 1.0f;
```

| Level | dL=20 | Scale Factor |
|-------|-------|-------------|
| 1 | 20 | 0.100 |
| 2 | 20 | 0.137 |
| 10 | 20 | 0.432 |
| 20 | 20 | 0.800 |
| 30 | 20 | 0.800 (capped at dL) |
| 30 (max) | 30 | 1.000 (boost=0.9) |

Note: `dL` (design level), not `h` (max level = 30), controls the scaling curve. This field comes from the item's `"dL"` key in the geardefinitionlist items array.

### Step 5: Per-Pick Scaling

The scale factor is applied to **each attribute pick independently** via `Multiply` (ij2_all_0169.c:2522), then `GetValue` (ij2_all_0168.c:16966) **sums** all picks per stat:

```c
// Multiply (called per-pick via vtable[3])
int scaled = (int)(float)((float)this->value * scaleFactor);
this->value = max(1, scaled);

// GetValue (called per-stat to read display value)
for each attribute matching stat_id:
    total += attribute->value;
```

This is critical: `int(147 * 0.432) + int(83 * 0.432) = 63 + 35 = 98`, **not** `int((147+83) * 0.432) = 99`.

## Power Is Not Used for Stats

The geardefinitionlist sets `usePowerWhenGeneratingItems: false`. This means the `Power` field in the inventory **does not affect stat generation**. It only affects the sell price calculation in `GetSellValue` (ij2_all_0168.c:15196).

The server accepts any Power value (0-255) without validation. The game client defaults uninitialized power to 256 (stored as `*(_WORD *)&__that.mPower = 256` in ij2_all_0169.c:8290), which is clamped to 100 in `CacheGeneratedData`.

## Attribute Sets

Each attribute set (indexed by the selector's `s` field) contains an array of attribute definitions:

```json
{
  "a": {
    "a": [
      {
        "i": 2,
        "p": [
          {"i": {"h": 150, "l": 70}, "id": "value"},
          {"i": {"h": 0, "l": 0}, "id": "set"}
        ]
      }
    ]
  }
}
```

| Field | Meaning |
|-------|---------|
| `i` | Attribute ID (0=Health, 1=Defense, 2=Strength, 3=Ability) |
| `p[].id` | Parameter name (`"value"` = stat value, `"set"` = stat group) |
| `p[].i.l` | Integer minimum |
| `p[].i.h` | Integer maximum |
| `p[].f.l` | Float minimum (for non-base-stat attributes) |
| `p[].f.h` | Float maximum |

## Remaining Work: Asset Group Counts

The `has_multiple_assets` parameter (whether `GetRandomAsset` advances the LCG) cannot be determined from the geardefinitionlist alone. It requires parsing `ITEMDEFINITIONSAUX.xxx` to check each procedural item's asset group member count.

The `p` flag in the catalog indicates a procedural/group item, but groups can have 1 or more assets. Only groups with >1 asset trigger the LCG advance.
