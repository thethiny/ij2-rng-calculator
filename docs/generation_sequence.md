# Procedural Items & Asset Groups

## What the player sees

Every gear piece in Injustice 2 has a 3D model (mesh) displayed on the character. Some gear pieces always look the same — Batman's Nanotech Greaves always show the same shin guard model. Other gear pieces have **multiple visual variants** — a "Common Cowl" might appear as any of several different cowl shapes, picked at random when the item is generated.

This is the difference between **single-asset** and **multi-asset** items.

## Single-asset items (non-procedural)

These items have one fixed appearance. The item definition stores a single MD5 hash (`mAssetHash`) pointing to one specific mesh/material combination.

In the canonical catalog payload (online `geardefinitionlist` or offline
`ItemDefinitions` from `TweakVars`), these items have **no `"p"` field** (or
`"p": false`). In the game binary, `mAssetGroupIndex == -1`.

Examples: Epic gear, Legendary gear, Set pieces — visually unique items that always look the same regardless of seed.

**Count:** 17,569 items in the catalog.

## Multi-asset items (procedural)

These items can take on different appearances. Instead of storing a single hash, the item points to an **asset group** — an array of MD5 hashes, each representing a different visual variant of the same gear slot.

In the catalog: `"p": true`. In the binary: `mAssetGroupIndex >= 0`, indexing into a global `sAssetGroups` array.

Examples: Common and Rare gear — the same item ID (e.g. "BM_LegsRare15") can appear as several visually distinct leg armors.

**Count:** 7,063 items, spread across 7,459 groups (some groups are shared).

## How the visual variant is selected

When the game renders a gear piece, `ItemDefinition::GetRandomAsset()` runs:

```
if mAssetGroupIndex == -1:
    return mAssetHash                    # single-asset: fixed look
else:
    group = sAssetGroups[mAssetGroupIndex]
    return group.GetRandomAsset()        # multi-asset: seed-based pick
```

For multi-asset groups, `AssetGroupDefinition::GetRandomAsset()` picks a variant:

```
asset_count = group.mItems.length

if asset_count == 1:
    return group.mItems[0]               # only one variant, no randomness
else:
    seed = LCG_advance(GInventoryRandomStream.Seed)
    index = (seed & 0x7FFFFFFF) % asset_count
    return group.mItems[index]           # seeded selection
```

The `GInventoryRandomStream` is initialized from the item's `RandomSeed` in `CacheGeneratedData` **before** this function runs. So the same seed always produces the same visual variant.

## Why this matters for stat generation

`GetRandomAsset` is called **before** attribute generation in `CacheGeneratedData`. When the asset group has >1 variants, the LCG advances once to pick the visual. This shifts the entire LCG state for all subsequent stat rolls.

| Scenario | LCG advance in GetRandomAsset | Effect on stats |
|---|---|---|
| Single-asset item (`mAssetGroupIndex == -1`) | None | Stats start from raw seed |
| Multi-asset group with 1 entry | None (index is always 0) | Stats start from raw seed |
| Multi-asset group with 2+ entries | 1 advance | Stats start from LCG(seed) |

This is the `has_multiple_assets` parameter in `StatGenerator.generate()`. It
cannot be determined from the catalog payload alone - it requires the asset
group sizes from `ITEMDEFINITIONSAUX.xxx`.

## Group size distribution

From the 7,459 asset groups parsed from ITEMDEFINITIONSAUX:

- **2,924 groups** have exactly 1 asset (procedural flag but no visual variety)
- **4,531 groups** have 2-209 assets (actual visual variety)
- All multi-asset groups belong to a single character (no cross-character sharing)

## The `"p"` flag vs actual multi-asset

The `"p": true` flag in the catalog means the item is **procedural** (part of an asset group), but that does not guarantee multiple visual variants. A procedural item can point to a group with only 1 asset — in that case it behaves identically to a single-asset item for both visuals and stat generation.

```
p=false  OR  p absent   →  single-asset      →  0 LCG advances
p=true,  group size 1   →  procedural         →  0 LCG advances  (no visual variety)
p=true,  group size 2+  →  procedural         →  1 LCG advance   (seed picks the look)
```
