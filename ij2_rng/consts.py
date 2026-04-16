"""
IJ2-specific default values for the stat generation algorithm.

These values were reverse-engineered from the game binary and
cross-referenced in-game.
"""

# LCG (Linear Congruential Generator) constants - UE3 standard
# Used in: GInventoryRandomStream, GenerateChestContents, all RNG paths
# Formula: next_seed = (LCG_MULTIPLIER * seed + LCG_INCREMENT) mod 2^32
LCG_MULTIPLIER = 196314165  # 0x0BB40E65
LCG_INCREMENT = 907633515  # 0x36156A3B

# Attribute ID -> stat name mapping
# Source: attributeSettings array in geardefinitionlist, indices 0-3
ATTRIBUTES_MAP = {
    0: "Health",
    1: "Defense",
    2: "Strength",
    3: "Ability",
}

# The 4 base stats that are scaled by CalculateScaleFactor
BASE_STAT_IDS = {0, 1, 2, 3} # From ATTRIBUTES_MAP

# CalculateScaleFactor constants (from EquipmentSaveData::CalculateScaleFactor)
SCALE_BOOST_AT_MAX = 0.9  # Boost when effectiveLevel == maxLevel (FLOAT_0_89999998)
SCALE_BOOST_BELOW_MAX = 0.7  # Boost when effectiveLevel < maxLevel (FLOAT_0_69999999)
SCALE_BASE = 0.1  # Base scale factor added after boost multiplication
