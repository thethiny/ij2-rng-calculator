"""
LCG (Linear Congruential Generator) implementation.

Game-agnostic: the multiplier and increment are passed to LCGStream,
not imported from consts. Only IEEE 754 / 32-bit arithmetic constants
are hardcoded since those are universal.

Source:
  - SerializeFrom advancing the stream
  - AttributeList::Add advancing the stream
"""

import struct

# Universal constants — IEEE 754 / 32-bit integer arithmetic, never game-specific
_MANTISSA_MASK = 0x7FFFFF  # Low 23 bits (IEEE 754 single-precision mantissa)
_FLOAT_ONE_BITS = 0x3F800000  # IEEE 754 representation of 1.0f
_INT31_MASK = 0x7FFFFFFF  # 31-bit mask for unsigned int random values
_UINT32_MOD = 0x100000000  # 2^32
_PACK_FLOAT = struct.Struct("f").pack
_UNPACK_FLOAT = struct.Struct("f").unpack
_PACK_UINT32 = struct.Struct("I").pack


def f32(x: float) -> float:
    """Cast a Python float to IEEE 754 single-precision (float32), matching C float behavior."""
    return _UNPACK_FLOAT(_PACK_FLOAT(x))[0]


def lcg_to_float(seed: int) -> float:
    """
    Convert an LCG state to a float in [0.0, 1.0).

    C equivalent (from AttributeSelectorDefinition::GenerateAttributes):
        COERCE_FLOAT(Seed & 0x7FFFFF | 0x3F800000)
        - (float)(int)COERCE_FLOAT(Seed & 0x7FFFFF | 0x3F800000)

    Takes the low 23 bits as IEEE 754 mantissa with exponent=127 -> float in [1.0, 2.0),
    then subtracts 1.0 to produce [0.0, 1.0).
    """
    bits = (seed & _MANTISSA_MASK) | _FLOAT_ONE_BITS
    f = _UNPACK_FLOAT(_PACK_UINT32(bits))[0]
    return f - 1.0


def lcg_to_int_range(seed: int, min_val: int, max_val: int) -> int:
    """
    Convert an LCG state to an integer in [min_val, max_val].

    C equivalent (from AttributeParameterListSerializer::SerializeFrom):
        min + ((unsigned int)Seed & 0x7FFFFFFF) % (max - min + 1)
    """
    if min_val >= max_val:
        return min_val
    return min_val + ((seed & _INT31_MASK) % (max_val - min_val + 1))


class LCGStream:
    """
    Stateful LCG random stream.

    The multiplier and increment are constructor parameters so this class
    works with any LCG variant, not just UE3's.
    """

    def __init__(self, seed: int, multiplier: int, increment: int):
        """
        Args:
            seed: Initial seed value.
            multiplier: LCG multiplier (a in: next = a*seed + c).
            increment: LCG increment (c in: next = a*seed + c).
        """
        self.seed = seed & 0xFFFFFFFF
        self._multiplier = multiplier
        self._increment = increment

    def advance(self) -> int:
        """Advance the LCG by one step and return the new seed."""
        self.seed = (self._multiplier * self.seed + self._increment) % _UINT32_MOD
        return self.seed

    def next_float(self) -> float:
        """Advance and return float in [0.0, 1.0)."""
        return lcg_to_float(self.advance())

    def next_int_range(self, min_val: int, max_val: int) -> int:
        """Advance and return int in [min_val, max_val]. No advance if min >= max."""
        if min_val >= max_val:
            return min_val
        return lcg_to_int_range(self.advance(), min_val, max_val)
