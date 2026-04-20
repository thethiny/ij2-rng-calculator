from .stat_generator import StatGenerator
from .lcg import LCGStream, lcg_to_float, lcg_to_int_range
from .seed_archive import IJ2SeedArchive, SeedArchiveHeader


def __getattr__(name):
    if name == "FastStatGenerator":
        from .stat_fast import FastStatGenerator
        return FastStatGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StatGenerator",
    "FastStatGenerator",
    "LCGStream",
    "lcg_to_float",
    "lcg_to_int_range",
    "IJ2SeedArchive",
    "SeedArchiveHeader",
]
