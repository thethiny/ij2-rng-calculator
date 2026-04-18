import json
import struct
import zipfile
from dataclasses import dataclass
from typing import Any, BinaryIO, Optional


_MAGIC = b"I2SD"
_FORMAT_VERSION = 2
_HEADER_STRUCT = struct.Struct("<4sHIIIHHH")
_BEST_ENTRY_STRUCT = struct.Struct("<8H")


@dataclass(frozen=True)
class SeedArchiveHeader:
    format_version: int
    item_index: int
    item_level: int
    seed_count: int
    max_pick_count: int
    record_size: int
    summary_count: int


class IJ2SeedArchive:
    MAGIC = _MAGIC
    FORMAT_VERSION = _FORMAT_VERSION
    SEED_COUNT = 0x10000
    ITEM_CODEC = "none"
    SUMMARY_NAMES = ("Health", "Defense", "Strength", "Ability", "Total")
    BEST_ENTRY_SIZE = _BEST_ENTRY_STRUCT.size
    HEADER_SIZE = _HEADER_STRUCT.size

    @classmethod
    def build_meta(
        cls,
        *,
        archive_version: str,
        database_version: str,
        lcg_multiplier: int,
        lcg_increment: int,
        item_level: int,
        item_count: int,
        selector_item_count: int,
        seed_count: int = SEED_COUNT,
    ) -> dict[str, Any]:
        return {
            "format": "ij2-precomputed-item-seeds",
            "version": archive_version,
            "lcg_multiplier": lcg_multiplier,
            "lcg_increment": lcg_increment,
            "format_version": cls.FORMAT_VERSION,
            "item_codec": cls.ITEM_CODEC,
            "database_version": database_version,
            "item_level": item_level,
            "item_count": item_count,
            "selector_item_count": selector_item_count,
            "seed_count": seed_count,
        }

    @classmethod
    def build_item_bytes(
        cls,
        *,
        item_index: int,
        item_level: int,
        max_pick_count: int,
        best_entries: list[tuple[int, int, int, int, int, int, int, int]],
        records_blob: bytes,
        seed_count: int = SEED_COUNT,
    ) -> bytes:
        record_size = 0 if seed_count == 0 else 4 + max_pick_count + (max_pick_count * 2)
        header = _HEADER_STRUCT.pack(
            cls.MAGIC,
            cls.FORMAT_VERSION,
            item_index,
            item_level,
            seed_count,
            max_pick_count,
            record_size,
            len(best_entries),
        )
        summary = b"".join(_BEST_ENTRY_STRUCT.pack(*entry) for entry in best_entries)
        return header + summary + records_blob

    @classmethod
    def parse_header(cls, data: bytes) -> SeedArchiveHeader:
        (
            magic,
            format_version,
            item_index,
            item_level,
            seed_count,
            max_pick_count,
            record_size,
            summary_count,
        ) = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid archive magic: {magic!r}")
        return SeedArchiveHeader(
            format_version=format_version,
            item_index=item_index,
            item_level=item_level,
            seed_count=seed_count,
            max_pick_count=max_pick_count,
            record_size=record_size,
            summary_count=summary_count,
        )

    @classmethod
    def parse_best_entries(cls, data: bytes) -> dict[str, dict[str, Any]]:
        header = cls.parse_header(data)
        offset = cls.HEADER_SIZE
        entries: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(cls.SUMMARY_NAMES[: header.summary_count]):
            seed, value, total, visual_index, health, defense, strength, ability = _BEST_ENTRY_STRUCT.unpack_from(data, offset)
            entries[name] = {
                "seed": seed,
                "value": value,
                "total": total,
                "visual_index": visual_index,
                "stats": {
                    "Health": health,
                    "Defense": defense,
                    "Strength": strength,
                    "Ability": ability,
                },
            }
            offset += cls.BEST_ENTRY_SIZE
        return entries

    @classmethod
    def read_seed_record(cls, data: bytes, seed: int) -> dict[str, Any]:
        header = cls.parse_header(data)
        if seed < 0 or seed >= header.seed_count:
            raise IndexError(f"Seed out of range: {seed}")

        summary_size = header.summary_count * cls.BEST_ENTRY_SIZE
        record_offset = cls.HEADER_SIZE + summary_size + (seed * header.record_size)
        visual_index = struct.unpack_from("<H", data, record_offset)[0]
        pick_count = data[record_offset + 2]
        stat_ids_offset = record_offset + 4
        raw_values_offset = stat_ids_offset + header.max_pick_count

        picks = []
        for pick_index in range(pick_count):
            stat_id = data[stat_ids_offset + pick_index]
            raw_value = struct.unpack_from("<H", data, raw_values_offset + (pick_index * 2))[0]
            picks.append((stat_id, raw_value))

        return {
            "visual_index": visual_index,
            "pick_count": pick_count,
            "picks": picks,
        }

    @classmethod
    def write_zip_meta(cls, archive: zipfile.ZipFile, meta: dict[str, Any]) -> None:
        archive.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=4))

    @classmethod
    def write_item(cls, archive: zipfile.ZipFile, item_index: int, item_bytes: bytes) -> None:
        archive.writestr(cls.item_path(item_index), item_bytes)

    @staticmethod
    def item_path(item_index: int) -> str:
        return f"items/{item_index}.bin"

    @classmethod
    def load_meta_from_zip(cls, path: str) -> dict[str, Any]:
        with zipfile.ZipFile(path, "r") as archive:
            with archive.open("meta.json", "r") as f:
                return json.load(f)

    @classmethod
    def load_item_from_zip(cls, path: str, item_index: int) -> bytes:
        with zipfile.ZipFile(path, "r") as archive:
            with archive.open(cls.item_path(item_index), "r") as f:
                return f.read()

    @classmethod
    def read_best_entries_from_zip(cls, path: str, item_index: int) -> dict[str, dict[str, Any]]:
        return cls.parse_best_entries(cls.load_item_from_zip(path, item_index))

    @classmethod
    def read_seed_from_zip(cls, path: str, item_index: int, seed: int) -> dict[str, Any]:
        return cls.read_seed_record(cls.load_item_from_zip(path, item_index), seed)
