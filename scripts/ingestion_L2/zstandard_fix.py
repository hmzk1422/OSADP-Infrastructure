"""
zstandard_fix.py
================
Lifecycle utilities for the CryptoHFTData orderbook files.

Modes
-----
repair      Files sitting at *.parquet that are actually zstd streams (the
            raw *.parquet.zst downloads saved without renaming) are
            decompressed in place.
compress    Inverse of repair: recompress plain *.parquet files into
            *.parquet.zst (default zstd level 15) to reclaim disk space.
            The original .parquet is only removed after a byte-identical
            roundtrip verification.
decompress  Inverse of compress: restore *.parquet.zst -> *.parquet.

Usage
-----
python scripts/ingestion_L2/zstandard_fix.py                  # repair
python scripts/ingestion_L2/zstandard_fix.py compress --level 15
python scripts/ingestion_L2/zstandard_fix.py decompress
"""

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import zstandard as zstd


# FIX (2026-09-18): CryptoHFTData hourly files were moved into the unified
# parquet tree (convert_to_parquet.py). Point the lifecycle utilities there.
ROOT = Path(
    "research/data_v2/parquet/BTCUSDT/orderbook"
)

DEFAULT_LEVEL = 15

CHUNK = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as src:
        for chunk in iter(lambda: src.read(CHUNK), b""):
            digest.update(chunk)

    return digest.hexdigest()


def repair_file(path: Path):
    # Already valid Parquet
    try:
        pl.read_parquet_schema(path)
        print(f"OK      {path}")
        return
    except Exception:
        pass

    print(f"REPAIR  {path}")

    tmp = path.with_suffix(".decompressed.parquet")

    try:
        with open(path, "rb") as src:
            dctx = zstd.ZstdDecompressor()

            with dctx.stream_reader(src) as reader:
                with open(tmp, "wb") as dst:
                    shutil.copyfileobj(reader, dst, CHUNK)

        # Verify decompressed output before replacing original
        pl.read_parquet_schema(tmp)

        os.replace(tmp, path)

        print(f"FIXED   {path}")

    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"FAILED  {path}: {exc}")


def compress_file(path: Path, level: int, verify: bool = True):
    target = path.with_name(path.name + ".zst")

    if target.exists():
        print(f"CACHED  {target}")
        return

    tmp = path.with_name(path.name + ".zst.tmp")
    verify_tmp = path.with_name(path.name + ".zst_verify")

    try:
        original_hash = _sha256(path) if verify else None

        with open(path, "rb") as src, open(tmp, "wb") as dst:
            cctx = zstd.ZstdCompressor(
                level=level,
                write_content_size=True,
                write_checksum=True,
                threads=-1,
            )
            cctx.copy_stream(src, dst)

        if verify:
            # Roundtrip-verify before touching the original file.
            with open(tmp, "rb") as src:
                dctx = zstd.ZstdDecompressor()

                with dctx.stream_reader(src) as reader:
                    with open(verify_tmp, "wb") as dst:
                        shutil.copyfileobj(reader, dst, CHUNK)

            if _sha256(verify_tmp) != original_hash:
                raise ValueError("roundtrip verification mismatch")

            verify_tmp.unlink()

        os.replace(tmp, target)
        os.remove(path)

        print(f"ZIPPED  {target}")

    except Exception as exc:
        tmp.unlink(missing_ok=True)
        verify_tmp.unlink(missing_ok=True)
        print(f"FAILED  {path}: {exc}")


def decompress_file(path: Path):
    target = path.with_suffix("")

    if target.exists():
        print(f"EXISTS  {target}")
        return

    tmp = target.with_suffix(".decompressed.parquet")

    try:
        with open(path, "rb") as src:
            dctx = zstd.ZstdDecompressor()

            with dctx.stream_reader(src) as reader:
                with open(tmp, "wb") as dst:
                    shutil.copyfileobj(reader, dst, CHUNK)

        pl.read_parquet_schema(tmp)

        os.replace(tmp, target)
        os.remove(path)

        print(f"UNPACK  {target}")

    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"FAILED  {path}: {exc}")


def _run(files, worker, workers, label):
    print(f"Found {len(files):,} files for '{label}'")

    if not files:
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, files))


def main():
    parser = argparse.ArgumentParser(
        description="Repair / compress / decompress CryptoHFTData orderbook files.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="repair",
        choices=["repair", "compress", "decompress"],
        help="repair (default): decompress mislabeled .parquet.zst in place, "
        "compress: .parquet -> .parquet.zst, decompress: .parquet.zst -> .parquet",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--level",
        type=int,
        default=DEFAULT_LEVEL,
        help=f"zstd compression level for compress mode (default: {DEFAULT_LEVEL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="concurrent files handled at once (default: 2)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the byte-identical roundtrip verification on compress",
    )
    args = parser.parse_args()

    if args.mode == "compress":
        files = sorted(args.root.rglob("*.parquet"))
        _run(
            files,
            lambda p: compress_file(p, args.level, not args.no_verify),
            args.workers,
            args.mode,
        )
    elif args.mode == "decompress":
        files = sorted(args.root.rglob("*.parquet.zst"))
        _run(files, decompress_file, args.workers, args.mode)
    else:
        files = sorted(args.root.rglob("*.parquet"))
        _run(files, repair_file, args.workers, args.mode)


if __name__ == "__main__":
    main()