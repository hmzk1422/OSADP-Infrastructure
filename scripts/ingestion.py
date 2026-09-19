"""
ingestion.py
=================================================
The main unified runner for ingestion, specifically ingestion_L2.
"""

from pathlib import Path
import argparse
import subprocess
import sys

def run(cmd: list[str], name: str) -> None:
    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise SystemExit(
            f"{name} failed with exit code {result.returncode}. "
            "Pipeline stopped."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--root", default="./research/data_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-raw", action="store_true")
    parser.add_argument(
        "--compress",
        action="store_true",
        help="After conversion, recompress CryptoHFTData parquet back to "
        ".parquet.zst via zstandard_fix.py (reclaims disk space; run "
        "zstandard_fix.py decompress --root <root> before reading them "
        "with research.MoE.data.live).",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=15,
        help="zstd compression level for --compress (default: 15)",
    )
    # FIX (2026-09-18): forward a dataset subset to both pipeline stages so a
    # partial re-run does not re-download/convert datasets that are already
    # complete (e.g. --datasets bookDepth,orderbook).
    parser.add_argument(
        "--datasets",
        default="aggTrades,bookDepth,orderbook",
        help="Comma-separated subset of datasets to process in both stages "
             "(default: all of aggTrades,bookDepth,orderbook)",
    )
    args = parser.parse_args()

    python = sys.executable

    # Directory containing this runner.
    project_root = Path(__file__).resolve().parent

    # Both implementation files live here.
    ingestion_dir = project_root / "ingestion_L2"

    downloader = ingestion_dir / "ofp_data_downloader.py"
    converter = ingestion_dir / "convert_to_parquet.py"

    # ------------------------------------------------------------
    # Stage 1: download native raw data
    # ------------------------------------------------------------
    downloader_cmd = [
        python,
        str(downloader),
        "--symbol", args.symbol,
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--root", args.root,
        "--workers", str(args.workers),
        "--datasets", args.datasets,
    ]

    run(downloader_cmd, "STAGE 1 — RAW DATA INGESTION")

    # ------------------------------------------------------------
    # Stage 2: convert/re-home into Parquet
    # ------------------------------------------------------------
    converter_cmd = [
        python,
        str(converter),
        "--root", args.root,
        "--symbol", args.symbol,
    ]

    converter_cmd.append("--datasets")
    converter_cmd.append(args.datasets)

    if args.overwrite:
        converter_cmd.append("--overwrite")

    if not args.delete_raw:
        converter_cmd.append("--keep-raw")

    run(converter_cmd, "STAGE 2 — PARQUET CONVERSION")

    # ------------------------------------------------------------
    # Stage 3 (optional): recompress CryptoHFTData parquet to .zst
    # ------------------------------------------------------------
    if args.compress:
        fixer = ingestion_dir / "zstandard_fix.py"

        fixer_cmd = [
            python,
            str(fixer),
            "compress",
            "--level", str(args.compress_level),
        ]

        run(fixer_cmd, "STAGE 3 — ZSTD ARCHIVAL COMPRESSION")

        print(
            "\nNote: orderbook parquet files are now .parquet.zst archives. "
            "Run 'zstandard_fix.py decompress' before fetching source data."
        )

    print("\nIngestion completed successfully.")


if __name__ == "__main__":
    main()