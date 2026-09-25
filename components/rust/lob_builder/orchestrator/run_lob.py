"""
run_lob.py
==========
Orchestrator for executing the compiled Rust LOB (Limit Order Book) builder binary.

Role in Pipeline:
    Acts as the entry point for running the high-performance Rust LOB engines 
    (`btree_ver` or `hash_ver`). It resolves cross-platform paths, validates 
    executable availability, and forwards CLI parameters directly to Rust.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_binary_path(engine_type: str) -> Path:
    """
    Locates and returns the path to the compiled Rust LOB binary.

    Handles cross-platform executable extensions (.exe on Windows) and checks
    both local package target directories and the workspace-level target folder.

    Args:
        engine_type (str): LOB engine backend implementation ('btree' or 'hash').

    Returns:
        Path: Resolved filesystem path to the compiled Rust binary.

    Raises:
        FileNotFoundError: If the compiled release binary does not exist.
    """
    ext = ".exe" if sys.platform.startswith("win") else ""
    # Selects target binary based on implementation choice
    binary_name = f"{engine_type}_ver{ext}"
    target_path = Path(__file__).parents[1] / engine_type / "target" / "release" / binary_name
    
    if not target_path.exists():
        # Fallback to root workspace target folder
        target_path = Path(__file__).parents[2] / "target" / "release" / binary_name
        
    if not target_path.exists():
        raise FileNotFoundError(
            f"Binary not found at {target_path}. Run 'cargo build --release' first."
        )
    return target_path


def main():
    """
    Parses arguments and invokes the selected Rust LOB builder binary.
    """
    parser = argparse.ArgumentParser(description="LOB Builder Python Orchestrator")
    parser.add_argument("--engine", choices=["btree", "hash"], default="btree", help="LOB engine backend")
    parser.add_argument("--root", required=True, help="Market data root folder")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    args, unknown_args = parser.parse_known_args()

    binary = get_binary_path(args.engine)
    cmd = [
        str(binary),
        "--root", args.root,
        "--symbol", args.symbol,
        "--date", args.date,
    ] + unknown_args

    print(f"[LOB_BUILDER] Running {args.engine}_ver for {args.date}...")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] LOB Builder failed with exit code {res.returncode}")
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()