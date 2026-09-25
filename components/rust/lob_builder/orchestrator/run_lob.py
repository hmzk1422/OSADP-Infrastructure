"""
run_lob.py
==========
Orchestrator for the compiled Rust LOB builder binary (lob_replay).

Role in Pipeline:
    Thin, faithful wrapper around the real lob_replay CLI. It resolves the
    compiled binary, exposes each Rust mode as a Python function so higher-level
    commands can call it in-process, and forwards the exact flags the binary
    actually accepts.

    Real CLI (see components/rust/lob_builder/btree_ver/main.rs):

        lob_replay ingest-parquet --input FILE|DIR --output-dir DIR
            --tick-size F [--snapshot-policy day_start|every_event] [--warm-start]
        lob_replay columnar --input FILE --output FILE --tick-size F
            --levels N --snapshot-policy day_start|every_event
            [--snapshot-dir DIR] [--snapshot-levels N]
            [--seed-snapshot FILE] [--skip-until MS] [--warm-start]
        lob_replay matched --input FILE --output FILE --tick-size F [--warm-start]

    Every other flag is rejected by the binary with `unknown flag`.
    """

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Script entry: put the repo root on sys.path before importing components.*.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODES = ("ingest-parquet", "columnar", "matched")

# Flags each mode accepts, in the order they are emitted.
MODE_FLAGS: dict[str, tuple[str, ...]] = {
    "ingest-parquet": ("--input", "--output-dir", "--tick-size", "--snapshot-policy", "--warm-start"),
    "columnar": (
        "--input",
        "--output",
        "--tick-size",
        "--levels",
        "--snapshot-policy",
        "--snapshot-dir",
        "--snapshot-levels",
        "--seed-snapshot",
        "--skip-until",
        "--warm-start",
    ),
    "matched": ("--input", "--output", "--tick-size", "--warm-start"),
}


def resolve_binary() -> Path:
    """
    Locate the compiled ``lob_replay`` executable.

    Checks ``components/target/{release,debug}`` with the platform executable
    extension, falling back to the extension-less Unix name.

    Raises:
        FileNotFoundError: if no binary exists, with the build command to run.
    """
    ext = ".exe" if sys.platform.startswith("win") else ""
    components_root = Path(__file__).resolve().parents[3]
    candidates = [
        components_root / "target" / "release" / f"lob_replay{ext}",
        components_root / "target" / "debug" / f"lob_replay{ext}",
        components_root / "target" / "release" / "lob_replay",
        components_root / "target" / "debug" / "lob_replay",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "lob_replay binary not found; run:\n"
        "  cargo build --manifest-path components/Cargo.toml --bin lob_replay --release"
    )


def build_command(
    mode: str,
    *,
    input_path: str | Path,
    tick_size: float,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    snapshot_policy: str | None = None,
    levels: int | None = None,
    snapshot_dir: str | Path | None = None,
    snapshot_levels: int | None = None,
    seed_snapshot: str | Path | None = None,
    skip_until: int | None = None,
    warm_start: bool = False,
) -> list[str]:
    """
    Assemble a ``lob_replay`` argument vector for one of the three modes.

    Only flags valid for the requested mode are emitted; supplying a flag the
    mode does not accept raises ``ValueError`` here rather than letting the Rust
    binary fail later with ``unknown flag``.
    """
    if mode not in MODE_FLAGS:
        raise ValueError(f"unknown lob_replay mode {mode!r}; expected one of {MODES}")

    allowed = set(MODE_FLAGS[mode])
    values: dict[str, object] = {"--input": input_path, "--tick-size": tick_size}

    if mode in ("columnar", "matched"):
        if output is None:
            raise ValueError(f"mode {mode!r} requires output=")
        values["--output"] = output
    if mode == "ingest-parquet":
        if output_dir is None:
            raise ValueError("mode 'ingest-parquet' requires output_dir=")
        values["--output-dir"] = output_dir

    optional: dict[str, object] = {
        "--snapshot-policy": snapshot_policy,
        "--levels": levels,
        "--snapshot-dir": snapshot_dir,
        "--snapshot-levels": snapshot_levels,
        "--seed-snapshot": seed_snapshot,
        "--skip-until": skip_until,
    }
    for flag, value in optional.items():
        if value is None:
            continue
        if flag not in allowed:
            raise ValueError(f"flag {flag!r} is not accepted by lob_replay mode {mode!r}")
        values[flag] = value

    if warm_start and "--warm-start" not in allowed:
        raise ValueError(f"flag '--warm-start' is not accepted by lob_replay mode {mode!r}")

    command: list[str] = [str(resolve_binary()), mode]
    for flag in MODE_FLAGS[mode]:
        if flag == "--warm-start":
            if warm_start:
                command.append("--warm-start")
            continue
        if flag in values:
            command.extend([flag, str(values[flag])])
    return command


def run(command: list[str], label: str = "lob_replay") -> str:
    """Execute a built command, surfacing stdout/stderr on failure."""
    print(f"[LOB_BUILDER] {label}: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    if result.returncode != 0:
        sys.stderr.write(result.stderr or "")
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def run_ingest_parquet(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    tick_size: float,
    snapshot_policy: str = "every_event",
    warm_start: bool = True,
    label: str | None = None,
) -> str:
    """
    Convenience wrapper: replay ``input_path`` (file *or* directory) with
    ``ingest-parquet``.

    Passing a directory uses one shared ReplayEngine across every parquet file
    under it (discovered in sorted order), which is how a full UTC day is
    reconstructed with the book carrying across hour boundaries.
    """
    command = build_command(
        "ingest-parquet",
        input_path=input_path,
        output_dir=output_dir,
        tick_size=tick_size,
        snapshot_policy=snapshot_policy,
        warm_start=warm_start,
    )
    return run(command, label=label or f"ingest-parquet {Path(input_path).name}")


def main(argv: list[str] | None = None) -> int:
    """CLI mirroring the Rust binary, one flag per accepted lob_replay option."""
    parser = argparse.ArgumentParser(
        description="Run the compiled Rust LOB builder (lob_replay).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=list(MODES), default="ingest-parquet")
    parser.add_argument("--input", required=True, help="Input parquet file, or directory (ingest-parquet).")
    parser.add_argument("--output", help="Output parquet file (columnar, matched).")
    parser.add_argument("--output-dir", help="Output directory (ingest-parquet).")
    parser.add_argument("--tick-size", type=float, required=True, help="Instrument tick size, e.g. 0.1")
    parser.add_argument("--snapshot-policy", choices=["day_start", "every_event"], default=None)
    parser.add_argument("--levels", type=int, default=None, help="columnar: book levels per side.")
    parser.add_argument("--snapshot-dir", default=None, help="columnar: checkpoint output directory.")
    parser.add_argument("--snapshot-levels", type=int, default=None, help="columnar: checkpoint levels.")
    parser.add_argument("--seed-snapshot", default=None, help="columnar: seed snapshot file.")
    parser.add_argument("--skip-until", type=int, default=None, help="columnar: skip events at/before this ms.")
    parser.add_argument("--warm-start", action="store_true", help="Accept diff-only input and rebase after gaps.")
    args = parser.parse_args(argv)

    command = build_command(
        args.mode,
        input_path=args.input,
        output=args.output,
        output_dir=args.output_dir,
        tick_size=args.tick_size,
        snapshot_policy=args.snapshot_policy,
        levels=args.levels,
        snapshot_dir=args.snapshot_dir,
        snapshot_levels=args.snapshot_levels,
        seed_snapshot=args.seed_snapshot,
        skip_until=args.skip_until,
        warm_start=args.warm_start,
    )
    run(command, label=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())