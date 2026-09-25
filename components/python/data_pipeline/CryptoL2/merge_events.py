"""Step 4 CLI: merge three source streams through the Rust event merger."""

from __future__ import annotations

from typing import Optional

from components.python.data_pipeline.matched import merge as _merge


def main(argv: Optional[list[str]] = None) -> int:
    return _merge.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
