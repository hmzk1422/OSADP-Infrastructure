"""One-shot: strip mojibake / non-ASCII from a source file.

The file was written through PowerShell, which double-encoded the em-dashes
into UTF-8 mojibake. rustc tolerates it (the bytes are valid UTF-8) but
rust-analyzer flags the corrupted sequences as syntax errors. This rewrites
the file as pure ASCII so every tool agrees.

Usage: python tools/fix_encoding.py <path>
"""

import sys
from pathlib import Path

# Multi-byte mojibake first, longest patterns before shorter substrings.
REPLACEMENTS = [
    ("\u00e2\u20ac\u201d", "--"),  # em dash
    ("\u00e2\u20ac\u201c", "-"),  # en dash
    ("\u00e2\u20ac\u2122", "'"),  # right single quote
    ("\u00e2\u20ac\u0153", '"'),  # left double quote
    ("\u00e2\u20ac\u009d", '"'),  # right double quote
    ("\u00e2\u20ac", "-"),
    ("\u00c3\u00a9", "e"),
    ("\u00e2\u0080\u0094", "--"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    before = sum(1 for ch in text if ord(ch) > 127)
    for bad, good in REPLACEMENTS:
        text = text.replace(bad, good)

    # Anything still non-ASCII is dropped: these are all comments, so the
    # punctuation is cosmetic and losing it beats leaving a corrupted byte.
    text = "".join(ch if ord(ch) < 128 else "-" for ch in text)

    after = sum(1 for ch in text if ord(ch) > 127)
    path.write_text(text, encoding="utf-8", newline="\n")

    print(f"{path}")
    print(f"  non-ascii before: {before}")
    print(f"  non-ascii after:  {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
