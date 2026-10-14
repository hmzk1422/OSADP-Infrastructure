"""Profile a live capture: is the feed keeping up, or stalling and catching up?

The question this answers: does the CSV advance steadily at ~10 rows/s of
*exchange* time, or does it sit flat and then dump a batch?

Two numbers matter:

* `exchange_span` - first timestamp to last timestamp, i.e. how much market
  time the file covers.
* `wall_clock`   - how long the capture actually ran.

If `exchange_span` is far smaller than `wall_clock`, rows were buffered and
released late. If a handful of very large timestamp gaps sit inside an
otherwise 102 ms stream, those are feed stalls, and the rows either side of
one were written to disk in the same instant.

Usage: python tools/latency_profile.py <capture.csv>
"""

import sys
from collections import Counter
from pathlib import Path

MS = 1_000_000  # timestamp column is epoch nanoseconds

BUCKETS = [
    (0, 150, "~100ms (nominal)"),
    (150, 250, "150-250ms (one dropped)"),
    (250, 1_000, "250ms-1s"),
    (1_000, 5_000, "1-5s"),
    (5_000, 30_000, "5-30s"),
    (30_000, float("inf"), ">30s"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    path = Path(sys.argv[1])
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if len(lines) < 3:
        print(f"too few rows in {path}")
        return 1

    header = lines[0].split(",")
    i_ts = header.index("timestamp")
    i_type = header.index("event_type")
    i_uid = header.index("lob__final_update_id")

    rows = [l.split(",") for l in lines[1:]]
    stamps = [int(r[i_ts]) for r in rows]
    kinds = Counter(r[i_type] for r in rows)

    gaps = [(stamps[i] - stamps[i - 1]) // MS for i in range(1, len(stamps))]

    print(f"file           {path}")
    print(f"rows           {len(rows)}   {dict(kinds)}")
    print(
        f"exchange span  {(stamps[-1] - stamps[0]) / 1e9:.1f}s "
        f"({stamps[0]} -> {stamps[-1]})"
    )

    print("\nrow-to-row timestamp gap distribution:")
    counted = Counter()
    for g in gaps:
        for lo, hi, label in BUCKETS:
            if lo <= g < hi:
                counted[label] += 1
                break
    total = len(gaps)
    for _, _, label in BUCKETS:
        n = counted.get(label, 0)
        bar = "#" * int(48 * n / total) if total else ""
        print(f"  {label:<22} {n:>6}  {100*n/total:5.1f}%  {bar}")

    big = sorted(g for g in gaps if g >= 1_000)
    if big:
        print(f"\ngaps >= 1s: {len(big)}  total {sum(big)/1000:.1f}s of stalled market time")
        print(f"  largest: {max(big)/1000:.1f}s   median: {sorted(big)[len(big)//2]/1000:.1f}s")
        # Where the stalls sit, as a fraction through the file.
        where = [
            i / len(gaps)
            for i, g in enumerate(gaps)
            if g >= 1_000
        ]
        print(f"  positions in file: {', '.join(f'{w:.0%}' for w in where[:14])}")

    # Duplicate consecutive update ids: rows that carried no new book state.
    uids = [r[i_uid] for r in rows if r[i_type] == "LOB" and r[i_uid]]
    dup = sum(1 for a, b in zip(uids, uids[1:]) if a == b)
    if uids:
        print(f"\nLOB rows with an update id:      {len(uids)}")
        print(f"  repeating the previous uid:    {dup} ({100*dup/(len(uids)-1):.1f}%)")

    # The verdict the file itself supports.
    stalled = sum(g for g in gaps if g >= 1_000)
    if total:
        share = stalled / sum(gaps) if sum(gaps) else 0
        print(
            f"\nstalled share of market time: {share:.1%}"
            + ("  <-- the feed is stalling, not merely stale" if share > 0.20 else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
