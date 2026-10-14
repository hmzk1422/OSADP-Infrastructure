"""Why does a batch of rows land at once?

`write_cadence.py` measures disk-write bursts. This explains them by checking
the two hypotheses that actually matter:

1. RELEASE BATCHING - rows share a timestamp, so the reorder buffer let a
   group go together.
2. SOURCE FLURRY - many events with the *same exchange millisecond* arrived
   from the feed, so the rows are genuinely simultaneous market events.

Distinguishing them is the whole point: (1) is our bug, (2) is the market.

Usage: python tools/burst_attrib.py <capture.csv>
"""

import sys
from pathlib import Path

MS = 1_000_000


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    path = Path(sys.argv[1])
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    header = lines[0].split(",")
    i_ts = header.index("timestamp")
    i_type = header.index("event_type")

    rows = [l.split(",") for l in lines[1:]]
    stamps = [int(r[i_ts]) for r in rows]
    kinds = [r[i_type] for r in rows]

    # Group rows that share an exact timestamp: a "simultaneous" group.
    groups: list[list[int]] = []
    run = [0]
    for i in range(1, len(stamps)):
        if stamps[i] == stamps[i - 1]:
            run.append(i)
        else:
            groups.append(run)
            run = [i]
    groups.append(run)

    multi = [g for g in groups if len(g) > 1]
    singles = [g for g in groups if len(g) == 1]

    print(f"file                  {path}")
    print(f"rows                  {len(rows)}")
    print(f"distinct timestamps   {len(groups)}")
    print(f"rows in a shared-ts group     {sum(len(g) for g in multi)}")
    print(f"rows with a unique timestamp  {len(singles)}")

    print("\nlargest simultaneous groups (rows sharing one timestamp):")
    for g in sorted(multi, key=len, reverse=True)[:8]:
        mix = {}
        for i in g:
            mix[kinds[i]] = mix.get(kinds[i], 0) + 1
        print(f"  {len(g):4} rows at ts={stamps[g[0]]}  {mix}")

    # Where do the biggest jumps in row count happen? Those are the bursts a
    # human watching the file sees.
    print("\ntimestamp gaps around the largest bursts:")
    gaps = [(i, (stamps[i] - stamps[i - 1]) // MS) for i in range(1, len(stamps))]
    for i, g in sorted(gaps, key=lambda p: -p[1])[:6]:
        print(f"  row {i}: previous row {g}ms earlier")

    # The verdict: if the largest groups are all one event type and the feed
    # genuinely produced them in one millisecond, this is the market.
    biggest = max(multi, key=len) if multi else None
    if biggest:
        mix = {kinds[i] for i in biggest}
        print(
            f"\nlargest group is {len(biggest)} rows of type(s) {sorted(mix)}\n"
            + (
                "  -> one event type: simultaneous market events (the feed sent\n"
                "     them in the same exchange millisecond), not release batching."
                if len(mix) == 1
                else "  -> mixed types in one timestamp: the buffer released them together."
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
