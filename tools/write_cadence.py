"""Watch a live capture's file grow, and report the *disk-write* cadence.

`latency_profile.py` reads exchange timestamps from a finished file. This one
answers the question those timestamps cannot: are rows hitting the disk
steadily, or does the file sit flat and then dump a batch?

A flat-then-burst profile is the signature of buffering -- rows produced on
time but written late. A steady ~10 rows/s profile means the pipeline keeps up.

Usage: python tools/write_cadence.py <capture_dir> [seconds]
"""

import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    out_dir = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    uets = Path(__file__).resolve().parent.parent / "uets"
    cmd = [
        "cargo", "run", "--quiet", "--bin", "dev_test", "--",
        "record", "btcusdt", str(out_dir), "--rows", "800",
    ]
    proc = subprocess.Popen(
        cmd, cwd=uets, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print(f"capturing into {out_dir} for ~{seconds}s\n")
    samples: list[tuple[float, int]] = []
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline and proc.poll() is None:
            csvs = list(out_dir.glob("*.csv"))
            if csvs:
                rows = sum(1 for _ in csvs[0].open("rb")) - 1
                samples.append((time.monotonic(), rows))
            time.sleep(0.25)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    if len(samples) < 3:
        print("too few samples; the capture may have failed")
        return 1

    t0, r0 = samples[0]
    print("  elapsed   rows   delta   interval")
    prev_t, prev_r = None, None
    deltas = []
    for t, r in samples:
        if prev_t is not None:
            d = r - prev_r
            deltas.append((d, (t - prev_t) * 1000))
            print(f"  {t-t0:7.2f}s {r:6}   {d:+5}    {(t-prev_t)*1000:6.0f}ms")
        prev_t, prev_r = t, r

    total_s = samples[-1][0] - t0
    total_r = samples[-1][1] - r0
    print(f"\nrows written: {total_r} in {total_s:.1f}s "
          f"=> {total_r/total_s:.1f} rows/s")
    print(f"expected at 100ms cadence: ~{total_s*10:.0f} rows")

    zero = [d for d, _ in deltas if d == 0]
    burst = [d for d, _ in deltas if d >= 20]
    print(f"\nsamples with ZERO new rows: {len(zero)}/{len(deltas)}")
    print(f"samples with >=20 new rows: {len(burst)}/{len(deltas)}  <-- bursts")
    if total_s:
        lag = (total_s * 10 - total_r) / (total_s * 10) if total_s else 0
        print(f"shortfall vs real-time:     {lag:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
