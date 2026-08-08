"""Inspect Aug 4 session and 0.528 occurrences in executor log."""
import re
from collections import Counter

path = 'main/logs/executor.log'

# 1. Find all p_win values seen in "Signal ignored" messages, most recent first
ignored = []
with open(path, encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = re.search(r'P\(WIN\) confidence \(([0-9.]+)\) is below ([0-9.]+)\.', line)
        if m:
            ignored.append((line[:19], float(m.group(1)), float(m.group(2)), line.rstrip()))

print(f"Total 'Signal ignored' lines: {len(ignored)}")
print("\n--- Last 30 'Signal ignored' (timestamp, p_win, threshold) ---")
for ts, pw, thr, _ in ignored[-30:]:
    print(f"{ts} | p_win={pw:.3f} | threshold={thr}")

# 2. Count p_win values >= 0.5 that were ignored (contradictions)
contradictions = [x for x in ignored if x[1] >= x[2] - 1e-9]
print(f"\n'Signal ignored' lines where p_win >= threshold: {len(contradictions)}")
for ts, pw, thr, line in contradictions[-10:]:
    print(f"  {ts} | p_win={pw:.3f} | threshold={thr} | {line[-120:]}")

# 3. Look for 0.528 specifically
print("\n--- Lines containing 0.528 ---")
count_528 = 0
with open(path, encoding='utf-8', errors='ignore') as f:
    for line in f:
        if '0.528' in line:
            count_528 += 1
            if count_528 <= 20:
                print(line[:160].rstrip())
print(f"Total lines with 0.528: {count_528}")

# 4. Aug 4 session: find 'ENTER' and 'Vol filter' and 'Signal' activity after 2026-08-04
print("\n--- Aug 4 session activity ---")
with open(path, encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('2026-08-04') and re.search(r'ENTER|Vol filter|Signal ignored|DEBOUNCED|RSI_FILTERED|LOCKOUT', line):
            print(line[:200].rstrip())