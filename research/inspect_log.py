"""Inspect executor.log for key decision actions and confidence values."""
import re

path = 'main/logs/executor.log'
patterns = re.compile(
    r'0\.528|ENTER_SHORT|ENTER LONG|DEBOUNCED|QTY_TOO_SMALL|Vol filter|LOCKOUT|'
    r'Signal ignored|P\(WIN\) confidence',
)

matches = []
with open(path, encoding='utf-8', errors='ignore') as f:
    for line in f:
        if patterns.search(line):
            matches.append(line.rstrip())

print(f"Total matches: {len(matches)}")
print("\n--- Last 100 matches ---")
for line in matches[-100:]:
    print(line)