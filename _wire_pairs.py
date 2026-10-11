"""One-off: wrap the GP (state, target) pair loop in a thread-safe per-pair function."""
import re
from pathlib import Path

p = Path('strategy_system/strategy/moe/gp/evolution.py')
lines = p.read_text(encoding='utf-8').splitlines(keepends=True)

assert lines[641].strip() == 'rng = np.random.default_rng(seed)', lines[641]
assert lines[651].strip().startswith('# Each state-target pair'), lines[651]
assert lines[652].strip().startswith('for state_id in tqdm(states'), lines[652]
assert lines[654].strip().startswith('for target_column in tqdm(targets_to_use'), lines[654]
assert lines[876].strip() == 'break', lines[876]

body = lines[655:877]  # lines 656..877
raw = ''.join(body)
nl = '\r\n' if '\r\n' in raw else '\n'

# Only the two pair-level skips become early returns; inner model loops keep `continue`.
raw = raw.replace(
    f'            if int(train_mask_pair.sum()) < required_samples:{nl}'
    f'                continue{nl}',
    f'            if int(train_mask_pair.sum()) < required_samples:{nl}'
    f'                return [], []{nl}')
raw = raw.replace(
    f'            if not cheap_names:{nl}'
    f'                continue{nl}',
    f'            if not cheap_names:{nl}'
    f'                return [], []{nl}')
assert raw.count('continue') == 6, raw.count('continue')
assert raw.count('return [], []') == 2, raw.count('return [], []')

text = ''.join(ln[4:] if ln.startswith('    ') else ln
               for ln in raw.splitlines(keepends=True))

text = re.sub(r'(?<![A-Za-z_])rng(?![A-Za-z_])', 'pair_rng', text)
text = re.sub(r'(?<![A-Za-z_])selected_values(?![A-Za-z_])', 'pair_selected_values', text)
text = re.sub(r'(?<![A-Za-z_])selected(?![A-Za-z_])', 'pair_selected', text)
text = re.sub(r'(?<![A-Za-z_])report_rows(?![A-Za-z_])', 'pair_report', text)

# Only the two pair-level skips become early returns; inner model loops keep `continue`.
text = re.sub(r'(\s+)continue\r?\n', r'\1continue\r\n', text)
text = text.replace(
    '            if int(train_mask_pair.sum()) < required_samples:\r\n'
    '                continue\r\n',
    '            if int(train_mask_pair.sum()) < required_samples:\r\n'
    '                return [], []\r\n')
text = text.replace(
    '            if not cheap_names:\r\n'
    '                continue\r\n',
    '            if not cheap_names:\r\n'
    '                return [], []\r\n')
assert text.count('continue') == 6, text.count('continue')
assert text.count('return [], []') == 2, text.count('return [], []')

header = '''    # Each state-target pair has an independent population and objective, so the
    # pairs can be searched concurrently. See gp/parallel.py for the micro-benchmark
    # that justifies threads and for the determinism contract.
    def search_pair(state_id: str, target_column: str):
        """Search one (latent state, target) pair; return its selections and rows.

        Thread-safe by construction: every mutable object below is created per
        call, and the RNG is derived from ``(seed, state_id, target_column)``
        rather than a shared stream, so results never depend on the order in
        which pairs are evaluated.
        """
        pair_rng = np.random.default_rng(derive_pair_seed(seed, state_id, target_column))
        pair_selected: list[SelectedExpression] = []
        pair_report: list[dict[str, Any]] = []
        state_mask = state_keys == state_id.casefold()
'''

footer = '''        return pair_selected, pair_report

    pairs = [(state_id, target_column)
             for state_id in states for target_column in targets_to_use]
    for pair_selected, pair_report in run_pairs(
        search_pair, pairs, workers=workers, show_progress=True,
    ):
        selected.extend(pair_selected)
        report_rows.extend(pair_report)

'''

new = lines[:641] + [header] + [text] + [footer] + lines[877:]
p.write_text(''.join(new), encoding='utf-8')
print('rewrote', p, 'lines', len(new))
