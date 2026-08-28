import re
p = r'c:\Users\Himaska\Documents\Projects\Business_Projects\intraday-momentum-composite\uets_rs\engine\src\lib.rs'
lines = open(p, encoding='utf-8').read().split('\n')

# 1) Remove the duplicate HYSTERESIS_BUFFER const (keep the config-block one
#    near line 68; blank the later ORDERFLOW_Z_WINDOW-group one).
for i, line in enumerate(lines):
    if line.strip().startswith('pub const HYSTERESIS_BUFFER') and i > 70:
        lines[i] = ''

s = '\n'.join(lines)

# 2) Normalize stray over-indentation on a few single statements whose edits
#    lacked leading whitespace (editor appends, leaving the original indent).
fix = {
    'let rsi_x_ma = rsi_z * ma_slope_5;': 8,
    'let (in_zone, side) = self.resolve_zone_side(&feat);': 8,
    'self.macro_close = *closes.last().expect("non-empty");': 8,
    'self.last_trade_price = Some(price);': 12,
    '(*w.last().expect("non-empty window") - mean) / sd': 4,
}
s = s.split('\n')
for i, line in enumerate(s):
    stripped = line.lstrip(' ')
    if stripped in fix:
        s[i] = ' ' * fix[stripped] + stripped
s = '\n'.join(s)

# 3) Convert `self.push_roll_z(...)` calls into free-function calls so the
#    borrow checker treats `&mut self.xxx_buffer` as a disjoint field borrow.
s = re.sub(r'self\s*\n\s*\.push_roll_z\(', 'push_roll_z(', s)
s = s.replace('self.push_roll_z(', 'push_roll_z(')

# 4) Insert the free-function `push_roll_z` right before the Tests banner.
marker = '\n// ============================================================================\n// Tests'
assert marker in s, 'tests marker not found'
free_fn = '''
/// Append `value` to `buf`, trim to `MAX_Z_HIST`, then return its rolling
/// z-score vs the trailing `window` samples (population std, matches
/// features.py `_rolling_zscore` → `calculate_zscore(window=…)`).
fn push_roll_z(buf: &mut VecDeque<f64>, value: f64, window: usize) -> f64 {
    buf.push_back(value);
    if buf.len() > MAX_Z_HIST {
        buf.pop_front();
    }
    let vals: Vec<f64> = buf.iter().copied().collect();
    roll_z(&vals, window)
}
'''
s = s.replace(marker, '\n' + free_fn + marker, 1)

open(p, 'w', encoding='utf-8').write(s)
print('all fixes applied')



