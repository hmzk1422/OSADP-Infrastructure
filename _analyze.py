import re, pathlib

root = pathlib.Path('strategy_system/strategy/moe/ood')
chatgpt = pathlib.Path('strategy_system/strategy/moe/ood/metrics_fix/chatgpt/apply_tdmi_fixes.py').read_text(encoding='utf-8')

print('=== chatgpt replace_top_function(metrics, ...) targets ===')
print(sorted(set(re.findall(r'replace_top_function\(metrics, "([^"]+)"', chatgpt))))
print('=== chatgpt replace_method(validator, ...) targets ===')
print(sorted(set(re.findall(r'replace_method\(validator, "([^"]+)"', chatgpt))))
print('=== chatgpt replace_literal_once labels ===')
print(re.findall(r'replace_literal_once\(\s*metrics,\s*(?:.*?\n)*?\s*"[^"]*",\s*"[^"]*",\s*"([^"]+)"', chatgpt))

# Live validator methods
lv = (root / 'regime_validator.py').read_text(encoding='utf-8')
live_methods = set(re.findall(r'^    def (\w+)', lv, re.M))
chatgpt_methods = set(re.findall(r'replace_method\(validator, "([^"]+)"', chatgpt))
print('=== chatgpt validator methods MISSING from live ===')
print(sorted(chatgpt_methods - live_methods))
print('=== chatgpt validator methods present in live ===')
print(sorted(chatgpt_methods & live_methods))

# Live metrics funcs
lm = (root / 'metrics.py').read_text(encoding='utf-8')
live_funcs = set(re.findall(r'^def (\w+)', lm, re.M))
chatgpt_funcs = set(re.findall(r'replace_top_function\(metrics, "([^"]+)"', chatgpt))
print('=== chatgpt metrics funcs MISSING from live ===')
print(sorted(chatgpt_funcs - live_funcs))
print('=== chatgpt metrics funcs present in live ===')
print(sorted(chatgpt_funcs & live_funcs))
