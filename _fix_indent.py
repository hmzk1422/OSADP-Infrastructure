#!/usr/bin/env python
import io, sys
p = r"uets_rs/live-trade/src/model.rs"
lines = io.open(p, encoding="utf-8").read().split("\n")
# Fix line 89 (0-indexed 88): should be exactly 4-space indent
for i in range(len(lines)):
    if lines[i].strip() == 'let per_sym = format!("build/regime_gate_{symbol}.json");':
        lines[i] = "    let per_sym = format!(\"build/regime_gate_{symbol}.json\");"
        print(f"Fixed line {i+1}")
# Check for duplicate let path lines
path_count = sum(1 for l in lines if l.strip().startswith("let path = find_model_path"))
print(f"path lines found: {path_count}")
io.open(p, "w", encoding="utf-8").write("\n".join(lines))
print("written")
